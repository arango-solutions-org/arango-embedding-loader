"""The embedding pipeline: read → label/extract text → embed → write → (vector index)."""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from arango.database import StandardDatabase

from app.config import get_settings
from app.errors import AppError
from app.ingest.text import get_path, render_template, set_path, split_path, value_to_text
from app.models.embedding import EmbeddingFieldSpec, VectorIndexSpec
from app.models.load import LoadOptions, ReembedRequest
from app.models.profiles import Profile
from app.providers.base import EmbeddingProvider, ProviderOptions
from app.providers.long_input import TextPlan
from app.providers.presets import get_preset
from app.providers.registry import get_provider
from app.services import profiles as profile_store
from app.services.analysis import Analysis, long_input_of
from app.services.jobs import JobHandle
from app.services.sources import DocumentSource
from app.services.vector_index import ensure_vector_index
from app.services.writer import WriteResult, insert_batch, update_batch

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- specs


@dataclass
class ResolvedSpec:
    spec: EmbeddingFieldSpec
    provider: EmbeddingProvider
    model: str
    options: ProviderOptions
    dimensions: int | None

    @property
    def target(self) -> str:
        return self.spec.resolved_target

    @property
    def meta(self) -> str:
        return self.spec.meta_field

    def text_for(self, doc: dict[str, Any]) -> str | None:
        if self.spec.template:
            return render_template(self.spec.template, doc)
        assert self.spec.source_field
        return value_to_text(get_path(doc, self.spec.source_field))

    def describe(self) -> dict[str, Any]:
        return {
            "source_field": self.spec.source_field,
            "template": self.spec.template,
            "target_field": self.target,
            "provider": self.provider.name,
            "model": self.model,
            "dimensions": self.dimensions,
        }


def resolve_spec(spec: EmbeddingFieldSpec) -> ResolvedSpec:
    provider_name = spec.provider
    model = spec.model
    raw_options: dict[str, Any] = {}
    if spec.preset:
        preset = get_preset(spec.preset)
        provider_name = spec.provider or preset.provider
        model = model or preset.model
        raw_options.update(preset.provider_options)
    raw_options.update(spec.provider_options)
    assert provider_name
    provider = get_provider(provider_name)
    options = provider.parse_options(raw_options)
    model = model or provider.default_model
    if model:
        options = provider.apply_long_input(options, model, spec.long_input)
    provider.ensure_available(options)
    if not model:
        raise AppError(400, "model_required", f"Provider '{provider.name}' needs a model")
    return ResolvedSpec(
        spec=spec, provider=provider, model=model, options=options, dimensions=spec.dimensions
    )


def resolve_specs(specs: list[EmbeddingFieldSpec]) -> list[ResolvedSpec]:
    resolved = [resolve_spec(s) for s in specs]
    targets = [r.target for r in resolved]
    if len(set(targets)) != len(targets):
        raise AppError(400, "duplicate_target", f"Duplicate target fields: {targets}")
    return resolved


# ----------------------------------------------------------------------- embedding


@dataclass
class DocOutcome:
    error: str | None = None
    embedded: bool = False
    skipped: bool = False
    #: The text of at least one field was longer than the model's token limit.
    over_limit: bool = False
    #: Extra documents to write, one per piece (``chunk_docs``).
    chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Split:
    spec: ResolvedSpec
    pieces: list[str]
    vectors: list[list[float]]
    weights: list[int] | None


def _meta(spec: ResolvedSpec, status: str, dimension: int | None = None, **extra: Any) -> dict:
    meta: dict[str, Any] = {
        "status": status,
        "provider": spec.provider.name,
        "model": spec.model,
        "at": datetime.now(UTC).isoformat(),
    }
    if dimension is not None:
        meta["dimension"] = dimension
    meta.update(extra)
    return meta


def _is_pending(doc: dict[str, Any], spec: ResolvedSpec) -> bool:
    return get_path(doc, f"{spec.meta}.status") is None


def _doc_ref(doc: dict[str, Any], batch_index: int) -> str:
    key = doc.get("_key")
    return f"_key '{key}'" if key is not None else f"batch position {batch_index}"


def _log_overrun(spec: ResolvedSpec, doc: dict[str, Any], index: int, plan: TextPlan) -> None:
    st = plan.stats
    if plan.action == "error":
        level, outcome = logging.ERROR, "stopping the run"
        _emit_overrun(spec, doc, index, st, level, outcome)
        return
    level = logging.WARNING if plan.warn else logging.INFO
    if plan.combine in ("mean", "split"):
        outcome = f"split into {st['chunks']} chunks"
    elif plan.action == "skip":
        outcome = "not embedded"
    else:
        outcome = f"{st['tokens_dropped']} tokens dropped"
    _emit_overrun(spec, doc, index, st, level, outcome)


def _emit_overrun(
    spec: ResolvedSpec,
    doc: dict[str, Any],
    index: int,
    st: dict[str, Any],
    level: int,
    outcome: str,
) -> None:
    log.log(
        level,
        "Text for '%s' in document %s is %s tokens, over the %s-token limit of %s/%s; "
        "long_input=%s (%s count), %s",
        spec.target,
        _doc_ref(doc, index),
        st.get("input_tokens"),
        st.get("max_tokens"),
        spec.provider.name,
        spec.model,
        st.get("long_input"),
        st.get("token_count"),
        outcome,
    )


_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9_\-:.@()+,=;$!*'%]")


def _chunk_docs(
    parent: dict[str, Any], splits: list[_Split], specs: list[ResolvedSpec]
) -> list[dict[str, Any]]:
    """One document per piece: a copy of ``parent`` holding the piece text and its vector."""
    if "_key" not in parent:
        parent["_key"] = uuid.uuid4().hex  # chunks need a key to point back to
    parent_key = str(parent["_key"])
    out: list[dict[str, Any]] = []
    for split in splits:
        spec = split.spec
        count = len(split.pieces)
        for n, (piece, vec) in enumerate(zip(split.pieces, split.vectors, strict=True)):
            child = copy.deepcopy(parent)
            for system in ("_id", "_rev"):
                child.pop(system, None)
            child["_key"] = f"{parent_key}_{_KEY_UNSAFE.sub('_', spec.target)}_chunk_{n}"
            if spec.spec.source_field:
                set_path(child, spec.spec.source_field, piece)
            else:
                set_path(child, f"{spec.target}_text", piece)
            set_path(child, spec.target, vec)
            set_path(
                child,
                spec.meta,
                _meta(
                    spec,
                    "embedded",
                    len(vec),
                    parent_key=parent_key,
                    chunk_index=n,
                    chunk_count=count,
                    input_tokens=split.weights[n] if split.weights else None,
                ),
            )
            # The other fields' vectors describe the parent; keep them off the chunks so vector
            # searches on those fields don't return duplicates, and mark them done for the
            # watcher.
            for other in specs:
                if other is spec:
                    continue
                set_path(child, other.target, None)
                set_path(
                    child,
                    other.meta,
                    _meta(other, "skipped", reason="chunk document", parent_key=parent_key),
                )
            out.append(child)
    return out


class InputTooLongError(AppError):
    def __init__(self, message: str):
        super().__init__(422, "input_too_long", message)


async def embed_documents(
    docs: list[dict[str, Any]],
    specs: list[ResolvedSpec],
    *,
    store_metadata: bool,
    only_pending: bool = False,
) -> list[DocOutcome]:
    """Embed ``docs`` in place according to ``specs``. Returns one outcome per document.

    Documents whose text is split into pieces (``chunk_docs``) get the extra documents to
    write in ``DocOutcome.chunks``. A text that is too long with ``long_input: error`` raises
    :class:`InputTooLongError`, which stops the run.
    """
    outcomes = [DocOutcome() for _ in docs]
    splits: dict[int, list[_Split]] = {}
    for spec in specs:
        texts: list[str] = []
        positions: list[int] = []
        for i, doc in enumerate(docs):
            if outcomes[i].error or (only_pending and not _is_pending(doc, spec)):
                continue
            text = spec.text_for(doc)
            if text is not None:
                texts.append(text)
                positions.append(i)
                continue
            missing = spec.spec.template or spec.spec.source_field
            if spec.spec.on_missing == "error":
                outcomes[i].error = f"Missing or empty source for '{spec.target}': {missing}"
                if store_metadata:
                    set_path(doc, spec.meta, _meta(spec, "error", reason="missing source"))
                continue
            outcomes[i].skipped = True
            if spec.spec.on_missing == "null":
                set_path(doc, spec.target, None)
            if store_metadata:
                set_path(doc, spec.meta, _meta(spec, "skipped", reason="missing source"))
        if not texts:
            continue
        plans = await asyncio.to_thread(
            spec.provider.plan_inputs,
            texts,
            model=spec.model,
            options=spec.options,
            input_type="document",
            long_input=spec.spec.long_input_settings,
        )
        if len(plans) != len(texts):
            raise AppError(
                500, "plan_mismatch", f"Provider '{spec.provider.name}' planned the wrong count"
            )
        for pos, plan in zip(positions, plans, strict=True):
            if plan.overrun:
                outcomes[pos].over_limit = True
                _log_overrun(spec, docs[pos], pos, plan)
            if plan.action == "error":
                st = plan.stats
                raise InputTooLongError(
                    f"Text for '{spec.target}' in document {_doc_ref(docs[pos], pos)} is "
                    f"{st.get('input_tokens')} tokens, over the {st.get('max_tokens')}-token "
                    f"limit of {spec.provider.name}/{spec.model} (long_input: error)."
                )

        flat: list[str] = []
        for plan in plans:
            if plan.action == "embed":
                flat.extend(plan.pieces)
        vectors = await spec.provider.embed(
            flat,
            model=spec.model,
            options=spec.options,
            dimensions=spec.dimensions,
            input_type="document",
        )
        cursor = 0
        for pos, plan in zip(positions, plans, strict=True):
            doc = docs[pos]
            if plan.action == "skip":
                outcomes[pos].skipped = True
                if store_metadata:
                    set_path(doc, spec.meta, _meta(spec, "skipped_too_long", **plan.stats))
                continue
            piece_vectors = vectors[cursor : cursor + len(plan.pieces)]
            cursor += len(plan.pieces)
            if plan.combine == "split":
                set_path(doc, spec.target, None)
                if store_metadata:
                    set_path(doc, spec.meta, _meta(spec, "chunked", **plan.stats))
                splits.setdefault(pos, []).append(
                    _Split(spec, plan.pieces, piece_vectors, plan.weights)
                )
                outcomes[pos].embedded = True
                continue
            if plan.combine == "mean":
                vec = spec.provider.combine_vectors(piece_vectors, plan.weights, spec.options)
            else:
                vec = piece_vectors[0]
            set_path(doc, spec.target, vec)
            if store_metadata:
                set_path(doc, spec.meta, _meta(spec, "embedded", len(vec), **plan.stats))
            outcomes[pos].embedded = True
    for pos, doc_splits in splits.items():
        if not outcomes[pos].error:
            outcomes[pos].chunks = _chunk_docs(docs[pos], doc_splits, specs)
    return outcomes


def chunk_documents(outcomes: list[DocOutcome]) -> list[dict[str, Any]]:
    return [c for o in outcomes for c in o.chunks]


# ---------------------------------------------------------------------------- load


@dataclass
class EffectiveLoad:
    specs: list[ResolvedSpec]
    key_field: str | None
    on_duplicate: str
    batch_size: int
    store_metadata: bool
    vector_index: VectorIndexSpec | None
    profile: Profile | None


async def effective_load_options(
    db: StandardDatabase, opts: LoadOptions | ReembedRequest, collection: str
) -> EffectiveLoad:
    settings = get_settings()
    profile = await asyncio.to_thread(profile_store.get_profile, db, collection)
    embeddings = opts.embeddings
    if embeddings is None:
        if profile is None:
            raise AppError(
                400,
                "embeddings_required",
                f"No `embeddings` given and collection '{collection}' has no embedding profile.",
            )
        embeddings = profile.embeddings
    store_metadata = True if opts.store_metadata is None else opts.store_metadata
    if profile is not None and profile.auto_watch:
        store_metadata = True  # the watcher relies on <target>_meta.status
    key_field = getattr(opts, "key_field", None) or (profile.key_field if profile else None)
    on_duplicate = (
        getattr(opts, "on_duplicate", None)
        or (profile.on_duplicate if profile else None)
        or "error"
    )
    return EffectiveLoad(
        specs=resolve_specs(embeddings),
        key_field=key_field,
        on_duplicate=on_duplicate,
        batch_size=opts.batch_size
        or (profile.batch_size if profile else None)
        or settings.default_batch_size,
        store_metadata=store_metadata,
        vector_index=opts.vector_index or (profile.vector_index if profile else None),
        profile=profile,
    )


def _apply_key(doc: dict[str, Any], key_field: str | None) -> str | None:
    if key_field:
        value = get_path(doc, key_field)
        if value is not None and "_key" not in doc:
            doc["_key"] = str(value)
    key = doc.get("_key")
    return str(key) if key is not None else None


def _preview(doc: dict[str, Any], specs: list[ResolvedSpec]) -> dict[str, Any]:
    out = copy.deepcopy(doc)
    for spec in specs:
        vec = get_path(out, spec.target)
        if isinstance(vec, list) and len(vec) > 8:
            set_path(out, spec.target, [round(v, 6) for v in vec[:8]] + [f"… ({len(vec)} dims)"])
    return out


def long_input_any(eff: EffectiveLoad, mode: str) -> bool:
    return any(long_input_of(s) == mode for s in eff.specs)


def _record_dims(
    dims: dict[str, int], docs: list[dict[str, Any]], specs: list[ResolvedSpec]
) -> None:
    for spec in specs:
        if spec.target in dims:
            continue
        for doc in docs:
            vec = get_path(doc, spec.target)
            if isinstance(vec, list) and vec:
                dims[spec.target] = len(vec)
                break


async def _write_chunks(collection: Any, chunks: list[dict[str, Any]], handle: JobHandle) -> None:
    if not chunks:
        return
    result: WriteResult = await asyncio.to_thread(insert_batch, collection, chunks, "replace")
    handle.job.counts.chunks += result.written
    for _, key, msg in result.errors:
        handle.add_error(msg, key=key)


async def _ensure_collection(db: StandardDatabase, name: str, create: bool) -> None:
    exists = await asyncio.to_thread(db.has_collection, name)
    if exists:
        return
    if not create:
        raise AppError(404, "collection_not_found", f"Collection '{name}' does not exist")
    await asyncio.to_thread(db.create_collection, name)


async def _maybe_vector_index(
    db: StandardDatabase,
    collection: str,
    eff: EffectiveLoad,
    dimension: int | None,
    handle: JobHandle,
) -> None:
    spec = eff.vector_index
    if spec is None or not spec.create or not eff.specs:
        return
    field = spec.field or eff.specs[0].target
    handle.job.message = f"Building vector index on '{field}'…"
    handle.touch()
    try:
        info = await asyncio.to_thread(
            ensure_vector_index,
            db,
            collection,
            field,
            metric=spec.metric,
            dimension=dimension,
            n_lists=spec.n_lists,
            default_n_probe=spec.default_n_probe,
            training_iterations=spec.training_iterations,
            name=spec.name,
            parallelism=spec.parallelism,
        )
        handle.job.result["vector_index"] = info
        handle.job.message = None
    except AppError as exc:
        handle.job.result["vector_index"] = {"status": "failed", "error": str(exc.detail)}
        handle.job.message = f"Documents loaded, but vector index failed: {exc.detail}"


async def _source_batches(
    source: DocumentSource, batch_size: int, max_documents: int | None
) -> AsyncIterator[list[dict[str, Any]]]:
    remaining = max_documents
    while True:
        size = batch_size if remaining is None else min(batch_size, remaining)
        if size <= 0:
            return
        batch = await source.next_batch(size)
        if not batch:
            return
        if remaining is not None:
            remaining -= len(batch)
        yield batch


async def _scan(
    handle: JobHandle,
    analysis: Analysis,
    batches: AsyncIterator[list[dict[str, Any]]],
    *,
    count: bool,
) -> None:
    offset = 0
    async for batch in batches:
        over = await analysis.add_batch(batch, offset)
        offset += len(batch)
        if count:
            handle.job.counts.read += len(batch)
            handle.job.counts.over_limit += over
        handle.touch()


async def _analyze(
    handle: JobHandle,
    eff: EffectiveLoad,
    batches: AsyncIterator[list[dict[str, Any]]],
) -> None:
    """``analyze: true``: count tokens and report; embed and write nothing."""
    analysis = Analysis(eff.specs, eff.key_field)
    await _scan(handle, analysis, batches, count=True)
    handle.job.result["analysis"] = analysis.as_dict()


async def _prescan(
    handle: JobHandle,
    eff: EffectiveLoad,
    batches: AsyncIterator[list[dict[str, Any]]] | None,
) -> None:
    """Before embedding anything, fail if a field with ``long_input: error`` has a long text.

    ``batches`` is a second read of the input, or ``None`` when the input can only be read once.
    """
    specs = [s for s in eff.specs if long_input_of(s) == "error"]
    if not specs:
        return
    job = handle.job
    if batches is None:
        job.result["prescan"] = {
            "status": "skipped",
            "reason": "This source is streamed and can only be read once.",
        }
        log.info("Job %s: no token pre-scan, the source can only be read once", job.id)
        return
    job.message = "Checking text lengths before embedding…"
    handle.touch()
    analysis = Analysis(specs, eff.key_field)
    await _scan(handle, analysis, batches, count=False)
    job.message = None
    message = analysis.error_message()
    job.result["prescan"] = {"status": "failed" if message else "passed"}
    if message:
        job.result["analysis"] = analysis.as_dict()
        job.counts.over_limit = analysis.over_limit_documents
        log.error("Job %s: %s", job.id, message)
        raise InputTooLongError(message)


async def run_load(
    handle: JobHandle,
    db: StandardDatabase,
    opts: LoadOptions,
    source: DocumentSource,
) -> None:
    """Run a load job: stream documents from ``source`` into ``opts.collection``."""
    job = handle.job
    handle.bytes_probe = lambda: source.bytes_read
    handle.cleanup.append(source.close)
    eff = await effective_load_options(db, opts, opts.collection)
    job.config["embeddings"] = [s.describe() for s in eff.specs]
    job.config["key_field"] = eff.key_field
    job.config["on_duplicate"] = eff.on_duplicate
    job.config["from_profile"] = opts.embeddings is None
    if opts.analyze:
        await _analyze(handle, eff, _source_batches(source, eff.batch_size, opts.max_documents))
        return
    if long_input_any(eff, "error"):
        scan = source.reopen() if source.reopen else None
        try:
            await _prescan(
                handle,
                eff,
                _source_batches(scan, eff.batch_size, opts.max_documents) if scan else None,
            )
        finally:
            if scan is not None:
                scan.close()
    if not opts.dry_run:
        await _ensure_collection(db, opts.collection, opts.create_collection)
    collection = db.collection(opts.collection)

    concurrency = max(1, get_settings().embed_concurrency)
    pending: set[asyncio.Task[None]] = set()
    dims: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    offset = 0
    remaining = opts.max_documents

    async def process(batch: list[dict[str, Any]], start: int) -> None:
        keys = [_apply_key(doc, eff.key_field) for doc in batch]
        outcomes = await embed_documents(batch, eff.specs, store_metadata=eff.store_metadata)
        good: list[dict[str, Any]] = []
        good_pos: list[int] = []
        good_outcomes: list[DocOutcome] = []
        for i, (doc, outcome) in enumerate(zip(batch, outcomes, strict=True)):
            if outcome.over_limit:
                job.counts.over_limit += 1
            if outcome.error:
                handle.add_error(outcome.error, index=start + i, key=keys[i])
                continue
            if outcome.embedded:
                job.counts.embedded += 1
            elif outcome.skipped:
                job.counts.skipped += 1
            good.append(doc)
            good_pos.append(start + i)
            good_outcomes.append(outcome)
        _record_dims(dims, good + chunk_documents(good_outcomes), eff.specs)
        if opts.dry_run:
            for doc in good + chunk_documents(good_outcomes):
                if len(samples) < 5:
                    samples.append(_preview(doc, eff.specs))
        else:
            result: WriteResult = await asyncio.to_thread(
                insert_batch, collection, good, eff.on_duplicate
            )
            job.counts.written += result.written
            job.counts.updated += result.updated
            failed = {idx for idx, _, _ in result.errors}
            for idx, key, msg in result.errors:
                handle.add_error(msg, index=good_pos[idx], key=key)
            # Chunks only for parents that were written; replace so a re-run doesn't collide.
            chunks = chunk_documents([o for i, o in enumerate(good_outcomes) if i not in failed])
            await _write_chunks(collection, chunks, handle)
        handle.touch()

    try:
        while True:
            size = eff.batch_size if remaining is None else min(eff.batch_size, remaining)
            if size <= 0:
                break
            batch = await source.next_batch(size)
            if not batch:
                break
            job.counts.read += len(batch)
            if remaining is not None:
                remaining -= len(batch)
            task = asyncio.create_task(process(batch, offset))
            offset += len(batch)
            pending.add(task)
            if len(pending) >= concurrency:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    t.result()
        if pending:
            for fut in asyncio.as_completed(pending):
                await fut
            pending = set()
    finally:
        for t in pending:
            t.cancel()

    job.result["dimensions"] = dims
    if opts.dry_run:
        job.result["samples"] = samples
        return
    first_dim = dims.get(eff.specs[0].target) if eff.specs else None
    await _maybe_vector_index(db, opts.collection, eff, first_dim, handle)


# -------------------------------------------------------------------------- reembed


def _bind_path(path: str) -> list[str]:
    return split_path(path)


async def run_reembed(
    handle: JobHandle, db: StandardDatabase, collection_name: str, req: ReembedRequest
) -> None:
    job = handle.job
    if not await asyncio.to_thread(db.has_collection, collection_name):
        raise AppError(404, "collection_not_found", f"Collection '{collection_name}' not found")
    eff = await effective_load_options(db, req, collection_name)
    job.config["embeddings"] = [s.describe() for s in eff.specs]
    collection = db.collection(collection_name)

    bind: dict[str, Any] = {"@col": collection_name}
    filters: list[str] = []
    for i, (path, value) in enumerate(req.filter.items()):
        bind[f"f{i}"] = _bind_path(path)
        bind[f"v{i}"] = value
        filters.append(f"d.@f{i} == @v{i}")
    if req.only_missing:
        ors = []
        for i, spec in enumerate(eff.specs):
            bind[f"t{i}"] = _bind_path(spec.target)
            ors.append(f"d.@t{i} == null")
        filters.append("(" + " OR ".join(ors) + ")")
    where = "".join(f" FILTER {f}" for f in filters)

    count_query = f"RETURN LENGTH(FOR d IN @@col{where} RETURN 1)"
    total = list(await asyncio.to_thread(db.aql.execute, count_query, bind_vars=bind))  # type: ignore[arg-type]
    job.total = int(total[0]) if total else None
    if req.max_documents and job.total:
        job.total = min(job.total, req.max_documents)
    handle.touch()

    query = f"FOR d IN @@col FILTER d._key > @last{where} SORT d._key LIMIT @n RETURN d"

    async def batches() -> AsyncIterator[list[dict[str, Any]]]:
        last = ""
        remaining = req.max_documents
        while True:
            n = eff.batch_size if remaining is None else min(eff.batch_size, remaining)
            if n <= 0:
                return
            cursor = await asyncio.to_thread(
                db.aql.execute, query, bind_vars=cast(Any, {**bind, "last": last, "n": n})
            )
            batch: list[dict[str, Any]] = list(cursor)  # type: ignore[arg-type]
            if not batch:
                return
            last = batch[-1]["_key"]
            if remaining is not None:
                remaining -= len(batch)
            yield batch

    if req.analyze:
        await _analyze(handle, eff, batches())
        return
    await _prescan(handle, eff, batches())

    dims: dict[str, int] = {}
    async for batch in batches():
        job.counts.read += len(batch)
        outcomes = await embed_documents(batch, eff.specs, store_metadata=eff.store_metadata)
        updates: list[dict[str, Any]] = []
        for doc, outcome in zip(batch, outcomes, strict=True):
            if outcome.over_limit:
                job.counts.over_limit += 1
            if outcome.error:
                handle.add_error(outcome.error, key=doc["_key"])
            elif outcome.embedded:
                job.counts.embedded += 1
            elif outcome.skipped:
                job.counts.skipped += 1
            update: dict[str, Any] = {"_key": doc["_key"]}
            for spec in eff.specs:
                for path in (spec.target, spec.meta):
                    top = split_path(path)[0]
                    if top in doc:
                        update[top] = doc[top]
                vec = get_path(doc, spec.target)
                if isinstance(vec, list) and vec:
                    dims.setdefault(spec.target, len(vec))
            updates.append(update)
        result = await asyncio.to_thread(update_batch, collection, updates)
        job.counts.written += result.written
        job.counts.updated += result.updated
        for _, key, msg in result.errors:
            handle.add_error(msg, key=key)
        chunks = chunk_documents(outcomes)
        _record_dims(dims, chunks, eff.specs)
        await _write_chunks(collection, chunks, handle)
        handle.touch()
    job.result["dimensions"] = dims
    first_dim = dims.get(eff.specs[0].target) if eff.specs else None
    await _maybe_vector_index(db, collection_name, eff, first_dim, handle)
