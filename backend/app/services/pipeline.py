"""The embedding pipeline: read → label/extract text → embed → write → (vector index)."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
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
from app.providers.presets import get_preset
from app.providers.registry import get_provider
from app.services import profiles as profile_store
from app.services.jobs import JobHandle
from app.services.sources import DocumentSource
from app.services.vector_index import ensure_vector_index
from app.services.writer import WriteResult, insert_batch, update_batch

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
    provider.ensure_available(options)
    model = model or provider.default_model
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


async def embed_documents(
    docs: list[dict[str, Any]],
    specs: list[ResolvedSpec],
    *,
    store_metadata: bool,
    only_pending: bool = False,
) -> list[DocOutcome]:
    """Embed ``docs`` in place according to ``specs``. Returns one outcome per document."""
    outcomes = [DocOutcome() for _ in docs]
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
        vectors = await spec.provider.embed(
            texts,
            model=spec.model,
            options=spec.options,
            dimensions=spec.dimensions,
            input_type="document",
        )
        for pos, vec in zip(positions, vectors, strict=True):
            set_path(docs[pos], spec.target, vec)
            if store_metadata:
                set_path(docs[pos], spec.meta, _meta(spec, "embedded", len(vec)))
            outcomes[pos].embedded = True
    return outcomes


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
        for i, (doc, outcome) in enumerate(zip(batch, outcomes, strict=True)):
            if outcome.error:
                handle.add_error(outcome.error, index=start + i, key=keys[i])
                continue
            if outcome.embedded:
                job.counts.embedded += 1
            elif outcome.skipped:
                job.counts.skipped += 1
            good.append(doc)
            good_pos.append(start + i)
        for spec in eff.specs:
            if spec.target not in dims:
                for doc in good:
                    vec = get_path(doc, spec.target)
                    if isinstance(vec, list) and vec:
                        dims[spec.target] = len(vec)
                        break
        if opts.dry_run:
            for doc in good:
                if len(samples) < 5:
                    samples.append(_preview(doc, eff.specs))
        else:
            result: WriteResult = await asyncio.to_thread(
                insert_batch, collection, good, eff.on_duplicate
            )
            job.counts.written += result.written
            job.counts.updated += result.updated
            for idx, key, msg in result.errors:
                handle.add_error(msg, index=good_pos[idx], key=key)
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
    last = ""
    dims: dict[str, int] = {}
    remaining = req.max_documents
    while True:
        n = eff.batch_size if remaining is None else min(eff.batch_size, remaining)
        if n <= 0:
            break
        cursor = await asyncio.to_thread(
            db.aql.execute, query, bind_vars=cast(Any, {**bind, "last": last, "n": n})
        )
        batch: list[dict[str, Any]] = list(cursor)  # type: ignore[arg-type]
        if not batch:
            break
        last = batch[-1]["_key"]
        job.counts.read += len(batch)
        if remaining is not None:
            remaining -= len(batch)
        outcomes = await embed_documents(batch, eff.specs, store_metadata=eff.store_metadata)
        updates: list[dict[str, Any]] = []
        for doc, outcome in zip(batch, outcomes, strict=True):
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
        handle.touch()
    job.result["dimensions"] = dims
    first_dim = dims.get(eff.specs[0].target) if eff.specs else None
    await _maybe_vector_index(db, collection_name, eff, first_dim, handle)
