"""The embedding pipeline: read → label/extract text → embed → write → (vector index)."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable
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
from app.providers.base import EmbeddingProvider, InputType, ProviderError, ProviderOptions
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
    #: Fields split by ``chunk_docs``, and the values the split replaced (path -> value, or
    #: ``_MISSING``), to put back if the document or its chunks fail.
    split_specs: list[ResolvedSpec] = field(default_factory=list)
    restore: dict[str, Any] = field(default_factory=dict)
    #: The chunks failed to write (see :func:`write_chunks_first`). The document is still
    #: written, with its split fields restored and ``status: "error"``.
    chunk_write_failed: bool = False

    def undo_split(self, doc: dict[str, Any]) -> None:
        for path, value in self.restore.items():
            _restore_path(doc, path, value)
        self.chunks = []


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


#: ``<target>_meta.reason`` on a chunk document for the fields it was not split for.
CHUNK_DOCUMENT = "chunk document"


def _is_pending(doc: dict[str, Any], spec: ResolvedSpec) -> bool:
    return get_path(doc, f"{spec.meta}.status") is None


def mark_failed(doc: dict[str, Any], specs: list[ResolvedSpec], reason: str, message: str) -> None:
    """Mark every field of ``doc`` not yet processed as failed (``status: "error"``)."""
    for spec in specs:
        if _is_pending(doc, spec):
            set_path(doc, spec.meta, _meta(spec, "error", reason=reason, error=message))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def is_unchanged_chunked(doc: dict[str, Any], spec: ResolvedSpec, text: str | None) -> bool:
    """``doc`` was split by ``chunk_docs`` and nothing that produced its chunks has changed.

    Its vector is ``null``, so ``only_missing`` would select it every time. It is re-embedded
    when its text, the model or ``long_input`` changes, or when its ``<target>_meta`` is removed.
    """
    meta = get_path(doc, spec.meta)
    return (
        text is not None
        and isinstance(meta, dict)
        and meta.get("status") == "chunked"
        and spec.spec.long_input == "chunk_docs"
        and meta.get("long_input") == "chunk_docs"
        and meta.get("provider") == spec.provider.name
        and meta.get("model") == spec.model
        and meta.get("text_hash") == text_hash(text)
    )


def is_settled(doc: dict[str, Any], spec: ResolvedSpec) -> bool:
    """Nothing for ``only_missing`` to do: the vector is there, the chunks are current, or this
    is a chunk document of another field (whose vectors for this field are left off on purpose).
    """
    if get_path(doc, spec.target) is not None:
        return True
    if get_path(doc, f"{spec.meta}.reason") == CHUNK_DOCUMENT:
        return True
    return is_unchanged_chunked(doc, spec, spec.text_for(doc))


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
                    _meta(other, "skipped", reason=CHUNK_DOCUMENT, parent_key=parent_key),
                )
            out.append(child)
    return out


class InputTooLongError(AppError):
    def __init__(self, message: str):
        super().__init__(422, "input_too_long", message)


class StillTooLongError(ProviderError):
    """The provider kept rejecting a text as too long after every smaller attempt."""


async def embed_documents(
    docs: list[dict[str, Any]],
    specs: list[ResolvedSpec],
    *,
    store_metadata: bool,
    only_pending: bool = False,
    keep_chunked: bool = False,
    input_type: InputType = "document",
    errors_per_document: bool = False,
) -> list[DocOutcome]:
    """Embed ``docs`` in place according to ``specs``. Returns one outcome per document.

    Documents whose text is split into pieces (``chunk_docs``) get the extra documents to
    write in ``DocOutcome.chunks``. A text that is too long with ``long_input: error`` raises
    :class:`InputTooLongError`, which stops the run. With ``keep_chunked``, fields whose
    chunks are current (see :func:`is_unchanged_chunked`) are left as they are.

    With ``errors_per_document`` (the watcher), a text too long under ``long_input: error``, or
    one the provider still rejects as too long after every smaller attempt
    (:class:`StillTooLongError`), does not raise: that document gets
    ``<target>_meta.status: "error"`` and an error outcome, and the others are embedded.

    A chunk document's other fields are never embedded (their vectors are left off on purpose),
    and a chunk document that is embedded again keeps its link to its original.
    """
    outcomes = [DocOutcome() for _ in docs]
    splits: dict[int, list[_Split]] = {}
    links = _chunk_links(docs, specs)

    def too_long(
        spec: ResolvedSpec, pos: int, exc: InputTooLongError | StillTooLongError, stats: dict
    ) -> None:
        if not errors_per_document:
            raise exc
        outcomes[pos].error = str(exc.detail if isinstance(exc, AppError) else exc)
        if store_metadata:
            set_path(docs[pos], spec.meta, _meta(spec, "error", reason="too long", **stats))

    for spec in specs:
        texts: list[str] = []
        positions: list[int] = []
        for i, doc in enumerate(docs):
            if outcomes[i].error or (only_pending and not _is_pending(doc, spec)):
                continue
            if get_path(doc, f"{spec.meta}.reason") == CHUNK_DOCUMENT:
                continue
            text = spec.text_for(doc)
            if keep_chunked and is_unchanged_chunked(doc, spec, text):
                continue
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
            input_type=input_type,
            long_input=spec.spec.long_input_settings,
        )
        if len(plans) != len(texts):
            raise AppError(
                500, "plan_mismatch", f"Provider '{spec.provider.name}' planned the wrong count"
            )
        for pos, plan in zip(positions, plans, strict=True):
            try:
                _check_plan(spec, docs, pos, plan, outcomes)
            except InputTooLongError as exc:
                too_long(spec, pos, exc, plan.stats)

        plans, plan_vectors = await _embed_plans(
            spec, docs, positions, texts, plans, input_type, outcomes, too_long
        )
        for pos, text, plan, piece_vectors in zip(
            positions, texts, plans, plan_vectors, strict=True
        ):
            doc = docs[pos]
            if plan.action == "error":
                continue  # recorded by too_long()
            if plan.action == "skip":
                outcomes[pos].skipped = True
                if store_metadata:
                    set_path(doc, spec.meta, _meta(spec, "skipped_too_long", **plan.stats))
                continue
            if plan.combine == "split":
                outcome = outcomes[pos]
                outcome.split_specs.append(spec)
                for path in (spec.target, spec.meta):
                    outcome.restore.setdefault(path, get_path(doc, path, _MISSING))
                set_path(doc, spec.target, None)
                if store_metadata:
                    meta = _meta(spec, "chunked", text_hash=text_hash(text), **plan.stats)
                    set_path(doc, spec.meta, meta)
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
            continue
        # Another field of this document failed, so its chunks are not written. Put back what
        # the split replaced, so the stored vector is not lost and the field is not marked as
        # chunked without chunks.
        outcomes[pos].undo_split(docs[pos])
    _restore_chunk_links(docs, links)
    return outcomes


_MISSING = object()


def _restore_path(doc: dict[str, Any], path: str, value: Any) -> None:
    """Set ``path`` back to ``value``, or remove it if it was missing (``_MISSING``)."""
    if value is not _MISSING:
        set_path(doc, path, value)
        return
    parts = split_path(path)
    node: Any = doc
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict):
        node.pop(parts[-1], None)


_LINK_FIELDS = ("parent_key", "chunk_index", "chunk_count")


def _chunk_links(
    docs: list[dict[str, Any]], specs: list[ResolvedSpec]
) -> dict[tuple[int, str], dict[str, Any]]:
    """For chunk documents among ``docs``: their link to the original, per field."""
    links: dict[tuple[int, str], dict[str, Any]] = {}
    for i, doc in enumerate(docs):
        for spec in specs:
            meta = get_path(doc, spec.meta)
            if isinstance(meta, dict) and meta.get("parent_key") is not None:
                links[(i, spec.meta)] = {k: meta[k] for k in _LINK_FIELDS if k in meta}
    return links


def _restore_chunk_links(
    docs: list[dict[str, Any]], links: dict[tuple[int, str], dict[str, Any]]
) -> None:
    """A chunk document embedded again (e.g. a full re-embed) keeps its link to its original."""
    for (i, meta_path), link in links.items():
        meta = get_path(docs[i], meta_path)
        if isinstance(meta, dict):
            for key, value in link.items():
                meta.setdefault(key, value)


def _check_plan(
    spec: ResolvedSpec,
    docs: list[dict[str, Any]],
    pos: int,
    plan: TextPlan,
    outcomes: list[DocOutcome],
) -> None:
    """Record and log an over-limit text; raise for ``long_input: error``."""
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


#: How often a text is planned again, against a smaller limit, after the provider rejects it as
#: too long. Each attempt uses 70% of the previous limit.
_SHRINK_ATTEMPTS = 6
_SHRINK_FACTOR = 0.7


async def _embed_plans(
    spec: ResolvedSpec,
    docs: list[dict[str, Any]],
    positions: list[int],
    texts: list[str],
    plans: list[TextPlan],
    input_type: InputType,
    outcomes: list[DocOutcome],
    too_long: Callable[[ResolvedSpec, int, InputTooLongError | StillTooLongError, dict], None],
) -> tuple[list[TextPlan], list[list[list[float]]]]:
    """Embed the pieces of every plan. Returns the plans (some may be re-planned) and, per plan,
    the vectors of its pieces (none when it is skipped).

    Requests are sized by token count. If the provider still rejects a request as too long
    (estimated counts can be low), each text is sent on its own, and a text the provider rejects
    is planned again against a smaller limit until it fits.
    """
    flat: list[str] = []
    counts: list[int] = []
    for plan in plans:
        if plan.action == "embed":
            flat.extend(plan.pieces)
            counts.extend(plan.piece_tokens or [0] * len(plan.pieces))
    try:
        vectors = await spec.provider.embed(
            flat,
            model=spec.model,
            options=spec.options,
            dimensions=spec.dimensions,
            input_type=input_type,
            token_counts=counts if len(counts) == len(flat) else None,
        )
    except ProviderError as exc:
        if not spec.provider.is_input_too_long(exc):
            raise
        log.warning(
            "%s/%s rejected a request as too long (%s); embedding each text on its own",
            spec.provider.name,
            spec.model,
            exc,
        )
    else:
        out: list[list[list[float]]] = []
        cursor = 0
        for plan in plans:
            n = len(plan.pieces) if plan.action == "embed" else 0
            out.append(vectors[cursor : cursor + n])
            cursor += n
        return plans, out

    new_plans: list[TextPlan] = []
    out = []
    for pos, text, plan in zip(positions, texts, plans, strict=True):
        try:
            plan, vecs = await _embed_one(spec, docs, pos, text, plan, input_type, outcomes)
        except (InputTooLongError, StillTooLongError) as exc:
            too_long(spec, pos, exc, plan.stats)  # raises unless errors are per document
            plan, vecs = dataclasses.replace(plan, action="error"), []
        new_plans.append(plan)
        out.append(vecs)
    return new_plans, out


async def _embed_one(
    spec: ResolvedSpec,
    docs: list[dict[str, Any]],
    pos: int,
    text: str,
    plan: TextPlan,
    input_type: InputType,
    outcomes: list[DocOutcome],
) -> tuple[TextPlan, list[list[float]]]:
    settings = spec.spec.long_input_settings
    original: int | None = None
    for _ in range(_SHRINK_ATTEMPTS):
        if plan.action != "embed":
            return plan, []
        try:
            vectors = await spec.provider.embed(
                plan.pieces,
                model=spec.model,
                options=spec.options,
                dimensions=spec.dimensions,
                input_type=input_type,
                token_counts=plan.piece_tokens or None,
            )
            return plan, vectors
        except ProviderError as exc:
            if not spec.provider.is_input_too_long(exc):
                raise
            if settings.mode == "error":
                raise InputTooLongError(
                    f"{spec.provider.name}/{spec.model} rejected the text for '{spec.target}' "
                    f"in document {_doc_ref(docs[pos], pos)} as too long (long_input: error; "
                    f"the estimated count was {plan.stats.get('input_tokens')} tokens): {exc}"
                ) from exc
            current = plan.stats.get("max_tokens") or plan.stats.get("input_tokens") or 1
            limit = max(1, int(current * _SHRINK_FACTOR))
            log.warning(
                "%s/%s rejected the text for '%s' in document %s as too long; planning it "
                "again with a %s-token limit (was %s)",
                spec.provider.name,
                spec.model,
                spec.target,
                _doc_ref(docs[pos], pos),
                limit,
                current,
            )
            (plan,) = await asyncio.to_thread(
                spec.provider.plan_inputs,
                [text],
                model=spec.model,
                options=spec.options,
                input_type=input_type,
                long_input=dataclasses.replace(settings, max_tokens=limit),
                allow_native=False,
            )
            original = original or current
            plan.stats["max_tokens_reduced_from"] = original
            _check_plan(spec, docs, pos, plan, outcomes)
    raise StillTooLongError(
        f"{spec.provider.name}/{spec.model} kept rejecting the text for '{spec.target}' in "
        f"document {_doc_ref(docs[pos], pos)} as too long after {_SHRINK_ATTEMPTS} smaller "
        "attempts",
        status_code=502,
    )


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


@dataclass
class ChunkWrite:
    written: int = 0
    removed: int = 0
    keys: list[str] = field(default_factory=list)
    #: Keys of chunk documents deleted as left over from an earlier, longer split.
    removed_keys: list[str] = field(default_factory=list)
    #: Keys of chunk documents deleted because another chunk of the same document failed.
    failed_keys: list[str] = field(default_factory=list)


def write_chunks_first(
    db: StandardDatabase,
    collection: Any,
    docs: list[dict[str, Any]],
    outcomes: list[DocOutcome],
    specs: list[ResolvedSpec],
    *,
    store_metadata: bool,
    skip_existing_parents: bool = False,
) -> ChunkWrite:
    """Write the chunk documents of ``docs`` BEFORE the documents themselves are written.

    If any chunk of a document fails, that document's written chunks are deleted, its split
    fields are put back as they were (so a stored vector is not replaced by ``null``), its
    ``<target>_meta`` is marked ``error`` (``chunk write failed``) and its outcome gets the
    error. Callers then write the documents as usual.

    ``skip_existing_parents`` (loads with ``on_duplicate`` ``error``/``ignore``): documents that
    already exist will not be written, so their chunks must not be either; they would replace
    the chunks of the stored version.
    """
    out = ChunkWrite()
    parents = [i for i, o in enumerate(outcomes) if o.chunks and not o.error]
    if not parents:
        return out
    # The same key twice in one batch: only the copy that will be stored may write chunks.
    # Inserts keep the first copy when existing documents are not replaced, the last otherwise.
    by_key: dict[str, int] = {}
    for i in parents:
        parent_key = str(docs[i]["_key"])
        if parent_key not in by_key or not skip_existing_parents:
            by_key[parent_key] = i
    for i in parents:
        if by_key[str(docs[i]["_key"])] != i:
            outcomes[i].chunks = []
    parents = sorted(by_key.values())
    if skip_existing_parents:
        cursor = db.aql.execute(
            "FOR k IN @keys FILTER DOCUMENT(@@col, k) != null RETURN k",
            bind_vars=cast(
                Any, {"keys": [str(docs[i]["_key"]) for i in parents], "@col": collection.name}
            ),
        )
        existing = set(cast(Any, cursor))
        for i in parents:
            if str(docs[i]["_key"]) in existing:
                outcomes[i].chunks = []
        parents = [i for i in parents if outcomes[i].chunks]
    chunks: list[dict[str, Any]] = []
    owners: list[int] = []
    for i in parents:
        chunks.extend(outcomes[i].chunks)
        owners.extend([i] * len(outcomes[i].chunks))
    if not chunks:
        return out
    result = insert_batch(collection, chunks, "replace")
    failed: dict[int, str] = {}
    for idx, key, msg in result.errors:
        failed.setdefault(owners[idx], f"chunk {key}: {msg}")
    if failed:
        orphans = [c["_key"] for c, o in zip(chunks, owners, strict=True) if o in failed]
        collection.delete_many([{"_key": k} for k in orphans], silent=True)
        out.failed_keys = [str(k) for k in orphans]
        for i, msg in failed.items():
            outcome, doc = outcomes[i], docs[i]
            outcome.undo_split(doc)
            if store_metadata:
                for spec in outcome.split_specs:
                    set_path(doc, spec.meta, _meta(spec, "error", reason="chunk write failed"))
            outcome.embedded = False
            outcome.chunk_write_failed = True
            outcome.error = f"Could not write the chunk documents of '{doc.get('_key')}': {msg}"
            log.error("%s", outcome.error)
    good = [c for c, o in zip(chunks, owners, strict=True) if o not in failed]
    out.written = len(good)
    out.keys = [str(c["_key"]) for c in good]
    out.removed_keys = remove_stale_chunks(db, collection, good, specs)
    out.removed = len(out.removed_keys)
    return out


def _count_chunks(handle: JobHandle, write: ChunkWrite) -> None:
    job = handle.job
    job.counts.chunks += write.written
    if write.removed:
        job.result["stale_chunks_removed"] = (
            job.result.get("stale_chunks_removed", 0) + write.removed
        )


#: Chunk keys checked per query when looking for chunks left from an earlier, longer split.
_STALE_WINDOW = 64


def remove_stale_chunks(
    db: StandardDatabase,
    collection: Any,
    chunks: list[dict[str, Any]],
    specs: list[ResolvedSpec],
) -> list[str]:
    """Delete chunk documents left over from an earlier split of the same text into more chunks.

    Only documents that are certainly such chunks are removed: the key is exactly
    ``<parent>_<target>_chunk_<n>`` with ``n`` at or past the new chunk count, and the
    document's ``<target>_meta`` names the same ``parent_key`` and ``chunk_index`` ``n``.
    Returns the keys removed.
    """
    removed: list[str] = []
    for spec in specs:
        parents: dict[str, tuple[str, int]] = {}  # parent key -> (key prefix, chunk count)
        for chunk in chunks:
            meta = get_path(chunk, spec.meta)
            if not isinstance(meta, dict) or meta.get("chunk_index") != 0:
                continue
            parent, count = meta.get("parent_key"), meta.get("chunk_count")
            key = str(chunk.get("_key", ""))
            if parent is None or not isinstance(count, int) or not key.endswith("_chunk_0"):
                continue
            parents[str(parent)] = (key[: -len("0")], count)
        start = {p: count for p, (_, count) in parents.items()}
        while start:
            probe = [{"parent": p, "prefix": parents[p][0], "from": n} for p, n in start.items()]
            cursor = db.aql.execute(
                """
                FOR p IN @probe
                  FOR n IN p.from..(p.from + @window - 1)
                    LET d = DOCUMENT(@@col, CONCAT(p.prefix, n))
                    FILTER d != null AND d.@pk == p.parent AND d.@ci == n
                    RETURN {parent: p.parent, key: d._key}
                """,
                bind_vars=cast(
                    Any,
                    {
                        "probe": probe,
                        "window": _STALE_WINDOW,
                        "@col": collection.name,
                        "pk": [*split_path(spec.meta), "parent_key"],
                        "ci": [*split_path(spec.meta), "chunk_index"],
                    },
                ),
            )
            found: dict[str, list[str]] = {}
            for row in cast(Any, cursor):
                found.setdefault(row["parent"], []).append(row["key"])
            keys = [k for ks in found.values() for k in ks]
            if keys:
                collection.delete_many([{"_key": k} for k in keys], silent=True)
                removed.extend(keys)
            # A full window may mean more beyond it.
            start = {
                p: start[p] + _STALE_WINDOW for p, ks in found.items() if len(ks) == _STALE_WINDOW
            }
    return removed


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
        if source.reopen is None and source.spool is not None:
            # Read-once input (URLs): save it to disk once, so it can be scanned and then
            # loaded without downloading it twice.
            spooled = await asyncio.to_thread(source.spool)
            source.close()
            source = spooled
            handle.cleanup.append(source.close)
            job.result["spooled"] = True
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

    write_lock = asyncio.Lock()

    async def process(batch: list[dict[str, Any]], start: int) -> None:
        keys = [_apply_key(doc, eff.key_field) for doc in batch]
        outcomes = await embed_documents(batch, eff.specs, store_metadata=eff.store_metadata)
        # Chunks, then documents, one batch at a time: concurrent batches with the same key
        # could otherwise leave a document with another version's chunks.
        async with write_lock:
            if not opts.dry_run:
                # Chunks first: a document whose chunks fail is written with an error status.
                write = await asyncio.to_thread(
                    write_chunks_first,
                    db,
                    collection,
                    batch,
                    outcomes,
                    eff.specs,
                    store_metadata=eff.store_metadata,
                    skip_existing_parents=eff.on_duplicate in ("error", "ignore"),
                )
                _count_chunks(handle, write)
            good: list[dict[str, Any]] = []
            good_pos: list[int] = []
            good_outcomes: list[DocOutcome] = []
            for i, (doc, outcome) in enumerate(zip(batch, outcomes, strict=True)):
                if outcome.over_limit:
                    job.counts.over_limit += 1
                if outcome.error:
                    handle.add_error(outcome.error, index=start + i, key=keys[i])
                    if not outcome.chunk_write_failed:
                        continue
                    # Written anyway, with its split fields restored and status "error", so a
                    # stored version is not left marked as chunked without its chunks.
                elif outcome.embedded:
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


def _parent_in(doc: dict[str, Any], specs: list[ResolvedSpec], keys: set[str]) -> bool:
    """``doc`` is a chunk document whose original's key is in ``keys``."""
    for spec in specs:
        parent = get_path(doc, f"{spec.meta}.parent_key")
        if parent is not None and str(parent) != str(doc.get("_key")) and str(parent) in keys:
            return True
    return False


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
    matched = int(total[0]) if total else None
    unchanged = 0

    def set_total() -> None:
        job.total = None if matched is None else matched - unchanged
        if req.max_documents and job.total is not None:
            job.total = min(job.total, req.max_documents)

    set_total()
    handle.touch()

    query = f"FOR d IN @@col FILTER d._key > @last{where} SORT d._key LIMIT @n RETURN d"
    #: Chunk documents written by this run. Their keys can sort after the current page, so the
    #: query would read them again; they were just embedded and are skipped.
    written_chunks: set[str] = set()

    async def batches(count_unchanged: bool = False) -> AsyncIterator[list[dict[str, Any]]]:
        nonlocal unchanged
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
            batch = [d for d in batch if d["_key"] not in written_chunks]
            if req.only_missing:
                # Originals split by chunk_docs have a null vector, so the query selects them.
                # Leave out those whose chunks are current, and take them off the total.
                kept = [d for d in batch if not all(is_settled(d, s) for s in eff.specs)]
                if count_unchanged and len(kept) < len(batch):
                    dropped = len(batch) - len(kept)
                    job.result["chunked_unchanged"] = (
                        job.result.get("chunked_unchanged", 0) + dropped
                    )
                    unchanged += dropped
                    set_total()
                batch = kept
            # max_documents counts documents processed, not ones skipped above, so repeated
            # "next N" runs always make progress.
            if remaining is not None:
                remaining -= len(batch)
            if batch:
                yield batch

    if req.analyze:
        await _analyze(handle, eff, batches())
        return
    await _prescan(handle, eff, batches())

    dims: dict[str, int] = {}
    #: Chunk documents deleted by this run as left over from a longer split.
    gone: set[str] = set()

    async def process(batch: list[dict[str, Any]]) -> None:
        job.counts.read += len(batch)
        outcomes = await embed_documents(
            batch, eff.specs, store_metadata=eff.store_metadata, keep_chunked=req.only_missing
        )
        # Chunks first: a document whose chunks fail keeps its previous vector.
        write = await asyncio.to_thread(
            write_chunks_first,
            db,
            collection,
            batch,
            outcomes,
            eff.specs,
            store_metadata=eff.store_metadata,
        )
        _count_chunks(handle, write)
        written_chunks.update(write.keys)
        gone.update(write.removed_keys)
        gone.update(write.failed_keys)
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
        _record_dims(dims, chunk_documents(outcomes), eff.specs)
        handle.touch()

    async for batch in batches(count_unchanged=True):
        # A chunk document whose original is in the same batch waits until the original is
        # done: if the original is split again, its chunks are rewritten (or removed) and the
        # old chunk must not be embedded from its old text and written back over them.
        keys = {str(d["_key"]) for d in batch}
        later = [d for d in batch if _parent_in(d, eff.specs, keys)]
        if later:
            first = [d for d in batch if not _parent_in(d, eff.specs, keys)]
            if first:
                await process(first)
            batch = [d for d in later if d["_key"] not in written_chunks and d["_key"] not in gone]
            if not batch:
                continue
        await process(batch)
    # The starting total counts documents matched up front. Some are then left out (current
    # chunks, chunk documents this run re-created), so end on the number actually processed.
    job.total = job.counts.read
    job.result["dimensions"] = dims
    first_dim = dims.get(eff.specs[0].target) if eff.specs else None
    await _maybe_vector_index(db, collection_name, eff, first_dim, handle)
