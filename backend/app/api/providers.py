from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.ingest.text import get_path
from app.models.common import ERROR_RESPONSES
from app.models.embedding import (
    ChunkOverlapField,
    EmbeddingFieldSpec,
    LongInputModeField,
    LongInputSettingsMixin,
    MaxTokensField,
    TokenCharsField,
    WarnOnLongInputField,
)
from app.providers.base import Availability, InputType, ModelInfo
from app.providers.long_input import DEFAULT_TOKEN_CHARS
from app.providers.presets import Preset, list_presets
from app.providers.registry import get_provider, list_providers

router = APIRouter(tags=["Providers"], responses=ERROR_RESPONSES)


class ProviderInfo(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    name: str
    label: str
    description: str
    available: bool
    reason: str | None = None
    default_model: str | None
    supports_dimensions: bool
    max_batch_size: int
    install_hint: str | None
    options_schema: dict[str, Any] = Field(..., description="JSON Schema of `provider_options`.")
    known_models: list[ModelInfo]


class EmbedTestRequest(LongInputSettingsMixin):
    """Texts are embedded exactly as a load would embed them, including ``long_input``."""

    texts: list[str] = Field(..., min_length=1, max_length=256, examples=[["BRCA1 mutation"]])
    model: str | None = None
    dimensions: int | None = Field(None, ge=1)
    input_type: InputType = "document"
    provider_options: dict[str, Any] = Field(default_factory=dict)
    include_vectors: bool = Field(False, description="Return full vectors (otherwise first 8).")
    long_input: LongInputModeField = "chunk_docs"
    chunk_overlap_tokens: ChunkOverlapField = 25
    max_tokens: MaxTokensField = None
    token_chars: TokenCharsField = DEFAULT_TOKEN_CHARS
    warn_on_long_input: WarnOnLongInputField = True


class EmbedTestResult(BaseModel):
    """What a load would store for one text: its `<target>_meta` and its vector(s)."""

    status: str = Field(
        ...,
        description=(
            "`embedded`, `chunked` (`chunk_docs`: one vector per chunk), `skipped_too_long` "
            "or `skipped` (empty text)."
        ),
    )
    meta: dict[str, Any] = Field(
        ..., description="The `<target>_meta` a load would store (token counts, truncation...)."
    )
    vectors: list[list[float]] = Field(
        ..., description="One vector, one per chunk (`chunk_docs`), or none when skipped."
    )


class EmbedTestResponse(BaseModel):
    provider: str
    model: str
    dimension: int
    count: int = Field(..., description="Number of vectors (more than texts when chunked).")
    elapsed_ms: float
    vectors: list[list[float]] = Field(..., description="Every vector, in text order.")
    results: list[EmbedTestResult] = Field(..., description="One per text, in order.")


@router.get("/providers", response_model=list[ProviderInfo], summary="List embedding providers")
async def providers() -> list[ProviderInfo]:
    out = []
    for provider in list_providers():
        try:
            status = provider.availability()
        except Exception as exc:  # noqa: BLE001
            status = Availability(available=False, reason=str(exc))
        out.append(ProviderInfo(**provider.describe(), **status.model_dump()))
    return out


@router.get(
    "/providers/{name}/models",
    response_model=list[ModelInfo],
    summary="List models for a provider",
)
async def provider_models(name: str) -> list[ModelInfo]:
    provider = get_provider(name)
    return await provider.list_models(provider.parse_options(None))


@router.post(
    "/providers/{name}/embed",
    response_model=EmbedTestResponse,
    summary="Embed sample texts (connectivity / dimension check)",
)
async def provider_embed(name: str, body: EmbedTestRequest) -> EmbedTestResponse:
    # Built as an embedding spec and run through the load pipeline, so the result (planning,
    # long_input, provider truncation, batching, retries) is what a load would produce.
    from app.services.pipeline import embed_documents, resolve_spec

    spec = EmbeddingFieldSpec(
        source_field="text",
        target_field="embedding",
        provider=name,
        model=body.model,
        dimensions=body.dimensions,
        provider_options=body.provider_options,
        long_input=body.long_input,
        chunk_overlap_tokens=body.chunk_overlap_tokens,
        max_tokens=body.max_tokens,
        token_chars=body.token_chars,
        warn_on_long_input=body.warn_on_long_input,
    )
    resolved = resolve_spec(spec)
    docs: list[dict[str, Any]] = [{"_key": str(i), "text": t} for i, t in enumerate(body.texts)]
    start = time.perf_counter()
    outcomes = await embed_documents(
        docs, [resolved], store_metadata=True, input_type=body.input_type
    )
    elapsed = (time.perf_counter() - start) * 1000

    def shown(v: list[float]) -> list[float]:
        return v if body.include_vectors else v[:8]

    results: list[EmbedTestResult] = []
    dimension = 0
    for doc, outcome in zip(docs, outcomes, strict=True):
        meta = get_path(doc, resolved.meta) or {}
        vector = get_path(doc, resolved.target)
        if outcome.chunks:
            ordered = sorted(outcome.chunks, key=lambda c: c[resolved.meta]["chunk_index"])
            vectors = [c[resolved.target] for c in ordered]
        else:
            vectors = [vector] if isinstance(vector, list) else []
        dimension = dimension or (len(vectors[0]) if vectors else 0)
        results.append(
            EmbedTestResult(
                status=str(meta.get("status", "embedded")),
                meta={k: v for k, v in meta.items() if k != "at"},
                vectors=[shown(v) for v in vectors],
            )
        )
    flat = [v for r in results for v in r.vectors]
    return EmbedTestResponse(
        provider=resolved.provider.name,
        model=resolved.model,
        dimension=dimension,
        count=len(flat),
        elapsed_ms=round(elapsed, 1),
        vectors=flat,
        results=results,
    )


@router.get("/presets", response_model=list[Preset], summary="List provider/model presets")
async def presets() -> list[Preset]:
    return list_presets()
