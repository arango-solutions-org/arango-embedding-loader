from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.models.common import ERROR_RESPONSES
from app.providers.base import Availability, InputType, ModelInfo
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


class EmbedTestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256, examples=[["BRCA1 mutation"]])
    model: str | None = None
    dimensions: int | None = Field(None, ge=1)
    input_type: InputType = "document"
    provider_options: dict[str, Any] = Field(default_factory=dict)
    include_vectors: bool = Field(False, description="Return full vectors (otherwise first 8).")


class EmbedTestResponse(BaseModel):
    provider: str
    model: str
    dimension: int
    count: int
    elapsed_ms: float
    vectors: list[list[float]]


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
    provider = get_provider(name)
    options = provider.parse_options(body.provider_options)
    provider.ensure_available(options)
    model = body.model or provider.default_model or ""
    start = time.perf_counter()
    vectors = await provider.embed(
        body.texts,
        model=model,
        options=options,
        dimensions=body.dimensions,
        input_type=body.input_type,
    )
    elapsed = (time.perf_counter() - start) * 1000
    return EmbedTestResponse(
        provider=provider.name,
        model=model,
        dimension=len(vectors[0]) if vectors else 0,
        count=len(vectors),
        elapsed_ms=round(elapsed, 1),
        vectors=vectors if body.include_vectors else [v[:8] for v in vectors],
    )


@router.get("/presets", response_model=list[Preset], summary="List provider/model presets")
async def presets() -> list[Preset]:
    return list_presets()
