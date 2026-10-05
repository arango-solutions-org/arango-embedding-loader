"""Cohere embeddings (v2 API)."""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field, SecretStr

from app.providers.base import (
    Availability,
    EmbeddingProvider,
    InputType,
    ModelInfo,
    ProviderOptions,
)


class CohereOptions(ProviderOptions):
    api_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(v) if (v := os.getenv("COHERE_API_KEY")) else None,
        description="API key. Defaults to $COHERE_API_KEY.",
    )
    base_url: str | None = Field(None, description="Override API base URL.")
    truncate: str = Field("END", description="NONE | START | END")
    timeout: float = Field(60.0, description="Request timeout in seconds.")


class CohereProvider(EmbeddingProvider):
    name = "cohere"
    label = "Cohere"
    description = "Cohere Embed models (v4 and v3, English and multilingual)."
    Options = CohereOptions
    requires = ("cohere",)
    install_hint = "uv sync --extra cohere"
    max_batch_size = 96
    default_model = "embed-v4.0"
    supports_dimensions = True
    known_models = (
        ModelInfo(id="embed-v4.0", dimension=1536),
        ModelInfo(id="embed-english-v3.0", dimension=1024),
        ModelInfo(id="embed-multilingual-v3.0", dimension=1024),
        ModelInfo(id="embed-english-light-v3.0", dimension=384),
        ModelInfo(id="embed-multilingual-light-v3.0", dimension=384),
    )

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def check_configuration(self, options: ProviderOptions) -> Availability:
        assert isinstance(options, CohereOptions)
        if options.api_key is None:
            return Availability(
                available=False, reason="No API key: set COHERE_API_KEY or pass api_key."
            )
        return Availability(available=True)

    def _client(self, options: CohereOptions) -> Any:
        assert options.api_key is not None
        secret = options.api_key.get_secret_value()
        key = f"{secret}|{options.base_url}"
        client = self._clients.get(key)
        if client is None:
            import cohere

            kwargs: dict[str, Any] = {"api_key": secret, "timeout": options.timeout}
            if options.base_url:
                kwargs["base_url"] = options.base_url
            client = cohere.AsyncClientV2(**kwargs)
            self._clients[key] = client
        return client

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        assert isinstance(options, CohereOptions)
        kwargs: dict[str, Any] = {
            "texts": texts,
            "model": model,
            "input_type": "search_document" if input_type == "document" else "search_query",
            "embedding_types": ["float"],
            "truncate": options.truncate,
        }
        if dimensions:
            kwargs["output_dimension"] = dimensions
        response = await self._client(options).embed(**kwargs)
        return [list(v) for v in response.embeddings.float_]
