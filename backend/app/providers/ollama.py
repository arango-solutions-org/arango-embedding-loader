"""Ollama embeddings (local or remote Ollama server)."""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import Field

from app.providers.base import EmbeddingProvider, InputType, ModelInfo, ProviderOptions


class OllamaOptions(ProviderOptions):
    host: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        description="Ollama server URL. Defaults to $OLLAMA_HOST.",
    )
    keep_alive: str | None = Field(None, description="How long to keep the model loaded, e.g. 5m.")
    truncate: bool = Field(True, description="Truncate inputs that exceed the context length.")
    timeout: float = Field(120.0, description="Request timeout in seconds.")


class OllamaProvider(EmbeddingProvider):
    name = "ollama"
    label = "Ollama"
    description = "Embedding models served by Ollama (nomic-embed-text, mxbai-embed-large, ...)."
    Options = OllamaOptions
    max_batch_size = 64
    max_concurrency = 2
    default_model = "nomic-embed-text"
    known_models = (
        ModelInfo(id="nomic-embed-text", dimension=768),
        ModelInfo(id="mxbai-embed-large", dimension=1024),
        ModelInfo(id="all-minilm", dimension=384),
        ModelInfo(id="bge-m3", dimension=1024),
        ModelInfo(id="snowflake-arctic-embed", dimension=1024),
    )

    @staticmethod
    def _url(options: OllamaOptions, path: str) -> str:
        host = options.host if "://" in options.host else f"http://{options.host}"
        return host.rstrip("/") + path

    async def list_models(self, options: ProviderOptions) -> list[ModelInfo]:
        assert isinstance(options, OllamaOptions)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self._url(options, "/api/tags"))
                resp.raise_for_status()
                names = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return list(self.known_models)
        return [
            ModelInfo(id=n, dimension=self.known_dimension(n.split(":")[0])) for n in names
        ] or list(self.known_models)

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        assert isinstance(options, OllamaOptions)
        body: dict[str, Any] = {"model": model, "input": texts, "truncate": options.truncate}
        if options.keep_alive:
            body["keep_alive"] = options.keep_alive
        async with httpx.AsyncClient(timeout=options.timeout) as client:
            resp = await client.post(self._url(options, "/api/embed"), json=body)
            resp.raise_for_status()
            return [list(v) for v in resp.json()["embeddings"]]
