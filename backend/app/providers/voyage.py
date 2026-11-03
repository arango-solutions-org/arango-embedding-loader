"""Voyage AI embeddings."""

from __future__ import annotations

import logging
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
from app.providers.long_input import HFTokenizer, LongInputMode, Tokenizer

log = logging.getLogger(__name__)


class VoyageOptions(ProviderOptions):
    api_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(v) if (v := os.getenv("VOYAGE_API_KEY")) else None,
        description="API key. Defaults to $VOYAGE_API_KEY.",
    )
    truncation: bool = Field(True, description="Truncate inputs longer than the model context.")
    timeout: float = Field(60.0, description="Request timeout in seconds.")


class VoyageProvider(EmbeddingProvider):
    name = "voyage"
    label = "Voyage AI"
    description = "Voyage AI embedding models (general, code, finance, legal, multilingual)."
    Options = VoyageOptions
    requires = ("voyageai",)
    install_hint = "uv sync --extra voyage"
    max_batch_size = 128
    default_model = "voyage-3.5"
    supports_dimensions = True
    known_models = (
        ModelInfo(id="voyage-3.5", dimension=1024, max_tokens=32000),
        ModelInfo(id="voyage-3.5-lite", dimension=1024, max_tokens=32000),
        ModelInfo(id="voyage-3-large", dimension=1024, max_tokens=32000),
        ModelInfo(id="voyage-code-3", dimension=1024, max_tokens=32000),
        ModelInfo(id="voyage-finance-2", dimension=1024, max_tokens=32000),
        ModelInfo(id="voyage-law-2", dimension=1024, max_tokens=16000),
        ModelInfo(id="voyage-multilingual-2", dimension=1024, max_tokens=32000),
    )

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self._tokenizers: dict[str, Tokenizer | None] = {}

    def exact_tokenizer(self, model: str, options: ProviderOptions) -> Tokenizer | None:
        """Voyage publishes its tokenizers on the Hugging Face Hub (``voyageai/<model>``)."""
        if model in self._tokenizers:
            return self._tokenizers[model]
        tokenizer: Tokenizer | None = None
        try:
            from tokenizers import Tokenizer as HubTokenizer

            hub = HubTokenizer.from_pretrained(f"voyageai/{model}")
            hub.no_truncation()
            tokenizer = HFTokenizer(hub)
        except Exception as exc:  # noqa: BLE001 - not installed, unknown model or no network
            log.warning("No tokenizer for voyageai/%s (%s); estimating token counts", model, exc)
        self._tokenizers[model] = tokenizer
        return tokenizer

    def native_long_input(self, model: str) -> frozenset[str]:
        return frozenset({"truncate_end"})

    def apply_long_input(
        self, options: ProviderOptions, model: str, mode: LongInputMode
    ) -> ProviderOptions:
        # Voyage truncates the end, or rejects long inputs. Every mode but `error` keeps
        # truncation on as a backstop for estimated counts.
        return options.model_copy(update={"truncation": mode != "error"})

    def check_configuration(self, options: ProviderOptions) -> Availability:
        assert isinstance(options, VoyageOptions)
        if options.api_key is None:
            return Availability(
                available=False, reason="No API key: set VOYAGE_API_KEY or pass api_key."
            )
        return Availability(available=True)

    def _client(self, options: VoyageOptions) -> Any:
        assert options.api_key is not None
        secret = options.api_key.get_secret_value()
        client = self._clients.get(secret)
        if client is None:
            import voyageai

            client = voyageai.AsyncClient(api_key=secret, max_retries=0, timeout=options.timeout)
            self._clients[secret] = client
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
        assert isinstance(options, VoyageOptions)
        kwargs: dict[str, Any] = {
            "model": model,
            "input_type": input_type,
            "truncation": options.truncation,
        }
        if dimensions:
            kwargs["output_dimension"] = dimensions
        result = await self._client(options).embed(texts, **kwargs)
        return [list(v) for v in result.embeddings]
