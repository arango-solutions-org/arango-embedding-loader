"""OpenAI embeddings.

Also Azure OpenAI and any OpenAI-compatible server (vLLM, Hugging Face TEI, LM Studio...).
"""

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
from app.providers.long_input import TiktokenTokenizer, Tokenizer

log = logging.getLogger(__name__)


def _env_secret(name: str) -> SecretStr | None:
    value = os.getenv(name)
    return SecretStr(value) if value else None


class OpenAIOptions(ProviderOptions):
    api_key: SecretStr | None = Field(
        default_factory=lambda: _env_secret("OPENAI_API_KEY"),
        description="API key. Defaults to $OPENAI_API_KEY.",
    )
    base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None,
        description="Override for OpenAI-compatible servers (vLLM, TEI, LM Studio...).",
    )
    organization: str | None = Field(None, description="OpenAI organization id.")
    azure_endpoint: str | None = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT") or None,
        description="Set to use Azure OpenAI, e.g. https://my-resource.openai.azure.com",
    )
    api_version: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_VERSION") or None,
        description="Azure OpenAI API version.",
    )
    timeout: float = Field(60.0, description="Request timeout in seconds.")


class OpenAIProvider(EmbeddingProvider):
    name = "openai"
    label = "OpenAI"
    description = (
        "OpenAI embeddings API. Also works with Azure OpenAI and OpenAI-compatible servers via "
        "`base_url`."
    )
    Options = OpenAIOptions
    requires = ("openai",)
    install_hint = "uv sync --extra openai"
    max_batch_size = 256
    default_model = "text-embedding-3-small"
    supports_dimensions = True
    known_models = (
        ModelInfo(id="text-embedding-3-small", dimension=1536, max_tokens=8191),
        ModelInfo(id="text-embedding-3-large", dimension=3072, max_tokens=8191),
        ModelInfo(id="text-embedding-ada-002", dimension=1536, max_tokens=8191),
    )

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self._tokenizers: dict[str, Tokenizer | None] = {}

    def check_configuration(self, options: ProviderOptions) -> Availability:
        assert isinstance(options, OpenAIOptions)
        if options.api_key is None and not options.base_url:
            return Availability(
                available=False, reason="No API key: set OPENAI_API_KEY or pass api_key."
            )
        return Availability(available=True)

    def exact_tokenizer(self, model: str, options: ProviderOptions) -> Tokenizer | None:
        """tiktoken for OpenAI models; ``None`` (estimate) for other OpenAI-compatible models."""
        if model in self._tokenizers:
            return self._tokenizers[model]
        tokenizer: Tokenizer | None = None
        try:
            import tiktoken

            tokenizer = TiktokenTokenizer(tiktoken.encoding_for_model(model))
        except ImportError:
            log.info("tiktoken is not installed; estimating token counts for %s", model)
        except KeyError:
            pass  # not an OpenAI model name
        except Exception as exc:  # noqa: BLE001 - e.g. no network to fetch the encoding
            log.warning("No tiktoken encoding for %s (%s); estimating token counts", model, exc)
        self._tokenizers[model] = tokenizer
        return tokenizer

    def _client(self, options: OpenAIOptions) -> Any:
        key = options.model_dump_json()
        if options.api_key:
            key += options.api_key.get_secret_value()
        client = self._clients.get(key)
        if client is None:
            import openai

            api_key = options.api_key.get_secret_value() if options.api_key else "not-needed"
            if options.azure_endpoint:
                client = openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=options.azure_endpoint,
                    api_version=options.api_version or "2024-10-21",
                    timeout=options.timeout,
                    max_retries=0,
                )
            else:
                client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=options.base_url,
                    organization=options.organization,
                    timeout=options.timeout,
                    max_retries=0,
                )
            self._clients[key] = client
        return client

    async def list_models(self, options: ProviderOptions) -> list[ModelInfo]:
        assert isinstance(options, OpenAIOptions)
        if options.base_url and not options.azure_endpoint:
            try:
                page = await self._client(options).models.list()
                return [ModelInfo(id=m.id, dimension=self.known_dimension(m.id)) for m in page.data]
            except Exception:
                pass
        return list(self.known_models)

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        assert isinstance(options, OpenAIOptions)
        kwargs: dict[str, Any] = {"model": model, "input": texts}
        if dimensions:
            kwargs["dimensions"] = dimensions
        response = await self._client(options).embeddings.create(**kwargs)
        data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in data]
