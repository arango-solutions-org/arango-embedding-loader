"""Embedding provider interface.

Every embedding backend implements :class:`EmbeddingProvider`. Providers are registered by name
in :mod:`app.providers.registry`; third-party packages can add providers through the
``arango_embedding_loader.providers`` entry-point group without touching this code base.
"""

from __future__ import annotations

import asyncio
import importlib.util
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.providers.long_input import (
    EstimateTokenizer,
    LongInput,
    LongInputMode,
    TextPlan,
    Tokenizer,
    plan_text,
)

InputType = Literal["document", "query"]


class ProviderError(Exception):
    """Raised for provider misconfiguration or embedding failures."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class ProviderOptions(BaseModel):
    """Base class for provider-specific options (API keys, endpoints, devices...)."""

    model_config = ConfigDict(extra="forbid")


class ModelInfo(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    dimension: int | None = Field(None, description="Output dimension, when known.")
    max_tokens: int | None = Field(None, description="Input token limit, when known.")
    description: str | None = None


class Availability(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    available: bool
    reason: str | None = None


_RETRYABLE_NAMES = (
    "RateLimit",
    "Timeout",
    "APIConnection",
    "InternalServer",
    "ServiceUnavailable",
    "Throttling",
    "TooManyRequests",
    "ConnectError",
    "ReadError",
    "RemoteProtocolError",
)


def default_is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    name = type(exc).__name__
    return any(token in name for token in _RETRYABLE_NAMES)


class EmbeddingProvider(ABC):
    #: Registry name, e.g. ``"openai"``.
    name: ClassVar[str]
    #: Human readable label for UIs.
    label: ClassVar[str]
    description: ClassVar[str] = ""
    #: Pydantic model describing accepted ``provider_options``.
    Options: ClassVar[type[ProviderOptions]] = ProviderOptions
    #: Importable modules required by this provider.
    requires: ClassVar[tuple[str, ...]] = ()
    install_hint: ClassVar[str | None] = None
    #: Largest number of texts sent in a single request.
    max_batch_size: ClassVar[int] = 128
    #: Max concurrent requests per pipeline.
    max_concurrency: ClassVar[int] = 4
    default_model: ClassVar[str | None] = None
    known_models: ClassVar[tuple[ModelInfo, ...]] = ()
    #: Whether the model accepts a ``dimensions`` override.
    supports_dimensions: ClassVar[bool] = False

    # ----------------------------------------------------------------- options

    def parse_options(self, raw: dict[str, Any] | None) -> ProviderOptions:
        try:
            return self.Options.model_validate(raw or {})
        except ValidationError as exc:
            raise ProviderError(f"Invalid options for provider '{self.name}': {exc}") from exc

    def missing_requirements(self) -> list[str]:
        return [m for m in self.requires if importlib.util.find_spec(m) is None]

    def availability(self, options: ProviderOptions | None = None) -> Availability:
        missing = self.missing_requirements()
        if missing:
            hint = f" Install with: {self.install_hint}" if self.install_hint else ""
            return Availability(
                available=False, reason=f"Missing Python package(s): {', '.join(missing)}.{hint}"
            )
        return self.check_configuration(options or self.parse_options(None))

    def check_configuration(self, options: ProviderOptions) -> Availability:
        """Override to report missing credentials etc. Must not do network I/O."""
        return Availability(available=True)

    def ensure_available(self, options: ProviderOptions) -> None:
        status = self.availability(options)
        if not status.available:
            raise ProviderError(f"Provider '{self.name}' is not available: {status.reason}")

    # ------------------------------------------------------------------ models

    async def list_models(self, options: ProviderOptions) -> list[ModelInfo]:
        return list(self.known_models)

    def known_dimension(self, model: str) -> int | None:
        for info in self.known_models:
            if info.id == model:
                return info.dimension
        return None

    # -------------------------------------------------------------- long input

    def token_limit(self, model: str, options: ProviderOptions) -> int | None:
        """The model's input token limit. Defaults to ``known_models``; ``None`` if unknown."""
        for info in self.known_models:
            if info.id == model:
                return info.max_tokens
        return None

    def exact_tokenizer(self, model: str, options: ProviderOptions) -> Tokenizer | None:
        """The model's own tokenizer, if available. Otherwise token counts are estimated."""
        return None

    def native_long_input(self, model: str) -> frozenset[str]:
        """Truncate modes the provider applies itself (the whole text is sent)."""
        return frozenset()

    def apply_long_input(
        self, options: ProviderOptions, model: str, mode: LongInputMode
    ) -> ProviderOptions:
        """Set the provider's own truncation option to match ``mode``. Default: no change."""
        return options

    # --------------------------------------------------------------- embedding

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        """Embed at most ``max_batch_size`` texts in a single provider call."""

    def plan_inputs(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        input_type: InputType,
        long_input: LongInput | None = None,
    ) -> list[TextPlan]:
        """Count each text's tokens and decide how it is sent. Called in a worker thread."""
        settings = long_input or LongInput()
        limit = settings.max_tokens or self.token_limit(model, options)
        tokenizer = self.exact_tokenizer(model, options) or EstimateTokenizer(settings.token_chars)
        native = self.native_long_input(model)
        return [
            plan_text(text, tokens, limit, settings, exact=tokenizer.exact, native=native)
            for text, tokens in zip(texts, tokenizer.tokenize(texts), strict=True)
        ]

    def combine_vectors(
        self, vectors: list[list[float]], weights: list[int] | None, options: ProviderOptions
    ) -> list[float]:
        """Weighted mean of the vectors of one text's pieces."""
        w = weights or [1] * len(vectors)
        total = float(sum(w)) or 1.0
        return [
            sum(wi * v[k] for v, wi in zip(vectors, w, strict=True)) / total
            for k in range(len(vectors[0]))
        ]

    def is_retryable(self, exc: BaseException) -> bool:
        return default_is_retryable(exc)

    async def embed(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None = None,
        input_type: InputType = "document",
        max_attempts: int = 5,
    ) -> list[list[float]]:
        """Embed any number of texts: splits into batches, runs them concurrently, retries."""
        if not texts:
            return []
        chunks = [
            texts[i : i + self.max_batch_size] for i in range(0, len(texts), self.max_batch_size)
        ]
        sem = asyncio.Semaphore(self.max_concurrency)

        async def run(chunk: list[str]) -> list[list[float]]:
            async with sem:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception(self.is_retryable),
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential_jitter(initial=1, max=30),
                    reraise=True,
                ):
                    with attempt:
                        vectors = await self.embed_batch(
                            chunk,
                            model=model,
                            options=options,
                            dimensions=dimensions,
                            input_type=input_type,
                        )
                if len(vectors) != len(chunk):
                    raise ProviderError(
                        f"Provider '{self.name}' returned {len(vectors)} vectors "
                        f"for {len(chunk)} inputs",
                        status_code=502,
                    )
                return [list(map(float, v)) for v in vectors]

        try:
            results = await asyncio.gather(*(run(c) for c in chunks))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Provider '{self.name}' failed: {type(exc).__name__}: {exc}", status_code=502
            ) from exc
        vectors = [v for chunk in results for v in chunk]
        dims = {len(v) for v in vectors}
        if len(dims) > 1:
            raise ProviderError(
                f"Provider '{self.name}' returned inconsistent dimensions: {sorted(dims)}",
                status_code=502,
            )
        return vectors

    # ------------------------------------------------------------------- info

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "default_model": self.default_model,
            "supports_dimensions": self.supports_dimensions,
            "max_batch_size": self.max_batch_size,
            "install_hint": self.install_hint,
            "options_schema": self.Options.model_json_schema(),
            "known_models": [m.model_dump() for m in self.known_models],
        }
