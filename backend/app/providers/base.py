"""Embedding provider interface.

Every embedding backend implements :class:`EmbeddingProvider`. Providers are registered by name
in :mod:`app.providers.registry`; third-party packages can add providers through the
``arango_embedding_loader.providers`` entry-point group without touching this code base.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    ValidationError,
    model_validator,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.providers.long_input import (
    LEGACY_TRUNCATE_OPTIONS,
    EstimateTokenizer,
    LongInput,
    LongInputMode,
    TextPlan,
    Tokenizer,
    legacy_long_input,
    plan_text,
)

log = logging.getLogger(__name__)

InputType = Literal["document", "query"]


class ProviderError(Exception):
    """Raised for provider misconfiguration or embedding failures."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class ProviderOptions(BaseModel):
    """Base class for provider-specific options (API keys, endpoints, devices...).

    The spec's ``long_input`` mode travels with the options, but is not one of them: providers
    with their own truncation option set it from :attr:`long_input_mode`.
    """

    model_config = ConfigDict(extra="forbid")

    _long_input: LongInputMode = PrivateAttr("truncate_end")

    @property
    def long_input_mode(self) -> LongInputMode:
        return self._long_input

    def with_long_input(self, mode: LongInputMode) -> Self:
        copy = self.model_copy()
        copy._long_input = mode
        return copy


class NativeTruncationOptions(ProviderOptions):
    """Options of a provider that has its own truncation option.

    That option used to be a provider option (``truncate`` or ``truncation``); it is now set from
    ``long_input``. The old keys are still accepted, mapped to a mode, and logged as deprecated.
    """

    @model_validator(mode="wrap")
    @classmethod
    def _legacy_truncate(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        mode: LongInputMode | None = None
        if isinstance(data, dict) and any(k in data for k in LEGACY_TRUNCATE_OPTIONS):
            data = dict(data)
            for key in LEGACY_TRUNCATE_OPTIONS:
                if key in data:
                    value = data.pop(key)
                    mode = legacy_long_input(value) or mode
                    log.warning(
                        "Provider option '%s' is deprecated and no longer a provider option; "
                        "use long_input on the embedding spec (%r is long_input=%s)",
                        key,
                        value,
                        legacy_long_input(value) or "not recognized, ignored",
                    )
        options = handler(data)
        if mode is not None:
            options._long_input = mode
        return options


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


#: Phrases in provider errors that reject an input or request for its length.
_TOO_LONG_MARKERS = (
    "context length",
    "context_length",
    "too many tokens",
    "too many input tokens",
    "maximum number of tokens",
    "max tokens",
    "max allowed tokens",
    "tokens per request",
    "tokens per submitted batch",
    "token limit",
    "input is too long",
    "input too long",
    "text is too long",
    "too long for",
    "exceeds the maximum",
    "maxlength",
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
    #: Most tokens one request may hold, summed over its texts (``None``: no limit).
    max_batch_tokens: ClassVar[int | None] = None
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
        """Attach the spec's ``long_input`` mode to ``options``.

        Providers with their own truncation option read it back from
        :attr:`ProviderOptions.long_input_mode` when they build a request.
        """
        return options.with_long_input(mode)

    def batch_token_limit(self, model: str) -> int | None:
        """Most tokens one request may hold, summed over its texts. ``None``: no limit."""
        return self.max_batch_tokens

    def is_input_too_long(self, exc: BaseException) -> bool:
        """``exc`` is the provider rejecting an input (or request) for having too many tokens."""
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            name = type(current).__name__
            if "Timeout" not in name and "Connect" not in name:
                text = str(current).lower()
                if any(marker in text for marker in _TOO_LONG_MARKERS):
                    return True
            current = current.__cause__ or current.__context__
        return False

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
        allow_native: bool = True,
    ) -> list[TextPlan]:
        """Count each text's tokens and decide how it is sent. Called in a worker thread.

        ``allow_native=False`` makes the loader apply every mode itself, e.g. when the provider
        rejected a text its own truncation should have handled.
        """
        settings = long_input or LongInput()
        limit = settings.max_tokens or self.token_limit(model, options)
        tokenizer = self.exact_tokenizer(model, options) or EstimateTokenizer(settings.token_chars)
        native = self.native_long_input(model) if allow_native else frozenset()
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
        token_counts: list[int] | None = None,
    ) -> list[list[float]]:
        """Embed any number of texts: splits into batches, runs them concurrently, retries.

        A batch holds at most ``max_batch_size`` texts and, when ``token_counts`` are given, at
        most :meth:`batch_token_limit` tokens.
        """
        if not texts:
            return []
        chunks = _batches(texts, token_counts, self.max_batch_size, self.batch_token_limit(model))
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


def _batches(
    texts: list[str], token_counts: list[int] | None, size: int, token_budget: int | None
) -> list[list[str]]:
    """Consecutive batches of at most ``size`` texts and ``token_budget`` tokens.

    A single text over the budget gets a batch of its own. 80% of the budget is used, as token
    counts may be estimates.
    """
    if token_counts is None or token_budget is None:
        return [texts[i : i + size] for i in range(0, len(texts), size)]
    budget = max(1, int(token_budget * 0.8))
    out: list[list[str]] = []
    batch: list[str] = []
    used = 0
    for text, n in zip(texts, token_counts, strict=True):
        if batch and (len(batch) >= size or used + n > budget):
            out.append(batch)
            batch, used = [], 0
        batch.append(text)
        used += n
    if batch:
        out.append(batch)
    return out
