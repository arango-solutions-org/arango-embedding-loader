"""Provider registry with built-ins and entry-point plugin discovery.

Third-party providers: publish a package that declares

    [project.entry-points."arango_embedding_loader.providers"]
    my_provider = "my_package.module:MyProvider"

where ``MyProvider`` subclasses :class:`app.providers.base.EmbeddingProvider`.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from app.providers.base import EmbeddingProvider, ProviderError

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "arango_embedding_loader.providers"

_registry: dict[str, EmbeddingProvider] = {}
_discovered = False


def register_provider(cls: type[EmbeddingProvider]) -> type[EmbeddingProvider]:
    """Register a provider class (usable as a decorator)."""
    if not issubclass(cls, EmbeddingProvider):
        raise TypeError(f"{cls!r} is not an EmbeddingProvider")
    _registry[cls.name] = cls()
    return cls


def _register_builtins() -> None:
    from app.providers.bedrock import BedrockProvider
    from app.providers.cohere_provider import CohereProvider
    from app.providers.fake import FakeProvider
    from app.providers.ollama import OllamaProvider
    from app.providers.openai_provider import OpenAIProvider
    from app.providers.sentence_transformers_provider import SentenceTransformersProvider
    from app.providers.voyage import VoyageProvider

    for cls in (
        SentenceTransformersProvider,
        OpenAIProvider,
        VoyageProvider,
        CohereProvider,
        BedrockProvider,
        OllamaProvider,
        FakeProvider,
    ):
        if cls.name not in _registry:
            register_provider(cls)


def _discover_plugins() -> None:
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
            register_provider(cls)
            log.info("Loaded embedding provider plugin %s from %s", cls.name, ep.value)
        except Exception:
            log.exception("Failed to load embedding provider plugin %s", ep.value)


def _ensure_loaded() -> None:
    global _discovered
    if not _discovered:
        _register_builtins()
        _discover_plugins()
        _discovered = True


def get_provider(name: str) -> EmbeddingProvider:
    _ensure_loaded()
    provider = _registry.get(name)
    if provider is None:
        raise ProviderError(
            f"Unknown provider '{name}'. Available: {', '.join(sorted(_registry))}",
            status_code=404,
        )
    return provider


def list_providers() -> list[EmbeddingProvider]:
    _ensure_loaded()
    return list(_registry.values())
