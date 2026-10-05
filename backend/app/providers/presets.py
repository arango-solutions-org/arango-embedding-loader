"""Named provider/model presets, e.g. ``pubmedbert``.

Built-in presets can be extended or overridden with a YAML file referenced by ``PRESETS_FILE``::

    - name: my-e5
      provider: sentence_transformers
      model: intfloat/e5-base-v2
      dimensions: 768
      provider_options: {document_prefix: "passage: "}
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.providers.base import ProviderError


class Preset(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    name: str
    label: str | None = None
    description: str | None = None
    provider: str
    model: str
    dimensions: int | None = Field(None, description="Expected vector dimension.")
    provider_options: dict[str, Any] = Field(default_factory=dict)


BUILTIN_PRESETS: list[Preset] = [
    Preset(
        name="pubmedbert",
        label="PubMedBERT (biomedical, local)",
        description="NeuML/pubmedbert-base-embeddings via sentence-transformers. 768 dimensions.",
        provider="sentence_transformers",
        model="NeuML/pubmedbert-base-embeddings",
        dimensions=768,
    ),
    Preset(
        name="minilm",
        label="all-MiniLM-L6-v2 (fast, local)",
        provider="sentence_transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
    ),
    Preset(
        name="openai-small",
        label="OpenAI text-embedding-3-small",
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
    ),
    Preset(
        name="openai-large",
        label="OpenAI text-embedding-3-large",
        provider="openai",
        model="text-embedding-3-large",
        dimensions=3072,
    ),
    Preset(
        name="voyage",
        label="Voyage voyage-3.5",
        provider="voyage",
        model="voyage-3.5",
        dimensions=1024,
    ),
    Preset(
        name="cohere-v4",
        label="Cohere embed-v4.0",
        provider="cohere",
        model="embed-v4.0",
        dimensions=1536,
    ),
    Preset(
        name="titan-v2",
        label="Amazon Titan Text Embeddings v2",
        provider="bedrock",
        model="amazon.titan-embed-text-v2:0",
        dimensions=1024,
    ),
    Preset(
        name="nomic-ollama",
        label="nomic-embed-text (Ollama)",
        provider="ollama",
        model="nomic-embed-text",
        dimensions=768,
    ),
    Preset(
        name="fake",
        label="Fake 384-d (testing)",
        provider="fake",
        model="fake-384",
        dimensions=384,
    ),
]


@lru_cache
def list_presets() -> list[Preset]:
    presets = {p.name: p for p in BUILTIN_PRESETS}
    path = get_settings().presets_file
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or []
        for item in data:
            preset = Preset.model_validate(item)
            presets[preset.name] = preset
    return list(presets.values())


def get_preset(name: str) -> Preset:
    for preset in list_presets():
        if preset.name == name:
            return preset
    raise ProviderError(f"Unknown preset '{name}'", status_code=404)
