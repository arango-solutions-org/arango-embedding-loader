"""Local Hugging Face models through sentence-transformers.

Runs any sentence-transformers compatible model in-process (CPU, Apple MPS or CUDA), e.g.
``NeuML/pubmedbert-base-embeddings`` for biomedical text.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

from pydantic import Field

from app.providers.base import EmbeddingProvider, InputType, ModelInfo, ProviderOptions


class SentenceTransformersOptions(ProviderOptions):
    device: str | None = Field(
        default_factory=lambda: os.getenv("ST_DEVICE") or None,
        description="cpu | cuda | mps. Defaults to $ST_DEVICE or auto-detect.",
    )
    normalize_embeddings: bool = Field(False, description="Return unit-length vectors.")
    batch_size: int = Field(32, ge=1, le=1024, description="Model forward-pass batch size.")
    document_prefix: str = Field(
        "", description="Prepended to every document text (e.g. 'passage: ' for E5 models)."
    )
    query_prefix: str = Field("", description="Prepended to query texts (e.g. 'query: ').")
    trust_remote_code: bool = Field(False, description="Allow custom model code from the Hub.")
    revision: str | None = Field(None, description="Model revision / commit on the Hub.")


class SentenceTransformersProvider(EmbeddingProvider):
    name = "sentence_transformers"
    label = "Sentence-Transformers (local)"
    description = (
        "Runs any Hugging Face sentence-transformers model locally, e.g. "
        "NeuML/pubmedbert-base-embeddings. Models are downloaded on first use."
    )
    Options = SentenceTransformersOptions
    requires = ("sentence_transformers",)
    install_hint = "uv sync --extra local"
    max_batch_size = 256
    max_concurrency = 1
    default_model = "NeuML/pubmedbert-base-embeddings"
    known_models = (
        ModelInfo(
            id="NeuML/pubmedbert-base-embeddings",
            dimension=768,
            description="PubMedBERT fine-tuned for biomedical sentence embeddings.",
        ),
        ModelInfo(id="sentence-transformers/all-MiniLM-L6-v2", dimension=384),
        ModelInfo(id="sentence-transformers/all-mpnet-base-v2", dimension=768),
        ModelInfo(id="BAAI/bge-small-en-v1.5", dimension=384),
        ModelInfo(id="BAAI/bge-base-en-v1.5", dimension=768),
        ModelInfo(id="intfloat/e5-base-v2", dimension=768),
    )

    def __init__(self) -> None:
        self._models: dict[tuple[str, str | None, str | None], Any] = {}
        self._load_lock = threading.Lock()
        self._encode_locks: dict[int, threading.Lock] = {}

    def _model(self, model: str, options: SentenceTransformersOptions) -> Any:
        key = (model, options.device, options.revision)
        with self._load_lock:
            loaded = self._models.get(key)
            if loaded is None:
                from sentence_transformers import SentenceTransformer

                loaded = SentenceTransformer(
                    model,
                    device=options.device,
                    trust_remote_code=options.trust_remote_code,
                    revision=options.revision,
                )
                self._models[key] = loaded
                self._encode_locks[id(loaded)] = threading.Lock()
            return loaded

    def _encode(
        self, texts: list[str], model: str, options: SentenceTransformersOptions
    ) -> list[list[float]]:
        st_model = self._model(model, options)
        with self._encode_locks[id(st_model)]:
            vectors = st_model.encode(
                texts,
                batch_size=options.batch_size,
                normalize_embeddings=options.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vectors]

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        assert isinstance(options, SentenceTransformersOptions)
        prefix = options.document_prefix if input_type == "document" else options.query_prefix
        inputs = [prefix + t for t in texts] if prefix else texts
        vectors = await asyncio.to_thread(self._encode, inputs, model, options)
        if dimensions and dimensions < len(vectors[0]):
            # Matryoshka-style truncation for models trained for it.
            vectors = [v[:dimensions] for v in vectors]
        return vectors
