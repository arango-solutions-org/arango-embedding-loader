"""Local Hugging Face models through sentence-transformers.

Runs any sentence-transformers compatible model in-process (CPU, Apple MPS or CUDA), e.g.
``NeuML/pubmedbert-base-embeddings`` for biomedical text.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
from typing import Any

from pydantic import Field

from app.providers.base import EmbeddingProvider, InputType, ModelInfo, ProviderOptions
from app.providers.long_input import HFTokenizer, Tokenizer


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


def baked_model_path(model: str) -> str | None:
    """Directory of a model saved into the image at build time (``BAKE_MODELS``), if any.

    Models live at ``$ST_MODELS_DIR/<org>/<name>``. Loading from that path needs no network,
    which matters on deployments that cannot reach the Hugging Face Hub.
    """
    root = os.getenv("ST_MODELS_DIR", "/opt/models")
    path = os.path.join(root, model)
    if os.path.realpath(path).startswith(os.path.realpath(root) + os.sep) and os.path.isfile(
        os.path.join(path, "modules.json")
    ):
        return path
    return None


class SentenceTransformersProvider(EmbeddingProvider):
    name = "sentence_transformers"
    label = "Sentence-Transformers (local)"
    description = (
        "Runs any Hugging Face sentence-transformers model locally, e.g. "
        "NeuML/pubmedbert-base-embeddings. Models baked into the image are loaded from disk; "
        "others are downloaded on first use."
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

                # A baked copy wins unless a specific Hub revision was requested.
                baked = None if options.revision else baked_model_path(model)
                loaded = SentenceTransformer(
                    baked or model,
                    device=options.device,
                    trust_remote_code=options.trust_remote_code,
                    revision=None if baked else options.revision,
                )
                self._models[key] = loaded
                self._encode_locks[id(loaded)] = threading.Lock()
            return loaded

    def exact_tokenizer(self, model: str, options: ProviderOptions) -> Tokenizer | None:
        assert isinstance(options, SentenceTransformersOptions)
        st_model = self._model(model, options)
        tokenizer = getattr(st_model, "tokenizer", None)
        if tokenizer is None:
            return None
        return HFTokenizer(tokenizer, lock=self._encode_locks[id(st_model)])

    def token_limit(self, model: str, options: ProviderOptions) -> int | None:
        """``max_seq_length`` less the special tokens and the document prefix."""
        assert isinstance(options, SentenceTransformersOptions)
        st_model = self._model(model, options)
        max_len = getattr(st_model, "max_seq_length", None)
        tokenizer = getattr(st_model, "tokenizer", None)
        if not max_len or tokenizer is None:
            return None
        with self._encode_locks[id(st_model)]:
            reserved = tokenizer.num_special_tokens_to_add(pair=False)
            if options.document_prefix:
                reserved += len(
                    tokenizer(options.document_prefix, add_special_tokens=False)["input_ids"]
                )
        return max(1, int(max_len) - reserved)

    def combine_vectors(
        self, vectors: list[list[float]], weights: list[int] | None, options: ProviderOptions
    ) -> list[float]:
        mean = super().combine_vectors(vectors, weights, options)
        if isinstance(options, SentenceTransformersOptions) and options.normalize_embeddings:
            norm = math.sqrt(sum(v * v for v in mean)) or 1.0
            mean = [v / norm for v in mean]
        return mean

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
