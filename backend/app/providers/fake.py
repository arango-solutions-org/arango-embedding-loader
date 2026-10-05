"""Deterministic, dependency-free provider for tests, demos and dry runs."""

from __future__ import annotations

import hashlib
import math
import struct

from pydantic import Field

from app.providers.base import EmbeddingProvider, InputType, ModelInfo, ProviderOptions


class FakeOptions(ProviderOptions):
    seed: str = Field("", description="Salt mixed into the hash; changes every vector.")


class FakeProvider(EmbeddingProvider):
    name = "fake"
    label = "Fake (deterministic hash)"
    description = (
        "Generates deterministic pseudo-embeddings from a hash of the text. No model, no network."
        " Useful for testing pipelines and vector indexes."
    )
    Options = FakeOptions
    max_batch_size = 1024
    default_model = "fake-384"
    supports_dimensions = True
    known_models = (
        ModelInfo(id="fake-384", dimension=384),
        ModelInfo(id="fake-768", dimension=768),
        ModelInfo(id="fake-1536", dimension=1536),
    )

    @staticmethod
    def _vector(text: str, dim: int, seed: str) -> list[float]:
        out: list[float] = []
        counter = 0
        while len(out) < dim:
            digest = hashlib.sha256(f"{seed}|{counter}|{text}".encode()).digest()
            for (val,) in struct.iter_unpack(">i", digest):
                out.append(val / 2**31)
            counter += 1
        vec = out[:dim]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        assert isinstance(options, FakeOptions)
        dim = dimensions or self.known_dimension(model)
        if dim is None:
            try:
                dim = int(model.rsplit("-", 1)[-1])
            except ValueError:
                dim = 384
        return [self._vector(t, dim, options.seed) for t in texts]
