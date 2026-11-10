"""The watcher isolates documents that fail to embed, whatever the error."""

from __future__ import annotations

from typing import Any

import pytest

from app.models.embedding import EmbeddingFieldSpec
from app.providers.base import EmbeddingProvider, InputType, ProviderOptions
from app.providers.registry import get_provider, register_provider
from app.services.pipeline import resolve_specs
from app.services.watcher import embed_isolating


class PickyProvider(EmbeddingProvider):
    """Fails a whole request with a generic error when any text holds ``poison``; fails every
    request while ``down``. The vector is [words in the text, 1.0]."""

    name = "picky_test"
    label = "Picky (test)"
    default_model = "m"
    max_batch_size = 1000
    down = False

    def __init__(self) -> None:
        self.calls = 0

    def is_retryable(self, exc: BaseException) -> bool:
        return False

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        self.calls += 1
        if self.down:
            raise ConnectionError("provider unreachable")
        if any("poison" in t for t in texts):
            raise RuntimeError("500 Internal Server Error")
        return [[float(len(t.split())), 1.0] for t in texts]


@pytest.fixture
def picky() -> PickyProvider:
    register_provider(PickyProvider)  # a fresh instance per test
    p = get_provider(PickyProvider.name)
    assert isinstance(p, PickyProvider)
    return p


def _specs(**extra: Any):
    return resolve_specs(
        [
            EmbeddingFieldSpec(
                source_field="text", target_field="vec", provider="picky_test", **extra
            )
        ]
    )


async def test_batch_without_errors_is_embedded_once(picky):
    docs = [{"_key": "a", "text": "one two"}, {"_key": "b", "text": "three"}]
    out_docs, outcomes = await embed_isolating(docs, _specs())
    assert [d["vec"] for d in out_docs] == [[2.0, 1.0], [1.0, 1.0]]
    assert all(o.embedded for o in outcomes) and picky.calls == 1


async def test_bad_document_fails_alone(picky):
    docs = [
        {"_key": "a", "text": "one two"},
        {"_key": "bad", "text": "a poison pill"},
        {"_key": "c", "text": "three"},
    ]
    out_docs, outcomes = await embed_isolating(docs, _specs())
    by_key = {d["_key"]: (d, o) for d, o in zip(out_docs, outcomes, strict=True)}
    bad, bad_outcome = by_key["bad"]
    assert bad_outcome.error and "500 Internal Server Error" in bad_outcome.error
    assert bad["vec_meta"]["status"] == "error"
    assert bad["vec_meta"]["reason"] == "embedding failed"
    assert "500 Internal Server Error" in bad["vec_meta"]["error"]
    assert "vec" not in bad
    assert by_key["a"][0]["vec"] == [2.0, 1.0] and by_key["a"][1].embedded
    assert by_key["c"][0]["vec"] == [1.0, 1.0] and by_key["c"][1].embedded


async def test_lone_bad_document_fails_alone(picky):
    """One document can't be told apart from an outage by itself; the provider check can."""
    out_docs, outcomes = await embed_isolating([{"_key": "bad", "text": "poison"}], _specs())
    assert outcomes[0].error and out_docs[0]["vec_meta"]["status"] == "error"


async def test_outage_raises_and_marks_nothing(picky):
    picky.down = True
    docs = [{"_key": "a", "text": "one"}, {"_key": "b", "text": "two"}]
    with pytest.raises(Exception, match="unreachable"):
        await embed_isolating(docs, _specs())
    assert all("vec_meta" not in d for d in docs)


async def test_any_error_is_isolated(picky, monkeypatch):
    """Errors outside the provider request (here, the tokenizer) are isolated too."""
    plan = picky.plan_inputs

    def failing_plan(texts, **kwargs):
        if any("boom" in t for t in texts):
            raise ValueError("tokenizer exploded")
        return plan(texts, **kwargs)

    monkeypatch.setattr(picky, "plan_inputs", failing_plan)
    docs = [{"_key": "a", "text": "fine"}, {"_key": "b", "text": "boom"}]
    out_docs, outcomes = await embed_isolating(docs, _specs())
    assert out_docs[0]["vec"] == [1.0, 1.0]
    assert outcomes[1].error and "tokenizer exploded" in out_docs[1]["vec_meta"]["error"]


async def test_failed_document_is_put_back_as_it_was(picky):
    """A chunked document that fails is not left half-split: no null vector, no chunks."""
    long_bad = " ".join(["poison"] + [f"w{i}" for i in range(20)])
    docs = [
        {"_key": "ok", "text": "fine"},
        {"_key": "bad", "text": long_bad, "vec": [9.0, 9.0], "vec_meta": None},
    ]
    specs = _specs(long_input="chunk_docs", max_tokens=5, token_chars=100)
    out_docs, outcomes = await embed_isolating(docs, specs)
    bad = out_docs[1]
    assert bad["vec"] == [9.0, 9.0] and bad["vec_meta"]["status"] == "error"
    assert not outcomes[1].chunks and not outcomes[1].split_specs


class RateLimitError(Exception):
    status_code = 429


async def test_rate_limit_marks_nothing(picky, monkeypatch):
    """A rate limit says nothing about the documents: back off, mark nothing."""
    real = picky.embed_batch

    async def limited(texts, **kwargs):
        if len(texts) > 1:
            raise RateLimitError("too many tokens per minute")
        return await real(texts, **kwargs)

    monkeypatch.setattr(picky, "embed_batch", limited)
    docs = [{"_key": "a", "text": "one"}, {"_key": "b", "text": "two"}]
    with pytest.raises(Exception, match="per minute"):
        await embed_isolating(docs, _specs())
    assert all("vec_meta" not in d for d in docs)


async def test_rate_limit_during_one_by_one_marks_nothing(picky, monkeypatch):
    real = picky.embed_batch

    async def flaky(texts, **kwargs):
        if any("poison" in t for t in texts):
            raise RuntimeError("500 Internal Server Error")
        if texts == ["three"]:
            raise RateLimitError("slow down")
        return await real(texts, **kwargs)

    monkeypatch.setattr(picky, "embed_batch", flaky)
    docs = [
        {"_key": "a", "text": "one"},
        {"_key": "bad", "text": "poison"},
        {"_key": "c", "text": "three"},
    ]
    with pytest.raises(Exception, match="slow down"):
        await embed_isolating(docs, _specs())
    assert all("vec_meta" not in d for d in docs)


async def test_every_document_failing_is_a_provider_problem(picky, monkeypatch):
    """The probe passes, but every real text fails (e.g. out of memory on real inputs)."""
    real = picky.embed_batch

    async def only_ok(texts, **kwargs):
        if texts != ["ok"]:
            raise RuntimeError("CUDA out of memory")
        return await real(texts, **kwargs)

    monkeypatch.setattr(picky, "embed_batch", only_ok)
    docs = [{"_key": "a", "text": "one"}, {"_key": "b", "text": "two"}]
    with pytest.raises(Exception, match="out of memory"):
        await embed_isolating(docs, _specs())
    assert all("vec_meta" not in d for d in docs)


def test_rate_limit_is_never_input_too_long(picky):
    """A 429 mentioning tokens must not shrink texts; a real length error still does."""
    from app.providers.base import ProviderError

    def wrapped(inner: Exception) -> ProviderError:
        try:
            raise ProviderError(f"Provider failed: {inner}", status_code=502) from inner
        except ProviderError as exc:
            return exc

    assert not picky.is_input_too_long(wrapped(RateLimitError("too many tokens per minute")))
    assert picky.is_input_too_long(wrapped(ValueError("too many tokens in input")))
