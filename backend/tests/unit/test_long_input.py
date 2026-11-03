"""Inputs longer than the model's token limit (`long_input` on the embedding spec)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from app.models.embedding import EmbeddingFieldSpec
from app.providers.base import EmbeddingProvider, InputType, ProviderOptions
from app.providers.long_input import EstimateTokenizer
from app.providers.registry import get_provider, register_provider
from app.services.pipeline import (
    InputTooLongError,
    chunk_documents,
    embed_documents,
    resolve_specs,
)


class WordProvider(EmbeddingProvider):
    """No tokenizer, no known limit: tests set `max_tokens`, and `token_chars: 100` makes every
    word one estimated token. The vector is [words in the text, 1.0]."""

    name = "words_test"
    label = "Words (test)"
    default_model = "m"
    max_batch_size = 1000

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(len(t.split())), 1.0] for t in texts]


class NativeWordProvider(WordProvider):
    """Like a hosted API that truncates the end itself."""

    name = "native_words_test"

    def native_long_input(self, model: str) -> frozenset[str]:
        return frozenset({"truncate_end"})


@pytest.fixture
def provider() -> WordProvider:
    register_provider(WordProvider)  # registers a fresh instance
    p = get_provider(WordProvider.name)
    assert isinstance(p, WordProvider)
    return p


def _spec(mode: str, **extra: Any) -> EmbeddingFieldSpec:
    fields: dict[str, Any] = {
        "source_field": "text",
        "target_field": "vec",
        "provider": "words_test",
        "model": "m",
        "long_input": mode,
        "max_tokens": 5,
        "token_chars": 100,
        "chunk_overlap_tokens": 0,
    }
    fields.update(extra)
    return EmbeddingFieldSpec(**fields)


def _specs(mode: str, **extra: Any):
    return resolve_specs([_spec(mode, **extra)])


SHORT = "one two three"
LONG = "a b c d e f g h i j k l"  # 12 tokens, limit 5


# ------------------------------------------------------------------- settings


def test_defaults():
    spec = EmbeddingFieldSpec(source_field="t", preset="fake")
    assert spec.long_input == "truncate_end"
    assert spec.chunk_overlap_tokens == 25
    assert spec.max_tokens is None and spec.token_chars == 4.0
    assert spec.warn_on_long_input is True


def test_estimate_tokenizer():
    (tokens,) = EstimateTokenizer(4).tokenize(["abcdefgh ab  abcde"])
    assert len(tokens) == 2 + 1 + 2
    assert tokens.word_start == [True, False, True, True, False]


def test_known_limits():
    assert get_provider("openai").token_limit("text-embedding-3-small", None) == 8191  # type: ignore[arg-type]
    assert get_provider("cohere").token_limit("embed-english-v3.0", None) == 512  # type: ignore[arg-type]
    assert get_provider("ollama").token_limit("nomic-embed-text", None) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "model", "field", "expected"),
    [
        (
            "cohere",
            "embed-english-v3.0",
            "truncate",
            {
                "truncate_end": "END",
                "truncate_start": "START",
                "error": "NONE",
                "chunk_mean": "END",
            },
        ),
        (
            "bedrock",
            "cohere.embed-english-v3",
            "truncate",
            {"truncate_end": "END", "truncate_start": "START", "error": "NONE", "skip": "END"},
        ),
        (
            "voyage",
            "voyage-3.5",
            "truncation",
            {"truncate_end": True, "error": False, "chunk_docs": True},
        ),
        (
            "ollama",
            "nomic-embed-text",
            "truncate",
            {"truncate_end": True, "error": False, "truncate_middle": True},
        ),
    ],
)
def test_native_truncation_options(name, model, field, expected):
    p = get_provider(name)
    for mode, value in expected.items():
        options = p.apply_long_input(p.parse_options({}), model, mode)
        assert getattr(options, field) == value, mode


def test_native_modes():
    assert get_provider("cohere").native_long_input("embed-v4.0") == {
        "truncate_end",
        "truncate_start",
    }
    assert get_provider("bedrock").native_long_input("amazon.titan-embed-text-v2:0") == set()
    assert get_provider("openai").native_long_input("text-embedding-3-small") == set()


# ---------------------------------------------------------------------- modes


async def test_short_text_records_metadata(provider):
    docs = [{"text": SHORT}]
    outcomes = await embed_documents(docs, _specs("error"), store_metadata=True)
    assert outcomes[0].embedded and not outcomes[0].over_limit
    meta = docs[0]["vec_meta"]
    assert meta["input_tokens"] == 3 and meta["max_tokens"] == 5
    assert meta["truncated"] is False and meta["tokens_dropped"] == 0 and meta["chunks"] == 1
    assert meta["long_input"] == "error" and meta["token_count"] == "estimated"
    assert docs[0]["vec"] == [3.0, 1.0]


async def test_unknown_limit_embeds_whole_text(provider):
    docs = [{"text": LONG}]
    await embed_documents(docs, _specs("error", max_tokens=None), store_metadata=True)
    assert provider.seen == [LONG]
    assert docs[0]["vec_meta"]["max_tokens"] is None
    assert docs[0]["vec_meta"]["input_tokens"] == 12


async def test_truncate_end(provider):
    docs = [{"text": LONG}]
    outcomes = await embed_documents(docs, _specs("truncate_end"), store_metadata=True)
    assert outcomes[0].over_limit and outcomes[0].embedded
    assert provider.seen == ["a b c d e"]
    meta = docs[0]["vec_meta"]
    assert meta["truncated"] is True and meta["tokens_dropped"] == 7 and meta["chunks"] == 1


async def test_truncate_start(provider):
    docs = [{"text": LONG}]
    await embed_documents(docs, _specs("truncate_start"), store_metadata=True)
    assert provider.seen == ["h i j k l"]
    assert docs[0]["vec_meta"]["tokens_dropped"] == 7


async def test_truncate_middle(provider):
    docs = [{"text": LONG}]
    await embed_documents(docs, _specs("truncate_middle"), store_metadata=True)
    # 2 tokens from the start, 1 for the line break, 2 from the end.
    assert provider.seen == ["a b\nk l"]
    assert docs[0]["vec_meta"]["truncated"] is True
    assert docs[0]["vec_meta"]["tokens_dropped"] == 8


async def test_native_truncation_sends_whole_text():
    register_provider(NativeWordProvider)
    p = get_provider(NativeWordProvider.name)
    assert isinstance(p, NativeWordProvider)
    docs = [{"text": LONG}]
    await embed_documents(
        docs, _specs("truncate_end", provider=NativeWordProvider.name), store_metadata=True
    )
    assert p.seen == [LONG]  # the provider truncates
    meta = docs[0]["vec_meta"]
    assert meta["truncated"] is True and meta["tokens_dropped"] == 7


async def test_chunk_mean(provider):
    docs = [{"text": LONG}, {"text": SHORT}]
    await embed_documents(docs, _specs("chunk_mean"), store_metadata=True)
    assert provider.seen == ["a b c d e", "f g h i j", "k l", SHORT]
    # Token-weighted mean of [5,1], [5,1], [2,1] with weights 5, 5, 2.
    assert docs[0]["vec"] == pytest.approx([(25 + 25 + 4) / 12, 1.0])
    meta = docs[0]["vec_meta"]
    assert meta["chunks"] == 3 and meta["truncated"] is False and meta["tokens_dropped"] == 0
    assert docs[1]["vec"] == [3.0, 1.0]


async def test_chunk_overlap(provider):
    docs = [{"text": LONG}]
    await embed_documents(docs, _specs("chunk_mean", chunk_overlap_tokens=2), store_metadata=True)
    assert provider.seen == ["a b c d e", "d e f g h", "g h i j k", "j k l"]


async def test_chunk_overlap_capped_at_half_the_limit(provider):
    docs = [{"text": LONG}]
    await embed_documents(docs, _specs("chunk_mean", chunk_overlap_tokens=25), store_metadata=True)
    assert provider.seen == ["a b c d e", "d e f g h", "g h i j k", "j k l"]


async def test_chunk_overlap_rounds_to_nearest_word(provider):
    # token_chars 1: every word is 3 tokens. Limit 7, overlap 2 -> one shared word (3 tokens).
    docs = [{"text": "aaa bbb ccc ddd eee"}]
    specs = _specs("chunk_mean", max_tokens=7, token_chars=1, chunk_overlap_tokens=2)
    await embed_documents(docs, specs, store_metadata=True)
    assert provider.seen == ["aaa bbb", "bbb ccc", "ccc ddd", "ddd eee"]


async def test_chunk_mean_renormalizes():
    from app.providers.sentence_transformers_provider import (
        SentenceTransformersOptions,
        SentenceTransformersProvider,
    )

    p = SentenceTransformersProvider()
    opts = SentenceTransformersOptions(normalize_embeddings=True)
    assert p.combine_vectors([[3.0, 4.0], [3.0, 4.0]], [1, 1], opts) == pytest.approx([0.6, 0.8])


async def test_skip(provider):
    docs = [{"text": LONG}, {"text": SHORT}]
    outcomes = await embed_documents(docs, _specs("skip"), store_metadata=True)
    assert outcomes[0].skipped and not outcomes[0].embedded and outcomes[0].over_limit
    assert "vec" not in docs[0]
    meta = docs[0]["vec_meta"]
    assert meta["status"] == "skipped_too_long" and meta["chunks"] == 0
    assert meta["tokens_dropped"] == 12
    assert outcomes[1].embedded and provider.seen == [SHORT]


@pytest.mark.parametrize("warn", [True, False])
async def test_error_raises(provider, caplog, warn):
    caplog.set_level(logging.INFO, logger="app.services.pipeline")
    docs = [{"_key": "ok", "text": SHORT}, {"_key": "big", "text": LONG}]
    with pytest.raises(InputTooLongError) as exc:
        await embed_documents(docs, _specs("error", warn_on_long_input=warn), store_metadata=True)
    assert "'big'" in str(exc.value.detail) and "12 tokens" in str(exc.value.detail)
    assert provider.seen == []  # nothing in the batch was embedded
    records = [r for r in caplog.records if r.name == "app.services.pipeline"]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "stopping the run" in records[0].getMessage()


async def test_chunk_docs(provider):
    docs = [{"_key": "p1", "title": "T", "text": LONG}, {"_key": "p2", "text": SHORT}]
    outcomes = await embed_documents(docs, _specs("chunk_docs"), store_metadata=True)
    parent = docs[0]
    assert parent["vec"] is None
    assert parent["vec_meta"]["status"] == "chunked" and parent["vec_meta"]["chunks"] == 3
    chunks = chunk_documents(outcomes)
    assert [c["_key"] for c in chunks] == ["p1_vec_chunk_0", "p1_vec_chunk_1", "p1_vec_chunk_2"]
    assert [c["text"] for c in chunks] == ["a b c d e", "f g h i j", "k l"]
    assert all(c["title"] == "T" for c in chunks)
    assert chunks[2]["vec"] == [2.0, 1.0]
    meta = chunks[1]["vec_meta"]
    assert meta["parent_key"] == "p1" and meta["chunk_index"] == 1 and meta["chunk_count"] == 3
    assert meta["status"] == "embedded" and meta["input_tokens"] == 5
    assert docs[1]["vec"] == [3.0, 1.0] and not outcomes[1].chunks


async def test_chunk_docs_without_key_gets_one(provider):
    docs = [{"text": LONG}]
    outcomes = await embed_documents(docs, _specs("chunk_docs"), store_metadata=True)
    assert outcomes[0].chunks[0]["vec_meta"]["parent_key"] == docs[0]["_key"]


async def test_chunk_docs_other_specs_marked_done(provider):
    specs = resolve_specs(
        [
            _spec("chunk_docs"),
            EmbeddingFieldSpec(source_field="title", target_field="tvec", preset="fake"),
        ]
    )
    docs = [{"_key": "p", "title": "T", "text": LONG}]
    outcomes = await embed_documents(docs, specs, store_metadata=True)
    assert len(docs[0]["tvec"]) == 384
    for chunk in outcomes[0].chunks:
        assert chunk["tvec"] is None
        assert chunk["tvec_meta"]["status"] == "skipped"


@pytest.mark.parametrize(("warn", "level"), [(True, logging.WARNING), (False, logging.INFO)])
async def test_long_input_logging(provider, caplog, warn, level):
    caplog.set_level(logging.INFO, logger="app.services.pipeline")
    docs = [{"_key": "big", "text": LONG}, {"_key": "small", "text": SHORT}]
    await embed_documents(
        docs, _specs("truncate_end", warn_on_long_input=warn), store_metadata=True
    )
    records = [r for r in caplog.records if r.name == "app.services.pipeline"]
    assert len(records) == 1 and records[0].levelno == level
    msg = records[0].getMessage()
    assert "'big'" in msg and "12 tokens" in msg and "long_input=truncate_end" in msg
    assert "7 tokens dropped" in msg and "estimated" in msg


# ------------------------------------------------------------ analyze and pre-scan


class FakeCollection:
    def __init__(self) -> None:
        self.inserted: list[dict[str, Any]] = []

    def insert_many(self, docs: list[dict[str, Any]], **_: Any) -> list[dict[str, Any]]:
        self.inserted.extend(docs)
        return [{"_key": d.get("_key")} for d in docs]


class FakeDb:
    def __init__(self) -> None:
        self.col = FakeCollection()

    def has_collection(self, name: str) -> bool:
        return True

    def collection(self, name: str) -> FakeCollection:
        return self.col


@pytest.fixture
def no_profiles(monkeypatch):
    from app.services import profiles

    monkeypatch.setattr(profiles, "get_profile", lambda db, name: None)


def _handle():
    from datetime import UTC, datetime

    from app.models.jobs import Job
    from app.services.jobs import JobHandle

    return JobHandle(
        Job(
            id="j",
            kind="load_documents",
            collection="c",
            database="d",
            created_at=datetime.now(UTC),
        )
    )


def _load_opts(mode: str, **extra: Any):
    from app.models.load import LoadOptions

    return LoadOptions(
        collection="c", embeddings=[_spec(mode)], key_field="id", batch_size=2, **extra
    )


DOCS = [
    {"id": "s1", "text": SHORT},
    {"id": "s2", "text": "one two"},
    {"id": "s3", "text": SHORT},
    {"id": "big", "text": LONG},
    {"id": "huge", "text": LONG + " m n o"},
]


async def test_analyze_reports_and_writes_nothing(provider, no_profiles):
    from app.services.pipeline import run_load
    from app.services.sources import list_source

    db, handle = FakeDb(), _handle()
    await run_load(handle, db, _load_opts("chunk_mean", analyze=True), list_source(DOCS))  # type: ignore[arg-type]
    assert db.col.inserted == [] and provider.seen == []
    job = handle.job
    assert job.counts.read == 5 and job.counts.over_limit == 2 and job.counts.embedded == 0
    report = job.result["analysis"]
    assert report["documents"] == 5 and report["over_limit_documents"] == 2
    field = report["fields"]["vec"]
    assert field["max_tokens"] == 5 and field["over_limit"] == 2
    assert field["long_input"] == "chunk_mean" and field["token_count"] == "estimated"
    assert field["over_limit_pct"] == 40.0
    assert field["tokens"] == {"total": 35, "median": 3, "p90": 15, "p99": 15, "max": 15}
    assert field["tokens_over_limit"] == 7 + 10
    assert field["longest"] == [
        {"document": "huge", "tokens": 15},
        {"document": "big", "tokens": 12},
    ]


async def test_prescan_fails_before_embedding(provider, no_profiles, caplog):
    from app.services.pipeline import run_load
    from app.services.sources import list_source

    db, handle = FakeDb(), _handle()
    with pytest.raises(InputTooLongError) as exc:
        await run_load(handle, db, _load_opts("error"), list_source(DOCS))  # type: ignore[arg-type]
    # The long texts are the last rows, yet nothing was embedded or written.
    assert provider.seen == [] and db.col.inserted == []
    msg = str(exc.value.detail)
    assert "2 of 5 documents" in msg and "huge (15), big (12)" in msg and "long_input" in msg
    assert handle.job.result["prescan"] == {"status": "failed"}
    assert handle.job.result["analysis"]["fields"]["vec"]["over_limit"] == 2
    assert any(r.levelno == logging.ERROR and "2 of 5" in r.getMessage() for r in caplog.records)


async def test_prescan_passes_then_loads(provider, no_profiles):
    from app.services.pipeline import run_load
    from app.services.sources import list_source

    db, handle = FakeDb(), _handle()
    short = [d for d in DOCS if d["id"].startswith("s")]
    await run_load(handle, db, _load_opts("error"), list_source(short))  # type: ignore[arg-type]
    assert handle.job.result["prescan"] == {"status": "passed"}
    assert [d["_key"] for d in db.col.inserted] == ["s1", "s2", "s3"]
    assert handle.job.counts.read == 3  # the pre-scan does not count as reading


async def test_prescan_reads_uploaded_file_twice(provider, no_profiles, tmp_path):
    from app.services.pipeline import run_load
    from app.services.sources import file_source

    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in DOCS))
    db, handle = FakeDb(), _handle()
    source = file_source(str(path), filename="docs.jsonl")
    with pytest.raises(InputTooLongError):
        await run_load(handle, db, _load_opts("error"), source)  # type: ignore[arg-type]
    assert provider.seen == []


async def test_prescan_skipped_for_read_once_source(provider, no_profiles):
    from app.services.pipeline import run_load
    from app.services.sources import DocumentSource

    db, handle = FakeDb(), _handle()
    source = DocumentSource(iter([dict(d) for d in DOCS]))  # like a URL: no reopen
    with pytest.raises(InputTooLongError):
        await run_load(handle, db, _load_opts("error"), source)  # type: ignore[arg-type]
    assert handle.job.result["prescan"]["status"] == "skipped"
    # Without a pre-scan, the batches before the long row were already embedded.
    assert provider.seen


async def test_no_prescan_for_other_modes(provider, no_profiles):
    from app.services.pipeline import run_load
    from app.services.sources import list_source

    db, handle = FakeDb(), _handle()
    await run_load(handle, db, _load_opts("truncate_end"), list_source(DOCS))  # type: ignore[arg-type]
    assert "prescan" not in handle.job.result
    assert len(db.col.inserted) == 5 and handle.job.counts.over_limit == 2


# ------------------------------------------------------------ real tokenizers


ALL_MODES = ["truncate_end", "truncate_start", "truncate_middle", "chunk_mean", "chunk_docs"]


def _check_pieces(p: EmbeddingProvider, model: str, options: Any, mode: str, text: str):
    spec = EmbeddingFieldSpec(source_field="t", provider=p.name, model=model, long_input=mode)
    (plan,) = p.plan_inputs(
        [text],
        model=model,
        options=options,
        input_type="document",
        long_input=spec.long_input_settings,
    )
    assert plan.overrun and plan.stats["token_count"] == "exact"
    if mode in p.native_long_input(model):
        assert plan.pieces == [text] and plan.stats["truncated"]  # the API truncates
        return plan
    tokenizer = p.exact_tokenizer(model, options)
    assert tokenizer is not None
    limit = plan.stats["max_tokens"]
    for piece, tokens in zip(plan.pieces, tokenizer.tokenize(plan.pieces), strict=True):
        assert len(tokens) <= limit, (mode, len(tokens), limit)
        for part in piece.split("\n"):
            assert part in text
            assert (
                part.split()[0].startswith("word")
                or part.split()[0] == "antidisestablishmentarianism"
            )
    if mode == "chunk_mean":
        assert len(plan.pieces) > 1
    return plan


LONG_REAL = " ".join(f"word{i} antidisestablishmentarianism" for i in range(2500))


@pytest.mark.local
@pytest.mark.parametrize("mode", ALL_MODES)
async def test_sentence_transformers_pieces_fit(mode):
    """A real (fast) Hugging Face tokenizer: every piece fits and no word is cut in half."""
    p = get_provider("sentence_transformers")
    model = "sentence-transformers/all-MiniLM-L6-v2"
    opts = p.parse_options({})
    plan = _check_pieces(p, model, opts, mode, LONG_REAL[:4000])
    vectors = await p.embed(plan.pieces, model=model, options=opts)
    assert len(vectors) == len(plan.pieces)


@pytest.mark.local
@pytest.mark.parametrize("mode", ALL_MODES)
def test_openai_tiktoken_pieces_fit(mode):
    """OpenAI models are counted exactly with tiktoken (downloads the encoding once)."""
    pytest.importorskip("tiktoken")
    p = get_provider("openai")
    _check_pieces(p, "text-embedding-3-small", p.parse_options({}), mode, LONG_REAL)


@pytest.mark.local
@pytest.mark.parametrize("mode", ALL_MODES)
def test_voyage_tokenizer_pieces_fit(mode):
    """Voyage models are counted with their tokenizer from the Hugging Face Hub."""
    pytest.importorskip("tokenizers")
    p = get_provider("voyage")
    _check_pieces(p, "voyage-3.5", p.parse_options({}), mode, LONG_REAL * 4)
