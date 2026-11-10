"""Inputs longer than the model's token limit (`long_input` on the embedding spec)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from app.models.embedding import EmbeddingFieldSpec
from app.providers.base import EmbeddingProvider, InputType, ProviderOptions
from app.providers.long_input import EstimateTokenizer, LongInput, plan_text
from app.providers.registry import get_provider, register_provider
from app.services.pipeline import (
    InputTooLongError,
    StillTooLongError,
    chunk_documents,
    embed_documents,
    is_settled,
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
    assert spec.long_input == "chunk_docs"
    assert spec.chunk_overlap_tokens == 25
    assert spec.max_tokens is None and spec.token_chars == 3.0
    assert spec.warn_on_long_input is True


def test_estimate_tokenizer():
    (tokens,) = EstimateTokenizer(4).tokenize(["abcdefgh ab  abcde"])
    assert len(tokens) == 2 + 1 + 2
    assert tokens.word_start == [True, False, True, True, False]


def test_known_limits():
    assert get_provider("openai").token_limit("text-embedding-3-small", None) == 8191  # type: ignore[arg-type]
    assert get_provider("cohere").token_limit("embed-english-v3.0", None) == 512  # type: ignore[arg-type]
    assert get_provider("ollama").token_limit("nomic-embed-text", None) is None  # type: ignore[arg-type]


class _Capture:
    """Stands in for a provider's SDK client or HTTP client and records the request."""

    def __init__(self) -> None:
        self.sent: dict[str, Any] = {}

    async def embed(self, *args: Any, **kwargs: Any) -> Any:  # Cohere and Voyage clients
        self.sent = kwargs

        class R:
            class embeddings:  # noqa: N801
                float_ = [[0.0]]

        r = R()
        if "truncation" in kwargs:  # Voyage
            r.embeddings = [[0.0]]  # type: ignore[assignment]
        return r


def _sent(name: str, model: str, options: Any, monkeypatch: pytest.MonkeyPatch) -> dict:
    """The request body the provider sends for ``options``."""
    p = get_provider(name)
    capture = _Capture()
    if name in ("cohere", "voyage"):
        monkeypatch.setattr(p, "_client", lambda o: capture)
    elif name == "bedrock":
        monkeypatch.setattr(p, "_client", lambda o: None)

        def invoke(client: Any, model: str, body: dict) -> dict:
            capture.sent = body
            return {"embeddings": [[0.0]]}

        monkeypatch.setattr(p, "_invoke", invoke)
    elif name == "ollama":
        import httpx

        class Client:
            def __init__(self, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def post(self, url: str, json: dict) -> httpx.Response:
                capture.sent = json
                return httpx.Response(
                    200, json={"embeddings": [[0.0]]}, request=httpx.Request("POST", url)
                )

        monkeypatch.setattr(httpx, "AsyncClient", Client)
    import asyncio

    asyncio.run(
        p.embed_batch(["x"], model=model, options=options, dimensions=None, input_type="document")
    )
    return capture.sent


NATIVE = [
    (
        "cohere",
        "embed-english-v3.0",
        "truncate",
        {"truncate_end": "END", "truncate_start": "START", "error": "NONE", "chunk_mean": "END"},
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
]


@pytest.mark.parametrize(("name", "model", "field", "expected"), NATIVE)
def test_native_truncation_sent_from_long_input(name, model, field, expected, monkeypatch):
    p = get_provider(name)
    for mode, value in expected.items():
        options = p.apply_long_input(p.parse_options({}), model, mode)
        assert _sent(name, model, options, monkeypatch)[field] == value, mode


@pytest.mark.parametrize(("name", "model", "field", "expected"), NATIVE)
def test_truncate_options_are_not_provider_options(name, model, field, expected):
    """One setting, everywhere: the provider option schemas (API, GUI) no longer list them."""
    schema = get_provider(name).describe()["options_schema"]
    assert not {"truncate", "truncation"} & set(schema.get("properties", {}))


@pytest.mark.parametrize(
    ("options", "mode"),
    [
        ({"truncate": "START"}, "truncate_start"),
        ({"truncate": "end"}, "truncate_end"),
        ({"truncate": "NONE"}, "error"),
        ({"truncation": False}, "error"),
        ({"truncation": True}, "truncate_end"),
        ({"truncate": False}, "error"),
    ],
)
def test_legacy_truncate_option_sets_long_input(options, mode, caplog):
    spec = EmbeddingFieldSpec(source_field="t", provider="cohere", provider_options=options)
    assert spec.long_input == mode
    assert spec.provider_options == {}
    assert "deprecated" in caplog.text


def test_legacy_truncate_option_loses_to_long_input(caplog):
    spec = EmbeddingFieldSpec(
        source_field="t",
        provider="cohere",
        provider_options={"truncate": "START", "base_url": "http://x"},
        long_input="chunk_mean",
    )
    assert spec.long_input == "chunk_mean"
    assert spec.provider_options == {"base_url": "http://x"}
    assert "ignored" in caplog.text


def test_legacy_truncate_option_unrecognized_keeps_default():
    spec = EmbeddingFieldSpec(source_field="t", provider="voyage", provider_options={"truncate": 3})
    assert spec.long_input == "chunk_docs" and spec.provider_options == {}


@pytest.mark.parametrize(("name", "model", "field", "expected"), NATIVE)
def test_legacy_truncate_option_on_options_directly(name, model, field, expected, monkeypatch):
    """Library callers building provider options with the old key keep its behaviour."""
    p = get_provider(name)
    options = p.parse_options({field: False if field == "truncation" else "NONE"})
    assert options.long_input_mode == "error"
    assert _sent(name, model, options, monkeypatch)[field] == expected["error"]


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


def _url_source_with(monkeypatch, body: bytes, gets: list[str]):
    """A URL source served from memory; ``gets`` records each download."""
    import httpx

    from app.services import sources

    def handler(request: httpx.Request) -> httpx.Response:
        gets.append(str(request.url))
        return httpx.Response(200, content=body)

    real = httpx.Client
    monkeypatch.setattr(
        sources.httpx,
        "Client",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return sources.url_source(["https://example.com/docs.jsonl"])


async def test_url_load_is_spooled_once_for_the_prescan(
    provider, no_profiles, monkeypatch, tmp_path
):
    from app.config import get_settings
    from app.services.pipeline import run_load

    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    short = [d for d in DOCS if d["id"].startswith("s")]
    gets: list[str] = []
    source = _url_source_with(monkeypatch, "\n".join(json.dumps(d) for d in short).encode(), gets)
    db, handle = FakeDb(), _handle()
    await run_load(handle, db, _load_opts("error"), source)  # type: ignore[arg-type]
    for cleanup in handle.cleanup:
        cleanup()
    assert len(gets) == 1  # downloaded once, read twice from disk
    assert handle.job.result["prescan"] == {"status": "passed"}
    assert handle.job.result["spooled"] is True
    assert [d["_key"] for d in db.col.inserted] == ["s1", "s2", "s3"]
    assert list(tmp_path.iterdir()) == []  # the spooled copy is removed


async def test_url_prescan_fails_before_embedding(provider, no_profiles, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.services.pipeline import run_load

    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    gets: list[str] = []
    source = _url_source_with(monkeypatch, "\n".join(json.dumps(d) for d in DOCS).encode(), gets)
    db, handle = FakeDb(), _handle()
    with pytest.raises(InputTooLongError):
        await run_load(handle, db, _load_opts("error"), source)  # type: ignore[arg-type]
    assert provider.seen == [] and len(gets) == 1


# ------------------------------------------------- provider rejections and batching


class StrictWordProvider(WordProvider):
    """Rejects any request holding a text over ``real_limit`` words (an estimate was too low),
    or more than ``request_limit`` words in total."""

    name = "strict_words_test"
    real_limit = 4
    request_limit = 1000

    async def embed_batch(self, texts, **kwargs):  # type: ignore[override]
        if any(len(t.split()) > self.real_limit for t in texts):
            raise ValueError("This model's maximum context length is 4 tokens")
        if sum(len(t.split()) for t in texts) > self.request_limit:
            raise ValueError("Requested too many tokens: max tokens per request is 6")
        return await super().embed_batch(texts, **kwargs)


@pytest.fixture
def strict() -> StrictWordProvider:
    register_provider(StrictWordProvider)
    p = get_provider(StrictWordProvider.name)
    assert isinstance(p, StrictWordProvider)
    return p


def _strict(mode: str, **extra: Any):
    return _specs(mode, provider="strict_words_test", max_tokens=6, **extra)


@pytest.mark.parametrize("mode", ["truncate_end", "chunk_mean", "chunk_docs"])
async def test_rejected_text_is_planned_again_smaller(strict, mode, caplog):
    docs = [{"_key": "a", "text": LONG}, {"_key": "b", "text": SHORT}]
    outcomes = await embed_documents(docs, _strict(mode), store_metadata=True)
    meta = docs[0]["vec_meta"]
    assert meta["max_tokens"] == 4 and meta["max_tokens_reduced_from"] == 6
    assert all(len(t.split()) <= 4 for t in strict.seen)
    if mode == "chunk_docs":
        assert [len(c["text"].split()) for c in chunk_documents(outcomes)] == [4, 4, 4]
    assert docs[1]["vec"] == [3.0, 1.0] and "max_tokens_reduced_from" not in docs[1]["vec_meta"]
    assert "rejected" in caplog.text


async def test_rejected_text_with_error_mode_raises(strict):
    docs = [{"_key": "a", "text": "a b c d e"}]
    with pytest.raises(InputTooLongError, match="rejected"):
        await embed_documents(docs, _strict("error"), store_metadata=True)


async def test_request_too_large_falls_back_to_one_text_per_request(strict):
    strict.request_limit = 6
    try:
        docs = [{"_key": str(i), "text": SHORT} for i in range(4)]  # 3 words each
        await embed_documents(docs, _strict("truncate_end"), store_metadata=True)
        assert all(d["vec"] == [3.0, 1.0] for d in docs)
        assert all("max_tokens_reduced_from" not in d["vec_meta"] for d in docs)
    finally:
        strict.request_limit = 1000


def test_batches_respect_token_budget():
    from app.providers.base import _batches

    texts = list("abcdef")
    assert _batches(texts, None, 4, 100) == [["a", "b", "c", "d"], ["e", "f"]]
    # 80% of 10 tokens per batch; a text over the budget gets a batch of its own.
    assert _batches(texts, [3, 3, 3, 20, 1, 1], 10, 10) == [
        ["a", "b"],
        ["c"],
        ["d"],
        ["e", "f"],
    ]


async def test_errors_per_document_marks_and_continues(provider, caplog):
    """The watcher's mode: the long document fails alone and leaves the queue."""
    docs = [{"_key": "a", "text": LONG}, {"_key": "b", "text": SHORT}]
    outcomes = await embed_documents(
        docs, _specs("error"), store_metadata=True, errors_per_document=True
    )
    assert outcomes[0].error and "12 tokens" in outcomes[0].error
    meta = docs[0]["vec_meta"]
    assert meta["status"] == "error" and meta["reason"] == "too long"
    assert meta["input_tokens"] == 12 and "vec" not in docs[0]
    assert docs[1]["vec"] == [3.0, 1.0] and outcomes[1].embedded
    assert provider.seen == [SHORT]
    assert any(r.levelno == logging.ERROR for r in caplog.records)


async def test_errors_per_document_after_provider_rejection(strict):
    docs = [{"_key": "a", "text": "a b c d e"}, {"_key": "b", "text": SHORT}]
    outcomes = await embed_documents(
        docs, _strict("error"), store_metadata=True, errors_per_document=True
    )
    assert outcomes[0].error and docs[0]["vec_meta"]["status"] == "error"
    assert docs[1]["vec"] == [3.0, 1.0]


class PoisonProvider(WordProvider):
    """Rejects as too long any request holding the word ``poison``, however short."""

    name = "poison_words_test"

    async def embed_batch(self, texts, **kwargs):  # type: ignore[override]
        if any("poison" in t for t in texts):
            raise ValueError("This model's maximum context length is 4 tokens")
        return await super().embed_batch(texts, **kwargs)


@pytest.mark.parametrize("mode", ["truncate_end", "chunk_docs"])
async def test_still_rejected_text_fails_alone_per_document(mode):
    """The watcher: a text the provider keeps rejecting fails its own document, not the batch."""
    register_provider(PoisonProvider)
    specs = _specs(mode, provider="poison_words_test", max_tokens=6)
    docs = [{"_key": "a", "text": "poison " + LONG}, {"_key": "b", "text": SHORT}]
    outcomes = await embed_documents(docs, specs, store_metadata=True, errors_per_document=True)
    assert outcomes[0].error and "kept rejecting" in outcomes[0].error
    assert docs[0]["vec_meta"]["status"] == "error" and "vec" not in docs[0]
    assert not outcomes[0].chunks
    assert docs[1]["vec"] == [3.0, 1.0] and outcomes[1].embedded

    with pytest.raises(StillTooLongError, match="kept rejecting"):
        await embed_documents([{"_key": "a", "text": "poison " + LONG}], specs, store_metadata=True)


@pytest.mark.parametrize(
    "text",
    ["\u6c49" * 3000, " ".join(["ab"] * 10) + " " + "x" * 6000],
    ids=["no-spaces", "one-long-word"],
)
@pytest.mark.parametrize(
    "mode", ["truncate_end", "truncate_start", "truncate_middle", "chunk_docs"]
)
def test_estimated_counts_cut_text_without_spaces(text, mode):
    """With an estimate, a word longer than the limit is cut by characters."""
    tokens = EstimateTokenizer(3.0).tokenize([text])[0]
    plan = plan_text(text, tokens, 512, LongInput(mode=mode, chunk_overlap_tokens=0), exact=False)
    assert all(t <= 512 for t in plan.piece_tokens)
    assert all(len(p) <= 512 * 3 + 1 for p in plan.pieces)  # + the truncate_middle join
    if mode == "chunk_docs":
        assert "".join(plan.pieces).replace(" ", "") == text.replace(" ", "")
        assert len(plan.pieces) > 1
    else:
        assert plan.stats["truncated"] and len(plan.pieces[0]) < len(text)


async def test_failed_document_keeps_its_vector_when_another_field_is_chunked(provider):
    """Field `vec` would be split, but field `tvec` fails: nothing about `vec` may change."""
    specs = resolve_specs(
        [
            _spec("chunk_docs"),
            EmbeddingFieldSpec(
                source_field="missing",
                target_field="tvec",
                preset="fake",
                on_missing="error",
            ),
        ]
    )
    old_meta = {"status": "embedded", "model": "m"}
    docs = [
        {"_key": "p", "text": LONG, "vec": [9.0, 9.0], "vec_meta": old_meta},
        {"_key": "q", "text": LONG},
    ]
    outcomes = await embed_documents(docs, specs, store_metadata=True)
    assert outcomes[0].error and not outcomes[0].chunks
    assert docs[0]["vec"] == [9.0, 9.0] and docs[0]["vec_meta"] == old_meta
    assert "vec" not in docs[1] and "vec_meta" not in docs[1]  # still pending
    assert docs[0]["tvec_meta"]["status"] == "error"


class ChunkStore:
    """A collection and database: insert fails for keys in ``fail``; ``existing`` documents
    are found by the existence check."""

    name = "c"

    def __init__(self, fail: set[str] = frozenset(), existing: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        self.fail, self.existing = fail, existing
        self.docs: dict[str, dict] = {}
        self.deleted: list[str] = []

        class Aql:
            def execute(inner, query: str, bind_vars: dict) -> list:  # noqa: N805
                if "FOR k IN @keys" in query:
                    return [k for k in bind_vars["keys"] if k in self.existing]
                return []  # no stale chunks

        self.aql = Aql()

    def insert_many(self, docs: list[dict], **_: Any) -> list:
        out: list = []
        for d in docs:
            if d["_key"] in self.fail:
                out.append(ValueError("illegal document key"))
            else:
                self.docs[d["_key"]] = d
                out.append({"_key": d["_key"]})
        return out

    def delete_many(self, docs: list[dict], **_: Any) -> bool:
        for d in docs:
            self.deleted.append(d["_key"])
            self.docs.pop(d["_key"], None)
        return True


async def test_chunk_write_failure_keeps_the_original(provider, caplog):
    from app.services.pipeline import write_chunks_first

    specs = _specs("chunk_docs")
    old = {"status": "embedded", "model": "m"}
    docs = [
        {"_key": "p", "text": LONG, "vec": [9.0, 9.0], "vec_meta": old},
        {"_key": "q", "text": LONG},
    ]
    outcomes = await embed_documents(docs, specs, store_metadata=True)
    store = ChunkStore(fail={"p_vec_chunk_1"})
    write = write_chunks_first(store, store, docs, outcomes, specs, store_metadata=True)  # type: ignore[arg-type]
    # p: its written chunks are removed, its old vector stays, the failure is recorded.
    assert outcomes[0].error and "p_vec_chunk_1" in outcomes[0].error
    assert outcomes[0].chunk_write_failed
    assert docs[0]["vec"] == [9.0, 9.0]
    assert docs[0]["vec_meta"]["status"] == "error"
    assert docs[0]["vec_meta"]["reason"] == "chunk write failed"
    assert not any(k.startswith("p_") for k in store.docs)
    assert "p_vec_chunk_0" in store.deleted
    # q is unaffected.
    assert not outcomes[1].error and docs[1]["vec_meta"]["status"] == "chunked"
    assert sorted(store.docs) == ["q_vec_chunk_0", "q_vec_chunk_1", "q_vec_chunk_2"]
    assert write.written == 3 and write.keys == sorted(store.docs)


async def test_chunks_of_existing_documents_skipped_when_not_written(provider):
    """on_duplicate ignore/error: the stored original won't be replaced, so neither may its
    chunks be."""
    from app.services.pipeline import write_chunks_first

    specs = _specs("chunk_docs")
    docs = [{"_key": "p", "text": LONG}, {"_key": "q", "text": LONG}]
    outcomes = await embed_documents(docs, specs, store_metadata=True)
    store = ChunkStore(existing={"p"})
    write = write_chunks_first(  # type: ignore[arg-type]
        store, store, docs, outcomes, specs, store_metadata=True, skip_existing_parents=True
    )
    assert not any(k.startswith("p_") for k in store.docs) and not outcomes[0].chunks
    assert write.written == 3 and all(k.startswith("q_") for k in store.docs)


@pytest.mark.parametrize(("skip_existing", "kept"), [(True, "first"), (False, "second")])
async def test_duplicate_keys_in_a_batch_write_one_set_of_chunks(provider, skip_existing, kept):
    """Only the copy that the insert will keep writes chunks: the first under error/ignore,
    the last under update/replace."""
    from app.services.pipeline import write_chunks_first

    specs = _specs("chunk_docs")
    first, second = LONG, "n o p q r s t u v w x y z"  # 12 and 13 words: 3 chunks each
    docs = [{"_key": "p", "text": first}, {"_key": "p", "text": second}]
    outcomes = await embed_documents(docs, specs, store_metadata=True)
    store = ChunkStore()
    write_chunks_first(  # type: ignore[arg-type]
        store,
        store,
        docs,
        outcomes,
        specs,
        store_metadata=True,
        skip_existing_parents=skip_existing,
    )
    source = first if kept == "first" else second
    assert store.docs["p_vec_chunk_0"]["text"] == " ".join(source.split()[:5])
    assert len(store.docs) == 3


async def test_load_writes_document_whose_chunks_failed(provider, no_profiles):
    """The document is stored with its error, not left out (a stored version would otherwise
    stay marked as chunked without its chunks)."""
    from app.services.pipeline import run_load
    from app.services.sources import list_source

    store = ChunkStore(fail={"big_vec_chunk_1"})
    store.inserted = []  # type: ignore[attr-defined]
    real_insert = store.insert_many

    def insert_many(docs: list[dict], **kw: Any) -> list:
        if not docs[0]["_key"].count("_chunk_"):
            store.inserted.extend(docs)  # type: ignore[attr-defined]
        return real_insert(docs, **kw)

    store.insert_many = insert_many  # type: ignore[method-assign]

    class Db:
        aql = store.aql

        def has_collection(self, name: str) -> bool:
            return True

        def collection(self, name: str) -> ChunkStore:
            return store

    handle = _handle()
    docs = [{"id": "big", "text": LONG}, {"id": "s1", "text": SHORT}]
    await run_load(handle, Db(), _load_opts("chunk_docs"), list_source(docs))  # type: ignore[arg-type]
    stored = {d["_key"]: d for d in store.inserted}  # type: ignore[attr-defined]
    assert stored["big"]["vec_meta"]["status"] == "error"
    assert stored["big"]["vec_meta"]["reason"] == "chunk write failed"
    assert "vec" not in stored["big"] and stored["s1"]["vec"] == [3.0, 1.0]
    assert handle.job.counts.chunks == 0 and len(handle.job.errors) == 1


async def test_reembedded_chunk_keeps_its_link(provider):
    specs = resolve_specs(
        [
            _spec("chunk_docs"),
            EmbeddingFieldSpec(source_field="title", target_field="tvec", preset="fake"),
        ]
    )
    _, chunks = await _chunked(specs)
    chunk = chunks[1]
    # A full re-embed (no only_missing) reads the chunk back and embeds it on its own.
    await embed_documents([chunk], specs, store_metadata=True)
    meta = chunk["vec_meta"]
    assert meta["status"] == "embedded" and meta["parent_key"] == "p"
    assert meta["chunk_index"] == 1 and meta["chunk_count"] == 3
    # Its other field stays off, so vector searches on it don't return the chunk.
    assert chunk["tvec"] is None and chunk["tvec_meta"]["reason"] == "chunk document"


async def test_url_download_cap(provider, no_profiles, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.errors import AppError
    from app.services.pipeline import run_load

    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    monkeypatch.setattr(get_settings(), "max_url_download_mb", 0)
    source = _url_source_with(monkeypatch, json.dumps(DOCS[0]).encode(), [])
    db, handle = FakeDb(), _handle()
    with pytest.raises(AppError) as exc:
        await run_load(handle, db, _load_opts("error"), source)  # type: ignore[arg-type]
    assert exc.value.error == "url_download_too_large"
    assert provider.seen == [] and list(tmp_path.iterdir()) == []


# ------------------------------------------------------------ test endpoint


def _client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


def test_embed_endpoint_applies_long_input(provider):
    body = {"texts": [LONG, SHORT], "max_tokens": 5, "token_chars": 100, "chunk_overlap_tokens": 0}
    r = _client().post("/api/v1/providers/words_test/embed", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    chunked, short = out["results"]
    assert chunked["status"] == "chunked" and len(chunked["vectors"]) == 3
    assert chunked["meta"]["chunks"] == 3 and chunked["meta"]["long_input"] == "chunk_docs"
    assert short["status"] == "embedded" and short["vectors"] == [[3.0, 1.0]]
    assert out["count"] == 4 and out["dimension"] == 2 and len(out["vectors"]) == 4

    mean = (
        _client()
        .post("/api/v1/providers/words_test/embed", json=body | {"long_input": "chunk_mean"})
        .json()
    )
    assert [len(r["vectors"]) for r in mean["results"]] == [1, 1]

    err = _client().post("/api/v1/providers/words_test/embed", json=body | {"long_input": "error"})
    assert err.status_code == 422 and err.json()["error"] == "input_too_long"


def test_embed_endpoint_maps_legacy_truncate_option(provider):
    body = {
        "texts": [LONG],
        "max_tokens": 5,
        "token_chars": 100,
        "provider_options": {"truncate": "START"},
    }
    r = _client().post("/api/v1/providers/words_test/embed", json=body)
    assert r.status_code == 200, r.text
    result = r.json()["results"][0]
    assert result["meta"]["long_input"] == "truncate_start" and result["vectors"] == [[5.0, 1.0]]


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


# ------------------------------------------------------- only_missing and chunk_docs


async def _chunked(specs):
    docs = [{"_key": "p", "title": "T", "text": LONG}]
    outcomes = await embed_documents(docs, specs, store_metadata=True)
    return docs[0], chunk_documents(outcomes)


async def test_chunked_original_is_settled_until_it_changes(provider):
    specs = _specs("chunk_docs")
    parent, chunks = await _chunked(specs)
    spec = specs[0]
    assert parent["vec"] is None and "text_hash" in parent["vec_meta"]
    assert is_settled(parent, spec)
    assert all(is_settled(c, spec) for c in chunks)

    # The user shortens the text (e.g. to chunk it themselves): embedded again.
    edited = parent | {"text": SHORT}
    assert not is_settled(edited, spec)
    # ... or removes the metadata, or the source.
    assert not is_settled({k: v for k, v in parent.items() if k != "vec_meta"}, spec)
    assert not is_settled({k: v for k, v in parent.items() if k != "text"}, spec)
    # A different mode or model also re-embeds it.
    assert not is_settled(parent, _specs("chunk_mean")[0])
    assert not is_settled(parent, _specs("chunk_docs", model="other")[0])


async def test_keep_chunked_leaves_current_chunks_alone(provider):
    specs = _specs("chunk_docs")
    parent, _ = await _chunked(specs)
    provider.seen.clear()
    outcomes = await embed_documents([parent], specs, store_metadata=True, keep_chunked=True)
    assert provider.seen == [] and not outcomes[0].embedded and not outcomes[0].chunks

    edited = parent | {"text": SHORT}
    outcomes = await embed_documents([edited], specs, store_metadata=True, keep_chunked=True)
    assert edited["vec"] == [3.0, 1.0] and edited["vec_meta"]["status"] == "embedded"
    assert outcomes[0].embedded and not outcomes[0].chunks


async def test_other_fields_of_chunk_documents_are_settled(provider):
    specs = resolve_specs(
        [
            _spec("chunk_docs"),
            EmbeddingFieldSpec(source_field="title", target_field="tvec", preset="fake"),
        ]
    )
    _, chunks = await _chunked(specs)
    assert chunks and all(is_settled(c, s) for c in chunks for s in specs)


def test_overlap_inside_long_words_stays_near_the_overlap():
    """A chunk that ends inside a long word overlaps the next by about `chunk_overlap_tokens`,
    not by most of the word."""
    text = ("a " + "x" * 3000 + " ") * 5
    tokens = EstimateTokenizer(1.0).tokenize([text])[0]
    plain, overlapped = (
        plan_text(
            text, tokens, 512, LongInput(mode="chunk_docs", chunk_overlap_tokens=o), exact=False
        )
        for o in (0, 25)
    )
    extra = sum(overlapped.piece_tokens) - sum(plain.piece_tokens)
    assert 0 < extra <= 25 * len(overlapped.pieces)
