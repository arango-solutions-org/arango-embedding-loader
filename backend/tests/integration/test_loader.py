from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SAMPLE = Path(__file__).parents[3] / "examples" / "pubmed" / "sample.jsonl"
FAKE = {"source_field": "text", "target_field": "embedding", "preset": "fake"}


def test_load_json_body_and_duplicates(client, db):
    body = {
        "collection": "docs",
        "key_field": "id",
        "embeddings": [FAKE],
        "documents": [{"id": 1, "text": "alpha"}, {"id": 2, "text": "beta"}, {"id": 3}],
    }
    job = client.post("/api/v1/load?wait=true", json=body).json()
    assert job["status"] == "succeeded", job
    assert job["counts"] == {
        "read": 3,
        "embedded": 2,
        "written": 3,
        "updated": 0,
        "skipped": 1,
        "failed": 0,
        "over_limit": 0,
        "chunks": 0,
    }
    doc = db.collection("docs").get("1")
    assert len(doc["embedding"]) == 384 and doc["text"] == "alpha"
    assert doc["embedding_meta"]["status"] == "embedded"
    assert "embedding" not in db.collection("docs").get("3")

    dup = client.post("/api/v1/load?wait=true", json=body | {"documents": body["documents"][:1]})
    assert dup.json()["counts"]["failed"] == 1

    upd = client.post(
        "/api/v1/load?wait=true",
        json=body | {"on_duplicate": "update", "documents": [{"id": 1, "text": "changed"}]},
    ).json()
    assert upd["counts"]["updated"] == 1
    assert db.collection("docs").get("1")["text"] == "changed"


def test_async_job_and_events(client):
    body = {"collection": "docs_async", "embeddings": [FAKE], "documents": [{"text": "x"}] * 10}
    r = client.post("/api/v1/load", json=body)
    assert r.status_code == 202
    job_id = r.json()["id"]
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
        events = [line for line in stream.iter_lines() if line.startswith("data:")]
    final = json.loads(events[-1][5:])
    assert final["status"] == "succeeded" and final["counts"]["written"] == 10


def test_profile_file_load_watcher_and_vector_search(client, db):
    profile = {
        "embeddings": [FAKE | {"model": "fake-768"}],
        "key_field": "id",
        "vector_index": {"create": True, "metric": "cosine"},
        "auto_watch": True,
        "watch_interval_s": 1,
    }
    r = client.put("/api/v1/collections/pubmed/profile", json=profile)
    assert r.status_code == 200, r.text
    assert r.json()["watcher"]["running"]

    with SAMPLE.open("rb") as fh:
        job = client.post(
            "/api/v1/load/file?collection=pubmed&wait=true", files={"file": ("s.jsonl", fh)}
        ).json()
    assert job["status"] == "succeeded", job
    assert job["config"]["from_profile"] is True
    assert job["counts"]["written"] == 120
    assert job["result"]["vector_index"]["status"] == "created"
    first = json.loads(SAMPLE.read_text().splitlines()[0])
    assert len(db.collection("pubmed").get(str(first["id"]))["embedding"]) == 768

    # Inserted by "another client": the watcher embeds it.
    db.collection("pubmed").insert({"_key": "raw1", "text": "inserted directly"})
    for _ in range(30):
        doc = db.collection("pubmed").get("raw1")
        if doc.get("embedding"):
            break
        time.sleep(0.5)
    assert len(doc["embedding"]) == 768

    status = client.get("/api/v1/collections/pubmed/profile").json()
    assert status["pending_documents"] == 0
    assert status["embedded_documents"]["embedding"] == 121

    cursor = db.aql.execute(
        "FOR d IN pubmed SORT APPROX_NEAR_COSINE(d.embedding, @q) DESC LIMIT 3 RETURN d._key",
        bind_vars={"q": doc["embedding"]},
    )
    assert list(cursor)[0] == "raw1"

    assert client.post("/api/v1/collections/pubmed/profile/watcher/stop").json()["running"] is False
    assert client.delete("/api/v1/collections/pubmed/profile").status_code == 204


def test_reembed_only_missing_and_nested(client, db):
    col = db.create_collection("existing")
    col.insert_many([{"_key": "a", "c": {"body": "one"}}, {"_key": "b", "c": {"body": "two"}}])
    spec = {"source_field": "c.body", "target_field": "vec.body", "preset": "fake"}
    job = client.post(
        "/api/v1/collections/existing/embed?wait=true", json={"embeddings": [spec]}
    ).json()
    assert job["status"] == "succeeded", job
    assert job["counts"]["updated"] == 2
    assert len(col.get("a")["vec"]["body"]) == 384
    again = client.post(
        "/api/v1/collections/existing/embed?wait=true", json={"embeddings": [spec]}
    ).json()
    assert again["total"] == 0 and again["counts"]["read"] == 0


def test_csv_and_collections_endpoints(client):
    csv_bytes = b"name,desc\nx,first item\ny,second item\n"
    job = client.post(
        "/api/v1/load/file?wait=true",
        files={"file": ("t.csv", csv_bytes)},
        data={
            "config": json.dumps(
                {
                    "collection": "csvdocs",
                    "embeddings": [{"source_field": "desc", "preset": "fake"}],
                }
            )
        },
    ).json()
    assert job["status"] == "succeeded" and job["counts"]["written"] == 2
    cols = {c["name"]: c for c in client.get("/api/v1/collections").json()}
    assert cols["csvdocs"]["count"] == 2
    sample = client.get("/api/v1/collections/csvdocs/sample").json()
    assert {"desc", "desc_embedding"} <= {f["path"] for f in sample["fields"]}


@pytest.mark.local
def test_pubmedbert_customer_scenario(client, db):
    pytest.importorskip("sentence_transformers")
    profile = json.loads((SAMPLE.parent / "profile.json").read_text())
    profile["auto_watch"] = False
    assert client.put("/api/v1/collections/pubmed_real/profile", json=profile).status_code == 200
    with SAMPLE.open("rb") as fh:
        job = client.post(
            "/api/v1/load/file?collection=pubmed_real&wait=true", files={"file": ("s.jsonl", fh)}
        ).json()
    assert job["status"] == "succeeded", job
    assert job["result"]["dimensions"] == {"embedding": 768}
    doc = db.collection("pubmed_real").get(
        str(json.loads(SAMPLE.read_text().splitlines()[0])["id"])
    )
    assert doc["embedding_meta"]["model"] == "NeuML/pubmedbert-base-embeddings"


@pytest.mark.local
def test_over_limit_inputs(client, db):
    """Over-limit texts: split into chunk documents (the default), truncated, or the run stops."""
    pytest.importorskip("sentence_transformers")
    long_text = " ".join(f"token{i}" for i in range(600))  # well over MiniLM's 256 tokens
    spec = {
        "source_field": "text",
        "target_field": "embedding",
        "provider": "sentence_transformers",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    body = {
        "collection": "overrun",
        "key_field": "id",
        "embeddings": [spec],
        "documents": [{"id": "long", "text": long_text}, {"id": "short", "text": "hello"}],
    }
    job = client.post("/api/v1/load?wait=true", json=body).json()
    assert job["status"] == "succeeded", job
    assert job["counts"]["over_limit"] == 1 and job["counts"]["written"] == 2
    chunks = job["counts"]["chunks"]
    assert chunks >= 2

    col = db.collection("overrun")
    parent = col.get("long")
    assert parent["embedding"] is None and parent["embedding_meta"]["status"] == "chunked"
    assert parent["embedding_meta"]["chunks"] == chunks
    assert parent["embedding_meta"]["long_input"] == "chunk_docs"
    first = col.get("long_embedding_chunk_0")
    assert len(first["embedding"]) == 384 and first["text"] in long_text
    assert first["embedding_meta"]["parent_key"] == "long"
    short_meta = col.get("short")["embedding_meta"]
    assert short_meta["truncated"] is False and short_meta["chunks"] == 1
    assert short_meta["token_count"] == "exact"

    # Loading again replaces the chunks instead of failing on duplicate keys.
    again = client.post("/api/v1/load?wait=true", json=body | {"on_duplicate": "replace"}).json()
    assert again["status"] == "succeeded" and again["counts"]["failed"] == 0, again

    # A shorter text, still over the limit, splits into fewer chunks: the extra chunk
    # documents from the earlier split are removed.
    shorter = " ".join(f"token{i}" for i in range(200))
    docs = [{"id": "long", "text": shorter}, {"id": "short", "text": "hello"}]
    fewer = client.post(
        "/api/v1/load?wait=true", json=body | {"documents": docs, "on_duplicate": "replace"}
    ).json()
    assert fewer["status"] == "succeeded", fewer
    new_count = col.get("long")["embedding_meta"]["chunks"]
    assert 1 < new_count < chunks
    assert fewer["result"]["stale_chunks_removed"] == chunks - new_count
    assert col.has(f"long_embedding_chunk_{new_count - 1}")
    assert not col.has(f"long_embedding_chunk_{new_count}")
    chunks = new_count

    # Re-embedding only missing vectors leaves the chunked original alone...
    reembed = "/api/v1/collections/overrun/embed?wait=true"
    job = client.post(reembed, json={"embeddings": [spec]}).json()
    assert job["status"] == "succeeded", job
    assert job["counts"]["embedded"] == 0 and job["result"]["chunked_unchanged"] == 1, job
    # ...until the user edits it, e.g. after splitting it into their own chunks.
    col.update({"_key": "long", "text": "a shorter version"})
    job = client.post(reembed, json={"embeddings": [spec]}).json()
    assert job["counts"]["embedded"] == 1 and "chunked_unchanged" not in job["result"], job
    edited = col.get("long")
    assert len(edited["embedding"]) == 384 and edited["embedding_meta"]["status"] == "embedded"
    assert col.has("long_embedding_chunk_0")  # removing old chunks is up to the user

    report = client.post(
        "/api/v1/load?wait=true", json=body | {"collection": "overrun_scan", "analyze": True}
    ).json()
    assert report["status"] == "succeeded", report
    field = report["result"]["analysis"]["fields"]["embedding"]
    assert field["over_limit"] == 1 and field["longest"][0]["document"] == "long"
    assert not db.has_collection("overrun_scan")

    # The long row comes first here, but the pre-scan means nothing is written either way.
    spec["long_input"] = "error"
    job = client.post("/api/v1/load?wait=true", json=body | {"collection": "overrun_err"}).json()
    assert job["status"] == "failed" and "long" in job["message"], job
    assert job["result"]["prescan"] == {"status": "failed"}
    assert job["counts"]["embedded"] == 0

    # truncate_end: one vector from the first tokens, truncation recorded.
    spec["long_input"] = "truncate_end"
    job = client.post("/api/v1/load?wait=true", json=body | {"collection": "overrun_trunc"}).json()
    assert job["status"] == "succeeded" and job["counts"]["over_limit"] == 1, job
    meta = db.collection("overrun_trunc").get("long")["embedding_meta"]
    assert meta["long_input"] == "truncate_end" and meta["truncated"] is True
    # At least the overflow is dropped (a little more when the cut moves back to a word start).
    assert meta["tokens_dropped"] >= meta["input_tokens"] - meta["max_tokens"] > 0


@pytest.mark.local
def test_reembed_with_chunks_in_key_order(client, db):
    """Chunk keys can sort after other documents ("12_embedding_chunk_0" > "120"): a full
    re-embed must not read back the chunks it just wrote, and `max_documents` with
    `only_missing` must reach documents past the chunked ones."""
    pytest.importorskip("sentence_transformers")
    spec = {
        "source_field": "text",
        "target_field": "embedding",
        "provider": "sentence_transformers",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    long_text = " ".join(f"token{i}" for i in range(600))
    body = {
        "collection": "keyorder",
        "key_field": "id",
        "embeddings": [spec],
        "documents": [{"id": "12", "text": long_text}, {"id": "120", "text": "short"}],
    }
    job = client.post("/api/v1/load?wait=true", json=body).json()
    assert job["status"] == "succeeded", job
    chunks = job["counts"]["chunks"]
    assert chunks >= 2

    reembed = "/api/v1/collections/keyorder/embed?wait=true"
    full = client.post(
        reembed, json={"embeddings": [spec], "only_missing": False, "batch_size": 1}
    ).json()
    assert full["status"] == "succeeded", full
    assert full["counts"]["read"] == 2 and full["counts"]["embedded"] == 2, full
    assert full["counts"]["chunks"] == chunks and full["total"] == 2
    col = db.collection("keyorder")
    assert col.get("12_embedding_chunk_0")["embedding_meta"]["parent_key"] == "12"

    # A re-load that is ignored for the stored original leaves its chunks alone, even when
    # the new text would split into fewer chunks.
    last = f"12_embedding_chunk_{chunks - 1}"
    before = col.get("12_embedding_chunk_0")["text"]
    shorter = " ".join(f"token{i}" for i in range(150))
    ignored = client.post(
        "/api/v1/load?wait=true",
        json=body | {"documents": [{"id": "12", "text": shorter}], "on_duplicate": "ignore"},
    ).json()
    assert ignored["status"] == "succeeded" and ignored["counts"]["chunks"] == 0, ignored
    assert "stale_chunks_removed" not in ignored["result"]
    assert col.get("12_embedding_chunk_0")["text"] == before and col.has(last)

    col.insert_many([{"_key": "13", "text": "new one"}, {"_key": "14", "text": "new two"}])
    first = client.post(
        reembed, json={"embeddings": [spec], "max_documents": 1, "batch_size": 1}
    ).json()
    assert first["status"] == "succeeded" and first["counts"]["read"] == 1, first
    assert col.get("13")["embedding"] and not col.get("14").get("embedding")


def test_full_reembed_rewrites_chunks_of_an_edited_original(client, db):
    """The original and its old chunks come back in one batch: the old chunks must not be
    embedded from their old text and written back over the new chunks."""
    from app.providers.fake import FakeProvider

    spec = FAKE | {
        "long_input": "chunk_docs",
        "max_tokens": 10,
        "chunk_overlap_tokens": 0,
        "token_chars": 100,
    }
    body = {
        "collection": "edited",
        "embeddings": [spec],
        "documents": [{"_key": "p", "text": " ".join(f"old{i}" for i in range(40))}],
    }
    job = client.post("/api/v1/load?wait=true", json=body).json()
    assert job["status"] == "succeeded" and job["counts"]["chunks"] == 4, job

    col = db.collection("edited")
    col.update({"_key": "p", "text": " ".join(f"new{i}" for i in range(25))})
    full = client.post(
        "/api/v1/collections/edited/embed?wait=true",
        json={"embeddings": [spec], "only_missing": False, "batch_size": 100},
    ).json()
    assert full["status"] == "succeeded" and not full.get("errors"), full
    assert full["result"]["stale_chunks_removed"] == 1
    assert full["counts"]["read"] == 1 and full["counts"]["chunks"] == 3, full

    keys = sorted(d["_key"] for d in col.all())
    assert keys == ["p", "p_embedding_chunk_0", "p_embedding_chunk_1", "p_embedding_chunk_2"]
    for n in range(3):
        chunk = col.get(f"p_embedding_chunk_{n}")
        assert chunk["text"].startswith("new"), chunk["text"]
        assert chunk["embedding_meta"]["chunk_count"] == 3
        assert chunk["embedding"] == pytest.approx(FakeProvider._vector(chunk["text"], 384, ""))


def test_watcher_marks_long_text_and_carries_on(client, db):
    """long_input: error in a watcher: the long document fails alone; the rest are embedded."""
    profile = {
        "embeddings": [FAKE | {"long_input": "error", "max_tokens": 5}],
        "auto_watch": True,
        "watch_interval_s": 1,
    }
    assert client.put("/api/v1/collections/watched/profile", json=profile).status_code == 200
    col = db.collection("watched")
    col.insert_many(
        [
            {"_key": "long", "text": " ".join(f"w{i}" for i in range(50))},
            {"_key": "short", "text": "hello"},
        ]
    )
    for _ in range(40):
        if col.get("short").get("embedding") and col.get("long").get("embedding_meta"):
            break
        time.sleep(0.5)
    long_meta = col.get("long")["embedding_meta"]
    assert long_meta["status"] == "error" and long_meta["reason"] == "too long"
    assert len(col.get("short")["embedding"]) == 384

    # Later documents are not held up behind the failed one.
    col.insert({"_key": "later", "text": "inserted after"})
    for _ in range(40):
        if col.get("later").get("embedding"):
            break
        time.sleep(0.5)
    assert len(col.get("later")["embedding"]) == 384
    status = client.get("/api/v1/collections/watched/profile").json()
    assert status["pending_documents"] == 0
    assert status["watcher"]["running"] and status["watcher"]["failed"] >= 1
    assert client.delete("/api/v1/collections/watched/profile").status_code == 204


def test_platform_token_from_gateway(platform_client, test_db, db):
    """Signed in with the forwarded JWT only: the server's own password is wrong."""
    info = platform_client.get("/api/v1/connection").json()
    assert info["connected"], info
    assert info["database"] == test_db and info["user"] == "root"
    assert info["connection"]["auth"] == "platform"
    assert test_db in platform_client.get("/api/v1/databases").json()

    body = {
        "collection": "gateway_docs",
        "key_field": "id",
        "embeddings": [FAKE],
        "documents": [{"id": "a", "text": "hello"}],
    }
    job = platform_client.post("/api/v1/load?wait=true", json=body).json()
    assert job["status"] == "succeeded", job
    assert len(db.collection("gateway_docs").get("a")["embedding"]) == 384


def test_bad_token_rejected(platform_client):
    r = platform_client.get(
        "/api/v1/collections", headers={"Authorization": "bearer not-a-real-token"}
    )
    assert r.status_code == 401


def test_platform_user_without_system_access(arango_settings, test_db, monkeypatch):
    """A typical platform user: access to one database only, signed in by the gateway."""
    import uuid

    import httpx
    from arango import ArangoClient
    from fastapi.testclient import TestClient
    from pydantic import SecretStr

    from app.config import get_settings
    from app.main import create_app

    endpoint, root, password = arango_settings
    sys_db = ArangoClient(hosts=endpoint).db("_system", username=root, password=password)
    user = f"ael_user_{uuid.uuid4().hex[:6]}"
    sys_db.create_user(user, "pw")
    try:
        sys_db.update_permission(user, "rw", test_db)
        jwt = httpx.post(
            f"{endpoint}/_open/auth", json={"username": user, "password": "pw"}
        ).json()["jwt"]
        settings = get_settings()
        monkeypatch.setattr(settings, "arango_endpoint", endpoint)
        monkeypatch.setattr(settings, "arango_password", SecretStr("wrong"))
        with TestClient(create_app(), headers={"Authorization": f"bearer {jwt}"}) as c:
            assert c.get("/api/v1/databases").json() == [test_db]
            assert not c.get("/api/v1/connection").json()["connected"]  # _system
            info = c.get("/api/v1/connection", headers={"X-Arango-Database": test_db}).json()
            assert info["connected"] and info["user"] == user
    finally:
        sys_db.delete_user(user, ignore_missing=True)
