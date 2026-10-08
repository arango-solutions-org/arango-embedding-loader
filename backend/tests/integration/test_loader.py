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
