from fastapi.testclient import TestClient

from app.main import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_health_and_openapi():
    c = client()
    assert c.get("/api/v1/health").json()["status"] == "ok"
    spec = c.get("/openapi.json").json()
    paths = spec["paths"]
    for path in (
        "/api/v1/load",
        "/api/v1/load/file",
        "/api/v1/load/url",
        "/api/v1/preview",
        "/api/v1/collections/{name}/profile",
        "/api/v1/collections/{name}/embed",
        "/api/v1/providers",
        "/api/v1/jobs/{job_id}/events",
    ):
        assert path in paths, path
    params = {p["name"] for p in paths["/api/v1/collections"]["get"]["parameters"]}
    assert {"x-arango-endpoint", "x-arango-password"} <= {p.lower() for p in params}


def test_providers_and_presets():
    c = client()
    providers = {p["name"]: p for p in c.get("/api/v1/providers").json()}
    assert providers["fake"]["available"]
    assert "options_schema" in providers["openai"]
    presets = {p["name"] for p in c.get("/api/v1/presets").json()}
    assert "pubmedbert" in presets


def test_test_embed_fake():
    c = client()
    r = c.post("/api/v1/providers/fake/embed", json={"texts": ["a"], "model": "fake-768"})
    assert r.status_code == 200 and r.json()["dimension"] == 768


def test_preview_upload():
    c = client()
    r = c.post(
        "/api/v1/preview",
        files={"file": ("p.jsonl", b'{"id":1,"text":"some abstract text here ok"}\n')},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["format"] == "jsonl" and body["estimated_total"] == 1


def test_override_forbidden(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "allow_connection_override", False)
    r = client().get("/api/v1/collections", headers={"X-Arango-Database": "other"})
    assert r.status_code == 403
    assert r.json()["error"] == "override_disabled"
