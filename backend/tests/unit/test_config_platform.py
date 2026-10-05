from app.config import Settings


def test_platform_endpoint_used_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("ARANGO_ENDPOINT", raising=False)
    monkeypatch.delenv("ARANGO_VERIFY_SSL", raising=False)
    monkeypatch.setenv("ARANGO_DEPLOYMENT_ENDPOINT", "https://arango-coordinator.ns.svc:8529/")
    s = Settings(_env_file=None)
    assert s.arango_endpoint == "https://arango-coordinator.ns.svc:8529"
    assert s.arango_verify_ssl is False


def test_explicit_endpoint_wins(monkeypatch):
    monkeypatch.setenv("ARANGO_DEPLOYMENT_ENDPOINT", "https://injected:8529")
    monkeypatch.setenv("ARANGO_ENDPOINT", "https://explicit:8529")
    s = Settings(_env_file=None)
    assert s.arango_endpoint == "https://explicit:8529"
    assert s.arango_verify_ssl is True
