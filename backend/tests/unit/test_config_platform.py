from app.config import Settings


def test_platform_endpoint_used_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("ARANGO_ENDPOINT", raising=False)
    monkeypatch.delenv("ARANGO_VERIFY_SSL", raising=False)
    monkeypatch.delenv("ARANGO_DEPLOYMENT_CA", raising=False)
    monkeypatch.setenv("ARANGO_DEPLOYMENT_ENDPOINT", "https://arango-coordinator.ns.svc:8529/")
    s = Settings(_env_file=None)
    assert s.arango_endpoint == "https://arango-coordinator.ns.svc:8529"
    assert s.arango_verify_ssl is False


def test_platform_ca_used_for_verification(monkeypatch, tmp_path):
    ca = tmp_path / "ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.delenv("ARANGO_ENDPOINT", raising=False)
    monkeypatch.delenv("ARANGO_VERIFY_SSL", raising=False)
    monkeypatch.setenv("ARANGO_DEPLOYMENT_ENDPOINT", "https://arango-coordinator.ns.svc:8529")
    monkeypatch.setenv("ARANGO_DEPLOYMENT_CA", str(ca))
    s = Settings(_env_file=None)
    assert s.arango_verify_ssl == str(ca)


def test_explicit_endpoint_wins(monkeypatch):
    monkeypatch.setenv("ARANGO_DEPLOYMENT_ENDPOINT", "https://injected:8529")
    monkeypatch.setenv("ARANGO_ENDPOINT", "https://explicit:8529")
    monkeypatch.delenv("ARANGO_VERIFY_SSL", raising=False)
    s = Settings(_env_file=None)
    assert s.arango_endpoint == "https://explicit:8529"
    assert s.arango_verify_ssl is True


def test_verify_ssl_parses_booleans(monkeypatch):
    monkeypatch.setenv("ARANGO_VERIFY_SSL", "false")
    assert Settings(_env_file=None).arango_verify_ssl is False
    monkeypatch.setenv("ARANGO_VERIFY_SSL", "/etc/ca.pem")
    assert Settings(_env_file=None).arango_verify_ssl == "/etc/ca.pem"


def test_sidecar_address_from_injected_env(monkeypatch):
    monkeypatch.setenv("INTEGRATION_HTTP_ADDRESS_FULL", "http://127.0.0.1:9203")
    s = Settings(_env_file=None)
    assert s.on_platform and s.integration_http_address == "http://127.0.0.1:9203"
