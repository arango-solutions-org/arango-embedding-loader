import base64
import json
import time

import httpx
import pytest

from app import arango as arango_mod
from app import auth
from app.arango import background, request_connection
from app.auth import PasswordToken, SidecarToken, StaticToken, bearer, claims, expires_at
from app.config import Settings

SIDECAR = "http://127.0.0.1:9203"


def jwt(**payload) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{enc({'alg': 'HS256'})}.{enc(payload)}.sig"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(arango_mod, "_sources", {})
    monkeypatch.setattr(auth, "_identities", {})


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_bearer():
    assert bearer("bearer abc") == "abc"
    assert bearer("Bearer abc") == "abc"
    assert bearer("Basic abc") is None
    assert bearer("bearer ") is None
    assert bearer(None) is None


def test_claims_and_expiry():
    token = jwt(preferred_username="alice", exp=123)
    assert claims(token)["preferred_username"] == "alice"
    assert expires_at(token) == 123
    assert claims("not-a-jwt") == {} and expires_at("x.y.z") is None


def test_static_token_key_hides_token():
    t = StaticToken("secret-token", "alice")
    assert "secret-token" not in t.key and t.token() == t.refresh() == "secret-token"


def test_sidecar_token_mints_for_user_and_renews(monkeypatch):
    calls = []

    def post(url, json, timeout):
        calls.append((url, json))
        # First token is about to expire, so the next call renews it.
        exp = time.time() + (60 if len(calls) == 1 else 3600)
        return httpx.Response(200, json={"token": jwt(exp=exp)}, request=httpx.Request("POST", url))

    monkeypatch.setattr(auth.httpx, "post", post)
    source = SidecarToken(sidecar=SIDECAR, user="alice", lifetime_s=3600)
    first = source.token()
    second = source.token()
    third = source.token()
    assert first != second and second == third
    assert calls[0] == (
        f"{SIDECAR}/_integration/authn/v1/createToken",
        {"lifetime": "3600s", "user": "alice"},
    )
    assert len(calls) == 2


def test_password_token_rejected(monkeypatch):
    def post(url, json, verify, timeout):
        return httpx.Response(401, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(auth.httpx, "post", post)
    with pytest.raises(auth.AppError) as err:
        PasswordToken(endpoint="http://db:8529", user="root", password="bad").token()
    assert err.value.status_code == 401


def test_request_connection_uses_forwarded_token(monkeypatch):
    def get(url, headers, timeout):
        assert headers == {"Authorization": "bearer tok"}
        return httpx.Response(200, json={"user": "alice"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(auth.httpx, "get", get)
    conn = request_connection(settings(integration_http_address=SIDECAR), "tok")
    assert conn is not None
    assert isinstance(conn.credentials, StaticToken)
    assert conn.user == "alice" and conn.describe()["auth"] == "platform"


def test_request_connection_without_token():
    assert request_connection(settings(integration_http_address=SIDECAR), None) is None
    off_platform = request_connection(settings(), None)
    assert off_platform is not None and isinstance(off_platform.credentials, PasswordToken)


def test_background_mints_for_the_same_user():
    s = settings(integration_http_address=SIDECAR)
    conn = arango_mod.default_connection(s).with_credentials(StaticToken("tok", "alice"))
    bg = background(conn, s)
    assert isinstance(bg.credentials, SidecarToken) and bg.credentials.user == "alice"
    assert background(conn, s).credentials is bg.credentials  # shared, minted once


def test_background_never_escalates_unknown_user():
    s = settings(integration_http_address=SIDECAR)
    conn = arango_mod.default_connection(s).with_credentials(StaticToken("tok", None))
    assert background(conn, s).credentials == conn.credentials


def test_default_connection_on_platform_uses_service_user():
    s = settings(integration_http_address="127.0.0.1:9203", arango_service_user="loader")
    creds = arango_mod.default_connection(s).credentials
    assert isinstance(creds, SidecarToken)
    assert creds.user == "loader" and creds.sidecar == SIDECAR
