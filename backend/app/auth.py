"""Where ArangoDB tokens come from. Every connection authenticates with a JWT.

- On the Arango Platform the gateway authenticates the user and forwards their token in the
  ``Authorization`` header (route ``passMode: pass``), so requests use it as is.
- Work that outlives a request (load jobs, watchers) asks the injected integration sidecar to
  mint a token for the same user, renewed before it expires. Watchers started at boot have no
  user and run as ``ARANGO_SERVICE_USER``.
- Off the platform the server's ``ARANGO_USERNAME`` / ``ARANGO_PASSWORD`` are exchanged for a
  token at ``/_open/auth``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.errors import AppError

# Renew a token this long before it expires.
REFRESH_MARGIN_S = 300


class TokenSource(Protocol):
    @property
    def key(self) -> str:
        """Identifies the credential (never the token itself) for caching connections."""
        ...

    @property
    def user(self) -> str | None: ...

    def token(self) -> str:
        """A valid token, renewed first if it is about to expire."""
        ...

    def refresh(self) -> str:
        """A new token, after the server rejected the current one."""
        ...


def claims(token: str) -> dict[str, Any]:
    """The token's payload, unverified: the gateway or ArangoDB has already checked it."""
    try:
        payload = token.split(".")[1]
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def expires_at(token: str) -> float | None:
    exp = claims(token).get("exp")
    return float(exp) if isinstance(exp, int | float) else None


def bearer(header: str | None) -> str | None:
    """The token from an ``Authorization: bearer <token>`` header."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


@dataclass(frozen=True)
class StaticToken:
    """A token supplied with the request. It cannot be renewed."""

    value: str
    user: str | None = None

    @property
    def key(self) -> str:
        return "token:" + hashlib.sha256(self.value.encode()).hexdigest()

    def token(self) -> str:
        return self.value

    def refresh(self) -> str:
        return self.value


@dataclass
class _Renewing:
    """Caches a minted token and mints a new one shortly before it expires."""

    _token: str | None = field(default=None, init=False, repr=False)
    _expires: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _mint(self) -> str:
        raise NotImplementedError

    def token(self) -> str:
        with self._lock:
            if self._token is None or time.time() > self._expires - REFRESH_MARGIN_S:
                self._renew()
            assert self._token is not None
            return self._token

    def refresh(self) -> str:
        with self._lock:
            self._renew()
            assert self._token is not None
            return self._token

    def _renew(self) -> None:
        token = self._mint()
        self._token = token
        self._expires = expires_at(token) or time.time() + REFRESH_MARGIN_S * 2


@dataclass
class SidecarToken(_Renewing):
    """Tokens for ``user`` minted by the platform's integration sidecar."""

    sidecar: str = ""
    user: str | None = None
    lifetime_s: int = 3600

    @property
    def key(self) -> str:
        return f"sidecar:{self.user}"

    def _mint(self) -> str:
        body: dict[str, Any] = {"lifetime": f"{self.lifetime_s}s"}
        if self.user:
            body["user"] = self.user
        try:
            r = httpx.post(
                f"{self.sidecar}/_integration/authn/v1/createToken", json=body, timeout=10
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise AppError(
                503, "token_unavailable", f"Could not get a token from the sidecar: {exc}"
            ) from exc
        return str(r.json()["token"])


@dataclass
class PasswordToken(_Renewing):
    """Tokens from ``/_open/auth`` for a username and password (off the platform)."""

    endpoint: str = ""
    user: str | None = None
    password: str = field(default="", repr=False)
    verify: bool | str = True

    @property
    def key(self) -> str:
        digest = hashlib.sha256(f"{self.user}|{self.password}".encode()).hexdigest()
        return f"password:{digest}"

    def _mint(self) -> str:
        try:
            r = httpx.post(
                f"{self.endpoint}/_open/auth",
                json={"username": self.user, "password": self.password},
                verify=self.verify,
                timeout=10,
            )
        except httpx.HTTPError as exc:
            raise AppError(502, "arango_unreachable", f"Could not reach ArangoDB: {exc}") from exc
        if r.status_code == 401:
            raise AppError(401, "arango_auth_failed", "ArangoDB rejected the credentials.")
        r.raise_for_status()
        return str(r.json()["jwt"])


_identities: dict[str, str] = {}


def identify(token: str, sidecar: str | None) -> str | None:
    """The user a token belongs to. Asks the sidecar, which validates the token; off the
    platform it falls back to the token's ``preferred_username`` claim."""
    key = hashlib.sha256(token.encode()).hexdigest()
    if key in _identities:
        return _identities[key]
    user: str | None = None
    if sidecar:
        try:
            r = httpx.get(
                f"{sidecar}/_integration/authn/v1/identity",
                headers={"Authorization": f"bearer {token}"},
                timeout=5,
            )
            if r.status_code == 200:
                user = r.json().get("user") or None
        except httpx.HTTPError:
            user = None
    if user is None:
        name = claims(token).get("preferred_username")
        user = name if isinstance(name, str) and name else None
    if user is not None:
        if len(_identities) > 1000:  # gateway tokens rotate; don't grow without bound
            _identities.clear()
        _identities[key] = user
    return user
