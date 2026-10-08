"""ArangoDB connection handling.

A connection is described by :class:`ArangoConnection`: the server's endpoint, a database and a
:class:`~app.auth.TokenSource` that supplies the JWT (see :mod:`app.auth`).
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, replace
from typing import Any

from arango import ArangoClient
from arango.connection import JwtConnection
from arango.database import StandardDatabase
from arango.request import Request
from arango.response import Response

from app.auth import PasswordToken, SidecarToken, StaticToken, TokenSource, expires_at, identify
from app.config import Settings


@dataclass(frozen=True)
class ArangoConnection:
    endpoint: str
    database: str
    credentials: TokenSource
    verify_ssl: bool | str = True
    timeout: int = 120

    @property
    def user(self) -> str | None:
        return self.credentials.user

    def with_database(self, database: str | None) -> ArangoConnection:
        return replace(self, database=database) if database else self

    def with_credentials(self, credentials: TokenSource) -> ArangoConnection:
        return replace(self, credentials=credentials)

    @property
    def cache_key(self) -> str:
        return f"{self.endpoint}|{self.database}|{self.credentials.key}|{self.verify_ssl}"

    def describe(self) -> dict[str, str | bool | None]:
        return {
            "endpoint": self.endpoint,
            "database": self.database,
            "user": self.user,
            "auth": _auth_kind(self.credentials),
        }


def _auth_kind(credentials: TokenSource) -> str:
    if isinstance(credentials, StaticToken):
        return "platform"
    if isinstance(credentials, SidecarToken):
        return "service"
    return "password"


def default_connection(settings: Settings) -> ArangoConnection:
    """The server's own connection: the configured username/password off the platform, or
    ``ARANGO_SERVICE_USER`` via the sidecar on it. Used when no request is involved."""
    credentials: TokenSource
    if settings.integration_http_address:
        credentials = _shared(
            f"sidecar:{settings.arango_service_user}",
            lambda: SidecarToken(
                sidecar=settings.integration_http_address or "",
                user=settings.arango_service_user,
                lifetime_s=settings.token_lifetime_s,
            ),
        )
    else:
        password = settings.arango_password.get_secret_value()
        credentials = _shared(
            f"password:{settings.arango_endpoint}|{settings.arango_username}|{password}",
            lambda: PasswordToken(
                endpoint=settings.arango_endpoint,
                user=settings.arango_username,
                password=password,
                verify=settings.arango_verify_ssl,
            ),
        )
    return ArangoConnection(
        endpoint=settings.arango_endpoint,
        database=settings.arango_database,
        credentials=credentials,
        verify_ssl=settings.arango_verify_ssl,
        timeout=settings.arango_request_timeout,
    )


def request_connection(settings: Settings, token: str | None) -> ArangoConnection | None:
    """The connection for a request carrying ``token`` (forwarded by the platform gateway).
    Without a token, the server's own connection off the platform and ``None`` on it."""
    if token:
        user = identify(token, settings.integration_http_address)
        base = default_connection(settings)
        return base.with_credentials(StaticToken(token, user))
    if settings.on_platform:
        return None
    return default_connection(settings)


def background(conn: ArangoConnection, settings: Settings) -> ArangoConnection:
    """The connection for work that outlives the request (load jobs, watchers). A forwarded
    token expires, so on the platform the sidecar mints renewable tokens for the same user.
    If the user is unknown the request's token is kept: the sidecar would default to root."""
    if (
        isinstance(conn.credentials, StaticToken)
        and conn.credentials.user
        and settings.integration_http_address
    ):
        user = conn.credentials.user
        sidecar = settings.integration_http_address
        credentials = _shared(
            f"sidecar:{user}",
            lambda: SidecarToken(sidecar=sidecar, user=user, lifetime_s=settings.token_lifetime_s),
        )
        return conn.with_credentials(credentials)
    return conn


_lock = threading.Lock()
_sources: dict[str, TokenSource] = {}
_clients: dict[str, ArangoClient] = {}
_dbs: dict[str, StandardDatabase] = {}


def _shared(key: str, make: Any) -> TokenSource:
    """One renewing token source per identity, so tokens are minted once and shared."""
    with _lock:
        source = _sources.get(key)
        if source is None:
            source = _sources[key] = make()
        return source


class _TokenConnection(JwtConnection):
    """A JWT connection that gets its tokens from a :class:`TokenSource`: before each request
    (so renewed tokens are picked up) and again when the server rejects one."""

    def __init__(self, *, source: TokenSource, **kwargs: Any) -> None:
        self._source = source
        super().__init__(user_token=source.token(), **kwargs)

    def set_token(self, token: str) -> None:
        # The source checks expiry; python-arango's own check also rejects tokens it can't
        # parse (e.g. without ``iat``) that ArangoDB accepts.
        self._token = token
        self._token_exp = int(expires_at(token) or sys.maxsize)
        self._auth_header = f"bearer {token}"

    def refresh_token(self) -> None:
        self.set_token(self._source.refresh())

    def send_request(self, request: Request) -> Response:
        token = self._source.token()
        if token != self._token:
            self.set_token(token)
        return super().send_request(request)


def get_database(conn: ArangoConnection) -> StandardDatabase:
    """Return a (cached) database handle for ``conn``."""
    key = conn.cache_key
    with _lock:
        db = _dbs.get(key)
        if db is not None:
            return db
        client = _clients.get(f"{conn.endpoint}|{conn.verify_ssl}")
        if client is None:
            client = ArangoClient(
                hosts=conn.endpoint,
                verify_override=conn.verify_ssl,
                request_timeout=conn.timeout,
            )
            _clients[f"{conn.endpoint}|{conn.verify_ssl}"] = client
        connection = _TokenConnection(
            source=conn.credentials,
            hosts=client._hosts,
            host_resolver=client._host_resolver,
            sessions=client._sessions,
            db_name=conn.database,
            http_client=client._http,
            serializer=client._serializer,
            deserializer=client._deserializer,
            request_compression=client._request_compression,
            response_compression=client._response_compression,
        )
        db = StandardDatabase(connection)
        if isinstance(conn.credentials, StaticToken):
            # A forwarded token is used for one request; caching it would only leak memory.
            return db
        _dbs[key] = db
        return db


def get_system_database(conn: ArangoConnection) -> StandardDatabase:
    return get_database(conn.with_database("_system"))
