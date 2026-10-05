"""ArangoDB connection handling.

A connection is described by :class:`ArangoConnection`. The server's environment provides the
default; individual requests may override any part of it with ``X-Arango-*`` headers.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from arango import ArangoClient
from arango.database import StandardDatabase

from app.config import Settings


@dataclass(frozen=True)
class ArangoConnection:
    endpoint: str
    database: str
    username: str
    password: str
    verify_ssl: bool = True
    timeout: int = 120

    @classmethod
    def from_settings(cls, settings: Settings) -> ArangoConnection:
        return cls(
            endpoint=settings.arango_endpoint,
            database=settings.arango_database,
            username=settings.arango_username,
            password=settings.arango_password.get_secret_value(),
            verify_ssl=settings.arango_verify_ssl,
            timeout=settings.arango_request_timeout,
        )

    def with_overrides(
        self,
        endpoint: str | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> ArangoConnection:
        return ArangoConnection(
            endpoint=endpoint or self.endpoint,
            database=database or self.database,
            username=username or self.username,
            password=password if password is not None else self.password,
            verify_ssl=self.verify_ssl,
            timeout=self.timeout,
        )

    @property
    def cache_key(self) -> str:
        raw = f"{self.endpoint}|{self.database}|{self.username}|{self.password}|{self.verify_ssl}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def masked(self) -> dict[str, str | bool]:
        return {
            "endpoint": self.endpoint,
            "database": self.database,
            "username": self.username,
            "password": "********" if self.password else "",
            "verify_ssl": self.verify_ssl,
        }


_lock = threading.Lock()
_clients: dict[str, ArangoClient] = {}
_dbs: dict[str, StandardDatabase] = {}


def get_database(conn: ArangoConnection) -> StandardDatabase:
    """Return a (cached) database handle for ``conn``."""
    key = conn.cache_key
    with _lock:
        db = _dbs.get(key)
        if db is not None:
            return db
        client = _clients.get(conn.endpoint)
        if client is None:
            client = ArangoClient(
                hosts=conn.endpoint,
                verify_override=conn.verify_ssl,
                request_timeout=conn.timeout,
            )
            _clients[conn.endpoint] = client
        db = client.db(conn.database, username=conn.username, password=conn.password)
        _dbs[key] = db
        return db


def get_system_database(conn: ArangoConnection) -> StandardDatabase:
    return get_database(conn.with_overrides(database="_system"))
