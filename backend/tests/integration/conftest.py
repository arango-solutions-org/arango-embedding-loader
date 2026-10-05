"""Integration fixtures: a throw-away database on the ArangoDB from docker-compose.

Run with:  docker compose up -d arangodb && uv run pytest -m integration
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from arango import ArangoClient
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture(scope="session")
def arango_settings():
    settings = get_settings()
    endpoint = os.getenv("ARANGO_TEST_ENDPOINT", settings.arango_endpoint)
    password = os.getenv("ARANGO_TEST_PASSWORD", settings.arango_password.get_secret_value())
    return endpoint, settings.arango_username, password


@pytest.fixture(scope="session")
def test_db(arango_settings) -> Iterator[str]:
    endpoint, user, password = arango_settings
    sys_db = ArangoClient(hosts=endpoint).db("_system", username=user, password=password)
    try:
        sys_db.version()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ArangoDB not reachable at {endpoint}: {exc}")
    name = f"ael_test_{uuid.uuid4().hex[:8]}"
    sys_db.create_database(name)
    yield name
    sys_db.delete_database(name, ignore_missing=True)


@pytest.fixture
def db(arango_settings, test_db):
    endpoint, user, password = arango_settings
    return ArangoClient(hosts=endpoint).db(test_db, username=user, password=password)


@pytest.fixture
def client(arango_settings, test_db) -> Iterator[TestClient]:
    endpoint, user, password = arango_settings
    headers = {
        "X-Arango-Endpoint": endpoint,
        "X-Arango-Database": test_db,
        "X-Arango-Username": user,
        "X-Arango-Password": password,
    }
    with TestClient(create_app(), headers=headers) as c:
        yield c
