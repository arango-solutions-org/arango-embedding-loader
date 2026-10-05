from __future__ import annotations

import asyncio

from arango.exceptions import ArangoError
from fastapi import APIRouter

from app.arango import get_database
from app.deps import ConnectionDep
from app.errors import describe_arango_error
from app.models.collections import ConnectionInfo
from app.models.common import ERROR_RESPONSES

router = APIRouter(tags=["Connection"], responses=ERROR_RESPONSES)


def _supports_vector(version: str | None) -> bool | None:
    if not version:
        return None
    try:
        parts = [int(p) for p in version.split("-")[0].split(".")[:3]]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts) >= (3, 12, 4)


@router.get(
    "/connection",
    response_model=ConnectionInfo,
    summary="Test the ArangoDB connection",
    description=(
        "Connects with the server defaults or the `X-Arango-*` header overrides and reports the "
        "server version. Never returns the password."
    ),
)
async def test_connection(conn: ConnectionDep) -> ConnectionInfo:
    db = get_database(conn)
    try:
        version = await asyncio.to_thread(db.version)
        return ConnectionInfo(
            connected=True,
            server_version=str(version),
            database=conn.database,
            connection=conn.masked(),
            vector_index_supported=_supports_vector(str(version)),
        )
    except ArangoError as exc:
        return ConnectionInfo(
            connected=False,
            database=conn.database,
            connection=conn.masked(),
            error=describe_arango_error(exc),
        )
    except Exception as exc:  # noqa: BLE001 - connection refused etc.
        return ConnectionInfo(
            connected=False, database=conn.database, connection=conn.masked(), error=str(exc)
        )
