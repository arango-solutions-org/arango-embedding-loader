"""FastAPI dependencies: resolving the ArangoDB connection for a request."""

from __future__ import annotations

from typing import Annotated

from arango.database import StandardDatabase
from fastapi import Depends, Header

from app.arango import ArangoConnection, get_database
from app.config import Settings, get_settings
from app.errors import AppError


def get_connection(
    settings: Annotated[Settings, Depends(get_settings)],
    x_arango_endpoint: Annotated[
        str | None,
        Header(
            description="Override the ArangoDB endpoint, e.g. `https://my-db.arangodb.cloud:8529`."
        ),
    ] = None,
    x_arango_database: Annotated[
        str | None, Header(description="Override the ArangoDB database name.")
    ] = None,
    x_arango_username: Annotated[
        str | None, Header(description="Override the ArangoDB username.")
    ] = None,
    x_arango_password: Annotated[
        str | None, Header(description="Override the ArangoDB password.")
    ] = None,
) -> ArangoConnection:
    base = ArangoConnection.from_settings(settings)
    overrides = (x_arango_endpoint, x_arango_database, x_arango_username, x_arango_password)
    if any(v is not None for v in overrides):
        if not settings.allow_connection_override:
            raise AppError(
                403, "override_disabled", "Connection overrides are disabled on this server."
            )
        return base.with_overrides(*overrides)
    return base


ConnectionDep = Annotated[ArangoConnection, Depends(get_connection)]


def get_db(conn: ConnectionDep) -> StandardDatabase:
    return get_database(conn)


DbDep = Annotated[StandardDatabase, Depends(get_db)]
