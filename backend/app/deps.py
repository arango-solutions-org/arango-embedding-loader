"""FastAPI dependencies: resolving the ArangoDB connection for a request."""

from __future__ import annotations

from typing import Annotated

from arango.database import StandardDatabase
from fastapi import Depends, Header

from app.arango import ArangoConnection, background, get_database, request_connection
from app.auth import bearer
from app.config import Settings, get_settings
from app.errors import AppError


def get_connection(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[
        str | None,
        Header(
            include_in_schema=False,
            description="Set by the Arango Platform gateway for the signed-in user.",
        ),
    ] = None,
    x_arango_database: Annotated[
        str | None, Header(description="The ArangoDB database to use (default: the server's).")
    ] = None,
) -> ArangoConnection:
    conn = request_connection(settings, bearer(authorization))
    if conn is None:
        raise AppError(
            401,
            "unauthenticated",
            "No ArangoDB credentials: open the app through the Arango Platform.",
        )
    return conn.with_database(x_arango_database)


ConnectionDep = Annotated[ArangoConnection, Depends(get_connection)]


def get_db(conn: ConnectionDep) -> StandardDatabase:
    return get_database(conn)


DbDep = Annotated[StandardDatabase, Depends(get_db)]


def get_background_db(
    conn: ConnectionDep, settings: Annotated[Settings, Depends(get_settings)]
) -> StandardDatabase:
    """A database handle for jobs that outlive the request (see :func:`app.arango.background`)."""
    return get_database(background(conn, settings))


BackgroundDbDep = Annotated[StandardDatabase, Depends(get_background_db)]
