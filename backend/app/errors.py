"""Application errors and their HTTP mapping."""

from __future__ import annotations

from typing import Any

from arango.exceptions import ArangoClientError, ArangoServerError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.ingest.parsers import ParseError
from app.providers.base import ProviderError


class AppError(Exception):
    def __init__(self, status_code: int, error: str, detail: Any = None):
        super().__init__(detail or error)
        self.status_code = status_code
        self.error = error
        self.detail = detail


def describe_arango_error(exc: BaseException) -> str:
    message = getattr(exc, "error_message", None) or str(exc)
    code = getattr(exc, "error_code", None)
    return f"{message} (error {code})" if code else message


def arango_status(exc: ArangoServerError) -> int:
    http = getattr(exc, "http_code", None) or 502
    if http in (400, 401, 403, 404, 409):
        return http
    return 502


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.error, "detail": exc.detail}
        )

    @app.exception_handler(ProviderError)
    async def _provider_error(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": "provider_error", "detail": str(exc)}
        )

    @app.exception_handler(ParseError)
    async def _parse_error(_: Request, exc: ParseError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "parse_error", "detail": str(exc)})

    @app.exception_handler(ArangoServerError)
    async def _arango_server(_: Request, exc: ArangoServerError) -> JSONResponse:
        return JSONResponse(
            status_code=arango_status(exc),
            content={"error": "arango_error", "detail": describe_arango_error(exc)},
        )

    @app.exception_handler(ArangoClientError)
    async def _arango_client(_: Request, exc: ArangoClientError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"error": "arango_unreachable", "detail": describe_arango_error(exc)},
        )
