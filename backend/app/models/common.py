from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Machine readable error code.", examples=["bad_request"])
    detail: Any = Field(None, description="Human readable explanation or validation details.")


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid request or provider configuration."},
    401: {"model": ErrorResponse, "description": "ArangoDB rejected the credentials."},
    403: {"model": ErrorResponse, "description": "Connection override not permitted."},
    404: {"model": ErrorResponse, "description": "Resource not found."},
    502: {"model": ErrorResponse, "description": "ArangoDB or embedding provider failure."},
}
