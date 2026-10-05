from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__

router = APIRouter(tags=["System"])


class Health(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=Health, summary="Liveness check")
async def health() -> Health:
    return Health(status="ok", version=__version__)
