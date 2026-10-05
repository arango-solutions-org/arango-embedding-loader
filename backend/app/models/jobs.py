from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
JobKind = Literal["load_documents", "load_file", "load_url", "reembed"]


class DocumentError(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    index: int | None = Field(None, description="Position of the document in the input.")
    key: str | None = None
    error: str


class JobCounts(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    read: int = 0
    embedded: int = 0
    written: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0


class Job(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    kind: JobKind
    status: JobStatus = "queued"
    collection: str
    database: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total: int | None = Field(None, description="Total documents when known.")
    bytes_total: int | None = None
    bytes_read: int | None = None
    progress: float | None = Field(None, description="0..1 when it can be estimated.")
    counts: JobCounts = Field(default_factory=JobCounts)
    errors: list[DocumentError] = Field(default_factory=list, description="First 100 errors.")
    message: str | None = None
    config: dict[str, Any] = Field(default_factory=dict, description="Sanitized job config.")
    result: dict[str, Any] = Field(
        default_factory=dict, description="Extra results: vector index info, dry-run samples."
    )
