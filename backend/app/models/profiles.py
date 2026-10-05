from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.embedding import EmbeddingFieldSpec, VectorIndexSpec
from app.models.load import OnDuplicate


class ProfileConfig(BaseModel):
    """Auto-embedding configuration attached to a collection."""

    embeddings: list[EmbeddingFieldSpec] = Field(..., min_length=1)
    key_field: str | None = Field(None, examples=["id"])
    on_duplicate: OnDuplicate = "update"
    batch_size: int = Field(64, ge=1, le=10000)
    vector_index: VectorIndexSpec | None = None
    auto_watch: bool = Field(
        False,
        description=(
            "Run a background watcher that embeds documents inserted into the collection by any "
            "client (arangoimport, drivers, web UI)."
        ),
    )
    watch_interval_s: float = Field(10.0, ge=1, le=3600)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "embeddings": [
                        {
                            "source_field": "text",
                            "target_field": "embedding",
                            "preset": "pubmedbert",
                        }
                    ],
                    "key_field": "id",
                    "on_duplicate": "update",
                    "vector_index": {"create": True, "metric": "cosine"},
                    "auto_watch": True,
                }
            ]
        }
    }


class Profile(ProfileConfig):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    collection: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WatcherState(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    running: bool = False
    last_run_at: datetime | None = None
    last_error: str | None = None
    embedded: int = 0
    failed: int = 0
    skipped: int = 0


class ProfileStatus(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    profile: Profile
    total_documents: int
    embedded_documents: dict[str, int] = Field(
        ..., description="Per target field: documents that have a vector."
    )
    pending_documents: int = Field(..., description="Documents not yet processed by any spec.")
    watcher: WatcherState
    indexes: list[dict[str, Any]] = Field(default_factory=list)
