from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ingest.fieldpaths import FieldPathInfo


class CollectionInfo(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    name: str
    type: str
    count: int | None = None
    has_profile: bool = False


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., examples=["pubmed"])
    shards: int | None = Field(None, ge=1, description="Cluster only: number of shards.")
    replication_factor: int | None = Field(None, ge=1, description="Cluster only.")


class SampleResponse(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    collection: str
    count: int
    sample: list[dict[str, Any]]
    fields: list[FieldPathInfo]


class VectorIndexRequest(BaseModel):
    field: str = Field(..., examples=["embedding"])
    metric: str = Field("cosine", description="cosine | l2 | innerProduct")
    dimension: int | None = Field(
        None, description="Vector dimension; detected from a document when omitted."
    )
    n_lists: int | None = Field(None, ge=1)
    default_n_probe: int | None = Field(None, ge=1)
    training_iterations: int | None = Field(None, ge=1)
    name: str | None = None
    parallelism: int | None = Field(None, ge=1)


class ConnectionInfo(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    connected: bool
    server_version: str | None = None
    database: str
    connection: dict[str, Any]
    vector_index_supported: bool | None = Field(
        None, description="Whether the server version supports vector indexes (3.12.4+)."
    )
    error: str | None = None
