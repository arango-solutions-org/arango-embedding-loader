from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ingest.parsers import CsvOptions, FileFormat
from app.models.embedding import EmbeddingFieldSpec, VectorIndexSpec

OnDuplicate = Literal["error", "update", "replace", "ignore"]

ANALYZE_DESCRIPTION = (
    "Count tokens only: report how many texts are longer than the model's limit, token "
    "percentiles and the longest documents (in `result.analysis`). Nothing is embedded or written."
)


class LoadOptions(BaseModel):
    collection: str = Field(..., description="Target collection.", examples=["pubmed"])
    create_collection: bool = Field(True, description="Create the collection if it is missing.")
    embeddings: list[EmbeddingFieldSpec] | None = Field(
        None,
        description=(
            "Fields to embed. Omit to use the collection's embedding profile "
            "(see `PUT /collections/{name}/profile`)."
        ),
    )
    key_field: str | None = Field(
        None,
        description="Field whose value becomes the document `_key` (stringified).",
        examples=["id"],
    )
    on_duplicate: OnDuplicate | None = Field(
        None,
        description=(
            "Behaviour when `_key` already exists: `error` (report per-document error), "
            "`update` (merge), `replace` or `ignore`. Defaults to the profile's setting or `error`."
        ),
    )
    batch_size: int | None = Field(None, ge=1, le=10000, description="Documents per batch.")
    store_metadata: bool | None = Field(
        None,
        description=(
            "Store `<target>_meta` ({status, provider, model, dimension, embedded_at}) next to "
            "each vector. Defaults to true."
        ),
    )
    vector_index: VectorIndexSpec | None = Field(
        None, description="Create an ArangoDB vector index after the load."
    )
    dry_run: bool = Field(False, description="Embed but do not write; returns sample output.")
    analyze: bool = Field(False, description=ANALYZE_DESCRIPTION)
    max_documents: int | None = Field(None, ge=1, description="Stop after this many documents.")


class FileLoadOptions(LoadOptions):
    format: FileFormat = Field("auto", description="Input format; `auto` detects it.")
    csv: CsvOptions = Field(default_factory=CsvOptions)
    json_path: str | None = Field(
        None,
        description="ijson prefix of the document array inside a JSON file, e.g. `data.item`.",
    )


class LoadDocumentsRequest(LoadOptions):
    documents: list[dict[str, Any]] = Field(..., description="Documents to load.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection": "pubmed",
                    "key_field": "id",
                    "embeddings": [
                        {
                            "source_field": "text",
                            "target_field": "embedding",
                            "preset": "pubmedbert",
                        }
                    ],
                    "documents": [
                        {"id": 1234, "text": "Title. Abstract text... Keywords: oncology"}
                    ],
                }
            ]
        }
    }


class LoadUrlRequest(FileLoadOptions):
    urls: list[str] = Field(
        ...,
        min_length=1,
        description="HTTP(S) or public `gs://bucket/object` URLs of JSON/JSONL/CSV files (.gz ok).",
        examples=[["https://storage.googleapis.com/my-bucket/data/part-0001.jsonl"]],
    )


class ReembedRequest(BaseModel):
    embeddings: list[EmbeddingFieldSpec] | None = Field(
        None, description="Fields to embed. Omit to use the collection's profile."
    )
    only_missing: bool = Field(
        True, description="Only embed documents where the target vector is missing."
    )
    filter: dict[str, Any] = Field(
        default_factory=dict,
        description='Equality filters on dot-path fields, e.g. `{"lang": "en"}`.',
    )
    batch_size: int | None = Field(None, ge=1, le=10000)
    store_metadata: bool | None = None
    vector_index: VectorIndexSpec | None = None
    max_documents: int | None = Field(None, ge=1)
    analyze: bool = Field(False, description=ANALYZE_DESCRIPTION)


class PreviewResponse(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    filename: str | None
    format: str
    sample: list[dict[str, Any]]
    fields: list[Any] = Field(..., description="Detected field paths (FieldPathInfo).")
    estimated_total: int | None = Field(None, description="Record count for JSONL/CSV.")
    size_bytes: int | None = None
