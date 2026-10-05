from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.ingest.text import meta_field_for, split_path


class EmbeddingFieldSpec(BaseModel):
    """Labels a field (or template of fields) to embed and where to store the vector."""

    source_field: str | None = Field(
        None,
        description="Dot-path of the field to embed, e.g. `text` or `content.body`.",
        examples=["text"],
    )
    template: str | None = Field(
        None,
        description=(
            "Alternative to `source_field`: compose text from several fields using `{path}` "
            "placeholders, e.g. `{title}\\n\\n{abstract}`."
        ),
    )
    target_field: str | None = Field(
        None,
        description=(
            "Field that receives the vector. Defaults to `<source_field>_embedding` "
            "(or `embedding` when using a template)."
        ),
        examples=["embedding"],
    )
    preset: str | None = Field(
        None,
        description="Named preset (see `GET /presets`) that sets provider, model and options.",
        examples=["pubmedbert"],
    )
    provider: str | None = Field(
        None,
        description="Provider name (see `GET /providers`).",
        examples=["sentence_transformers"],
    )
    model: str | None = Field(
        None,
        description="Model id. Defaults to the provider's default model.",
        examples=["NeuML/pubmedbert-base-embeddings"],
    )
    dimensions: int | None = Field(
        None, ge=1, le=65536, description="Requested output dimension, if the model supports it."
    )
    provider_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific options (API key, base URL, device...). Never stored.",
    )
    on_missing: Literal["skip", "null", "error"] = Field(
        "skip",
        description=(
            "What to do when the source is missing/empty: `skip` (store document without vector), "
            "`null` (set the target to null) or `error` (reject the document)."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> EmbeddingFieldSpec:
        if not self.source_field and not self.template:
            raise ValueError("Either source_field or template is required")
        if not self.preset and not self.provider:
            raise ValueError("Either preset or provider is required")
        if self.target_field is not None and not split_path(self.target_field):
            raise ValueError("target_field must not be empty")
        return self

    @property
    def resolved_target(self) -> str:
        if self.target_field:
            return self.target_field
        if self.source_field:
            return f"{self.source_field.replace('.', '_')}_embedding"
        return "embedding"

    @property
    def meta_field(self) -> str:
        return meta_field_for(self.resolved_target)

    def public_dict(self) -> dict[str, Any]:
        """Serializable form with secrets removed (safe to persist in profiles / job info)."""
        data = self.model_dump(exclude_none=True)
        data["provider_options"] = {
            k: v
            for k, v in self.provider_options.items()
            if not any(s in k.lower() for s in ("key", "secret", "token", "password"))
        }
        return data


class VectorIndexSpec(BaseModel):
    """ArangoDB vector index settings.

    Requires ArangoDB 3.12.4+ started with ``--experimental-vector-index``.
    """

    create: bool = Field(True, description="Create the index after loading if missing.")
    field: str | None = Field(
        None, description="Field to index. Defaults to the first embedding target field."
    )
    metric: Literal["cosine", "l2", "innerProduct"] = "cosine"
    n_lists: int | None = Field(
        None, ge=1, description="Number of IVF clusters. Default ≈ 15·√N, capped at N/40."
    )
    default_n_probe: int | None = Field(
        None, ge=1, description="Default clusters probed per query."
    )
    training_iterations: int | None = Field(None, ge=1)
    name: str | None = Field(None, description="Index name. Defaults to `vec_<field>`.")
    parallelism: int | None = Field(None, ge=1, description="Threads used to build the index.")
