from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.ingest.text import meta_field_for, split_path
from app.providers.long_input import (
    DEFAULT_TOKEN_CHARS,
    LEGACY_TRUNCATE_OPTIONS,
    LongInput,
    LongInputMode,
    legacy_long_input,
)

log = logging.getLogger(__name__)

# The long-input settings, shared by every model that embeds text (embedding specs and the
# provider test endpoint), so they are the same everywhere.
LongInputModeField = Annotated[
    LongInputMode,
    Field(
        description=(
            "What to do with a text longer than the model's token limit: `truncate_end` (keep "
            "the first tokens), `truncate_start` (keep the last tokens), `truncate_middle` (keep "
            "the start and the end), `chunk_mean` (embed overlapping chunks, store their mean), "
            "`chunk_docs` (store each chunk as its own document, linked by `parent_key`), `skip` "
            "(don't embed; status `skipped_too_long`) or `error` (fail the job). Replaces the "
            "old `truncate`/`truncation` provider options, which are still accepted."
        ),
    ),
]
ChunkOverlapField = Annotated[
    int,
    Field(
        ge=0,
        description=(
            "Tokens shared by consecutive chunks (`chunk_mean`, `chunk_docs`), rounded to the "
            "nearest word. Values above half the token limit are capped at half the limit, so "
            "each chunk starts at least half a chunk after the previous one."
        ),
    ),
]
MaxTokensField = Annotated[
    int | None,
    Field(
        ge=1,
        description=(
            "Advanced: token limit to apply. Defaults to the model's limit, when known. Needed "
            "for models whose limit the loader doesn't know (e.g. Ollama)."
        ),
    ),
]
TokenCharsField = Annotated[
    float,
    Field(
        gt=0,
        description=(
            "Advanced: characters per token, used to estimate token counts for models without "
            "a local tokenizer. The default, 3, over-counts typical English (about 4) on purpose."
        ),
    ),
]
WarnOnLongInputField = Annotated[
    bool,
    Field(
        description=(
            "Log each over-limit text as a warning; when off, at info level. With `error` it is "
            "always logged as an error."
        ),
    ),
]


class LongInputSettingsMixin(BaseModel):
    """Behaviour shared by models with the long-input fields (declared on each model)."""

    if TYPE_CHECKING:
        long_input: LongInputMode
        chunk_overlap_tokens: int
        max_tokens: int | None
        token_chars: float
        warn_on_long_input: bool

    @model_validator(mode="before")
    @classmethod
    def _legacy_truncate_options(cls, data: Any) -> Any:
        """Map the old ``truncate``/``truncation`` provider options to ``long_input``.

        They are removed from ``provider_options``. When ``long_input`` is not given, a
        recognized old value sets it, so existing profiles and requests keep their behaviour;
        when it is given, it wins.
        """
        if not isinstance(data, dict):
            return data
        options = data.get("provider_options")
        if not isinstance(options, dict) or not any(k in options for k in LEGACY_TRUNCATE_OPTIONS):
            return data
        options = dict(options)
        found = [(k, options.pop(k)) for k in LEGACY_TRUNCATE_OPTIONS if k in options]
        data = {**data, "provider_options": options}
        explicit = data.get("long_input")
        for key, value in found:
            mode = legacy_long_input(value)
            if explicit is None and mode is not None:
                data["long_input"] = explicit = mode
                outcome = f"using long_input={mode}"
            elif mode is None:
                outcome = "value not recognized, ignored"
            else:
                outcome = f"ignored, long_input={explicit} is set"
            log.warning(
                "Provider option '%s'=%r is deprecated; set long_input instead (%s)",
                key,
                value,
                outcome,
            )
        return data

    @property
    def long_input_settings(self) -> LongInput:
        return LongInput(
            mode=self.long_input,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
            max_tokens=self.max_tokens,
            token_chars=self.token_chars,
            warn=self.warn_on_long_input,
        )


class EmbeddingFieldSpec(LongInputSettingsMixin):
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

    long_input: LongInputModeField = "chunk_docs"
    chunk_overlap_tokens: ChunkOverlapField = 25
    max_tokens: MaxTokensField = None
    token_chars: TokenCharsField = DEFAULT_TOKEN_CHARS
    warn_on_long_input: WarnOnLongInputField = True

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
