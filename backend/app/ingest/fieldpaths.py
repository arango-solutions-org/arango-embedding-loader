"""Detect the field paths present in a sample of documents (used by the GUI's labeling step)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldPathInfo(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    path: str = Field(..., description="Dot-path of the field, e.g. `content.body`.")
    types: list[str] = Field(..., description="JSON types observed for this field.")
    coverage: float = Field(..., description="Fraction of sampled documents containing it.")
    avg_length: float | None = Field(None, description="Average length for string values.")
    example: Any = Field(None, description="Truncated example value.")
    embeddable: bool = Field(..., description="Heuristic: looks like text worth embedding.")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _walk(doc: dict[str, Any], prefix: str, out: dict[str, list[Any]], depth: int) -> None:
    for key, value in doc.items():
        path = f"{prefix}.{key}" if prefix else key
        out.setdefault(path, []).append(value)
        if isinstance(value, dict) and depth < 6:
            _walk(value, path, out, depth + 1)


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 160:
        return value[:157] + "..."
    if isinstance(value, list):
        return [_truncate(v) for v in value[:5]]
    if isinstance(value, dict):
        return "{…}"
    return value


def detect_field_paths(docs: Iterable[dict[str, Any]]) -> list[FieldPathInfo]:
    docs = list(docs)
    if not docs:
        return []
    values: dict[str, list[Any]] = {}
    for doc in docs:
        _walk(doc, "", values, 0)
    result: list[FieldPathInfo] = []
    for path, vals in values.items():
        if path in ("_id", "_rev"):
            continue
        types = sorted({_type_name(v) for v in vals})
        strings = [v for v in vals if isinstance(v, str)]
        avg_len = sum(len(s) for s in strings) / len(strings) if strings else None
        string_lists = [
            v for v in vals if isinstance(v, list) and v and all(isinstance(x, str) for x in v)
        ]
        embeddable = bool(
            (avg_len is not None and avg_len >= 20 and "object" not in types)
            or (string_lists and len(string_lists) >= len(vals) / 2)
        )
        example = next((v for v in vals if v is not None), None)
        result.append(
            FieldPathInfo(
                path=path,
                types=types,
                coverage=round(len(vals) / len(docs), 3),
                avg_length=round(avg_len, 1) if avg_len is not None else None,
                example=_truncate(example),
                embeddable=embeddable,
            )
        )
    result.sort(key=lambda f: (not f.embeddable, -(f.avg_length or 0), f.path))
    return result
