"""Batch writes to ArangoDB with per-document error reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arango.collection import StandardCollection

from app.errors import describe_arango_error


@dataclass
class WriteResult:
    written: int = 0
    updated: int = 0
    errors: list[tuple[int, str | None, str]] = field(default_factory=list)


def _collect(results: Any, docs: list[dict[str, Any]], out: WriteResult) -> None:
    if not isinstance(results, list):
        out.written += len(docs)
        return
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            key = docs[i].get("_key") if i < len(docs) else None
            msg = describe_arango_error(res) if hasattr(res, "error_message") else str(res)
            out.errors.append((i, str(key) if key is not None else None, msg))
        else:
            out.written += 1
            if isinstance(res, dict) and (res.get("_old_rev") or res.get("_oldRev")):
                out.updated += 1


def insert_batch(
    collection: StandardCollection, docs: list[dict[str, Any]], on_duplicate: str
) -> WriteResult:
    out = WriteResult()
    if not docs:
        return out
    overwrite_mode = None if on_duplicate == "error" else on_duplicate
    results = collection.insert_many(
        docs, overwrite_mode=overwrite_mode, raise_on_document_error=False
    )
    _collect(results, docs, out)
    return out


def update_batch(collection: StandardCollection, docs: list[dict[str, Any]]) -> WriteResult:
    out = WriteResult()
    if not docs:
        return out
    results = collection.update_many(
        docs, check_rev=False, merge=True, keep_none=True, raise_on_document_error=False
    )
    _collect(results, docs, out)
    out.updated = out.written
    return out
