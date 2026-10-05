"""ArangoDB vector index management.

Notes (ArangoDB 3.12.4+, server started with ``--experimental-vector-index``):
- The index is trained on existing documents, so the collection must contain vectors first.
- ``nLists`` defaults to ~15·√N (capped so each list gets ~40 training points).
- Creating the index on a large collection can exceed the client timeout; in that case we poll
  until the index shows up.
"""

from __future__ import annotations

import math
import time
from typing import Any

from arango.collection import StandardCollection
from arango.database import StandardDatabase
from arango.exceptions import ArangoClientError, ArangoServerError
from requests.exceptions import RequestException

from app.errors import AppError, describe_arango_error
from app.ingest.text import split_path


def default_n_lists(count: int) -> int:
    return max(1, min(int(15 * math.sqrt(count)), count // 40 or 1))


def find_vector_index(collection: StandardCollection, field: str) -> dict[str, Any] | None:
    for index in collection.indexes():  # type: ignore[union-attr]
        if index.get("type") == "vector" and index.get("fields") == [field]:
            return dict(index)
    return None


def detect_dimension(db: StandardDatabase, collection: str, field: str) -> int | None:
    cursor = db.aql.execute(
        "FOR d IN @@col FILTER IS_ARRAY(d.@field) LIMIT 1 RETURN LENGTH(d.@field)",
        bind_vars={"@col": collection, "field": split_path(field)},
    )
    values = list(cursor)  # type: ignore[arg-type]
    return int(values[0]) if values else None


def ensure_vector_index(
    db: StandardDatabase,
    collection_name: str,
    field: str,
    *,
    metric: str = "cosine",
    dimension: int | None = None,
    n_lists: int | None = None,
    default_n_probe: int | None = None,
    training_iterations: int | None = None,
    name: str | None = None,
    parallelism: int | None = None,
    poll_timeout_s: float = 3600,
) -> dict[str, Any]:
    collection = db.collection(collection_name)
    existing = find_vector_index(collection, field)
    if existing:
        return {"status": "exists", "index": existing}

    count = int(collection.count())  # type: ignore[arg-type]
    if count == 0:
        raise AppError(
            400,
            "vector_index_requires_documents",
            "A vector index can only be created on a collection that already contains vectors.",
        )
    if dimension is None:
        dimension = detect_dimension(db, collection_name, field)
    if dimension is None:
        raise AppError(400, "vector_index_no_vectors", f"No document has an array in '{field}'.")

    params: dict[str, Any] = {
        "metric": metric,
        "dimension": dimension,
        "nLists": n_lists or default_n_lists(count),
    }
    if default_n_probe:
        params["defaultNProbe"] = default_n_probe
    if training_iterations:
        params["trainingIterations"] = training_iterations
    body: dict[str, Any] = {
        "type": "vector",
        "name": name or f"vec_{field.replace('.', '_')}",
        "fields": [field],
        "sparse": True,
        "params": params,
    }
    if parallelism:
        body["parallelism"] = parallelism

    try:
        index = collection.add_index(body)
        return {"status": "created", "index": index}
    except ArangoServerError as exc:
        hint = ""
        if getattr(exc, "error_code", None) in (10, 1555) or "vector" in str(exc).lower():
            hint = (
                " Make sure ArangoDB >= 3.12.4 runs with --experimental-vector-index and all "
                "vectors have the same dimension."
            )
        raise AppError(502, "vector_index_failed", describe_arango_error(exc) + hint) from exc
    except (ArangoClientError, RequestException) as exc:
        # Most likely a client-side timeout while the server keeps building the index.
        deadline = time.monotonic() + poll_timeout_s
        while time.monotonic() < deadline:
            time.sleep(5)
            try:
                found = find_vector_index(collection, field)
            except (ArangoClientError, RequestException):
                continue
            if found:
                return {"status": "created", "index": found}
        raise AppError(
            502, "vector_index_timeout", "Timed out waiting for the vector index."
        ) from exc


def ensure_persistent_index(
    collection: StandardCollection, field: str, name: str
) -> dict[str, Any]:
    for index in collection.indexes():  # type: ignore[union-attr]
        if index.get("type") == "persistent" and index.get("fields") == [field]:
            return dict(index)
    return dict(
        collection.add_index(  # type: ignore[arg-type]
            {"type": "persistent", "fields": [field], "sparse": False, "name": name}
        )
    )
