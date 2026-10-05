from __future__ import annotations

import asyncio
from typing import Annotated, Any, cast

from fastapi import APIRouter, Query, Response

from app.api.common import WaitQuery, job_response
from app.config import get_settings
from app.deps import ConnectionDep, DbDep
from app.errors import AppError
from app.ingest.fieldpaths import detect_field_paths
from app.models.collections import (
    CollectionInfo,
    CreateCollectionRequest,
    SampleResponse,
    VectorIndexRequest,
)
from app.models.common import ERROR_RESPONSES
from app.models.jobs import Job
from app.models.load import ReembedRequest
from app.services import profiles as profile_store
from app.services.jobs import job_manager
from app.services.pipeline import run_reembed
from app.services.vector_index import ensure_vector_index

router = APIRouter(tags=["Collections"], responses=ERROR_RESPONSES)


@router.get("/collections", response_model=list[CollectionInfo], summary="List collections")
async def list_collections(
    db: DbDep,
    include_counts: Annotated[bool, Query(description="Include document counts.")] = True,
) -> list[CollectionInfo]:
    def work() -> list[CollectionInfo]:
        profiles_name = get_settings().profiles_collection
        profiled = {p.collection for p in profile_store.list_profiles(db)}
        out = []
        for col in db.collections():  # type: ignore[union-attr]
            if col["system"] or col["name"] == profiles_name:
                continue
            count = db.collection(col["name"]).count() if include_counts else None
            out.append(
                CollectionInfo(
                    name=col["name"],
                    type=col["type"],
                    count=count,
                    has_profile=col["name"] in profiled,
                )
            )
        return sorted(out, key=lambda c: c.name)

    return await asyncio.to_thread(work)


@router.post(
    "/collections", response_model=CollectionInfo, status_code=201, summary="Create a collection"
)
async def create_collection(body: CreateCollectionRequest, db: DbDep) -> CollectionInfo:
    if await asyncio.to_thread(db.has_collection, body.name):
        raise AppError(409, "collection_exists", f"Collection '{body.name}' already exists")
    kwargs: dict[str, Any] = {}
    if body.shards:
        kwargs["shard_count"] = body.shards
    if body.replication_factor:
        kwargs["replication_factor"] = body.replication_factor
    await asyncio.to_thread(lambda: db.create_collection(body.name, **kwargs))
    return CollectionInfo(name=body.name, type="document", count=0)


def _get_collection_or_404(db: Any, name: str) -> Any:
    if not db.has_collection(name):
        raise AppError(404, "collection_not_found", f"Collection '{name}' not found")
    return db.collection(name)


@router.get(
    "/collections/{name}/sample",
    response_model=SampleResponse,
    summary="Sample documents and detect field paths",
)
async def sample_collection(
    name: str, db: DbDep, limit: Annotated[int, Query(ge=1, le=500)] = 20
) -> SampleResponse:
    def work() -> SampleResponse:
        col = _get_collection_or_404(db, name)
        cursor = db.aql.execute(
            "FOR d IN @@col LIMIT @n RETURN d", bind_vars=cast(Any, {"@col": name, "n": limit})
        )
        docs = list(cursor)  # type: ignore[arg-type]
        sample = []
        for doc in docs:
            trimmed = {}
            for k, v in doc.items():
                if isinstance(v, list) and len(v) > 8 and all(isinstance(x, float) for x in v[:8]):
                    trimmed[k] = [round(x, 6) for x in v[:8]] + [f"… ({len(v)} dims)"]
                else:
                    trimmed[k] = v
            sample.append(trimmed)
        return SampleResponse(
            collection=name,
            count=int(col.count()),
            sample=sample,
            fields=detect_field_paths(docs),
        )

    return await asyncio.to_thread(work)


@router.get(
    "/collections/{name}/indexes", response_model=list[dict[str, Any]], summary="List indexes"
)
async def list_indexes(name: str, db: DbDep) -> list[dict[str, Any]]:
    def work() -> list[dict[str, Any]]:
        return [dict(i) for i in _get_collection_or_404(db, name).indexes()]

    return await asyncio.to_thread(work)


@router.post(
    "/collections/{name}/vector-index",
    response_model=dict[str, Any],
    summary="Create a vector index",
    description=(
        "Creates an ArangoDB vector index (requires ArangoDB 3.12.4+ started with "
        "`--experimental-vector-index`). The collection must already contain vectors. "
        "Returns `{status: created|exists, index}`."
    ),
)
async def create_vector_index(name: str, body: VectorIndexRequest, db: DbDep) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        _get_collection_or_404(db, name)
        return ensure_vector_index(
            db,
            name,
            body.field,
            metric=body.metric,
            dimension=body.dimension,
            n_lists=body.n_lists,
            default_n_probe=body.default_n_probe,
            training_iterations=body.training_iterations,
            name=body.name,
            parallelism=body.parallelism,
        )

    return await asyncio.to_thread(work)


@router.post(
    "/collections/{name}/embed",
    response_model=Job,
    summary="Embed documents already in a collection",
    description=(
        "Re-embeds existing documents (by default only those missing the target vector). "
        "Omit `embeddings` to use the collection's profile."
    ),
    responses={202: {"model": Job, "description": "Job accepted (when `wait=false`)."}},
)
async def reembed_collection(
    name: str,
    body: ReembedRequest,
    db: DbDep,
    conn: ConnectionDep,
    response: Response,
    wait: WaitQuery = False,
) -> Job:
    handle = job_manager.submit(
        kind="reembed",
        collection=name,
        database=conn.database,
        config={"only_missing": body.only_missing, "filter": body.filter},
        runner=lambda h: run_reembed(h, db, name, body),
    )
    return await job_response(handle, wait, response)
