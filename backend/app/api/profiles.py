from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Path

from app.deps import ConnectionDep, DbDep
from app.errors import AppError
from app.ingest.text import split_path
from app.models.common import ERROR_RESPONSES
from app.models.profiles import Profile, ProfileConfig, ProfileStatus, WatcherState
from app.services import profiles as profile_store
from app.services.pipeline import resolve_specs
from app.services.vector_index import ensure_persistent_index, ensure_vector_index
from app.services.watcher import count_pending, embedded_counts, watcher_manager

router = APIRouter(tags=["Profiles"], responses=ERROR_RESPONSES)

CollectionPath = Annotated[str, Path(description="Collection name.", examples=["pubmed"])]


def _get_or_404(db: object, collection: str) -> Profile:
    profile = profile_store.get_profile(db, collection)  # type: ignore[arg-type]
    if profile is None:
        raise AppError(404, "profile_not_found", f"Collection '{collection}' has no profile")
    return profile


@router.get("/profiles", response_model=list[Profile], summary="List embedding profiles")
async def list_profiles(db: DbDep) -> list[Profile]:
    return await asyncio.to_thread(profile_store.list_profiles, db)


@router.put(
    "/collections/{name}/profile",
    response_model=ProfileStatus,
    summary="Set up an auto-embedding collection",
    description=(
        "Creates the collection if needed and attaches an embedding profile. Afterwards every "
        "load into the collection is embedded automatically (no `embeddings` needed), and with "
        "`auto_watch` a background watcher also embeds documents inserted by any other client. "
        "Secrets in `provider_options` are never persisted - configure API keys via environment "
        "variables for profiles."
    ),
)
async def put_profile(
    name: CollectionPath, body: ProfileConfig, db: DbDep, conn: ConnectionDep
) -> ProfileStatus:
    resolve_specs(body.embeddings)  # validate providers/models before saving

    def work() -> Profile:
        if not db.has_collection(name):
            db.create_collection(name)
        col = db.collection(name)
        for spec in body.embeddings:
            meta_status = f"{spec.meta_field}.status"
            ensure_persistent_index(
                col, meta_status, f"idx_{'_'.join(split_path(spec.meta_field))}_status"
            )
        profile = profile_store.save_profile(db, name, body)
        vi = body.vector_index
        if vi and vi.create and col.count():
            try:
                ensure_vector_index(
                    db,
                    name,
                    vi.field or body.embeddings[0].resolved_target,
                    metric=vi.metric,
                    n_lists=vi.n_lists,
                    default_n_probe=vi.default_n_probe,
                    training_iterations=vi.training_iterations,
                    name=vi.name,
                    parallelism=vi.parallelism,
                )
            except AppError:
                pass  # no vectors yet - created after the first load
        return profile

    await asyncio.to_thread(work)
    if body.auto_watch:
        watcher_manager.start(conn, name)
    else:
        watcher_manager.stop(conn, name)
    return await get_profile_status(name, db, conn)


@router.get(
    "/collections/{name}/profile",
    response_model=ProfileStatus,
    summary="Get a profile and its embedding status",
)
async def get_profile_status(name: CollectionPath, db: DbDep, conn: ConnectionDep) -> ProfileStatus:
    def work() -> ProfileStatus:
        profile = _get_or_404(db, name)
        if not db.has_collection(name):
            total, embedded, pending, indexes = 0, {}, 0, []
        else:
            col = db.collection(name)
            total = int(col.count())  # type: ignore[arg-type]
            embedded = embedded_counts(db, name, profile)
            try:
                pending = count_pending(db, name, resolve_specs(profile.embeddings))
            except Exception:  # noqa: BLE001 - provider unavailable shouldn't break status
                pending = -1
            indexes = [dict(i) for i in col.indexes()]  # type: ignore[union-attr]
        return ProfileStatus(
            profile=profile,
            total_documents=total,
            embedded_documents=embedded,
            pending_documents=pending,
            watcher=watcher_manager.state(conn, name),
            indexes=indexes,
        )

    return await asyncio.to_thread(work)


@router.delete(
    "/collections/{name}/profile", status_code=204, summary="Remove a profile (keeps data)"
)
async def delete_profile(name: CollectionPath, db: DbDep, conn: ConnectionDep) -> None:
    watcher_manager.stop(conn, name)
    if not await asyncio.to_thread(profile_store.delete_profile, db, name):
        raise AppError(404, "profile_not_found", f"Collection '{name}' has no profile")


@router.post(
    "/collections/{name}/profile/watcher/start",
    response_model=WatcherState,
    summary="Start the auto-embed watcher",
)
async def start_watcher(name: CollectionPath, db: DbDep, conn: ConnectionDep) -> WatcherState:
    profile = await asyncio.to_thread(_get_or_404, db, name)
    if not profile.auto_watch:
        config = ProfileConfig.model_validate(
            profile.model_dump(exclude={"collection", "created_at", "updated_at"})
            | {"auto_watch": True}
        )
        await asyncio.to_thread(profile_store.save_profile, db, name, config)
    return watcher_manager.start(conn, name)


@router.post(
    "/collections/{name}/profile/watcher/stop",
    response_model=WatcherState,
    summary="Stop the auto-embed watcher",
)
async def stop_watcher(name: CollectionPath, db: DbDep, conn: ConnectionDep) -> WatcherState:
    profile = await asyncio.to_thread(_get_or_404, db, name)
    if profile.auto_watch:
        config = ProfileConfig.model_validate(
            profile.model_dump(exclude={"collection", "created_at", "updated_at"})
            | {"auto_watch": False}
        )
        await asyncio.to_thread(profile_store.save_profile, db, name, config)
    return watcher_manager.stop(conn, name)
