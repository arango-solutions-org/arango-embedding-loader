"""Background watchers that embed documents inserted into profiled collections by any client.

A watcher polls for documents whose ``<target>_meta.status`` is missing (i.e. not yet processed
by this loader), embeds them per the collection profile and writes the vectors back. A persistent
index on ``<target>_meta.status`` keeps the lookup cheap on large collections.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

from arango.database import StandardDatabase

from app.arango import ArangoConnection, get_database
from app.ingest.text import split_path
from app.models.profiles import Profile, WatcherState
from app.services import profiles as profile_store
from app.services.pipeline import ResolvedSpec, embed_documents, resolve_specs
from app.services.writer import update_batch

log = logging.getLogger(__name__)


def pending_query(specs: list[ResolvedSpec]) -> tuple[str, dict[str, Any]]:
    bind: dict[str, Any] = {}
    ors = []
    for i, spec in enumerate(specs):
        bind[f"m{i}"] = [*split_path(spec.meta), "status"]
        ors.append(f"d.@m{i} == null")
    query = "FOR d IN @@col FILTER " + " OR ".join(ors) + " LIMIT @n RETURN d"
    return query, bind


def count_pending(db: StandardDatabase, collection: str, specs: list[ResolvedSpec]) -> int:
    query, bind = pending_query(specs)
    count_query = query.replace(" LIMIT @n RETURN d", " COLLECT WITH COUNT INTO c RETURN c")
    cursor = db.aql.execute(count_query, bind_vars={**bind, "@col": collection})
    values = list(cursor)  # type: ignore[arg-type]
    return int(values[0]) if values else 0


async def process_pending(
    db: StandardDatabase,
    collection: str,
    profile: Profile,
    specs: list[ResolvedSpec],
    state: WatcherState | None = None,
) -> tuple[int, int]:
    """Embed one batch of pending documents. Returns (documents processed, write errors)."""
    query, bind = pending_query(specs)
    cursor = await asyncio.to_thread(
        db.aql.execute,
        query,
        bind_vars=cast(Any, {**bind, "@col": collection, "n": profile.batch_size}),
    )
    docs: list[dict[str, Any]] = list(cursor)  # type: ignore[arg-type]
    if not docs:
        return 0, 0
    outcomes = await embed_documents(docs, specs, store_metadata=True, only_pending=True)
    updates: list[dict[str, Any]] = []
    for doc, outcome in zip(docs, outcomes, strict=True):
        update: dict[str, Any] = {"_key": doc["_key"]}
        for spec in specs:
            for path in (spec.target, spec.meta):
                top = split_path(path)[0]
                if top in doc:
                    update[top] = doc[top]
        updates.append(update)
        if state is not None:
            if outcome.error:
                state.failed += 1
            elif outcome.embedded:
                state.embedded += 1
            elif outcome.skipped:
                state.skipped += 1
    result = await asyncio.to_thread(update_batch, db.collection(collection), updates)
    return len(docs), len(result.errors)


class WatcherManager:
    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._states: dict[tuple[str, str], WatcherState] = {}

    @staticmethod
    def _key(conn: ArangoConnection, collection: str) -> tuple[str, str]:
        return conn.cache_key, collection

    def state(self, conn: ArangoConnection, collection: str) -> WatcherState:
        key = self._key(conn, collection)
        state = self._states.get(key) or WatcherState()
        task = self._tasks.get(key)
        state.running = bool(task and not task.done())
        return state

    def start(self, conn: ArangoConnection, collection: str) -> WatcherState:
        key = self._key(conn, collection)
        task = self._tasks.get(key)
        if task and not task.done():
            return self.state(conn, collection)
        self._states.setdefault(key, WatcherState())
        self._tasks[key] = asyncio.create_task(
            self._loop(key, conn, collection), name=f"watcher-{collection}"
        )
        return self.state(conn, collection)

    def stop(self, conn: ArangoConnection, collection: str) -> WatcherState:
        key = self._key(conn, collection)
        task = self._tasks.pop(key, None)
        if task:
            task.cancel()
        state = self._states.get(key) or WatcherState()
        state.running = False
        return state

    async def start_all(self, conn: ArangoConnection) -> int:
        db = get_database(conn)
        profiles = await asyncio.to_thread(profile_store.list_profiles, db)
        started = 0
        for profile in profiles:
            if profile.auto_watch:
                self.start(conn, profile.collection)
                started += 1
        return started

    async def shutdown(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()

    async def _loop(self, key: tuple[str, str], conn: ArangoConnection, collection: str) -> None:
        state = self._states[key]
        failures = 0
        while True:
            interval = 10.0
            try:
                db = get_database(conn)
                profile = await asyncio.to_thread(profile_store.get_profile, db, collection)
                if profile is None or not profile.auto_watch:
                    log.info("Watcher for %s stopping: profile removed or disabled", collection)
                    return
                interval = profile.watch_interval_s
                if not await asyncio.to_thread(db.has_collection, collection):
                    await asyncio.sleep(interval)
                    continue
                specs = resolve_specs(profile.embeddings)
                processed, write_errors = await process_pending(
                    db, collection, profile, specs, state
                )
                state.last_run_at = datetime.now(UTC)
                state.last_error = (
                    f"{write_errors} document update(s) failed" if write_errors else None
                )
                failures = 0
                if processed < profile.batch_size or write_errors:
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                failures += 1
                state.last_error = f"{type(exc).__name__}: {exc}"
                state.last_run_at = datetime.now(UTC)
                log.warning("Watcher for %s failed: %s", collection, exc)
                await asyncio.sleep(min(300.0, interval * (2 ** min(failures, 5))))


def embedded_counts(db: StandardDatabase, collection: str, profile: Profile) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in profile.embeddings:
        target = spec.resolved_target
        cursor = db.aql.execute(
            "FOR d IN @@col FILTER IS_ARRAY(d.@t) COLLECT WITH COUNT INTO c RETURN c",
            bind_vars={"@col": collection, "t": split_path(target)},
        )
        values = list(cursor)  # type: ignore[arg-type]
        counts[target] = int(values[0]) if values else 0
    return counts


watcher_manager = WatcherManager()
