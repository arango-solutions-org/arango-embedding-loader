"""Background watchers that embed documents inserted into profiled collections by any client.

A watcher polls for documents whose ``<target>_meta.status`` is missing (i.e. not yet processed
by this loader), embeds them per the collection profile and writes the vectors back. A persistent
index on ``<target>_meta.status`` keeps the lookup cheap on large collections.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from datetime import UTC, datetime
from typing import Any, cast

from arango.database import StandardDatabase

from app.arango import ArangoConnection, get_database
from app.ingest.text import split_path
from app.models.profiles import Profile, WatcherState
from app.providers.base import ProviderError
from app.services import profiles as profile_store
from app.services.pipeline import (
    DocOutcome,
    ResolvedSpec,
    embed_documents,
    mark_failed,
    resolve_specs,
    write_chunks_first,
)
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
    # A document that can't be embedded is marked failed (`<target>_meta.status: "error"`, so
    # it leaves the queue) instead of failing the batch, which would retry it forever.
    docs, outcomes = await embed_isolating(docs, specs)
    col = db.collection(collection)
    # Chunks first: a document whose chunks fail is marked `status: "error"` and keeps any
    # previous vector, instead of being left chunked without chunks.
    await asyncio.to_thread(write_chunks_first, db, col, docs, outcomes, specs, store_metadata=True)
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
    result = await asyncio.to_thread(update_batch, col, updates)
    return len(docs), len(result.errors)


async def _provider_works(specs: list[ResolvedSpec]) -> bool:
    """Every spec's provider embeds a tiny text: an error is not a provider outage."""
    try:
        for spec in specs:
            await spec.provider.embed(
                ["ok"],
                model=spec.model,
                options=spec.options,
                dimensions=spec.dimensions,
                max_attempts=1,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Provider check failed (%s: %s); treating it as an outage", type(exc).__name__, exc
        )
        return False
    return True


#: Errors that say nothing about a document's content: the provider is unavailable, limiting
#: requests or rejecting the credentials. They never mark a document as failed.
_OUTAGE_STATUS = frozenset({401, 403, 408, 429, 502, 503, 504})
_OUTAGE_NAMES = (
    "RateLimit",
    "Throttl",
    "TooManyRequests",
    "Timeout",
    "ServiceUnavailable",
    "Connection",
    "ConnectError",
    "ReadError",
    "RemoteProtocolError",
    "Authentication",
    "PermissionDenied",
)


def is_outage(exc: BaseException) -> bool:
    """``exc``, or an error it was raised from, is about the provider, not the input."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # A ProviderError's own status_code is the loader's HTTP status, not the provider's.
        status = None if isinstance(cur, ProviderError) else getattr(cur, "status_code", None)
        status = status or getattr(getattr(cur, "response", None), "status_code", None)
        if isinstance(status, int) and status in _OUTAGE_STATUS:
            return True
        if any(n in type(cur).__name__ for n in _OUTAGE_NAMES):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


async def embed_isolating(
    docs: list[dict[str, Any]], specs: list[ResolvedSpec]
) -> tuple[list[dict[str, Any]], list[DocOutcome]]:
    """Embed ``docs`` as one batch; if that fails, find the documents that cause it.

    Any error in the batch (from the provider, a tokenizer, a template...) leads to a check
    that each provider still embeds a tiny text. If it doesn't, the provider is down: the
    error is raised and the watcher backs off and retries the batch later. If it does, each
    document is embedded on its own; one that still fails, with any error, is put back as it
    was and marked ``<target>_meta.status: "error"`` (``reason: "embedding failed"``), and the
    others are stored. Returns the documents (copies, after a fallback) and their outcomes.

    Good documents are not marked failed for the provider's problems: an outage error
    (:func:`is_outage`: rate limits, timeouts, connection or auth errors), in the batch or in
    any single document, is raised and nothing is marked; and if every document of a batch of
    two or more fails on its own, the last error is raised instead.
    """
    originals = copy.deepcopy(docs)
    try:
        outcomes = await embed_documents(
            docs, specs, store_metadata=True, only_pending=True, errors_per_document=True
        )
        return docs, outcomes
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        batch_error = exc
    log.warning(
        "Embedding a batch of %d document(s) failed (%s: %s); checking the provider",
        len(originals),
        type(batch_error).__name__,
        batch_error,
    )
    if is_outage(batch_error) or not await _provider_works(specs):
        raise batch_error
    out_docs: list[dict[str, Any]] = []
    out: list[DocOutcome] = []
    last_error: Exception = batch_error
    raised = 0
    for doc in originals:
        attempt = copy.deepcopy(doc)
        try:
            (outcome,) = await embed_documents(
                [attempt], specs, store_metadata=True, only_pending=True, errors_per_document=True
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any error marks this document only
            if is_outage(exc):
                raise
            last_error = exc
            raised += 1
            attempt = doc
            message = f"{type(exc).__name__}: {exc}"
            mark_failed(attempt, specs, "embedding failed", message)
            outcome = DocOutcome(error=f"Document {doc.get('_key')!r}: {message}")
            log.error("Watcher could not embed %r: %s", doc.get("_key"), message)
        out_docs.append(attempt)
        out.append(outcome)
    if len(originals) > 1 and raised == len(originals):
        log.warning(
            "Every document of the batch failed on its own; treating it as a provider problem"
        )
        raise last_error
    return out_docs, out


class WatcherManager:
    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._states: dict[tuple[str, str], WatcherState] = {}

    @staticmethod
    def _key(conn: ArangoConnection, collection: str) -> tuple[str, str]:
        # One watcher per collection, whoever started it (credentials differ per user).
        return f"{conn.endpoint}|{conn.database}", collection

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
