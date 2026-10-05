"""In-process job manager for long-running loads, with progress events for SSE."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.errors import AppError, describe_arango_error
from app.models.jobs import DocumentError, Job, JobKind
from app.providers.base import ProviderError

log = logging.getLogger(__name__)

MAX_ERRORS = 100
MAX_JOBS = 500
TERMINAL = {"succeeded", "failed", "cancelled"}


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobHandle:
    def __init__(self, job: Job):
        self.job = job
        self.task: asyncio.Task[None] | None = None
        self._changed = asyncio.Event()
        self.cleanup: list[Callable[[], None]] = []
        self.bytes_probe: Callable[[], int | None] | None = None

    def add_error(self, error: str, *, index: int | None = None, key: str | None = None) -> None:
        self.job.counts.failed += 1
        if len(self.job.errors) < MAX_ERRORS:
            self.job.errors.append(DocumentError(index=index, key=key, error=error))

    def touch(self) -> None:
        job = self.job
        if self.bytes_probe is not None:
            job.bytes_read = self.bytes_probe()
        if job.total:
            done = job.counts.read
            job.progress = min(1.0, done / job.total)
        elif job.bytes_total and job.bytes_read is not None:
            job.progress = min(1.0, job.bytes_read / job.bytes_total)
        self._changed.set()

    async def wait_changed(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), timeout)
        except TimeoutError:
            pass
        self._changed.clear()

    @property
    def done(self) -> bool:
        return self.job.status in TERMINAL


Runner = Callable[[JobHandle], Awaitable[None]]


class JobManager:
    def __init__(self) -> None:
        self._jobs: OrderedDict[str, JobHandle] = OrderedDict()

    def submit(
        self,
        *,
        kind: JobKind,
        collection: str,
        database: str,
        config: dict[str, Any],
        runner: Runner,
        total: int | None = None,
        bytes_total: int | None = None,
    ) -> JobHandle:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            collection=collection,
            database=database,
            created_at=utcnow(),
            total=total,
            bytes_total=bytes_total,
            config=config,
        )
        handle = JobHandle(job)
        self._jobs[job.id] = handle
        self._evict()
        handle.task = asyncio.create_task(self._run(handle, runner), name=f"job-{job.id}")
        return handle

    async def _run(self, handle: JobHandle, runner: Runner) -> None:
        job = handle.job
        job.status = "running"
        job.started_at = utcnow()
        handle.touch()
        try:
            await runner(handle)
            job.status = "succeeded"
            if job.total is None or job.progress is not None:
                job.progress = 1.0
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.message = "Cancelled by user"
        except AppError as exc:
            job.status = "failed"
            job.message = str(exc.detail or exc.error)
        except ProviderError as exc:
            job.status = "failed"
            job.message = str(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("Job %s failed", job.id)
            job.status = "failed"
            msg = describe_arango_error(exc) if hasattr(exc, "error_message") else str(exc)
            job.message = f"{type(exc).__name__}: {msg}"
        finally:
            job.finished_at = utcnow()
            for fn in handle.cleanup:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.warning("Job cleanup failed", exc_info=True)
            handle.touch()

    def _evict(self) -> None:
        while len(self._jobs) > MAX_JOBS:
            for job_id, handle in self._jobs.items():
                if handle.done:
                    del self._jobs[job_id]
                    break
            else:
                break

    def get(self, job_id: str) -> JobHandle:
        handle = self._jobs.get(job_id)
        if handle is None:
            raise AppError(404, "job_not_found", f"Job '{job_id}' not found")
        return handle

    def list(self) -> list[Job]:
        return [h.job for h in reversed(self._jobs.values())]

    def cancel(self, job_id: str) -> Job:
        handle = self.get(job_id)
        if not handle.done and handle.task is not None:
            handle.task.cancel()
        return handle.job

    async def wait(self, handle: JobHandle) -> Job:
        if handle.task is not None:
            await asyncio.shield(handle.task)
        return handle.job

    async def events(self, handle: JobHandle, interval: float = 1.0) -> AsyncIterator[Job]:
        yield handle.job
        while not handle.done:
            await handle.wait_changed(interval)
            yield handle.job

    async def shutdown(self) -> None:
        for handle in self._jobs.values():
            if handle.task is not None and not handle.done:
                handle.task.cancel()


job_manager = JobManager()
