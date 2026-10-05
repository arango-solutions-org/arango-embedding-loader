from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.models.common import ERROR_RESPONSES
from app.models.jobs import Job
from app.services.jobs import job_manager

router = APIRouter(tags=["Jobs"], responses=ERROR_RESPONSES)


@router.get("/jobs", response_model=list[Job], summary="List recent jobs")
async def list_jobs() -> list[Job]:
    return job_manager.list()


@router.get("/jobs/{job_id}", response_model=Job, summary="Get a job")
async def get_job(job_id: str) -> Job:
    return job_manager.get(job_id).job


@router.get(
    "/jobs/{job_id}/events",
    summary="Stream job progress (Server-Sent Events)",
    description=(
        "`text/event-stream` of `job` events whose data is the Job JSON. "
        "The stream closes after the job reaches a terminal state."
    ),
    response_class=EventSourceResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def job_events(job_id: str) -> EventSourceResponse:
    handle = job_manager.get(job_id)

    async def stream() -> AsyncIterator[dict[str, str]]:
        async for job in job_manager.events(handle):
            yield {"event": "job", "data": job.model_dump_json()}

    return EventSourceResponse(stream())


@router.delete("/jobs/{job_id}", response_model=Job, summary="Cancel a running job")
async def cancel_job(job_id: str) -> Job:
    return job_manager.cancel(job_id)
