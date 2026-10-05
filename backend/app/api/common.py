from __future__ import annotations

from typing import Annotated

from fastapi import Query, Response

from app.models.jobs import Job
from app.services.jobs import JobHandle, job_manager

WaitQuery = Annotated[
    bool,
    Query(
        description=(
            "Block until the job finishes and return the final job (200). Otherwise return the "
            "queued job immediately (202) and track it with `GET /jobs/{id}` or SSE."
        )
    ),
]


async def job_response(handle: JobHandle, wait: bool, response: Response) -> Job:
    if wait:
        return await job_manager.wait(handle)
    response.status_code = 202
    return handle.job
