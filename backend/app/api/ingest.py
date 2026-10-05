from __future__ import annotations

import asyncio
import json
import os
import tempfile
from itertools import islice
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.api.common import WaitQuery, job_response
from app.config import get_settings
from app.deps import ConnectionDep, DbDep
from app.errors import AppError
from app.ingest.fieldpaths import detect_field_paths
from app.ingest.parsers import (
    CsvOptions,
    FileFormat,
    as_buffered,
    count_records,
    detect_format,
    iter_documents,
    maybe_decompress,
)
from app.models.common import ERROR_RESPONSES
from app.models.jobs import Job
from app.models.load import FileLoadOptions, LoadDocumentsRequest, LoadUrlRequest, PreviewResponse
from app.services.jobs import job_manager
from app.services.pipeline import run_load
from app.services.sources import (
    check_url_allowed,
    file_source,
    head_sizes,
    list_source,
    normalize_url,
    url_source,
)

router = APIRouter(tags=["Ingest"], responses=ERROR_RESPONSES)

CHUNK = 1024 * 1024


async def save_upload(upload: UploadFile) -> str:
    """Persist an upload to a temp file the background job owns (uploads close on response)."""
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024
    directory = str(settings.upload_dir) if settings.upload_dir else None
    suffix = os.path.splitext(upload.filename or "")[1]
    fd, path = tempfile.mkstemp(prefix="ael-", suffix=suffix, dir=directory)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise AppError(
                        413, "upload_too_large", f"Upload exceeds {settings.max_upload_mb} MB"
                    )
                await asyncio.to_thread(out.write, chunk)
    except BaseException:
        os.unlink(path)
        raise
    return path


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _preview_file(
    path: str,
    filename: str | None,
    fmt: FileFormat,
    csv_opts: CsvOptions,
    json_path: str | None,
    limit: int,
) -> PreviewResponse:
    with open(path, "rb") as fh:
        data = maybe_decompress(fh, filename)
        buffered = as_buffered(data)
        resolved = detect_format(filename, buffered.peek(4096)) if fmt == "auto" else fmt
        docs = list(
            islice(
                iter_documents(
                    buffered, filename=None, fmt=resolved, csv_options=csv_opts, json_path=json_path
                ),
                limit,
            )
        )
    return PreviewResponse(
        filename=filename,
        format=resolved,
        sample=docs[: min(limit, 20)],
        fields=[f.model_dump() for f in detect_field_paths(docs)],
        estimated_total=count_records(path, filename, resolved),
        size_bytes=os.path.getsize(path),
    )


@router.post(
    "/preview",
    response_model=PreviewResponse,
    summary="Preview a file and detect its fields",
    description=(
        "Parses the first documents of an uploaded JSON / JSONL / CSV file (optionally gzipped) "
        "and returns a sample plus detected field paths, so you can choose which field to embed."
    ),
)
async def preview(
    file: Annotated[UploadFile, File(description="JSON, JSONL/NDJSON or CSV file (.gz ok).")],
    format: Annotated[FileFormat, Form()] = "auto",
    csv_delimiter: Annotated[str, Form(min_length=1, max_length=1)] = ",",
    json_path: Annotated[str | None, Form()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> PreviewResponse:
    path = await save_upload(file)
    try:
        return await asyncio.to_thread(
            _preview_file,
            path,
            file.filename,
            format,
            CsvOptions(delimiter=csv_delimiter),
            json_path,
            limit,
        )
    finally:
        _remove(path)


@router.post(
    "/load",
    response_model=Job,
    summary="Load documents from the request body",
    description=(
        "Embeds the labeled field(s) of each document and inserts the documents - with the "
        "vectors stored in the same document - into `collection`. If `embeddings` is omitted, "
        "the collection's embedding profile is used."
    ),
    status_code=200,
    responses={202: {"model": Job, "description": "Job accepted (when `wait=false`)."}},
)
async def load_documents(
    body: LoadDocumentsRequest,
    db: DbDep,
    conn: ConnectionDep,
    response: Response,
    wait: WaitQuery = False,
) -> Job:
    opts = body.model_copy(update={"documents": []})
    source = list_source(body.documents)
    handle = job_manager.submit(
        kind="load_documents",
        collection=body.collection,
        database=conn.database,
        config={"documents": len(body.documents), "dry_run": body.dry_run},
        total=len(body.documents)
        if not body.max_documents
        else min(len(body.documents), body.max_documents),
        runner=lambda h: run_load(h, db, opts, source),
    )
    return await job_response(handle, wait, response)


def _parse_config(raw: str | None, collection: str | None) -> FileLoadOptions:
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise AppError(400, "invalid_config", f"`config` is not valid JSON: {exc}") from exc
    if collection:
        data["collection"] = collection
    try:
        return FileLoadOptions.model_validate(data)
    except ValidationError as exc:
        raise AppError(422, "invalid_config", exc.errors(include_url=False)) from exc


@router.post(
    "/load/file",
    response_model=Job,
    summary="Load documents from an uploaded file",
    description=(
        "Multipart upload of a JSON array, JSONL/NDJSON or CSV file (optionally gzipped). "
        "`config` is a JSON-encoded `FileLoadOptions` object; `collection` may be given as a "
        "query parameter instead. With an embedding profile on the collection, `config` can be "
        "omitted entirely:\n\n"
        "`curl -F file=@part1.jsonl 'http://localhost:8000/api/v1/load/file?collection=pubmed'`"
    ),
    responses={202: {"model": Job, "description": "Job accepted (when `wait=false`)."}},
)
async def load_file(
    db: DbDep,
    conn: ConnectionDep,
    response: Response,
    file: Annotated[UploadFile, File(description="JSON, JSONL/NDJSON or CSV file (.gz ok).")],
    config: Annotated[
        str | None, Form(description="JSON-encoded FileLoadOptions (see schema).")
    ] = None,
    collection: Annotated[str | None, Query(description="Target collection.")] = None,
    wait: WaitQuery = False,
) -> Job:
    opts = _parse_config(config, collection)
    path = await save_upload(file)
    try:
        total = await asyncio.to_thread(count_records, path, file.filename, opts.format)
        if opts.max_documents and total:
            total = min(total, opts.max_documents)
        source = file_source(
            path,
            filename=file.filename,
            fmt=opts.format,
            csv_options=opts.csv,
            json_path=opts.json_path,
            total=total,
        )
    except BaseException:
        _remove(path)
        raise
    handle = job_manager.submit(
        kind="load_file",
        collection=opts.collection,
        database=conn.database,
        config={"filename": file.filename, "format": opts.format, "dry_run": opts.dry_run},
        total=total,
        bytes_total=source.bytes_total,
        runner=lambda h: run_load(h, db, opts, source),
    )
    handle.cleanup.append(lambda: _remove(path))
    return await job_response(handle, wait, response)


@router.post(
    "/load/url",
    response_model=Job,
    summary="Load documents from remote file URLs",
    description=(
        "Streams one or more HTTP(S) or public `gs://` files (JSON / JSONL / CSV, optionally "
        "gzipped) through the embedding pipeline without staging them on disk."
    ),
    responses={202: {"model": Job, "description": "Job accepted (when `wait=false`)."}},
)
async def load_url(
    body: LoadUrlRequest,
    db: DbDep,
    conn: ConnectionDep,
    response: Response,
    wait: WaitQuery = False,
) -> Job:
    settings = get_settings()
    if not settings.allow_url_sources:
        raise AppError(403, "url_sources_disabled", "URL sources are disabled on this server.")
    urls = [normalize_url(u) for u in body.urls]
    for url in urls:
        check_url_allowed(url, settings.url_allowed_hosts)
    bytes_total = await asyncio.to_thread(head_sizes, urls)
    source = url_source(
        urls,
        fmt=body.format,
        csv_options=body.csv,
        json_path=body.json_path,
        bytes_total=bytes_total,
    )
    handle = job_manager.submit(
        kind="load_url",
        collection=body.collection,
        database=conn.database,
        config={"urls": urls, "format": body.format, "dry_run": body.dry_run},
        bytes_total=bytes_total,
        runner=lambda h: run_load(h, db, body, source),
    )
    return await job_response(handle, wait, response)


class UrlPreviewRequest(BaseModel):
    url: str = Field(..., description="HTTP(S) or public gs:// URL.")
    format: FileFormat = "auto"
    csv: CsvOptions = Field(default_factory=CsvOptions)
    json_path: str | None = None
    limit: int = Field(50, ge=1, le=1000)


@router.post(
    "/preview/url",
    response_model=PreviewResponse,
    summary="Preview a remote file and detect its fields",
    description="Streams only the first `limit` documents of the remote file.",
)
async def preview_url(body: UrlPreviewRequest) -> PreviewResponse:
    settings = get_settings()
    if not settings.allow_url_sources:
        raise AppError(403, "url_sources_disabled", "URL sources are disabled on this server.")
    url = normalize_url(body.url)
    check_url_allowed(url, settings.url_allowed_hosts)
    source = url_source(
        [url], fmt=body.format, csv_options=body.csv, json_path=body.json_path, timeout=30
    )
    try:
        docs = await source.next_batch(body.limit)
    finally:
        source.close()
    size = await asyncio.to_thread(head_sizes, [url])
    filename = url.rsplit("/", 1)[-1]
    return PreviewResponse(
        filename=filename,
        format=body.format if body.format != "auto" else _guess_format(filename),
        sample=docs[:20],
        fields=[f.model_dump() for f in detect_field_paths(docs)],
        estimated_total=None,
        size_bytes=size,
    )


def _guess_format(filename: str) -> str:
    name = filename.lower().removesuffix(".gz")
    if name.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if name.endswith((".csv", ".tsv")):
        return "csv"
    return "json" if name.endswith(".json") else "auto"
