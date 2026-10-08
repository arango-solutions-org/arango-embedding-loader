"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import collections, connection, ingest, jobs, profiles, providers, system
from app.arango import default_connection
from app.config import get_settings
from app.errors import install_error_handlers
from app.services.jobs import job_manager
from app.services.watcher import watcher_manager

log = logging.getLogger("arango_embedding_loader")

API_PREFIX = "/api/v1"

DESCRIPTION = """
Load NoSQL documents (JSON, JSONL/NDJSON, CSV - optionally gzipped - or a JSON body) into an
ArangoDB collection, **embedding the field(s) you label** with the embedding provider of your
choice and storing each vector **inside the same document**.

### Concepts
* **Embedding spec** - `{source_field | template, target_field, provider | preset, model}`.
  One or more per load.
* **Providers** - pluggable: Sentence-Transformers (local Hugging Face models such as
  `NeuML/pubmedbert-base-embeddings`), OpenAI (and compatible servers), Voyage AI, Cohere,
  Amazon Bedrock, Ollama, plus any third-party plugin. See `GET /providers`.
* **Profiles** - attach an embedding spec to a collection once (`PUT /collections/{name}/profile`);
  every later load into it is embedded automatically, and an optional watcher embeds documents
  inserted by any other client.
* **Jobs** - loads run as background jobs. Pass `?wait=true` for a synchronous response, or
  follow progress with `GET /jobs/{id}` / `GET /jobs/{id}/events` (SSE).

### Connection
On the Arango Platform the app connects as the signed-in user: the gateway forwards their token
and the database endpoint is injected, so there is nothing to configure. Elsewhere the server
signs in with `ARANGO_USERNAME` / `ARANGO_PASSWORD`. Choose the database per request with the
`X-Arango-Database` header (default `ARANGO_DATABASE`); `GET /databases` lists the ones you can
access.
"""

TAGS = [
    {"name": "System", "description": "Health and version."},
    {"name": "Connection", "description": "Verify ArangoDB connectivity."},
    {"name": "Collections", "description": "Browse collections, sample fields, manage indexes."},
    {"name": "Profiles", "description": "Auto-embedding collections (embed on load / watcher)."},
    {"name": "Providers", "description": "Embedding providers, models and presets."},
    {"name": "Ingest", "description": "Preview files and load documents with embeddings."},
    {"name": "Jobs", "description": "Track, stream and cancel load jobs."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.start_watchers_on_startup:
        try:
            started = await watcher_manager.start_all(default_connection(settings))
            if started:
                log.info("Started %d auto-embed watcher(s)", started)
        except Exception as exc:  # noqa: BLE001 - DB may not be reachable yet
            log.warning("Could not start watchers from default database: %s", exc)
    yield
    await watcher_manager.shutdown()
    await job_manager.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Arango Embedding Loader API",
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        license_info={"name": "Apache-2.0"},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    for module in (system, connection, collections, profiles, providers, ingest, jobs):
        app.include_router(module.router, prefix=API_PREFIX)

    static_dir = settings.static_dir
    if static_dir and Path(static_dir).is_dir():
        static = Path(static_dir)
        if (static / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            candidate = (static / path).resolve()
            if path and candidate.is_file() and static.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static / "index.html")

    return app


app = create_app()
