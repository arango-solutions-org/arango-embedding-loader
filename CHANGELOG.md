# Changelog

All notable changes to Arango Embedding Loader are documented here.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions in the `0.0.x` series are the initial development history of the application.
The Python package, the web UI, and the OpenAPI document are at `0.0.6`.

## [0.0.6] - 2026-10-06

Published image tags are split into three variants. `latest` is the plain image again.

### Changed

- `latest`, `<version>`, `<major>.<minor>`, and `sha-<sha>` are the plain image: local and hosted providers, with no models baked in. Local models download from the Hugging Face Hub on first use.
- `latest-pubmedbert` (and the matching version and sha tags) bake `NeuML/pubmedbert-base-embeddings` at `/opt/models` for deployments that cannot reach the Hub.
- `latest-slim` (and the matching version and sha tags) ships hosted providers only. It omits PyTorch and local models.
- `BAKE_MODELS` defaults to empty, so a plain `docker build` matches `latest`.
- Makefile targets are `build`, `build-pubmedbert`, and `build-slim`. Another model can be baked with `BAKED_MODEL` and `BAKED_SUFFIX`.
- The container runs as UID `10001` so `runAsNonRoot` can verify the user.
- CI runs an offline import smoke test (`--network none`) on every variant. Baked variants also embed offline for each baked model.
- GitHub Actions in the publish workflow are pinned by commit SHA.

### Removed

- The base image's `pip` is removed. The app runs from `/opt/venv`. This clears the fixable Trivy findings that came from pip 25.0.1.
- The temporary `feat/baked-models` CI trigger is removed.

## [0.0.5] - 2026-10-06

### Added

- Security workflow (`.github/workflows/security.yml`) that runs on pull requests, pushes to `main`, weekly, and on demand.
- Semgrep `p/default` over the repository. The job fails on ERROR-severity findings.
- Trivy filesystem scan of `uv.lock`, `package-lock.json`, secrets, and the Dockerfile.
- Trivy scan of the built image (without baked model weights). The job fails on HIGH or CRITICAL vulnerabilities that have a fix, and on HIGH or CRITICAL secrets. Full reports are uploaded as artifacts.
- Scanner images are pinned by version and actions are pinned by commit SHA.

## [0.0.4] - 2026-10-06

Sentence-Transformers models can be baked into the image so the loader works without the Hugging Face Hub.

### Added

- `BAKE_MODELS` build argument. Each listed model is saved to `/opt/models/<org>/<name>` and made world-readable so the non-root `loader` user can read weights written as root.
- The sentence-transformers provider loads `$ST_MODELS_DIR/<model>` when a baked copy exists. A request that sets `provider_options.revision` still fetches that revision from the Hub. Path lookup cannot leave the models directory.
- Makefile targets for `build`, `build-nomodels`, `build-slim`, `smoke-offline`, `check`, and `test-integration`. The Makefile uses Docker when it is installed, otherwise Podman (`CONTAINER` overrides this). Podman builds use `--format docker` so the image `HEALTHCHECK` is kept.
- CI integration job against ArangoDB 3.12 with `--experimental-vector-index`.
- CI publish of a PubMedBERT image and a `-nomodels` image, with an offline smoke test on the baked image before it is pushed. Non-default branches are tagged with the branch name.

### Changed

- README and `.env.example` document `BAKE_MODELS`, `ST_MODELS_DIR`, published tags, Make targets, and `normalize_embeddings`.

## [0.0.3] - 2026-10-05

### Fixed

- The web UI no longer renders a white screen when the app is mounted under a path prefix, such as Bring Your Own Code (BYOC) on the Arango Platform (`/_service/uds/...`).
- Vite builds assets with a relative base (`base: "./"`). API calls, the favicon, and the API docs link use the directory the page was served from. Locally that prefix is empty.

## [0.0.2] - 2026-10-05

### Added

- On the Arango Platform, the loader uses the operator-injected `ARANGO_DEPLOYMENT_ENDPOINT` as the ArangoDB coordinator address when `ARANGO_ENDPOINT` is not set.
- TLS verification is left off for that injected endpoint unless `ARANGO_VERIFY_SSL` is set explicitly, because the certificate comes from the cluster CA.

## [0.0.1] - 2026-10-05

### Added

- GitHub Actions workflow that lints, tests, and publishes the container image to GHCR.

## [0.0.0] - 2026-10-05

Initial application: load JSON, JSONL, and CSV documents into ArangoDB with embeddings stored on the same document.

### Added

- FastAPI service (`/api/v1`) for connection checks, collections, profiles, providers, ingest, and background jobs.
- Ingest from a JSON body, uploaded JSON, JSONL/NDJSON, or CSV/TSV (optionally gzipped), from HTTP(S) and public `gs://` URLs, and from documents already in a collection.
- Pluggable embedding providers: Sentence-Transformers, OpenAI (including Azure and OpenAI-compatible servers), Voyage AI, Cohere, Amazon Bedrock, and Ollama, plus a fake provider for tests. Extra providers can be registered as entry points.
- Named presets (including PubMedBERT) and optional YAML presets via `PRESETS_FILE`.
- Embedding specs that label a source field or a template, write the vector to a target field, and record embedding metadata.
- Batched pipeline with concurrency, retries, and per-document duplicate handling (`error`, `update`, `replace`, `ignore`).
- Optional ArangoDB vector index (cosine, l2, or innerProduct) on ArangoDB 3.12.4 or later.
- Auto-embedding profiles stored in `embedding_loader_profiles`. Loads into a profiled collection embed automatically. An optional watcher embeds documents inserted by other clients.
- Job status and a Server-Sent Events progress stream, with cancel support.
- React GUI (load wizard, collections, jobs) with a typed client generated from the OpenAPI document.
- Swagger UI at `/docs`, committed specs in `docs/openapi.yaml` and `docs/openapi.json`, and an endpoint guide in `docs/API.md`.
- Docker Compose stack (ArangoDB 3.12 with the vector index enabled, plus the loader) and a PubMed example under `examples/pubmed`.
- Unit tests and an integration test against a throw-away ArangoDB database.

[0.0.6]: https://github.com/arango-solutions/arango-embedding-loader/pull/6
[0.0.5]: https://github.com/arango-solutions/arango-embedding-loader/pull/5
[0.0.4]: https://github.com/arango-solutions/arango-embedding-loader/pull/4
[0.0.3]: https://github.com/arango-solutions/arango-embedding-loader/pull/3
[0.0.2]: https://github.com/arango-solutions/arango-embedding-loader/pull/2
[0.0.1]: https://github.com/arango-solutions/arango-embedding-loader/pull/1
[0.0.0]: https://github.com/arango-solutions/arango-embedding-loader/commit/3b23c7c
