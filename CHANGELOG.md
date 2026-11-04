# Changelog

All notable changes to Arango Embedding Loader are documented here.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions in the `0.0.x` series are the initial development history of the application.
The Python package, the web UI, and the OpenAPI document are at `0.0.6`.

## [Unreleased]

On the Arango Platform the app connects as the signed-in user. Users no longer enter a password.

### Changed

- Requests use the token the platform gateway forwards for the signed-in user (`Authorization: bearer`), with that user's ArangoDB permissions. Without one, platform requests get `401 unauthenticated`.
- Load jobs, re-embed jobs and watchers get renewable tokens for the same user from the injected integration sidecar, so they keep running after the request's token expires. Watchers started at boot run as `ARANGO_SERVICE_USER` (default `root`).
- On the platform, TLS is verified against the injected CA (`ARANGO_DEPLOYMENT_CA`) instead of being turned off.
- Off the platform the server exchanges `ARANGO_USERNAME` / `ARANGO_PASSWORD` for a JWT. Every connection now authenticates with a token.
- There is one watcher per collection, whoever started it.
- The header shows the signed-in user and a database picker. The choice is remembered in the browser.
- Texts longer than the model's token limit are no longer cut silently. By default (`long_input: chunk_docs`) they are split into overlapping chunks, and each chunk is stored as its own document. Set `long_input: truncate_end` to keep the old behaviour.
- The provider options `truncate` (Cohere, Bedrock, Ollama) and `truncation` (Voyage) are replaced by `long_input` and no longer listed in the provider options, API schema or GUI. They are still accepted (deprecated): when `long_input` is not set they set it (`END`/`true` → `truncate_end`, `START` → `truncate_start`, `NONE`/`false` → `error`).
- `POST /providers/{name}/embed` runs texts through the same code as a load. It takes the `long_input` settings and returns, per text, the status, `<target>_meta` and vectors (`results`). `vectors` and `count` include every chunk vector.
- Estimated token counts use 3 characters per token by default (was 4), so pieces stay under the real limit. Requests to OpenAI and Voyage are sized to their per-request token limits. If a provider still rejects a text as too long, it is embedded again with a smaller limit (`max_tokens_reduced_from` in its metadata), so a load does not fail on a low estimate.
- URL loads with `long_input: error` are downloaded once to a temporary file, so the pre-scan runs without downloading twice.
- When a text is split into fewer chunks than before, the extra chunk documents are deleted (`result.stale_chunks_removed`).
- The watcher marks a document whose text is too long under `long_input: error` as `status: "error"` and carries on, instead of failing the batch and retrying it forever.
- A chunk document embedded again on its own (a full re-embed) keeps its `parent_key`, `chunk_index` and `chunk_count`, and its other fields are never embedded.
- `MAX_URL_DOWNLOAD_MB` (default 500) caps a URL load saved to disk for the pre-scan.
- Chunk documents are written before their original. If they fail, the original keeps its previous vector and gets `status: "error"`. Loads with `on_duplicate` `ignore`/`error` don't write chunks for existing documents.
- Long words and text without spaces (Chinese, Japanese, base64, minified JSON, long URLs) are cut inside the word when no word boundary is near enough, so estimated counts can truncate and chunk them.
- A full re-embed processes an original before its chunk documents, so an edited original's chunks get vectors of their new text.
- The watcher marks a document `status: "error"` when the provider keeps rejecting its text as too long after every smaller attempt, instead of retrying the batch forever.
- When a watcher batch fails with any error, the watcher checks the provider with a tiny text. If the provider works, each document is embedded on its own, and only those that still fail are marked `status: "error"` (`reason: "embedding failed"`). If it doesn't, the batch is retried later as before. Rate limits, timeouts, connection and auth errors, or every document failing on its own, mark nothing.
- A rate-limit error whose message mentions tokens is no longer taken for an input that is too long, so texts aren't cut smaller for it.
- Re-embeds with `only_missing` skip originals split by `chunk_docs` whose chunks are current (same text, model and `long_input`), even though their vector is `null`. Editing the original re-embeds it. The job's `result.chunked_unchanged` counts the originals skipped.

### Added

- `GET /api/v1/databases` lists the databases the user can access.
- `long_input` on the embedding spec and profile, for every provider, sets what happens to texts longer than the model's token limit: `truncate_end` (the behaviour before), `truncate_start`, `truncate_middle`, `chunk_mean`, `chunk_docs` (the default), `skip` or `error`. There are also `chunk_overlap_tokens` (default 25, rounded to the nearest word), `warn_on_long_input`, and the advanced `max_tokens` and `token_chars`.
- Tokens are counted with the model's own tokenizer for Sentence-Transformers, OpenAI (`tiktoken`, added to the `openai` extra) and Voyage, and estimated for other providers. Cohere, Bedrock-Cohere, Voyage and Ollama use their own truncation options where one matches the mode.
- `<target>_meta` records `input_tokens`, `truncated`, `tokens_dropped` and `chunks` on every document. Job counts include `over_limit` and `chunks`.
- The GUI shows enum provider options as drop-downs.
- `analyze: true` on loads and re-embeds (**Analyze only** in the GUI) reports token lengths and over-limit documents without embedding or writing anything.
- With `long_input: error`, the input is checked for over-limit texts before anything is embedded, so the job fails up front rather than partway through. URL loads skip this check.

### Fixed

- File preview and file upload now work when the app is served under a platform path prefix.

### Removed

- The "ArangoDB connection" panel, and the `X-Arango-Endpoint`, `X-Arango-Username` and `X-Arango-Password` request headers. `X-Arango-Database` remains.
- The `ALLOW_CONNECTION_OVERRIDE` setting.

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
