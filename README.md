# Arango Embedding Loader

Load NoSQL documents into ArangoDB with embeddings: **label the field(s) to embed**, choose any
embedding provider and model, and each vector is stored **inside the same document** in your
collection. You can use it through the REST API (scripts, pipelines, curl) or the web GUI.

- **Inputs:** JSON body, JSON arrays, JSONL/NDJSON, CSV/TSV (optionally gzipped) uploaded or streamed
  from HTTP(S) / public `gs://` URLs, plus documents already stored in a collection
- **Providers (pluggable):** Sentence-Transformers (any local Hugging Face model, such as
  `NeuML/pubmedbert-base-embeddings`), OpenAI (also Azure and OpenAI-compatible servers such as
  vLLM, TEI and LM Studio), Voyage AI, Cohere, Amazon Bedrock and Ollama. Third-party providers can
  be added as plugins.
- **Auto-embedding collections:** attach a profile to a collection once, and every later load into
  it is embedded automatically. An optional watcher also embeds documents inserted by any other
  client.
- **Vector index:** optionally creates an ArangoDB vector index (cosine / l2 / innerProduct)
- **Swagger:** `/docs`, plus a committed spec in [`docs/openapi.yaml`](docs/openapi.yaml) and an
  endpoint guide in [`docs/API.md`](docs/API.md)

```
backend/   Python 3.12 · uv · FastAPI · python-arango
frontend/  TypeScript · React 19 · Vite · Tailwind v4 · TanStack Query (typed client generated from OpenAPI)
docs/      openapi.yaml / openapi.json (Swagger), API.md
examples/  pubmed/ - PubMedBERT auto-embedding walkthrough
```

## Quick start (Docker)

```bash
cp .env.example .env              # set ARANGO_PASSWORD (and any provider API keys)
docker compose up --build         # ArangoDB 3.12 (vector index enabled) + loader
```

- GUI: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ArangoDB web UI: http://localhost:8530

The image includes sentence-transformers with CPU PyTorch, so local models such as PubMedBERT work
out of the box. Downloaded models are cached in the `hf_cache` volume. For a smaller image without
local models, set `APP_EXTRAS=hosted`.

### Image variants

CI publishes three variants of the image to GHCR:

| Tags | Contents | Use when |
|---|---|---|
| `latest`, `<version>`, `<major>.<minor>`, `sha-<sha>` | Local and hosted providers. No baked models: local models download from the Hugging Face Hub on first use. | The deployment can reach the Hub, or only uses hosted providers. |
| `latest-pubmedbert`, `<version>-pubmedbert`, `sha-<sha>-pubmedbert` | As `latest`, with `NeuML/pubmedbert-base-embeddings` baked in at `/opt/models`. | The deployment cannot reach the Hub (air-gapped or egress-restricted, e.g. BYOC on the Arango Platform) and needs PubMedBERT. |
| `latest-slim`, `<version>-slim`, `sha-<sha>-slim` | Hosted providers only (OpenAI-compatible, Voyage, Cohere, Bedrock, Ollama). No PyTorch, no local models. | Embeddings come from an API, and image size matters. |

On a deployment without Hub access, use a tag with the model baked in. On `latest`, a local model
fails at the first embedding because it tries to download.

Baked models load from disk with no network access. Other local models still download on first
use. To bake models into your own build, pass a comma-separated list of Hugging Face ids:

```bash
docker build --build-arg BAKE_MODELS=NeuML/pubmedbert-base-embeddings .
docker build --build-arg BAKE_MODELS=NeuML/pubmedbert-base-embeddings,BAAI/bge-base-en-v1.5 .
```

A request that sets `provider_options.revision` bypasses the baked copy and fetches that revision
from the Hub.

Sentence-Transformers returns unnormalized vectors by default. To compare against unit-length
vectors produced elsewhere, enable **Normalize Embeddings** under *Sentence-Transformers (local)
options* in the GUI, or set `"provider_options": {"normalize_embeddings": true}` in the spec or
profile.

### Make targets

The Makefile works with Docker or Podman. It uses `docker` when installed, otherwise `podman`.
Override it with `make CONTAINER=podman …`.

| Target | |
|---|---|
| `make build` / `build-pubmedbert` / `build-slim` | `latest` / `latest-pubmedbert` / `latest-slim` equivalents. Bake another model with `make build-pubmedbert BAKED_MODEL=<hf-id> BAKED_SUFFIX=<suffix>` |
| `make smoke-offline` | Embed one text in the baked-model image with `--network none` |
| `make check` | Lint, type-check, unit tests and frontend checks |
| `make test-integration` | Start a throw-away ArangoDB 3.12, run the integration tests, remove it |

## Local development

```bash
docker compose up -d arangodb                      # database only, on :8530

cd backend
uv sync --all-extras                               # or: --extra local / --extra openai / ...
uv run uvicorn app.main:app --reload               # API on :8000

cd ../frontend
npm install
npm run dev                                        # GUI on :5173 (proxies /api to :8000)
```

| Command | What |
|---|---|
| `uv run pytest` | Unit tests (no database needed) |
| `uv run pytest -m integration` | Integration tests against the compose ArangoDB (throw-away database) |
| `uv run ruff check . && uv run mypy app` | Lint and type-check |
| `uv run python ../scripts/export_openapi.py` | Regenerate `docs/openapi.{yaml,json}` |
| `npm run gen:api` | Regenerate the GUI's typed API client from `docs/openapi.json` |
| `npm run typecheck && npm test && npm run build` | Frontend checks |

## How it works

1. **Label fields.** An embedding spec names a `source_field` (dot-path) or a `template` that
   combines several fields, the `target_field` that receives the vector, and a `preset`, or a
   `provider` plus `model`.
2. **Embed.** Documents stream through the pipeline in batches. Texts go to the provider, which
   handles batching, concurrency and retries with backoff, and each vector is written into the
   document under `target_field`.
3. **Write.** Documents are written with `insert_many` (`on_duplicate`: error / update / replace /
   ignore). An error in one document is reported without stopping the batch.
4. **Index.** Optionally, a vector index is built once the vectors exist.

Each embedded document looks like this:

```json
{
  "_key": "1234",
  "id": 1234,
  "text": "title + abstract + keywords",
  "embedding": [0.0123, -0.0456, "… 768 floats"],
  "embedding_meta": {"status": "embedded", "provider": "sentence_transformers",
                     "model": "NeuML/pubmedbert-base-embeddings", "dimension": 768, "at": "…"}
}
```

### Texts longer than the model's limit

Every embedding model reads a fixed number of tokens: 512 for PubMedBERT, 256 for MiniLM, 8,191
for OpenAI's models. The loader counts each text's tokens and handles texts over the limit
according to `long_input` on the embedding spec or profile. In the GUI, it's under **Texts longer
than the model's limit** in the model step.

| `long_input` | What happens to a text over the limit |
| --- | --- |
| `truncate_end` | Keep the first tokens. This was the behaviour before this setting existed. |
| `truncate_start` | Keep the last tokens. |
| `truncate_middle` | Keep the start and the end, joined by a line break. |
| `chunk_mean` | Split into overlapping chunks, embed each, store the token-weighted mean as one vector. |
| `chunk_docs` (default) | Split into overlapping chunks and store each as its own document (see below). No text is lost. |
| `skip` | Don't embed. The document is stored with `<target>_meta.status: "skipped_too_long"`. |
| `error` | Fail the job. See [Checking lengths before a load](#checking-lengths-before-a-load). |

These settings are the same everywhere: on embedding specs and profiles, on the provider test
endpoint (`POST /providers/{name}/embed`, which runs texts through the same code as a load and
returns the same metadata), and in the GUI.

Options on the spec:

| Option | Default | Meaning |
| --- | --- | --- |
| `chunk_overlap_tokens` | `25` | Tokens shared by consecutive chunks, rounded to the nearest word. Values above half the limit are capped at half the limit (256 for a 512-token model), so each chunk starts at least half a chunk after the previous one. |
| `warn_on_long_input` | `true` | Log each long text as a warning. When `false`, they're logged at info level. With `error` they're always logged as errors. |
| `max_tokens` | model's limit | Advanced. The token limit to apply. Needed when the loader doesn't know the model's limit, e.g. Ollama. |
| `token_chars` | `3` | Advanced. Characters per token, used to estimate counts for models without a local tokenizer. English averages about 4, so 3 over-counts on purpose and pieces stay under the real limit. |

**How tokens are counted.**

| Provider | Count | Modes handed to the provider's API (the loader applies all others) |
| --- | --- | --- |
| Sentence-Transformers | Exact, with the model's own tokenizer | None |
| OpenAI | Exact, with `tiktoken` (estimated for other OpenAI-compatible models, or without network to fetch the encoding) | None (the API has no truncation option) |
| Voyage | Exact, with Voyage's tokenizer from the Hugging Face Hub (estimated without network) | `truncate_end` |
| Cohere, Bedrock (Cohere models) | Estimated | `truncate_end`, `truncate_start` |
| Ollama | Estimated; the limit must be set with `max_tokens` | `truncate_end` |
| Bedrock (Titan) | Estimated | None (the API has no truncation option) |

Only `truncate_end` and `truncate_start` can be handed over, and only to APIs with a matching
option (Cohere's `truncate: END/START`, Voyage's and Ollama's on/off truncation). No API can
truncate the middle or chunk. When the API truncates, the loader sends the whole text and the API
cuts it with its own exact count, which matters where the loader's count is an estimate. `error`
turns the API's truncation off; every other mode leaves it on as a backstop. Pieces are cut on
word boundaries, unless the nearest one would drop more than half of what fits. A single word
longer than that (text without spaces such as Chinese or Japanese, base64, minified JSON, long
URLs) is then cut inside the word; with estimated counts, by characters (about `token_chars` per
token).

The provider options `truncate` (Cohere, Bedrock, Ollama) and `truncation` (Voyage)
are replaced by `long_input` and no longer listed. They are still accepted: when `long_input` is
not set, `END`/`true` becomes `truncate_end`, `START` becomes `truncate_start` and `NONE`/`false`
becomes `error`, with a deprecation warning in the log. When both are set, `long_input` wins.

**Texts the provider rejects anyway.** Estimated counts can be low (code, numbers, non-Latin
scripts). Requests are sized to the provider's per-request token limit where it has one (OpenAI,
Voyage). If the provider still rejects a request as too long, the loader sends each text on its
own, and a text the provider rejects is planned again against 70% of the previous limit, up to six
times, until it fits. A rate limit is never taken for a length error, even when its message
mentions tokens. Its `<target>_meta.max_tokens` then shows the limit used and
`max_tokens_reduced_from` the original one. With `error`, the rejection fails the job. If the
provider still rejects the text after six attempts, the load fails; the watcher instead marks
that document `status: "error"` and carries on.

**What's recorded.** Every document's `<target>_meta` (for example `embedding_meta`) records
`input_tokens`, `truncated` (`true`/`false`), `tokens_dropped` and `chunks`, plus `max_tokens`,
`long_input` and `token_count` (`exact` or `estimated`). The job's `counts.over_limit` shows how
many documents went over the limit. Each long text is also logged with its key, token count, limit
and what was done.

**`chunk_docs`.** The original document keeps its fields, gets `<target>: null` and
`<target>_meta.status: "chunked"`. Each chunk is written to the **same collection** as a copy of
the original. The copy has the chunk's text in `source_field` (or in `<target>_text` for templates)
and its own vector. It's keyed `<parent _key>_<target>_chunk_<n>`, and its `<target>_meta` holds
`parent_key`, `chunk_index` and `chunk_count`. This is a plain document collection, not a graph:
the link is the `parent_key` value, not an edge. A vector search can return the original:

```aql
FOR c IN pubmed
  SORT APPROX_NEAR_COSINE(c.embedding, @q) DESC LIMIT 10
  RETURN DOCUMENT(pubmed, c.embedding_meta.parent_key || c._key)
```

Chunks are written with overwrite, so re-running a load or re-embed replaces them. A chunk
document's other fields are never embedded, and a chunk embedded again on its own (a full
re-embed) keeps its `parent_key`, `chunk_index` and `chunk_count`. In a full re-embed
(`only_missing: false`), an original is processed before its chunk documents: if it is split
again, its chunks are rewritten from the current text, and the old chunk documents are not
embedded again on their own. If a text is
split into fewer chunks than before, the extra chunk documents from the earlier split are deleted
(`result.stale_chunks_removed`). Only documents that are certainly those chunks are deleted: the
key is exactly `<parent _key>_<target>_chunk_<n>` past the new count, and the document's
`<target>_meta` names the same `parent_key` and `chunk_index`. If an original is no longer split
at all (it got shorter, or the mode changed), its old chunks are left for you to remove.

Chunks are written before their original. If a chunk can't be written (for example, a chunk key
would exceed ArangoDB's 254-character key limit, as chunk keys are about 20 characters longer than
the original's), that document's chunks are removed and the document is stored with its previous
vector, if any, and `<target>_meta.status: "error"` (`reason: "chunk write failed"`). The job
reports the error, and a later re-embed retries it. Loads with `on_duplicate: "ignore"` or
`"error"` don't write chunks for documents that already exist. If the same key appears twice in
one load, only the copy that is stored writes chunks.

Each chunk is a copy of its original, so a **unique index** on any copied field makes chunking
fail for that collection (the chunks collide with each other and with the original). Use a
different `long_input` mode for such collections.

A re-embed with `only_missing` (the default) leaves a chunked original alone, even though its
vector is `null`, as long as its chunks are current: same text, model and `long_input`. The
original's `<target>_meta.text_hash` records the text it was split from. If you edit the original,
for example after splitting it into chunks of your own, the next re-embed embeds it again. Deleting
its `<target>_meta` does the same. Removing the old chunk documents is up to you. The job's
`result.chunked_unchanged` counts the originals it left alone.

#### Checking lengths before a load

`"analyze": true` on a load or re-embed (**Analyze only** in the GUI) counts tokens without
embedding or writing anything. `result.analysis` reports, for each field: the limit, how many
documents are over it and by how many tokens, the median, p90, p99 and max token counts, and the
20 longest over-limit documents. Use it to choose `long_input` before a long run.

With `long_input: error`, the loader runs this check automatically before embedding anything. If
any text is too long, the job fails straight away. The error log and job message name the longest
documents, and `result.analysis` has the full report. Without this, a long row near the end would
fail the job after hours of embedding. Uploaded files, JSON bodies and re-embeds are read twice
for this. URL loads are downloaded once to a temporary file (in `UPLOAD_DIR`, or the system temp
directory), which is read twice and deleted when the job ends (`result.spooled: true`). Downloads
over `MAX_URL_DOWNLOAD_MB` (500 by default) fail the job; other modes stream URLs without saving
them.

### Auto-embedding collections

```bash
curl -X PUT localhost:8000/api/v1/collections/pubmed/profile \
     -H 'content-type: application/json' -d @examples/pubmed/profile.json
curl -F file=@part1.jsonl 'localhost:8000/api/v1/load/file?collection=pubmed&wait=true'
```

ArangoDB has no triggers and doesn't call models itself, so the loader provides this feature:

- **On load.** Any load into a collection with a profile uses the profile's specs, key field and
  vector index settings. Labels in the request take precedence over the profile.
- **Watcher** (`auto_watch: true`). A background task finds documents with no
  `<target>_meta.status`, using a persistent index, and embeds them. This covers documents written by
  `arangoimport`, drivers or the web UI. Watchers restart with the server for profiles in the
  default database. A document the watcher can't embed (a text too long under
  `long_input: error`, a text the provider keeps rejecting as too long, or a missing source
  under `on_missing: error`) is marked
  `<target>_meta.status: "error"` and logged, and the watcher carries on with the others. It
  doesn't see later edits to documents it has already processed; run a re-embed for those.
- **When a batch fails.** The watcher embeds documents in batches. If a batch fails with any
  error (provider errors are retried first), the watcher checks that each provider still embeds
  a tiny text. If it doesn't, the provider is treated as down: nothing is marked, and the batch
  is retried later with backoff (up to 5 minutes). If it does, each document is embedded on its
  own. One that still fails is marked `status: "error"` (`reason: "embedding failed"`, with the
  message in `error`) and the others are stored, so one bad document can't hold up the queue.
  Good documents aren't marked for the provider's problems. Rate limits, timeouts, connection
  and auth errors (in the batch or in any single document) mean the provider is the problem:
  nothing is marked and the batch is retried later. The same goes when every document of a
  batch fails on its own. A document marked this way has no vector, so a re-embed with
  `only_missing` tries it again.

Profiles are stored in the `embedding_loader_profiles` collection in the same database. Secrets
in `provider_options` are never persisted. Configure API keys for profiles through server
environment variables.

See [`examples/pubmed`](examples/pubmed) for the full PubMedBERT walkthrough.

## Configuration

All settings are environment variables (or `.env`). See [`.env.example`](.env.example).

| Variable | Default | |
|---|---|---|
| `ARANGO_ENDPOINT` / `ARANGO_DATABASE` / `ARANGO_USERNAME` / `ARANGO_PASSWORD` | `http://localhost:8529` / `_system` / `root` / – | Connection off the platform. On the Arango Platform the endpoint is injected and users are signed in by the gateway (see [Connection](docs/API.md#connection)) |
| `ARANGO_VERIFY_SSL` | `true` (platform: the injected CA) | `true`, `false`, or a CA bundle path |
| `ARANGO_SERVICE_USER` | `root` | Platform only: the user for watchers started at boot |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AZURE_OPENAI_ENDPOINT` | – | OpenAI / compatible / Azure |
| `VOYAGE_API_KEY`, `COHERE_API_KEY` | – | Hosted providers |
| `AWS_REGION` (plus the standard AWS credential chain) | – | Bedrock |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama |
| `ST_DEVICE`, `HF_HOME` | auto | Sentence-Transformers device and model cache |
| `ST_MODELS_DIR` | `/opt/models` | Models baked in at build time, loaded from `<dir>/<org>/<name>` |
| `EMBED_CONCURRENCY` / `DEFAULT_BATCH_SIZE` | `4` / `64` | Pipeline tuning |
| `MAX_UPLOAD_MB` | `1024` | Upload limit |
| `MAX_URL_DOWNLOAD_MB` | `500` | Limit on a URL load saved to disk for the `long_input: error` pre-scan |
| `ALLOW_URL_SOURCES` / `URL_ALLOWED_HOSTS` | `true` / any | Restrict `POST /load/url` (for example `["storage.googleapis.com"]`) |
| `PRESETS_FILE` | – | YAML file with extra presets |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | For a separately hosted GUI |

The vector index needs **ArangoDB 3.12.4 or later** started with `--experimental-vector-index`.
The compose file already sets this.

## Adding an embedding provider

Subclass `EmbeddingProvider` and implement `embed_batch`:

```python
from app.providers.base import EmbeddingProvider, ProviderOptions

class MyOptions(ProviderOptions):
    endpoint: str = "http://my-embedder:8080"

class MyProvider(EmbeddingProvider):
    name = "my_provider"
    label = "My embedder"
    Options = MyOptions
    default_model = "my-model"
    max_batch_size = 64

    async def embed_batch(self, texts, *, model, options, dimensions, input_type):
        ...  # return one list[float] per text
```

Register it in one of two ways:

- Built-in: add it to `_register_builtins()` in `backend/app/providers/registry.py`.
- External package: declare an entry point, with no change to this repo:

  ```toml
  [project.entry-points."arango_embedding_loader.providers"]
  my_provider = "my_package.provider:MyProvider"
  ```

The provider's `Options` model drives validation, the Swagger schema and the GUI's options form.
