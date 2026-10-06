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
  default database.

Profiles are stored in the `embedding_loader_profiles` collection in the same database. Secrets
in `provider_options` are never persisted. Configure API keys for profiles through server
environment variables.

See [`examples/pubmed`](examples/pubmed) for the full PubMedBERT walkthrough.

## Configuration

All settings are environment variables (or `.env`). See [`.env.example`](.env.example).

| Variable | Default | |
|---|---|---|
| `ARANGO_ENDPOINT` / `ARANGO_DATABASE` / `ARANGO_USERNAME` / `ARANGO_PASSWORD` | `http://localhost:8529` / `_system` / `root` / – | Default connection |
| `ALLOW_CONNECTION_OVERRIDE` | `true` | Allow `X-Arango-*` header overrides |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AZURE_OPENAI_ENDPOINT` | – | OpenAI / compatible / Azure |
| `VOYAGE_API_KEY`, `COHERE_API_KEY` | – | Hosted providers |
| `AWS_REGION` (plus the standard AWS credential chain) | – | Bedrock |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama |
| `ST_DEVICE`, `HF_HOME` | auto | Sentence-Transformers device and model cache |
| `ST_MODELS_DIR` | `/opt/models` | Models baked in at build time, loaded from `<dir>/<org>/<name>` |
| `EMBED_CONCURRENCY` / `DEFAULT_BATCH_SIZE` | `4` / `64` | Pipeline tuning |
| `MAX_UPLOAD_MB` | `1024` | Upload limit |
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
