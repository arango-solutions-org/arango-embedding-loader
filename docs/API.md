# Arango Embedding Loader API

The machine-readable Swagger/OpenAPI 3.1 document is [`openapi.yaml`](openapi.yaml)
([`openapi.json`](openapi.json)). A running server also serves:

| URL | What |
|---|---|
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc |
| `http://localhost:8000/openapi.json` | Live OpenAPI document |

After changing endpoints, regenerate the committed files and the GUI's typed client:

```bash
cd backend && uv run python ../scripts/export_openapi.py && cd ../frontend && npm run gen:api
```

## Connection

The server's `.env` provides the default ArangoDB connection. Any request can override it with
these headers:

| Header | Example |
|---|---|
| `X-Arango-Endpoint` | `https://abc123.arangodb.cloud:8529` |
| `X-Arango-Database` | `research` |
| `X-Arango-Username` | `loader` |
| `X-Arango-Password` | `…` |

Set `ALLOW_CONNECTION_OVERRIDE=false` to disable overrides.

## Endpoints

All paths are prefixed with `/api/v1`.

| Tag | Method | Path | Purpose |
|---|---|---|---|
| System | GET | `/health` | Liveness and version |
| Connection | GET | `/connection` | Test the connection; returns the server version and whether vector indexes are supported |
| Collections | GET | `/collections` | List collections with counts and profile flag |
| | POST | `/collections` | Create a collection |
| | GET | `/collections/{name}/sample` | Sample documents and detect field paths |
| | GET | `/collections/{name}/indexes` | List indexes |
| | POST | `/collections/{name}/vector-index` | Create a vector index (cosine / l2 / innerProduct) |
| | POST | `/collections/{name}/embed` | Embed documents already stored in the collection (job) |
| Profiles | GET | `/profiles` | List auto-embed profiles |
| | PUT | `/collections/{name}/profile` | Set up an auto-embedding collection |
| | GET | `/collections/{name}/profile` | Profile plus status: embedded / total / pending, watcher, indexes |
| | DELETE | `/collections/{name}/profile` | Remove the profile (data is kept) |
| | POST | `/collections/{name}/profile/watcher/start` | Start the watcher |
| | POST | `/collections/{name}/profile/watcher/stop` | Stop the watcher |
| Providers | GET | `/providers` | Registered providers, availability and options JSON schema |
| | GET | `/providers/{name}/models` | Models for a provider, with dimensions |
| | POST | `/providers/{name}/embed` | Embed sample texts (connectivity and dimension check) |
| | GET | `/presets` | Named provider/model presets, such as `pubmedbert` |
| Ingest | POST | `/preview` | Upload a file; returns a sample and detected fields |
| | POST | `/preview/url` | Preview a remote file |
| | POST | `/load` | Load documents from the JSON body (job) |
| | POST | `/load/file` | Load from a multipart file upload (job) |
| | POST | `/load/url` | Stream HTTP(S) or `gs://` files (job) |
| Jobs | GET | `/jobs` | Recent jobs |
| | GET | `/jobs/{id}` | Job status and counts |
| | GET | `/jobs/{id}/events` | Server-Sent Events progress stream |
| | DELETE | `/jobs/{id}` | Cancel a running job |

Loads run as background jobs. By default a load returns **202** with the queued job. Add
`?wait=true` to block until the job finishes and get **200** with the final job.

## Embedding spec

Each spec labels one field to embed. A load can include several.

```jsonc
{
  "source_field": "text",              // dot-path; or use "template": "{title}. {abstract}"
  "target_field": "embedding",         // where the vector goes (default <source>_embedding)
  "preset": "pubmedbert",              // or "provider" + "model"
  "provider": "sentence_transformers",
  "model": "NeuML/pubmedbert-base-embeddings",
  "dimensions": null,                  // for models that support output dimensions
  "provider_options": {},              // api_key, base_url, device, ... (see GET /providers)
  "on_missing": "skip"                 // skip | null | error
}
```

When `store_metadata` is true (the default), each document also gets
`<target>_meta: {status, provider, model, dimension, at}`.

## Examples

### JSON body

```bash
curl -X POST 'http://localhost:8000/api/v1/load?wait=true' -H 'content-type: application/json' -d '{
  "collection": "pubmed",
  "key_field": "id",
  "embeddings": [{"source_field": "text", "target_field": "embedding", "preset": "pubmedbert"}],
  "vector_index": {"create": true, "metric": "cosine"},
  "documents": [{"id": 1234, "text": "Title. Abstract... Keywords: oncology"}]
}'
```

### File upload (JSON / JSONL / CSV, gzip OK)

```bash
curl -F file=@part1.jsonl \
     -F 'config={"collection":"pubmed","key_field":"id","embeddings":[{"source_field":"text","preset":"pubmedbert"}]}' \
     'http://localhost:8000/api/v1/load/file?wait=true'
```

### CSV with OpenAI and a combined-field template

```bash
curl -F file=@products.csv \
     -F 'config={"collection":"products","embeddings":[{"template":"{name}: {description}","target_field":"vec","provider":"openai","model":"text-embedding-3-small"}],"csv":{"delimiter":","}}' \
     'http://localhost:8000/api/v1/load/file?wait=true'
```

### Auto-embedding collection (embed on load and watch for inserts)

```bash
curl -X PUT http://localhost:8000/api/v1/collections/pubmed/profile -H 'content-type: application/json' \
     -d @examples/pubmed/profile.json
# After this, loads need no embeddings config:
curl -F file=@part2.jsonl 'http://localhost:8000/api/v1/load/file?collection=pubmed&wait=true'
```

### Embed an existing collection

```bash
curl -X POST 'http://localhost:8000/api/v1/collections/articles/embed?wait=true' \
     -H 'content-type: application/json' \
     -d '{"embeddings":[{"source_field":"abstract","provider":"ollama","model":"nomic-embed-text"}],"only_missing":true}'
```

### Follow progress

```bash
JOB=$(curl -s -F file=@big.jsonl 'http://localhost:8000/api/v1/load/file?collection=pubmed' | jq -r .id)
curl -N http://localhost:8000/api/v1/jobs/$JOB/events      # SSE stream
```

### Query with vector search (AQL)

```aql
FOR d IN pubmed
  SORT APPROX_NEAR_COSINE(d.embedding, @query_vector) DESC
  LIMIT 10
  RETURN { id: d.id, text: d.text }
```
