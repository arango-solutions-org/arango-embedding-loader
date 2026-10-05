# PubMed example: PubMedBERT auto-embedding collection

Reference scenario:

- **Data:** JSONL, one record per line: `{"id": 1234, "text": "title + abstract + keywords"}`
- **Model:** [`NeuML/pubmedbert-base-embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings),
  768 dimensions, run locally via sentence-transformers (`pubmedbert` preset)
- **Collection:** set up once so that every document loaded into it gets an `embedding` field
  automatically

| File | Purpose |
|---|---|
| `profile.json` | Profile: embed `text` into `embedding` with `pubmedbert`, `id` becomes `_key`, cosine vector index, watcher on |
| `sample.jsonl` | 120 synthetic records in the customer's format |
| `load.sh` | curl walkthrough: set up the profile, load the files, print status and a search query |

```bash
./examples/pubmed/load.sh                       # sample data
./examples/pubmed/load.sh /data/pm/*.jsonl      # your own files
```

To load remote files, such as the customer's files in Google Cloud Storage, pass their object URLs:

```bash
curl -X POST 'http://localhost:8000/api/v1/load/url?wait=true' \
  -H 'content-type: application/json' \
  -d '{"collection": "pubmed",
       "urls": ["https://storage.googleapis.com/ncbi-arangodb-share/pmdata/<file1>.jsonl",
                "gs://ncbi-arangodb-share/pmdata/<file2>.jsonl"]}'
```

The bucket does not allow public listing, so you need the exact object names. Because the
collection has a profile, the request doesn't need an `embeddings` section.
