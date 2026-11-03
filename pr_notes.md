# Configurable handling of inputs longer than the model limit (`long_input`)

Part of [SE-133](https://arangodb.atlassian.net/browse/SE-133).

## Summary

The `sentence_transformers` provider used to cut every text at the model's `max_seq_length` (512 tokens for PubMedBERT) without any warning. On a 22,953-abstract PubMed load, 12.3% of documents went over the limit, and nothing showed it. Hosted providers each handled long inputs differently.

This PR adds a `long_input` setting on the embedding spec and profile. It works the same for every provider:

| `long_input` | Behaviour |
| --- | --- |
| `truncate_end` (default) | Keep the first N tokens. This was the behaviour before, and is now recorded. |
| `truncate_start` | Keep the last N tokens |
| `truncate_middle` | Keep the start and the end |
| `chunk_mean` | Split into overlapping chunks, embed each, store the token-weighted mean as one vector |
| `chunk_docs` | Split into overlapping chunks, write one document per chunk, linked to the original by `parent_key` (see the deviation below) |
| `skip` | Don't embed; `embedding_meta.status = "skipped_too_long"` |
| `error` | Raise `InputTooLongError` (HTTP 422 `input_too_long`); the job fails |

**Options on the spec**
- `chunk_overlap_tokens`: default 25, rounded to the nearest word, at most half the limit.
- `warn_on_long_input`: default `true`. It logs each long text as a warning, or at info level when `false`. With `error` it's always logged as an error.
- `max_tokens` (advanced): defaults to the model's limit.
- `token_chars` (advanced): default 4, the characters per token used for estimates.

**How tokens are counted**

| Provider | Count | Native truncation used for |
| --- | --- | --- |
| Sentence-Transformers | Exact (model's tokenizer) | — (the loader applies every mode) |
| OpenAI | Exact (`tiktoken`, added to the `openai` extra) | — |
| Voyage | Exact (Voyage's tokenizer from the Hugging Face Hub) | `truncate_end` |
| Cohere, Bedrock-Cohere | Estimated | `truncate_end`, `truncate_start` |
| Ollama | Estimated; the limit must be set with `max_tokens` | `truncate_end` |
| Bedrock-Titan | Estimated | — |

Where the provider truncates natively, the whole text is sent, so the cut is exact. `error` turns native truncation off. Every other mode leaves it on, as a backstop in case an estimated chunk is slightly too long. Pieces are cut on whitespace word boundaries.

**`embedding_meta`** records `input_tokens`, `truncated`, `tokens_dropped` and `chunks` on every document, plus `max_tokens`, `long_input` and `token_count` (`exact` or `estimated`). Job counts gain `over_limit` and `chunks`.

**Analyze and pre-scan**
- `analyze: true` on loads and re-embeds (**Analyze only** in the GUI) counts tokens and reports, in `result.analysis`, the over-limit count, the median, p90, p99 and max token counts, and the 20 longest documents. It embeds and writes nothing.
- With `long_input: error`, the input is checked like this before anything is embedded, so a long row near the end fails the job up front rather than after hours of compute.
- URL loads are streamed once, so they skip the pre-scan.

**GUI changes**
- A "Texts longer than the model's limit" section in the model step: mode, overlap, warning toggle, and the advanced max tokens and characters per token.
- Enum provider options now show as drop-downs.
- The job page shows the analysis report and the new counts.

## Acceptance criteria

| Criterion | Status |
| --- | --- |
| Each mode implemented for sentence-transformers | ✅ All 7 modes, with real-tokenizer tests |
| Hosted providers map to native truncation where it exists; the loader applies the mode where it doesn't | ✅ Native mapping for Cohere, Bedrock-Cohere, Voyage and Ollama; the loader applies every other mode, using exact counts (OpenAI, Voyage) or estimates |
| Token counting uses the model's own tokenizer for local models | ✅ |
| `embedding_meta` records truncation for every document | ✅ When `store_metadata` is on, which is the default |
| Job summary reports how many documents were over the limit | ✅ `counts.over_limit`, shown in the GUI |
| GUI exposes the mode and options; Swagger/OpenAPI updated | ✅ |
| Unit tests per mode; integration test with an over-limit document | ✅ |
| README documents the modes and the default | ✅ |

## Deviations from the requirements

- **`chunk_docs` writes chunks to the same collection, not a child collection.** Each chunk is a copy of the original, keyed `<parent _key>_<target>_chunk_<n>`, with `parent_key`, `chunk_index` and `chunk_count` in its `embedding_meta`. The original keeps its fields with `embedding: null` and `status: "chunked"`. The existing vector index covers the chunks with no extra setup.
- **This is NOT a graph.** The loader writes plain document collections. The chunk → original link is a `parent_key` value, not an edge, so there is no edge collection and nothing to traverse.

## Files changed

| File | Lines (+/−) |
| --- | --- |
| `backend/app/providers/long_input.py` (new) | +258 |
| `backend/app/services/pipeline.py` | +353 / −30 |
| `backend/app/services/analysis.py` (new) | +176 |
| `backend/app/providers/base.py` | +63 |
| `backend/app/models/embedding.py` | +53 |
| `backend/app/providers/voyage.py` | +38 / −7 |
| `backend/app/providers/sentence_transformers_provider.py` | +35 |
| `backend/app/providers/openai_provider.py` | +26 / −3 |
| `backend/app/providers/cohere_provider.py` | +24 / −7 |
| `backend/app/providers/bedrock.py` | +21 / −6 |
| `backend/app/providers/ollama.py` | +14 / −1 |
| `backend/app/services/sources.py` | +18 / −2 |
| `backend/app/services/watcher.py` | +15 / −4 |
| `backend/app/api/ingest.py` | +13 / −3 |
| `backend/app/models/load.py` | +7 |
| `backend/app/models/jobs.py` | +4 |
| `backend/pyproject.toml`, `backend/uv.lock` | +3 / −2, +32 |
| `backend/tests/unit/test_long_input.py` (new) | +555 |
| `backend/tests/integration/test_loader.py` | +68 |
| `frontend/src/components/LongInputForm.tsx` (new) | +89 |
| `frontend/src/components/JobProgress.tsx` | +90 / −1 |
| `frontend/src/components/ProviderOptionsForm.tsx` | +31 / −1 |
| `frontend/src/lib/spec.ts`, `spec.test.ts` | +48 / −1, +40 / −1 |
| `frontend/src/pages/LoadWizard.tsx` | +27 / −5 |
| `frontend/src/pages/Collections.tsx` | +1 / −1 |
| `frontend/src/api/schema.d.ts` (generated) | +65 |
| `docs/openapi.json`, `docs/openapi.yaml` (generated) | +94 / −1, +82 |
| `README.md` | +78 |
| `CHANGELOG.md` | +6 |

## Testing

- `make check` passes: ruff, mypy, 74 backend unit tests, and the frontend typecheck, 11 tests and the build.
- There are also 15 real-tokenizer tests (marked `local`), which pass. They cover MiniLM, OpenAI via tiktoken and Voyage via its Hugging Face tokenizer, and check that every piece fits the limit and that no word is cut in half.
- `make test-integration` passes: all 10 tests against ArangoDB 3.12.12, including those that use the PubMedBERT and MiniLM models. The new over-limit test covers `chunk_docs`, the default `truncate_end`, `analyze`, and the `error` pre-scan through the API.

## Open questions

1. **Document store or corpus graph?** The loader only writes document collections. Should its output fit into a corpus graph, such as the one AutoGraph builds?
2. **Should `chunk_docs` use a child collection, and should it be a graph?** The requirement says child collection. This PR keeps chunks in the same collection with `parent_key` (see Deviations). The options:
   - A `<collection>_chunks` collection, which needs its own vector index.
   - An edge collection (chunk → original) so the chunks can be traversed.
3. **Are the defaults OK?**
   - `truncate_end` keeps the old vectors for existing loads, but still drops text from about 12% of PubMed documents.
   - `chunk_overlap_tokens: 25`.
   - `warn_on_long_input: true` writes one warning per over-limit document, about 2,800 on that load.
4. **Hosted limits** come from provider documentation and should be checked:
   - OpenAI: 8,191
   - Voyage: 32,000 (16,000 for `voyage-law-2`)
   - Cohere v3: 512 (128,000 for `embed-v4.0`)
   - Titan: 8,192
   - Ollama has no default, because it depends on the model and the server's `num_ctx`.
5. **Native truncation options are overridden.** Cohere's `truncate`, Voyage's `truncation` and Ollama's `truncate` are now set from `long_input`. Values in `provider_options` are ignored, but they still appear in the provider options form. Should they be hidden?
6. **Estimated counts:** for Cohere, Ollama and Titan, a chunk can still be slightly over the real limit. Where native truncation exists it trims the overflow; Titan has none and could reject the request. Should Cohere's tokenizer be used, even though it needs an API call or a download?
7. **The Voyage tokenizer** is downloaded from the Hugging Face Hub on first use. If that fails (no network), the loader falls back to estimates and logs a warning. Is that acceptable for air-gapped deployments?
8. **URL loads and the pre-scan:** should they download twice, or spool to disk first? Today they skip it.
9. **Leftover chunks:** if a text is re-embedded into fewer chunks, the extra chunk documents from the earlier run stay behind. Should they be deleted?
10. **`only_missing` re-embed** re-processes `chunk_docs` originals every time, because their vector is `null`. Should it ignore `status: chunked`?
11. **`error` and the watcher:** the watcher doesn't stop. It retries with backoff and shows the error in its state, and it has no pre-scan. Should it stop instead?
12. **Not covered:** query embedding (SE-134) and the `/providers/{name}/embed` test endpoint don't use `long_input`.
