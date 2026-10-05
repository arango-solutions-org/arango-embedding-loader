#!/usr/bin/env bash
# PubMed walkthrough: auto-embedding collection with NeuML/pubmedbert-base-embeddings.
#
#   ./examples/pubmed/load.sh                         # uses examples/pubmed/sample.jsonl
#   ./examples/pubmed/load.sh part1.jsonl part2.jsonl # your own files
#   API=http://localhost:8000 COLLECTION=pubmed ./examples/pubmed/load.sh
set -euo pipefail

API=${API:-http://localhost:8000}/api/v1
COLLECTION=${COLLECTION:-pubmed}
HERE=$(cd "$(dirname "$0")" && pwd)
FILES=("$@")
[ ${#FILES[@]} -eq 0 ] && FILES=("$HERE/sample.jsonl")

echo "1) Set up '$COLLECTION' as an auto-embedding collection (pubmedbert preset)"
curl -sf -X PUT "$API/collections/$COLLECTION/profile" \
  -H 'content-type: application/json' -d @"$HERE/profile.json" | python3 -m json.tool | head -20

for f in "${FILES[@]}"; do
  echo "2) Load $f (no embedding config needed - the profile applies)"
  curl -sf -F "file=@$f" "$API/load/file?collection=$COLLECTION&wait=true" \
    | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j["status"], j["counts"], j.get("message") or "")'
done

echo "3) Status"
curl -sf "$API/collections/$COLLECTION/profile" \
  | python3 -c 'import json,sys; j=json.load(sys.stdin); print({k: j[k] for k in ("total_documents","embedded_documents","pending_documents")})'

cat <<AQL

4) Semantic search in ArangoDB (AQL):

   LET q = FIRST(FOR d IN $COLLECTION LIMIT 1 RETURN d.embedding)
   FOR d IN $COLLECTION
     SORT APPROX_NEAR_COSINE(d.embedding, q) DESC
     LIMIT 5
     RETURN { id: d.id, text: SUBSTRING(d.text, 0, 120) }
AQL
