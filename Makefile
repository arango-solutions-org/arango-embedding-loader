# Works with Docker or Podman. Uses docker when it is installed, otherwise podman;
# override with `make CONTAINER=podman ...`.
#
#   make build                 image with PubMedBERT baked in (runs with no network access)
#   make build-nomodels        same image, no baked models (models download on first use)
#   make build-slim            hosted providers only (no torch, no local models)
#   make smoke-offline         embed one text with --network none in the baked image
#   make check                 lint + unit tests + frontend checks (what CI runs)
#   make test-integration      starts a throw-away ArangoDB, runs integration tests, removes it

CONTAINER ?= $(shell command -v docker >/dev/null 2>&1 && echo docker || echo podman)
IMAGE     ?= arango-embedding-loader
TAG       ?= dev
BAKE_MODELS ?= NeuML/pubmedbert-base-embeddings

# Podman's default OCI format drops HEALTHCHECK; the docker format keeps it.
BUILD_FLAGS := $(if $(findstring podman,$(CONTAINER)),--format docker,)

TEST_DB_IMAGE    ?= docker.io/library/arangodb:3.12
TEST_DB_NAME     ?= ael-test-arangodb
TEST_DB_PORT     ?= 8530
TEST_DB_PASSWORD ?= test

.PHONY: help build build-nomodels build-slim smoke-offline \
        test lint frontend check test-integration db-up db-down

help:
	@sed -n 's/^#   make/  make/p' Makefile
	@echo "  (container tool: $(CONTAINER))"

build:
	$(CONTAINER) build $(BUILD_FLAGS) --build-arg BAKE_MODELS=$(BAKE_MODELS) -t $(IMAGE):$(TAG) .

build-nomodels:
	$(CONTAINER) build $(BUILD_FLAGS) --build-arg BAKE_MODELS= -t $(IMAGE):$(TAG)-nomodels .

build-slim:
	$(CONTAINER) build $(BUILD_FLAGS) --build-arg EXTRAS=hosted -t $(IMAGE):$(TAG)-slim .

smoke-offline:
	$(CONTAINER) run --rm --network none $(IMAGE):$(TAG) python -c "\
	import asyncio; \
	from app.providers.registry import get_provider; \
	from app.providers.sentence_transformers_provider import baked_model_path; \
	m = '$(firstword $(subst $(comma), ,$(BAKE_MODELS)))'; \
	assert baked_model_path(m), 'not baked: ' + m; \
	p = get_provider('sentence_transformers'); \
	v = asyncio.run(p.embed(['offline smoke test'], model=m, options=p.parse_options(None))); \
	print('ok', m, len(v[0]), 'dims, no network')"

test:
	cd backend && uv run pytest -m "not integration and not local"

lint:
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app

frontend:
	cd frontend && npm ci --no-audit --no-fund && npm run typecheck && npm test && npm run build

check: lint test frontend

db-up:
	$(CONTAINER) run -d --rm --name $(TEST_DB_NAME) -p $(TEST_DB_PORT):8529 \
	    -e ARANGO_ROOT_PASSWORD=$(TEST_DB_PASSWORD) $(TEST_DB_IMAGE) \
	    arangod --server.endpoint tcp://0.0.0.0:8529 --experimental-vector-index
	@until curl -sf -u root:$(TEST_DB_PASSWORD) http://127.0.0.1:$(TEST_DB_PORT)/_api/version >/dev/null; do sleep 2; done
	@curl -s -u root:$(TEST_DB_PASSWORD) http://127.0.0.1:$(TEST_DB_PORT)/_api/version; echo

db-down:
	-$(CONTAINER) rm -f $(TEST_DB_NAME) >/dev/null 2>&1

test-integration: db-up
	cd backend && ARANGO_TEST_ENDPOINT=http://127.0.0.1:$(TEST_DB_PORT) \
	    ARANGO_TEST_PASSWORD=$(TEST_DB_PASSWORD) uv run pytest -m integration; \
	    rc=$$?; $(MAKE) -C .. db-down; exit $$rc

comma := ,
