# Single image: FastAPI backend + built React GUI.
#   docker build -t arango-embedding-loader .                       # local + hosted providers
#   docker build --build-arg EXTRAS=hosted -t ael:slim .             # no local models (no torch)

# ---- GUI -------------------------------------------------------------------------------
FROM node:20-slim AS ui
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
COPY docs/openapi.json /src/docs/openapi.json
RUN npm run build

# ---- API -------------------------------------------------------------------------------
FROM python:3.12-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
ARG EXTRAS="local,hosted"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    STATIC_DIR=/app/static \
    HF_HOME=/data/hf

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e; args=""; for e in $(echo "$EXTRAS" | tr ',' ' '); do args="$args --extra $e"; done; \
    uv sync --frozen --no-dev --no-install-project $args
COPY backend/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e; args=""; for e in $(echo "$EXTRAS" | tr ',' ' '); do args="$args --extra $e"; done; \
    uv sync --frozen --no-dev $args
COPY --from=ui /src/frontend/dist /app/static

RUN useradd --create-home --uid 10001 loader && mkdir -p /data/hf && chown -R loader /data
USER loader
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
