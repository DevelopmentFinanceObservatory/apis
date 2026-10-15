# Build stage: resolve the locked dependency set into a self-contained venv.
#
# uv installs a managed CPython into /opt/uv-python rather than using the base
# image's interpreter, and both /opt/uv-python and the venv are copied to the
# runtime stage at identical paths, so the venv needs no relocation.
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:bookworm-slim AS build

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# uv needs git on PATH to resolve the dfo-spec-kit git dependency, and git's
# own HTTPS transport needs ca-certificates to verify github.com -- neither
# ships in the base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --extra datasets pulls in dataset_store's driver so `dataset_store.smoke`/
# `.migrate` can run from this image via a one-off Cloud Run Job.
#
# Manifests first for layer caching; install deps without the workspace members.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/dfo-data-tools/pyproject.toml packages/dfo-data-tools/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install \
    && uv sync --no-install-workspace --extra engine --extra datasets --extra graph --no-default-groups --frozen

# Then the source; build and install the project non-editable.
COPY . ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra engine --extra datasets --extra graph --no-default-groups --frozen --no-editable

# Builds the dfo_graph pack from this repo's own committed graph export,
# composes it with resolvekit's bundled geo packs into one store, and
# converts every affected SQLite file to rollback-journal mode (see the
# script's own docstring for why: Cloud Run's read-only filesystem cannot
# open a WAL pack). Baking this in means entity_resolution.acquire_shared()
# hits the cache on every cold start. The venv interpreter is called
# directly, not `uv run`, so nothing can re-resolve the locked environment.
ENV RESOLVEKIT_CACHE_DIR=/app/resolvekit-cache
RUN /app/.venv/bin/python scripts/warm_resolvekit_cache.py


# Runtime stage: the interpreter, the venv, and the warmed resolvekit cache.
FROM --platform=linux/amd64 debian:bookworm-slim AS runtime

# ca-certificates is required: every outbound call (the Data Commons graph, the
# Gemini API) is HTTPS and fails certificate verification without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/uv-python /opt/uv-python
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/resolvekit-cache /app/resolvekit-cache
COPY log-config.json /app/log-config.json

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    RESOLVEKIT_CACHE_DIR=/app/resolvekit-cache

RUN useradd --create-home --uid 1000 app
USER app

# The deploy workflow supplies this build argument as github.sha. It is
# baked into the image rather than set on the Cloud Run service because
# dfo-infra owns the service's own environment.
ARG DFO_BUILD_COMMIT=unknown
ENV DFO_BUILD_COMMIT=${DFO_BUILD_COMMIT}

EXPOSE 8080

# Cloud Run restarts crashed containers, so uvicorn runs in the foreground with
# no supervisor. exec keeps it as PID 1 so SIGTERM reaches it for graceful drain.
# The log config writes every log line as JSON for Cloud Logging
# (dfo_platform.cloud_logging).
CMD exec uvicorn --factory dfo_apis.app:create_app --host 0.0.0.0 --port "${PORT}" \
    --log-config /app/log-config.json
