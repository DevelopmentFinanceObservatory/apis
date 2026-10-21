# DFO APIs — Developer Commands
# Run 'just' or 'just --list' to see all available targets.
#
# Override defaults from the command line:
#   just PORT=9000 dev
#   just SEARCH_GRAPH_BASE=http://127.0.0.1:8080 dev
#   just IMAGE_TAG=v1.2.0 build
#   just DEPLOY_ENV=prod deploy

# Configuration (override any of these from the command line)
IMAGE_NAME := "dfo-apis"
IMAGE_TAG  := "latest"
IMAGE      := IMAGE_NAME + ":" + IMAGE_TAG
DOCKERFILE := "Dockerfile"
ENV_FILE   := ".env"
ENV_SAMPLE := ".env.example"
GCP_PROJECT := "dev-fin-observatory"

# Host port for the dev server and for run-local (default: 8083).
PORT := "8083"

# Graph the engine reads. dfo-dev carries the current V2 surface the engine
# targets; prod (datacommons.one.org) serves the ONE custom data on the legacy surface.
SEARCH_GRAPH_BASE := env_var_or_default("SEARCH_GRAPH_BASE", "https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app")

# Dependency set for every local command. Matches ci.yml's UV_SYNC_ARGS, so a
# green `just check` predicts a green CI run.
SYNC_ARGS := "--frozen --extra engine --extra eval --extra datasets --extra graph"

# Deployment tier. Selects the image tag prefix (dfo-<tier>-<sha>) and the branch
# `deploy` expects to be on. Override for a prod cut:
#   just DEPLOY_ENV=prod deploy
DEPLOY_ENV := "dev"

# Show available commands (default)
_default:
    @just --list --unsorted

# ── Setup & Configuration ────────────────────────────

# First-time setup: create the env file, sync dependencies, install pre-commit hooks
setup: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Syncing dependencies..."
    uv sync {{SYNC_ARGS}}
    echo ""
    # Hooks run `ty`, which resolves imports against .venv — so the extras have to
    # be synced before the hooks are installed, not after.
    echo "Installing pre-commit hooks..."
    uv run pre-commit install
    echo ""
    dir=$(just _spec-kit-dir)
    if [ -z "$dir" ]; then
        echo "No dfo-spec-kit checkout found (DFO_SPEC_KIT_DIR unset, no ../dfo-spec-kit"
        echo "sibling). 'just check' needs one for the identity vector suite -- see"
        echo "CONTRIBUTING.md for how to get one."
        echo ""
    fi
    echo "Setup complete. Run 'just dev' to start the Search API."

# Create the local env file (fetches the Gemini key from Secret Manager, prompts as fallback)
env:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGET="{{ENV_FILE}}"
    SAMPLE="{{ENV_SAMPLE}}"
    if [ ! -f "$SAMPLE" ]; then
        echo "Error: $SAMPLE not found. Cannot create env file."
        exit 1
    fi
    if [ ! -f "$TARGET" ]; then
        echo "Copying $SAMPLE -> $TARGET..."
        cp "$SAMPLE" "$TARGET"
        echo ""
    fi
    current() {
        grep -s "^${1}=" "$TARGET" | head -1 | cut -d= -f2-
    }
    write() {
        # Fill the empty KEY= line the sample ships, or append if there is none.
        local key="$1" value="$2"
        if grep -q "^${key}=$" "$TARGET" 2>/dev/null; then
            sed -i.bak "s|^${key}=$|${key}=${value}|" "$TARGET" && rm -f "${TARGET}.bak"
        else
            echo "${key}=${value}" >> "$TARGET"
        fi
    }
    echo "Checking environment variables..."

    # GEMINI_API_KEY — required to boot the engine. Never blocks: if it cannot be
    # resolved and nothing is typed at the prompt, the line is left empty and the
    # rest of the file is still written.
    if [ -n "$(current GEMINI_API_KEY)" ]; then
        echo "  GEMINI_API_KEY already set."
    else
        key=$(just _gemini-key)
        if [ -z "$key" ]; then
            read -rp "  Gemini API key (enter to skip): " key || true
        fi
        write GEMINI_API_KEY "$key"
        if [ -n "$key" ]; then
            echo "  Set GEMINI_API_KEY."
        else
            echo "  GEMINI_API_KEY left blank — the engine will not boot until it is set."
        fi
    fi

    # Langfuse credentials are only needed for live eval runs, and are not in
    # Secret Manager. Taken from the environment when present, left blank otherwise.
    for key in LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
        if [ -n "$(current "$key")" ]; then
            echo "  $key already set."
        elif [ -n "${!key:-}" ]; then
            write "$key" "${!key}"
            echo "  $key taken from environment."
        else
            echo "  $key left blank (eval runs need it; the test suite does not)."
        fi
    done

    # The sample only carries SEARCH_GRAPH_BASE as a commented override. Write the
    # real default so the target a dev run reads is visible in the file.
    if [ -n "$(current SEARCH_GRAPH_BASE)" ]; then
        echo "  SEARCH_GRAPH_BASE already set."
    else
        echo "SEARCH_GRAPH_BASE={{SEARCH_GRAPH_BASE}}" >> "$TARGET"
        echo "  Set SEARCH_GRAPH_BASE to {{SEARCH_GRAPH_BASE}}."
    fi
    echo ""
    echo "$TARGET is ready. Run 'just dev' to start."

# Resolve a Gemini key: current env > env file > Secret Manager. Prints the key to
# stdout (empty if none); progress and hints go to stderr.
[private]
_gemini-key:
    #!/usr/bin/env bash
    set -euo pipefail
    key="${GEMINI_API_KEY:-}"
    if [ -z "$key" ] && [ -f {{ENV_FILE}} ]; then
        # `|| true`: grep exits 1 when the env file has no GEMINI line — benign, not
        # fatal under set -e/pipefail.
        key=$(grep -s '^GEMINI_API_KEY=' {{ENV_FILE}} | head -1 | cut -d= -f2- || true)
    fi
    if [ -z "$key" ]; then
        echo "  GEMINI_API_KEY not in env or {{ENV_FILE}}; querying Secret Manager ({{GCP_PROJECT}})..." >&2
        if all=$(gcloud secrets list --project={{GCP_PROJECT}} --format="value(name)" 2>&1); then
            name=$(printf '%s\n' "$all" | grep -iE 'gemini|genai|llm' | head -1 || true)
            if [ -n "$name" ]; then
                key=$(gcloud secrets versions access latest --secret="$name" --project={{GCP_PROJECT}} 2>/dev/null || true)
                [ -n "$key" ] && echo "  Got GEMINI_API_KEY from Secret Manager ($name)." >&2
            else
                echo "  No secret matching gemini/genai/llm. Available secrets:" >&2
                printf '%s\n' "$all" | sed 's/^/    /' >&2
            fi
        else
            echo "  gcloud could not list secrets:" >&2
            printf '%s\n' "$all" | sed 's/^/    /' >&2
            echo "  -> run 'gcloud auth login' and confirm access to {{GCP_PROJECT}}, then retry." >&2
        fi
    fi
    printf '%s' "$key"

# ── Development ──────────────────────────────────────

# By default, reads the remote Data Commons graph. To override and point at
# an alternate endpoint (e.g., a local development mirror):
#   just SEARCH_GRAPH_BASE=http://127.0.0.1:8080 dev
#
# Hot-reload the Search API on the host (uvicorn --reload)
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    key=$(just _gemini-key)
    if [ -z "$key" ]; then
        echo "ERROR: no GEMINI_API_KEY (env, {{ENV_FILE}}, or Secret Manager) — see hints above." >&2
        exit 1
    fi
    # uv does not override the process env from --env-file, and config.py seeds the
    # remaining variables from {{ENV_FILE}} itself at import time.
    export GEMINI_API_KEY="$key"
    export SEARCH_GRAPH_BASE="{{SEARCH_GRAPH_BASE}}"
    # entity_resolution.shared.acquire_shared() raises RuntimeError on a cold
    # environment without a cache dir; default it so a cold `just dev` and
    # `just eval-app` can start.
    if [ -z "${RESOLVEKIT_CACHE_DIR:-}" ]; then
        export RESOLVEKIT_CACHE_DIR="{{justfile_directory()}}/runs/dev_resolvekit"
        mkdir -p "$RESOLVEKIT_CACHE_DIR"
    fi
    echo "Search API (reload) → http://localhost:{{PORT}}/api/search/healthz"
    echo "  graph: {{SEARCH_GRAPH_BASE}}"
    uv run {{SYNC_ARGS}} \
        uvicorn --factory dfo_apis.app:create_app --reload --host 127.0.0.1 --port {{PORT}} \
        --log-config log-config.json

# ── Quality ──────────────────────────────────────────

# Resolve the dfo-spec-kit checkout the identity vector suite reads:
# DFO_SPEC_KIT_DIR from the environment when set, otherwise the sibling
# ../dfo-spec-kit next to this checkout. Prints the resolved path, or nothing
# when neither resolves. A git worktree has its own directory, separate from
# the main checkout's, so the sibling fallback only finds a checkout next to
# the worktree itself, not next to the repo it was created from.
[private]
_spec-kit-dir:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "${DFO_SPEC_KIT_DIR:-}" ]; then
        echo "$DFO_SPEC_KIT_DIR"
    elif [ -d "{{justfile_directory()}}/../dfo-spec-kit" ]; then
        echo "{{justfile_directory()}}/../dfo-spec-kit"
    fi

# Run the tests affected by changes since the last run (offline, against
# recorded fixtures). Extra args go to pytest and run exactly what they name.
# Exports DFO_SPEC_KIT_DIR when a dfo-spec-kit checkout resolves (env var, or
# the sibling ../dfo-spec-kit), so the identity vector suite runs alongside the
# rest of the suite; it skips on its own when neither resolves.
test *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    dir=$(just _spec-kit-dir)
    if [ -n "$dir" ]; then
        export DFO_SPEC_KIT_DIR="$dir"
    fi
    # Any ARGS means a targeted run, which stays serial for clean output and a
    # working pdb.
    if [ -n "{{ARGS}}" ]; then
        uv run {{SYNC_ARGS}} pytest -q {{ARGS}}
        exit
    fi
    # A bare `just test` parallelises with -n auto and runs only the tests
    # whose code changed since the last bare run: pytest-testmon records the
    # lines each test executes in .testmondata. testmon reads Python only, so
    # the run covers the whole suite whenever the other inputs differ from the
    # last finished run: every non-Python file in the repo (fixtures, code
    # index, SQL, YAML, uv.lock) and the dfo-spec-kit checkout, compared by
    # content hash in .testmondata-inputs.
    inputs=$(uv run {{SYNC_ARGS}} python - "${DFO_SPEC_KIT_DIR:-}" <<'PY'
    import hashlib, pathlib, subprocess, sys

    digest = hashlib.sha256()

    def add_tree(root, keep):
        listing = subprocess.run(
            ["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True,
        ).stdout.decode()
        for name in sorted(n for n in listing.split("\0") if n and keep(n)):
            digest.update(name.encode() + b"\0")
            path = pathlib.Path(root, name)
            if path.is_file():
                with path.open("rb") as f:
                    digest.update(hashlib.file_digest(f, "sha256").digest())

    add_tree(".", lambda name: not name.endswith(".py"))
    if spec_kit := sys.argv[1]:
        digest.update(spec_kit.encode() + b"\0")
        add_tree(spec_kit, lambda name: True)
    print(digest.hexdigest())
    PY
    )
    if [ -f .testmondata ] && [ "$inputs" = "$(cat .testmondata-inputs 2>/dev/null)" ]; then
        select=--testmon
    else
        echo "just test: non-Python inputs changed since the last run; running the whole suite"
        select=--testmon-noselect
    fi
    status=0
    uv run {{SYNC_ARGS}} pytest -q -n auto "$select" || status=$?
    # Exit 0 or 1 means the run finished, so .testmondata covers these inputs.
    if [ "$status" -le 1 ]; then
        echo "$inputs" > .testmondata-inputs
    fi
    exit "$status"

# Run the whole suite in parallel without testmon's selection.
test-all:
    #!/usr/bin/env bash
    set -euo pipefail
    dir=$(just _spec-kit-dir)
    if [ -n "$dir" ]; then
        export DFO_SPEC_KIT_DIR="$dir"
    fi
    uv run {{SYNC_ARGS}} pytest -q -n auto

# Run the dc_sdmx live suite against a real deployment. Points at the same
# graph as `just dev` (SEARCH_GRAPH_BASE) plus the Data Commons service's
# /core/api prefix, since SdmxClient appends its own /sdmx/v3/... paths to the
# API root. Reads the x-graph-token shared secret from SEARCH_GRAPH_TOKEN in
# the environment or {{ENV_FILE}}. DFO_SDMX_LIVE_REQUIRED makes a missing base
# a failure rather than a silent skip.
sdmx-live *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    token="${SEARCH_GRAPH_TOKEN:-}"
    if [ -z "$token" ] && [ -f {{ENV_FILE}} ]; then
        # `|| true`: grep exits 1 when the env file carries no such line, which
        # is benign under set -e/pipefail.
        token=$(grep -s '^SEARCH_GRAPH_TOKEN=' {{ENV_FILE}} | head -1 | cut -d= -f2- || true)
    fi
    export DFO_SDMX_LIVE_BASE="{{SEARCH_GRAPH_BASE}}/core/api"
    export DFO_SDMX_LIVE_TOKEN="$token"
    export DFO_SDMX_LIVE_REQUIRED=1
    echo "SDMX live → $DFO_SDMX_LIVE_BASE"
    uv run {{SYNC_ARGS}} pytest -q tests/dc_sdmx/test_live.py {{ARGS}}

# Lint and check formatting, exactly as CI does
lint:
    uv run {{SYNC_ARGS}} ruff check .
    uv run {{SYNC_ARGS}} ruff format --check .

# Apply lint fixes and format in place
format:
    uv run {{SYNC_ARGS}} ruff check --fix .
    uv run {{SYNC_ARGS}} ruff format .

# Type-check src/ with ty
typecheck:
    uv run {{SYNC_ARGS}} ty check

# Everything CI runs, in CI's order: lint, typecheck, then the full test
# suite. Requires a dfo-spec-kit checkout (env var, or the sibling
# ../dfo-spec-kit) and fails immediately if none resolves, so the identity
# vector suite always runs, matching CI's DFO_SPEC_KIT_REQUIRED=1. Green here
# means green there.
check:
    #!/usr/bin/env bash
    set -euo pipefail
    dir=$(just _spec-kit-dir)
    if [ -z "$dir" ]; then
        echo "ERROR: no dfo-spec-kit checkout found." >&2
        echo "" >&2
        echo "tests/engine/test_identity_vectors.py needs one to run. Clone it as a" >&2
        echo "sibling of this checkout:" >&2
        echo "  git clone https://github.com/DevelopmentFinanceObservatory/spec-kit.git ../dfo-spec-kit" >&2
        echo "or point DFO_SPEC_KIT_DIR at an existing checkout instead. See" >&2
        echo "CONTRIBUTING.md for details." >&2
        echo "" >&2
        echo "A git worktree needs DFO_SPEC_KIT_DIR set explicitly: the sibling fallback" >&2
        echo "resolves relative to the worktree's own directory, not the main checkout's." >&2
        exit 1
    fi
    just lint
    just typecheck
    # -n auto lives here rather than in [tool.pytest.ini_options] addopts so
    # it never applies to a targeted `just test`, where it would interleave
    # output and break pdb.
    DFO_SPEC_KIT_DIR="$dir" DFO_SPEC_KIT_REQUIRED=1 uv run {{SYNC_ARGS}} pytest -q -n auto

# ── Migrations ────────────────────────────────────────

# Apply pending forward migrations (see .env.example, deploy/postgres.md).
# Forward-only -- fix a bad migration with a new one, never a rollback.
# Needs a .env file (`just env` first); `uv run --env-file` errors without it.
migrate:
    uv run --env-file {{ENV_FILE}} {{SYNC_ARGS}} python -m dataset_store.migrate up

# Apply presentation_store's own pending forward migrations (see
# deploy/postgres.md's presentation section) -- a distinct entrypoint from
# `migrate` above: its own PRESENTATION_STORE_* credentials, its own
# `presentation` database, its own migrations/ directory.
migrate-presentation:
    uv run --env-file {{ENV_FILE}} {{SYNC_ARGS}} python -m presentation_store.migrate up

# Prove the runtime role's connection works. Needs network access to the
# real instance (see deploy/postgres.md) or DATASET_STORE_HOST plus
# DATASET_STORE_RUNTIME_DB_PASSWORD pointed at a local Postgres for a dry
# run.
dataset-store-smoke:
    uv run --env-file {{ENV_FILE}} {{SYNC_ARGS}} python -m dataset_store.smoke

# ── Eval cases & evals ──────────────────────────────

# Validate corpus/*.yaml and corpus/families/*.yaml, and list the cases that cannot gate
corpus-check *ARGS:
    uv run {{SYNC_ARGS}} python scripts/eval_corpus.py check {{ARGS}}

# Print the JSON schema of a case file, generated from search.eval.cases.Case
corpus-schema:
    uv run {{SYNC_ARGS}} python scripts/eval_corpus.py schema

# Trim a sentence panel run (a directory of raw-*.jsonl) into tests/fixtures/sentence_panel
sentence-golden RUN:
    uv run {{SYNC_ARGS}} python scripts/sentence_golden.py {{RUN}}
# Send the sentence panel to the deployed resolver and write raw and digest files (live, ~5 min; pass slice names such as `crs ids`)
sentence-panel *ARGS:
    uv run {{SYNC_ARGS}} python scripts/sentence_panel/run.py {{ARGS}}

# Pass --nightly URL [--summary FILE] to also run the nightly experiment on the app at URL; it exits 0, 1 or 3.
#
# Mirror corpus/*.yaml to the Langfuse datasets. Live; reads credentials from the env file
eval-sync *ARGS: _check-env
    uv run --env-file {{ENV_FILE}} {{SYNC_ARGS}} \
        python -m search.eval.sync {{ARGS}}

# Exits 0 when nothing blocks, 1 on a blocking verdict, 3 on infrastructure failure.
#
# Run the paired gate: score every eligible case on two running apps, head against base
eval-paired HEAD_URL BASE_URL BASE_REF *ARGS:
    uv run {{SYNC_ARGS}} python scripts/eval_paired.py \
        --head-url {{HEAD_URL}} --base-url {{BASE_URL}} --base-ref {{BASE_REF}} {{ARGS}}

# Filter with --case <id> (repeatable) or --file <stem>; --samples N sets the samples per case.
#
# Score eligible cases on one running app and print every failing check per case
eval-run URL *ARGS:
    uv run {{SYNC_ARGS}} python scripts/eval_paired.py --single-url {{URL}} {{ARGS}}

# Reads GEMINI_API_KEY, GRAPH_RELEASE_URI and ORIGIN_CATALOGUE_URI from the environment.
#
# Serve the app with the eval profile: no response cache, no LLM memo, no extraction store
eval-app PORT:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${GEMINI_API_KEY:?set GEMINI_API_KEY in the environment}"
    export SEARCH_RESPONSE_CACHE_ENABLED=false
    export SEARCH_LLM_MEMO_MAX_BYTES=0
    export SEARCH_EXTRACT_STORE=off
    export SEARCH_GRAPH_BASE="{{SEARCH_GRAPH_BASE}}"
    if [ -z "${RESOLVEKIT_CACHE_DIR:-}" ]; then
        export RESOLVEKIT_CACHE_DIR="{{justfile_directory()}}/runs/dev_resolvekit"
        mkdir -p "$RESOLVEKIT_CACHE_DIR"
    fi
    echo "Search API (eval profile) → http://127.0.0.1:{{PORT}}/api/search/healthz"
    uv run {{SYNC_ARGS}} \
        uvicorn --factory dfo_apis.app:create_app --host 127.0.0.1 --port {{PORT}} \
        --log-config log-config.json

# Live; records every extract/bind call for later cell replay (see promptlab/README.md)
#
# Run one live capture over the corpus cases (scripts/ladder.py capture)
ladder-capture *ARGS: _check-env
    uv run --env-file {{ENV_FILE}} {{SYNC_ARGS}} \
        python scripts/ladder.py capture {{ARGS}}

# Pass --mode bind|extract|e2e (default e2e) and --dry-run for an offline check
#
# Run a rung as one cell against a completed capture (scripts/ladder.py run)
ladder-run RUNG *ARGS: _check-env
    uv run --env-file {{ENV_FILE}} {{SYNC_ARGS}} \
        python scripts/ladder.py run {{RUNG}} {{ARGS}}

# Pass --against <ceiling-cell-id> for the noise-band comparison, --rescore to
# re-derive scores offline, --format json for a machine-readable artifact
#
# Render a cell or family's scores as a report (scripts/ladder.py report)
ladder-report *ARGS: _check-env
    uv run --env-file {{ENV_FILE}} {{SYNC_ARGS}} \
        python scripts/ladder.py report {{ARGS}}

# ── Container ────────────────────────────────────────

# Build the image for linux/amd64 (the Cloud Run target).
build:
    @echo "Building {{IMAGE}} from {{DOCKERFILE}} for linux/amd64..."
    docker buildx build --platform linux/amd64 --provenance=false --load --tag {{IMAGE}} -f {{DOCKERFILE}} .

# Run the built image locally with the env file (container 8080 -> the dev PORT)
#
# --read-only --tmpfs /tmp matches Cloud Run's filesystem: without them a
# local run masks a read-only startup failure and rewrites the image's baked
# resolvekit cache back to WAL on the host's writable layer.
run-local: _check-env
    docker run --rm -it --init \
        --read-only --tmpfs /tmp \
        --env-file {{ENV_FILE}} \
        -p {{PORT}}:8080 \
        {{IMAGE}}

# Alias for run-local (familiar shorthand)
run: run-local

# Show branch, remote, built images, and env file status
status:
    #!/usr/bin/env bash
    echo "Git branch:"
    git branch --show-current
    echo ""
    echo "Remote:"
    git remote -v
    echo ""
    echo "Docker images:"
    docker images | grep -E "{{IMAGE_NAME}}|REPOSITORY" || echo "  (none found)"
    echo ""
    echo "Env file:"
    if [ -f "{{ENV_FILE}}" ]; then echo "  ok {{ENV_FILE}}"; else echo "  missing {{ENV_FILE}} (run 'just env')"; fi

# Remove built images and local caches
clean:
    #!/usr/bin/env bash
    docker images --filter "reference={{IMAGE_NAME}}" -q | sort -u | xargs -r docker rmi -f 2>/dev/null || true
    rm -rf .pytest_cache .ruff_cache .testmondata*
    echo "Cleaned."

# ── Deploying ────────────────────────────────────────

# Defaults to the dev tier; for a production cut: just DEPLOY_ENV=prod deploy
#
# Build and tag an immutable dfo-<tier>-<sha> image
deploy: build
    #!/usr/bin/env bash
    set -euo pipefail
    TIER="{{DEPLOY_ENV}}"
    SHA=$(git rev-parse --short HEAD)
    BRANCH=$(git rev-parse --abbrev-ref HEAD)

    # Each tier is cut from one branch: dev from dev, prod from main. A mismatch
    # is usually a mistake, so warn loudly rather than silently tag the wrong tier.
    case "$TIER" in
        dev)  EXPECTED=dev ;;
        prod) EXPECTED=main ;;
        *)    EXPECTED="" ;;
    esac
    if [ -n "$EXPECTED" ] && [ "$BRANCH" != "$EXPECTED" ]; then
        echo "WARNING: DEPLOY_ENV=$TIER expects branch '$EXPECTED' but you are on '$BRANCH'." >&2
    fi

    # A dirty tree is a warning for dev and a hard stop for prod: a prod image must
    # be reproducible from a committed sha.
    if ! git diff --quiet || ! git diff --cached --quiet; then
        if [ "$TIER" = prod ]; then
            echo "ERROR: uncommitted changes present. Commit before cutting a prod image." >&2
            exit 1
        fi
        echo "WARNING: uncommitted changes present — tagging $SHA from a dirty tree." >&2
    fi

    REF="{{IMAGE_NAME}}:dfo-$TIER-$SHA"
    echo "Tagging {{IMAGE}} -> $REF"
    docker tag {{IMAGE}} "$REF"
    echo ""
    echo "Built and tagged: $REF"
    echo "This is a local tag for hand testing only -- it is never pushed."
    echo "The real deploy path is .github/workflows/deploy-dev.yml, which builds,"
    echo "pushes, and points Cloud Run at a fresh image on every push to dev."
    echo "See deploy/README.md for the vars/secrets it needs."

# ── Internal ─────────────────────────────────────────

_check-env:
    #!/usr/bin/env bash
    if [ ! -f "{{ENV_FILE}}" ]; then
        echo "Error: {{ENV_FILE}} not found."
        echo "Run 'just setup' or 'just env' first."
        exit 1
    fi
