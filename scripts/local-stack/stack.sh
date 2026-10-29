#!/usr/bin/env bash
# The local dfo-apis stack: Postgres 18, fake-gcs-server and dfo-apis with the Datasets and
# Presentation surfaces (scripts/local-stack/README.md).
#
#   stack.sh up      start the containers, migrate both stores, publish the release and
#                    catalogue, start dfo-apis on 127.0.0.1:58083 and wait until it answers
#   stack.sh seed    mint the committed readings into the running stores (idempotent)
#   stack.sh smoke   check every seeded reading over HTTP
#   stack.sh down    stop dfo-apis, remove the containers and everything they held
#   stack.sh status  print what is running
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUNS="$ROOT/runs/local-stack"
COMPOSE=(docker compose -f "$HERE/compose.yaml")
PORT=58083
BASE_URL="http://127.0.0.1:$PORT"
PID_FILE="$RUNS/server.pid"
LOG_FILE="$RUNS/server.log"
SYNC_ARGS=(--frozen --extra engine --extra eval --extra datasets --extra graph --group test)

export GRAPH_RELEASE_URI="file://$RUNS/graph-release"
export ORIGIN_CATALOGUE_URI="file://$RUNS/origin-catalogue"
export LOCAL_STACK_BASE_URL="$BASE_URL"
export LOCAL_STACK_SEEDED="$RUNS/seeded.json"

# The release publisher and the seed import the test suite's fixture builders and doubles.
run() {
    (cd "$ROOT" && PYTHONPATH="$ROOT" uv run --env-file "$HERE/local.env" "${SYNC_ARGS[@]}" "$@")
}

server_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

create_buckets() {
    local bucket status count=0
    for bucket in dfo-local-working dfo-local-pinned; do
        status=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
            -H 'content-type: application/json' -d "{\"name\":\"$bucket\"}" \
            "http://127.0.0.1:54443/storage/v1/b?project=dfo-local")
        case "$status" in
            200|409) count=$((count + 1)) ;;
            *) echo "creating bucket $bucket answered $status" >&2; exit 1 ;;
        esac
    done
    [ "$count" -eq 2 ] || { echo "expected 2 buckets, made $count" >&2; exit 1; }
    echo "buckets ready: $count"
}

wait_for_gcs() {
    local tries
    for tries in $(seq 1 30); do
        curl -sf "http://127.0.0.1:54443/storage/v1/b?project=dfo-local" >/dev/null && return 0
        sleep 1
    done
    echo "fake-gcs-server did not answer after $tries tries" >&2
    exit 1
}

start_server() {
    if server_running; then
        echo "dfo-apis already running (pid $(cat "$PID_FILE"))"
        return 0
    fi
    # Job control puts the server in its own process group, so `down` stops uv and uvicorn.
    set -m
    (cd "$ROOT" && PYTHONPATH="$ROOT" exec uv run --env-file "$HERE/local.env" "${SYNC_ARGS[@]}" \
        uvicorn --factory local_app:create_app --app-dir "$HERE" \
        --host 127.0.0.1 --port "$PORT" --log-config log-config.json) \
        >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    set +m
    local tries
    for tries in $(seq 1 120); do
        if ! server_running; then
            echo "dfo-apis exited during startup; last lines of $LOG_FILE:" >&2
            tail -n 40 "$LOG_FILE" >&2
            exit 1
        fi
        if curl -sf "$BASE_URL/api/datasets/healthz" >/dev/null \
            && curl -sf "$BASE_URL/api/presentation/healthz" >/dev/null; then
            echo "dfo-apis ready at $BASE_URL (pid $(cat "$PID_FILE"), log $LOG_FILE)"
            return 0
        fi
        sleep 1
    done
    echo "dfo-apis did not answer healthz after $tries s; see $LOG_FILE" >&2
    exit 1
}

stop_server() {
    if server_running; then
        local pid
        pid=$(cat "$PID_FILE")
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
        local tries
        for tries in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        echo "dfo-apis stopped (pid $pid)"
    fi
    rm -f "$PID_FILE"
}

up() {
    mkdir -p "$RUNS"
    "${COMPOSE[@]}" up -d --wait
    wait_for_gcs
    create_buckets
    run python -m dataset_store.migrate up
    run python -m presentation_store.migrate up
    run python scripts/local-stack/publish_inputs.py
    start_server
}

down() {
    stop_server
    "${COMPOSE[@]}" down -v --remove-orphans
    rm -rf "$RUNS"
}

status() {
    "${COMPOSE[@]}" ps
    if server_running; then
        echo "dfo-apis running at $BASE_URL (pid $(cat "$PID_FILE"))"
    else
        echo "dfo-apis not running"
    fi
}

case "${1:-}" in
    up) up ;;
    seed) run python scripts/local-stack/seed.py ;;
    smoke) run python scripts/local-stack/smoke.py ;;
    down) down ;;
    status) status ;;
    *) echo "usage: $0 up|seed|smoke|down|status" >&2; exit 2 ;;
esac
