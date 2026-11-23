"""Test `deploy_warm.sh` against a local server standing in for the service and the IAM
Credentials API, with a stub `gcloud`."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "deploy_warm.sh"

_DEPLOYER = "dfo-apis-deployer@dev-fin-observatory.iam.gserviceaccount.com"
_MINT_PATH = f"/v1/projects/-/serviceAccounts/{_DEPLOYER}:generateIdToken"
_WARM_PATHS = ("/api/skills", "/api/graph/schema")

# A 503 the script retries: `Retry-After: 0` keeps the retry immediate.
_BUILDING = (503, {"Retry-After": "0"})


class _Server:
    """Answers each path from its scripted statuses in order, repeating the last, and records
    every request."""

    def __init__(self, replies: dict[str, list[tuple[int, dict[str, str]]]]) -> None:
        self.replies = replies
        self.requests: list[tuple[str, dict[str, str], bytes]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                server.requests.append((self.path, dict(self.headers), body))
                queue = server.replies.get(self.path, [(404, {})])
                status, headers = queue.pop(0) if len(queue) > 1 else queue[0]
                payload = (
                    json.dumps({"token": "id-token"}).encode()
                    if self.path == _MINT_PATH and status == 200
                    else b"{}"
                )
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _answer
            do_POST = _answer

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def paths(self) -> list[str]:
        return [path for path, _, _ in self.requests]


@pytest.fixture
def serve() -> Iterator:
    servers: list[_Server] = []

    def start(replies: dict[str, list[tuple[int, dict[str, str]]]]) -> _Server:
        server = _Server(replies)
        threading.Thread(target=server.httpd.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.httpd.shutdown()
        server.httpd.server_close()


def _fake_gcloud(bin_dir: Path, *, url: str, describe_exit: int = 0) -> None:
    """A PATH shim for `gcloud run services describe` (prints `url`) and `gcloud auth
    print-access-token`."""
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "gcloud"
    script.write_text(
        f"""#!/usr/bin/env bash
if [ "$1" = auth ]; then
  echo access-token
  exit 0
fi
if [ "{describe_exit}" != 0 ]; then
  echo "describe failed" >&2
  exit {describe_exit}
fi
echo "{url}"
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _run_warm(
    server: _Server, tmp_path: Path, *, deadline: int = 30, describe_exit: int = 0
) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, url=server.url, describe_exit=describe_exit)
    output_file = tmp_path / "github_output"
    output_file.write_text("")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SERVICE": "dfo-dev-apis",
            "REGION": "northamerica-northeast1",
            "PROJECT_ID": "dev-fin-observatory",
            "DEPLOYER": _DEPLOYER,
            "WARM_DEADLINE_SECONDS": str(deadline),
            "IAM_CREDENTIALS_BASE": server.url,
            "GITHUB_OUTPUT": str(output_file),
        },
    )
    return result, output_file.read_text()


def _ok() -> dict[str, list[tuple[int, dict[str, str]]]]:
    return {_MINT_PATH: [(200, {})]} | {path: [(200, {})] for path in _WARM_PATHS}


def test_every_path_answering_200_warms_with_the_deployers_id_token(serve, tmp_path: Path):
    server = serve(_ok())

    result, output = _run_warm(server, tmp_path)

    assert result.returncode == 0, result.stderr
    assert output == "status=warm\n"
    assert server.paths() == [_MINT_PATH, *_WARM_PATHS]
    _, mint_headers, mint_body = server.requests[0]
    assert mint_headers["Authorization"] == "Bearer access-token"
    assert json.loads(mint_body) == {"audience": server.url, "includeEmail": True}
    for _, headers, _ in server.requests[1:]:
        assert headers["X-Serverless-Authorization"] == "Bearer id-token"


@pytest.mark.parametrize(
    ("replies", "deadline", "status", "annotation"),
    [
        # Slowness past the bound warns: the build runs on for the next request.
        ({"/api/skills": [_BUILDING]}, 1, "slow", "::warning::"),
        ({"/api/skills": [(504, {"Retry-After": "0"})]}, 1, "slow", "::warning::"),
        ({"/api/skills": [(502, {"Retry-After": "0"})]}, 1, "slow", "::warning::"),
        # The grants are not applied yet.
        ({_MINT_PATH: [(403, {})]}, 30, "forbidden", "::warning::"),
        ({"/api/skills": [(403, {})]}, 30, "forbidden", "::warning::"),
        # The service answers an error.
        ({"/api/graph/schema": [(500, {})]}, 30, "error", "::error::"),
        ({"/api/skills": [(404, {})]}, 30, "error", "::error::"),
        # A later refusal ends probing but keeps an earlier error.
        ({"/api/skills": [(500, {})], "/api/graph/schema": [(403, {})]}, 30, "error", "::error::"),
    ],
)
def test_outcomes(serve, tmp_path: Path, replies, deadline, status, annotation):
    server = serve(_ok() | replies)

    result, output = _run_warm(server, tmp_path, deadline=deadline)

    assert result.returncode == 0, result.stderr
    assert output == f"status={status}\n"
    assert annotation in result.stdout


def test_retries_until_the_build_lands(serve, tmp_path: Path):
    server = serve(_ok() | {"/api/graph/schema": [_BUILDING, _BUILDING, (200, {})]})

    result, output = _run_warm(server, tmp_path)

    assert result.returncode == 0, result.stderr
    assert output == "status=warm\n"
    assert server.paths().count("/api/graph/schema") == 3


def test_a_refused_token_calls_no_path(serve, tmp_path: Path):
    server = serve(_ok() | {_MINT_PATH: [(403, {})]})

    _run_warm(server, tmp_path)

    assert server.paths() == [_MINT_PATH]


def test_an_error_on_one_path_still_warms_the_others(serve, tmp_path: Path):
    server = serve(_ok() | {"/api/skills": [(500, {})]})

    _run_warm(server, tmp_path)

    assert server.paths() == [_MINT_PATH, *_WARM_PATHS]


def test_an_unreadable_service_url_warns(serve, tmp_path: Path):
    server = serve(_ok())

    result, output = _run_warm(server, tmp_path, describe_exit=1)

    assert result.returncode == 0, result.stderr
    assert output == "status=unavailable\n"
    assert "::warning::" in result.stdout
    assert server.paths() == []
