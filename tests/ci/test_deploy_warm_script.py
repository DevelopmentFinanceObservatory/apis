"""Test `deploy_warm.sh` against a local server standing in for the service and the IAM
Credentials API, with a stub `gcloud`."""

from __future__ import annotations

import base64
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
_WARM_PATHS = (
    "/api/skills",
    "/api/graph/schema",
    "/api/graph/schema/terms/un_system",
    "/api/search/stated-scope",
)
_UN_TERMS = "/api/graph/schema/terms/un_system"
_STATED_SCOPE = "/api/search/stated-scope"
_REVISION = "dfo-dev-apis-00002-bbb"

# A 503 the script retries: `Retry-After: 0` keeps the retry immediate.
_BUILDING = (503, {"Retry-After": "0"})


def _id_token(audience: str) -> str:
    """An unsigned JWT whose payload carries `audience` as its `aud` claim."""
    payload = base64.urlsafe_b64encode(json.dumps({"aud": audience}).encode()).rstrip(b"=")
    return f"eyJhbGciOiJSUzI1NiJ9.{payload.decode()}.sig"


class _Server:
    """Answers each path from its scripted statuses in order, repeating the last, and records
    every request. A 200 from the mint path carries an ID token for the requested audience,
    or for `audience` when set."""

    def __init__(self, replies: dict[str, list[tuple[int, dict[str, str]]]]) -> None:
        self.replies = replies
        self.requests: list[tuple[str, dict[str, str], bytes]] = []
        self.bodies: dict[str, bytes] = {}
        self.audience: str | None = None
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                server.requests.append((self.path, dict(self.headers), body))
                queue = server.replies.get(self.path, [(404, {})])
                status, headers = queue.pop(0) if len(queue) > 1 else queue[0]
                if self.path == _MINT_PATH and status == 200:
                    audience = server.audience or json.loads(body)["audience"]
                    payload = json.dumps({"token": _id_token(audience)}).encode()
                else:
                    payload = server.bodies.get(self.path, b"{}")
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


def _fake_gcloud(
    bin_dir: Path,
    *,
    url: str,
    tagged_url: str,
    tag_revision: str | None,
    describe_exit: int = 0,
) -> None:
    """A PATH shim for `gcloud auth print-access-token` and `gcloud run services describe`,
    which prints the service as JSON: its URL, the serving revision, and a traffic entry tagged
    `candidate` on `tag_revision` at `tagged_url`, or none when `tag_revision` is None."""
    traffic: list[dict] = [{"percent": 100, "revisionName": "dfo-dev-apis-00001-aaa"}]
    if tag_revision is not None:
        traffic.append({"revisionName": tag_revision, "tag": "candidate", "url": tagged_url})
    described = json.dumps({"status": {"url": url, "traffic": traffic}})
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
for arg in "$@"; do
  if [ "$arg" = --format=json ]; then
    echo '{described}'
    exit 0
  fi
done
echo "{url}"
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _run_warm(
    server: _Server,
    tmp_path: Path,
    *,
    deadline: int = 30,
    describe_exit: int = 0,
    tagged_url: str | None = None,
    tag_revision: str | None = _REVISION,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, str]:
    """Runs the script against `server` as the service and the IAM Credentials API. The
    candidate answers at `tagged_url`, which defaults to `server` too."""
    bin_dir = tmp_path / "bin"
    _fake_gcloud(
        bin_dir,
        url=server.url,
        tagged_url=tagged_url or server.url,
        tag_revision=tag_revision,
        describe_exit=describe_exit,
    )
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
            "REVISION": _REVISION,
            "PROBE_TAG": "candidate",
            "ROUTE_DEADLINE_SECONDS": str(deadline),
            "IAM_CREDENTIALS_BASE": server.url,
            "GITHUB_OUTPUT": str(output_file),
            **(extra_env or {}),
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
        assert headers["X-Serverless-Authorization"] == f"Bearer {_id_token(server.url)}"


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


def test_a_tagged_candidate_is_probed_at_its_own_url_with_a_token_for_the_service_url(
    serve, tmp_path: Path
):
    service = serve({_MINT_PATH: [(200, {})]})
    candidate = serve({path: [(200, {})] for path in _WARM_PATHS})

    result, output = _run_warm(service, tmp_path, tagged_url=candidate.url)

    assert result.returncode == 0, result.stderr
    assert output == "status=warm\n"
    assert service.paths() == [_MINT_PATH]
    assert json.loads(service.requests[0][2]) == {"audience": service.url, "includeEmail": True}
    assert candidate.paths() == list(_WARM_PATHS)
    _, headers, body = candidate.requests[-1]
    assert headers["X-Serverless-Authorization"] == f"Bearer {_id_token(service.url)}"
    assert json.loads(body)["texts"]
    assert "ID token audience: the service URL." in result.stdout


@pytest.mark.parametrize("required", ["false", "true"])
@pytest.mark.parametrize(
    "tag_revision", [None, "dfo-dev-apis-00003-ccc"], ids=["no-tag", "tag-on-another-revision"]
)
def test_a_revision_without_its_tagged_url_is_an_error_and_nothing_is_probed(
    serve, tmp_path: Path, tag_revision, required
):
    """The service URL reaches the serving revision, so probing it proves nothing about the
    candidate."""
    server = serve(_ok())

    result, output = _run_warm(
        server, tmp_path, tag_revision=tag_revision, extra_env={"REQUIRE_PROBE": required}
    )

    assert output == "status=error\n"
    assert "::error::" in result.stdout
    assert server.paths() == []


def test_each_route_has_its_own_deadline(serve, tmp_path: Path):
    """A route slow to its deadline leaves the next route its full budget."""
    server = serve(
        _ok()
        | {"/api/skills": [(503, {"Retry-After": "1"})]}
        | {_STATED_SCOPE: [(503, {"Retry-After": "1"}), (200, {})]}
    )

    _, output = _run_warm(server, tmp_path, deadline=2)

    assert output == "status=slow\n"
    assert server.paths().count(_STATED_SCOPE) == 2


def test_a_token_for_another_audience_is_an_error(serve, tmp_path: Path):
    server = serve(_ok())
    server.audience = "https://elsewhere.example"

    _, output = _run_warm(server, tmp_path)

    assert output == "status=error\n"
    assert server.paths() == [_MINT_PATH]


def test_the_token_mint_retries_a_transient_failure(serve, tmp_path: Path):
    server = serve(_ok() | {_MINT_PATH: [(503, {"Retry-After": "0"}), (200, {})]})

    _, output = _run_warm(server, tmp_path, extra_env={"REQUIRE_PROBE": "true"})

    assert output == "status=warm\n"
    assert server.paths() == [_MINT_PATH, _MINT_PATH, *_WARM_PATHS]


@pytest.mark.parametrize(
    ("replies", "required", "status"),
    [
        ({_MINT_PATH: [(403, {})]}, "true", "error"),
        ({_MINT_PATH: [(403, {})]}, "false", "forbidden"),
        ({_MINT_PATH: [(500, {"Retry-After": "0"})]}, "true", "error"),
        ({"/api/skills": [(401, {})]}, "true", "error"),
        ({"/api/skills": [(401, {})]}, "false", "forbidden"),
        # A route that never answers 200 within its deadline.
        ({_STATED_SCOPE: [_BUILDING]}, "true", "error"),
        ({_STATED_SCOPE: [_BUILDING]}, "false", "slow"),
        ({"/api/skills": [(502, {"Retry-After": "0"})]}, "true", "error"),
    ],
)
def test_require_probe_turns_an_unprobed_revision_into_an_error(
    serve, tmp_path: Path, replies, required, status
):
    """Production requires every route to answer 200."""
    server = serve(_ok() | replies)

    result, output = _run_warm(server, tmp_path, deadline=1, extra_env={"REQUIRE_PROBE": required})

    assert result.returncode == 0, result.stderr
    assert output == f"status={status}\n"


def test_an_unreadable_service_url_is_an_error_when_the_probe_is_required(serve, tmp_path: Path):
    server = serve(_ok())

    _, output = _run_warm(server, tmp_path, describe_exit=1, extra_env={"REQUIRE_PROBE": "true"})

    assert output == "status=error\n"


@pytest.mark.parametrize(
    ("path", "body", "status"),
    [
        (_UN_TERMS, {"cause": {"code": "origin_not_served"}}, "warm"),
        (_UN_TERMS, {"cause": {"code": "node_not_held"}}, "error"),
        (_UN_TERMS, {}, "error"),
        # Only the UN terms route answers `origin_not_served`.
        (_STATED_SCOPE, {"cause": {"code": "origin_not_served"}}, "error"),
    ],
)
def test_a_404_origin_not_served_skips_the_un_terms_route_and_any_other_404_fails(
    serve, tmp_path: Path, path, body, status
):
    server = serve(_ok() | {path: [(404, {})]})
    server.bodies[path] = json.dumps(body).encode()

    _, output = _run_warm(server, tmp_path)

    assert output == f"status={status}\n"
    assert server.paths() == [_MINT_PATH, *_WARM_PATHS], "the probe goes on after the skip"
