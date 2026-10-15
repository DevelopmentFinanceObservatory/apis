"""Log lines are JSON objects that Cloud Logging groups under the request's trace."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from dfo_platform.app import Surface, create_app
from dfo_platform.cloud_logging import PROJECT_ENV, TRACE_FIELD, JsonFormatter, stdout_handler
from dfo_platform.trace import cloud_trace_id_from_headers

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"


def _app_logging_to(stream: io.StringIO, *, project: str | None):
    """An app whose one route logs a WARNING through a JSON handler writing to `stream`."""
    log = logging.getLogger("tests.platform.cloud_logging")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(project=project))
    log.addHandler(handler)
    log.propagate = False

    router = APIRouter()

    @router.get("/api/t/log")
    async def emit():
        log.warning("inside %s", "the request")
        return {}

    surface = Surface(name="t", prefix="/api/t", router=router, capability_version="v-t", codes={})
    return create_app([surface])


@pytest.fixture
def stream():
    stream = io.StringIO()
    yield stream
    log = logging.getLogger("tests.platform.cloud_logging")
    log.handlers.clear()
    log.propagate = True


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"X-Cloud-Trace-Context": "abc123/1;o=1"}, "projects/p-1/traces/abc123"),
        (
            {"traceparent": f"00-{_TRACE_ID}-00f067aa0ba902b7-01"},
            f"projects/p-1/traces/{_TRACE_ID}",
        ),
        ({}, None),
    ],
    ids=["cloud-trace-context", "traceparent", "no-header"],
)
def test_a_line_logged_inside_a_request_carries_its_trace(stream, headers, expected):
    app = _app_logging_to(stream, project="p-1")

    TestClient(app).get("/api/t/log", headers=headers)

    entry = json.loads(stream.getvalue())
    assert entry["severity"] == "WARNING"
    assert entry["message"] == "inside the request"
    assert entry["logger"] == "tests.platform.cloud_logging"
    assert entry.get(TRACE_FIELD) == expected


def test_a_line_omits_the_trace_without_a_project(stream):
    app = _app_logging_to(stream, project=None)

    TestClient(app).get("/api/t/log", headers={"X-Cloud-Trace-Context": "abc123/1;o=1"})

    assert TRACE_FIELD not in json.loads(stream.getvalue())


def test_a_record_json_fields_become_top_level_entries_that_cannot_replace_the_reserved_ones():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "request", (), None)
    record.json_fields = {"caller_email": "a@b", "severity": "forged"}

    entry = json.loads(JsonFormatter(project="p-1").format(record))

    assert entry["caller_email"] == "a@b"
    assert entry["severity"] == "INFO"


def test_a_traceback_is_folded_into_the_message():
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", (), None)
    try:
        raise ValueError("boom")
    except ValueError:
        record.exc_info = sys.exc_info()

    message = json.loads(JsonFormatter(project="p-1").format(record))["message"]

    assert message.startswith("failed\nTraceback")
    assert "ValueError: boom" in message


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-cloud-trace-context": "ABC123"}, "abc123"),
        ({"x-cloud-trace-context": "abc123;o=1"}, "abc123"),
        ({"x-cloud-trace-context": "not-a-trace/1"}, None),
        ({"x-cloud-trace-context": "000/1;o=1"}, None),
        ({"traceparent": "00-" + "0" * 32 + "-00f067aa0ba902b7-01"}, None),
        ({"traceparent": "00-short-00f067aa0ba902b7-01"}, None),
        (
            {
                "x-cloud-trace-context": "abc123/1",
                "traceparent": f"00-{_TRACE_ID}-00f067aa0ba902b7-01",
            },
            "abc123",
        ),
    ],
)
def test_cloud_trace_id_parsing(headers, expected):
    assert cloud_trace_id_from_headers(headers) == expected


def test_the_handler_warns_once_when_the_project_is_unset(monkeypatch, capsys):
    monkeypatch.delenv(PROJECT_ENV, raising=False)

    stdout_handler()

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["severity"] == "WARNING"
    assert PROJECT_ENV in entry["message"]


def test_the_handler_is_silent_when_the_project_is_set(monkeypatch, capsys):
    monkeypatch.setenv(PROJECT_ENV, "p-1")

    stdout_handler()

    assert capsys.readouterr().out == ""


def test_the_dockerfile_log_config_sends_search_datasets_and_request_info_to_stdout_as_json():
    """Applies the committed config in a subprocess, since dictConfig replaces global state.

    The Datasets doors log their `datasets.timings` line at INFO on `datasets_api.*`."""
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text()
    assert "--log-config /app/log-config.json" in dockerfile
    assert "COPY log-config.json /app/log-config.json" in dockerfile

    script = (
        "import json, logging, logging.config, sys;"
        "logging.config.dictConfig(json.load(open(sys.argv[1])));"
        "logging.getLogger('search.engine.fanout').info('from search');"
        "logging.getLogger('datasets_api.doors.definition').info('from datasets');"
        "logging.getLogger('dfo_platform.request').info('from request');"
        "logging.getLogger('httpx').info('below the root level')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(_REPO_ROOT / "log-config.json")],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, PROJECT_ENV: "p-1"},
    )

    entries = [json.loads(line) for line in result.stdout.splitlines()]
    assert [(e["logger"], e["message"]) for e in entries] == [
        ("search.engine.fanout", "from search"),
        ("datasets_api.doors.definition", "from datasets"),
        ("dfo_platform.request", "from request"),
    ]
