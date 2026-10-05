"""Write pytest outcome details for the LLM fixture harvest.

The summary counts passed calls, call failures, setup errors, and teardown errors. Each failure
records its phase, exception type, MissingFixtureError key when present, and the number of FakeLLM
misses logged to the harvest file while its test ran. A failure other than MissingFixtureError that
follows a miss is collateral: an app test turns the miss into a 500, for example. An unset output
path leaves ordinary pytest runs unchanged.

Pytest imports this `-p` plugin before `tests/conftest.py` sets the offline `SEARCH_GRAPH_BASE`, so
the module sits outside the `tests.fixtures` package and imports no engine code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.llm_miss_guard import HARVEST_MISS_PATH_ENV

OUTCOME_PATH_ENV = "SEARCH_LLM_HARVEST_OUTCOME_PATH"
"""Output path for the summary; unset paths produce no file."""

SCHEMA_VERSION = 3
"""Summary layout version; change it when summary fields or phases change."""

_COUNT_KEYS = ("passed", "failed", "errored", "teardown_errored")


def _harvest_file_size() -> int | None:
    """Return the harvest file's size in bytes (0 before it exists), or None when logging is off."""
    path = os.environ.get(HARVEST_MISS_PATH_ENV)
    if not path:
        return None
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _misses_since(offset: int | None) -> int:
    """Count miss lines appended to the harvest file after `offset` bytes."""
    path = os.environ.get(HARVEST_MISS_PATH_ENV)
    if offset is None or not path:
        return 0
    with open(path, "rb") as f:
        f.seek(offset)
        tail = f.read().decode("utf-8", errors="replace")
    return sum(1 for line in tail.splitlines() if '"outcome": "miss"' in line)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item) -> None:
    item._harvest_offset = _harvest_file_size()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Record call, setup, and teardown failures with exception details."""
    report = yield
    counts = item.session._harvest_counts
    if report.passed:
        if report.when == "call":
            counts["passed"] += 1
        return report
    if not report.failed:
        # Skips and xfails do not count as failures.
        return report
    if report.when == "teardown":
        counts["teardown_errored"] += 1
    else:
        counts["failed" if report.when == "call" else "errored"] += 1
    excinfo = call.excinfo
    item.session._harvest_failures.append(
        {
            "nodeid": report.nodeid,
            "when": report.when,
            "exception_type": excinfo.type.__name__ if excinfo is not None else "",
            "fixture_key": getattr(excinfo.value, "fixture_key", None)
            if excinfo is not None
            else None,
            "misses_during": _misses_since(getattr(item, "_harvest_offset", None)),
        }
    )
    return report


def pytest_sessionstart(session) -> None:
    session._harvest_counts = dict.fromkeys(_COUNT_KEYS, 0)
    session._harvest_failures = []


def pytest_sessionfinish(session, exitstatus) -> None:
    outcome_path = os.environ.get(OUTCOME_PATH_ENV)
    if not outcome_path:
        return
    summary = {
        "schema_version": SCHEMA_VERSION,
        "counts": dict(session._harvest_counts),
        "failures": list(session._harvest_failures),
    }
    Path(outcome_path).write_text(json.dumps(summary, indent=2) + "\n")
