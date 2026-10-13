"""Write pytest outcome details for the LLM fixture harvest.

The summary counts passed calls, call failures, setup and collection errors, and teardown errors.
Each failure records its phase, exception type, MissingFixtureError key when present, and the number
of FakeLLM misses logged to the harvest file while its test ran. A module that fails to collect is
recorded as a failure with phase `collect` and an empty exception type, which the harvest refuses;
under xdist such a module does not stop the run, so the summary is the only place it shows. The
summary also carries `collected`, the session's collected test count, and `reported`, the number of
tests that produced a call report, a setup failure or skip, or a crash report. A run that ends with
fewer reported than collected tests stopped early, as xdist does after `--max-worker-restart`
crashes, and the harvest refuses it. A failure other than MissingFixtureError that
follows a miss is collateral: an app test turns the miss into a 500, for example. An unset output
path leaves ordinary pytest runs unchanged.

Under pytest-xdist each worker classifies its own failures in `pytest_runtest_makereport`, where
`call.excinfo` is present, and attaches the record to the report as `harvest_failure`. Pytest
serialises extra report attributes, so the controller's `pytest_runtest_logreport` receives the
record and the controller alone writes the summary. A failure report without the record, such as
xdist's report for a crashed worker, counts with an empty exception type, which the harvest refuses.
Each worker also logs FakeLLM calls to its own file, `worker_harvest_path`, so concurrent workers
never append to one file and `misses_during` counts only the worker's own misses.

Pytest imports this `-p` plugin before `tests/conftest.py` sets the offline `SEARCH_GRAPH_BASE`, so
the module sits outside the `tests.fixtures` package and imports no engine code.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.llm_miss_guard import HARVEST_MISS_PATH_ENV

OUTCOME_PATH_ENV = "SEARCH_LLM_HARVEST_OUTCOME_PATH"
"""Output path for the summary; unset paths produce no file."""

SCHEMA_VERSION = 4
"""Summary layout version; change it when summary fields or phases change."""

_COUNT_KEYS = ("passed", "failed", "errored", "teardown_errored")


def worker_harvest_path(harvest_path: Path, worker: str) -> Path:
    """Return the harvest file an xdist worker such as `gw0` logs to beside `harvest_path`."""
    return harvest_path.with_name(f"{harvest_path.name}.{worker}")


def worker_harvest_paths(harvest_path: Path) -> list[Path]:
    """Return every xdist worker's harvest file beside `harvest_path`, in name order."""
    pattern = re.compile(rf"{re.escape(harvest_path.name)}\.gw\d+")
    return sorted(
        path
        for path in harvest_path.parent.glob(f"{harvest_path.name}.gw*")
        if pattern.fullmatch(path.name)
    )


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
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            tail = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        # A worker's harvest file appears on its first logged call.
        return 0
    return sum(1 for line in tail.splitlines() if '"outcome": "miss"' in line)


def _unclassified_failure(nodeid: str, when: str) -> dict:
    """Return a failure record with an empty exception type, which the harvest refuses."""
    return {
        "nodeid": nodeid,
        "when": when,
        "exception_type": "",
        "fixture_key": None,
        "misses_during": 0,
    }


class _Summary:
    """Count every report and keep its failure record, then write the summary at session end."""

    def __init__(self, outcome_path: str) -> None:
        self._outcome_path = outcome_path
        self._counts = dict.fromkeys(_COUNT_KEYS, 0)
        self._failures: list[dict] = []
        self._reported: set[str] = set()

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if not report.failed:
            return
        self._counts["errored"] += 1
        self._failures.append(_unclassified_failure(report.nodeid, "collect"))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # xdist's report for a test whose worker crashed has phase "???".
        if report.when in ("call", "???") or (report.when == "setup" and not report.passed):
            self._reported.add(report.nodeid)
        if report.passed:
            if report.when == "call":
                self._counts["passed"] += 1
            return
        if not report.failed:
            # Skips and xfails do not count as failures.
            return
        if report.when == "teardown":
            self._counts["teardown_errored"] += 1
        else:
            self._counts["failed" if report.when == "call" else "errored"] += 1
        self._failures.append(
            getattr(report, "harvest_failure", None)
            or _unclassified_failure(report.nodeid, report.when)
        )

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "counts": self._counts,
            "collected": session.testscollected,
            "reported": len(self._reported),
            "failures": self._failures,
        }
        Path(self._outcome_path).write_text(json.dumps(summary, indent=2) + "\n")


def pytest_configure(config: pytest.Config) -> None:
    if hasattr(config, "workerinput"):
        # An xdist worker logs to its own harvest file and leaves the summary to the controller.
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        path = os.environ.get(HARVEST_MISS_PATH_ENV)
        if worker and path:
            os.environ[HARVEST_MISS_PATH_ENV] = str(worker_harvest_path(Path(path), worker))
        return
    outcome_path = os.environ.get(OUTCOME_PATH_ENV)
    if outcome_path:
        config.pluginmanager.register(_Summary(outcome_path), "harvest_outcome_summary")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    item._harvest_offset = _harvest_file_size()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """Attach the failure's exception details to its report while `call.excinfo` is present."""
    report = yield
    if report.failed:
        excinfo = call.excinfo
        report.harvest_failure = {
            "nodeid": report.nodeid,
            "when": report.when,
            "exception_type": excinfo.type.__name__ if excinfo is not None else "",
            "fixture_key": getattr(excinfo.value, "fixture_key", None)
            if excinfo is not None
            else None,
            "misses_during": _misses_since(getattr(item, "_harvest_offset", None)),
        }
    return report
