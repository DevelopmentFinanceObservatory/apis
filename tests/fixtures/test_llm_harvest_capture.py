"""Tests for FakeLLM hit/miss logging and the harvest outcome plugin.

Harvest paths use pytest temporary files.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from tests._harvest_outcome_plugin import OUTCOME_PATH_ENV, SCHEMA_VERSION
from tests.fixtures import HARVEST_MISS_PATH_ENV, FakeLLM, MissingFixtureError
from tests.fixtures.keys import fixture_key

pytest_plugins = ["pytester"]

# Every test here drives FakeLLM, or a pytester run in this process, into a miss on purpose.
pytestmark = pytest.mark.llm_misses_expected


class _Schema(BaseModel):
    value: str


def test_env_unset_hit_returns_fixture_unchanged(monkeypatch):
    """With logging unset, a hit returns the stored response and zero usage."""
    monkeypatch.delenv(HARVEST_MISS_PATH_ENV, raising=False)
    key = fixture_key(_Schema, "sys", "prompt")
    llm = FakeLLM(responses={key: {"value": "ok"}})

    parsed, usage = asyncio.run(
        llm.generate_structured(prompt="prompt", system="sys", schema=_Schema)
    )

    assert parsed == _Schema(value="ok")
    assert usage == {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


def test_env_unset_miss_raises_keyerror_and_writes_nothing(monkeypatch, tmp_path):
    """With logging unset, a miss raises KeyError without writing a record."""
    monkeypatch.delenv(HARVEST_MISS_PATH_ENV, raising=False)
    llm = FakeLLM(responses={})

    with pytest.raises(KeyError, match="_Schema"):
        asyncio.run(llm.generate_structured(prompt="prompt", system="sys", schema=_Schema))

    assert list(tmp_path.iterdir()) == []


def test_miss_raises_missingfixtureerror_that_is_a_keyerror(monkeypatch):
    """MissingFixtureError exposes the missing key and remains a KeyError."""
    monkeypatch.delenv(HARVEST_MISS_PATH_ENV, raising=False)
    llm = FakeLLM(responses={})

    with pytest.raises(MissingFixtureError) as excinfo:
        asyncio.run(
            llm.generate_structured(prompt="the prompt", system="the system", schema=_Schema)
        )

    assert issubclass(MissingFixtureError, KeyError)
    assert isinstance(excinfo.value, KeyError)
    assert excinfo.value.fixture_key == fixture_key(_Schema, "the system", "the prompt")
    assert excinfo.value.args[0].startswith("FakeLLM: missing fixture for schema='_Schema'")


def test_env_set_miss_appends_one_wellformed_line_and_still_raises(monkeypatch, tmp_path):
    miss_path = tmp_path / "misses.jsonl"
    monkeypatch.setenv(HARVEST_MISS_PATH_ENV, str(miss_path))
    llm = FakeLLM(responses={})

    with pytest.raises(KeyError):
        asyncio.run(
            llm.generate_structured(prompt="the prompt", system="the system", schema=_Schema)
        )

    lines = miss_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["outcome"] == "miss"
    assert record["schema_name"] == "_Schema"
    assert record["system"] == "the system"
    assert record["prompt"] == "the prompt"
    assert record["key"] == fixture_key(_Schema, "the system", "the prompt")
    # Miss records include the prompt fields needed to recreate the key.
    assert set(record) == {"key", "outcome", "schema_name", "system", "prompt"}


def test_env_set_hit_appends_a_hit_line_carrying_only_the_key(monkeypatch, tmp_path):
    """Hit records contain only the key and outcome for pruning."""
    miss_path = tmp_path / "misses.jsonl"
    monkeypatch.setenv(HARVEST_MISS_PATH_ENV, str(miss_path))
    key = fixture_key(_Schema, "sys", "prompt")
    llm = FakeLLM(responses={key: {"value": "ok"}})

    asyncio.run(llm.generate_structured(prompt="prompt", system="sys", schema=_Schema))

    lines = miss_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["key"] == key
    assert record["outcome"] == "hit"
    assert set(record) == {"key", "outcome"}


def test_env_set_two_misses_append_two_lines(monkeypatch, tmp_path):
    """Each miss appends one line; the harvest handles deduplication."""
    miss_path = tmp_path / "misses.jsonl"
    monkeypatch.setenv(HARVEST_MISS_PATH_ENV, str(miss_path))
    llm = FakeLLM(responses={})

    for _ in range(2):
        with pytest.raises(KeyError):
            asyncio.run(
                llm.generate_structured(prompt="the prompt", system="the system", schema=_Schema)
            )

    lines = miss_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == json.loads(lines[1])


_GENERATED_TESTS = """
def test_passes():
    assert True


def test_assertion_failure():
    assert 1 == 2


def test_missing_fixture():
    from tests.fixtures import MissingFixtureError

    raise MissingFixtureError("no such fixture", fixture_key="Extraction:deadbeef")
"""


def _run_generated_pass(pytester, *, outcome_path: Path | None) -> dict:
    """Run three generated tests and parse the plugin summary."""
    pytester.makepyfile(test_generated=_GENERATED_TESTS)
    args = ["-q", "-p", "tests._harvest_outcome_plugin"]
    if outcome_path is not None:
        pytester._monkeypatch.setenv(OUTCOME_PATH_ENV, str(outcome_path))
    else:
        pytester._monkeypatch.delenv(OUTCOME_PATH_ENV, raising=False)
    pytester.runpytest(*args)
    return json.loads(outcome_path.read_text()) if outcome_path is not None else {}


def test_plugin_summary_records_each_failure_type_and_the_fixture_key(pytester):
    summary = _run_generated_pass(pytester, outcome_path=pytester.path / "out.json")

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["counts"] == {
        "passed": 1,
        "failed": 2,
        "errored": 0,
        "teardown_errored": 0,
    }
    by_nodeid = {f["nodeid"]: f for f in summary["failures"]}
    assert set(by_nodeid) == {
        "test_generated.py::test_assertion_failure",
        "test_generated.py::test_missing_fixture",
    }
    assert (
        by_nodeid["test_generated.py::test_assertion_failure"]["exception_type"] == "AssertionError"
    )
    assert by_nodeid["test_generated.py::test_assertion_failure"]["fixture_key"] is None
    missing = by_nodeid["test_generated.py::test_missing_fixture"]
    assert missing["exception_type"] == MissingFixtureError.__name__
    assert missing["fixture_key"] == "Extraction:deadbeef"
    assert missing["when"] == "call"


def test_plugin_writes_nothing_when_the_outcome_path_is_unset(pytester):
    """An unset output path leaves ordinary pytest runs inert."""
    _run_generated_pass(pytester, outcome_path=None)

    assert sorted(p.name for p in pytester.path.iterdir() if p.is_file()) == ["test_generated.py"]


_TEARDOWN_FAILURE_TEST = """
import pytest


@pytest.fixture
def teardown_raises():
    yield
    raise RuntimeError("boom in teardown")


def test_passes_call_but_fails_teardown(teardown_raises):
    assert True
"""


def test_plugin_records_a_teardown_error_the_call_phase_never_saw(pytester):
    """A teardown error follows a passing call and still appears in the summary."""
    pytester.makepyfile(test_generated=_TEARDOWN_FAILURE_TEST)
    outcome_path = pytester.path / "out.json"
    pytester._monkeypatch.setenv(OUTCOME_PATH_ENV, str(outcome_path))

    pytester.runpytest("-q", "-p", "tests._harvest_outcome_plugin")
    summary = json.loads(outcome_path.read_text())

    assert summary["counts"] == {
        "passed": 1,
        "failed": 0,
        "errored": 0,
        "teardown_errored": 1,
    }
    assert len(summary["failures"]) == 1
    failure = summary["failures"][0]
    assert failure["nodeid"] == "test_generated.py::test_passes_call_but_fails_teardown"
    assert failure["when"] == "teardown"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["fixture_key"] is None


_COLLATERAL_TESTS = """
import asyncio

import pytest
from pydantic import BaseModel

from tests.fixtures import FakeLLM, MissingFixtureError


class _S(BaseModel):
    value: str


def test_miss_swallowed_then_assertion_fails():
    try:
        asyncio.run(FakeLLM(responses={}).generate_structured(prompt="p", system="s", schema=_S))
    except MissingFixtureError:
        pass
    assert False, "the code under test turned the miss into another failure"


def test_unrelated_assertion_after_a_clean_test():
    assert False
"""


def test_plugin_counts_the_misses_logged_while_each_failing_test_ran(pytester):
    """A failure after a logged miss carries the miss count; an unrelated failure carries zero."""
    pytester.makepyfile(test_generated=_COLLATERAL_TESTS)
    outcome_path = pytester.path / "out.json"
    pytester._monkeypatch.setenv(OUTCOME_PATH_ENV, str(outcome_path))
    pytester._monkeypatch.setenv(HARVEST_MISS_PATH_ENV, str(pytester.path / "misses.jsonl"))

    pytester.runpytest("-q", "-p", "tests._harvest_outcome_plugin")
    by_nodeid = {f["nodeid"]: f for f in json.loads(outcome_path.read_text())["failures"]}

    assert (
        by_nodeid["test_generated.py::test_miss_swallowed_then_assertion_fails"]["misses_during"]
        == 1
    )
    assert (
        by_nodeid["test_generated.py::test_unrelated_assertion_after_a_clean_test"]["misses_during"]
        == 0
    )


def test_plugin_loads_without_search_graph_base():
    """The harvest child starts pytest with the plugin and no SEARCH_GRAPH_BASE.

    Pytest imports `-p` plugins before `tests/conftest.py` sets its offline placeholder, so an
    engine import in the plugin's chain fails at startup. A fresh interpreter is needed because
    this process has already imported the engine settings. Collecting one small module keeps the
    run to a few seconds.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "SEARCH_GRAPH_BASE" and not k.startswith("PYTEST_")
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:xdist",
            "-p",
            "tests._harvest_outcome_plugin",
            "--collect-only",
            "tests/test_llm_miss_guard.py",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
