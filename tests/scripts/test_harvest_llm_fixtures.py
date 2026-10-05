"""Pins fixture harvesting with fake callers and in-memory banks."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

import scripts.harvest_llm_fixtures as m
import search.engine.extract as extract_module
from search.engine.bind import _UniversalBindOutput
from search.engine.extract import Extraction
from tests.fixtures import MissingFixtureError, _append_harvest_line
from tests.fixtures._harvest_outcome_plugin import OUTCOME_PATH_ENV, SCHEMA_VERSION
from tests.fixtures.keys import fixture_key


def _miss(schema: type[BaseModel], system: str, prompt: str) -> m.Miss:
    key = fixture_key(schema, system, prompt)
    return m.Miss(key=key, schema_name=schema.__name__, system=system, prompt=prompt)


def _obs(
    misses: dict[str, m.Miss], *, requested: frozenset[str] | None = None
) -> m.PassObservations:
    """Builds fake pass observations, defaulting requested keys to the misses."""
    return m.PassObservations(misses=misses, requested=requested or frozenset(misses))


class _FakeCaller:
    """Returns a scripted result by key and raises for configured failures."""

    def __init__(self, results: dict[str, object], raise_for: frozenset[str] = frozenset()):
        self._results = results
        self._raise_for = raise_for
        self.calls: list[str] = []

    async def generate_structured(self, *, prompt: str, system: str, schema):
        key = fixture_key(schema, system, prompt)
        self.calls.append(key)
        if key in self._raise_for:
            raise RuntimeError(f"simulated schema-rejected response for {key}")
        return self._results[key], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


# ---------------------------------------------------------------------------
# _parse_miss_lines / _parse_requested_keys
# ---------------------------------------------------------------------------


def _miss_line(path: Path, miss: m.Miss) -> str:
    """Writes and returns one miss line in the production format."""
    _append_harvest_line(
        str(path),
        outcome="miss",
        key=miss.key,
        schema_name=miss.schema_name,
        system=miss.system,
        prompt=miss.prompt,
    )
    return path.read_text().splitlines()[-1]


def _hit_line(path: Path, key: str) -> str:
    _append_harvest_line(str(path), outcome="hit", key=key)
    return path.read_text().splitlines()[-1]


def test_parse_miss_lines_dedupes_same_key_from_two_sources(tmp_path):
    """sweep_corpus.py and pytest both append the same golden's Extraction
    miss to the same file in one pass; that is one Miss, not two."""
    miss = _miss(Extraction, "sys", "prompt")
    line = _miss_line(tmp_path / "misses.jsonl", miss)

    parsed = m._parse_miss_lines(tmp_path / "misses.jsonl", [line, line])

    assert parsed == {miss.key: miss}


def test_parse_miss_lines_skips_blank_lines(tmp_path):
    miss = _miss(Extraction, "sys", "prompt")
    line = _miss_line(tmp_path / "misses.jsonl", miss)

    parsed = m._parse_miss_lines(tmp_path / "misses.jsonl", ["", line, "   ", ""])

    assert parsed == {miss.key: miss}


def test_parse_miss_lines_ignores_hit_lines(tmp_path):
    """Hit lines lack miss fields and must be skipped."""
    miss = _miss(Extraction, "sys", "prompt")
    path = tmp_path / "misses.jsonl"

    parsed = m._parse_miss_lines(
        path, [_hit_line(path, "Extraction:someotherkey"), _miss_line(path, miss)]
    )

    assert parsed == {miss.key: miss}


def test_parse_requested_keys_includes_both_hits_and_misses(tmp_path):
    """The reached set includes hit and miss keys, including after convergence."""
    miss = _miss(Extraction, "sys", "prompt")
    hit_key = "Extraction:hit-only-key"
    path = tmp_path / "misses.jsonl"

    requested = m._parse_requested_keys(path, [_miss_line(path, miss), _hit_line(path, hit_key)])

    assert requested == frozenset({miss.key, hit_key})


def test_parse_requested_keys_skips_blank_lines(tmp_path):
    hit_key = "Extraction:hit-only-key"
    path = tmp_path / "misses.jsonl"

    requested = m._parse_requested_keys(path, ["", _hit_line(path, hit_key), "   "])

    assert requested == frozenset({hit_key})


def test_parse_miss_lines_reports_a_truncated_line_with_its_file_and_number(tmp_path):
    """A truncated JSON line raises `HarvestError` with its file and line number."""
    path = tmp_path / "misses.jsonl"
    miss = _miss(Extraction, "sys", "prompt")
    good = _miss_line(path, miss)
    path.write_text(good + "\n" + '{"key": "Extraction:truncated", "outcome": "mis')

    with pytest.raises(m.HarvestError, match=r"misses\.jsonl line 2"):
        m._parse_miss_lines(path, path.read_text().splitlines())


def test_parse_requested_keys_reports_a_non_object_line_with_its_file_and_number(tmp_path):
    """A JSON line that parses but is not an object (a bare number, a list)
    names no key, so it can only be a foreign or corrupted file."""
    path = tmp_path / "misses.jsonl"
    path.write_text('["not", "a", "record"]\n')

    with pytest.raises(m.HarvestError, match=r"misses\.jsonl line 1.*list"):
        m._parse_requested_keys(path, path.read_text().splitlines())


def test_read_harvest_lines_reports_an_unreadable_path(tmp_path):
    """A directory at the harvest path raises `HarvestError` as an unreadable file."""
    unreadable = tmp_path / "misses.jsonl"
    unreadable.mkdir()
    try:
        with pytest.raises(m.HarvestError, match=r"misses\.jsonl.*unreadable"):
            m._read_harvest_lines(unreadable)
    finally:
        unreadable.rmdir()


def test_read_harvest_lines_reports_a_file_the_process_cannot_open(tmp_path):
    """The mode-000 shape of the same axis, for a filesystem that does not
    let a directory stand in (root, or a platform where the directory read
    succeeds)."""
    path = tmp_path / "misses.jsonl"
    path.write_text("")
    path.chmod(0o000)
    try:
        with pytest.raises(m.HarvestError, match="unreadable"):
            m._read_harvest_lines(path)
    finally:
        path.chmod(0o600)


def test_check_pytest_outcome_reports_an_unreadable_harvest_file(tmp_path):
    """An rc-1 summary needs the harvest file; unreadable files raise `HarvestError`."""
    outcome_path = tmp_path / "pytest_outcome.json"
    _write_outcome(
        outcome_path,
        [
            {
                "nodeid": "tests/x.py::test_a",
                "when": "call",
                "exception_type": MissingFixtureError.__name__,
                "fixture_key": "Extraction:aaa",
            }
        ],
    )
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.mkdir()
    try:
        with pytest.raises(m.HarvestError, match="unreadable"):
            m._check_pytest_outcome(
                returncode=1,
                outcome_path=outcome_path,
                harvest_file=harvest_path,
                where="pytest collector",
            )
    finally:
        harvest_path.rmdir()


# ---------------------------------------------------------------------------
# _schema_and_stage
# ---------------------------------------------------------------------------


def test_schema_and_stage_known_schemas():
    assert m._schema_and_stage("Extraction") == (Extraction, "extract")
    assert m._schema_and_stage("_UniversalBindOutput") == (_UniversalBindOutput, "bind")


def test_schema_and_stage_unknown_name_fails_closed():
    with pytest.raises(m.HarvestError, match="NotARealSchema"):
        m._schema_and_stage("NotARealSchema")


# ---------------------------------------------------------------------------
# harvest()
# ---------------------------------------------------------------------------


def test_harvest_stops_immediately_when_first_pass_finds_nothing_new():
    caller = _FakeCaller(results={})
    writes: list[dict] = []

    result = asyncio.run(
        m.harvest(
            existing={"already": "known"},
            llm=caller,
            collect_pass=lambda: _obs({}),
            write_bank=writes.append,
        )
    )

    assert result.passes_run == 1
    assert result.added_per_pass == [0]
    assert result.total_added == 0
    assert result.bank == {"already": "known"}
    assert result.final_pass_keys == frozenset()
    assert caller.calls == []
    assert writes == []


def test_harvest_reaches_fixed_point_after_resolving_new_keys():
    """The repeated misses on pass 2 are already in the bank, so harvesting stops."""
    miss_a = _miss(Extraction, "sys", "prompt a")
    miss_b = _miss(Extraction, "sys", "prompt b")
    misses = {miss_a.key: miss_a, miss_b.key: miss_b}
    caller = _FakeCaller(
        results={
            miss_a.key: Extraction(kind="resolution", variables=["a"]),
            miss_b.key: Extraction(kind="resolution", variables=["b"]),
        }
    )
    writes: list[dict] = []
    calls = {"n": 0}

    def collect_pass():
        calls["n"] += 1
        return _obs(misses)  # the same two keys every pass; only new-vs-bank matters

    result = asyncio.run(
        m.harvest(
            existing={},
            llm=caller,
            collect_pass=collect_pass,
            write_bank=lambda bank: writes.append(dict(bank)),
        )
    )

    assert result.passes_run == 2
    assert result.added_per_pass == [2, 0]
    assert result.total_added == 2
    assert result.bank[miss_a.key] == Extraction(kind="resolution", variables=["a"]).model_dump(
        mode="json"
    )
    assert result.bank[miss_b.key] == Extraction(kind="resolution", variables=["b"]).model_dump(
        mode="json"
    )
    assert result.final_pass_keys == frozenset(misses)
    # Committed once, after pass 1's resolution -- not once per pass.
    assert len(writes) == 1
    assert writes[0] == result.bank
    assert calls["n"] == 2


def test_harvest_final_pass_keys_is_requested_not_misses():
    """The fixed-point pass's PassObservations.requested can carry hits the
    miss set never saw -- final_pass_keys must come from `requested`, not be
    re-derived from `misses` (which converged to empty)."""
    miss = _miss(Extraction, "sys", "prompt")
    hit_only_key = "Extraction:already-in-bank"
    caller = _FakeCaller(results={miss.key: Extraction(kind="resolution", variables=["x"])})
    calls = {"n": 0}

    def collect_pass():
        calls["n"] += 1
        if calls["n"] == 1:
            return _obs({miss.key: miss}, requested=frozenset({miss.key}))
        # Pass 2: the key is now in the bank, so nothing misses -- but a hit
        # for it, and for a second key already known, are still requested.
        return _obs({}, requested=frozenset({miss.key, hit_only_key}))

    result = asyncio.run(
        m.harvest(
            existing={hit_only_key: {"already": "known"}},
            llm=caller,
            collect_pass=collect_pass,
            write_bank=lambda bank: None,
        )
    )

    assert result.passes_run == 2
    assert result.final_pass_keys == frozenset({miss.key, hit_only_key})


def test_harvest_two_sources_reporting_same_key_produce_one_llm_call():
    miss = _miss(Extraction, "sys", "prompt")

    def collect_pass():
        # Simulates sweep_corpus.py and pytest both writing the same miss to
        # the harvest file in one pass; _parse_miss_lines already collapsed
        # it to one entry by the time collect_pass returns.
        return _obs({miss.key: miss})

    caller = _FakeCaller(results={miss.key: Extraction(kind="resolution", variables=["x"])})

    result = asyncio.run(
        m.harvest(existing={}, llm=caller, collect_pass=collect_pass, write_bank=lambda bank: None)
    )

    assert caller.calls == [miss.key]
    assert result.total_added == 1


def test_harvest_aborts_and_leaves_bank_unmodified_when_schema_rejects_response():
    miss = _miss(Extraction, "sys", "prompt")
    caller = _FakeCaller(results={}, raise_for=frozenset({miss.key}))
    writes: list[dict] = []

    with pytest.raises(RuntimeError, match="simulated schema-rejected response"):
        asyncio.run(
            m.harvest(
                existing={"pre-existing": "entry"},
                llm=caller,
                collect_pass=lambda: _obs({miss.key: miss}),
                write_bank=writes.append,
            )
        )

    assert writes == []


def test_harvest_bound_exceeded_fails_loudly_without_writing():
    """A new key on every pass raises at `--max-passes` without a call or write."""
    caller = _FakeCaller(results={})
    writes: list[dict] = []
    pass_count = {"n": 0}

    def collect_pass():
        pass_count["n"] += 1
        # A distinct key every call -> collect_pass never repeats, so harvest()
        # can never see a pass with nothing new.
        fresh = _miss(Extraction, "sys", f"prompt {pass_count['n']}")
        return _obs({fresh.key: fresh})

    with pytest.raises(m.HarvestError, match=r"max.passes|pass 1"):
        asyncio.run(
            m.harvest(
                existing={},
                llm=caller,
                collect_pass=collect_pass,
                write_bank=writes.append,
                max_passes=1,
            )
        )

    assert writes == []
    assert caller.calls == []


# ---------------------------------------------------------------------------
# _direction_corpus_collect
#
# extract() raises MissingFixtureError (a KeyError subclass) on a genuine
# FakeLLM miss, which this collector must swallow -- the miss line FakeLLM
# already appended to harvest_path is the record of it. Catching bare
# KeyError instead (finding 2) would also swallow an ORDINARY KeyError raised
# by anything else inside extraction, silently dropping that query from the
# reached set.
# ---------------------------------------------------------------------------


def _raising_extract(exc: BaseException):
    async def _extract(query, *, llm):
        raise exc

    return _extract


def test_direction_corpus_collect_swallows_a_captured_missing_fixture(tmp_path, monkeypatch):
    """A captured `MissingFixtureError` is skipped so collection can continue."""
    queries = ["a query", "second query"]
    monkeypatch.setattr(m, "_direction_corpus_queries", lambda: queries)
    extracted: list[str] = []

    async def _extract(query, *, llm):
        extracted.append(query)
        if query == "a query":
            raise MissingFixtureError("no such fixture", fixture_key="Extraction:aaa")

    monkeypatch.setattr(extract_module, "extract", _extract)
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.write_text("")

    m._direction_corpus_collect(harvest_path=harvest_path)

    assert extracted == queries


def test_direction_corpus_collect_propagates_an_unrelated_keyerror(tmp_path, monkeypatch):
    """An unrelated `KeyError` propagates instead of disappearing from the reached set.

    `MissingFixtureError` subclasses `KeyError`, so a bare `except KeyError` would swallow it.
    """
    monkeypatch.setattr(m, "_direction_corpus_queries", lambda: ["a query"])
    monkeypatch.setattr(extract_module, "extract", _raising_extract(KeyError("unrelated bug")))
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.write_text("")

    with pytest.raises(KeyError, match="unrelated bug"):
        m._direction_corpus_collect(harvest_path=harvest_path)


# ---------------------------------------------------------------------------
# _check_pytest_outcome / _live_collect_pass
#
# pytest is a subprocess and reports only an exit code, so a suite that
# failed on missing fixtures (the misses a pass exists to collect) and one
# broken by an unrelated edit both exit 1. _check_pytest_outcome reads the
# plugin's own summary and refuses anything but an unambiguously converged
# pass -- see tests/fixtures/_harvest_outcome_plugin.py.
# ---------------------------------------------------------------------------


def _write_outcome(path, failures: list[dict], *, counts: dict | None = None) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "counts": counts or {"passed": 0, "failed": len(failures), "errored": 0},
                "failures": failures,
            }
        )
    )


def _empty_harvest(tmp_path) -> Path:
    path = tmp_path / "misses.jsonl"
    path.write_text("")
    return path


def _write_sweep_outcome(path, *, attempted: int, completed: int, failures: list[dict]) -> None:
    """scripts/sweep_corpus.py's own outcome summary shape, written the way
    its main() writes it (see m.SWEEP_OUTCOME_PATH_ENV/SWEEP_OUTCOME_SCHEMA_VERSION,
    re-exported from scripts.sweep_corpus)."""
    Path(path).write_text(
        json.dumps(
            {
                "schema_version": m.SWEEP_OUTCOME_SCHEMA_VERSION,
                "attempted": attempted,
                "completed": completed,
                "failures": failures,
            }
        )
    )


def _write_miss_line(harvest_path, key: str) -> None:
    """One FakeLLM miss line as _append_harvest_line writes it."""
    _append_harvest_line(
        str(harvest_path),
        outcome="miss",
        key=key,
        schema_name=key.split(":", 1)[0],
        system="sys",
        prompt="prompt",
    )


def _check(
    tmp_path,
    *,
    returncode: int,
    failures: list[dict],
    harvest_keys: set[str] | frozenset[str] = frozenset(),
    summary_json: str | None = None,
):
    """_check_pytest_outcome over a summary and harvest file built from
    `failures` -- `harvest_keys` names the keys the same pass wrote miss
    lines for. `summary_json` writes the summary file verbatim instead, for
    the shapes `_write_outcome` cannot build (malformed JSON, a non-object).
    Returns None on a converged pass, raises otherwise."""
    outcome_path = tmp_path / "pytest_outcome.json"
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.write_text("")
    for key in harvest_keys:
        _write_miss_line(harvest_path, key)
    if summary_json is None:
        _write_outcome(outcome_path, failures)
    else:
        outcome_path.write_text(summary_json)
    return m._check_pytest_outcome(
        returncode=returncode,
        outcome_path=outcome_path,
        harvest_file=harvest_path,
        where="pytest collector",
    )


def test_check_pytest_outcome_accepts_a_clean_run(tmp_path):
    assert _check(tmp_path, returncode=0, failures=[]) is None


def test_check_pytest_outcome_accepts_failures_that_are_all_captured_misses(tmp_path):
    """A run with only captured fixture misses is a converged collection pass."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": MissingFixtureError.__name__,
            "fixture_key": "Extraction:aaa",
        },
        {
            "nodeid": "tests/x.py::test_b",
            "when": "call",
            "exception_type": MissingFixtureError.__name__,
            "fixture_key": "_UniversalBindOutput:bbb",
        },
    ]

    assert (
        _check(
            tmp_path,
            returncode=1,
            failures=failures,
            harvest_keys={"Extraction:aaa", "_UniversalBindOutput:bbb", "X:other"},
        )
        is None
    )


def test_check_pytest_outcome_refuses_an_assertion_failure(tmp_path):
    """An assertion failure refuses collection before an LLM call or bank write."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": "AssertionError",
            "fixture_key": None,
        }
    ]

    with pytest.raises(m.HarvestError, match="AssertionError"):
        _check(tmp_path, returncode=1, failures=failures)


def test_check_pytest_outcome_accepts_a_failure_that_followed_a_logged_miss(tmp_path):
    """An app test turns a miss into a 500, so its assertion fails after the miss was captured."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": "AssertionError",
            "fixture_key": None,
            "misses_during": 2,
        }
    ]

    assert _check(tmp_path, returncode=1, failures=failures) is None


def test_check_pytest_outcome_refuses_a_failure_with_no_miss_during_its_test(tmp_path):
    """A zero `misses_during` leaves an assertion failure unexplained."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": "AssertionError",
            "fixture_key": None,
            "misses_during": 0,
        }
    ]

    with pytest.raises(m.HarvestError, match="AssertionError"):
        _check(tmp_path, returncode=1, failures=failures)


def test_check_pytest_outcome_refuses_a_missing_fixture_with_no_harvest_key(tmp_path):
    """A missing fixture key absent from the harvest file blocks convergence."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": MissingFixtureError.__name__,
            "fixture_key": "Extraction:never-written",
        }
    ]

    with pytest.raises(m.HarvestError, match="never-written"):
        _check(
            tmp_path,
            returncode=1,
            failures=failures,
            harvest_keys={"Extraction:aaa"},
        )


@pytest.mark.parametrize("returncode", [2, 3, 4, 5])
def test_check_pytest_outcome_refuses_a_returncode_other_than_zero_or_one(tmp_path, returncode):
    """Exit codes 2-5 do not establish a completed collection pass."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": MissingFixtureError.__name__,
            "fixture_key": "Extraction:aaa",
        }
    ]

    with pytest.raises(m.HarvestError, match=str(returncode)):
        _check(
            tmp_path,
            returncode=returncode,
            failures=failures,
            harvest_keys={"Extraction:aaa"},
        )


def test_check_pytest_outcome_refuses_an_absent_summary(tmp_path):
    """A missing outcome summary refuses the pass."""
    outcome_path = tmp_path / "pytest_outcome.json"
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.write_text("")

    with pytest.raises(m.HarvestError, match="missing or unreadable"):
        m._check_pytest_outcome(
            returncode=0,
            outcome_path=outcome_path,
            harvest_file=harvest_path,
            where="pytest collector",
        )


def test_check_pytest_outcome_refuses_a_summary_at_an_unknown_schema_version(tmp_path):
    outcome_path = tmp_path / "pytest_outcome.json"
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.write_text("")
    outcome_path.write_text(json.dumps({"schema_version": 99, "failures": []}))

    with pytest.raises(m.HarvestError, match="schema_version"):
        m._check_pytest_outcome(
            returncode=0,
            outcome_path=outcome_path,
            harvest_file=harvest_path,
            where="pytest collector",
        )


def test_check_pytest_outcome_refuses_a_summary_that_is_not_valid_json(tmp_path):
    """A truncated outcome summary refuses the pass."""
    with pytest.raises(m.HarvestError, match="missing or unreadable"):
        _check(
            tmp_path,
            returncode=1,
            failures=[],
            summary_json='{"schema_version": 1, "failures": [',
        )


def test_check_pytest_outcome_refuses_a_summary_that_is_not_an_object(tmp_path):
    """A JSON list where the summary object belongs carries no
    schema_version, so it is refused rather than read field by field."""
    with pytest.raises(m.HarvestError, match="not a JSON object"):
        _check(tmp_path, returncode=1, failures=[], summary_json="[]")


def test_check_pytest_outcome_refuses_a_failures_field_that_is_not_a_list(tmp_path):
    """The outcome requires a `failures` list for either accepted exit code."""
    for returncode in (0, 1):
        with pytest.raises(m.HarvestError, match="no `failures` list"):
            _check(
                tmp_path,
                returncode=returncode,
                failures=[],
                summary_json=json.dumps({"schema_version": SCHEMA_VERSION, "failures": {}}),
            )


def test_check_pytest_outcome_refuses_a_zero_exit_that_records_failures(tmp_path):
    """An exit code of 0 conflicts with a recorded failure."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": MissingFixtureError.__name__,
            "fixture_key": "Extraction:aaa",
        }
    ]

    with pytest.raises(m.HarvestError, match="exited 0"):
        _check(
            tmp_path,
            returncode=0,
            failures=failures,
            harvest_keys={"Extraction:aaa"},
        )


def test_check_pytest_outcome_refuses_a_one_exit_that_records_no_failure(tmp_path):
    """Exit code 1 without a recorded failure does not establish convergence."""
    with pytest.raises(m.HarvestError, match="no failing test"):
        _check(tmp_path, returncode=1, failures=[])


def test_check_pytest_outcome_refuses_a_teardown_only_failure(tmp_path):
    """A teardown failure is recorded and refuses convergence."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "teardown",
            "exception_type": "RuntimeError",
            "fixture_key": None,
        }
    ]

    with pytest.raises(m.HarvestError, match="RuntimeError"):
        _check(tmp_path, returncode=1, failures=failures)


def test_check_pytest_outcome_refuses_a_captured_miss_mixed_with_a_teardown_error(tmp_path):
    """A captured miss plus a teardown error still refuses convergence."""
    failures = [
        {
            "nodeid": "tests/x.py::test_a",
            "when": "call",
            "exception_type": MissingFixtureError.__name__,
            "fixture_key": "Extraction:aaa",
        },
        {
            "nodeid": "tests/x.py::test_b",
            "when": "teardown",
            "exception_type": "RuntimeError",
            "fixture_key": None,
        },
    ]

    with pytest.raises(m.HarvestError, match="RuntimeError"):
        _check(
            tmp_path,
            returncode=1,
            failures=failures,
            harvest_keys={"Extraction:aaa"},
        )


# ---------------------------------------------------------------------------
# _check_sweep_outcome
#
# scripts/sweep_corpus.py's own exit code is always 0, so it cannot signal
# that a golden's run raised and was skipped -- that signal exists only in
# the outcome summary main() writes to SWEEP_OUTCOME_PATH_ENV's path. A
# golden's remaining FakeLLM calls -- the ones its exception preempted --
# never reach the harvest file, so a `reached` set built from a pass that
# skipped one is missing exactly the keys that golden would have requested.
# ---------------------------------------------------------------------------


def test_check_sweep_outcome_accepts_a_clean_run(tmp_path):
    outcome_path = tmp_path / "sweep_outcome.json"
    _write_sweep_outcome(outcome_path, attempted=5, completed=5, failures=[])

    assert (
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))
        is None
    )


def test_check_sweep_outcome_refuses_a_failed_golden(tmp_path):
    """A skipped golden appears in the summary and blocks the pass."""
    outcome_path = tmp_path / "sweep_outcome.json"
    _write_sweep_outcome(
        outcome_path,
        attempted=5,
        completed=4,
        failures=[{"gid": "golden-b", "error_class": "ValueError", "message": "boom"}],
    )

    with pytest.raises(m.HarvestError, match="golden-b"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


def _missing_fixture_failure(gid: str) -> dict:
    return {"gid": gid, "error_class": MissingFixtureError.__name__, "message": "missing"}


def test_check_sweep_outcome_accepts_goldens_that_stopped_on_a_captured_miss(tmp_path):
    """A golden that raised `MissingFixtureError` is a miss to collect, not a skipped golden."""
    outcome_path = tmp_path / "sweep_outcome.json"
    _write_sweep_outcome(
        outcome_path,
        attempted=3,
        completed=1,
        failures=[_missing_fixture_failure("a"), _missing_fixture_failure("b")],
    )
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.write_text("")
    _write_miss_line(harvest_path, "Extraction:aaa")

    assert m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=harvest_path) is None


def test_check_sweep_outcome_refuses_a_missing_fixture_that_left_no_miss_line(tmp_path):
    outcome_path = tmp_path / "sweep_outcome.json"
    _write_sweep_outcome(
        outcome_path, attempted=2, completed=1, failures=[_missing_fixture_failure("a")]
    )

    with pytest.raises(m.HarvestError, match="no miss line"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


def test_check_sweep_outcome_refuses_a_missing_fixture_mixed_with_another_failure(tmp_path):
    outcome_path = tmp_path / "sweep_outcome.json"
    _write_sweep_outcome(
        outcome_path,
        attempted=3,
        completed=1,
        failures=[
            _missing_fixture_failure("a"),
            {"gid": "golden-b", "error_class": "ValueError", "message": "boom"},
        ],
    )
    harvest_path = tmp_path / "misses.jsonl"
    harvest_path.write_text("")
    _write_miss_line(harvest_path, "Extraction:aaa")

    with pytest.raises(m.HarvestError, match="golden-b"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=harvest_path)


def test_check_sweep_outcome_refuses_a_completed_count_short_of_attempted(tmp_path):
    """A completed count below attempted refuses the pass, even without failures."""
    outcome_path = tmp_path / "sweep_outcome.json"
    _write_sweep_outcome(outcome_path, attempted=5, completed=3, failures=[])

    with pytest.raises(m.HarvestError, match="3 of 5"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


def test_check_sweep_outcome_refuses_an_absent_summary(tmp_path):
    """A missing sweep outcome summary refuses the pass."""
    outcome_path = tmp_path / "sweep_outcome.json"

    with pytest.raises(m.HarvestError, match="missing or unreadable"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


def test_check_sweep_outcome_refuses_a_summary_that_is_not_valid_json(tmp_path):
    outcome_path = tmp_path / "sweep_outcome.json"
    outcome_path.write_text('{"schema_version": 1, "attempted": 1, "completed": 1, "failures": [')

    with pytest.raises(m.HarvestError, match="missing or unreadable"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


def test_check_sweep_outcome_refuses_a_summary_that_is_not_an_object(tmp_path):
    outcome_path = tmp_path / "sweep_outcome.json"
    outcome_path.write_text("[]")

    with pytest.raises(m.HarvestError, match="not a JSON object"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


def test_check_sweep_outcome_refuses_a_summary_at_an_unknown_schema_version(tmp_path):
    outcome_path = tmp_path / "sweep_outcome.json"
    outcome_path.write_text(
        json.dumps({"schema_version": 99, "attempted": 1, "completed": 1, "failures": []})
    )

    with pytest.raises(m.HarvestError, match="schema_version"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


def test_check_sweep_outcome_refuses_a_failures_field_that_is_not_a_list(tmp_path):
    outcome_path = tmp_path / "sweep_outcome.json"
    outcome_path.write_text(
        json.dumps(
            {
                "schema_version": m.SWEEP_OUTCOME_SCHEMA_VERSION,
                "attempted": 1,
                "completed": 1,
                "failures": {},
            }
        )
    )

    with pytest.raises(m.HarvestError, match="attempted/completed ints and failures list"):
        m._check_sweep_outcome(outcome_path=outcome_path, harvest_file=_empty_harvest(tmp_path))


class _FakePassRunner:
    """Records commands and writes fake sweep and pytest outcomes.
    An empty `sweep_failures` list means every golden succeeded."""

    def __init__(
        self,
        *,
        pytest_returncode: int,
        miss_keys: list[str],
        failure_type: str,
        sweep_failures: list[dict] | None = None,
        sweep_attempted: int = 1,
    ):
        self.pytest_returncode = pytest_returncode
        self.miss_keys = miss_keys
        self.failure_type = failure_type
        self.sweep_failures = sweep_failures or []
        self.sweep_attempted = sweep_attempted
        self.commands: list[list[str]] = []
        self.envs: list[dict] = []

    def __call__(self, *, cmd, env, ok_returncodes):
        self.commands.append(list(cmd))
        self.envs.append(dict(env))
        if "pytest" not in cmd:
            # The sweep_corpus.py collector, which always exits 0 -- its own
            # completeness signal is the outcome summary, not the returncode.
            _write_sweep_outcome(
                env[m.SWEEP_OUTCOME_PATH_ENV],
                attempted=self.sweep_attempted,
                completed=self.sweep_attempted - len(self.sweep_failures),
                failures=self.sweep_failures,
            )
            return subprocess.CompletedProcess(args=list(cmd), returncode=0)
        for key in self.miss_keys:
            _write_miss_line(env[m.HARVEST_MISS_PATH_ENV], key)
        # A green run records no failure, matching the exit code the caller
        # asked for -- a summary that disagrees with its own rc is read as a
        # disagreement by _check_pytest_outcome, which is a different test.
        failures = (
            [
                {
                    "nodeid": "tests/x.py::test_a",
                    "when": "call",
                    "exception_type": self.failure_type,
                    "fixture_key": self.miss_keys[0] if self.miss_keys else None,
                }
            ]
            if self.pytest_returncode == 1
            else []
        )
        _write_outcome(env[OUTCOME_PATH_ENV], failures)
        return subprocess.CompletedProcess(args=list(cmd), returncode=self.pytest_returncode)


def test_live_collect_pass_reads_the_misses_off_a_converged_pytest_run(tmp_path, monkeypatch):
    """The pass the harvest expects: pytest exited 1 on one missing fixture,
    which the run captured, so the pass reports that miss for resolution."""
    runner = _FakePassRunner(
        pytest_returncode=1, miss_keys=["Extraction:aaa"], failure_type=MissingFixtureError.__name__
    )
    monkeypatch.setattr(m, "_run_pass_subprocess", runner)
    monkeypatch.setattr(m, "_direction_corpus_collect", lambda *, harvest_path: None)

    obs = m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=["-k", "engine"])

    assert set(obs.misses) == {"Extraction:aaa"}
    assert obs.requested == frozenset({"Extraction:aaa"})
    # pytest ran with the outcome plugin loaded, xdist off, and its own
    # summary path. `python -m pytest`, because the plugin's dotted path is
    # resolved before any conftest runs and only `python -m` puts the
    # subprocess's working directory (this repo's root) on sys.path.
    pytest_cmd = next(cmd for cmd in runner.commands if "pytest" in cmd)
    assert pytest_cmd[:6] == ["uv", "run", "python", "-m", "pytest", "-q"]
    # Each `-p` flag is an adjacent pair, so this pins the plugin being
    # loaded and xdist being off rather than a list position.
    assert pytest_cmd[6:8] == ["-p", "no:xdist"]
    assert pytest_cmd[8:10] == ["-p", "tests.fixtures._harvest_outcome_plugin"]
    assert pytest_cmd[10:] == ["-k", "engine"]


@pytest.mark.parametrize(
    "var, value", [("PYTEST_ADDOPTS", "-k not_slow"), ("PYTEST_PLUGINS", "some_plugin")]
)
def test_live_collect_pass_strips_inherited_pytest_collection_env(
    tmp_path, monkeypatch, capsys, var, value
):
    """PYTEST_ADDOPTS and PYTEST_PLUGINS narrow or otherwise change pytest
    collection independent of any CLI argument, but main()'s decision to
    skip pruning trusts `--pytest-args` alone to know whether a run was
    narrowed. An inherited value for either must not reach any collector
    subprocess (sweep_corpus.py included, since the same env is shared), and
    the caller is told it was dropped rather than losing effect silently."""
    runner = _FakePassRunner(pytest_returncode=0, miss_keys=[], failure_type="")
    monkeypatch.setattr(m, "_run_pass_subprocess", runner)
    monkeypatch.setattr(m, "_direction_corpus_collect", lambda *, harvest_path: None)
    monkeypatch.setenv(var, value)

    m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=[])

    assert runner.envs, "the fake subprocess runner was never called"
    for call_env in runner.envs:
        assert var not in call_env
    assert f"{var}={value!r}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "forwarded",
    ["-n auto", "-nauto", "-n4", "-n=4", "--numprocesses=2", "--numprocesses 2", "--dist loadfile"],
)
def test_live_collect_pass_refuses_forwarded_xdist_args(tmp_path, monkeypatch, forwarded):
    """xdist keeps the plugin's recording in worker processes and re-fires
    only pytest_runtest_logreport on the controller, whose serialised report
    carries no excinfo -- so the controller's summary records no failing
    test and the harvest would abort on a pass that missed fixtures. The
    flag is refused by name, with no subprocess double installed, so the
    refusal is provably reached before any subprocess could run."""
    monkeypatch.setattr(m, "_run_pass_subprocess", lambda **kwargs: pytest.fail("a subprocess ran"))

    with pytest.raises(m.HarvestError, match="xdist"):
        m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=forwarded.split())


def test_live_collect_pass_allows_forwarded_args_that_are_not_xdist(tmp_path, monkeypatch):
    """The refusal is narrow: an ordinary selection argument still runs."""
    runner = _FakePassRunner(pytest_returncode=0, miss_keys=[], failure_type="")
    monkeypatch.setattr(m, "_run_pass_subprocess", runner)
    monkeypatch.setattr(m, "_direction_corpus_collect", lambda *, harvest_path: None)

    obs = m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=["-k", "x"])

    assert obs.misses == {}
    assert len(runner.commands) == 2


def test_live_collect_pass_refuses_a_run_broken_by_an_unrelated_failure(tmp_path, monkeypatch):
    """An AssertionError in the suite exits 1 and writes no miss line, which
    the old exit-code check read as a converged pass. It must raise here,
    before the direction driver runs and before any LLM call."""
    runner = _FakePassRunner(pytest_returncode=1, miss_keys=[], failure_type="AssertionError")
    monkeypatch.setattr(m, "_run_pass_subprocess", runner)
    direction_ran: list[bool] = []
    monkeypatch.setattr(
        m, "_direction_corpus_collect", lambda *, harvest_path: direction_ran.append(True)
    )

    with pytest.raises(m.HarvestError, match="AssertionError"):
        m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=[])

    assert direction_ran == []


def test_live_collect_pass_refuses_a_pytest_exit_code_other_than_zero_or_one(tmp_path, monkeypatch):
    """rc 2 (interrupted) never reached the old ok_returncodes=(0, 1) check
    either, but the point here is that it is refused for the pass's own
    reason -- the suite did not complete -- and not by the subprocess
    wrapper's generic exit-code message."""
    runner = _FakePassRunner(
        pytest_returncode=2, miss_keys=["Extraction:aaa"], failure_type=MissingFixtureError.__name__
    )
    monkeypatch.setattr(m, "_run_pass_subprocess", runner)
    monkeypatch.setattr(m, "_direction_corpus_collect", lambda *, harvest_path: None)

    with pytest.raises(m.HarvestError, match="neither 0"):
        m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=[])


def test_live_collect_pass_refuses_a_sweep_pass_that_skipped_a_golden(tmp_path, monkeypatch):
    """The exact hole this fix closes: sweep_corpus.py's own per-golden loop
    catches and skips any golden whose run raised, then the script still
    exits 0 (see its module docstring). Before this fix nothing here read
    that skip, so the pass's `reached` set silently omitted whatever that
    golden's remaining FakeLLM calls would have requested, and a later prune
    could drop those still-live fixtures. It must be refused here, before
    pytest or the direction driver runs."""
    runner = _FakePassRunner(
        pytest_returncode=0,
        miss_keys=[],
        failure_type="",
        sweep_failures=[{"gid": "golden-b", "error_class": "ValueError", "message": "boom"}],
        sweep_attempted=2,
    )
    monkeypatch.setattr(m, "_run_pass_subprocess", runner)
    direction_ran: list[bool] = []
    monkeypatch.setattr(
        m, "_direction_corpus_collect", lambda *, harvest_path: direction_ran.append(True)
    )

    with pytest.raises(m.HarvestError, match="golden-b"):
        m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=[])

    # Refused right after the sweep subprocess -- pytest never ran.
    assert len(runner.commands) == 1
    assert direction_ran == []


def test_live_collect_pass_refuses_a_sweep_pass_with_no_outcome_file(tmp_path, monkeypatch):
    """A sweep subprocess double that exits 0 without writing an outcome
    file at all -- e.g. a sweep_corpus.py predating this fix -- must not be
    read as a clean pass; a missing outcome fails closed exactly like a
    missing pytest_outcome.json does."""

    def _sweep_only_no_outcome(*, cmd, env, ok_returncodes):
        if "pytest" not in cmd:
            return subprocess.CompletedProcess(args=list(cmd), returncode=0)
        pytest.fail("pytest ran despite a missing sweep outcome")

    monkeypatch.setattr(m, "_run_pass_subprocess", _sweep_only_no_outcome)

    with pytest.raises(m.HarvestError, match="missing or unreadable"):
        m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=[])


def test_live_collect_pass_prunes_normally_on_a_clean_sweep_outcome(tmp_path, monkeypatch):
    """The non-regression: a sweep pass that completed every attempted
    golden still lets the pass through to the pytest collector and back, the
    same as before this fix."""
    runner = _FakePassRunner(
        pytest_returncode=1,
        miss_keys=["Extraction:aaa"],
        failure_type=MissingFixtureError.__name__,
        sweep_failures=[],
        sweep_attempted=5,
    )
    monkeypatch.setattr(m, "_run_pass_subprocess", runner)
    monkeypatch.setattr(m, "_direction_corpus_collect", lambda *, harvest_path: None)

    obs = m._live_collect_pass(harvest_path=tmp_path / "misses.jsonl", pytest_args=[])

    assert set(obs.misses) == {"Extraction:aaa"}
    assert len(runner.commands) == 2


def test_harvest_runs_the_collector_off_the_event_loop():
    """The direction collector calls `asyncio.run`, which fails on a thread with a running loop."""
    miss = _miss(Extraction, "sys", "prompt")
    passes = iter([_obs({miss.key: miss}), _obs({})])

    async def _noop() -> None:
        return None

    def _collect() -> m.PassObservations:
        asyncio.run(_noop())
        return next(passes)

    caller = _FakeCaller(
        {miss.key: Extraction(kind="resolution", variables=[], entities=[], dates=[])}
    )
    result = asyncio.run(
        m.harvest(existing={}, llm=caller, collect_pass=_collect, write_bank=lambda bank: None)
    )

    assert result.added_per_pass == [1, 0]


def test_collector_env_removes_credentials_and_defaults_the_offline_surfaces(tmp_path, monkeypatch):
    for var in ("COVERAGE_ARTIFACT_URI", "COVERAGE_RECHECK_SECONDS", "ORIGIN_CATALOGUE_URI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setenv("SEARCH_GRAPH_TOKEN", "secret")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")

    env = m._collector_env(harvest_path=tmp_path / "misses.jsonl")

    assert not {"GEMINI_API_KEY", "SEARCH_GRAPH_TOKEN", "LANGFUSE_SECRET_KEY"} & set(env)
    assert env[m.HARVEST_MISS_PATH_ENV] == str(tmp_path / "misses.jsonl")
    assert env["COVERAGE_ARTIFACT_URI"] == f"file://{m._FIXTURES_DIR / 'coverage_slice'}"
    assert env["COVERAGE_RECHECK_SECONDS"] == "0"
    assert env["ORIGIN_CATALOGUE_URI"] == f"file://{tmp_path / 'origin-catalogue'}"
    assert (tmp_path / "origin-catalogue").is_dir()


def test_collector_env_can_be_built_again_for_the_next_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("ORIGIN_CATALOGUE_URI", raising=False)
    harvest_path = tmp_path / "misses.jsonl"

    first = m._collector_env(harvest_path=harvest_path)
    second = m._collector_env(harvest_path=harvest_path)

    assert first["ORIGIN_CATALOGUE_URI"] == second["ORIGIN_CATALOGUE_URI"]


def test_collector_env_keeps_surfaces_the_environment_already_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", "file:///elsewhere")
    monkeypatch.setenv("ORIGIN_CATALOGUE_URI", "file:///catalogue")

    env = m._collector_env(harvest_path=tmp_path / "misses.jsonl")

    assert env["COVERAGE_ARTIFACT_URI"] == "file:///elsewhere"
    assert env["ORIGIN_CATALOGUE_URI"] == "file:///catalogue"
    assert not (tmp_path / "origin-catalogue").exists()


# ---------------------------------------------------------------------------
# _prune_bank / _prune_with_safeguards
#
# A fake bank and fake pass results throughout -- no live call, no
# dependence on the real tests/fixtures/llm_responses.json. These pin the
# fix for the run that emptied the real bank down to one key: the reachable
# set must be what a pass REQUESTED (hit + miss), never the pass's miss set
# alone, which is empty by definition once a harvest has converged.
# ---------------------------------------------------------------------------


def _fake_bank(n: int, *, schema: str = "Extraction") -> dict[str, dict]:
    return {f"{schema}:{i:03d}": {"variables": [f"v{i}"]} for i in range(n)}


def test_prune_bank_drops_nothing_when_every_key_is_reached():
    bank = {**_fake_bank(3), "_comment": "metadata, never a fixture"}

    pruned, dropped = m._prune_bank(bank, reached=frozenset(_fake_bank(3)))

    assert pruned == bank
    assert dropped == {}


def test_prune_bank_drops_exactly_the_unreached_complement():
    bank = _fake_bank(5)
    reached = frozenset({"Extraction:000", "Extraction:002", "Extraction:004"})

    pruned, dropped = m._prune_bank(bank, reached=reached)

    assert set(pruned) == reached
    assert dropped == {"Extraction": 2}


def test_prune_bank_never_drops_a_key_with_no_colon():
    """The bank's own "_comment" metadata string names no schema/prompt
    pair, so nothing "reaches" it -- it must survive an empty reached set,
    the exact scenario that wiped the real bank when this was still wrong."""
    bank = {"_comment": "metadata", **_fake_bank(2)}

    pruned, dropped = m._prune_bank(bank, reached=frozenset())

    assert pruned == {"_comment": "metadata"}
    assert dropped == {"Extraction": 2}


def test_prune_with_safeguards_drops_nothing_when_reached_covers_the_bank(tmp_path):
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(4)

    pruned, dropped = m._prune_with_safeguards(bank, reached=frozenset(bank), bank_path=bank_path)

    assert pruned == bank
    assert dropped == {}


def test_prune_with_safeguards_proceeds_on_a_legitimate_large_drop(tmp_path):
    """A real prompt change orphans most of the bank -- 8 of 10 keys here,
    an 80% drop -- with the reachable set correctly small and nothing newly
    recorded among the dropped keys. This must proceed, not refuse: this is
    exactly the case the old 25%-ceiling design got backwards (it refused
    twice on real harvests where the drop was entirely correct)."""
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(10)
    reached = frozenset({"Extraction:000", "Extraction:001"})

    pruned, dropped = m._prune_with_safeguards(bank, reached=reached, bank_path=bank_path)

    assert set(pruned) == {"Extraction:000", "Extraction:001"}
    assert dropped == {"Extraction": 8}


def test_prune_with_safeguards_empty_reached_set_refuses_rather_than_emptying_the_bank(tmp_path):
    """The exact scenario that shipped: a converged pass's miss set is empty
    by definition, so if that were ever passed as `reached` again, this must
    refuse instead of writing an empty bank."""
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(6)

    with pytest.raises(m.PruneRefused, match="empty"):
        m._prune_with_safeguards(bank, reached=frozenset(), bank_path=bank_path)

    # Refusing must not have touched the real bank path.
    assert not bank_path.exists()


def test_prune_with_safeguards_refuses_when_a_newly_recorded_key_would_be_dropped(tmp_path):
    """The invariant that replaces the percentage ceiling: a key THIS run
    just resolved and paid for can never legitimately be in the drop set,
    since resolving it required it to have been requested (a miss) on the
    very pass that became `reached`. If it ends up dropped anyway, the
    reachable-set accounting itself is broken -- refuse rather than
    silently discarding paid-for work, regardless of how small the drop is."""
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(4)
    # Extraction:000 is reached (survives on its own) but NOT in `reached`
    # here, simulating a bug in how `reached` was assembled.
    reached = frozenset({"Extraction:001", "Extraction:002", "Extraction:003"})
    newly_recorded = frozenset({"Extraction:000"})

    with pytest.raises(m.PruneRefused, match="Extraction:000"):
        m._prune_with_safeguards(
            bank, reached=reached, newly_recorded=newly_recorded, bank_path=bank_path
        )


def test_prune_with_safeguards_refuses_past_the_sanity_bound(tmp_path):
    """20 keys, only 1 reached -> a 95% drop, past the much-higher 90%
    backstop -- refuses even though no newly-recorded key is involved,
    covering the one gap the two invariant predicates cannot: a run that
    resolves nothing new (so the newly-recorded check is a no-op) whose
    collection scope was accidentally narrowed."""
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(20)

    with pytest.raises(m.PruneRefused, match="90%"):
        m._prune_with_safeguards(bank, reached=frozenset({"Extraction:000"}), bank_path=bank_path)


def test_prune_with_safeguards_writes_a_sidecar_holding_the_pre_prune_bank(tmp_path):
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(4)
    # Drop 1 of 4 (25%, comfortably under every bound) -- no refusal here.
    reached = frozenset({"Extraction:000", "Extraction:001", "Extraction:002"})

    m._prune_with_safeguards(bank, reached=reached, bank_path=bank_path)

    sidecar = m._sidecar_path(bank_path)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == bank


def test_prune_with_safeguards_writes_the_sidecar_even_when_it_refuses(tmp_path):
    """The pre-prune bank must be recoverable from the sidecar even when the
    prune itself refuses -- the sidecar is the safety net for exactly the
    case where the reachable set turns out to be wrong."""
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(6)

    with pytest.raises(m.PruneRefused):
        m._prune_with_safeguards(bank, reached=frozenset(), bank_path=bank_path)

    sidecar = m._sidecar_path(bank_path)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == bank


def test_main_skips_pruning_when_pytest_args_narrows_collection(tmp_path, monkeypatch, capsys):
    """Filtered pytest arguments make `final_pass_keys` incomplete, so `main()` skips pruning."""
    bank_path = tmp_path / "llm_responses.json"
    bank = _fake_bank(2)
    bank_path.write_text(json.dumps(bank))
    monkeypatch.setattr(m, "_BANK_PATH", bank_path)

    written: list[dict] = []
    monkeypatch.setattr(m, "_write_bank", lambda pruned, *, path=None: written.append(pruned))

    class _FakeLLM:
        def __init__(self) -> None:
            pass

    monkeypatch.setattr("search.engine.clients.llm.LLM", _FakeLLM)

    result = m.HarvestResult(
        bank=bank,
        added_per_pass=[0],
        passes_run=1,
        final_pass_keys=frozenset({"Extraction:000"}),
        newly_recorded=frozenset(),
    )

    async def _fake_harvest(**kwargs):
        return result

    monkeypatch.setattr(m, "harvest", _fake_harvest)
    monkeypatch.setattr(sys, "argv", ["harvest_llm_fixtures.py", "--pytest-args", "-k engine"])

    m.main()

    assert written == []  # _write_bank never called -- nothing was pruned
    assert json.loads(bank_path.read_text()) == bank  # both keys survive on disk
    assert not m._sidecar_path(bank_path).exists()  # _prune_with_safeguards never ran
    assert "Pruning skipped" in capsys.readouterr().out
