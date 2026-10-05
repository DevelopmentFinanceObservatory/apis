"""Pins latency selection, aggregation, artifact output, and CLI exits without live calls."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

import scripts.measure_corpus_latency as m
from search import Resolution, ResolveRequest, ResolveResponse
from search.eval.cases import Case, validate_case
from tests.conftest import base_response, minimal_spec
from tests.scripts.conftest import SKILLS_INDEX_SENTINEL


def _case(case_id: str, *, why: str = "A definite answer for a Kenya aid query.") -> Case:
    return validate_case(
        {
            "id": case_id,
            "query": "aid disbursements to Kenya",
            "why": why,
            "expect": {"definite": {}},
        },
        families={},
    )


def _response(
    status: str = "definite",
    *,
    warning_codes: list[str] | None = None,
    timed_out: bool | None = False,
) -> ResolveResponse:
    diagnostics = {
        "engine_build": "test-build",
        "trace_id": "test-trace",
        "warnings": [
            {"code": code, "severity": "warn", "message": f"{code} fired"}
            for code in (warning_codes or [])
        ],
        "timed_out": timed_out,
    }
    if status == "definite":
        parts = [{"status": "definite", "spec": minimal_spec("s1")}]
    else:
        parts = [{"status": "no_data", "no_data": {"reason": "variable_not_resolved"}}]
    return Resolution.model_validate(
        base_response(status=status, parts=parts, diagnostics=diagnostics)
    )


def _fake_resolve(outcomes):
    """A resolve callable returning each outcome in turn; an Exception is raised."""
    calls = iter(outcomes)

    async def resolve(request: ResolveRequest) -> ResolveResponse:
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return resolve


def _run(seconds: float, status: str = "definite", *, warning_codes=None, timed_out=False) -> dict:
    return {
        "seconds": seconds,
        "status": status,
        "warning_codes": list(warning_codes or []),
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# default_selection / select_cases
# ---------------------------------------------------------------------------


def test_default_selection_picks_only_cases_whose_why_has_a_timing_claim():
    cases = [
        _case("no-claim", why="Count_Person is confirmed on the graph."),
        _case("claim", why="Takes 67 to 98 seconds against the composed stack."),
        _case("deadline", why="Exceeds the service's 45-second request deadline."),
    ]

    assert [c.id for c in m.default_selection(cases)] == ["claim", "deadline"]


def test_select_cases_ids_replaces_the_default_selection_and_sorts_by_id():
    cases = [_case("claim", why="Takes 67 seconds."), _case("b"), _case("a")]

    selected = m.select_cases(cases, "b,a")

    assert [c.id for c in selected] == ["a", "b"]


def test_select_cases_none_uses_the_default_selection():
    cases = [_case("claim", why="Takes 67 to 98 seconds."), _case("other")]

    assert [c.id for c in m.select_cases(cases, None)] == ["claim"]


# ---------------------------------------------------------------------------
# summarize_runs
# ---------------------------------------------------------------------------


def test_summarize_runs_computes_the_min_and_max_wall_clock():
    runs = [_run(67.0), _run(98.5), _run(12.25)]

    row = m.summarize_runs(runs)

    assert row["seconds"] == [67.0, 98.5, 12.25]
    assert row["min_seconds"] == 12.25
    assert row["max_seconds"] == 98.5


def test_summarize_runs_flags_a_status_flip_between_runs():
    runs = [_run(1.0, "definite"), _run(1.0, "no_data"), _run(1.0, "definite")]

    row = m.summarize_runs(runs)

    assert row["status_flip"] is True
    assert row["statuses"] == ["definite", "no_data", "definite"]


def test_summarize_runs_ignores_errored_runs_when_deciding_a_status_flip():
    """An errored run is an outage, not the engine answering differently: four
    errors and one status is a single observed status, so status_flip is False."""
    runs = [_run(0.1, "error"), _run(0.2, "error"), _run(1.0, "definite")]

    row = m.summarize_runs(runs)

    assert row["status_flip"] is False
    assert row["statuses"] == ["error", "error", "definite"]


def test_summarize_runs_unions_warning_codes_and_any_timeout():
    runs = [
        _run(1.0, warning_codes=["retrieval_score_weak"]),
        _run(1.0, warning_codes=["query_qualifier_dropped", "retrieval_score_weak"]),
        _run(1.0, timed_out=True),
    ]

    row = m.summarize_runs(runs)

    assert row["warning_codes"] == ["query_qualifier_dropped", "retrieval_score_weak"]
    assert row["timed_out"] is True


def test_summarize_runs_of_no_runs_reports_zero_timings():
    row = m.summarize_runs([])

    assert row["min_seconds"] == 0.0
    assert row["max_seconds"] == 0.0
    assert row["status_flip"] is False


# ---------------------------------------------------------------------------
# probe_case: the injected resolve callable and per-run readings
# ---------------------------------------------------------------------------


def test_probe_case_records_status_warning_codes_and_timed_out():
    resolve = _fake_resolve(
        [
            _response("definite", warning_codes=["retrieval_score_weak"]),
            _response("definite", warning_codes=["retrieval_score_weak"], timed_out=True),
        ]
    )

    record = asyncio.run(m.probe_case(_case("g-01"), runs=2, resolve=resolve))

    assert record["statuses"] == ["definite", "definite"]
    assert record["warning_codes"] == ["retrieval_score_weak"]
    assert record["timed_out"] is True
    assert len(record["seconds"]) == 2
    assert all(seconds >= 0 for seconds in record["seconds"])
    assert record["min_seconds"] <= record["max_seconds"]


def test_probe_case_records_an_errored_run_and_keeps_going():
    resolve = _fake_resolve([RuntimeError("graph 500"), _response("definite")])

    record = asyncio.run(m.probe_case(_case("g-01"), runs=2, resolve=resolve))

    assert record["statuses"] == ["error", "definite"]
    assert record["runs"][0]["warning_codes"] == []
    assert record["runs"][0]["timed_out"] is False


def test_probe_case_builds_the_eval_runner_request_shape():
    """The request handed to resolve is the one the eval runner builds for this
    case, so a timing taken here describes the same call the gate makes."""
    seen: list[ResolveRequest] = []

    async def resolve(request: ResolveRequest) -> ResolveResponse:
        seen.append(request)
        return _response("definite")

    asyncio.run(m.probe_case(_case("g-01"), runs=1, resolve=resolve))

    assert len(seen) == 1
    dumped = seen[0].model_dump(mode="json")
    assert dumped["input"]["kind"] == "raw_text"
    assert dumped["input"]["query"] == "aid disbursements to Kenya"
    assert dumped["response_options"]["include_skills"] is True


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def _artifact(cases: dict, *, deadline: float = 45.0) -> dict:
    return {
        "measurement": m._MEASUREMENT,
        "runs_per_case": 2,
        "engine_build": "test-build",
        "deadline_seconds": deadline,
        "cases": cases,
    }


def test_render_report_names_a_case_whose_max_exceeds_the_deadline():
    artifact = _artifact(
        {
            "fast": m.summarize_runs([_run(2.0), _run(3.0)]),
            "slow": m.summarize_runs([_run(67.0), _run(98.0)]),
        },
        deadline=45.0,
    )

    report = m.render_report(artifact)

    assert "slow: max 98.00s exceeds 45.0s" in report
    assert "fast: max" not in report


def test_render_report_says_so_when_no_case_exceeds_the_deadline():
    artifact = _artifact({"fast": m.summarize_runs([_run(2.0), _run(3.0)])}, deadline=45.0)

    assert "no case exceeded the 45.0s deadline" in m.render_report(artifact)


def test_render_report_names_a_case_whose_status_changed_between_runs():
    artifact = _artifact(
        {
            "stable": m.summarize_runs([_run(1.0, "definite"), _run(1.0, "definite")]),
            "flipping": m.summarize_runs([_run(1.0, "definite"), _run(1.0, "no_data")]),
        }
    )

    report = m.render_report(artifact)

    assert "flipping: definite -> no_data" in report
    assert "stable: definite ->" not in report


def test_render_report_says_so_when_no_status_changed_between_runs():
    artifact = _artifact({"stable": m.summarize_runs([_run(1.0, "definite")] * 2)})

    assert "no case changed status between runs" in m.render_report(artifact)


# ---------------------------------------------------------------------------
# measure: artifact shape and byte stability
# ---------------------------------------------------------------------------


async def _stub_resolve(request: ResolveRequest) -> ResolveResponse:
    """A resolve callable that never runs: measure() is driven with a stubbed probe_case."""
    raise AssertionError("not called -- probe_case is stubbed in these tests")


def _stub_probe_case(records: dict[str, dict]):
    async def probe_case(case, *, runs, resolve):
        return records[case.id]

    return probe_case


def test_measure_writes_a_stable_artifact(tmp_path, monkeypatch):
    """Two identical measurements produce byte-identical artifacts, and the
    written JSON round-trips to the returned dict."""
    cases = [_case("b"), _case("a")]
    monkeypatch.setattr(
        m,
        "probe_case",
        _stub_probe_case(
            {
                "a": m.summarize_runs([_run(1.0), _run(2.0)]),
                "b": m.summarize_runs([_run(3.0, "no_data"), _run(4.0, "no_data")]),
            }
        ),
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    def run_once(out_path):
        return asyncio.run(
            m.measure(
                cases,
                runs=2,
                resolve=_stub_resolve,
                out_path=out_path,
                deadline_seconds=45.0,
                engine_build="test-build",
            )
        )

    artifact = run_once(first_path)
    run_once(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text()) == artifact
    assert sorted(artifact["cases"]) == ["a", "b"]
    assert artifact["measurement"] == m._MEASUREMENT
    assert artifact["runs_per_case"] == 2
    assert artifact["engine_build"] == "test-build"
    assert artifact["deadline_seconds"] == 45.0


def test_measure_writes_the_artifact_after_every_case(tmp_path, monkeypatch):
    """A crash between cases leaves the already-probed results on disk."""
    cases = [_case("g1"), _case("g2")]
    out_path = tmp_path / "latency.json"
    seen_before_g2: dict = {}
    records = {
        "g1": m.summarize_runs([_run(1.0)]),
        "g2": m.summarize_runs([_run(2.0)]),
    }

    async def probe_case(case, *, runs, resolve):
        if case.id == "g2":
            seen_before_g2.update(json.loads(out_path.read_text())["cases"])
        return records[case.id]

    monkeypatch.setattr(m, "probe_case", probe_case)

    asyncio.run(
        m.measure(
            cases,
            runs=1,
            resolve=_stub_resolve,
            out_path=out_path,
            deadline_seconds=45.0,
            engine_build="test-build",
        )
    )

    assert "g1" in seen_before_g2
    assert "g2" not in seen_before_g2


def test_measure_records_no_measurement_date(tmp_path, monkeypatch):
    """Wall-clock timings are machine-dependent, so the artifact pins what it
    ran against (the engine build) and never when."""
    monkeypatch.setattr(m, "probe_case", _stub_probe_case({"a": m.summarize_runs([])}))
    out_path = tmp_path / "latency.json"

    artifact = asyncio.run(
        m.measure(
            [_case("a")],
            runs=1,
            resolve=_stub_resolve,
            out_path=out_path,
            deadline_seconds=45.0,
            engine_build="test-build",
        )
    )

    assert "measured_at" not in artifact
    assert "measured_at" not in out_path.read_text()


# ---------------------------------------------------------------------------
# main: dry-run, empty selection, deadline default
# ---------------------------------------------------------------------------


def test_main_dry_run_lists_the_selected_ids_without_building_a_client(monkeypatch, capsys):
    monkeypatch.setattr(
        m,
        "load_cases",
        lambda: [_case("deferred-claim", why="Takes 67 to 98 seconds."), _case("other")],
    )
    monkeypatch.setattr(
        m,
        "_acquire_latency_resolvekit",
        lambda: pytest.fail("--dry-run must not acquire a resolvekit"),
    )
    monkeypatch.setattr(sys, "argv", ["measure_corpus_latency.py", "--dry-run"])

    m.main()

    out = capsys.readouterr().out
    assert "1 case(s) selected:" in out
    assert "  deferred-claim" in out


def test_main_exits_non_zero_on_an_empty_default_selection(monkeypatch):
    monkeypatch.setattr(m, "load_cases", lambda: [_case("other")])
    monkeypatch.setattr(sys, "argv", ["measure_corpus_latency.py"])

    with pytest.raises(SystemExit, match="no cases selected"):
        m.main()


def test_main_exits_non_zero_when_ids_match_nothing(monkeypatch):
    monkeypatch.setattr(m, "load_cases", lambda: [_case("other")])
    monkeypatch.setattr(sys, "argv", ["measure_corpus_latency.py", "--ids", "absent-case"])

    with pytest.raises(SystemExit, match="no cases selected"):
        m.main()


def test_main_measures_with_the_configured_deadline(
    tmp_path, monkeypatch, search_settings, fake_live_clients
):
    """With no --deadline-seconds, the deadline recorded is the engine's own
    configured resolve deadline."""
    search_settings(request_timeout_s=45.0)
    monkeypatch.setattr(m, "load_cases", lambda: [_case("deferred-claim", why="Takes 67 seconds.")])
    monkeypatch.setattr(m, "_acquire_latency_resolvekit", lambda: object())
    monkeypatch.setattr(m, "release_shared", lambda shared: None)
    monkeypatch.setattr(m, "probe_case", _stub_probe_case({"deferred-claim": m.summarize_runs([])}))
    out_path = tmp_path / "latency.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["measure_corpus_latency.py", "--out", str(out_path), "--runs", "1"],
    )

    m.main()

    assert json.loads(out_path.read_text())["deadline_seconds"] == 45.0


def test_main_records_a_deadline_seconds_override(tmp_path, monkeypatch, fake_live_clients):
    monkeypatch.setattr(m, "load_cases", lambda: [_case("deferred-claim", why="Takes 67 seconds.")])
    monkeypatch.setattr(m, "_acquire_latency_resolvekit", lambda: object())
    monkeypatch.setattr(m, "release_shared", lambda shared: None)
    monkeypatch.setattr(m, "probe_case", _stub_probe_case({"deferred-claim": m.summarize_runs([])}))
    out_path = tmp_path / "latency.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_corpus_latency.py",
            "--out",
            str(out_path),
            "--runs",
            "1",
            "--deadline-seconds",
            "20",
        ],
    )

    m.main()

    assert json.loads(out_path.read_text())["deadline_seconds"] == 20.0


def test_main_resolves_with_the_shared_skills_index(tmp_path, monkeypatch, fake_live_clients):
    """The eval request asks for skills, so the timed call carries the skills index built on
    the run's graph client and the timing includes skill matching."""
    monkeypatch.setattr(m, "load_cases", lambda: [_case("deferred-claim", why="Takes 67 seconds.")])
    monkeypatch.setattr(m, "_acquire_latency_resolvekit", lambda: object())
    monkeypatch.setattr(m, "release_shared", lambda shared: None)
    captured: dict = {}

    async def fake_measure(cases, *, runs, resolve, out_path, deadline_seconds, engine_build):
        captured["resolve"] = resolve
        return {"cases": {}}

    monkeypatch.setattr(m, "measure", fake_measure)
    monkeypatch.setattr(m, "render_report", lambda artifact: "")
    monkeypatch.setattr(
        sys, "argv", ["measure_corpus_latency.py", "--out", str(tmp_path / "latency.json")]
    )

    m.main()

    assert captured["resolve"].keywords["shared_skills_index"] is SKILLS_INDEX_SENTINEL
    assert fake_live_clients == ["graph-stub"]
