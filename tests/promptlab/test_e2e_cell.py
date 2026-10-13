"""Runs the production rung offline against three fixture-backed corpus queries.

The production rung composes prompts byte-identical to the shipped constants, so the recorded
FakeLLM fixtures answer them; a synthetic rung would not. The cases (maternal-mortality, sdg-03, sdg-04)
are single-leg queries that resolve cleanly against the offline graph fixtures.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import yaml

from promptlab.e2e_cell import _aggregate, _check_rates, _score_context, run_e2e_cell
from promptlab.graphcache import GraphCache
from promptlab.manifest import PROMPTLAB_ROOT, Rung, load_rung
from promptlab.store import (
    CaptureManifest,
    CaptureStore,
    CellManifest,
    CellStore,
    ExtractRecord,
)
from search.eval.cases import Case, validate_case
from search.eval.checks import score_results
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM

_CASES_PATH = Path(__file__).parent / "fixtures" / "e2e_cases.yaml"


def _load_cases() -> list[Case]:
    return [validate_case(raw, families={}) for raw in yaml.safe_load(_CASES_PATH.read_text())]


_CASES = _load_cases()
_CASE_QUERIES: dict[str, str] = {case.id: case.query for case in _CASES}
_CASE_IDS: tuple[str, ...] = tuple(_CASE_QUERIES)

_EMPTY_SDMX = FakeSdmxClient(observations={}, availability={})


class _SimulatedKill(RuntimeError):
    """Raised by _FailingLLM to simulate a driver process dying mid-run."""


class _FailingLLM:
    """Wraps FakeLLM, counting generate_structured calls and raising
    _SimulatedKill once `fail_after` calls have already succeeded -- a
    controlled stand-in for a driver process that dies mid-`asyncio.gather`,
    after some calls already landed in `out` but before `finish()` runs."""

    def __init__(self, inner: FakeLLM, *, fail_after: int) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self.calls = 0

    async def generate_structured(self, *, prompt, system, schema):
        self.calls += 1
        if self.calls > self._fail_after:
            raise _SimulatedKill(f"simulated kill on call {self.calls}")
        return await self._inner.generate_structured(prompt=prompt, system=system, schema=schema)


def _production_rung(seeds: tuple[int, ...]) -> Rung:
    """The real production rung, with run.seeds overridden to keep tests
    small. extract/bind block lists are untouched, so compose() still
    reproduces the shipped prompts byte for byte."""
    rung = load_rung(
        PROMPTLAB_ROOT / "rungs" / "production-extract.json",
        PROMPTLAB_ROOT / "rungs" / "production-bind.json",
    )
    return dataclasses.replace(rung, run=dataclasses.replace(rung.run, seeds=seeds))


def _make_capture(root: Path, *, case_ids: tuple[str, ...] = _CASE_IDS) -> CaptureStore:
    """Build a finished capture at the date pinned in the FakeLLM fixtures."""
    manifest = CaptureManifest(
        capture_id="test-capture",
        git_sha="deadbeef",
        engine_build_id="test-build",
        model="gemini-3.5-flash-lite",
        top_k=8,
        substituted_date=PINNED_DATE.isoformat(),
        graph_base="https://graph.example",
        corpus_sha256="0" * 64,
        seeds=(1, 2, 3),
        extract_system_sha1="e" * 40,
        bind_system_sha1="b" * 40,
    )
    store = CaptureStore.create(root, manifest=manifest)
    for case_id in case_ids:
        store.append_extract(
            ExtractRecord(
                case_id=case_id,
                query=_CASE_QUERIES[case_id],
                system_sha1="unused",
                output={},
                usage=None,
                elapsed_ms=0,
            )
        )
    store.finish()
    return store


def _make_cell_store(root: Path, rung: Rung, *, cell_id: str = "e2e-test") -> CellStore:
    manifest = CellManifest(
        cell_id=cell_id,
        rung=dataclasses.asdict(rung),
        capture_id="test-capture",
        git_sha="deadbeef",
        model=rung.run.model,
        seeds=rung.run.seeds,
        extract_system="unused",
        bind_system="unused",
    )
    return CellStore.create(root, manifest=manifest)


def _graph_cache(root: Path) -> GraphCache:
    root.mkdir(parents=True, exist_ok=True)
    return GraphCache(FakeGraph(), path=root / "graph.sqlite")


def test_run_e2e_cell_runs_end_to_end_offline(tmp_path):
    rung = _production_rung(seeds=(1, 2))
    capture = _make_capture(tmp_path / "capture")
    out = _make_cell_store(tmp_path / "cell", rung)
    graph = _graph_cache(tmp_path)

    asyncio.run(
        run_e2e_cell(
            rung,
            capture=capture,
            graph=graph,
            out=out,
            llm=FakeLLM(),
            sdmx=_EMPTY_SDMX,
            cases=_CASES,
        )
    )

    assert out.is_complete()
    assert out.done_keys() == {(case_id, 0, seed) for case_id in _CASE_IDS for seed in (1, 2)}

    records = {(r["case_id"], r["seed"]): r for r in out.call_records()}
    assert len(records) == len(_CASE_IDS) * 2
    for record in records.values():
        # Each case has one extract call and at least one bind call.
        stages = [c["stage"] for c in record["llm_calls"]]
        assert stages.count("extract") == 1
        assert stages.count("bind") >= 1
        assert record["checks"], "expected at least one check result to be recorded"

    scores = out.read_scores()
    assert set(scores["per_seed"]) == {"1", "2"}
    for seed_scores in scores["per_seed"].values():
        # Every case accepts the status the offline fixtures answer with.
        assert seed_scores["disposition_pass_rate"] == 1.0
    # Both seeds replay the same answer for each system and prompt.
    assert scores["seed_flip_rate"] == 0.0


def test_run_e2e_cell_resume_after_kill_matches_a_fresh_run(tmp_path):
    # One case, two seeds: with concurrency=1 the second seed's resolve
    # cannot start until the first's completes and appends, so "kill on the
    # 4th generate_structured call" deterministically means "seed 1's own
    # extract, bind and quantity read calls all succeeded and were persisted;
    # seed 2 never got as far as appending anything."
    case_id = _CASE_IDS[0]
    rung = _production_rung(seeds=(1, 2))
    killed_capture = _make_capture(tmp_path / "capture", case_ids=(case_id,))
    cell_root = tmp_path / "cell"
    out = _make_cell_store(cell_root, rung)
    graph = _graph_cache(tmp_path)

    failing_llm = _FailingLLM(FakeLLM(), fail_after=3)
    try:
        asyncio.run(
            run_e2e_cell(
                rung,
                capture=killed_capture,
                graph=graph,
                out=out,
                llm=failing_llm,
                sdmx=_EMPTY_SDMX,
                concurrency=1,
                cases=_CASES,
            )
        )
    except _SimulatedKill:
        pass
    else:
        raise AssertionError("expected the simulated kill to propagate")
    out.close()  # a real kill wouldn't call this either; append() already fsync'd.

    reopened = CellStore.open(cell_root)
    assert not reopened.is_complete()
    assert reopened.done_keys() == {(case_id, 0, 1)}

    resuming_capture = _make_capture(tmp_path / "capture-resume", case_ids=(case_id,))
    resuming_llm = _FailingLLM(FakeLLM(), fail_after=1000)  # never fails
    asyncio.run(
        run_e2e_cell(
            rung,
            capture=resuming_capture,
            graph=graph,
            out=reopened,
            llm=resuming_llm,
            sdmx=_EMPTY_SDMX,
            concurrency=1,
            cases=_CASES,
        )
    )
    assert resuming_llm.calls == 3, "seed 1 must not be re-run; only seed 2's 3 calls are new"
    assert reopened.is_complete()
    assert reopened.done_keys() == {(case_id, 0, 1), (case_id, 0, 2)}
    resumed_scores = reopened.read_scores()

    fresh_capture = _make_capture(tmp_path / "capture-fresh", case_ids=(case_id,))
    fresh_out = _make_cell_store(tmp_path / "cell-fresh", rung, cell_id="e2e-fresh")
    fresh_graph = _graph_cache(tmp_path / "fresh")
    asyncio.run(
        run_e2e_cell(
            rung,
            capture=fresh_capture,
            graph=fresh_graph,
            out=fresh_out,
            llm=FakeLLM(),
            sdmx=_EMPTY_SDMX,
            cases=_CASES,
        )
    )

    assert resumed_scores == fresh_out.read_scores()


def test_run_e2e_cell_records_the_checks_that_score_the_stored_output(tmp_path):
    """The recorded verdicts are what the eval checks give the stored response."""
    case = _CASES[0]
    rung = _production_rung(seeds=(1,))
    capture = _make_capture(tmp_path / "capture", case_ids=(case.id,))
    out = _make_cell_store(tmp_path / "cell", rung)
    graph = _graph_cache(tmp_path)

    asyncio.run(
        run_e2e_cell(
            rung,
            capture=capture,
            graph=graph,
            out=out,
            llm=FakeLLM(),
            sdmx=_EMPTY_SDMX,
            cases=_CASES,
        )
    )
    [record] = out.call_records()

    expected = score_results(case, record["output"], _score_context(graph, _EMPTY_SDMX))
    assert {name: r["verdict"] for name, r in record["checks"].items()} == {
        name: result.verdict.value for name, result in expected.items()
    }


def _record(case_id: str, seed: int, **verdicts: str) -> dict:
    return {
        "case_id": case_id,
        "seed": seed,
        "output": {},
        "checks": {name: {"verdict": verdict, "detail": ""} for name, verdict in verdicts.items()},
    }


def test_check_rates_count_only_pass_and_fail():
    records = [
        _record("c1", 1, measure="pass", entities="n/a", provenance="unverifiable"),
        _record("c2", 1, measure="fail", entities="pass", provenance="unverifiable"),
        _record("c3", 1, measure="pass"),
    ]

    assert _check_rates(records) == {"measure_pass_rate": 2 / 3, "entities_pass_rate": 1.0}


def test_aggregate_empty_seed_bucket_reports_no_rates():
    """`by_seed` is pre-seeded with an empty list for every configured seed; a seed that
    produced zero records reports no rate rather than a fabricated one."""
    records = (_record("g1", 1, measure="pass"),)

    scores = _aggregate(records, seeds=(1, 2))

    assert scores["per_seed"]["1"] == {"measure_pass_rate": 1.0}
    assert scores["per_seed"]["2"] == {}
