"""Checks capture, bind, and extract cells with fixture-backed dependencies."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from promptlab.bindmenu import MenuEntry
from promptlab.cells import (
    _aggregate_bind_scores,
    _aggregate_extract_scores,
    _assign_legs,
    _build_request,
    _shape_election_score,
    _shape_matches,
    _variable_split_matches,
    run_bind_cell,
    run_capture,
    run_extract_cell,
)
from promptlab.expectations import LegExpectation, leg_expectation
from promptlab.graphcache import GraphCache
from promptlab.llm import CallRecord
from promptlab.manifest import BlockRef, RunConfig, Rung, StageSpec
from promptlab.scoring import menu_five_tuple
from promptlab.store import BindRecord, CaptureManifest, CaptureStore, CellManifest, CellStore
from search.engine.errors import EngineInputError
from search.eval.cases import Case, load_cases, validate_case
from search.models import DatasetResubmitInput, RawTextInput
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM
from tests.fixtures.keys import fixture_key

_CASE_IDS = ("synth-tb-incidence-ethiopia", "life-expectancy", "conj-01")


def _cases(*case_ids: str) -> list[Case]:
    by_id = {c.id: c for c in load_cases()}
    return [by_id[case_id] for case_id in case_ids or _CASE_IDS]


def _raw_case(case_id: str, query: str) -> Case:
    return validate_case(
        {
            "id": case_id,
            "query": query,
            "why": "A query for a stubbed resolve.",
            "expect": {"definite": {}},
        },
        families={},
    )


def test_build_request_follows_the_case_door() -> None:
    raw, resubmit = _cases("life-expectancy")[0], _cases("spec-std-01")[0]

    assert isinstance(_build_request(raw).input, RawTextInput)
    assert isinstance(_build_request(resubmit).input, DatasetResubmitInput)


def _capture_manifest() -> CaptureManifest:
    return CaptureManifest(
        capture_id="test-capture",
        git_sha="abc123",
        engine_build_id="build-1",
        model="test-model",
        top_k=8,
        substituted_date=PINNED_DATE.isoformat(),
        graph_base="https://graph.example",
        corpus_sha256="deadbeef",
        seeds=(),
        extract_system_sha1="e" * 40,
        bind_system_sha1="b" * 40,
    )


async def _run_capture(tmp_path: Path, cases: list[Case], *, name: str = "capture") -> CaptureStore:
    store = CaptureStore.create(tmp_path / name, manifest=_capture_manifest())
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = None
        await run_capture(
            cases,
            out=store,
            llm=FakeLLM(),
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            concurrency=4,
        )
    store.finish()
    return store


_RUNG0 = Rung(
    name="rung0",
    description="test",
    extract=StageSpec(
        stage="extract",
        blocks=(BlockRef(id="r0-task"), BlockRef(id="r0-date"), BlockRef(id="r0-security")),
    ),
    bind=StageSpec(
        stage="bind",
        blocks=(
            BlockRef(id="r0-role"),
            BlockRef(id="r0-part1"),
            BlockRef(id="r0-part2"),
            BlockRef(id="r0-alternates"),
            BlockRef(id="r0-security"),
        ),
    ),
    run=RunConfig(
        capture="test-capture",
        top_k=3,
        seeds=(11, 12),
        model="test-model",
        thinking={"extract": None, "bind": "minimal"},
    ),
)


def _cell_manifest() -> CellManifest:
    return CellManifest(
        cell_id="test-cell",
        rung={"name": "rung0"},
        capture_id="test-capture",
        git_sha="abc123",
        model="test-model",
        seeds=_RUNG0.run.seeds,
        extract_system="",
        bind_system="",
    )


def _fake_llm_for_bind_cell(capture: CaptureStore, rung: Rung) -> FakeLLM:
    """Build a FakeLLM fixture for each captured, truncated bind prompt."""
    from promptlab.bindmenu import truncate_menu
    from promptlab.compose import compose
    from search.engine.bind import _UniversalBindOutput

    bind_system = compose(rung.bind)
    canned = _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[], ask=None)
    responses = {}
    for record in capture.bind_records():
        truncated = truncate_menu(record.prompt, rung.run.top_k)
        key = fixture_key(_UniversalBindOutput, bind_system, truncated)
        responses[key] = canned.model_dump(mode="json")
    return FakeLLM(responses=responses)


def _fake_llm_for_extract_cell(capture: CaptureStore, rung: Rung) -> FakeLLM:
    """Build a FakeLLM that replays each captured extraction for the rung."""
    from promptlab.compose import compose
    from search.engine.extract import Extraction

    extract_system = compose(rung.extract).replace("[[TODAY]]", PINNED_DATE.isoformat())
    responses = {}
    for record in capture.extract_records():
        key = fixture_key(Extraction, extract_system, record.query)
        responses[key] = record.output
    return FakeLLM(responses=responses)


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_menu_five_tuple_reads_the_five_menu_keys():
    entry = MenuEntry(
        index=0,
        label="x",
        five_tuple={
            "population_type": "PT",
            "measured_property": "MP",
            "stat_type": "ST",
            "measurement_qualifier": None,
            "measurement_denominator": None,
        },
        sample_dcids=(),
        retrieval_score=1.0,
    )
    assert menu_five_tuple(entry) == ("PT", "MP", "ST", None, None)


def _menu_entry(index: int, **five: str | None) -> MenuEntry:
    ft: dict[str, str | None] = {
        "population_type": None,
        "measured_property": None,
        "stat_type": None,
        "measurement_qualifier": None,
        "measurement_denominator": None,
        **five,
    }
    return MenuEntry(
        index=index, label=f"shape {index}", five_tuple=ft, sample_dcids=(), retrieval_score=1.0
    )


def _expectation(*shapes: dict, statuses=("definite",)) -> LegExpectation:
    return LegExpectation(statuses=statuses, shapes=tuple(shapes), stat_vars=())


_SHAPE = {"population_type": "PT", "measured_property": "MP", "stat_type": "ST"}
_EXPECTED_DEFINITE = _expectation(_SHAPE)


@pytest.mark.parametrize(
    ("population_type", "expected_score"),
    [
        pytest.param("PT", 1.0, id="matching"),
        pytest.param("OTHER", 0.0, id="one-field-mismatch"),
    ],
)
def test_shape_election_score_checks_population_type(population_type, expected_score):
    menu = [_menu_entry(0, population_type=population_type, measured_property="MP", stat_type="ST")]
    score = _shape_election_score({"chosen_shape_index": 0}, menu, _EXPECTED_DEFINITE)
    assert score == expected_score


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            {"chosen_shape_index": 0, "ask": "off-topic"},
            id="ask-with-index",
        ),
        pytest.param(
            {"chosen_shape_index": 5},
            id="out-of-range-index",
        ),
    ],
)
def test_shape_election_score_invalid_selection_is_a_miss(answer):
    menu = [_menu_entry(0, population_type="PT", measured_property="MP", stat_type="ST")]
    assert _shape_election_score(answer, menu, _EXPECTED_DEFINITE) == 0.0


def test_shape_election_score_accepts_any_of_the_expected_shapes():
    menu = [_menu_entry(0, population_type="B", measured_property="MP", stat_type="ST")]
    expected = _expectation(_SHAPE, {**_SHAPE, "population_type": "B"})

    assert _shape_election_score({"chosen_shape_index": 0}, menu, expected) == 1.0


@pytest.mark.parametrize(
    ("shape", "measurement_denominator", "matches"),
    [
        pytest.param(
            {"measurement_denominator": "PerCapita"}, "per person", True, id="schema-term"
        ),
        pytest.param(
            {"measurement_denominator": "Count_Person"},
            "per person",
            True,
            id="schema-term-other-spelling",
        ),
        pytest.param(
            {"measurement_denominator": "GiniIndex"}, "GiniIndex", True, id="statvar-dcid"
        ),
        pytest.param(
            {"measurement_denominator": "PerCapita"},
            "(present, unresolved)",
            True,
            id="asserted-denominator-matches-the-unresolved-sentinel",
        ),
        pytest.param({"measurement_denominator": "PerCapita"}, None, False, id="absent"),
        pytest.param({"measurement_denominator": None}, None, True, id="asserted-absent"),
        pytest.param(
            {"measurement_denominator": None}, "per person", False, id="asserted-absent-but-present"
        ),
        pytest.param({}, "per person", True, id="unasserted-dimension-is-not-compared"),
    ],
)
def test_shape_matches_compares_only_asserted_dimensions(shape, measurement_denominator, matches):
    entry = _menu_entry(0, measurement_denominator=measurement_denominator)

    assert _shape_matches(entry, shape) is matches


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(None, id="no-such-leg"),
        pytest.param(_expectation(statuses=("no_data",)), id="no-data-case"),
        pytest.param(_expectation(), id="no-expected-shape"),
    ],
)
def test_shape_election_score_is_none_without_a_shape_to_choose(expected):
    menu = [_menu_entry(0)]
    assert _shape_election_score({"chosen_shape_index": 0}, menu, expected) is None


def test_aggregate_bind_scores_computes_rates_and_means():
    records = [
        {"scores": {"shape_election_match": 1.0, "over_nomination_count": 0}},
        {"scores": {"shape_election_match": 0.0}},
        {"scores": {"over_nomination_count": 2}},
    ]
    agg = _aggregate_bind_scores(records)
    assert agg["n_calls"] == 3
    assert agg["shape_election_match_rate"] == 0.5
    assert agg["shape_election_scored"] == 2
    assert agg["over_nomination_count_mean"] == 1.0
    assert agg["over_nomination_scored"] == 2


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        pytest.param(
            ["GDP", "population"],
            ["population", "GDP"],
            True,
            id="reordered",
        ),
        pytest.param(
            ["population and GDP"],
            ["population", "GDP"],
            False,
            id="merged-phrases",
        ),
        pytest.param(
            ["population", "unrelated zzz"],
            ["population", "GDP"],
            False,
            id="unrelated-phrase",
        ),
        pytest.param(
            ["gdp per capita", "gdp growth"],
            ["gdp", "gdp per capita"],
            False,
            id="maximum-weight-matching",
        ),
    ],
)
def test_variable_split_matches_expected_phrases(actual, expected, matches):
    assert _variable_split_matches(actual, expected) is matches


def test_aggregate_extract_scores_computes_rates():
    records = [
        {
            "scores": {
                "date_structure_match": True,
                "shape_recall": 1.0,
                "variable_split_match": True,
            }
        },
        {
            "scores": {
                "date_structure_match": False,
                "shape_recall": 0.5,
                "variable_split_match": False,
            }
        },
    ]
    agg = _aggregate_extract_scores(records)
    assert agg["n_calls"] == 2
    assert agg["date_structure_match_rate"] == 0.5
    assert agg["shape_recall_mean"] == 0.75
    assert agg["variable_split_match_rate"] == 0.5


def _bind_call_record(variable: str) -> CallRecord:
    from search.engine.bind import _build_universal_bind_prompt
    from search.engine.shape import ShapeDraft

    shapes = [
        ShapeDraft(
            shape_id="s0",
            label="a shape",
            pop_type_dcid="Person",
            meas_prop_dcid="count",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            slot_taxonomy={},
            representative_score=1.0,
        )
    ]
    prompt = _build_universal_bind_prompt(variable, shapes)
    return CallRecord(
        stage="bind",
        schema="_UniversalBindOutput",
        system_sha1="s",
        prompt_sha1="p",
        prompt=prompt,
        output={},
        usage=None,
        elapsed_ms=1,
        seed=None,
        thinking_level=None,
    )


def test_assign_legs_matches_by_text_not_completion_order():
    call_gdp = _bind_call_record("GDP")
    call_population = _bind_call_record("population")
    # Bind calls complete in a different order from `variable_text`.
    pairs = _assign_legs([call_gdp, call_population], ["population", "GDP"])
    by_call = {id(call_gdp): None, id(call_population): None}
    for leg, call in pairs:
        by_call[id(call)] = leg
    assert by_call[id(call_gdp)] == 1  # GDP is variable_text[1]
    assert by_call[id(call_population)] == 0  # population is variable_text[0]


def test_assign_legs_raises_on_unmatched_phrase():
    call = _bind_call_record("nowhere")
    with pytest.raises(ValueError, match="does not match"):
        _assign_legs([call], ["population", "GDP"])


# ---------------------------------------------------------------------------
# run_capture
# ---------------------------------------------------------------------------


def test_run_capture_completes_end_to_end(tmp_path: Path):
    store = asyncio.run(_run_capture(tmp_path, _cases()))

    assert store.is_complete()
    extract_ids = {r.case_id for r in store.extract_records()}
    assert extract_ids == set(_CASE_IDS)

    bind_by_case: dict[str, list[int]] = {}
    for record in store.bind_records():
        bind_by_case.setdefault(record.case_id, []).append(record.leg)
    # The recorded conj-01 extraction names two variables.
    assert sorted(bind_by_case["synth-tb-incidence-ethiopia"]) == [0]
    assert sorted(bind_by_case["life-expectancy"]) == [0]
    assert sorted(bind_by_case["conj-01"]) == [0, 1]

    # conj-01's own extraction ordering the pipeline resolved this run.
    conj_extract = next(r for r in store.extract_records() if r.case_id == "conj-01")
    variable_text = conj_extract.output["variables"]
    conj_binds = {r.leg: r.variable for r in store.bind_records() if r.case_id == "conj-01"}
    assert conj_binds[0] == variable_text[0]
    assert conj_binds[1] == variable_text[1]


def test_run_capture_records_the_extract_of_a_catalogue_or_declined_answer(tmp_path: Path):
    """An answer with no parts keeps its extract record and makes no bind call."""
    cases = _cases("cat-what-data", "oos-sky-blue")

    store = asyncio.run(_run_capture(tmp_path, cases))

    assert store.is_complete()
    kinds = {r.case_id: r.output["kind"] for r in store.extract_records()}
    assert kinds == {"cat-what-data": "catalogue", "oos-sky-blue": "out_of_scope"}
    assert store.done_bind_keys() == set()


def test_run_capture_resume_has_no_duplicates_and_matches_a_fresh_run(tmp_path: Path):
    cases = _cases()

    # Close after one case to simulate an interrupted capture.
    partial_root = tmp_path / "partial"
    partial = CaptureStore.create(partial_root, manifest=_capture_manifest())
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = None
        asyncio.run(
            run_capture(
                cases[:1],
                out=partial,
                llm=FakeLLM(),
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                concurrency=4,
            )
        )
    partial.close()

    # Resume with all cases; the stored key must not be duplicated.
    resumed = CaptureStore.open(partial_root)
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = None
        asyncio.run(
            run_capture(
                cases,
                out=resumed,
                llm=FakeLLM(),
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                concurrency=4,
            )
        )
    resumed.finish()

    fresh = asyncio.run(_run_capture(tmp_path, cases, name="fresh"))

    assert resumed.status() == fresh.status()


def test_run_capture_resume_drops_stale_bind_before_a_different_split_lands(tmp_path: Path):
    """An orphaned bind record is removed before its missing extract is resumed."""
    cases = _cases()

    root = tmp_path / "killed-after-bind"
    store = CaptureStore.create(root, manifest=_capture_manifest())
    store.append_bind(
        BindRecord(
            case_id="conj-01",
            leg=0,
            variable="STALE-BOGUS-VARIABLE-FROM-A-KILLED-ATTEMPT",
            prompt="Variable phrase: STALE-BOGUS-VARIABLE-FROM-A-KILLED-ATTEMPT\n",
            prompt_sha1="s" * 40,
            system_sha1="b" * 40,
            output={"chosen_index": 0},
            usage=None,
            elapsed_ms=1,
        )
    )
    # Simulated kill: conj-01's extract record never became durable.
    store.close()

    # Reopening already repaired the store -- the orphaned leg-0 record for
    # conj-01 (a case with no durable extract record) must be gone before
    # run_capture ever sees it.
    resumed = CaptureStore.open(root)
    assert resumed.done_bind_keys() == set()

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = None
        asyncio.run(
            run_capture(
                cases,
                out=resumed,
                llm=FakeLLM(),
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                concurrency=4,
            )
        )
    resumed.finish()

    conj_extract = next(r for r in resumed.extract_records() if r.case_id == "conj-01")
    variable_text = conj_extract.output["variables"]
    conj_binds = {r.leg: r for r in resumed.bind_records() if r.case_id == "conj-01"}

    assert sorted(conj_binds) == [0, 1]
    for leg, record in conj_binds.items():
        assert record.variable == variable_text[leg]
        assert "STALE-BOGUS" not in record.variable
        assert "STALE-BOGUS" not in record.prompt


def test_run_capture_taskgroup_awaits_siblings_before_a_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failing task cancels and awaits its parked siblings before propagating."""
    started: list[str] = []
    finished: list[str] = []
    release = asyncio.Event()

    async def fake_resolve_async(request, *, graph, llm, sdmx):
        query = request.input.query
        started.append(query)
        try:
            if query == "boom-query":
                raise EngineInputError("refused", code="some_other_code")
            await release.wait()
            raise AssertionError(f"{query} should have been cancelled, not completed")
        finally:
            finished.append(query)

    monkeypatch.setattr("promptlab.cells.resolve_async", fake_resolve_async)

    cases = [
        _raw_case("waiter-a", "waiter-a-query"),
        _raw_case("waiter-b", "waiter-b-query"),
        _raw_case("boom", "boom-query"),
    ]
    store = CaptureStore.create(tmp_path / "capture", manifest=_capture_manifest())

    with pytest.raises(EngineInputError):
        asyncio.run(
            run_capture(
                cases,
                out=store,
                llm=FakeLLM(),
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                concurrency=3,
            )
        )

    assert set(started) == {"waiter-a-query", "waiter-b-query", "boom-query"}
    assert set(finished) == {"waiter-a-query", "waiter-b-query", "boom-query"}


def test_run_capture_facet_refusal_code_is_skipped_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Facet refusals are skipped without an LLM call or capture record."""

    async def fake_resolve_async(request, *, graph, llm, sdmx):
        raise EngineInputError("refused", code="facet_not_available")

    monkeypatch.setattr("promptlab.cells.resolve_async", fake_resolve_async)

    case = _raw_case("refused", "refused-query")
    store = CaptureStore.create(tmp_path / "capture", manifest=_capture_manifest())
    asyncio.run(
        run_capture(
            [case],
            out=store,
            llm=FakeLLM(),
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            concurrency=4,
        )
    )
    store.finish()

    assert store.is_complete()
    assert store.done_extract_keys() == set()
    assert store.done_bind_keys() == set()


def test_run_capture_non_facet_refusal_engine_input_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Engine input errors outside the facet-refusal set propagate."""

    async def fake_resolve_async(request, *, graph, llm, sdmx):
        raise EngineInputError("refused", code="some_other_code")

    monkeypatch.setattr("promptlab.cells.resolve_async", fake_resolve_async)

    case = _raw_case("broken", "broken-query")
    store = CaptureStore.create(tmp_path / "capture", manifest=_capture_manifest())
    with pytest.raises(EngineInputError) as exc_info:
        asyncio.run(
            run_capture(
                [case],
                out=store,
                llm=FakeLLM(),
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                concurrency=4,
            )
        )
    assert exc_info.value.code == "some_other_code"


# ---------------------------------------------------------------------------
# run_bind_cell
# ---------------------------------------------------------------------------


def test_run_bind_cell_completes_end_to_end(tmp_path: Path):
    cases = _cases()
    capture = asyncio.run(_run_capture(tmp_path, cases))
    llm = _fake_llm_for_bind_cell(capture, _RUNG0)

    out = CellStore.create(tmp_path / "bind-cell", manifest=_cell_manifest())
    asyncio.run(run_bind_cell(_RUNG0, cases, capture=capture, out=out, llm=llm, concurrency=4))

    assert out.is_complete()
    n_bind_records = len(capture.bind_records())
    assert len(out.call_records()) == n_bind_records * len(_RUNG0.run.seeds)
    scores = out.read_scores()
    assert scores["n_calls"] == n_bind_records * len(_RUNG0.run.seeds)
    # over_nomination_count is scored per leg, not per case: conj-01 is a
    # two-variable ask whose parts carry a definite population leg and a
    # candidates GDP leg, so only the legs that accept definite alone get the
    # dimension scored. shape_election_match may or may not resolve depending on
    # whether the canned shape 0 happens to match each leg's expected shape, but
    # the dimension must at least be attempted for every such leg.
    cases_by_id = {c.id: c for c in cases}
    n_definite_legs = sum(
        1
        for record in capture.bind_records()
        for _seed in _RUNG0.run.seeds
        if (expected := leg_expectation(cases_by_id[record.case_id], record.leg)) is not None
        and expected.sole_status == "definite"
    )
    assert scores["over_nomination_scored"] == n_definite_legs


def test_run_bind_cell_resume_produces_equal_scores(tmp_path: Path):
    cases = _cases()
    capture = asyncio.run(_run_capture(tmp_path, cases))
    llm = _fake_llm_for_bind_cell(capture, _RUNG0)

    full = CellStore.create(tmp_path / "bind-full", manifest=_cell_manifest())
    asyncio.run(run_bind_cell(_RUNG0, cases, capture=capture, out=full, llm=llm, concurrency=4))
    full_scores = full.read_scores()
    full_records = full.call_records()

    # Simulate a kill-and-resume: a fresh store pre-seeded with exactly one
    # call this same rung/capture/llm would have produced, closed without
    # finish(), then replayed to completion.
    partial = CellStore.create(tmp_path / "bind-partial", manifest=_cell_manifest())
    seed_record = dict(full_records[0])
    case_id = seed_record.pop("case_id")
    leg = seed_record.pop("leg")
    seed = seed_record.pop("seed")
    partial.append(case_id=case_id, leg=leg, seed=seed, call=seed_record)
    partial.close()

    resumed = CellStore.open(tmp_path / "bind-partial")
    asyncio.run(run_bind_cell(_RUNG0, cases, capture=capture, out=resumed, llm=llm, concurrency=4))

    assert resumed.is_complete()
    assert sorted(resumed.done_keys()) == sorted(full.done_keys())
    assert resumed.read_scores() == full_scores


def test_run_bind_cell_raises_when_case_missing_for_captured_record(tmp_path: Path):
    # One case, leg, and seed isolates the missing-case failure.
    cases = _cases()
    capture = asyncio.run(_run_capture(tmp_path, cases[1:2]))  # life-expectancy: one leg
    rung = dataclasses.replace(_RUNG0, run=dataclasses.replace(_RUNG0.run, seeds=(11,)))
    llm = _fake_llm_for_bind_cell(capture, rung)
    out = CellStore.create(tmp_path / "bind-cell-missing-case", manifest=_cell_manifest())

    with pytest.raises(ValueError, match="no case supplied"):
        asyncio.run(run_bind_cell(rung, [], capture=capture, out=out, llm=llm, concurrency=4))


# ---------------------------------------------------------------------------
# run_extract_cell
# ---------------------------------------------------------------------------


class _CountingGraph(FakeGraph):
    """Count graph calls made while resolving extracted phrases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detect_legs_calls = 0
        self.resolve_by_arc_calls = 0

    def detect_legs(self, query: str):
        self.detect_legs_calls += 1
        return super().detect_legs(query)

    def resolve_by_arc(
        self,
        names: list[str],
        *,
        arc: str,
        deadline: float | None = None,
    ):
        self.resolve_by_arc_calls += 1
        return super().resolve_by_arc(names, arc=arc)


def test_run_extract_cell_completes_end_to_end(tmp_path: Path):
    cases = _cases()
    capture = asyncio.run(_run_capture(tmp_path, cases))
    llm = _fake_llm_for_extract_cell(capture, _RUNG0)
    graph = _CountingGraph()

    out = CellStore.create(tmp_path / "extract-cell", manifest=_cell_manifest())
    asyncio.run(
        run_extract_cell(
            _RUNG0, cases, capture=capture, graph=graph, out=out, llm=llm, concurrency=4
        )
    )

    assert out.is_complete()
    n_extract_records = len(capture.extract_records())
    assert len(out.call_records()) == n_extract_records * len(_RUNG0.run.seeds)
    scores = out.read_scores()
    assert scores["n_calls"] == n_extract_records * len(_RUNG0.run.seeds)
    # The fixture replays the reference extraction verbatim.
    assert scores["date_structure_match_rate"] == 1.0
    assert scores["variable_split_match_rate"] == 1.0


def test_run_extract_cell_novel_phrase_writes_back_through_the_cache(tmp_path: Path):
    """A second cache over the same SQLite file serves the replay without graph calls."""
    cases = _cases()
    capture = asyncio.run(_run_capture(tmp_path, cases))
    llm = _fake_llm_for_extract_cell(capture, _RUNG0)
    db_path = tmp_path / "graph.sqlite"

    first_inner = _CountingGraph()
    first_cache = GraphCache(first_inner, path=db_path)
    out = CellStore.create(tmp_path / "extract-cell-novel", manifest=_cell_manifest())
    asyncio.run(
        run_extract_cell(
            _RUNG0, cases, capture=capture, graph=first_cache, out=out, llm=llm, concurrency=4
        )
    )
    first_cache.close()

    assert first_inner.detect_legs_calls > 0
    assert first_inner.resolve_by_arc_calls > 0

    second_inner = _CountingGraph()
    second_cache = GraphCache(second_inner, path=db_path)
    out2 = CellStore.create(tmp_path / "extract-cell-novel-replay", manifest=_cell_manifest())
    asyncio.run(
        run_extract_cell(
            _RUNG0, cases, capture=capture, graph=second_cache, out=out2, llm=llm, concurrency=4
        )
    )
    second_cache.close()

    assert second_inner.detect_legs_calls == 0
    assert second_inner.resolve_by_arc_calls == 0


def test_run_extract_cell_resume_produces_equal_scores(tmp_path: Path):
    cases = _cases()
    capture = asyncio.run(_run_capture(tmp_path, cases))
    llm = _fake_llm_for_extract_cell(capture, _RUNG0)

    full = CellStore.create(tmp_path / "extract-full", manifest=_cell_manifest())
    asyncio.run(
        run_extract_cell(
            _RUNG0, cases, capture=capture, graph=FakeGraph(), out=full, llm=llm, concurrency=4
        )
    )
    full_scores = full.read_scores()
    full_records = full.call_records()

    partial = CellStore.create(tmp_path / "extract-partial", manifest=_cell_manifest())
    seed_record = dict(full_records[0])
    case_id = seed_record.pop("case_id")
    leg = seed_record.pop("leg")
    seed = seed_record.pop("seed")
    partial.append(case_id=case_id, leg=leg, seed=seed, call=seed_record)
    partial.close()

    resumed = CellStore.open(tmp_path / "extract-partial")
    asyncio.run(
        run_extract_cell(
            _RUNG0, cases, capture=capture, graph=FakeGraph(), out=resumed, llm=llm, concurrency=4
        )
    )

    assert resumed.is_complete()
    assert sorted(resumed.done_keys()) == sorted(full.done_keys())
    assert resumed.read_scores() == full_scores


def test_run_extract_cell_raises_when_case_missing_for_captured_record(tmp_path: Path):
    # One case and seed isolates the missing-case failure.
    cases = _cases()
    capture = asyncio.run(_run_capture(tmp_path, cases[:1]))  # synth-tb-incidence-ethiopia
    rung = dataclasses.replace(_RUNG0, run=dataclasses.replace(_RUNG0.run, seeds=(11,)))
    llm = _fake_llm_for_extract_cell(capture, rung)
    out = CellStore.create(tmp_path / "extract-cell-missing-case", manifest=_cell_manifest())

    with pytest.raises(ValueError, match="no case supplied"):
        asyncio.run(
            run_extract_cell(
                rung, [], capture=capture, graph=FakeGraph(), out=out, llm=llm, concurrency=4
            )
        )
