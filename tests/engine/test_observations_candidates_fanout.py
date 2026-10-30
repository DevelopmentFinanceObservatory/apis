"""Tests that candidate fan-out applies bound-unit checks and builds facet
options for each surviving candidate.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest

import search.engine.region.candidate_collapse as candidate_collapse_module
import search.engine.region.candidate_options as candidate_options_module
from materialise import (
    GraphReader,
)
from materialise.rate_tables import RateTableCache
from search.engine.concurrency import scoped_concurrency
from search.engine.errors import GraphInfraError
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.state import EngineOptions
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    materialise_double_from_obs,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------------------
# region/candidates.py: the candidates fan-out (`if alt_indices:`) threading a
# bound unit_text request the same way the single-shape path already does --
# spec_id correctness (facet_bindings on every candidate), the per-candidate
# availability refusal, a graph outage during the check failing loud rather
# than serving an unverified "definite", and (separately) facet_options for
# every surviving candidate under include_option_coverage.
# ---------------------------------------------------------------------------


def _candidate_unit_std_arcs(label: str, *, pop_type: str, meas_prop: str) -> dict:
    """A plain STANDARD aggregate SV with no constraint slots -- bind_universal's
    own bindings are legitimately empty for a shape like this (see
    _UniversalBindOutput.bindings' own docstring)."""
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


class _CandidateUnitBindLLM:
    """Elects `chosen_shape_index` with no constraint bindings (every fixture
    SV below is a plain STANDARD aggregate) and flags `also_plausible` -- the
    model's own ambiguity signal that drives region/candidates.py's
    `if alt_indices:` candidates fan-out."""

    def __init__(self, also_plausible: list[int], *, chosen_shape_index: int = 0):
        self._also_plausible = also_plausible
        self._chosen_shape_index = chosen_shape_index

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(self._chosen_shape_index, also_plausible=self._also_plausible), None


_UNIT_CAND_SV_A = "Test_CandidateUnit_SV_A"
_UNIT_CAND_SV_B = "Test_CandidateUnit_SV_B"
_UNIT_CAND_QUERY = "candidate unit test variable in Kenya"
_UNIT_CAND_NODES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _UNIT_CAND_SV_A: {
        "label": "Candidate unit SV A",
        "type": "StatisticalVariable",
        "arcs": _candidate_unit_std_arcs(
            "Candidate unit test variable A", pop_type="EconomicActivity", meas_prop="amountA"
        ),
    },
    _UNIT_CAND_SV_B: {
        "label": "Candidate unit SV B",
        "type": "StatisticalVariable",
        "arcs": _candidate_unit_std_arcs(
            "Candidate unit test variable B", pop_type="EconomicActivity", meas_prop="amountB"
        ),
    },
}
_UNIT_CAND_DETECT = {
    _UNIT_CAND_QUERY: {
        "svs": [_UNIT_CAND_SV_A, _UNIT_CAND_SV_B],
        "cosine_scores": [0.9, 0.88],
    },
}
# Both candidates have one real, in-window observation under unit=USD -- the
# same (currency="USD", prices="current") triple "US dollars" binds to.
_BOTH_MATCH_USD_OBS = {
    f"{_UNIT_CAND_SV_A}|country/KEN": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/SrcA",
            "unit": "CurrencyUSD",
            "dates": ["2020"],
            "values": [10.0],
        }
    ],
    f"{_UNIT_CAND_SV_B}|country/KEN": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/SrcB",
            "unit": "CurrencyUSD",
            "dates": ["2020"],
            "values": [20.0],
        }
    ],
}
# SV_A (the elected shape) has real data, but in EUR, not the bound USD --
# SV_B (the also_plausible alternate) carries no observations at all, so
# _materialise_alt_shapes drops it, leaving SV_A as the lone survivor.
_LONE_SURVIVOR_WRONG_UNIT_OBS = {
    f"{_UNIT_CAND_SV_A}|country/KEN": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/SrcA",
            "unit": "CurrencyEUR",
            "dates": ["2020"],
            "values": [10.0],
        }
    ],
}
# SV_A's facet carries USD, so the bound unit matches. SV_B holds data in Kenya too, so the
# status rule offers it beside SV_A, but only in EUR, which declares no conversion recipe:
# the bound-unit filter drops it, and SV_A, the lone survivor, collapses to "definite".
_LONE_SURVIVOR_MATCHED_UNIT_OBS = {
    f"{_UNIT_CAND_SV_A}|country/KEN": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/SrcA",
            "unit": "CurrencyUSD",
            "dates": ["2020"],
            "values": [10.0],
        }
    ],
    f"{_UNIT_CAND_SV_B}|country/KEN": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/SrcB",
            "unit": "CurrencyEUR",
            "dates": ["2020"],
            "values": [20.0],
        }
    ],
}


def _run_candidate_unit(
    *,
    obs: dict,
    unit_text: str | None,
    also_plausible: list[int] | None = None,
    include_option_coverage: bool = False,
    stream_slot_options: bool = False,
    include_observations: bool = False,
    graph: FakeGraph | None = None,
    materialise_graph: GraphReader | None = None,
):
    if graph is None:
        graph = FakeGraph(
            nodes=_UNIT_CAND_NODES,
            detect=_UNIT_CAND_DETECT,
            resolve={"Kenya": "country/KEN"},
            obs=obs,
        )
    _materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    if materialise_graph is None:
        materialise_graph = _materialise_graph
        # region/candidates.py's per-candidate election (materialise.
        # decide_member_unit, via _candidate_unit_elections) resolves a
        # native currency match through
        # materialise.recipes.require_currency_unit even when no conversion
        # is needed. Declared here since materialise_double_from_obs'
        # FakeGraphReader carries no arcs by default, on the auto-built
        # double only; a caller-supplied materialise_graph (e.g.
        # _RaisingGraphReader) keeps its own arcs. CurrencyEUR carries no
        # conversionRecipe on its provenance, so a EUR-native facet under a
        # bound USD request refuses genuinely (RecipeNotDeclaredError),
        # matching every fixture below that relies on that mismatch.
        materialise_graph._arcs = {
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
            "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
        }
    return asyncio.run(
        resolve_variable(
            _UNIT_CAND_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_UNIT_CAND_QUERY,
            role_query=_UNIT_CAND_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CandidateUnitBindLLM(also_plausible if also_plausible is not None else [1]),
            base_timing={},
            unit_text=unit_text,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_option_coverage=include_option_coverage,
                stream_slot_options=stream_slot_options,
                include_observations=include_observations,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        )
    )


def test_candidates_bound_unit_carries_facets_unit_in_every_spec_identity():
    """A bound unit changes every candidate's identity, so its `spec_id`
    differs from the same query without the binding.
    """
    result = _run_candidate_unit(obs=_BOTH_MATCH_USD_OBS, unit_text="US dollars")
    assert result.status == "candidates"
    assert len(result.specs) == 2

    baseline = _run_candidate_unit(obs=_BOTH_MATCH_USD_OBS, unit_text=None)
    assert baseline.status == "candidates"
    bound_ids = {s.spec_id for s in result.specs}
    baseline_ids = {s.spec_id for s in baseline.specs}
    assert bound_ids.isdisjoint(baseline_ids), (
        "a bound unit request must mint different spec_ids than the same "
        "query without it -- facet_bindings feeds spec_id, so the two "
        "identities must never collide"
    )


def test_candidates_bound_unit_stamps_unit_lexicon_version_on_every_spec():
    """A bound-unit request stamps every surviving candidate, including when
    two candidates remain.
    """
    result = _run_candidate_unit(obs=_BOTH_MATCH_USD_OBS, unit_text="US dollars")
    assert result.status == "candidates"
    assert len(result.specs) == 2
    for spec in result.specs:
        assert spec.resolved.unit_lexicon_version is not None, (
            f"spec {spec.spec_id!r} missing unit_lexicon_version"
        )


def test_multi_candidate_response_skips_observations_when_requested():
    result = _run_candidate_unit(
        obs=_BOTH_MATCH_USD_OBS,
        unit_text="US dollars",
        include_observations=True,
    )
    assert result.status == "candidates"
    assert all(spec.resolved.observations is None for spec in result.specs)


def test_candidates_lone_survivor_matched_unit_collapses_to_definite_with_facets():
    """A matching USD facet lets the lone surviving candidate collapse to a
    definite result with bound facet metadata.
    """
    result = _run_candidate_unit(obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS, unit_text="US dollars")
    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.facets["unit"].ref == "CurrencyUSD"
    assert spec.resolved.facets["unit"].basis == "bound"


def test_candidates_lone_survivor_matched_unit_buffered_observations_keep_bound_basis():
    """Buffered observation attachment preserves the bound unit basis and
    populates observations on the collapsed definite result.
    """
    result = _run_candidate_unit(
        obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS,
        unit_text="US dollars",
        include_observations=True,
    )
    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.facets["unit"].ref == "CurrencyUSD"
    assert spec.resolved.facets["unit"].basis == "bound"
    assert spec.resolved.facets["unit"].request_basis is not None
    assert spec.resolved.observations is not None


def test_candidates_lone_survivor_wrong_unit_refuses_recipe_not_declared():
    """An EUR facet needs conversion to serve the bound USD request, and the StatVar declares
    no nominal amount to convert. The candidate election raises `RecipeNotDeclaredError`, and
    the candidates door answers `recipe_not_declared` with its cause, as the single-shape door
    does.
    """
    result = _run_candidate_unit(obs=_LONE_SURVIVOR_WRONG_UNIT_OBS, unit_text="US dollars")
    assert result.status == "unsupported", result
    assert result.unsupported is not None
    assert result.unsupported.code == "recipe_not_declared"
    assert result.unsupported.cause is not None
    assert result.unsupported.cause.code == "nominal_undeclared"


def test_candidates_bound_unit_graph_failure_raises_infra_error_not_definite():
    """A graph outage in the per-candidate unit probe propagates as
    `GraphInfraError`, which the app maps to a retryable 503. The probe reads
    through `materialise_graph`: its facet reads and, with the plan's candidates in hand,
    its facet-policy reads.
    """

    class _RaisingGraphReader(FakeGraphReader):
        def variable_facets_batch(
            self, *, stat_var, entities, needs_dates: bool = False, deadline: float | None = None
        ):
            raise GraphInfraError("graph down")

        def default_facet_policy(self, *, stat_var, axis):
            raise GraphInfraError("graph down")

    with pytest.raises(GraphInfraError):
        _run_candidate_unit(
            obs=_BOTH_MATCH_USD_OBS,
            unit_text="US dollars",
            materialise_graph=_RaisingGraphReader(),
        )


def test_candidates_include_option_coverage_populates_facet_options_per_candidate():
    """Each surviving candidate gets facet options from its own StatVars and
    observation entities.
    """
    result = _run_candidate_unit(
        obs=_BOTH_MATCH_USD_OBS,
        unit_text=None,
        include_option_coverage=True,
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2
    for spec in result.specs:
        assert spec.resolved.facet_options, (
            f"expected facet_options for candidate spec {spec.spec_id!r}, got []"
        )
        axes = {fo.axis for fo in spec.resolved.facet_options}
        assert "unit" in axes


def test_candidates_sse_trailing_facet_options_match_buffered_embed():
    """The SSE task emits the same facet-option axes as the buffered path for
    each candidate.
    """
    buffered = _run_candidate_unit(
        obs=_BOTH_MATCH_USD_OBS,
        unit_text=None,
        include_option_coverage=True,
    )
    assert buffered.status == "candidates"
    buffered_axes_by_spec = {
        s.spec_id: {fo.axis for fo in s.resolved.facet_options} for s in buffered.specs
    }
    assert all(buffered_axes_by_spec.values())

    async def _run_sse():
        graph = FakeGraph(
            nodes=_UNIT_CAND_NODES,
            detect=_UNIT_CAND_DETECT,
            resolve={"Kenya": "country/KEN"},
            obs=_BOTH_MATCH_USD_OBS,
        )
        async with scoped_concurrency():
            result = await resolve_variable(
                _UNIT_CAND_QUERY,
                entities=["Kenya"],
                date_request=None,
                detect_query=_UNIT_CAND_QUERY,
                role_query=_UNIT_CAND_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_CandidateUnitBindLLM([1]),
                base_timing={},
                unit_text=None,
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(stream_slot_options=True),
            )
            assert result.slot_options_task is not None
            events, warns = await result.slot_options_task
            return result, events, warns

    sse, events, warns = asyncio.run(_run_sse())
    assert sse.status == "candidates"
    # SSE never embeds facet_options directly -- the response stays empty,
    # the trailing task carries it instead.
    assert all(not s.resolved.facet_options for s in sse.specs)
    assert not warns

    sse_axes_by_spec: dict[str, set[str]] = {}
    for ev in events:
        if ev.kind == "facet_options":
            sse_axes_by_spec.setdefault(ev.spec_id, set()).add(ev.axis)

    assert set(sse_axes_by_spec) == set(buffered_axes_by_spec)
    for spec_id, buffered_axes in buffered_axes_by_spec.items():
        assert sse_axes_by_spec[spec_id] == buffered_axes


def test_single_spec_sse_trailing_facet_options_unchanged():
    """The single-spec SSE task includes unit facet options."""

    async def _run_sse():
        graph = FakeGraph(
            nodes=_UNIT_CAND_NODES,
            detect=_UNIT_CAND_DETECT,
            resolve={"Kenya": "country/KEN"},
            obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS,
        )
        materialise_graph, materialise_sdmx = materialise_double_from_obs(
            _LONE_SURVIVOR_MATCHED_UNIT_OBS
        )
        materialise_graph._arcs = {
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
        async with scoped_concurrency():
            result = await resolve_variable(
                _UNIT_CAND_QUERY,
                entities=["Kenya"],
                date_request=None,
                detect_query=_UNIT_CAND_QUERY,
                role_query=_UNIT_CAND_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_CandidateUnitBindLLM([1]),
                base_timing={},
                unit_text="US dollars",
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(
                    stream_slot_options=True,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=RateTableCache(),
                ),
            )
            assert result.slot_options_task is not None
            events, warns = await result.slot_options_task
            return result, events, warns

    result, events, warns = asyncio.run(_run_sse())
    assert result.status == "definite"
    assert not warns
    facet_axes = {ev.axis for ev in events if ev.kind == "facet_options"}
    assert "unit" in facet_axes


def test_candidates_lone_survivor_collapse_launches_place_options_task():
    """A lone-candidate collapse launches the place-options task when
    streaming and a materialisation graph are configured.
    """

    async def _run():
        graph = FakeGraph(
            nodes=_UNIT_CAND_NODES,
            detect=_UNIT_CAND_DETECT,
            resolve={"Kenya": "country/KEN"},
            obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS,
        )
        materialise_graph, materialise_sdmx = materialise_double_from_obs(
            _LONE_SURVIVOR_MATCHED_UNIT_OBS
        )
        materialise_graph._arcs = {
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
        place_options_pool = PlaceOptionsPool.build(max_workers=2)
        try:
            async with scoped_concurrency():
                result = await resolve_variable(
                    _UNIT_CAND_QUERY,
                    entities=["Kenya"],
                    date_request=None,
                    detect_query=_UNIT_CAND_QUERY,
                    role_query=_UNIT_CAND_QUERY,
                    pac=True,
                    graph=graph,
                    resolver=make_resolver(graph),
                    llm=_CandidateUnitBindLLM([1]),
                    base_timing={},
                    unit_text="US dollars",
                    sdmx=FakeSdmxClient(observations={}, availability={}),
                    coverage=empty_coverage_snapshot(),
                    options=EngineOptions(
                        stream_slot_options=True,
                        materialise_graph=materialise_graph,
                        materialise_sdmx=materialise_sdmx,
                        rate_cache=RateTableCache(),
                        place_options_pool=place_options_pool,
                    ),
                )
                assert result.status == "definite"
                assert result.place_options_task is not None
                assert result.place_options_relaunch is not None
                # scoped_concurrency cancels and awaits owned tasks at exit.
        finally:
            place_options_pool.shutdown()

    asyncio.run(_run())


def test_candidates_lone_survivor_cancellation_does_not_leak_observations_task(monkeypatch):
    """The facet probe blocks on an Event so cancellation lands inside `gather`.
    This await precedes observation-task creation; the wrapper records any task
    returned by `_resolve_spec_observations`. If that call ran first, the task would
    exist and leak uncancelled. Captured tasks must be cancelled, since fast fakes
    can finish one before the assertion.
    """
    entered = threading.Event()
    release = threading.Event()
    captured_tasks: list[asyncio.Task | None] = []

    real_resolve_spec_observations = candidate_collapse_module._resolve_spec_observations

    async def _capturing_resolve_spec_observations(*args, **kwargs):
        result = await real_resolve_spec_observations(*args, **kwargs)
        captured_tasks.append(result[1])
        return result

    def _blocking_maybe_build_facet_options(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return [], None

    monkeypatch.setattr(
        candidate_collapse_module,
        "_resolve_spec_observations",
        _capturing_resolve_spec_observations,
    )
    monkeypatch.setattr(
        candidate_options_module, "_maybe_build_facet_options", _blocking_maybe_build_facet_options
    )

    async def _run() -> None:
        graph = FakeGraph(
            nodes=_UNIT_CAND_NODES,
            detect=_UNIT_CAND_DETECT,
            resolve={"Kenya": "country/KEN"},
            obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS,
        )
        materialise_graph, materialise_sdmx = materialise_double_from_obs(
            _LONE_SURVIVOR_MATCHED_UNIT_OBS
        )
        materialise_graph._arcs = {
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
        resolve_task = asyncio.create_task(
            resolve_variable(
                _UNIT_CAND_QUERY,
                entities=["Kenya"],
                date_request=None,
                detect_query=_UNIT_CAND_QUERY,
                role_query=_UNIT_CAND_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_CandidateUnitBindLLM([1]),
                base_timing={},
                unit_text="US dollars",
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(
                    stream_observations=True,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=RateTableCache(),
                ),
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        while not entered.is_set():
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(
                    "the single-survivor branch never entered "
                    "_maybe_build_facet_options -- did the bound USD facet stop "
                    "matching the bound request?"
                )
            await asyncio.sleep(0)
        resolve_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resolve_task
        release.set()
        # Drain captured tasks after releasing the worker so the assertion
        # catches running and completed tasks alike.
        for t in captured_tasks:
            if t is not None:
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(asyncio.shield(t), timeout=2)

    asyncio.run(_run())

    assert all(t is not None and t.cancelled() for t in captured_tasks), (
        "an observations task survived the cancellation instead of being cancelled with it: "
        f"{captured_tasks!r}"
    )


_PURPOSE = "Test_CandidateUnit_HealthPurpose"
_PURPOSE_LABEL = "Candidate unit health purpose"


def _constrained_survivor_graph() -> FakeGraph:
    """SV_A as a constrained member: one constraint property whose value has a label."""
    arcs = {
        **_candidate_unit_std_arcs(
            "Candidate unit test variable A", pop_type="EconomicActivity", meas_prop="amountA"
        ),
        "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}]},
        "crsPurpose": {"nodes": [{"dcid": _PURPOSE}]},
    }
    return FakeGraph(
        nodes={
            **_UNIT_CAND_NODES,
            _UNIT_CAND_SV_A: {**_UNIT_CAND_NODES[_UNIT_CAND_SV_A], "arcs": arcs},
            _PURPOSE: {"label": _PURPOSE_LABEL},
        },
        detect=_UNIT_CAND_DETECT,
        resolve={"Kenya": "country/KEN"},
        obs=_LONE_SURVIVOR_MATCHED_UNIT_OBS,
    )


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_a_collapsed_survivor_labels_its_preview_from_its_member_facts(stream: bool):
    """The lone survivor's preview row labels the member by its constraint value's label, read
    from the survivor's own `member_facts` as the single-candidate path does."""

    async def _run():
        graph = _constrained_survivor_graph()
        materialise_graph, materialise_sdmx = materialise_double_from_obs(
            _LONE_SURVIVOR_MATCHED_UNIT_OBS
        )
        materialise_graph._arcs = {"CurrencyUSD": {"typeOf": ("CurrencyUnit",)}}
        async with scoped_concurrency():
            result = await resolve_variable(
                _UNIT_CAND_QUERY,
                entities=["Kenya"],
                date_request=None,
                detect_query=_UNIT_CAND_QUERY,
                role_query=_UNIT_CAND_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_CandidateUnitBindLLM([1]),
                base_timing={},
                unit_text="US dollars",
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(
                    include_observations=not stream,
                    stream_observations=stream,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=RateTableCache(),
                ),
            )
            assert result.status == "definite", result.no_data_reason
            if not stream:
                return result.specs[0].resolved.observations
            assert result.observations_task is not None
            (event,), _warns = await result.observations_task
            return event.observations

    observations = asyncio.run(_run())
    assert observations is not None
    (series,) = observations.series
    assert series.stat_var.label == _PURPOSE_LABEL
