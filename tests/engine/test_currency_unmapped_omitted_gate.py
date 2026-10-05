"""The bound-unit gate adds `currency_unmapped_omitted` to a definite answer whose
conversion omits entities no currency names, on a fresh request and a resubmit alike, from the
coverage grounding recorded (`convertibility.convertible_answer`)."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from entity_resolution import Admission, EntityResolver
from materialise import GraphReader
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.region import phase_answer, promote_finalize
from search.engine.region.convertibility import UnitCoverage
from search.engine.region.facet_probe import FacetProbe
from search.engine.region.promote_state import _PromoteState
from search.engine.region.state import ResolveState
from search.engine.retrieval.read_plan import BulkPlan
from search.engine.retrieval.retrieve import Materialised
from search.engine.unit.bind import BoundUnitRequest
from search.models import CoverageBare, ResubmitIdentityPayload
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine.test_unit_coverage import (
    _DONOR,
    _KEN,
    _REGIONS,
    _SV,
    SECOND_FLOW,
    two_flow_bulk_plan,
    two_flow_elected,
    two_flow_graph,
)

_OMITTED = UnitCoverage(
    omitted=frozenset(("sv", "financingRecipient", e) for e in ("africaRegion", "southOfSahara"))
)

# The gate reads the recorded coverage, so none of these is read.
_GRAPH = cast(EngineGraphClient, object())
_MATERIALISE_GRAPH = cast(GraphReader, object())
_MATERIALISE_SDMX = cast(MaterialiseSdmxClient, object())


def _mat_result() -> Materialised:
    return Materialised(sv_dcids=["sv"], has_data=True, coverage=CoverageBare(has_data=True))


def _fresh_state(*, materialise_graph: GraphReader = _MATERIALISE_GRAPH) -> ResolveState:
    return ResolveState(
        variable="aid",
        entities=[],
        date_request=None,
        detect_query="aid",
        role_query="aid",
        pac=True,
        graph=_GRAPH,
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        source_text=None,
        materialise_graph=materialise_graph,
        materialise_sdmx=_MATERIALISE_SDMX,
        pipeline_steps=[],
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
    )


def _fresh() -> tuple[object, list]:
    state = _fresh_state()
    state.unit_coverage = _OMITTED
    setup = phase_answer._DefiniteSetup(
        available_at_place=None,
        confidence=None,
        obs_entities={},
        probe_role_entities={},
        flat_probe_entities=[],
        bulk_scope=None,
        role_axis_facts=[],
        obs_fetch_facet_filter=None,
        # The gate reads only `obs_entities` of the setup.
        entity_task=cast(asyncio.Task, None),
        slot_options_raw=[],
        facet_probe=cast(FacetProbe, None),
        t0=0,
        facet_bindings=None,
    )
    result = phase_answer._bound_unit_refusal(state, _mat_result(), setup)
    return result, state.warnings


def _resubmit() -> tuple[object, list]:
    state = _PromoteState(
        inp=cast(ResubmitIdentityPayload, None),
        graph=_GRAPH,
        pipeline_steps=[],
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        materialise_graph=_MATERIALISE_GRAPH,
        materialise_sdmx=_MATERIALISE_SDMX,
        coverage=empty_coverage_snapshot(),
    )
    state.unit_coverage = _OMITTED
    return promote_finalize._bound_unit_refusal(state, _mat_result()), state.warnings


@pytest.mark.parametrize(
    ("module", "gate"),
    [(phase_answer, _fresh), (promote_finalize, _resubmit)],
    ids=["fresh", "resubmit"],
)
def test_a_passing_gate_that_omits_entities_warns_once(monkeypatch, module, gate):
    monkeypatch.setattr(module, "bound_unit_gate_reason", lambda *_args: None)

    result, warnings = gate()

    assert result is None
    assert [(w.code, w.severity) for w in warnings] == [("currency_unmapped_omitted", "warn")]
    assert warnings[0].message.endswith("2 entities: africaRegion, southOfSahara")


def test_a_bulk_latest_gate_reads_the_plan_its_read_serves(monkeypatch, search_settings):
    """Two flows read in bulk under `latest`: the first gives Kenya 2020 and a region 2021,
    the second Kenya 2020. The region's rows are omitted before the latest year is taken, so
    the gate reads 2020, where both flows serve Kenya, and the conversion runs, as Datasets
    serves Kenya 2020 for each flow."""
    search_settings(obs_preview_max_pairs=1)
    plan, mat = two_flow_bulk_plan(
        {_SV: {_KEN: (2020,), _REGIONS[0]: (2021,)}, SECOND_FLOW: {_KEN: (2020,)}}
    )
    state = _fresh_state(materialise_graph=two_flow_graph())
    state.bound_unit_outcome = cast(BoundUnitRequest, object())
    state.unit_elected_selections = two_flow_elected()
    monkeypatch.setattr(phase_answer, "_gate_carve_out", lambda *_args: False)
    monkeypatch.setattr(
        phase_answer,
        "_answer_observation_entities",
        lambda *_args: ({"financingRecipient": [_KEN], "financingProvider": [_DONOR]}, None),
    )

    assert phase_answer._runs_unit_conversion(state, mat, plan) is True
    assert isinstance(state.unit_gate_plan, BulkPlan)
    assert state.unit_gate_plan.member_years(_SV) == (2020, 2020)
    assert state.unit_gate_plan.member_years(SECOND_FLOW) == (2020, 2020)
