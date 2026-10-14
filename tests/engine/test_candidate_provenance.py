"""Text-recognised provenance reaches the elected candidate's forced filters.

The alternate shares its crsTypeOfFinance value but reports `binder` from its own bindings.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_candidates_member_count_ordering import (
    _MEAS_PROP_1,
    _MEAS_QUAL_1,
    _POP_TYPE_1,
    _RECALLED_SV1,
    _SHAPE1_BOUND_PROPS,
    _STAT_TYPE_1,
    _real_row,
    _sv_node,
)
from tests.engine.test_qualifier_suppression_and_provenance import (
    _E2E_KEN_DCID,
    _E2E_SV_DCID,
    _e2e_sv_arcs,
)
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "ODA standard grants"
# Registers no code-row synonym in the real index, so match_qualifiers admits
# nothing for this text -- the same two shapes recalled, with no forced binding.
_NEUTRAL_QUERY = "zzqxvville flerbnaxx wobbleglorp"

_recalled1_row = _real_row(_RECALLED_SV1)

_NODES = {
    _E2E_KEN_DCID: {"label": "Kenya", "type": "Country"},
    _E2E_SV_DCID: {
        "label": "test grant equivalent flow",
        "type": "StatisticalVariable",
        "arcs": _e2e_sv_arcs(),
    },
    "CRSTypeOfFlow10": {"label": "ODA"},
    "CRSTypeOfFinance110": {"label": "standard grant"},
    "CRSCooperationModalityA02": {"label": "test cooperation modality"},
    _RECALLED_SV1: _sv_node(
        _RECALLED_SV1,
        pop_type=_POP_TYPE_1,
        meas_prop=_MEAS_PROP_1,
        stat_type=_STAT_TYPE_1,
        meas_qual=_MEAS_QUAL_1,
        constraint_props=list(_SHAPE1_BOUND_PROPS),
    ),
    # standard_bindings_from_arcs' own "value" bindings need each bound value
    # dcid to resolve too (build_slot falls back to unbound otherwise).
    **{
        v: {"label": v}
        for p, v in _recalled1_row.constraint_properties.items()
        if p in _SHAPE1_BOUND_PROPS
    },
}

_OBS = {
    f"{_E2E_SV_DCID}|{_E2E_KEN_DCID}": [
        {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
    ],
    f"{_RECALLED_SV1}|{_E2E_KEN_DCID}": [
        {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
    ],
}


class _ElectShape0AlsoPlausibleShape1LLM:
    """Elect the CRS shape, leave its forced axes to admission, and mark the financial-obligation
    shape as plausible.
    """

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid="crsCooperationModality",
                    kind="value",
                    value_dcids=["CRSCooperationModalityA02"],
                )
            ],
            also_plausible=[1],
            ask=None,
        ), None


class _ExplicitBindNoAdmissionLLM:
    """Draft the same values directly, without qualifier admission."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="what",
                    property_dcid="crsTypeOfFlow",
                    kind="value",
                    value_dcids=["CRSTypeOfFlow10"],
                ),
                SlotBindingDraft(
                    axis="what",
                    property_dcid="crsTypeOfFinance",
                    kind="value",
                    value_dcids=["CRSTypeOfFinance110"],
                ),
                SlotBindingDraft(
                    axis="how",
                    property_dcid="crsCooperationModality",
                    kind="value",
                    value_dcids=["CRSCooperationModalityA02"],
                ),
            ],
            also_plausible=[1],
            ask=None,
        ), None


def _graph(query: str) -> FakeGraph:
    detect = {query: {"svs": [_E2E_SV_DCID, _RECALLED_SV1], "cosine_scores": [0.9, 0.85]}}
    return FakeGraph(nodes=_NODES, detect=detect, resolve={"Kenya": _E2E_KEN_DCID}, obs=_OBS)


def _run(query: str, llm):
    graph = _graph(query)
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=llm,
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def _spec_for(specs, sv_dcid: str):
    return next(
        spec for spec in specs if any(sv.ref.ref == sv_dcid for sv in spec.resolved.members)
    )


def _filter_for(spec, prop: str):
    return next(
        f
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None and f.key.property.ref == prop
    )


def test_elected_candidate_carries_text_recognized_alt_carries_binder():
    """Only the elected candidate's forced filters report `text_recognized`. The alt's matching
    crsTypeOfFinance value reports `binder`.
    """
    result = _run(_QUERY, _ElectShape0AlsoPlausibleShape1LLM())
    assert result.status == "candidates"
    assert result.admitted_qualifier_phrases == frozenset({"oda", "standard grants"})

    elected = _spec_for(result.specs, _E2E_SV_DCID)
    alt = _spec_for(result.specs, _RECALLED_SV1)

    assert _filter_for(elected, "crsTypeOfFlow").provenance == "text_recognized"
    assert _filter_for(elected, "crsTypeOfFinance").provenance == "text_recognized"
    assert _filter_for(elected, "crsCooperationModality").provenance == "binder"

    alt_finance = _filter_for(alt, "crsTypeOfFinance")
    assert alt_finance.provenance == "binder"
    assert alt_finance.refs and alt_finance.refs[0].ref == "CRSTypeOfFinance110"


def test_elected_candidate_carries_measure_elected_alt_does_not():
    """The elected candidate carries measure-election provenance. The alternate uses its own arcs.

    The fixture sets `measure_elected_properties` directly to isolate propagation.
    """
    import search.engine.region.pipeline as pipeline_module

    _orig_settle = pipeline_module._phase_settle
    _orig_phases = pipeline_module._PHASES

    async def _settle_and_force_election(state):
        outcome = await _orig_settle(state)
        state.measure_elected_properties = frozenset({"financialFlowType"})
        return outcome

    # `resolve_variable` iterates `_PHASES`, so replace the tuple entry.
    pipeline_module._PHASES = tuple(
        _settle_and_force_election if phase is _orig_settle else phase for phase in _orig_phases
    )
    try:
        result = _run(_QUERY, _ElectShape0AlsoPlausibleShape1LLM())
    finally:
        pipeline_module._PHASES = _orig_phases

    assert result.status == "candidates"

    elected = _spec_for(result.specs, _E2E_SV_DCID)
    alt = _spec_for(result.specs, _RECALLED_SV1)

    assert _filter_for(elected, "financialFlowType").provenance == "measure_elected"
    assert all(f.provenance != "measure_elected" for f in alt.resolved.trace.slot_filters)


def test_spec_id_is_byte_identical_with_and_without_admission_on_this_path():
    """The elected candidate has the same spec identity with or without admission provenance."""
    forced = _spec_for(_run(_QUERY, _ElectShape0AlsoPlausibleShape1LLM()).specs, _E2E_SV_DCID)
    drafted = _spec_for(_run(_NEUTRAL_QUERY, _ExplicitBindNoAdmissionLLM()).specs, _E2E_SV_DCID)
    assert forced.spec_id == drafted.spec_id
    assert forced.definition.identity == drafted.definition.identity
    assert _filter_for(forced, "crsTypeOfFlow").provenance == "text_recognized"
    assert _filter_for(drafted, "crsTypeOfFlow").provenance == "binder"
