"""A data-less also-plausible alternate leaves one survivor.
An unbound query qualifier still produces `qualifier_not_bound` (#971).
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.departures import warning_key
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from search.heading.departure_fact import DepartureFact
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

_QUERY = "non-concessional grants to Kenya"

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

# Shape 1 (the also_plausible alt) carries no observations entry here: it
# never survives _materialise_alt_shapes, so candidate_triples collapses to
# the elected shape alone -- the single-survivor path this test targets.
_OBS = {
    f"{_E2E_SV_DCID}|{_E2E_KEN_DCID}": [
        {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
    ],
}


class _ElectShape0AlsoPlausibleShape1LLM:
    """Elects shape 0 (the CRS grant-equivalent flow), drafts no binding of
    its own, and names shape 1 (the financial-obligation shape) as
    also_plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None),
            None,
        )


def _graph(query: str) -> FakeGraph:
    detect = {query: {"svs": [_E2E_SV_DCID, _RECALLED_SV1], "cosine_scores": [0.9, 0.85]}}
    return FakeGraph(nodes=_NODES, detect=detect, resolve={"Kenya": _E2E_KEN_DCID}, obs=_OBS)


def test_lone_also_plausible_survivor_refuses_qualifier_not_bound():
    graph = _graph(_QUERY)
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_ElectShape0AlsoPlausibleShape1LLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "qualifier_not_bound"


# --- 2+ genuinely distinct survivors: candidates carries the warning --------

# Unlike _OBS above, shape 1 (the also_plausible alt) carries its own
# observations entry here, so it survives _materialise_alt_shapes alongside
# the elected shape 0: candidate_triples keeps two genuinely distinct
# entries, and resolve_candidates' own "2+ survivors" branch (not the lone-
# survivor collapse the test above targets) answers "candidates" instead of
# refusing.
_OBS_TWO_SURVIVORS = {
    **_OBS,
    f"{_RECALLED_SV1}|{_E2E_KEN_DCID}": [
        {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
    ],
}


def _graph_two_survivors(query: str) -> FakeGraph:
    detect = {query: {"svs": [_E2E_SV_DCID, _RECALLED_SV1], "cosine_scores": [0.9, 0.85]}}
    return FakeGraph(
        nodes=_NODES, detect=detect, resolve={"Kenya": _E2E_KEN_DCID}, obs=_OBS_TWO_SURVIVORS
    )


def test_two_survivors_candidates_carries_query_qualifier_dropped_warning():
    graph = _graph_two_survivors(_QUERY)
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_ElectShape0AlsoPlausibleShape1LLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "candidates"
    assert len(result.specs) >= 2
    (dropped,) = (
        w
        for w in result.warnings
        if w.code == "query_qualifier_dropped"
        and w.message == 'The requested qualifier "non-concessional" was not applied.'
    )
    # The phrase names no code, so no one slot edit clears it.
    assert result.departure_facts[warning_key(dropped)] == DepartureFact(
        code="measure_not_applied", binding_ref=None, asked="non-concessional"
    )
