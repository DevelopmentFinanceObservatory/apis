"""Checks whether elected constraint values reach `ResolvedFilter.provenance` as `measure_elected`.

The real CRS grant-equivalent fixture forces `financialFlowType` unless the query binds it first.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_measure_election_inapplicable_slots import (
    _COVERAGE_ROWS,
    _DETECT,
    _EXTRA_PURPOSE,
    _EXTRA_SV,
    _FORCED_FLOW_TYPE_VALUE,
    _KEN_DCID,
    _NODES,
    _OBS,
    _PROP_FLOW_TYPE,
    _PROP_PURPOSE,
    _QUERY,
    _ROLE_NAMES,
)
from tests.engine.test_measure_election_inapplicable_slots import (
    _run as _run_purpose_only,
)
from tests.fixtures import FakeGraph, make_resolver


def _filter_for(spec, prop: str):
    return next(
        f
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None and f.key.property.ref == prop
    )


def test_the_axis_the_election_forced_reads_measure_elected(monkeypatch):
    """The election binds `financialFlowType`; query-bound `crsPurpose` keeps `binder` provenance."""
    result = _run_purpose_only(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]

    flow_type_filter = _filter_for(spec, _PROP_FLOW_TYPE)
    assert flow_type_filter.provenance == "measure_elected"
    assert flow_type_filter.refs and flow_type_filter.refs[0].ref == _FORCED_FLOW_TYPE_VALUE

    purpose_filter = _filter_for(spec, _PROP_PURPOSE)
    assert purpose_filter.provenance == "binder"


class _BindPurposeAndFlowTypeLLM:
    """Also drafts the family's sole forceable axis, leaving the election nothing to bind."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(
            0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_PURPOSE,
                    kind="value",
                    value_dcids=[_EXTRA_PURPOSE],
                ),
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_FLOW_TYPE,
                    kind="value",
                    value_dcids=[_FORCED_FLOW_TYPE_VALUE],
                ),
            ],
            also_plausible=[],
        ), None


def _run_purpose_and_flow_type(monkeypatch):
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _COVERAGE_ROWS, role_names={_EXTRA_SV: _ROLE_NAMES}
    )
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs={})
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            entity_roles=door_roles_for_test_query(_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations=_OBS, availability={}),
            llm=_BindPurposeAndFlowTypeLLM(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def test_no_axis_forced_means_no_filter_reads_measure_elected(monkeypatch):
    """When the query binds `financialFlowType`, no filter gets `measure_elected` provenance."""
    result = _run_purpose_and_flow_type(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]

    assert spec.resolved.trace.slot_filters
    assert all(f.provenance != "measure_elected" for f in spec.resolved.trace.slot_filters)
    assert _filter_for(spec, _PROP_FLOW_TYPE).provenance == "binder"
