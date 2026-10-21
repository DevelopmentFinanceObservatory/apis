"""Checks that measure-election drops join member-fact inapplicable slots without duplicates.

The real CRS fixture resolves an index-only member. A synthetic sibling adds `nettingBasis` to
`slot_keys`, while `financialObligationType` reaches the response only through election drops.
"""

from __future__ import annotations

import asyncio

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from series_coverage.schema import RoleNames
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "grant equivalent flows to Kenya"
_KEN_DCID = "country/KEN"

_POP_TYPE = "FinancialFlow"
_MEAS_PROP = "grantEquivalent"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"

_RECALLED_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_EXTRA_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)
_NETTING_ONLY_SV = "TESTFIXTURE/MeasureElectionInapplicableSlots/NettingOnlySibling"

_PROP_PURPOSE = "crsPurpose"
_EXTRA_PURPOSE = "CRSPurposeCode51010"

_PROP_FLOW_TYPE = "financialFlowType"
_FORCED_FLOW_TYPE_VALUE = "Disbursement"
_PROP_NETTING = "nettingBasis"
_NETTING_ONLY_VALUE = "TestGrossValue"
_PROP_OBLIGATION_TYPE = "financialObligationType"

# The destination-end role every packaged CRS row declares; "to Kenya" binds
# Kenya under this role.
_ROLE_RECIPIENT = "financingRecipient"
_ROLES = ("financingProvider", "financingRecipient")


def _facet(*, obs_count: int = 3, facet_id: str = "f1") -> Facet:
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
    )


def _sv_arcs(dcid: str, *, constraint_props: list[str], **constraint_values: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "measurementQualifier": {"nodes": [{"dcid": _MEAS_QUAL}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraint_props]},
        "observationProperties": {"nodes": [{"dcid": p} for p in _ROLES]},
        **{p: {"nodes": [{"dcid": constraint_values[p]}]} for p in constraint_props},
    }


def _sv_arcs_from_real_row(dcid: str) -> dict:
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    return _sv_arcs(
        dcid, constraint_props=list(row.constraint_properties), **row.constraint_properties
    )


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: {
        "label": "test recalled sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_RECALLED_SV),
    },
    # Grounding resolves this index-only member by name and reads its declared roles.
    _EXTRA_SV: {
        "label": "test index-only sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_EXTRA_SV),
    },
    # Adds `nettingBasis` to the shape's slot keys without becoming place-available.
    _NETTING_ONLY_SV: {
        "label": "test netting-only sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            _NETTING_ONLY_SV,
            constraint_props=[_PROP_NETTING],
            **{_PROP_NETTING: _NETTING_ONLY_VALUE},
        ),
    },
    _EXTRA_PURPOSE: {"label": "CRS purpose 51010"},
}

_DETECT = {
    _QUERY: {
        "svs": [_RECALLED_SV, _NETTING_ONLY_SV],
        "cosine_scores": [0.9, 0.8],
    }
}

# Only _EXTRA_SV carries observations, keyed under the role-bearing shape's
# own destination role -- the crsPurpose binding already excludes
# _RECALLED_SV and _NETTING_ONLY_SV before this role-keyed probe ever runs.
_OBS = {_EXTRA_SV: {((_ROLE_RECIPIENT, _KEN_DCID),): [_facet()]}}
# Only _EXTRA_SV confirms against coverage: the destination axis binds
# Kenya, the origin axis stays open, so the donor dcid here is a
# placeholder no assertion depends on.
_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])
_COVERAGE_ROWS = [(_EXTRA_SV, "f1", "country/USA", _KEN_DCID, [2018, 2019, 2020])]


class _BindPurposeToExtraLLM:
    """Binds `crsPurpose` to `_EXTRA_SV`, leaving other constraint slots open."""

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
                )
            ],
            also_plausible=[],
        ), None


def _run(monkeypatch):
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
            llm=_BindPurposeToExtraLLM(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def test_dropped_axis_absent_from_slot_keys_still_reaches_inapplicable_slots(monkeypatch):
    """`financialObligationType` reaches inapplicable slots only through election drops."""
    result = _run(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    assert _PROP_OBLIGATION_TYPE in spec.resolved.inapplicable_slots
    assert all(
        s.key.property is None or s.key.property.ref != _PROP_OBLIGATION_TYPE
        for s in spec.resolved.slots
    )


def test_a_property_both_sources_report_appears_once(monkeypatch):
    """Both member facts and the election drop `nettingBasis`; the union includes it once."""
    result = _run(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.inapplicable_slots.count(_PROP_NETTING) == 1


def test_forced_axis_is_never_reported_inapplicable(monkeypatch):
    """The election forces `financialFlowType=Disbursement`; forced axes stay out of inapplicable slots."""
    result = _run(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    assert _PROP_FLOW_TYPE not in spec.resolved.inapplicable_slots
    flow_type_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == _PROP_FLOW_TYPE
    )
    assert flow_type_slot.binding.kind == "value"
    assert flow_type_slot.binding.value.ref.ref == _FORCED_FLOW_TYPE_VALUE


def test_inapplicable_slots_order_is_stable_across_runs(monkeypatch):
    first = _run(monkeypatch).specs[0].resolved.inapplicable_slots
    second = _run(monkeypatch).specs[0].resolved.inapplicable_slots
    assert first == second


def test_the_served_measure_partition_reaches_the_definite_response(monkeypatch):
    """The resolved grant-equivalent answer publishes its observed measure partition."""
    result = _run(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.alternative_measures is not None
