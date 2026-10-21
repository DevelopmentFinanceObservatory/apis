"""Grounding uses the resolved member facts when the winner comes from the index.

The fixture recalls one real CRS StatVar, then resolves an index-only sibling for `to Kenya`. It
checks the winner's bound `crsPurpose` value and keeps its asserted, unbound properties applicable.
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

# "Flows" leaves financialFlowType unbound; "disbursements" would force its value.
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

_PROP_PURPOSE = "crsPurpose"
_RECALLED_PURPOSE = "CRSPurposeCode11110"
_EXTRA_PURPOSE = "CRSPurposeCode51010"

# The winner asserts all four axes. Three remain unbound; the measure election forces
# financialFlowType=Disbursement.
_PROP_MODALITY = "crsCooperationModality"
_PROP_FINANCE = "crsTypeOfFinance"
_PROP_FLOW = "crsTypeOfFlow"
_PROP_FLOW_TYPE = "financialFlowType"
_UNBOUND_PROPS = (_PROP_MODALITY, _PROP_FINANCE, _PROP_FLOW)
_FORCED_FLOW_TYPE_VALUE = "Disbursement"

# Every packaged CRS row declares both roles. "To Kenya" binds the recipient;
# coverage rows still include the unbound provider role.
_ROLE_RECIPIENT = "financingRecipient"
_ROLE_PROVIDER = "financingProvider"


def _facet(*, obs_count: int = 3, facet_id: str = "f1") -> Facet:
    """A role-keyed facet with the date and id required for LATEST confirmation."""
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
    )


def test_fixture_matches_the_real_packaged_index():
    """Validate fixture dcids, five-tuple, declared axes, and role arcs against the packaged index."""
    index = get_index()
    target = (_POP_TYPE, _MEAS_PROP, _STAT_TYPE, _MEAS_QUAL, None)

    def _row(dcid: str):
        return next(r for r in index.statvars if r.dcid == dcid)

    recalled_row = _row(_RECALLED_SV)
    extra_row = _row(_EXTRA_SV)
    for row in (recalled_row, extra_row):
        five_tuple = (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )
        assert five_tuple == target
    assert recalled_row.constraint_properties[_PROP_PURPOSE] == _RECALLED_PURPOSE
    assert extra_row.constraint_properties[_PROP_PURPOSE] == _EXTRA_PURPOSE
    assert _RECALLED_PURPOSE != _EXTRA_PURPOSE
    for prop in _UNBOUND_PROPS:
        assert prop in recalled_row.constraint_properties
        assert prop in extra_row.constraint_properties
    for row in (recalled_row, extra_row):
        assert row.constraint_properties[_PROP_FLOW_TYPE] == _FORCED_FLOW_TYPE_VALUE

    # _sv_arcs_from_real_row's own observationProperties arc must carry
    # _RECALLED_SV's real declared roles -- the fake node this module hands
    # derive_shapes must declare the same role signature as the real row it
    # claims to reproduce, or the shape it derives elects against the wrong
    # identity.
    synthetic_arcs = _sv_arcs_from_real_row(_RECALLED_SV)
    synthetic_roles = tuple(n["dcid"] for n in synthetic_arcs["observationProperties"]["nodes"])
    assert synthetic_roles == recalled_row.observation_properties


def _sv_arcs_from_real_row(dcid: str) -> dict:
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    constraint_props = list(row.constraint_properties)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "measurementQualifier": {"nodes": [{"dcid": _MEAS_QUAL}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraint_props]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: {
        "label": "test recalled sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_RECALLED_SV),
    },
    # Grounding looks up index-only winners by name, then reads their declared roles.
    # The synthetic node copies the real row's observationProperties arc.
    _EXTRA_SV: {
        "label": "test index-only sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_EXTRA_SV),
    },
    # The bound purpose value needs a node so `build_slot` can resolve it.
    _EXTRA_PURPOSE: {"label": "CRS purpose 51010"},
}

_DETECT = {_QUERY: {"svs": [_RECALLED_SV], "cosine_scores": [0.9]}}

# Only _EXTRA_SV carries observations, keyed under the role-bearing shape's
# own destination role. The crsPurpose binding already excludes _RECALLED_SV
# -- the only member the recall pool ever confirmed -- from surviving_svs
# before this role-keyed probe ever runs, so the member-probe can only
# succeed by reaching the index-sourced _EXTRA_SV.
_OBS = {_EXTRA_SV: {((_ROLE_RECIPIENT, _KEN_DCID),): [_facet()]}}


class _BindPurposeToExtraLLM:
    """Binds `crsPurpose` to the index-only member and leaves other constraint slots open."""

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
        monkeypatch,
        [(_EXTRA_SV, "f1", "country/FRA", _KEN_DCID, [2020])],
        role_names={_EXTRA_SV: RoleNames(donor=_ROLE_PROVIDER, recipient=_ROLE_RECIPIENT)},
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


def test_index_only_winner_renders_its_bound_slot_value_on_the_wire(monkeypatch):
    """The regression itself: _EXTRA_SV is resolved (it is the one member
    with observations), and it genuinely asserts crsPurpose=_EXTRA_PURPOSE --
    the exact value the query bound. Its StatVar.slot_values must carry that
    assertion, asserted on the rendered value, not merely on the call
    returning without raising."""
    result = _run(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    assert [sv.ref.ref for sv in spec.resolved.members] == [_EXTRA_SV]
    stat_var = spec.resolved.members[0]
    purpose_values = [
        sv.value for sv in stat_var.slot_values if sv.key.property.ref == _PROP_PURPOSE
    ]
    assert len(purpose_values) == 1
    assert purpose_values[0].ref.ref == _EXTRA_PURPOSE


def test_index_only_winner_is_not_reported_inapplicable_for_properties_it_asserts(monkeypatch):
    """_EXTRA_SV asserts all four of the other constraint axes too (real CRS
    rows always do), three of them left unbound by the query. None may be
    reported inapplicable: that would claim no resolved StatVar declares a
    property that _EXTRA_SV -- the very StatVar resolved -- genuinely asserts."""
    result = _run(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    for prop in _UNBOUND_PROPS:
        assert prop not in spec.resolved.inapplicable_slots
        slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == prop
        )
        assert slot.binding.kind == "unbound"

    # The measure election forces financialFlowType to the value every row carries.
    assert _PROP_FLOW_TYPE not in spec.resolved.inapplicable_slots
    flow_type_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == _PROP_FLOW_TYPE
    )
    assert flow_type_slot.binding.kind == "value"
    assert flow_type_slot.binding.value.ref.ref == _FORCED_FLOW_TYPE_VALUE
