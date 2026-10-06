"""The relaxed nearest-real probe can confirm members from the packaged index
that the recall pool missed. Its returned suggestion must use those members'
own facts. The fixture pairs two real CRS StatVars; only the index-only member
has observations.
"""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.region.grounding import _suggest_nearest_real
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._builders import bound_place
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

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
_PROP_FLOW = "crsTypeOfFlow"


def _real_row(dcid: str):
    index = get_index()
    return next(r for r in index.statvars if r.dcid == dcid)


def _sv_arcs_from_real_row(dcid: str) -> dict:
    row = _real_row(dcid)
    constraint_props = list(row.constraint_properties)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "measurementQualifier": {"nodes": [{"dcid": _MEAS_QUAL}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraint_props]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _shape() -> ShapeDraft:
    recalled_row = _real_row(_RECALLED_SV)
    return ShapeDraft(
        shape_id="test_nearest_real_crs_shape",
        label="test shape",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=_MEAS_QUAL,
        meas_denom_dcid=None,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=p, label=p)
            for p in recalled_row.constraint_properties
        ),
        sv_arc_facts={_RECALLED_SV: _sv_arcs_from_real_row(_RECALLED_SV)},
        sv_arc_facts_complete=False,
    )


_BINDINGS = [
    SlotBindingDraft(
        axis="how", property_dcid=_PROP_PURPOSE, kind="value", value_dcids=[_RECALLED_PURPOSE]
    )
]

_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: {"label": "test recalled sibling", "type": "StatisticalVariable"},
    _EXTRA_SV: {"label": "test index-only sibling", "type": "StatisticalVariable"},
}

_OBS = {f"{_EXTRA_SV}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}]}

_ROLES = {_KEN_DCID: "observationAbout"}


def _run():
    graph = FakeGraph(nodes=_NODES, obs=_OBS, detect={}, resolve={})
    return _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_KEN_DCID]},
            destination_dcid=_KEN_DCID,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )


def test_relaxed_probe_confirms_an_index_only_member():
    """Relaxing `crsPurpose` admits the index-only member with observations."""
    suggestions, warnings, _reads, _presence = _run()
    assert warnings == []
    assert len(suggestions) == 1
    assert [sv.ref.dcid for sv in suggestions[0].resolved.members] == [_EXTRA_SV]


def test_relaxed_index_only_member_is_not_reported_inapplicable():
    """The confirmed member declares `crsPurpose`, so the relaxed slot is
    unbound and the property stays applicable."""
    suggestions, _, _reads, _presence = _run()
    assert len(suggestions) == 1
    resolved = suggestions[0].resolved
    purpose_slot = next(
        s
        for s in resolved.slots
        if s.key.property is not None and s.key.property.dcid == _PROP_PURPOSE
    )
    assert purpose_slot.binding.kind == "unbound"
    assert _PROP_PURPOSE not in resolved.inapplicable_slots
    # Both index rows declare this second axis, so it also stays applicable.
    assert _PROP_FLOW not in resolved.inapplicable_slots
