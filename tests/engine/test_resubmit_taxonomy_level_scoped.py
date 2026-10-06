"""Fresh and resubmit shapes for the same CRS member publish identical,
level-scoped crsPurpose options. The packaged index supplies the hierarchy
for the member's five-tuple. A resubmit shape never index-sources its members, which once
flattened every how-slot menu.
"""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.region.grounding import _ground_answer, _shape_with_taxonomy
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.fixtures import FakeGraph

# A real CRS member: FinancialFlow/grantEquivalent/measuredValue/Nominal,
# crsPurpose bound to "primary education" (a leaf with no children, one real
# parent -- see test_slot_options_level_scoped.py's own health-family fixture
# for the general pattern).
_SV_DCID = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_"
    "CRSTypeOfFinance110_CRSCooperationModalityA02_CRSPurposeCode11220_Nominal"
)
_PURPOSE = "crsPurpose"
_PRIMARY_EDUCATION = "CRSPurposeCode11220"
_PRIMARY_EDUCATION_PARENT = "CRSPurposeCode112"


def _sv_arcs() -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "grantEquivalent"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {
            "nodes": [
                {"dcid": p}
                for p in (
                    "crsCooperationModality",
                    _PURPOSE,
                    "crsTypeOfFinance",
                    "crsTypeOfFlow",
                    "financialFlowType",
                )
            ]
        },
        "crsCooperationModality": {"nodes": [{"dcid": "CRSCooperationModalityA02"}]},
        _PURPOSE: {"nodes": [{"dcid": _PRIMARY_EDUCATION}]},
        "crsTypeOfFinance": {"nodes": [{"dcid": "CRSTypeOfFinance110"}]},
        "crsTypeOfFlow": {"nodes": [{"dcid": "CRSTypeOfFlow10"}]},
        "financialFlowType": {"nodes": [{"dcid": "Disbursement"}]},
    }


def _shape_draft(*, sv_arc_facts_complete: bool) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="grantEquivalent",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=_PURPOSE, label="sector"),),
        sv_arc_facts={_SV_DCID: _sv_arcs()},
        sv_arc_facts_complete=sv_arc_facts_complete,
    )


def test_fixture_matches_the_real_packaged_index():
    """Independent cross-check, so a corpus refresh that moves this family
    fails loudly here rather than the assertions below silently testing
    stale assumptions."""
    index = get_index()
    row = index.code_by_dcid(_PRIMARY_EDUCATION)
    assert row is not None
    assert row.parent == _PRIMARY_EDUCATION_PARENT
    counts = index.value_counts(_PURPOSE)
    assert counts.get(_PRIMARY_EDUCATION_PARENT, 0) > 0


def _crs_purpose_options(shape_draft: ShapeDraft):
    shape_draft = _shape_with_taxonomy(shape_draft)
    bindings = [
        SlotBindingDraft(
            axis="how", property_dcid=_PURPOSE, kind="value", value_dcids=[_PRIMARY_EDUCATION]
        )
    ]
    result = _ground_answer(
        shape_draft,
        bindings=bindings,
        sv_dcids=[],
        roles={},
        graph=FakeGraph(nodes={}),
        open_axes=frozenset(),
    )
    slot_options = result[7]
    (so,) = [o for o in slot_options if o.key.property.dcid == _PURPOSE]
    return so


def test_fresh_and_noop_resubmit_shapes_publish_the_same_level_scoped_options():
    """A fresh, recall-sourced shape (sv_arc_facts_complete=False, as
    derive_shapes stamps it) and a resubmit-sourced shape (sv_arc_facts_complete=True,
    as region/resubmit_promote.py's re-read stamps it) for the identical member
    and five-tuple publish the identical crsPurpose slot_options: the same
    relation, value, and member_count, exhaustive at their own level
    (truncated=False) rather than the resubmit path flattening to the one
    bound value with no relation."""
    fresh = _crs_purpose_options(_shape_draft(sv_arc_facts_complete=False))
    resubmit = _crs_purpose_options(_shape_draft(sv_arc_facts_complete=True))

    assert fresh.truncated is False
    assert resubmit.truncated is False
    assert [(o.relation, o.value.ref.dcid, o.member_count) for o in fresh.options] == [
        (o.relation, o.value.ref.dcid, o.member_count) for o in resubmit.options
    ]
    assert {o.value.ref.dcid for o in resubmit.options} == {_PRIMARY_EDUCATION_PARENT}
    assert all(o.relation == "parent" for o in resubmit.options)
