"""The confirm sample can omit a slot present in the full index population.

The fixture uses a fixture-index row without ``crsCooperationModality`` to pin that
``derive_shapes`` declares the widened slot."""

from __future__ import annotations

from code_index.query import row_five_tuple
from code_index.shared import get_index
from search.engine.retrieval.derive import derive_shapes
from tests.fixtures import FakeGraph

_POP_TYPE = "FinancialFlow"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"
_MEAS_PROP = "amount"
_PROP_MODALITY = "crsCooperationModality"
_PROP_PURPOSE = "crsPurpose"

# This fixture-index row omits modality, like a confirm sample missing its siblings.
_BARE_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode11110_Nominal"
)


def _sv_arcs_from_real_row(dcid: str) -> dict:
    """Builds graph arcs from the fixture index row for ``dcid``."""
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
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def test_fixture_sv_really_omits_the_widened_property():
    """Confirms the fixture row omits ``crsCooperationModality``."""
    row = next(r for r in get_index().statvars if r.dcid == _BARE_SV)
    assert row.measured_property == _MEAS_PROP
    assert _PROP_MODALITY not in row.constraint_properties
    assert _PROP_PURPOSE in row.constraint_properties


def test_a_capped_sample_missing_a_property_still_declares_the_index_populations_slot():
    """The index population declares a slot missing from the single-member confirm sample."""
    arcs = _sv_arcs_from_real_row(_BARE_SV)
    graph = FakeGraph(
        nodes={_BARE_SV: {"label": _BARE_SV, "type": "StatisticalVariable", "arcs": arcs}},
        detect={},
        resolve={},
        obs={},
    )

    shapes = derive_shapes(confirmed_svs=[_BARE_SV], graph=graph)

    assert len(shapes) == 1
    shape = shapes[0]
    assert shape.slot_taxonomy_index_sourced is True

    modality_slot = next((s for s in shape.slot_keys if s.property_dcid == _PROP_MODALITY), None)
    assert modality_slot is not None, (
        f"expected {_PROP_MODALITY} as a declared slot, got "
        f"{[s.property_dcid for s in shape.slot_keys]}"
    )
    assert shape.slot_taxonomy is not None
    taxonomy_key = f"{modality_slot.axis}:{_PROP_MODALITY}"
    assert shape.slot_taxonomy.get(taxonomy_key), "the widened slot must carry realizable values"

    # A property the sample already carried keeps its own ordinary slot,
    # unaffected by the widening.
    purpose_slot = next((s for s in shape.slot_keys if s.property_dcid == _PROP_PURPOSE), None)
    assert purpose_slot is not None


def test_a_crs_sample_declares_the_slots_of_its_own_five_tuples_rows_alone():
    """The slot keys describe the sample's whole family and nothing beyond it. An identity
    built from a CRS sample carries every property the CRS rows of the aid five-tuple carry,
    none that only the DAC1 rows of the same five-tuple carry, and none that only the
    imputed multilateral rows, on their own five-tuple, carry."""
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == _BARE_SV)
    five_tuple_properties = {
        prop
        for other in index.statvars
        if row_five_tuple(other) == row_five_tuple(row)
        and set(other.provenances) == set(row.provenances)
        for prop in other.constraint_properties
    }
    assert "dac1AidType" not in five_tuple_properties
    graph = FakeGraph(
        nodes={
            _BARE_SV: {
                "label": _BARE_SV,
                "type": "StatisticalVariable",
                "arcs": _sv_arcs_from_real_row(_BARE_SV),
            }
        },
        detect={},
        resolve={},
        obs={},
    )

    (shape,) = derive_shapes(confirmed_svs=[_BARE_SV], graph=graph)

    assert {s.property_dcid for s in shape.slot_keys if s.axis == "how"} == five_tuple_properties
    assert "dac2aAidType" not in five_tuple_properties
