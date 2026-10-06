"""Tests that a same-shape merge retains an agreed measure partition and re-mints its siblings.
The fixture varies recipients, so the merged identity must include both entities.
"""

from __future__ import annotations

import search.engine.spec_build as spec_build_module
from code_index.types import MeasureAxis, MeasureIndex, MeasurePartitionLabel, MeasureValueRow
from search.engine.retrieval.bound_values import MeasurePartition, mint_sibling_identity
from search.engine.same_shape import collapse_same_shape
from search.engine.spec_build import build_spec
from search.models import (
    BindingValue,
    CoverageBare,
    Entity,
    EntityRoleDirectional,
    GraphRef,
    ProvenanceRef,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
)
from tests.engine._conjoin_fixtures import _region

_RECIPIENT_PROP = "DevelopmentFinanceRecipient"

_AXES = (MeasureAxis(axis="nettingBasis", kind="constraint", role="basis", prefer=("Gross",)),)
_LABELS = (
    MeasurePartitionLabel(axis_values=(("nettingBasis", "Gross"),), label="gross disbursements"),
    MeasurePartitionLabel(axis_values=(("nettingBasis", "Net"),), label="net disbursements"),
)


_VALUES = tuple(
    MeasureValueRow(
        dcid=dcid, axis="nettingBasis", label=dcid, synonyms=(), short_name=None, matchable=True
    )
    for dcid in ("Gross", "Net")
)


def _measure_index() -> MeasureIndex:
    return MeasureIndex(axes=_AXES, values=_VALUES, partition_labels=_LABELS)


def _partition(value: str) -> MeasurePartition:
    return MeasurePartition(values=(("nettingBasis", value),), members=())


_GROSS = _partition("Gross")
_NET = _partition("Net")


def _gr(dcid: str) -> GraphRef:
    return GraphRef(dcid=dcid, label=dcid)


_CRS = ProvenanceRef(
    dcid="provenance/OECDCreditorReportingSystem", label="OECD Creditor Reporting System (CRS)"
)
_DAC1 = ProvenanceRef(dcid="provenance/OECDDAC1", label="OECD DAC1")


def _recipient_only_shape() -> Shape:
    return Shape(
        shape_id="recipient_only_test_shape",
        label="Test recipient-only shape",
        population_type=_gr("FinancialFlow"),
        measured_property=_gr("amount"),
        stat_type=_gr("measuredValue"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient")],
        member_count=1,
    )


def _where_bound_spec(
    shape: Shape,
    recipient_dcid: str,
    sv_dcid: str,
    *,
    measure_partition=None,
    sources=(),
    also_serves: tuple[str, ...] = (),
):
    where_slot = Slot(
        key=SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        binding=BindingValue(value=SlotValue(ref=_gr(recipient_dcid), value_kind="entity")),
    )
    entity = Entity(
        ref=_gr(recipient_dcid),
        role=EntityRoleDirectional(role=_gr(_RECIPIENT_PROP), axis="destination"),
    )
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[where_slot],
        stat_vars=[
            StatVar(
                ref=_gr(dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
            for dcid in (sv_dcid, *also_serves)
        ],
        entities=[entity],
        coverage=CoverageBare(has_data=True),
        resolved_sources=list(sources),
        measure_partition=measure_partition,
        measure_siblings=(_NET,) if measure_partition is _GROSS else (),
    )


def test_a_group_that_agrees_on_the_observed_partition_re_mints_its_siblings(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _measure_index)
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(
        shape, "country/KEN", "sv_kenya", measure_partition=_GROSS, sources=[_CRS]
    )
    spec_ethiopia = _where_bound_spec(
        shape, "country/ETH", "sv_ethiopia", measure_partition=_GROSS, sources=[_DAC1, _CRS]
    )
    r_a = _region(
        spec_kenya,
        "aid to Kenya",
        earliest_index=0,
        measure_partition=_GROSS,
        measure_siblings=(_NET,),
    )
    r_b = _region(
        spec_ethiopia,
        "aid to Ethiopia",
        earliest_index=1,
        measure_partition=_GROSS,
        measure_siblings=(_NET,),
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]
    # Both recipients appear in the merged identity.
    assert set(merged.spec.definition.identity.entities["destination"]) == {
        "country/KEN",
        "country/ETH",
    }
    assert merged.measure_partition is not None
    assert dict(merged.measure_partition.values) == {"nettingBasis": "Gross"}
    assert merged.measure_siblings == (_NET,)
    assert merged.spec.resolved.measure_label == "gross disbursements"
    # The merged answer cites every member's sources, each once, in group order.
    assert merged.spec.resolved.trace.resolved_sources == [_CRS, _DAC1]
    assert merged.spec.resolved.alternative_measures is not None
    assert len(merged.spec.resolved.alternative_measures) == 1
    assert merged.spec.resolved.alternative_measures[0].label == "net disbursements"

    # Re-minted against the merged spec's own (unioned-entity) identity, not
    # either member's own single-recipient identity.
    sibling_from_kenya_identity = mint_sibling_identity(
        spec_kenya.definition.identity, _NET, _AXES, _LABELS
    ).spec_id
    sibling_from_ethiopia_identity = mint_sibling_identity(
        spec_ethiopia.definition.identity, _NET, _AXES, _LABELS
    ).spec_id
    published_sibling_spec_id = merged.spec.resolved.alternative_measures[0].definition.spec_id
    assert published_sibling_spec_id != sibling_from_kenya_identity
    assert published_sibling_spec_id != sibling_from_ethiopia_identity


def test_a_group_that_disagrees_on_the_observed_partition_publishes_null(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _measure_index)
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya", measure_partition=_GROSS)
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia", measure_partition=_NET)
    r_a = _region(spec_kenya, "aid to Kenya", earliest_index=0, measure_partition=_GROSS)
    r_b = _region(spec_ethiopia, "aid to Ethiopia", earliest_index=1, measure_partition=_NET)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]
    assert merged.measure_partition is None
    assert merged.measure_siblings == ()
    assert merged.spec.resolved.measure_label is None
    assert merged.spec.resolved.alternative_measures is None


def test_a_member_with_no_observed_partition_at_all_also_publishes_null(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _measure_index)
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya", measure_partition=_GROSS)
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia", measure_partition=None)
    r_a = _region(spec_kenya, "aid to Kenya", earliest_index=0, measure_partition=_GROSS)
    r_b = _region(spec_ethiopia, "aid to Ethiopia", earliest_index=1, measure_partition=None)

    effective, _ = collapse_same_shape([r_a, r_b])

    merged = effective[0]
    assert merged.measure_partition is None
    assert merged.spec.resolved.alternative_measures is None
