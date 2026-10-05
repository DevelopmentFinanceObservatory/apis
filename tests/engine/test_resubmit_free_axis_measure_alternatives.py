"""A no-seed resubmit recovers the selected partition's measure label. The
real CRS family has four populated partitions; alternatives vary on free axes,
including nettingBasis and populationType.

Identities are built directly to test recovery apart from the minter. measurementQualifier has
one observed value graph-wide, so no real fixture flips it.
"""

from __future__ import annotations

from code_index.labels import compose_measure_label
from code_index.shared import get_index, get_measure_index
from code_index.types import label_for_pairs
from search.engine.region import resolve_spec_resubmit
from search.engine.retrieval.bound_values import MeasurePartition, measure_family_partitions
from search.engine.shape import ShapeDraft, identity_shape_for
from search.models import (
    BindingUnbound,
    BindingValue,
    GraphRef,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from series_coverage.schema import RoleNames
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._index_slots import query_for
from tests.engine.test_candidates_measure_alternatives import _ROLES
from tests.engine.test_resubmit_alternative_measure_round_trip import (
    _KEN,
    _family_graph,
    _sibling_sdmx,
)

_PURPOSE_PROP = "crsPurpose"
_PURPOSE = "CRSPurposeCode11330"
# The amount family's own declared role names (observationProperties on
# every member, see _arcs_for_row in test_resubmit_alternative_measure_round_trip.py).
_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])


def _slot(prop: str, value: str) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(dcid=value, label=value), value_kind="enum_value")
        ),
    )


def _unbound_slot(prop: str) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        binding=BindingUnbound(),
    )


def _purpose_scoped_partitions() -> tuple[tuple, tuple[MeasurePartition, ...]]:
    """Return the populated partitions for the packaged CRS-purpose family."""
    axes = get_measure_index().axes
    shape = ShapeDraft(
        shape_id="",
        label="",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(),
    )
    partitions = measure_family_partitions(
        shape, get_index(), axes, query_for(shape, {_PURPOSE_PROP: [_PURPOSE]})
    )
    return axes, partitions


def _partition(partitions: tuple[MeasurePartition, ...], **values: str | None) -> MeasurePartition:
    return next(p for p in partitions if dict(p.values) == values)


def _label_for(partition: MeasurePartition) -> str:
    measure_index = get_measure_index()
    match = label_for_pairs(partition.values, measure_index.partition_labels)
    assert match is not None
    return compose_measure_label(match, measure_index.values)


def _members_matching(**want: str | None) -> list:
    """Return the real family rows matching the requested measure partition."""
    index = get_index()

    def _match(row) -> bool:
        if row.constraint_properties.get(_PURPOSE_PROP) != _PURPOSE:
            return False
        for prop in ("nettingBasis", "financialFlowType", "financialObligationType"):
            if want.get(prop) != row.constraint_properties.get(prop):
                return False
        return (
            row.population_type == want["population_type"]
            and row.measured_property == want["measured_property"]
            and row.measurement_qualifier == want["measurement_qualifier"]
        )

    return [row for row in index.statvars if _match(row)]


def _round_trip(
    *,
    population_type: str,
    measurement_qualifier: str,
    extra_slots: list[Slot],
    monkeypatch,
    **member_axes: str | None,
) -> None:
    _axes, partitions = _purpose_scoped_partitions()
    partition_values = {
        "populationType": population_type,
        "measuredProperty": "amount",
        "measurementQualifier": measurement_qualifier,
        "financialFlowType": None,
        "financialObligationType": None,
        "nettingBasis": None,
        **{
            k: v
            for k, v in member_axes.items()
            if k in ("financialFlowType", "financialObligationType", "nettingBasis")
        },
    }
    partition = _partition(partitions, **partition_values)
    label = _label_for(partition)

    member_rows = _members_matching(
        population_type=population_type,
        measured_property="amount",
        measurement_qualifier=measurement_qualifier,
        **member_axes,
    )
    assert member_rows, "fixture drift: no real row matches this partition"

    # A resubmit's own declared slot keys are the posted identity's own
    # (region.resubmit_promote's _phase_build_shape). crsPurpose and the
    # measure axes are already explicitly bound above; every other property
    # the real member rows assert (crsCooperationModality, crsTypeOfFinance,
    # crsTypeOfFlow, the routing axes) is posted unbound here too, so each
    # one becomes a slot in the shape's own five-tuple family instead of
    # only surfacing as an open axis.
    already_named = {_PURPOSE_PROP, *(s.key.property.dcid for s in extra_slots if s.key.property)}
    other_props = {
        prop
        for row in member_rows
        for prop in row.constraint_properties
        if prop not in already_named
    }
    slots = [
        _slot(_PURPOSE_PROP, _PURPOSE),
        *extra_slots,
        *(_unbound_slot(p) for p in sorted(other_props)),
    ]

    shape = identity_shape_for(
        pop_type_dcid=population_type,
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=measurement_qualifier,
        meas_denom_dcid=None,
    )
    spec_id, identity = make_base(shape, slots, entities=[destination_entity(_KEN)])

    graph = _family_graph(member_rows)
    sdmx = _sibling_sdmx(member_rows[0].dcid)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(member_rows[0].dcid, "f1", "country/USA", _KEN, [2018, 2019, 2020])],
        role_names={member_rows[0].dcid: _ROLE_NAMES},
    )

    inp = ResubmitIdentityPayload(identity=identity, base_spec_id=spec_id, base_identity=identity)
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.measure_label == label


def test_a_netting_basis_flip_round_trips_to_its_own_label(monkeypatch):
    """The free `nettingBasis` axis (Gross -> Net), holding
    `measurementQualifier`/`populationType` fixed."""
    _round_trip(
        population_type="FinancialFlow",
        measurement_qualifier="Nominal",
        extra_slots=[_slot("nettingBasis", "Net"), _slot("financialFlowType", "Disbursement")],
        monkeypatch=monkeypatch,
        nettingBasis="Net",
        financialFlowType="Disbursement",
    )


def test_a_financial_obligation_type_flip_round_trips_to_its_own_label(monkeypatch):
    """The free `populationType` axis (FinancialFlow -> FinancialObligation),
    which carries `financialFlowType`'s single asserted value out and
    `financialObligationType`'s own in -- holding `measurementQualifier`
    fixed."""
    _round_trip(
        population_type="FinancialObligation",
        measurement_qualifier="Nominal",
        extra_slots=[_slot("financialObligationType", "Commitment")],
        monkeypatch=monkeypatch,
        financialObligationType="Commitment",
    )
