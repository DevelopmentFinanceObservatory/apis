"""Posts every real minted alternative back through resubmit without a member seed.

The CRSPurposeCode11330 fixture covers siblings that cross `populationType`, exercising identity padding.
"""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.retrieval.bound_values import (
    elect_default_partition,
    measure_family_partitions,
)
from search.engine.shape import ShapeDraft, identity_shape_for
from search.engine.spec_build import _measure_alternatives
from search.models import (
    BindingUnbound,
    BindingValue,
    GraphRef,
    IdentityBindingValue,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from series_coverage.schema import RoleNames
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._harness import walk_options
from tests.engine._index_slots import crs_flow_slot_keys, query_for
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
# All family routing axes are declared on the base identity for confirmation.
_ROUTING_PROPS = ("crsCooperationModality", "crsTypeOfFinance", "crsTypeOfFlow")


def _bound_slot(prop: str, value: str) -> Slot:
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


def _elected_identity():
    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    slots = [
        _bound_slot(_PURPOSE_PROP, _PURPOSE),
        *(_unbound_slot(p) for p in _ROUTING_PROPS),
    ]
    return make_base(shape, slots, entities=[destination_entity(_KEN)])


def _real_election():
    """Elects the default partition of the real CRSPurposeCode11330 family."""
    axes = get_measure_index().axes
    shape = ShapeDraft(
        shape_id="",
        label="",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=crs_flow_slot_keys(),
    )
    partitions = measure_family_partitions(
        shape, get_index(), axes, query_for(shape, {_PURPOSE_PROP: [_PURPOSE]})
    )
    election = elect_default_partition(partitions, axes)
    assert election is not None
    return election


def _bound_slot_value(identity, prop: str) -> str | None:
    for slot in identity.slots:
        if slot.property == prop and isinstance(slot.binding, IdentityBindingValue):
            return slot.binding.ref
    return None


def _members_matching(**want: str | None) -> list:
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


def _round_trip(alternative, monkeypatch) -> str:
    """Resubmits a minted identity and returns its recovered measure label."""
    identity = alternative.definition.identity
    shape = identity.shape
    member_rows = _members_matching(
        population_type=shape.population_type,
        measured_property=shape.measured_property,
        measurement_qualifier=shape.measurement_qualifier,
        nettingBasis=_bound_slot_value(identity, "nettingBasis"),
        financialFlowType=_bound_slot_value(identity, "financialFlowType"),
        financialObligationType=_bound_slot_value(identity, "financialObligationType"),
    )
    assert member_rows, "fixture drift: no real row matches this alternative's own partition"

    graph = _family_graph(member_rows)
    sdmx = _sibling_sdmx(member_rows[0].dcid)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(member_rows[0].dcid, "f1", "country/USA", _KEN, [2018, 2019, 2020])],
        role_names={member_rows[0].dcid: _ROLE_NAMES},
    )
    materialise_graph = axis_confirming_reader(*(row.dcid for row in member_rows), graph=graph)

    inp = ResubmitIdentityPayload(identity=identity)
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.measure_label == alternative.label
    return region.specs[0].resolved.measure_label


def _kind(identity) -> str:
    """Returns a readable name for an alternative's measure kind."""
    shape = identity.shape
    netting = _bound_slot_value(identity, "nettingBasis")
    obligation = _bound_slot_value(identity, "financialObligationType")
    bits = [shape.population_type, shape.measured_property, shape.measurement_qualifier]
    if netting:
        bits.append(netting)
    if obligation:
        bits.append(obligation)
    return "/".join(bits)


def test_every_real_amount_family_alternative_round_trips(monkeypatch):
    """Both real alternatives keep their labels."""
    election = _real_election()
    assert len(election.siblings) == 2

    _, elected_identity = _elected_identity()
    alternatives, (elected_label, _, _) = _measure_alternatives(
        elected_identity, election.elected, election.siblings
    )
    assert alternatives is not None
    assert elected_label is not None
    assert len(alternatives) == 2

    results: dict[str, str] = {}
    for alternative in alternatives:
        results[_kind(alternative.definition.identity)] = _round_trip(alternative, monkeypatch)

    assert len(results) == 2
    for kind, label in results.items():
        assert label, f"{kind} round-tripped with no label"
