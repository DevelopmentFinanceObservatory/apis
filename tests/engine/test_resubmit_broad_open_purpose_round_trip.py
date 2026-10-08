"""Pins resubmission of an ODA grant answer with open purpose and hierarchical finance bindings.

Finance rollups must expand to indexed leaf values when the family is recovered.
"""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from search.engine.clients.graph import Facet
from search.engine.region import resolve_spec_resubmit
from search.engine.region.promote_bindings import _covered_by_posted_binding
from search.engine.region.state import EngineOptions
from search.engine.retrieval.bound_values import (
    elect_default_partition,
    measure_family_partitions,
)
from search.engine.shape import ShapeDraft, identity_shape_for
from search.engine.spec_build import _measure_alternatives
from search.models import (
    BindingSet,
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
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._index_slots import crs_flow_slot_keys, query_for
from tests.engine.test_candidates_measure_alternatives import _ROLE_RECIPIENT, _ROLES
from tests.engine.test_resubmit_alternative_measure_round_trip import (
    _KEN,
    _arcs_for_row,
    _family_graph,
    _sibling_sdmx,
)
from tests.fixtures import FakeGraph

_FINANCE_ROLLUP = "CRSTypeOfFinance100"  # "grant", a real hierarchical parent
_FLOW = "CRSTypeOfFlow10"  # "official development assistance", a flat code
# The amount family's own declared role names (observationProperties on
# every member, see _arcs_for_row in test_resubmit_alternative_measure_round_trip.py).
_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])


def _family_graph_with_finance_rollup(member_rows) -> FakeGraph:
    """Adds a graph label for the rollup, which no member asserts."""
    nodes = {_KEN: {"label": "Kenya", "type": "Country"}, _FINANCE_ROLLUP: {"label": "grant"}}
    for row in member_rows:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(row),
        }
    return FakeGraph(nodes=nodes, obs={})


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


def _set_slot(prop: str, values: list[str]) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        binding=BindingSet(
            values=[
                SlotValue(ref=GraphRef(dcid=v, label=v), value_kind="enum_value") for v in values
            ]
        ),
    )


def _elected_identity():
    """Builds the broad ODA identity with purpose open and routing values bound."""
    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    slots = [
        _unbound_slot("crsCooperationModality"),
        _unbound_slot("crsPurpose"),
        _bound_slot("crsTypeOfFinance", _FINANCE_ROLLUP),
        _bound_slot("crsTypeOfFlow", _FLOW),
        _bound_slot("financialFlowType", "Disbursement"),
        _bound_slot("nettingBasis", "Gross"),
    ]
    return make_base(shape, slots, entities=[destination_entity(_KEN)])


def _real_election():
    """Elects the real amount-family partition with purpose open and finance values widened."""
    index = get_index()
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
    bound_values = {
        "crsTypeOfFinance": list(index.descendants(_FINANCE_ROLLUP)),
        "crsTypeOfFlow": list(index.descendants(_FLOW)),
    }
    partitions = measure_family_partitions(shape, index, axes, query_for(shape, bound_values))
    election = elect_default_partition(partitions, axes)
    assert election is not None
    return election


def _bound_slot_value(identity, prop: str) -> str | None:
    for slot in identity.slots:
        if slot.property == prop and isinstance(slot.binding, IdentityBindingValue):
            return slot.binding.ref
    return None


def _member_rows_for(dcids) -> list:
    index = get_index()
    by_dcid = {row.dcid: row for row in index.statvars}
    rows = [by_dcid[dcid] for dcid in dcids]
    assert rows, "fixture drift: election recovered no real rows"
    return rows


def _round_trip(*, name: str, identity, expect_label: str, monkeypatch) -> None:
    """Resubmits an identity and checks its label and finance binding.

    Family membership comes from the index election. The rollup value must be echoed.
    """
    shape = identity.shape
    dcids = _partition_members_for(shape, identity)
    member_rows = _member_rows_for(dcids)

    # The rollup needs a graph label because members assert only leaf values.
    graph = _family_graph_with_finance_rollup(member_rows)
    sdmx = _sibling_sdmx(member_rows[0].dcid)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(member_rows[0].dcid, "f1", "country/USA", _KEN, [2018, 2019, 2020])],
        role_names={member_rows[0].dcid: _ROLE_NAMES},
    )
    materialise_graph = axis_confirming_reader(*(row.dcid for row in member_rows), graph=graph)

    inp = ResubmitIdentityPayload(
        identity=identity,
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region.status == "definite", (
        name,
        region.no_data_reason if region.status != "definite" else None,
    )
    assert region.specs[0].resolved.measure_label == expect_label, name

    finance_slot = next(
        s
        for s in region.specs[0].resolved.slots
        if s.key.property is not None and s.key.property.dcid == "crsTypeOfFinance"
    )
    assert finance_slot.binding.kind == "value", name
    assert finance_slot.binding.value.ref is not None, name
    assert finance_slot.binding.value.ref.dcid == _FINANCE_ROLLUP, name


def _partition_members_for(shape, identity) -> tuple[str, ...]:
    """Returns the partition members after expanding hierarchical bindings."""
    index = get_index()
    axes = get_measure_index().axes
    synth_shape = ShapeDraft(
        shape_id="",
        label="",
        pop_type_dcid=shape.population_type or "",
        meas_prop_dcid=shape.measured_property or "",
        stat_type_dcid=shape.stat_type or "",
        meas_qual_dcid=shape.measurement_qualifier,
        meas_denom_dcid=None,
        slot_keys=(),
    )
    bound_values = {
        "crsTypeOfFinance": list(index.descendants(_FINANCE_ROLLUP)),
        "crsTypeOfFlow": list(index.descendants(_FLOW)),
    }
    fft = _bound_slot_value(identity, "financialFlowType")
    netting = _bound_slot_value(identity, "nettingBasis")
    obligation = _bound_slot_value(identity, "financialObligationType")
    if fft:
        bound_values["financialFlowType"] = [fft]
    if netting:
        bound_values["nettingBasis"] = [netting]
    if obligation:
        bound_values["financialObligationType"] = [obligation]
    partitions = measure_family_partitions(
        synth_shape, index, axes, query_for(synth_shape, bound_values)
    )
    values = {
        "populationType": shape.population_type,
        "measuredProperty": shape.measured_property,
        "measurementQualifier": shape.measurement_qualifier,
        "financialFlowType": fft,
        "nettingBasis": netting,
        "financialObligationType": obligation,
    }
    match = next(p for p in partitions if dict(p.values) == values)
    return match.members


def test_base_spec_and_every_real_alternative_round_trip_over_an_open_purpose_axis(monkeypatch):
    """The elected identity and both published alternatives recover their labels."""
    election = _real_election()
    assert len(election.siblings) == 2

    _spec_id, elected_identity = _elected_identity()
    alternatives, (elected_label, _, _) = _measure_alternatives(
        elected_identity, election.elected, election.siblings
    )
    assert alternatives is not None
    assert elected_label is not None
    assert len(alternatives) == 2

    # Check the base identity before its alternatives.
    _round_trip(
        name="BASE (self)",
        identity=elected_identity,
        expect_label=elected_label,
        monkeypatch=monkeypatch,
    )

    results: dict[str, str] = {}
    for alternative in alternatives:
        identity = alternative.definition.identity
        kind = "/".join(
            filter(
                None,
                (
                    identity.shape.population_type,
                    identity.shape.measurement_qualifier,
                    _bound_slot_value(identity, "nettingBasis"),
                    _bound_slot_value(identity, "financialObligationType"),
                ),
            )
        )
        _round_trip(
            name=kind,
            identity=identity,
            expect_label=alternative.label,
            monkeypatch=monkeypatch,
        )
        results[kind] = alternative.label

    assert len(results) == 2


def _narrowed_identity(finance_leaf: str):
    """Narrows the finance binding from the rollup to one real leaf."""
    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    slots = [
        _unbound_slot("crsCooperationModality"),
        _unbound_slot("crsPurpose"),
        _bound_slot("crsTypeOfFinance", finance_leaf),
        _bound_slot("crsTypeOfFlow", _FLOW),
        _bound_slot("financialFlowType", "Disbursement"),
        _bound_slot("nettingBasis", "Gross"),
    ]
    _, identity = make_base(shape, slots, entities=[destination_entity(_KEN)])
    return identity


def _multi_sibling_sdmx(sibling_svs) -> FakeSdmxClient:
    """Adds an observation for each member of a multi-member family."""
    observations = {
        sv: {
            ((_ROLE_RECIPIENT, _KEN),): [
                Facet(
                    earliest_date="2020",
                    latest_date="2020",
                    obs_count=3,
                    dates=["2020"],
                    values=[0.0],
                    facet_id="f1",
                )
            ]
        }
        for sv in sibling_svs
    }
    return FakeSdmxClient(observations=observations, availability={})


def _members_for_finance_values(finance_values: list[str]) -> tuple[str, ...]:
    """Returns indexed members matching finance leaves and the other bound routing values."""
    index = get_index()
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
    bound_values = {
        "crsTypeOfFinance": finance_values,
        "crsTypeOfFlow": list(index.descendants(_FLOW)),
        "financialFlowType": ["Disbursement"],
        "nettingBasis": ["Gross"],
    }
    partitions = measure_family_partitions(shape, index, axes, query_for(shape, bound_values))
    values = {
        "populationType": "FinancialFlow",
        "measuredProperty": "amount",
        "measurementQualifier": "Nominal",
        "financialFlowType": "Disbursement",
        "nettingBasis": "Gross",
        "financialObligationType": None,
    }
    match = next(p for p in partitions if dict(p.values) == values)
    return match.members


def test_narrowing_crs_type_of_finance_from_the_rollup_to_a_real_leaf_elects_that_leafs_own_family(
    monkeypatch,
):
    """The elected identity binds `crsTypeOfFinance` to the hierarchical rollup
    (`CRSTypeOfFinance100`), and the client narrows to one real leaf
    (`CRSTypeOfFinance110`): the packaged index's election recovers every real
    member sharing that leaf.
    """
    leaf = "CRSTypeOfFinance110"
    member_dcids = _members_for_finance_values([leaf])
    member_rows = _member_rows_for(member_dcids)

    graph = _family_graph(member_rows)
    sdmx = _multi_sibling_sdmx(member_dcids)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(dcid, "f1", "country/USA", _KEN, [2018, 2019, 2020]) for dcid in member_dcids],
        role_names=dict.fromkeys(member_dcids, _ROLE_NAMES),
    )
    materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

    inp = ResubmitIdentityPayload(identity=_narrowed_identity(leaf))
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region.status == "definite", (
        region.no_data_reason if region.status != "definite" else None
    )
    resolved = region.specs[0].resolved
    assert resolved.stat_vars.member_count == len(member_dcids)
    assert {sv.ref.dcid for sv in resolved.members} <= set(member_dcids)

    # The client's edit, narrowing crsTypeOfFinance from the rollup to one real leaf, reads
    # back as that leaf, not the rollup.
    finance_slot = next(
        s
        for s in region.specs[0].resolved.slots
        if s.key.property is not None and s.key.property.dcid == "crsTypeOfFinance"
    )
    assert finance_slot.binding.kind == "value"
    assert finance_slot.binding.value.ref is not None
    assert finance_slot.binding.value.ref.dcid == leaf


def _set_bound_identity(leaves: list[str]):
    """Binds finance to a set of real leaves instead of the rollup."""
    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    slots = [
        _unbound_slot("crsCooperationModality"),
        _unbound_slot("crsPurpose"),
        _set_slot("crsTypeOfFinance", leaves),
        _bound_slot("crsTypeOfFlow", _FLOW),
        _bound_slot("financialFlowType", "Disbursement"),
        _bound_slot("nettingBasis", "Gross"),
    ]
    return make_base(shape, slots, entities=[destination_entity(_KEN)])


def test_unedited_resubmit_of_a_posted_leaf_set_still_reads_as_that_set(monkeypatch):
    """An unedited leaf set is echoed unchanged."""
    leaf_a, leaf_b = "CRSTypeOfFinance110", "CRSTypeOfFinance210"
    member_dcids = _members_for_finance_values([leaf_a, leaf_b])
    member_rows = _member_rows_for(member_dcids)

    _base_spec_id, identity = _set_bound_identity([leaf_a, leaf_b])

    graph = _family_graph(member_rows)
    sdmx = _multi_sibling_sdmx(member_dcids)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(dcid, "f1", "country/USA", _KEN, [2018, 2019, 2020]) for dcid in member_dcids],
        role_names=dict.fromkeys(member_dcids, _ROLE_NAMES),
    )
    materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

    inp = ResubmitIdentityPayload(identity=identity)
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region.status == "definite", (
        region.no_data_reason if region.status != "definite" else None
    )
    stat_vars = region.specs[0].resolved.stat_vars
    assert stat_vars.member_count == len(member_dcids)
    finance_slot = next(
        s
        for s in region.specs[0].resolved.slots
        if s.key.property is not None and s.key.property.dcid == "crsTypeOfFinance"
    )
    assert finance_slot.binding.kind == "set"
    assert {v.ref.dcid for v in finance_slot.binding.values if v.ref is not None} == {
        leaf_a,
        leaf_b,
    }


def test_covered_by_posted_binding_accepts_a_rollup_covering_its_real_leaf_descendants():
    """The exact case `_round_trip`'s own crsTypeOfFinance assertion depends
    on: every real leaf descendant of the posted rollup is covered by it."""
    index = get_index()
    leaves = [d for d in index.descendants(_FINANCE_ROLLUP) if d != _FINANCE_ROLLUP]
    assert leaves, "fixture drift: expected real leaf descendants of the finance rollup"
    assert _covered_by_posted_binding([_FINANCE_ROLLUP], leaves)
    assert _covered_by_posted_binding([_FINANCE_ROLLUP], leaves[:1])


def test_covered_by_posted_binding_accepts_a_posted_leaf_set_matching_exactly():
    """A posted leaf set covers itself and any subset."""
    leaf_a, leaf_b = "CRSTypeOfFinance110", "CRSTypeOfFinance210"
    assert _covered_by_posted_binding([leaf_a, leaf_b], [leaf_a, leaf_b])
    assert _covered_by_posted_binding([leaf_a, leaf_b], [leaf_a])


def test_covered_by_posted_binding_refuses_a_value_outside_the_posted_rollup():
    """A leaf outside the posted rollup is not covered.

    A full resubmit never reaches this case, so the check is called directly.
    """
    index = get_index()
    finance_descendants = set(index.descendants(_FINANCE_ROLLUP))
    outside_leaf = next(
        leaf
        for row in index.statvars
        if row.population_type == "FinancialFlow"
        and row.measured_property == "amount"
        and row.stat_type == "measuredValue"
        and row.measurement_qualifier == "Nominal"
        and (leaf := row.constraint_properties.get("crsTypeOfFinance")) is not None
        and leaf not in finance_descendants
    )

    assert not _covered_by_posted_binding([_FINANCE_ROLLUP], [outside_leaf])
    inside_leaf = next(iter(finance_descendants - {_FINANCE_ROLLUP}))
    assert not _covered_by_posted_binding([_FINANCE_ROLLUP], [inside_leaf, outside_leaf])


def test_covered_by_posted_binding_refuses_an_empty_posted_value():
    assert not _covered_by_posted_binding([], ["CRSTypeOfFinance110"])
