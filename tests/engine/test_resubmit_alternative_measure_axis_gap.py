"""A resubmit of an identity that binds measure axes serves the family a Datasets mint of it
serves (`code_index.members.index_family`): members asserting a slot the identity omits stay
out (conformance-search C184, conformance-datasets C157), and a measure axis bound to a set
serves every value of the set.

The fixture index holds Net rows under CRSPurposeCode700 that carry crsCooperationModality,
a slot the Net alternative's identity omits, so the Net partition leaves them out. The second Net test
adds the modality to one more member, so the sibling's own resolution still round-trips.
"""

from __future__ import annotations

import dataclasses

from code_index import CodeIndex
from code_index.members import index_family
from code_index.shared import get_index, get_measure_index
from search.engine.clients.graph import Facet
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.retrieval import bound_values as bound_values_module
from search.engine.retrieval.bound_values import (
    elect_default_partition,
    identity_query,
    measure_family_partitions,
)
from search.engine.shape import ShapeDraft, identity_shape_for
from search.engine.spec_build import _measure_alternatives
from search.models import (
    BindingUnbound,
    BindingValue,
    GraphRef,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._builders import binding_set_slot
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine.test_candidates_measure_alternatives import _ROLE_NAMES, _ROLE_RECIPIENT
from tests.engine.test_resubmit_alternative_measure_round_trip import (
    _KEN,
    _arcs_for_row,
    _sibling_sdmx,
)
from tests.fixtures import FakeGraph

_PURPOSE_PROP = "crsPurpose"
_PURPOSE = "CRSPurposeCode700"
_MODALITY_PROP = "crsCooperationModality"
_MODALITY_VALUE = "CRSCooperationModalityA02"
# The elected identity omits the fixture's modality axis.
_ROUTING_PROPS = ("crsTypeOfFinance", "crsTypeOfFlow")


def _bound_slot(prop: str, value: str) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(ref=prop, label=prop), label=prop),
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(ref=value, label=value), value_kind="enum_value")
        ),
    )


def _unbound_slot(prop: str) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(ref=prop, label=prop), label=prop),
        binding=BindingUnbound(),
    )


_NOMINAL_FLOW_SHAPE = identity_shape_for(
    pop_type_dcid="FinancialFlow",
    meas_prop_dcid="amount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="Nominal",
    meas_denom_dcid=None,
)


def _elected_identity():
    slots = [
        _bound_slot(_PURPOSE_PROP, _PURPOSE),
        *(_unbound_slot(p) for p in _ROUTING_PROPS),
        _bound_slot("financialFlowType", "Disbursement"),
        _bound_slot("nettingBasis", "Gross"),
    ]
    return make_base(_NOMINAL_FLOW_SHAPE, slots, entities=[destination_entity(_KEN)])


def _real_election():
    """Elects the real Gross/Disbursement partition and its two siblings, read with the
    elected identity's query."""
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
        shape, get_index(), axes, identity_query(_elected_identity()[1])
    )
    election = elect_default_partition(partitions, axes)
    assert election is not None
    return election


def _net_partition(election):
    return next(p for p in election.siblings if dict(p.values).get("nettingBasis") == "Net")


def _net_alternative(alternatives):
    for alt in alternatives:
        for slot in alt.definition.identity.slots:
            if (
                slot.property == "nettingBasis"
                and slot.binding.state == "value"
                and slot.binding.ref == "Net"
            ):
                return alt
    raise AssertionError("fixture drift: no Net alternative published")


def _multi_sibling_sdmx(sibling_svs) -> FakeSdmxClient:
    """Gives every sibling member an observation so it remains in the served set."""
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


def _coverage_for_svs(monkeypatch, svs):
    """Adds a recipient=Kenya row for each member; the donor remains unbound."""
    svs = list(svs)
    rows = [(sv, "f1", "country/USA", _KEN, [2020]) for sv in svs]
    index = patch_coverage(monkeypatch, rows, role_names=dict.fromkeys(svs, _ROLE_NAMES))
    return FakeCoverageService(index).snapshot()


def _augmented_index(target_dcid: str) -> CodeIndex:
    """Adds the missing modality declaration to one row in a copy of the index."""
    real_index = get_index()
    statvars = tuple(
        dataclasses.replace(
            row,
            constraint_properties={**row.constraint_properties, _MODALITY_PROP: _MODALITY_VALUE},
        )
        if row.dcid == target_dcid
        else row
        for row in real_index.statvars
    )
    return CodeIndex(
        stamp=real_index.stamp,
        declarations=real_index.declarations.values(),
        codes=real_index.codes,
        statvars=statvars,
        datasets=real_index.datasets.values(),
    )


def _family_members(identity) -> frozenset[str]:
    """The members a Datasets mint of `identity` reads: its index family's rows."""
    family = index_family(get_index(), identity_query(identity))
    assert family is not None
    return frozenset(row.dcid for row in family.rows)


def _resubmit_over_rows(monkeypatch, identity, rows):
    """Resubmit `identity` with no seed over a graph, observations and coverage holding every
    row of `rows`, so only the member election decides which of them the answer serves."""
    nodes = {_KEN: {"label": "Kenya", "type": "Country"}}
    for row in rows:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(row),
        }
    graph = FakeGraph(nodes=nodes, obs={})
    dcids = [row.dcid for row in rows]
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=_multi_sibling_sdmx(dcids),
        coverage=_coverage_for_svs(monkeypatch, dcids),
        options=walk_options(
            graph, EngineOptions(materialise_graph=axis_confirming_reader(*dcids, graph=graph))
        ),
    )


def _rows_holding(**values: str) -> list:
    """The fixture index's rows asserting every value in `values`."""
    return [
        row
        for row in get_index().statvars
        if all(row.constraint_properties.get(prop) == value for prop, value in values.items())
    ]


def _net_alternative_identity():
    """The elected partition's Net sibling, and the identity of the alternative it mints."""
    election = _real_election()
    _, elected_identity = _elected_identity()
    alternatives, _ = _measure_alternatives(elected_identity, election.elected, election.siblings)
    assert alternatives is not None
    net_alt = _net_alternative(alternatives)
    return _net_partition(election), net_alt


def _net_rows_of_the_purpose() -> list:
    """Every Net disbursement row of the nominal flow five-tuple under the purpose, those
    asserting crsCooperationModality included."""
    purposes = set(get_index().descendants(_PURPOSE))
    return [
        row
        for row in _rows_holding(financialFlowType="Disbursement", nettingBasis="Net")
        if row.constraint_properties.get(_PURPOSE_PROP) in purposes
        and (row.population_type, row.measured_property, row.measurement_qualifier)
        == ("FinancialFlow", "amount", "Nominal")
        and row.measurement_denominator is None
    ]


def _five_slot_crs_resubmit():
    """Sixteen rows share the identity's five bound values. Fifteen also assert
    crsCooperationModality, which the identity omits, so its family is the one row
    asserting no other slot."""
    bound = {
        "crsPurpose": "CRSPurposeCode99810",
        "crsTypeOfFinance": "CRSTypeOfFinance110",
        "crsTypeOfFlow": "CRSTypeOfFlow10",
        "financialFlowType": "Disbursement",
        "nettingBasis": "Gross",
    }
    _, identity = make_base(
        _NOMINAL_FLOW_SHAPE,
        [_bound_slot(prop, value) for prop, value in bound.items()],
        entities=[destination_entity(_KEN)],
    )
    rows = [
        row
        for row in _rows_holding(**bound)
        if (row.population_type, row.measured_property) == ("FinancialFlow", "amount")
        and row.measurement_denominator is None
    ]
    return identity, rows


def _measure_axis_set_resubmit():
    """nettingBasis bound to {Gross, Net}: the family holds a member of each value, beside
    rows asserting slots the identity omits."""
    _, identity = make_base(
        _NOMINAL_FLOW_SHAPE,
        [
            _bound_slot("dac2aAidType", "DAC2aAidType206"),
            _unbound_slot("dacTypeOfFinance"),
            _bound_slot("financialFlowType", "Disbursement"),
            binding_set_slot("how", "nettingBasis", "nettingBasis", ["Gross", "Net"]),
        ],
        entities=[destination_entity(_KEN)],
    )
    rows = [
        row
        for row in _rows_holding(dac2aAidType="DAC2aAidType206", financialFlowType="Disbursement")
        if row.measurement_denominator is None
    ]
    return identity, rows


def test_the_net_sibling_partition_is_the_family_of_the_identity_it_mints():
    """A sibling partition holds the rows its minted identity's family holds: the Net
    alternative's identity omits crsCooperationModality, so the rows asserting it stay out
    of the partition."""
    net_partition, net_alt = _net_alternative_identity()

    family = _family_members(net_alt.definition.identity)

    assert set(net_partition.members) == family
    assert family < {row.dcid for row in _net_rows_of_the_purpose()}  # fixture guard


def test_no_seed_resubmit_of_the_net_alternative_serves_its_identitys_family(monkeypatch):
    """Rows asserting crsCooperationModality, which the Net alternative's identity omits,
    hold the alternative's values; the resubmit serves the identity's family, without
    them."""
    _, net_alt = _net_alternative_identity()
    identity = net_alt.definition.identity
    rows = _net_rows_of_the_purpose()
    family = _family_members(identity)
    assert family < {row.dcid for row in rows}  # fixture guard: the rows are wider

    region = _resubmit_over_rows(monkeypatch, identity, rows)

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.measure_label == net_alt.label
    assert region.specs[0].resolved.stat_vars.member_count == len(family)


def test_resubmit_of_a_five_slot_crs_identity_serves_the_member_asserting_no_other_slot(
    monkeypatch,
):
    """Sixteen rows share the identity's five bound values and fifteen of them also assert
    crsCooperationModality, a slot the identity omits; the answer serves the one row the
    mint serves."""
    identity, rows = _five_slot_crs_resubmit()
    family = _family_members(identity)
    assert (len(rows), len(family)) == (16, 1)  # fixture guard

    region = _resubmit_over_rows(monkeypatch, identity, rows)

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.stat_vars.member_count == 1
    assert region.specs[0].resolved.stat_vars.served_count == 1


def test_resubmit_with_an_unbound_measure_axis_elects_a_value_its_family_holds(monkeypatch):
    """Under these four values only rows asserting crsCooperationModality, a slot the
    identity omits, hold Gross; the rows the identity's family can hold are Net. The
    election forces Net, and the answer serves the family's one member."""
    bound = {
        "crsPurpose": "CRSPurposeCode99810",
        "crsTypeOfFinance": "CRSTypeOfFinance422",
        "crsTypeOfFlow": "CRSTypeOfFlow10",
        "financialFlowType": "Disbursement",
    }
    _, identity = make_base(
        _NOMINAL_FLOW_SHAPE,
        [
            *(_bound_slot(prop, value) for prop, value in bound.items()),
            _unbound_slot("nettingBasis"),
        ],
        entities=[destination_entity(_KEN)],
    )
    rows = [
        row
        for row in _rows_holding(**bound)
        if (row.population_type, row.measured_property) == ("FinancialFlow", "amount")
        and row.measurement_denominator is None
    ]
    _, net_identity = make_base(
        _NOMINAL_FLOW_SHAPE,
        [
            *(_bound_slot(prop, value) for prop, value in bound.items()),
            _bound_slot("nettingBasis", "Net"),
        ],
        entities=[destination_entity(_KEN)],
    )
    family = _family_members(net_identity)
    assert len(family) == 1  # fixture guard
    assert any(row.constraint_properties.get("nettingBasis") == "Gross" for row in rows)

    region = _resubmit_over_rows(monkeypatch, identity, rows)

    assert region.status == "definite", region.no_data_reason
    netting = next(
        s
        for s in region.specs[0].resolved.slots
        if s.key.property is not None and s.key.property.ref == "nettingBasis"
    )
    assert netting.binding.kind == "value"
    assert netting.binding.value.ref is not None
    assert netting.binding.value.ref.ref == "Net"
    assert {sv.ref.ref for sv in region.specs[0].resolved.members} == family
    assert region.specs[0].resolved.stat_vars.member_count == 1


def test_resubmit_of_a_measure_axis_set_binding_serves_every_value_of_the_set(monkeypatch):
    """An identity binding nettingBasis to {Gross, Net} serves a member of each value, as
    the mint does, where one measure partition holds only one."""
    identity, rows = _measure_axis_set_resubmit()
    family = _family_members(identity)
    assert len(family) == 2  # fixture guard: one Gross and one Net member

    region = _resubmit_over_rows(monkeypatch, identity, rows)

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.stat_vars.member_count == 2
    assert region.specs[0].resolved.stat_vars.served_count == 2


def test_the_net_alternative_round_trips_definite_and_remints_its_own_spec_id(
    monkeypatch,
):
    """Posting the Net alternative's `definition` alone, with no seed,
    resolves definite even though its own real members declare
    crsCooperationModality, an axis the minted sibling identity never names
    at all: `mint_sibling_identity` pads no member-only axis onto a
    sibling's slot set, and a resubmit's own declared slot keys are the
    posted identity's own (region.resubmit_promote's _phase_build_shape),
    not the union of what its resolved members happen to assert. The
    resolved spec re-mints the identical spec_id. This sibling round-trips
    through its own posted definition unchanged.
    """
    election = _real_election()
    net_partition = _net_partition(election)
    target_dcid = net_partition.members[0]

    monkeypatch.setattr(bound_values_module, "get_index", lambda: _augmented_index(target_dcid))

    _, elected_identity = _elected_identity()
    alternatives, _ = _measure_alternatives(elected_identity, election.elected, election.siblings)
    assert alternatives is not None
    net_alt = _net_alternative(alternatives)

    # No slot for crsCooperationModality at all (neither bound nor an
    # unbound placeholder), even though the fixture index (patched above)
    # declares it for one of this partition's own real members.
    assert all(slot.property != _MODALITY_PROP for slot in net_alt.definition.identity.slots)

    by_dcid = {row.dcid: row for row in get_index().statvars}
    member_rows = [by_dcid[dcid] for dcid in net_partition.members]
    # Augment the same one real member with the crsCooperationModality arc
    # the fixture-index patch above declared, so the posted graph and the
    # minted identity agree on which member carries it.
    augmented = [
        dataclasses.replace(
            row,
            constraint_properties={**row.constraint_properties, _MODALITY_PROP: _MODALITY_VALUE},
        )
        if row.dcid == target_dcid
        else row
        for row in member_rows
    ]

    nodes = {_KEN: {"label": "Kenya", "type": "Country"}}
    for row in augmented:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(row),
        }
    graph = FakeGraph(nodes=nodes, obs={})
    sdmx = _sibling_sdmx(augmented[0].dcid)
    materialise_graph = axis_confirming_reader(*(row.dcid for row in augmented), graph=graph)

    identity = net_alt.definition.identity
    inp = ResubmitIdentityPayload(identity=identity)

    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=_coverage_for_svs(monkeypatch, (row.dcid for row in augmented)),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.measure_label == net_alt.label
    assert region.specs[0].spec_id == net_alt.definition.spec_id
