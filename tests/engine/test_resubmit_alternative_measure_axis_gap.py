"""A CRSPurposeCode700 sibling omits a member-only slot, yet its no-seed resubmit must recover.

The fixture index lacks this purpose's modality variation, so the fixture adds it to one real member.
"""

from __future__ import annotations

import dataclasses

from code_index import CodeIndex
from code_index.shared import get_index, get_measure_index
from search.engine.clients.graph import Facet
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.retrieval import bound_values as bound_values_module
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
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._index_slots import query_for
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
    """Elects the real Gross/Disbursement partition and its two siblings."""
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


def test_no_seed_resubmit_of_the_net_alternative_recovers_its_own_partition_members(
    monkeypatch,
):
    """A no-seed resubmit recovers the Net partition's real members."""
    election = _real_election()
    net_partition = _net_partition(election)

    _, elected_identity = _elected_identity()
    alternatives, _ = _measure_alternatives(elected_identity, election.elected, election.siblings)
    assert alternatives is not None
    net_alt = _net_alternative(alternatives)

    by_dcid = {row.dcid: row for row in get_index().statvars}
    member_rows = [by_dcid[dcid] for dcid in net_partition.members]

    nodes = {_KEN: {"label": "Kenya", "type": "Country"}}
    for row in member_rows:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(row),
        }
    graph = FakeGraph(nodes=nodes, obs={})
    sdmx = _multi_sibling_sdmx(row.dcid for row in member_rows)
    materialise_graph = axis_confirming_reader(*(row.dcid for row in member_rows), graph=graph)

    identity = net_alt.definition.identity
    inp = ResubmitIdentityPayload(identity=identity)
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=_coverage_for_svs(monkeypatch, (row.dcid for row in member_rows)),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.measure_label == net_alt.label
    assert region.specs[0].resolved.stat_vars.member_count == len(net_partition.members)


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
