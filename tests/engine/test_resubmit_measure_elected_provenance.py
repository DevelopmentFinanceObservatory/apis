"""The packaged CRS grant-equivalent family forces
financialFlowType=Disbursement. These tests pin its `measure_elected`
provenance on no-seed, edited, and widened resubmits.
"""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.clients.graph import Facet
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.shape import identity_shape_for
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
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_candidates_measure_alternatives import _ROLE_RECIPIENT, _ROLES
from tests.engine.test_resubmit_alternative_measure_round_trip import (
    _KEN,
    _family_graph,
    _sibling_sdmx,
)

_CRS_PURPOSE = "CRSPurposeCode51010"
_FORCED_FLOW_TYPE_VALUE = "Disbursement"
_REP_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)
# The family's own declared role names (observationProperties on every
# member, see _arcs_for_row in test_resubmit_alternative_measure_round_trip.py).
_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])


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


def _member_rows():
    index = get_index()
    rows = [
        row
        for row in index.statvars
        if row.population_type == "FinancialFlow"
        and row.measured_property == "grantEquivalent"
        and row.measurement_qualifier == "Nominal"
        and row.constraint_properties.get("crsPurpose") == _CRS_PURPOSE
    ]
    assert rows, "fixture drift: expected real grantEquivalent/CRSPurposeCode51010 members"
    return rows


def _shape():
    return identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="grantEquivalent",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )


def _filter_for(spec, prop: str):
    return next(
        f
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None and f.key.property.dcid == prop
    )


def test_named_family_election_reads_measure_elected_on_resubmit(monkeypatch):
    """A no-seed resubmit that only names `crsPurpose` recovers this
    family's members through `elect_measure_family`
    (`state.stat_var_dcids_from_election`), and the axis that election
    forces -- `financialFlowType` -- must read `"measure_elected"` and be
    bound to the forced value rather than the "widened axes win last"
    unbound reset a client's own posted-unbound axis gets. `crsPurpose`,
    the client's own posted value, still reads `"binder"`.
    """
    member_rows = _member_rows()
    slots = [
        _unbound_slot("crsCooperationModality"),
        _bound_slot("crsPurpose", _CRS_PURPOSE),
        _unbound_slot("crsTypeOfFinance"),
        _unbound_slot("crsTypeOfFlow"),
        _unbound_slot("financialFlowType"),
    ]
    base_spec_id, identity = make_base(_shape(), slots, entities=[destination_entity(_KEN)])

    graph = _family_graph(member_rows)
    sdmx = _sibling_sdmx(_REP_SV)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_REP_SV, "f1", "country/USA", _KEN, [2018, 2019, 2020])],
        role_names={_REP_SV: _ROLE_NAMES},
    )
    materialise_graph = axis_confirming_reader(*(row.dcid for row in member_rows), graph=graph)

    inp = ResubmitIdentityPayload(
        identity=identity, base_spec_id=base_spec_id, base_identity=identity
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=EngineOptions(materialise_graph=materialise_graph),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]

    flow_type_filter = _filter_for(spec, "financialFlowType")
    assert flow_type_filter.provenance == "measure_elected"
    assert flow_type_filter.binding_kind == "value"
    assert flow_type_filter.refs and flow_type_filter.refs[0].dcid == _FORCED_FLOW_TYPE_VALUE

    purpose_filter = _filter_for(spec, "crsPurpose")
    assert purpose_filter.provenance == "binder"


def test_edited_slot_reads_binder_even_on_the_forceable_family(monkeypatch):
    """The other half: a named family's own membership always comes from
    the packaged index's election now, whether or not a seed is posted, so
    the seeded and unseeded paths agree -- `financialFlowType` reads
    `"measure_elected"` here too, the same axis the election forces on the
    no-seed path above. Narrowing `crsCooperationModality` -- the edit this
    resubmit actually makes -- reads `"binder"`, the client's own choice,
    since that axis is never one the election forces.
    """
    member_rows = _member_rows()
    seed_dcids = [row.dcid for row in member_rows]
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(dcid, "f1", "country/USA", _KEN, [2018, 2019, 2020]) for dcid in seed_dcids],
        role_names=dict.fromkeys(seed_dcids, _ROLE_NAMES),
    )

    base_slots = [
        _unbound_slot("crsCooperationModality"),
        _bound_slot("crsPurpose", _CRS_PURPOSE),
        _unbound_slot("crsTypeOfFinance"),
        _unbound_slot("crsTypeOfFlow"),
        _unbound_slot("financialFlowType"),
    ]
    base_spec_id, base_identity = make_base(
        _shape(), base_slots, entities=[destination_entity(_KEN)]
    )

    narrowed_slots = [
        _bound_slot("crsCooperationModality", "CRSCooperationModalityA01"),
        _bound_slot("crsPurpose", _CRS_PURPOSE),
        _unbound_slot("crsTypeOfFinance"),
        _unbound_slot("crsTypeOfFlow"),
        _unbound_slot("financialFlowType"),
    ]
    _, narrowed_identity = make_base(_shape(), narrowed_slots, entities=[destination_entity(_KEN)])

    graph = _family_graph(member_rows)
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
        for sv in seed_dcids
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})
    materialise_graph = axis_confirming_reader(*seed_dcids, graph=graph)

    inp = ResubmitIdentityPayload(
        identity=narrowed_identity, base_spec_id=base_spec_id, base_identity=base_identity
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=EngineOptions(materialise_graph=materialise_graph),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]

    cooperation_filter = _filter_for(spec, "crsCooperationModality")
    assert cooperation_filter.provenance == "binder"
    assert (
        cooperation_filter.refs and cooperation_filter.refs[0].dcid == "CRSCooperationModalityA01"
    )

    flow_type_filter = _filter_for(spec, "financialFlowType")
    assert flow_type_filter.provenance == "measure_elected"


def test_widening_a_base_bound_axis_still_reads_unbound_on_the_no_seed_path(monkeypatch):
    """A third case, alongside the two above: a property the client itself
    had bound in `base_identity`, posted unbound in this same no-seed
    resubmit, is a genuine widening -- distinct from `financialFlowType`
    (the election-forced axis, which the first test above shows stays
    bound). `crsCooperationModality` varies across this family's own members
    (unlike `financialFlowType`, which every member shares `Disbursement`
    for), so the election leaves it open rather than forcing it; posting it
    unbound over a base-bound value must still read unbound in the response,
    the same "widened axes win last" rule as before this fix, since
    `crsCooperationModality` was never in `measure_elected_properties` to
    begin with.
    """
    member_rows = _member_rows()

    base_slots = [
        _bound_slot("crsCooperationModality", "CRSCooperationModalityA01"),
        _bound_slot("crsPurpose", _CRS_PURPOSE),
        _unbound_slot("crsTypeOfFinance"),
        _unbound_slot("crsTypeOfFlow"),
        _unbound_slot("financialFlowType"),
    ]
    base_spec_id, base_identity = make_base(
        _shape(), base_slots, entities=[destination_entity(_KEN)]
    )

    posted_slots = [
        _unbound_slot("crsCooperationModality"),
        _bound_slot("crsPurpose", _CRS_PURPOSE),
        _unbound_slot("crsTypeOfFinance"),
        _unbound_slot("crsTypeOfFlow"),
        _unbound_slot("financialFlowType"),
    ]
    _, posted_identity = make_base(_shape(), posted_slots, entities=[destination_entity(_KEN)])

    graph = _family_graph(member_rows)
    sdmx = _sibling_sdmx(_REP_SV)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_REP_SV, "f1", "country/USA", _KEN, [2018, 2019, 2020])],
        role_names={_REP_SV: _ROLE_NAMES},
    )
    materialise_graph = axis_confirming_reader(*(row.dcid for row in member_rows), graph=graph)

    inp = ResubmitIdentityPayload(
        identity=posted_identity, base_spec_id=base_spec_id, base_identity=base_identity
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=EngineOptions(materialise_graph=materialise_graph),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]

    cooperation_filter = _filter_for(spec, "crsCooperationModality")
    assert cooperation_filter.binding_kind == "unbound"
    assert cooperation_filter.refs == []

    flow_type_filter = _filter_for(spec, "financialFlowType")
    assert flow_type_filter.binding_kind == "value"
    assert flow_type_filter.refs and flow_type_filter.refs[0].dcid == _FORCED_FLOW_TYPE_VALUE
    assert flow_type_filter.provenance == "measure_elected"
