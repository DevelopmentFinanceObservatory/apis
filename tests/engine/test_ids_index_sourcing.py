"""IDS flows share CRS's `FinancialFlow/amount/measuredValue/Nominal` five-tuple but are
not code-index rows. The index decides a member only when its constraint keys cover an index
row (`bound_values.member_family`), and an identity only when its slot keys do
(`index_covers_slot_keys`), so an IDS pool reads its own arcs: its taxonomy, its resubmit's
walked members, its bind prompt, and its options population."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest

from code_index.members import index_covers_slot_keys, index_family
from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import StatVarRow
from promptlab.expectations import case_stat_vars
from search.engine.bind import _build_universal_bind_prompt
from search.engine.errors import EngineInputError
from search.engine.options import population as options_population_module
from search.engine.qualifier_match import QualifierMatch
from search.engine.region import promote_reread as promote_reread_module
from search.engine.region import resolve_spec_resubmit
from search.engine.region.phase_rank_bind import _rank_key
from search.engine.region.promote_edits import _phase_detect_edits
from search.engine.region.promote_reread import _elected_members, _phase_reread_arcs
from search.engine.region.promote_shape import _phase_build_shape
from search.engine.region.promote_state import _PromoteState
from search.engine.region.promote_survivors import _phase_resolve_survivors
from search.engine.region.promote_validate import _phase_validate
from search.engine.region.result import RegionResult
from search.engine.retrieval import bound_values as bound_values_module
from search.engine.retrieval.bound_values import (
    ResubmitWalk,
    RoleClassWalks,
    collapse_measure_families,
    elect_five_tuple_family,
    elect_measure_family,
    held_codes,
    identity_query,
    index_carries_shape,
    query_phrases_read,
    shape_query,
)
from search.engine.retrieval.derive import derive_shapes, member_group_class
from search.engine.retrieval.facts import FiveTuple as ShapeFiveTuple
from search.engine.shape import ShapeDraft, identity_shape_for
from search.eval.cases import load_cases
from search.models import (
    BaseIdentity,
    BindingUnbound,
    BindingValue,
    GraphRef,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from series_coverage.schema import RoleNames
from statvar_members import DEFAULT_MEMBER_CAP, FiveTuple, MemberGroup
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import member_walk_reader
from tests.fixtures import FakeGraph
from tests.origin_catalogue.support import listing_base_variables
from tests.statvar_members.conftest import FakeMemberArcReader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KEN = "country/KEN"
_PPG = "PublicOrPubliclyGuaranteed"
_IDS_DEBT_SERVICE = "Amount_FinancialFlow_ServicingPayment_PublicOrPubliclyGuaranteed_Nominal"
_IDS_DISBURSEMENT = "Amount_FinancialFlow_Disbursement_PublicOrPubliclyGuaranteed_Nominal"
_IDS_INTEREST = "Amount_FinancialFlow_InterestPayment_PublicOrPubliclyGuaranteed_Nominal"
# The debt-service total's by-creditor twin declares the creditor role too, as on the
# dev graph, so the shape's roles admit CRS rows (both roles) to its measure family.
_IDS_DEBT_SERVICE_BY_CREDITOR = (
    "Amount_FinancialFlow_ServicingPayment_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
)
_IDS_FLOWS = {
    _IDS_DEBT_SERVICE: "ServicingPayment",
    _IDS_DEBT_SERVICE_BY_CREDITOR: "ServicingPayment",
    _IDS_DISBURSEMENT: "Disbursement",
    _IDS_INTEREST: "InterestPayment",
}
_CRS_PURPOSE = "CRSPurposeCode11110"
_FLOW_FIVE_TUPLE = FiveTuple(
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)


def _ids_arcs(flow_type: str, *, by_creditor: bool = False) -> dict:
    """An IDS PPG flow's arcs, as the dev graph serves the debt-service total."""
    roles = ["financingProvider", "financingRecipient"] if by_creditor else ["financingRecipient"]
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": f"{flow_type} on PPG external debt"}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {
            "nodes": [{"dcid": "financialFlowType"}, {"dcid": "guaranteeStatus"}]
        },
        "observationProperties": {"nodes": [{"dcid": role} for role in roles]},
        "financialFlowType": {"nodes": [{"dcid": flow_type}]},
        "guaranteeStatus": {"nodes": [{"dcid": _PPG, "name": "public or publicly guaranteed"}]},
    }


def _crs_rows() -> list:
    rows = [
        row
        for row in get_index().statvars
        if row.population_type == "FinancialFlow"
        and row.measured_property == "amount"
        and row.measurement_qualifier == "Nominal"
        and row.constraint_properties.get("crsPurpose") == _CRS_PURPOSE
    ][:3]
    assert rows, "fixture drift: expected CRS amount/Nominal rows for CRSPurposeCode11110"
    return rows


def _crs_arcs(row) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": row.dcid}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        "observationProperties": {
            "nodes": [{"dcid": p} for p in row.observation_properties or ("financingRecipient",)]
        },
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _graph(arcs_by_sv: dict[str, dict]) -> FakeGraph:
    nodes: dict = {_KEN: {"label": "Kenya", "type": "Country"}}
    for sv, arcs in arcs_by_sv.items():
        nodes[sv] = {"label": sv, "type": "StatisticalVariable", "arcs": arcs}
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _derive_one(arcs_by_sv: dict[str, dict]) -> ShapeDraft:
    svs = list(arcs_by_sv)
    shapes = derive_shapes(
        confirmed_svs=svs,
        graph=_graph(arcs_by_sv),
        sv_scores=dict.fromkeys(svs, 0.8),
        sv_types=dict.fromkeys(svs, "StatisticalVariable"),
    )
    assert len(shapes) == 1, [s.shape_id for s in shapes]
    return shapes[0]


def _ids_shape() -> ShapeDraft:
    return _derive_one(
        {
            sv: _ids_arcs(flow, by_creditor=sv == _IDS_DEBT_SERVICE_BY_CREDITOR)
            for sv, flow in _IDS_FLOWS.items()
        }
    )


def _crs_shape() -> ShapeDraft:
    return _derive_one({row.dcid: _crs_arcs(row) for row in _crs_rows()})


def _taxonomy_for(shape: ShapeDraft, prop: str) -> list[str]:
    taxonomy = shape.slot_taxonomy or {}
    return next(values for key, values in taxonomy.items() if key.endswith(f":{prop}"))


def _withheld_measure_axes() -> frozenset[str]:
    """The set `phase_rank_bind._phase_bind` passes to the binder."""
    return frozenset(axis.axis for axis in get_measure_index().axes if axis.kind == "constraint")


# Derive: the taxonomy an IDS pool binds against.


def test_an_ids_flow_pool_derives_its_taxonomy_from_its_own_members():
    shape = _ids_shape()

    assert shape.slot_taxonomy_index_sourced is False
    assert _taxonomy_for(shape, "guaranteeStatus") == [_PPG]
    assert set(_taxonomy_for(shape, "financialFlowType")) == set(_IDS_FLOWS.values())


def test_a_crs_pool_on_the_same_five_tuple_stays_index_sourced():
    shape = _crs_shape()

    assert shape.slot_taxonomy_index_sourced is True
    assert index_carries_shape(shape)


# Derive: a recall pool mixing index rows and other members splits on membership.


def _mixed_shapes() -> tuple[ShapeDraft, ShapeDraft]:
    """Derive a pool holding CRS index rows and the IDS PPG flows on one five-tuple."""
    arcs = {row.dcid: _crs_arcs(row) for row in _crs_rows()}
    arcs.update(
        (sv, _ids_arcs(flow, by_creditor=sv == _IDS_DEBT_SERVICE_BY_CREDITOR))
        for sv, flow in _IDS_FLOWS.items()
    )
    svs = list(arcs)
    shapes = derive_shapes(
        confirmed_svs=svs,
        graph=_graph(arcs),
        sv_scores=dict.fromkeys(svs, 0.8),
        sv_types=dict.fromkeys(svs, "StatisticalVariable"),
    )
    assert len(shapes) == 2, [s.shape_id for s in shapes]
    crs = next(s for s in shapes if s.slot_taxonomy_index_sourced)
    ids = next(s for s in shapes if not s.slot_taxonomy_index_sourced)
    return crs, ids


def test_a_mixed_pool_derives_an_index_shape_and_a_recall_shape():
    crs, ids = _mixed_shapes()

    assert set(crs.sv_arc_facts or {}) == {row.dcid for row in _crs_rows()}
    assert set(ids.sv_arc_facts or {}) == set(_IDS_FLOWS)
    assert index_carries_shape(crs)
    assert not index_carries_shape(ids)


def test_the_index_half_of_a_split_five_tuple_keeps_its_shape_id():
    """The index's family keeps the five-tuple's id and the other half takes a suffix."""
    crs, ids = _mixed_shapes()

    assert crs.shape_id == "financialflow_amount_measuredvalue_nominal"
    assert ids.shape_id.startswith(f"{crs.shape_id}_")
    assert set(crs.sv_arc_facts or {}).isdisjoint(ids.sv_arc_facts or {})


def test_the_index_half_of_a_split_ranks_first_on_a_full_tie():
    crs, ids = _mixed_shapes()
    tied = dataclasses.replace(ids, sv_arc_facts=dict(list((ids.sv_arc_facts or {}).items())[:3]))
    crs_tied = dataclasses.replace(
        crs, sv_arc_facts=dict(list((crs.sv_arc_facts or {}).items())[:3])
    )

    assert sorted([tied, crs_tied], key=_rank_key) == [crs_tied, tied]


def test_the_index_shape_of_a_mixed_pool_equals_the_pure_crs_shape():
    """The CRS half of a mixed pool binds exactly as the CRS rows would alone."""
    crs, _ = _mixed_shapes()

    assert crs == _crs_shape()


def test_the_index_shape_of_a_mixed_pool_elects_its_crs_family():
    crs, ids = _mixed_shapes()
    index = get_index()
    axes = get_measure_index().axes

    election = elect_measure_family(crs, index, axes, shape_query(crs, [], ()))

    assert election is not None
    assert set(election.elected.members) <= {row.dcid for row in index.statvars}
    assert elect_measure_family(ids, index, axes, shape_query(ids, [], ())) is None


def test_the_recall_shape_of_a_mixed_pool_keeps_its_own_taxonomy():
    _, ids = _mixed_shapes()

    assert _taxonomy_for(ids, "guaranteeStatus") == [_PPG]
    assert set(_taxonomy_for(ids, "financialFlowType")) == set(_IDS_FLOWS.values())


# Election: an unindexed family split off an index family's five-tuple is its rival, and
# the family holding fewer of the values the query names leaves the election.

_IDS_NET_FLOW = "Amount_FinancialFlow_Net_Disbursement_PrincipalRepayment_DebtInstrument_IDSConcessional_Nominal"


def _ids_net_flow_arcs() -> dict:
    """The IDS concessional net-flow member's arcs: it asserts nettingBasis=Net."""
    constraints = {
        "financialFlowType": "Disbursement",
        "financingResource": "DebtInstrument",
        "idsConcessionality": "IDSConcessional",
        "netOf": "PrincipalRepayment",
        "nettingBasis": "Net",
    }
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Net flows on concessional external debt"}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        "observationProperties": {"nodes": [{"dcid": "financingRecipient"}]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


def _named(code_system: str, code_dcid: str) -> QualifierMatch:
    return QualifierMatch(
        code_system=code_system,
        code_dcid=code_dcid,
        label=code_dcid,
        source="label",
        matched_text=code_dcid,
    )


def _half(shape: ShapeDraft) -> tuple[bool, frozenset[str]]:
    """A split half by its sourcing and members."""
    return index_carries_shape(shape), frozenset(shape.sv_arc_facts or {})


def _collapse(
    shapes: list[ShapeDraft], *named: QualifierMatch
) -> list[tuple[bool, frozenset[str]]]:
    return [
        _half(shape)
        for shape in elect_five_tuple_family(
            shapes, lambda shape: query_phrases_read(held_codes(shape), named, ())
        )
    ]


def test_a_named_value_only_the_index_shape_carries_drops_the_recall_shape():
    crs, ids = _mixed_shapes()

    assert _collapse([ids, crs], _named("nettingBasis", "Net")) == [_half(crs)]


def test_no_named_value_keeps_both_halves_of_a_mixed_pool():
    crs, ids = _mixed_shapes()

    assert _collapse([ids, crs]) == [_half(ids), _half(crs)]


def test_a_named_value_both_halves_carry_keeps_both():
    crs, ids = _mixed_shapes()

    assert _collapse([ids, crs], _named("financialFlowType", "Disbursement")) == [
        _half(ids),
        _half(crs),
    ]


def test_a_recall_shape_asserting_the_named_value_stays_beside_the_index_shape():
    crs, ids = _mixed_shapes()
    pool = {**(ids.sv_arc_facts or {}), _IDS_NET_FLOW: _ids_net_flow_arcs()}
    ids_with_net = dataclasses.replace(ids, sv_arc_facts=pool)

    assert _collapse([ids_with_net, crs], _named("nettingBasis", "Net")) == [
        _half(ids_with_net),
        _half(crs),
    ]


def test_an_ids_only_pool_naming_its_own_net_value_keeps_its_shape():
    ids = _derive_one({_IDS_NET_FLOW: _ids_net_flow_arcs()})

    assert _collapse([ids], _named("nettingBasis", "Net")) == [_half(ids)]


# Measure family: a recall-sourced shape never elects or collapses into a CRS family.


def test_a_recall_sourced_shape_elects_no_index_family():
    index = get_index()
    axes = get_measure_index().axes
    ids, crs = _ids_shape(), _crs_shape()

    assert elect_measure_family(ids, index, axes, shape_query(ids, [], ())) is None
    assert elect_measure_family(crs, index, axes, shape_query(crs, [], ())) is not None


def test_a_recall_sourced_shape_passes_through_family_collapse():
    """An IDS `FinancialFlow` shape keeps its place beside a CRS `FinancialObligation`
    shape; `populationType` is a free measure axis, so two index shapes would collapse."""
    ids_shape = _ids_shape()
    commitment_row = next(
        row
        for row in get_index().statvars
        if row.population_type == "FinancialObligation"
        and row.measured_property == "amount"
        and row.measurement_qualifier == "Nominal"
    )
    crs_sibling = dataclasses.replace(
        ids_shape,
        shape_id="financialobligation_amount_measuredvalue_nominal",
        pop_type_dcid="FinancialObligation",
        sv_arc_facts={commitment_row.dcid: _crs_arcs(commitment_row)},
    )

    collapsed = collapse_measure_families(
        [crs_sibling, ids_shape], get_index(), get_measure_index().axes
    )

    assert collapsed == (crs_sibling, ids_shape)


# Bind: a recall-sourced shape binds its measure axis; an index-sourced one withholds it.


def test_a_recall_sourced_financial_flow_shape_renders_its_flow_type_slot():
    prompt = _build_universal_bind_prompt(
        "public and publicly guaranteed external debt service",
        [_ids_shape()],
        withheld_measure_axes=_withheld_measure_axes(),
    )

    assert "property_dcid=financialFlowType" in prompt
    assert "ServicingPayment" in prompt
    assert "property_dcid=guaranteeStatus" in prompt


def test_an_index_sourced_financial_flow_shape_withholds_its_flow_type_slot():
    prompt = _build_universal_bind_prompt(
        "ODA disbursements", [_crs_shape()], withheld_measure_axes=_withheld_measure_axes()
    )

    assert "property_dcid=financialFlowType" not in prompt
    assert "property_dcid=crsPurpose" in prompt


def test_a_crs_bind_prompt_is_byte_identical_to_withholding_every_measure_slot():
    """A CRS prompt equals the one built when every shape withheld its measure slots."""
    crs = _crs_shape()
    withheld = _withheld_measure_axes()
    stripped = dataclasses.replace(
        crs, slot_keys=tuple(s for s in crs.slot_keys if s.property_dcid not in withheld)
    )

    assert _build_universal_bind_prompt(
        "ODA disbursements", [crs], withheld_measure_axes=withheld
    ) == _build_universal_bind_prompt("ODA disbursements", [stripped])


# Resubmit: an IDS identity elects its members by the graph walk.


def _slot(prop: str, value: str | None) -> Slot:
    key = SlotKey(axis="how", property=GraphRef(ref=prop, label=prop), label=prop)
    if value is None:
        return Slot(key=key, binding=BindingUnbound())
    return Slot(
        key=key,
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(ref=value, label=value), value_kind="enum_value")
        ),
    )


def _flow_identity(slots: list[Slot]) -> tuple[str, BaseIdentity]:
    shape = identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )
    return make_base(shape, slots, entities=[destination_entity(_KEN)])


def _promote_state(graph: FakeGraph, slots: list[Slot]) -> _PromoteState:
    """A resubmit's state over `graph`, whose StatVars the member walk reads."""
    _spec_id, identity = _flow_identity(slots)
    return _PromoteState(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=cast(Any, graph),
        sdmx=cast(Any, None),
        deadline=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=empty_coverage_snapshot(),
        role_class_walks=RoleClassWalks(
            member_walk_reader(graph),
            resubmit=ResubmitWalk(subject_ceiling=20_000, full_read_deadline=None),
        ),
    )


@pytest.mark.usefixtures("unlisted_index")
def test_an_ids_identity_with_guarantee_status_unbound_resubmits_over_its_walked_member(
    bind_catalogue,
):
    slots = [_slot("financialFlowType", None), _slot("guaranteeStatus", None)]
    arcs = {_IDS_DEBT_SERVICE: _ids_arcs("ServicingPayment")}
    graph = _graph(arcs)
    # The catalogue lists the walked member, so the member walk reads this graph.
    bind_catalogue(listing_base_variables(arcs))
    state = _promote_state(graph, slots)

    _phase_validate(state)
    assert _phase_reread_arcs(state) is None

    assert state.stat_var_dcids == [_IDS_DEBT_SERVICE]
    assert state.measure_elected_properties == frozenset()


def _index_listing(*rows: StatVarRow) -> CodeIndex:
    """The packaged index with `rows` as its only listed rows."""
    index = copy.copy(get_index())
    index.listed_statvars = rows
    index.listed_by_five_tuple = {_FLOW_FIVE_TUPLE: rows} if rows else {}
    return index


def test_an_ids_identity_elects_its_listed_members_without_walking_the_graph(monkeypatch):
    """A family holding a row the index lists is elected from the index: over a graph that
    holds no StatVar, where a walk finds nothing, the listed debt-service total is elected.
    Runs with `GRAPH_RELEASE_ORIGIN` off: with it on, the walk of this family reads the bound
    release, which holds IDS members, in place of the empty graph."""
    monkeypatch.delenv("GRAPH_RELEASE_ORIGIN", raising=False)
    listed = StatVarRow(
        dcid=_IDS_DEBT_SERVICE,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=("financingRecipient",),
        constraint_properties={"financialFlowType": "ServicingPayment", "guaranteeStatus": _PPG},
        provenances=("provenance/WorldBankInternationalDebtStatistics",),
    )
    slots = [_slot("financialFlowType", None), _slot("guaranteeStatus", None)]
    graph = _graph({})

    listing = _index_listing(listed)
    monkeypatch.setattr(bound_values_module, "get_index", lambda: listing)
    state = _promote_state(graph, slots)
    _phase_validate(state)
    assert _elected_members(state) == [_IDS_DEBT_SERVICE]

    unlisted = _index_listing()
    monkeypatch.setattr(bound_values_module, "get_index", lambda: unlisted)
    walked = _promote_state(graph, slots)
    _phase_validate(walked)
    assert not isinstance(_elected_members(walked), list)


def _resubmit_electing(monkeypatch, members: list[str]) -> RegionResult:
    """A resubmit of the IDS debt-service identity at Kenya whose election settles
    `members`, over a graph holding only the debt-service total, which holds data at Kenya."""
    monkeypatch.setattr(promote_reread_module, "_elected_members", lambda _state: members)
    graph = _graph({_IDS_DEBT_SERVICE: _ids_arcs("ServicingPayment")})
    coverage = build_coverage_index(
        [(_IDS_DEBT_SERVICE, "f", "", _KEN, [2020])],
        role_names={_IDS_DEBT_SERVICE: RoleNames(donor=None, recipient="financingRecipient")},
    )
    _spec_id, identity = _flow_identity(
        [_slot("financialFlowType", None), _slot("guaranteeStatus", None)]
    )
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=FakeSdmxClient(),
        coverage=FakeCoverageService(coverage).snapshot(),
    )


def test_an_elected_member_the_graph_lacks_drops_under_a_warning_naming_it(monkeypatch):
    """The code index can list a member the graph no longer holds. The served answer reads
    the members the graph holds, and its `members_absent_from_graph` warning names the one
    it left out; with every member held, no warning."""
    region = _resubmit_electing(monkeypatch, [_IDS_DEBT_SERVICE, _IDS_INTEREST])

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.stat_vars.member_count == 1
    [warning] = [w for w in region.warnings if w.code == "members_absent_from_graph"]
    assert _IDS_INTEREST in warning.message

    held = _resubmit_electing(monkeypatch, [_IDS_DEBT_SERVICE])
    assert held.status == "definite", held.no_data_reason
    assert not [w for w in held.warnings if w.code == "members_absent_from_graph"]


# Options and mint: an IDS identity enumerates from the graph.


@pytest.mark.usefixtures("unlisted_index")
def test_options_enumerates_an_ids_identity_from_the_graph(monkeypatch):
    enumerated: list[FiveTuple] = []

    class _RecordingEnumerator:
        member_cap = DEFAULT_MEMBER_CAP

        def enumerate(self, *, five_tuple, **_kwargs):
            enumerated.append(five_tuple)
            return MemberGroup(
                five_tuple=five_tuple,
                members=(),
                family=(),
                root_confirmed=False,
                candidates_considered=0,
                candidate_cap=1,
                truncated=False,
                proven_complete=True,
            )

    monkeypatch.setattr(
        options_population_module, "shared_enumerator", lambda: _RecordingEnumerator()
    )
    _base_spec_id, identity = _flow_identity(
        [_slot("financialFlowType", "ServicingPayment"), _slot("guaranteeStatus", _PPG)]
    )

    with pytest.raises(EngineInputError, match="denotes no StatVar population"):
        options_population_module.options_population(
            identity, cast(Any, FakeMemberArcReader(arcs={}))
        )

    assert len(enumerated) == 1


def test_the_mint_resolves_an_ids_identity_from_the_graph_and_a_crs_identity_from_the_index():
    _base_spec_id, ids_identity = _flow_identity(
        [_slot("financialFlowType", "ServicingPayment"), _slot("guaranteeStatus", _PPG)]
    )
    _base_spec_id, crs_identity = _flow_identity(_crs_identity_slots())

    assert index_family(get_index(), identity_query(ids_identity)) is None
    family = index_family(get_index(), identity_query(crs_identity))
    assert family is not None and _crs_rows()[0].dcid in {row.dcid for row in family.rows}


# One rule for a seedless identity: the index sources it only when a row fits it.

_IDS_DISBURSEMENT_INDICATOR = "DT.DIS.DPPG.CD"
# A probe identity of the shape Search serves for "Kenya disbursements on external debt"
# when it elects the IDS recall shape and leaves the guarantee status unbound.
_PROBE_SLOTS = [_slot("financialFlowType", "Disbursement"), _slot("guaranteeStatus", None)]
_PPG_BOUND_SLOTS = [_slot("financialFlowType", "Disbursement"), _slot("guaranteeStatus", _PPG)]
# The same probe as Search minted it while graph reads still listed the import-added
# `sourceIndicatorCode` as a constraint. The registry holds specs of this shape.
_PRE_STRIP_PROBE_SLOTS = [*_PROBE_SLOTS, _slot("sourceIndicatorCode", None)]


def _crs_identity_slots() -> list[Slot]:
    """A pure CRS identity: the slot keys of one CRS row, its flow type and purpose bound."""
    row = _crs_rows()[0]
    return [
        _slot(prop, value if prop in ("financialFlowType", "crsPurpose") else None)
        for prop, value in row.constraint_properties.items()
    ]


def _ids_disbursement_reader() -> FakeMemberArcReader:
    """The dev graph's IDS PPG disbursement total, the one member the probe identity denotes,
    as `GraphClient` reads it: the code's own arc without its constraint-list entry."""
    return FakeMemberArcReader(
        arcs={
            _IDS_DISBURSEMENT: {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("FinancialFlow",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "measurementQualifier": ("Nominal",),
                "measurementDenominator": (),
                "constraintProperties": ("financialFlowType", "guaranteeStatus"),
                "observationProperties": ("financingRecipient",),
                "financialFlowType": ("Disbursement",),
                "guaranteeStatus": (_PPG,),
                "sourceIndicatorCode": (_IDS_DISBURSEMENT_INDICATOR,),
            }
        }
    )


def test_the_mint_sources_the_probe_identity_from_the_graph():
    _base_spec_id, identity = _flow_identity(_PROBE_SLOTS)

    assert index_family(get_index(), identity_query(identity)) is None


def test_the_mint_sources_a_crs_identity_from_the_index():
    _base_spec_id, identity = _flow_identity(_crs_identity_slots())

    family = index_family(get_index(), identity_query(identity))

    assert family is not None and family.rows


def _crs_slots_with_purpose(purpose: str) -> list[Slot]:
    return [
        _slot(slot.key.property.ref, purpose) if slot.key.property.ref == "crsPurpose" else slot
        for slot in _crs_identity_slots()
        if slot.key.property is not None
    ]


def test_sourcing_reads_slot_keys_and_never_bound_values():
    """Stage one: the index sources a CRS identity whose bound purpose no row asserts,
    since its slot keys fit; the IDS probe identity's keys fit no row, whatever it binds."""
    index = get_index()
    for slots, sourced in (
        (_crs_identity_slots(), True),
        (_crs_slots_with_purpose("CRSPurposeCodeDoesNotExist999"), True),
        (_PROBE_SLOTS, False),
        (_PPG_BOUND_SLOTS, False),
        (_PRE_STRIP_PROBE_SLOTS, False),
    ):
        _base_spec_id, identity = _flow_identity(slots)
        slot_keys = {slot.property for slot in identity.slots}
        assert index_covers_slot_keys(index, _FLOW_FIVE_TUPLE, slot_keys) is sourced


def test_bound_values_decide_which_index_members_exist():
    """Stage two: an index-sourced identity whose values match no row has no members."""
    _base_spec_id, identity = _flow_identity(
        _crs_slots_with_purpose("CRSPurposeCodeDoesNotExist999")
    )

    family = index_family(get_index(), identity_query(identity))

    assert family is not None and family.rows == ()


def test_a_seedless_crs_identity_with_unmatched_values_resubmits_to_no_data():
    state = _promote_state(_graph({}), _crs_slots_with_purpose("CRSPurposeCodeDoesNotExist999"))

    _phase_validate(state)
    region = _phase_reread_arcs(state)

    assert region is not None and region.status == "no_data"


@pytest.mark.parametrize(
    "slots", [_PROBE_SLOTS, _PRE_STRIP_PROBE_SLOTS], ids=["probe", "minted-before-the-strip"]
)
def test_options_enumerates_the_probe_identity_from_the_graph(slots):
    """A spec minted before the strip, with an unbound `sourceIndicatorCode` slot its member
    no longer lists, still enumerates the same member."""
    _base_spec_id, identity = _flow_identity(slots)

    population = options_population_module.options_population(identity, _ids_disbursement_reader())

    assert population.source == "graph"
    assert population.stat_var_dcids == (_IDS_DISBURSEMENT,)


def test_options_sources_a_crs_identity_from_the_index():
    _base_spec_id, identity = _flow_identity(_crs_identity_slots())

    population = options_population_module.options_population(identity, _ids_disbursement_reader())

    assert population.source == "index"
    assert _IDS_DISBURSEMENT not in population.stat_var_dcids


@pytest.mark.usefixtures("unlisted_index")
@pytest.mark.parametrize("slots", [_PROBE_SLOTS, _PPG_BOUND_SLOTS], ids=["probe", "ppg-bound"])
def test_an_ids_identity_whose_graph_holds_no_member_answers_no_matching_slice(slots):
    state = _promote_state(_graph({}), slots)

    _phase_validate(state)
    region = _phase_reread_arcs(state)

    assert region is not None
    assert (region.status, region.no_data_reason) == ("no_data", "no_matching_slice")


def test_a_crs_identity_resubmits_over_the_index_election():
    state = _promote_state(_graph({}), _crs_identity_slots())

    _phase_validate(state)
    _phase_reread_arcs(state)

    family = index_family(get_index(), identity_query(state.inp.identity))
    assert family is not None
    assert set(state.stat_var_dcids) <= {row.dcid for row in family.rows}


def test_an_ids_shape_id_is_the_same_whether_or_not_crs_rows_were_recalled():
    """The id reads the group alone: the IDS members on a five-tuple the index carries take
    the `_by_role` suffix with or without a CRS group beside them, and the CRS group keeps
    the five-tuple's slug with or without the IDS group."""
    crs, ids = _mixed_shapes()

    assert _ids_shape().shape_id == ids.shape_id == f"{crs.shape_id}_by_role"
    assert _crs_shape().shape_id == crs.shape_id == "financialflow_amount_measuredvalue_nominal"


def _ids_no_op_resubmit() -> tuple[ShapeDraft, _PromoteState]:
    _, ids = _mixed_shapes()
    graph = _graph({_IDS_DEBT_SERVICE: _ids_arcs("ServicingPayment")})
    slots = [_slot("financialFlowType", "ServicingPayment"), _slot("guaranteeStatus", _PPG)]
    return ids, _promote_state(graph, slots)


def _crs_no_op_resubmit() -> tuple[ShapeDraft, _PromoteState]:
    rows = _crs_rows()
    graph = _graph({row.dcid: _crs_arcs(row) for row in rows})
    return _crs_shape(), _promote_state(graph, _crs_identity_slots())


@pytest.mark.parametrize(
    "no_op_resubmit", [_ids_no_op_resubmit, _crs_no_op_resubmit], ids=["ids", "crs-index"]
)
def test_a_no_op_resubmit_keeps_its_answers_shape_id_and_label(no_op_resubmit):
    """A resubmit rebuilds the shape derive gave the answer: the graph-decided IDS shape
    under its member's name, the index-decided CRS shape under its family label."""
    derived, state = no_op_resubmit()

    for phase in (
        _phase_validate,
        _phase_reread_arcs,
        _phase_detect_edits,
        _phase_resolve_survivors,
        _phase_build_shape,
    ):
        assert phase(state) is None

    assert state.regenerated_shape_id == derived.shape_id
    assert state.shape_draft.label == derived.label


# The constraint keys each IDS case's expected StatVar lists once the graph read drops the
# import-added `sourceIndicatorCode` (`dc_graph.IMPORT_ADDED_CONSTRAINT_PROPERTIES`).
_IDS_CASE_MEMBER_KEYS = {
    "Amount_FinancialFlow_ServicingPayment_PublicOrPubliclyGuaranteed_Nominal": (
        "financialFlowType",
        "guaranteeStatus",
    ),
    "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_Nominal": (
        "financialPositionType",
        "guaranteeStatus",
    ),
    "Amount_FinancialPosition_AmountOutstanding_ByCreditor_Nominal": ("financialPositionType",),
    "Amount_FinancialFlow_InterestPayment_Nominal": ("financialFlowType",),
}

# The IDS StatVars whose only constraint key, `financialFlowType`, every CRS row of their
# five-tuple also lists.
_IDS_FLOW_TYPE_ONLY_MEMBERS = (
    "Amount_FinancialFlow_InterestPayment_Nominal",
    "Amount_FinancialFlow_InterestPayment_ByCreditor_Nominal",
    "Amount_FinancialFlow_ServicingPayment_Nominal",
    "Amount_FinancialFlow_ServicingPayment_ByCreditor_Nominal",
)


def _member_five_tuple(sv: str) -> ShapeFiveTuple:
    """The five-tuple `derive_shapes` reads for an IDS `Amount_<population>_..._Nominal`."""
    return ShapeFiveTuple(sv.split("_")[1], "amount", "measuredValue", "Nominal", None)


def _member_arcs(keys: tuple[str, ...]) -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": key} for key in keys]},
        "observationProperties": {"nodes": [{"dcid": "financingRecipient"}]},
    }


@pytest.mark.parametrize(
    ("sv", "keys"),
    [
        *_IDS_CASE_MEMBER_KEYS.items(),
        *((sv, ("financialFlowType",)) for sv in _IDS_FLOW_TYPE_ONLY_MEMBERS),
    ],
)
def test_an_ids_member_is_graph_decided(sv, keys):
    """An IDS member's keys cover no index row of its five-tuple, so the mint walks the graph
    for its identity and derive puts it in a graph-decided shape. Every CRS row lists `crs*`
    keys no IDS member lists, so this holds even for a member whose every key a CRS row
    lists."""
    _role_bearing, family = member_group_class(sv, _member_five_tuple(sv), _member_arcs(keys))
    assert family is None


def test_every_ids_case_member_has_its_keys_recorded():
    members = {
        sv for case in load_cases() if case.id.startswith("ids-") for sv in case_stat_vars(case)
    }
    assert members == set(_IDS_CASE_MEMBER_KEYS), "record the keys of every IDS case member"


def test_no_index_row_lists_the_import_added_source_indicator_code():
    assert not any(
        "sourceIndicatorCode" in row.constraint_properties for row in get_index().statvars
    )
