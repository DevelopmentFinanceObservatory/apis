"""Search and the Datasets mint state one family by one digest and one served count.

The mint's `trace.family_member_digest` hashes the family its member resolution admits,
before antichain reduction, `trace.family_member_count` counts it, and
`trace.resolved_stat_vars` lists the members it serves after it. Search states the same
three facts (`bound_values.stated_members`) as `resolved.shape.member_digest`,
`resolved.shape.member_count` and `resolved.stat_vars.served_count`: from the same index rows
for a family the code index decides, and from the same `MemberGroup` for a walked one
(`MemberGroup.stated_family`). Both digests come from `statvar_members.member_digest`.
"""

from __future__ import annotations

import httpx
import pytest

import search.engine.retrieval.bound_values as bound_values_module
from code_index.members import FamilyQuery
from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members, to_materialisation_request
from search.engine.retrieval.bound_values import StatedMembers, WalkedMembers, stated_members
from search.models import BaseIdentity
from statvar_members import MemberEnumerator, MemberGroup, member_digest
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._builders import binding_set_slot
from tests.engine.test_crs_family_member_digest import _EXTRA_SV, _RECALLED_SV
from tests.engine.test_crs_family_member_digest import _run as _search_resolve
from tests.engine.test_resubmit_alternative_measure_axis_gap import (
    _KEN,
    _NOMINAL_FLOW_SHAPE,
    _bound_slot,
    _resubmit_over_rows,
    _rows_holding,
    _unbound_slot,
)

from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds
from .test_definition_door import _SCOPE, _STAT_VAR_REF, _client, _coverage_service
from .test_definition_door import _graph as _pinned_graph
from .test_resolve_definition_members_index import (
    _empty_code_index,
    _gender_family_identity,
    _person_family_graph,
    _person_family_index,
)

_PROVIDER = "country/FRA"
_RECIPIENT = "country/KEN"
# The dataset the fixture index records the two members under, which the mint serves from.
_PROVENANCE = "provenance/OECDCreditorReportingSystem"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)

_PERSON_FAMILY = {"Person", "Person_Male", "Person_Female"}


def _real_row(dcid: str):
    index = get_index()
    return next(r for r in index.statvars if r.dcid == dcid)


def _member_arcs(dcid: str) -> dict[str, tuple[str, ...]]:
    row = _real_row(dcid)
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": (row.population_type,),
        "measuredProperty": (row.measured_property,),
        "observationProperties": ("financingProvider", "financingRecipient"),
        "constraintProperties": tuple(row.constraint_properties),
        **{prop: (value,) for prop, value in row.constraint_properties.items()},
    }


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={_RECALLED_SV: _member_arcs(_RECALLED_SV), _EXTRA_SV: _member_arcs(_EXTRA_SV)},
        labels={_PROVIDER: "France", _RECIPIENT: "Kenya"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    body = _CSV_HEADER + (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_RECALLED_SV},{_PROVIDER},{_RECIPIENT},USD,"
        f"GrantEquivalent,NotApplicable,{_PROVENANCE},2020,1000,,facet1\r\n"
    )
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def test_the_mints_family_digest_equals_the_digest_of_the_search_resolve_it_mints(monkeypatch):
    """Search resolves a two-member CRS family from the fixture index, and the mint is
    posted the spec that resolve minted. No fixture row dominates another, so the mint
    serves the whole family."""
    region = _search_resolve(monkeypatch)
    assert region.status == "definite"
    spec = region.specs[0]
    assert spec.resolved.shape.member_digest is not None

    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        coverage=_coverage_service(
            stat_vars=(_RECALLED_SV, _EXTRA_SV),
            donor_role="financingProvider",
            donor=_PROVIDER,
            recipient=_RECIPIENT,
            provenance=_PROVENANCE,
        ),
    )
    definition = {
        "schema_version": 1,
        "spec_id": spec.spec_id,
        "identity": spec.definition.identity.model_dump(mode="json", exclude_none=True),
    }

    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec.spec_id, "definition": definition},
            "options": {"bundle": False},
        },
    )

    assert resp.status_code == 200, resp.text
    trace = resp.json()["trace"]
    assert trace["family_member_digest"] == spec.resolved.shape.member_digest
    assert trace["family_member_count"] == spec.resolved.shape.member_count == 2
    assert {entry["ref"] for entry in trace["resolved_stat_vars"]} == {
        f"dcid:{_RECALLED_SV}",
        f"dcid:{_EXTRA_SV}",
    }


def test_a_family_with_a_dominated_member_has_one_digest_and_a_reduced_served_set(monkeypatch):
    """The root `Person` dominates the two gendered members. The code index decides the
    identity, and Search and the mint both state the family of three and serve the root
    alone."""
    index = _person_family_index()
    monkeypatch.setattr(bound_values_module, "get_index", lambda: index)
    identity = _gender_family_identity()

    stated = stated_members(BaseIdentity.model_validate(identity), None)
    resolved = resolve_definition_members(
        identity,
        spec_id="spec_person",
        graph=FakeMintGraphReader(arcs={}),
        enumerator=MemberEnumerator(),
        code_index=index,
        bounds=mint_bounds(),
    )
    (served,) = resolved.shapes
    request = to_materialisation_request(identity, shape=served, code_index=index)

    assert served.stat_var == "dcid:Person"
    assert request.family is not None
    assert (request.family.digest, request.family.count) == (member_digest(_PERSON_FAMILY), 3)
    assert stated == StatedMembers(served_count=1, digest=request.family.digest, family_count=3)


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


@pytest.mark.parametrize(
    ("resubmit", "family_count"),
    [(_five_slot_crs_resubmit, 1), (_measure_axis_set_resubmit, 2)],
    ids=["five-slot-crs-identity", "measure-axis-set-binding"],
)
def test_a_resubmit_counts_the_family_the_members_door_counts(
    monkeypatch, resubmit, family_count: int
):
    """The answer's graph holds rows asserting slots the identity omits, which the member
    walk admits and the identity's family leaves out. The Spec counts the family it digests,
    as the members door does."""
    identity, rows = resubmit()
    region = _resubmit_over_rows(monkeypatch, identity, rows)
    assert region.status == "definite", region.no_data_reason
    (spec,) = region.specs
    shape = spec.resolved.shape
    assert shape.member_digest is not None

    resp = _client(graph=FakeMintGraphReader(arcs={})).post(
        f"/api/datasets/{spec.spec_id}/members",
        json={
            "schema_version": 1,
            "definition": {
                "schema_version": 1,
                "spec_id": spec.spec_id,
                "identity": spec.definition.identity.model_dump(mode="json", exclude_none=True),
            },
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (shape.member_digest, shape.member_count) == (
        body["family_member_digest"],
        body["family_member_count"],
    )
    assert shape.member_count == family_count


class _RecordingEnumerator(MemberEnumerator):
    """An enumerator that keeps every group it returns."""

    def __init__(self) -> None:
        super().__init__()
        self.groups: list[MemberGroup] = []

    def enumerate(self, **kwargs) -> MemberGroup:  # type: ignore[override]
        group = super().enumerate(**kwargs)
        self.groups.append(group)
        return group


def test_a_graph_walk_that_reads_every_candidate_states_the_same_family():
    """Search states what the mint states of one walk's group: its antichain-minimal members
    served and its whole family digested."""
    identity = _gender_family_identity()
    graph = _person_family_graph()
    enumerator = _RecordingEnumerator()

    resolved = resolve_definition_members(
        identity,
        spec_id="spec_person",
        graph=graph,
        enumerator=enumerator,
        code_index=_empty_code_index(),
        bounds=mint_bounds(),
    )

    (served,) = resolved.shapes
    (group,) = enumerator.groups
    walked = WalkedMembers.of(FamilyQuery(group.five_tuple, slots={}, source=None), group)
    assert served.stat_var == "dcid:Person"
    assert served.family is not None
    assert served.family.digest == member_digest(_PERSON_FAMILY)
    assert served.family.count == 3
    assert walked.stated == StatedMembers(
        served_count=len(resolved.shapes), digest=served.family.digest, family_count=3
    )


@pytest.mark.parametrize("stat_var", ["Person", "Person_Male"])
def test_a_pinned_identity_admits_its_one_stat_var(stat_var: str):
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": stat_var},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    graph = _person_family_graph()
    (shape,) = resolve_definition_members(
        identity,
        spec_id="spec_pinned",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=_empty_code_index(),
        bounds=mint_bounds(),
    ).shapes

    request = to_materialisation_request(identity, shape=shape, code_index=_empty_code_index())

    assert request.family is not None
    assert request.family.digest == member_digest([stat_var])
    assert request.family.count == 1


def test_a_one_member_mint_states_its_family_in_the_trace():
    """A `statvar_set` of one StatVar mints a pinned spec, whose family is that StatVar."""
    client = _client(graph=_pinned_graph())

    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
            "options": {"bundle": False},
        },
    )

    assert resp.status_code == 200, resp.text
    trace = resp.json()["trace"]
    assert trace["family_member_digest"] == member_digest([_STAT_VAR_REF.removeprefix("dcid:")])
    assert trace["family_member_count"] == 1
