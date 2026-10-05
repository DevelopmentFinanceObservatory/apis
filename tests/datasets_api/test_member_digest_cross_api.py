"""Search and the Datasets mint state one admitted population by one digest.

Search's `resolved.shape.member_digest` hashes the population its bindings admit. The mint's
`trace.family_member_digest` hashes the family its member resolution admits, before antichain
reduction, and `trace.resolved_stat_vars` lists the members it serves after it. Both digests
come from `statvar_members.member_digest`.
"""

from __future__ import annotations

import httpx
import pytest

import search.engine.retrieval.bound_values as bound_values_module
from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members, to_materialisation_request
from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import compute_member_digest
from statvar_members import MemberEnumerator, member_digest
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_crs_family_member_digest import _EXTRA_SV, _RECALLED_SV
from tests.engine.test_crs_family_member_digest import _run as _search_resolve
from tests.fixtures import FakeGraph

from ._graph_doubles import FakeMintGraphReader
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
# The dataset the packaged index records the two members under, which the mint serves from.
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
    """Search resolves a two-member CRS family from the packaged index, and the mint is
    posted the spec that resolve minted. No packaged row dominates another, so the mint
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


def _person_shape() -> ShapeDraft:
    """The shape Search derives for the person family: one slot key and a recall sample of
    one member."""
    return ShapeDraft(
        shape_id="count_person",
        label="population",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="gender", label="gender"),),
        sv_arc_facts={"Person_Male": {}},
        sv_arc_facts_complete=False,
    )


def test_a_family_with_a_dominated_member_has_one_digest_and_a_reduced_served_set(monkeypatch):
    """The root `Person` dominates the two gendered members. Search admits all three, and
    the mint admits all three and serves the root alone, so a digest over the members served
    differs from Search's and the family digest equals it."""
    index = _person_family_index()
    monkeypatch.setattr(bound_values_module, "get_index", lambda: index)
    place = "country/KEN"
    searched = materialise(
        _person_shape(),
        [],
        {"observationAbout": [place]},
        graph=FakeGraph(
            nodes={},
            obs={
                f"Person|{place}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
            },
            detect={},
            resolve={},
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(searched, Materialised)
    search_digest = compute_member_digest(searched.population)
    assert search_digest == member_digest(_PERSON_FAMILY)

    identity = _gender_family_identity()
    graph = FakeMintGraphReader(arcs={})
    resolved = resolve_definition_members(
        identity,
        spec_id="spec_person",
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=index,
    )
    (served,) = resolved.shapes
    request = to_materialisation_request(identity, shape=served, code_index=index)

    assert served.stat_var == "dcid:Person"
    assert request.family is not None
    assert request.family.digest == search_digest
    assert request.family.count == 3
    assert member_digest(["Person"]) != search_digest


def test_a_graph_walk_that_reads_every_candidate_states_the_same_family():
    identity = _gender_family_identity()
    graph = _person_family_graph()

    resolved = resolve_definition_members(
        identity,
        spec_id="spec_person",
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=_empty_code_index(),
    )

    (served,) = resolved.shapes
    assert served.stat_var == "dcid:Person"
    assert served.family is not None
    assert served.family.digest == member_digest(_PERSON_FAMILY)
    assert served.family.count == 3


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
        enumerator=MemberEnumerator(graph=graph),
        code_index=_empty_code_index(),
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
