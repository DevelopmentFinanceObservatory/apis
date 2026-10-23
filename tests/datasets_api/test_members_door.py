"""`POST /api/datasets/{id}/members` serves the members a mint of the spec serves.

The composed app covers each resolution path, the graph walk, the code index and the pinned
arm, for a registered id and for an unregistered definition the body carries. The members,
their count and the family's count and digest equal the mint's `trace`. The door writes
nothing, refuses as the mint does, and no input takes a posted member list.
"""

from __future__ import annotations

import httpx
import pytest

from code_index.query import CodeIndex
from code_index.types import IndexStamp, StatVarRow
from materialise.graph import Facet
from spec_registry.identity import recompute_identity

from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import described_identity, pinned_identity
from .conftest import FakeDatasetStore, wire_definition_for
from .test_definition_door import _client, _coverage_service
from .test_resolve_definition_members_guards import _InconsistentReadGraphReader

_RECIPIENT = "country/KEN"
_ROOT = "Count_Person"
_MALE = "Count_Person_Male"
_FEMALE = "Count_Person_Female"
_HOUSEHOLD = "Count_Household"

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_LABELS = {_ROOT: "Population", _MALE: "Male population", _FEMALE: "Female population"}


def _arcs(*, population_type: str = "Person", gender: str | None = None) -> dict:
    arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": (population_type,),
        "measuredProperty": ("count",),
        "observationProperties": ("recipient",),
    }
    if gender is not None:
        arcs["constraintProperties"] = ("gender",)
        arcs["gender"] = (gender,)
    return arcs


def _graph(*, with_root: bool = True) -> FakeMintGraphReader:
    """A `Person`/`count` family beside an unrelated `Household` StatVar.

    With the root, the identity resolves to `Count_Person` alone, since the root dominates the
    gendered members. Without it, the identity resolves to the two gendered members.
    """
    arcs = {
        _MALE: _arcs(gender="Male"),
        _FEMALE: _arcs(gender="Female"),
        _HOUSEHOLD: _arcs(population_type="Household"),
    }
    if with_root:
        arcs[_ROOT] = _arcs()
    return FakeMintGraphReader(
        arcs=arcs,
        facets={(stat_var, _RECIPIENT): [_FACET] for stat_var in (_ROOT, _MALE, _FEMALE)},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya", **_LABELS},
    )


def _sdmx_transport() -> httpx.MockTransport:
    stat_vars = (_ROOT, _MALE, _FEMALE)

    def _respond(request: httpx.Request) -> httpx.Response:
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_RECIPIENT},USD,GrantEquivalent,"
            "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
            for stat_var in stat_vars
            if stat_var in str(request.url).replace(f"{stat_var}_", "")
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(_respond)


def _family_index() -> CodeIndex:
    """Carries the `Person`/`count` five-tuple, so resolution reads the index and skips the walk."""

    def _row(dcid: str, constraints: dict[str, str]) -> StatVarRow:
        return StatVarRow(
            dcid=dcid,
            population_type="Person",
            measured_property="count",
            stat_type="measuredValue",
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=("recipient",),
            constraint_properties=constraints,
            provenances=("OECD_DAC2A",),
        )

    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=(),
        statvars=(
            _row(_ROOT, {}),
            _row(_MALE, {"gender": "Male"}),
            _row(_FEMALE, {"gender": "Female"}),
        ),
    )


def _described(*, stat_type: str | None = None) -> dict:
    shape = {"kind": "statvar_schema", "population_type": "Person", "measured_property": "count"}
    if stat_type is not None:
        shape["stat_type"] = stat_type
    return described_identity(
        shape=shape,
        slots=[{"property": "gender", "binding": {"state": "unbound"}}],
        entities={"recipient": [_RECIPIENT]},
    )


def _definition(document: dict) -> dict:
    return wire_definition_for(spec_id=recompute_identity(document).spec_id, identity=document)


def _mint_body(document: dict) -> dict:
    return {
        "schema_version": 1,
        "input": {
            "kind": "spec_id",
            "spec_id": recompute_identity(document).spec_id,
            "definition": _definition(document),
        },
        "options": {"bundle": False},
    }


def _members(client, document: dict, *, body: dict | None = None) -> httpx.Response:
    spec_id = recompute_identity(document).spec_id
    if body is None:
        body = {"schema_version": 1, "definition": _definition(document)}
    return client.post(f"/api/datasets/{spec_id}/members", json=body)


def _members_client(
    *,
    graph: FakeMintGraphReader | None = None,
    store: FakeDatasetStore | None = None,
    code_index: CodeIndex | None = None,
):
    return _client(
        graph=graph if graph is not None else _graph(),
        store=store,
        sdmx_transport=_sdmx_transport(),
        code_index=code_index,
        coverage=_coverage_service(stat_vars=(_ROOT, _MALE, _FEMALE), recipient=_RECIPIENT),
    )


@pytest.mark.parametrize(
    ("document", "with_root", "index", "served"),
    [
        (_described(), True, False, [_ROOT]),
        (_described(), False, False, [_FEMALE, _MALE]),
        (_described(stat_type="measuredValue"), True, True, [_ROOT]),
        (
            pinned_identity(stat_var=_MALE, entities={"recipient": [_RECIPIENT]}),
            True,
            False,
            [_MALE],
        ),
    ],
    ids=["walk_root", "walk_no_root", "code_index", "pinned"],
)
@pytest.mark.parametrize("registered", [False, True], ids=["sidecar", "registry_hit"])
def test_the_door_serves_the_members_and_family_the_mint_states(
    document: dict, with_root: bool, index: bool, served: list[str], registered: bool
):
    store = FakeDatasetStore()
    client = _members_client(
        graph=_graph(with_root=with_root),
        store=store,
        code_index=_family_index() if index else None,
    )
    minted = client.post("/api/datasets", json=_mint_body(document))
    assert minted.status_code == 200, minted.text
    if not registered:
        store.state.spec_registry.clear()
        store.state.spec_definition.clear()

    resp = _members(client, document, body={} if registered else None)

    assert resp.status_code == 200, resp.text
    body, trace = resp.json(), minted.json()["trace"]
    assert body["spec_id"] == recompute_identity(document).spec_id
    assert body["members"] == [
        {
            "ref": f"dcid:{stat_var}",
            "label": _LABELS[stat_var],
            "roles": ["recipient"],
            "constraints": {"gender": stat_var.removeprefix("Count_Person_")}
            if stat_var != _ROOT
            else {},
        }
        for stat_var in served
    ]
    assert [member["ref"] for member in body["members"]] == [
        entry["ref"] for entry in trace["resolved_stat_vars"]
    ]
    assert body["served_count"] == len(served)
    assert body["family_member_digest"] is not None
    assert (body["family_member_count"], body["family_member_digest"]) == (
        trace["family_member_count"],
        trace["family_member_digest"],
    )


def test_an_unregistered_definition_writes_no_registry_row():
    store = FakeDatasetStore()

    resp = _members(_members_client(store=store), _described())

    assert resp.status_code == 200, resp.text
    assert store.state.spec_registry == {}
    assert store.state.spec_definition == {}


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({}, 404, "spec_not_found"),
        (None, 404, "spec_not_found"),
        ({"definition": "other"}, 422, "spec_id_mismatch"),
        ({"resolved_stat_var_dcids": [_ROOT]}, 400, "request_invalid"),
    ],
    ids=["no_definition", "no_body", "other_definition", "posted_members"],
)
def test_the_door_refuses_what_it_cannot_resolve(body: dict | None, status: int, code: str):
    document = _described()
    spec_id = recompute_identity(document).spec_id
    client = _members_client()
    if body is not None and body.get("definition") == "other":
        other = described_identity(
            shape={
                "kind": "statvar_schema",
                "population_type": "Household",
                "measured_property": "count",
            },
            slots=[],
            entities={"recipient": [_RECIPIENT]},
        )
        body = {"definition": {**_definition(other), "spec_id": spec_id}}

    resp = (
        client.post(f"/api/datasets/{spec_id}/members")
        if body is None
        else client.post(f"/api/datasets/{spec_id}/members", json=body)
    )

    assert resp.status_code == status, resp.text
    assert resp.json()["code"] == code


def test_a_member_refusal_is_the_mint_s():
    """No StatVar holds the `Animal`/`count` five-tuple, so both routes refuse
    `spec_not_exportable`."""
    document = described_identity(
        shape={"kind": "statvar_schema", "population_type": "Animal", "measured_property": "count"},
        slots=[],
        entities={"recipient": [_RECIPIENT]},
    )
    client = _members_client()

    minted = client.post("/api/datasets", json=_mint_body(document))
    members = _members(client, document)

    assert minted.status_code == members.status_code == 409, members.text
    assert minted.json()["code"] == members.json()["code"] == "spec_not_exportable"


class _CompleteInconsistentReadGraphReader(_InconsistentReadGraphReader):
    """Reads every node in full (`statvar_members.CompleteArcReads`), as the origin
    catalogue's tables do, so a contradiction it returns is the graph's own."""

    def arcs_complete(self, dcid: str) -> bool:
        return True


@pytest.mark.parametrize(
    ("reader_type", "retryable"),
    [(_InconsistentReadGraphReader, True), (_CompleteInconsistentReadGraphReader, False)],
    ids=["live_reads", "complete_reads"],
)
def test_an_unsettled_member_refuses_retryable_only_where_a_later_read_can_settle_it(
    reader_type: type[_InconsistentReadGraphReader], retryable: bool
):
    """A member whose forward read contradicts the reverse index is left undismissed. On a
    live graph a later batch read can settle it, so the refusal is retryable. A reader that
    holds every arc reads the same contradiction again, so it is final."""
    graph = reader_type(_graph(with_root=False), stale_dcids=(_MALE,))

    resp = _members(_members_client(graph=graph), _described())  # ty: ignore[invalid-argument-type]

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("member_not_resolvable", retryable)
    assert problem["issues"] == [{"member": f"dcid:{_MALE}"}]


def test_a_query_key_refuses_request_invalid():
    document = _described()
    spec_id = recompute_identity(document).spec_id

    resp = _members_client().post(
        f"/api/datasets/{spec_id}/members?format=csv",
        json={"definition": _definition(document)},
    )

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert (problem["code"], problem["pointer"]) == ("request_invalid", "/options/format")


@pytest.mark.parametrize("route", ["mint", "preview", "options"])
def test_no_input_takes_a_posted_member_list(route: str):
    """`resolved_stat_var_dcids` is no field of any datasets input, so posting it refuses
    `request_invalid` before resolution."""
    document = _described()
    spec_id = recompute_identity(document).spec_id
    store = FakeDatasetStore()
    client = _members_client(store=store)
    if route == "options":
        path = f"/api/datasets/{spec_id}/options"
        body: dict = {"schema_version": 1, "definition": _definition(document)}
        body["resolved_stat_var_dcids"] = [_ROOT]
    else:
        path = "/api/datasets"
        body = _mint_body(document)
        body["input"]["resolved_stat_var_dcids"] = [_ROOT]
        if route == "preview":
            body["preview"] = {"limit": 10}

    resp = client.post(path, json=body)

    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "request_invalid"
    assert store.state.spec_registry == {}
