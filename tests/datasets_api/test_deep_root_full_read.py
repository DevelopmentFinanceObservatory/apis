"""A described identity whose root sorts past the first reverse page settles by the full read.

The graph truncates the first page at `CANDIDATE_CAP` and leaves the root off it, as the dev
graph does for `Person`/`count` by gender. A read wider than that page returns the whole
filtered set. No request names a member, so the mint, a GET with no head, the worker and the
members door each settle the family from the identity alone. A set past
`DatasetsBuildSettings.member_subject_ceiling` refuses `candidate_window`, and a walk past
`DatasetsBuildSettings.member_read_seconds` answers a retryable `timeout`.
"""

from __future__ import annotations

import httpx
import pytest

from datasets_api import kinds
from datasets_api.config import DatasetsBuildSettings
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from spec_registry.identity import recompute_identity
from statvar_members import member_digest

from ._graph_doubles import FirstPageWithholdingGraphReader
from ._identity_builders import described_identity, register_body
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import wire_definition_for
from .test_definition_door import _client as _base_client
from .test_definition_door import _coverage_service
from .test_definition_door_currency import _FakeHeadStore
from .test_worker_e2e import api_and_worker

_RECIPIENT = "country/KEN"
_ROOT = "Count_Person"
_MALE = "Count_Person_Male"
_FEMALE = "Count_Person_Female"

_CSV_BODY = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    f"dataflow,DC:DF_OBS(1.0.0),I,{_ROOT},{_RECIPIENT},USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)


def _member_arcs(gender: str | None) -> dict[str, tuple[str, ...]]:
    arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "observationProperties": ("recipient",),
    }
    if gender is not None:
        arcs["constraintProperties"] = ("gender",)
        arcs["gender"] = (gender,)
    return arcs


def _graph() -> FirstPageWithholdingGraphReader:
    return FirstPageWithholdingGraphReader(
        withheld=frozenset({_ROOT}),
        arcs={
            _ROOT: _member_arcs(None),
            _MALE: _member_arcs("Male"),
            _FEMALE: _member_arcs("Female"),
        },
        facets={(_ROOT, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya", _ROOT: "Population"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _identity() -> dict:
    return described_identity(
        shape={"kind": "statvar_schema", "population_type": "Person", "measured_property": "count"},
        slots=[{"property": "gender", "binding": {"state": "unbound"}}],
        entities={"recipient": [_RECIPIENT]},
    )


def _mint_body(document: dict) -> dict:
    spec_id = recompute_identity(document).spec_id
    return {
        "schema_version": 1,
        "input": {
            "kind": "spec_id",
            "spec_id": spec_id,
            "definition": wire_definition_for(spec_id=spec_id, identity=document),
        },
        "options": {"bundle": False},
    }


def _client(*, settings: DatasetsBuildSettings | None = None, **kwargs):
    return _base_client(
        graph=_graph(),
        head_writer=_FakeHeadStore(),
        sdmx_transport=_sdmx_transport(),
        coverage=_coverage_service(stat_vars=(_ROOT,), recipient=_RECIPIENT),
        settings=settings,
        **kwargs,
    )


_FAMILY_DIGEST = member_digest(frozenset({_ROOT, _MALE, _FEMALE}))


def test_a_mint_settles_a_root_past_the_first_page_and_states_its_family():
    resp = _client().post("/api/datasets", json=_mint_body(_identity()))

    assert resp.status_code == 200, resp.text
    trace = resp.json()["trace"]
    assert [entry["ref"] for entry in trace["resolved_stat_vars"]] == [f"dcid:{_ROOT}"]
    assert (trace["family_member_count"], trace["family_member_digest"]) == (3, _FAMILY_DIGEST)


def test_a_get_with_no_head_settles_the_registered_definition():
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    client = _client(version_store=FakeVersionRecordDatasetStore())
    registered = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=document)),
    )
    assert registered.status_code == 200, registered.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    refs = [entry["ref"] for entry in resp.json()["trace"]["resolved_stat_vars"]]
    assert refs == [f"dcid:{_ROOT}"]


def test_the_members_door_serves_what_the_mint_serves():
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    client = _client()

    members = client.post(
        f"/api/datasets/{spec_id}/members",
        json={
            "schema_version": 1,
            "definition": wire_definition_for(spec_id=spec_id, identity=document),
        },
    )
    minted = client.post("/api/datasets", json=_mint_body(document))

    assert members.status_code == 200, members.text
    assert minted.status_code == 200, minted.text
    body, trace = members.json(), minted.json()["trace"]
    assert [member["ref"] for member in body["members"]] == [
        entry["ref"] for entry in trace["resolved_stat_vars"]
    ]
    assert body["members"][0]["label"] == "Population"
    assert body["served_count"] == len(trace["resolved_stat_vars"])
    assert (body["family_member_count"], body["family_member_digest"]) == (
        trace["family_member_count"],
        trace["family_member_digest"],
    )


def test_an_over_budget_first_mint_runs_in_the_worker_with_the_members_the_door_resolved(
    monkeypatch: pytest.MonkeyPatch,
):
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
    api_client, operation_store, forwarded = api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(
            base_url="https://datacommons.one.org/core/api", transport=_sdmx_transport()
        ),
        coverage=lambda: _coverage_service(stat_vars=(_ROOT,), recipient=_RECIPIENT),
    )

    submitted = api_client.post("/api/datasets", json=_mint_body(document))

    assert submitted.status_code == 202, submitted.text
    operation_id = submitted.json()["operation"]["operation_id"]
    assert forwarded == [(operation_id, 0)]
    stored = operation_store.get(operation_id)
    assert stored is not None
    assert stored.request_identity == {"subject_id": spec_id, "member_dcids": [_ROOT]}
    polled = api_client.get(f"/api/datasets/operations/{operation_id}")
    assert polled.status_code == 200, polled.text
    assert polled.json()["operation"]["status"] == "done"


def _post(client, document: dict, *, route: str) -> httpx.Response:
    if route == "mint":
        return client.post("/api/datasets", json=_mint_body(document))
    spec_id = recompute_identity(document).spec_id
    return client.post(
        f"/api/datasets/{spec_id}/members",
        json={"definition": wire_definition_for(spec_id=spec_id, identity=document)},
    )


@pytest.mark.parametrize("route", ["mint", "members"])
def test_a_set_past_the_ceiling_refuses_candidate_window(route: str):
    """A whole set larger than the ceiling refuses with the ceiling as its cap."""
    client = _client(settings=DatasetsBuildSettings(member_subject_ceiling=2))

    resp = _post(client, _identity(), route=route)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert (problem["code"], problem["ceiling_kind"]) == ("over_ceiling", "candidate_window")
    assert (problem["count"], problem["cap"]) == (2, 2)
    assert "resolved_stat_var_dcids" not in problem["detail"]


@pytest.mark.parametrize("route", ["mint", "members"])
def test_a_walk_past_its_deadline_answers_a_retryable_timeout(route: str):
    """Running out of time says nothing about the set's size, so the door answers 504
    `timeout`, retryable, and states no ceiling."""
    client = _client(settings=DatasetsBuildSettings(member_read_seconds=1e-9))

    resp = _post(client, _identity(), route=route)

    assert resp.status_code == 504, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("timeout", True)
    assert "ceiling_kind" not in problem
