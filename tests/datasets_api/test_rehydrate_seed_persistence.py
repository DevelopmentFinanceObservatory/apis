"""A posted StatVar seed must survive in the stored head so a described identity resolves
past the head recheck interval, and on the operation of an over-budget first mint so the
worker resolves the family the door resolved.

The composed app uses real materialisation and a SQLite-backed version writer. The graph
hides `Count_Person` from truncated reverse walks, so GET succeeds only when rehydration
recovers the seed.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from datasets_api import kinds
from datasets_api.config import DatasetsBuildSettings
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from spec_registry.identity import recompute_identity

from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import described_identity, register_body
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore, wire_definition_for
from .test_definition_door import _client as _base_client
from .test_definition_door import _coverage_service
from .test_definition_door_currency import _FakeHeadStore
from .test_worker_e2e import api_and_worker

_RECIPIENT = "country/KEN"
_ROOT = "Count_Person"
_MALE = "Count_Person_Male"
_FEMALE = "Count_Person_Female"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_ROOT},{_RECIPIENT},USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
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

_PAST_RECHECK_INTERVAL = timedelta(seconds=DatasetsBuildSettings().head_recheck_seconds + 100)


def _member_arcs(*, gendered: tuple[str, str] | None) -> dict[str, tuple[str, ...]]:
    arcs: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "observationProperties": ("recipient",),
    }
    if gendered is not None:
        property_name, value = gendered
        arcs["constraintProperties"] = (property_name,)
        arcs[property_name] = (value,)
    return arcs


class _RootWithheldGraphReader(FakeMintGraphReader):
    """Reverse-arc pages are truncated and omit `_ROOT`.

    Arc reads retain it for seed checks.
    """

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        return type(page)(dcids=tuple(d for d in page.dcids if d != _ROOT), truncated=True)


def _graph() -> _RootWithheldGraphReader:
    return _RootWithheldGraphReader(
        arcs={
            _ROOT: _member_arcs(gendered=None),
            _MALE: _member_arcs(gendered=("gender", "Male")),
            _FEMALE: _member_arcs(gendered=("gender", "Female")),
        },
        facets={(_ROOT, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
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


def _spec_id_body(document: dict, spec_id: str, *, seed: list[str] | None) -> dict:
    input_: dict = {
        "kind": "spec_id",
        "spec_id": spec_id,
        "definition": wire_definition_for(spec_id=spec_id, identity=document),
    }
    if seed is not None:
        input_["resolved_stat_var_dcids"] = seed
    return {"schema_version": 1, "input": input_, "options": {"bundle": False}}


def test_the_withheld_root_graph_genuinely_refuses_over_ceiling_unseeded():
    """Control: this graph refuses the identity without a posted seed."""
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    client = _base_client(
        graph=_graph(),
        head_writer=_FakeHeadStore(),
        sdmx_transport=_sdmx_transport(),
        coverage=_coverage_service(stat_vars=(_ROOT,), recipient=_RECIPIENT),
    )

    resp = client.post("/api/datasets", json=_spec_id_body(document, spec_id, seed=None))

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "over_ceiling"


def test_a_seed_recovered_from_the_head_keeps_a_described_identity_resolving_past_the_recheck_interval():
    """After the head exceeds the recheck interval, GET recovers `Count_Person` as the
    rehydration seed and resolves without a request body.
    """
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    store = FakeDatasetStore()
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    client = _base_client(
        graph=_graph(),
        store=store,
        head_writer=head_store,
        version_store=version_store,
        sdmx_transport=_sdmx_transport(),
        coverage=_coverage_service(stat_vars=(_ROOT,), recipient=_RECIPIENT),
    )

    minted = client.post("/api/datasets", json=_spec_id_body(document, spec_id, seed=[_ROOT]))
    assert minted.status_code == 200, minted.text
    minted_refs = {entry["ref"] for entry in minted.json()["trace"]["resolved_stat_vars"]}
    assert minted_refs == {f"dcid:{_ROOT}"}

    head_store.age_head(spec_id, age=_PAST_RECHECK_INTERVAL)

    # `bundle=false` matches the mint options, so a stale head serves its stored rendering.
    # Seed recovery runs in `_resolve_currency` before `_serve_head`.
    resp = client.get(f"/api/datasets/{spec_id}?bundle=false")

    assert resp.status_code == 200, resp.text
    resolved_refs = {entry["ref"] for entry in resp.json()["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {f"dcid:{_ROOT}"}


def test_a_described_identity_with_no_head_yet_resolves_exactly_as_it_does_today():
    """A registered definition has no head to seed its first GET, so the withheld-root graph
    still returns `over_ceiling`.
    """
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    client = _base_client(
        graph=_graph(),
        head_writer=_FakeHeadStore(),
        version_store=FakeVersionRecordDatasetStore(),
        sdmx_transport=_sdmx_transport(),
    )
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=document)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "over_ceiling"


def _api_and_worker():
    return api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(
            base_url="https://datacommons.one.org/core/api", transport=_sdmx_transport()
        ),
        coverage=lambda: _coverage_service(stat_vars=(_ROOT,), recipient=_RECIPIENT),
    )


def test_an_over_budget_first_mint_runs_in_the_worker_with_the_posted_seed(
    monkeypatch: pytest.MonkeyPatch,
):
    """The spec has no head when the worker runs, so the operation's stored members are the
    only seed that proves the withheld-root family complete there."""
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
    api_client, operation_store, forwarded = _api_and_worker()

    submitted = api_client.post(
        "/api/datasets", json=_spec_id_body(document, spec_id, seed=[_ROOT])
    )

    assert submitted.status_code == 202, submitted.text
    operation_id = submitted.json()["operation"]["operation_id"]
    assert forwarded == [(operation_id, 0)]
    stored = operation_store.get(operation_id)
    assert stored is not None
    assert stored.request_identity == {"subject_id": spec_id, "member_dcids": [_ROOT]}
    polled = api_client.get(f"/api/datasets/operations/{operation_id}")
    assert polled.status_code == 200, polled.text
    body = polled.json()
    assert body["operation"]["status"] == "done"
    resolved_refs = {entry["ref"] for entry in body["manifest"]["trace"]["resolved_stat_vars"]}
    assert resolved_refs == {f"dcid:{_ROOT}"}


def test_an_unseeded_repost_refuses_at_the_door_and_does_not_reach_the_seeded_operation(
    monkeypatch: pytest.MonkeyPatch,
):
    """The mint door resolves with the posted seed alone, so the same spec posted without
    one refuses `candidate_window` synchronously and records no operation."""
    document = _identity()
    spec_id = recompute_identity(document).spec_id
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
    api_client, _operation_store, forwarded = _api_and_worker()
    seeded = api_client.post("/api/datasets", json=_spec_id_body(document, spec_id, seed=[_ROOT]))
    assert seeded.status_code == 202, seeded.text

    unseeded = api_client.post("/api/datasets", json=_spec_id_body(document, spec_id, seed=None))

    assert unseeded.status_code == 422, unseeded.text
    problem = unseeded.json()
    assert (problem["code"], problem["ceiling_kind"]) == ("over_ceiling", "candidate_window")
    assert "this same spec_id again with resolved_stat_var_dcids" in problem["detail"]
    assert len(forwarded) == 1
