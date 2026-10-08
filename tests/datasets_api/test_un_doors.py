"""Mints of a UN System Data Commons variable through the datasets doors, over the
UN deployment double. The mint doors run the facet probe before the prewrite opens its UN record,
so these tests post to the doors. A request that binds no entity carries the
`un_system_unavailable` warning a request naming the same places carries. A request with no
candidate is the empty result at every mint door, with that warning, and refuses
`upstream_error` when the only places it names are ones the UN refused.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from datasets_api import kinds
from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.shape import resolve_shape
from dc_sdmx import SdmxClient
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.materialise.test_observation_rows import _sdmx_never_called
from tests.materialise.un_deployment import (
    OURS_API,
    UN_API,
    UN_SECOND_VARIABLE,
    UN_VARIABLE,
    UnDeployments,
)

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_mint_unbound_entities import _coverage
from .test_pin_door import FakeArtifactReferenceWriter
from .test_worker_e2e import api_and_worker

_GRC, _KEN, _GHA, _NGA = "country/GRC", "country/KEN", "country/GHA", "country/NGA"
_VARIABLES = (UN_VARIABLE, UN_SECOND_VARIABLE)
_STAT_VAR_ARCS = {"typeOf": ["StatisticalVariable"], "measuredProperty": ["value"]}
# Kenya holds five observations and Ghana two.
_ROWS = 7
_UNREAD_GREECE = f"No UN data was read for {_GRC}."


@pytest.fixture
def deployments() -> UnDeployments:
    """The UN deployment with its front door refusing Greece, whose availability lists Greece
    beside Kenya and Ghana."""
    return UnDeployments(
        refused={_GRC},
        arcs=dict.fromkeys(_VARIABLES, _STAT_VAR_ARCS),
        available=[_GRC, _KEN, _GHA],
    )


@pytest.fixture
def graph(deployments: UnDeployments) -> Iterator[OriginGraphClient]:
    client = build_graph_client(
        base_url=OURS_API,
        token=None,
        un_base_url=UN_API,
        un_availability=SdmxClient(
            base_url=UN_API, transport=httpx.MockTransport(deployments.handler)
        ),
        transport=httpx.MockTransport(deployments.handler),
        max_attempts=1,
    )
    yield client
    client.close()


def _client(graph: OriginGraphClient) -> TestClient:
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=_sdmx_never_called(),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage(),
    )


def _sidecar(graph: OriginGraphClient, stat_var: str, places: tuple[str, ...]) -> dict:
    """The canonical definition of `stat_var` at `places`, or binding no entity without any."""
    shape = resolve_shape(f"dcid:{stat_var}", graph=graph)
    scope = Scope(entities={"observationAbout": list(places)}) if places else Scope()
    document = build_identity_document(shape, scope=scope)
    return {
        "schema_version": 1,
        "spec_id": recompute_identity(document).spec_id,
        "identity": document,
    }


def _spec_body(graph: OriginGraphClient, places: tuple[str, ...] = ()) -> dict:
    sidecar = _sidecar(graph, UN_VARIABLE, places)
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": sidecar["spec_id"], "definition": sidecar},
        "options": {"bundle": False},
    }


def _through_the_worker(
    body: dict, graph: OriginGraphClient, monkeypatch: pytest.MonkeyPatch
) -> httpx.Response:
    """Submit `body` over budget and return the poll of the operation the worker ran."""
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
    api_client, _operation_store, forwarded = api_and_worker(
        graph=graph, sdmx=_sdmx_never_called, coverage=_coverage
    )
    submitted = api_client.post("/api/datasets", json=body)
    assert submitted.status_code == 202, submitted.text
    operation_id = submitted.json()["operation"]["operation_id"]
    assert forwarded == [(operation_id, 0)]
    return api_client.get(f"/api/datasets/operations/{operation_id}")


def _un_warnings(holder: dict) -> list[str]:
    return [w["message"] for w in holder["warnings"] if w["code"] == "un_system_unavailable"]


_NAMED = (_GRC, _KEN, _GHA)
_BINDINGS = pytest.mark.parametrize("places", [(), _NAMED], ids=["no_entity", "named"])


@_BINDINGS
def test_the_spec_door_warns_of_the_place_the_un_refused(graph, places):
    resp = _client(graph).post("/api/datasets", json=_spec_body(graph, places))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["coverage"]["rows"] == _ROWS
    (warning,) = _un_warnings(manifest)
    assert warning.endswith(_UNREAD_GREECE)


@_BINDINGS
@pytest.mark.parametrize("body", [_spec_body])
def test_the_worker_warns_of_the_place_the_un_refused(graph, monkeypatch, body, places):
    polled = _through_the_worker(body(graph, places), graph, monkeypatch)

    assert polled.status_code == 200, polled.text
    assert polled.json()["operation"]["status"] == "done", polled.text
    (warning,) = _un_warnings(polled.json()["manifest"])
    assert warning.endswith(_UNREAD_GREECE)


def test_a_mint_binding_no_entity_reads_the_un_availability_once(graph, deployments):
    # The facet probe reads it. The `withheld_by_registry` check, which runs because Greece came
    # back without rows, takes the places from the result.
    resp = _client(graph).post("/api/datasets", json=_spec_body(graph))

    assert resp.status_code == 200, resp.text
    assert len(deployments.availability_requests) == 1


_DOORS = pytest.mark.parametrize("body", [_spec_body])


def _warning_codes(manifest: dict) -> set[str]:
    return {warning["code"] for warning in manifest["warnings"]}


@_DOORS
def test_a_request_binding_no_entity_with_no_un_data_is_the_empty_result_at_every_door(
    graph, deployments, body
):
    # The UN's availability lists Nigeria, and its facet read there holds nothing.
    deployments.available = [_NGA]

    resp = _client(graph).post("/api/datasets", json=body(graph))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["coverage"]["rows"] == 0
    assert "empty_result" in _warning_codes(manifest)
    assert _un_warnings(manifest) == []


@_DOORS
def test_an_empty_result_binding_no_entity_warns_of_the_place_the_un_refused(
    graph, deployments, body
):
    # Greece is refused and Nigeria holds nothing. The UN front door refuses Greece on every
    # read, so a retry would meet the same refusal. The request is the empty result and its
    # warning names Greece.
    deployments.available = [_GRC, _NGA]

    resp = _client(graph).post("/api/datasets", json=body(graph))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["coverage"]["rows"] == 0
    assert "empty_result" in _warning_codes(manifest)
    (warning,) = _un_warnings(manifest)
    assert warning.endswith(_UNREAD_GREECE)


@pytest.mark.parametrize("body", [_spec_body])
def test_the_worker_serves_a_request_binding_no_entity_with_no_un_data_as_the_empty_result(
    graph, deployments, monkeypatch, body
):
    deployments.available = [_GRC, _NGA]

    polled = _through_the_worker(body(graph), graph, monkeypatch)

    assert polled.status_code == 200, polled.text
    assert polled.json()["operation"]["status"] == "done", polled.text
    manifest = polled.json()["manifest"]
    assert manifest["coverage"]["rows"] == 0
    assert "empty_result" in _warning_codes(manifest)
    (warning,) = _un_warnings(manifest)
    assert warning.endswith(_UNREAD_GREECE)


@_DOORS
def test_a_request_naming_only_a_refused_place_refuses_upstream_error_at_every_door(graph, body):
    # The empty probe says nothing about Greece, which the UN refused to read.
    resp = _client(graph).post("/api/datasets", json=body(graph, (_GRC,)))

    assert resp.status_code == 502, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("upstream_error", True)
    assert _GRC in problem["detail"]
