"""Definition-door concurrency tests cover admission, timeout permits, deadlines, shutdown,
and broken-pool handling on `GET /api/datasets/{id}`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator

import httpx
import pytest

from datasets_api import door_executor
from datasets_api.config import DatasetsBuildSettings
from datasets_api.operation_store import NullOperationStore
from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._executor_doubles import BrokenSubmitExecutor
from ._graph_doubles import BlockingFakeMintGraphReader, FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"

# Mirrors `test_materialise_semaphore.py`'s own `_SETTINGS_ENV`: the four
# `DATASETS_*` settings `datasets_surface()`'s own composed `lifespan` reads
# unconditionally through `DatasetsSettings.from_env()`, needed by every
# test below that enters the client (`with client:`) to keep its concurrent
# thread's own request on the same shared portal loop.
_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_ROW = (
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)

_SCOPE = {
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}

_MINT_BODY = {
    "schema_version": 1,
    "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
    "options": {"bundle": False},
}


def _sdmx_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_CSV_HEADER + _CSV_ROW,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _default_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        # Role-bearing candidates come from coverage, not these facets.
        facets={(_STAT_VAR, _RECIPIENT): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )


def _coverage_service() -> FakeCoverageService:
    """A one-facet `CoverageIndex` for `_STAT_VAR`, shaped to reproduce
    `_FACET`: same facet_id/provenance/measurement_method/unit, and
    `n_obs`/`last_year` from the one 2015 row `_CSV_ROW` carries.
    """
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", "country/USA", _RECIPIENT, [2015])],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


class _SlowStreamingTransport(httpx.BaseTransport):
    """Streams a row every 0.01 s so the worker deadline expires during a live SDMX read.

    The deadline guard compares `time.monotonic()` instants, so real time must pass.
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            content=self._chunks(),
        )

    def _chunks(self) -> Iterator[bytes]:
        yield _CSV_HEADER.encode()
        while True:
            time.sleep(0.01)
            yield _CSV_ROW.encode()


def _client(
    *,
    graph,
    store=None,
    definition_pool_size: int = 1,
    sdmx: SdmxClient | None = None,
    settings: DatasetsBuildSettings | None = None,
):
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        settings=settings,
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx or SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        # The composed lifespan starts snapshot and pin doors too.
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        definition_pool_size=definition_pool_size,
        # The composed lifespan also requires the meta door's operation store.
        operation_store=NullOperationStore(),
        coverage=_coverage_service(),
    )


def _mint_one_spec(client) -> str:
    resp = client.post("/api/datasets", json=_MINT_BODY)
    assert resp.status_code == 200, resp.text
    return resp.json()["definition"]["spec_id"]


def test_over_admission_at_pool_size_one_refuses_definition_backpressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full one-worker pool refuses the second retrieval with backpressure."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    graph = BlockingFakeMintGraphReader(_default_graph())
    graph.release.set()  # let the mint below (and its own node_arcs calls) through unblocked
    client = _client(graph=graph, definition_pool_size=1)

    with client:
        spec_id = _mint_one_spec(client)
        graph.entered.clear()
        graph.release.clear()  # Block the retrieval's graph read.

        first_response: list = []
        first_thread = threading.Thread(
            target=lambda: first_response.append(client.get(f"/api/datasets/{spec_id}"))
        )
        first_thread.start()
        assert graph.entered.wait(timeout=5.0), (
            "first request never reached the blocking graph read"
        )

        second_response = client.get(f"/api/datasets/{spec_id}")

        assert second_response.status_code == 503, second_response.text
        second_problem = second_response.json()
        assert second_problem["code"] == "service_unavailable"
        assert second_problem["retryable"] is True

        graph.release.set()
        first_thread.join(timeout=5.0)
        assert first_response, "first request never returned"
        assert first_response[0].status_code == 200, first_response[0].text


def test_timed_out_call_keeps_its_permit_held_until_it_actually_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout does not release the permit while the worker runs. The next request gets
    backpressure until that worker finishes."""
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    graph = BlockingFakeMintGraphReader(_default_graph())
    graph.release.set()
    client = _client(
        graph=graph,
        definition_pool_size=1,
        settings=DatasetsBuildSettings(definition_timeout_seconds=0.05, metadata_read_seconds=0.01),
    )

    # The client context keeps both requests on one portal loop.
    with client:
        spec_id = _mint_one_spec(client)
        graph.entered.clear()
        graph.release.clear()

        first_response: list = []
        first_thread = threading.Thread(
            target=lambda: first_response.append(client.get(f"/api/datasets/{spec_id}"))
        )
        first_thread.start()
        assert graph.entered.wait(timeout=5.0), (
            "first request never reached the blocking graph read"
        )

        second_response = client.get(f"/api/datasets/{spec_id}")

        first_thread.join(timeout=5.0)
        assert first_response, "first request never returned"
        assert first_response[0].status_code == 504, first_response[0].text
        first_problem = first_response[0].json()
        assert first_problem["code"] == "timeout"
        assert first_problem["retryable"] is True

        assert second_response.status_code == 503, second_response.text
        second_problem = second_response.json()
        assert second_problem["code"] == "service_unavailable"
        assert second_problem["retryable"] is True

        graph.release.set()


def test_permit_frees_once_the_background_call_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same timed-out-then-still-running setup, but this time the blocked
    call is released and allowed to finish: a request arriving afterward
    succeeds, proving the permit frees on the executor call's own
    completion.
    """
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    graph = BlockingFakeMintGraphReader(_default_graph())
    graph.release.set()
    # The door reads this value per request. Restore the normal timeout for the unblocked poll.
    settings = DatasetsBuildSettings(definition_timeout_seconds=0.05)
    client = _client(graph=graph, definition_pool_size=1, settings=settings)

    # The client context keeps both requests on one portal loop.
    with client:
        spec_id = _mint_one_spec(client)
        graph.entered.clear()
        graph.release.clear()

        first_thread = threading.Thread(target=lambda: client.get(f"/api/datasets/{spec_id}"))
        first_thread.start()
        assert graph.entered.wait(timeout=5.0), (
            "first request never reached the blocking graph read"
        )

        timed_out = client.get(f"/api/datasets/{spec_id}")
        assert timed_out.status_code == 503, timed_out.text

        # Restore the normal timeout before polling an unblocked request.
        settings.definition_timeout_seconds = 30.0

        graph.release.set()
        first_thread.join(timeout=5.0)

        # `join` waits for the response, not the worker that holds the permit.
        deadline = time.monotonic() + 5.0
        while True:
            resp = client.get(f"/api/datasets/{spec_id}")
            if resp.status_code == 200 or time.monotonic() > deadline:
                break
            assert resp.status_code == 503, resp.text

        assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("suffix", ["", "/options"])
def test_sdmx_deadline_finishes_definition_worker_before_outer_timeout(suffix: str) -> None:
    store = FakeDatasetStore()
    graph = _default_graph()
    spec_id = _mint_one_spec(_client(graph=graph, store=store))
    client = _client(
        graph=graph,
        store=store,
        definition_pool_size=1,
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            timeout=10,
            transport=_SlowStreamingTransport(),
        ),
        settings=DatasetsBuildSettings(definition_timeout_seconds=0.2),
    )

    started = time.monotonic()
    first = client.get(f"/api/datasets/{spec_id}{suffix}")
    first_elapsed = time.monotonic() - started

    assert first.status_code == 504, first.text
    assert first.json()["code"] == "timeout"
    assert first_elapsed < 0.5

    # The SDMX deadline releases the sole permit before the next request.
    second = client.get(f"/api/datasets/{spec_id}{suffix}")
    assert second.status_code == 504, second.text


def test_request_after_lifespan_shutdown_gets_backpressure_not_a_bare_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once this door's own lifespan has exited, `executor.submit` raises a
    bare post-shutdown `RuntimeError` that no classifier arm recognizes on
    its own. Without `_run_retrieval`'s own translation, that reaches the
    platform's catch-all as a non-retryable 500. During a graceful drain
    the honest word is the same retryable backpressure a full pool gives.
    """
    monkeypatch.setenv("DATASETS_GRAPH_BASE_URL", "https://graph.example.org")
    monkeypatch.setenv("DATASETS_SDMX_BASE_URL", "https://sdmx.example.org")
    monkeypatch.setenv("DATASETS_ARTIFACT_BUCKET_WORKING", "working-bucket")
    monkeypatch.setenv("DATASETS_ARTIFACT_BUCKET_PINNED", "pinned-bucket")
    client = _client(graph=_default_graph(), definition_pool_size=1)

    with client:
        pass  # enters and exits this door's own lifespan, with nothing in flight

    resp = client.get("/api/datasets/spec_never_minted")

    assert resp.status_code == 503, resp.text
    problem = resp.json()
    assert problem["code"] == "service_unavailable"
    assert problem["retryable"] is True


def test_broken_thread_pool_surfaces_as_internal_error_not_backpressure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A permanently broken pool maps to the platform's non-retryable 500 and logs the failure."""
    store = FakeDatasetStore()
    graph = _default_graph()
    spec_id = _mint_one_spec(_client(graph=graph, store=store, definition_pool_size=1))

    monkeypatch.setattr(door_executor, "ThreadPoolExecutor", BrokenSubmitExecutor)
    client = _client(graph=graph, store=store, definition_pool_size=1)

    with caplog.at_level(logging.ERROR):
        resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 500, resp.text
    problem = resp.json()
    assert problem["code"] == "internal_error"
    assert problem["retryable"] is False
    assert "permanently broken" in caplog.text
