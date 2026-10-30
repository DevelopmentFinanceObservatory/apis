"""Tests graph and SDMX client construction, facet chunking, and semaphore limits.

No network access is required."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

import dc_graph
import search.engine.clients.materialise_bridge as materialise_bridge
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import GraphReadError
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.materialise_bridge import (
    build_materialise_sdmx_client,
    observation_facets_via_materialise,
)
from search.engine.errors import GraphInfraError
from tests.materialise.conftest import FakeGraphReader


class TestLiveGraphClientFromSearchSettings:
    def test_no_args_reads_config_module_attributes(self, search_settings):
        search_settings(graph_base="https://cfg.example.org")
        search_settings(graph_token="cfg-token")
        search_settings(graph_timeout_s=12.5)

        client = LiveGraphClient.from_search_settings()
        try:
            wire_client = client._client._ours._client
            assert str(wire_client.base_url).rstrip("/") == "https://cfg.example.org/core/api"
            assert wire_client.headers["x-graph-token"] == "cfg-token"
            assert wire_client.timeout.read == 12.5
        finally:
            client.close()

    def test_no_token_omits_header(self, search_settings):
        search_settings(graph_base="http://localhost:9999", graph_token="")

        client = LiveGraphClient.from_search_settings()
        try:
            assert "x-graph-token" not in client._client._ours._client.headers
        finally:
            client.close()

    def test_explicit_settings_object_overrides_module_config(self, search_settings):
        search_settings(graph_base="https://cfg.example.org")
        search_settings(graph_timeout_s=30.0)
        explicit = materialise_bridge.config.settings().model_copy(
            update={"graph_base": "https://override.example.org", "graph_timeout_s": 5.0}
        )

        client = LiveGraphClient.from_search_settings(explicit)
        try:
            wire_client = client._client._ours._client
            assert str(wire_client.base_url).rstrip("/") == "https://override.example.org/core/api"
            assert wire_client.timeout.read == 5.0
        finally:
            client.close()

    def test_max_attempts_defaults_to_one_no_retry(self, search_settings):
        # Search sets one attempt; `dc_graph.GraphClient` defaults to three.
        search_settings(graph_base="http://localhost:9999")

        client = LiveGraphClient.from_search_settings()
        try:
            assert client._client._ours._max_attempts == 1
        finally:
            client.close()

    def test_a_transport_failure_raises_an_error_in_both_families(self, search_settings):
        """`dc_graph.GraphClient`'s own read failure, substituted through
        `read_error_cls`/`deadline_error_cls` (see `graph_client.py`'s own
        module docstring), must be an instance of Search's own
        `GraphInfraError` (every search fail-open `except GraphInfraError`
        site) AND `dc_graph.GraphReadError` (every `materialise`-driven
        `except GraphReadError` site sharing this same client), never one
        family at the other's expense."""
        search_settings(graph_base="http://localhost:9999")

        client = LiveGraphClient.from_search_settings()
        try:

            def fake_post(self, path, json=None, **kwargs):
                raise httpx.ConnectError("simulated connection failure")

            with (
                pytest.MonkeyPatch.context() as mp,
                pytest.raises(GraphInfraError) as exc_info,
            ):
                mp.setattr(httpx.Client, "post", fake_post)
                client.node_label("country/KEN")
        finally:
            client.close()

        exc = exc_info.value
        assert isinstance(exc, GraphInfraError)
        assert isinstance(exc, dc_graph.GraphReadError)


class TestBuildMaterialiseSdmxClient:
    def test_no_args_reads_config_module_attributes(self, search_settings):
        search_settings(graph_base="https://cfg.example.org")
        search_settings(graph_token="cfg-token")
        search_settings(graph_timeout_s=12.5)

        client = build_materialise_sdmx_client()
        try:
            assert str(client._client.base_url).rstrip("/") == "https://cfg.example.org/core/api"
            assert client._client.headers["x-graph-token"] == "cfg-token"
            assert client._client.timeout.read == 12.5
        finally:
            client.close()

    def test_base_url_carries_core_api_suffix_not_v2(self, search_settings):
        search_settings(graph_base="https://cfg.example.org")

        client = build_materialise_sdmx_client()
        try:
            base_url = str(client._client.base_url).rstrip("/")
            assert base_url == "https://cfg.example.org/core/api"
            assert "/core/api/v2" not in base_url
        finally:
            client.close()

    def test_base_and_timeout_override_config_defaults(self, search_settings):
        search_settings(graph_base="https://cfg.example.org")
        search_settings(graph_timeout_s=30.0)

        client = build_materialise_sdmx_client(base="https://override.example.org", timeout=5.0)
        try:
            assert (
                str(client._client.base_url).rstrip("/") == "https://override.example.org/core/api"
            )
            assert client._client.timeout.read == 5.0
        finally:
            client.close()

    def test_shares_the_process_wide_total_semaphore_with_config(self):
        # Both builders must share this semaphore instance, not just its capacity.
        client = build_materialise_sdmx_client()
        try:
            assert client._total_semaphore is materialise_bridge.SDMX_TOTAL_SEMAPHORE
            assert (
                client._chunk_concurrency
                == materialise_bridge.config.settings().sdmx_chunk_concurrency
            )
        finally:
            client.close()


def test_live_sdmx_client_and_materialise_client_share_one_semaphore():
    # The two clients must share one semaphore to enforce the process-wide GET cap.
    from search.engine.clients.sdmx import SDMX_TOTAL_SEMAPHORE, LiveSdmxClient

    live = LiveSdmxClient()
    materialise_client = build_materialise_sdmx_client()
    try:
        assert live._client._total_semaphore is materialise_client._total_semaphore
        assert live._client._total_semaphore is SDMX_TOTAL_SEMAPHORE
    finally:
        live.close()
        materialise_client.close()


class TestObservationFacetsViaMaterialise:
    def test_one_call_covers_a_grid_under_the_pair_budget(self):
        """A grid within the pair budget uses one `observation_facets_batch_many` POST."""
        graph = FakeGraphReader(
            facets={
                ("sv/A", "country/USA"): [
                    MaterialiseFacet(
                        facet_id="f1",
                        provenance_id="dc/base/Src",
                        measurement_method="Meth",
                        unit="Unit",
                        obs_count=3,
                        earliest_date="2020",
                        latest_date="2022",
                        scaling_factor="1",
                    )
                ],
                ("sv/B", "country/USA"): [],
            }
        )
        result = observation_facets_via_materialise(
            stat_vars=["sv/A", "sv/B"], entities=["country/USA"], materialise_graph=graph
        )
        assert graph.calls["observation_facets_batch_many"] == 1
        assert graph.calls["variable_facets_batch"] == 0
        assert result == {
            "sv/A": {
                "country/USA": [
                    MaterialiseFacet(
                        facet_id="f1",
                        provenance_id="dc/base/Src",
                        measurement_method="Meth",
                        unit="Unit",
                        obs_count=3,
                        earliest_date="2020",
                        latest_date="2022",
                        scaling_factor="1",
                    )
                ]
            },
            "sv/B": {"country/USA": []},
        }

    def test_grid_larger_than_one_chunk_issues_multiple_posts_matching_pair_arithmetic(
        self, monkeypatch
    ):
        """POST count follows `max(1, budget // entities)` chunking exactly."""
        monkeypatch.setattr(materialise_bridge, "_FACET_PROBE_MAX_PAIRS_PER_POST", 10)
        stat_vars = [f"sv/{i}" for i in range(7)]
        entities = [f"country/{i}" for i in range(5)]
        # chunk = max(1, 10 // 5) = 2 StatVars/POST; 7 StatVars -> ceil(7/2) = 4 POSTs.
        graph = FakeGraphReader(facets={})

        result = observation_facets_via_materialise(
            stat_vars=stat_vars, entities=entities, materialise_graph=graph
        )

        assert graph.calls["observation_facets_batch_many"] == 4
        assert set(result) == set(stat_vars)
        for sv in stat_vars:
            assert set(result[sv]) == set(entities)

    def test_entity_count_drives_chunk_size_not_stat_var_count(self, monkeypatch):
        """Entity count determines chunk size for a fixed StatVar count."""
        monkeypatch.setattr(materialise_bridge, "_FACET_PROBE_MAX_PAIRS_PER_POST", 12)
        stat_vars = [f"sv/{i}" for i in range(6)]

        few_entities_graph = FakeGraphReader(facets={})
        observation_facets_via_materialise(
            stat_vars=stat_vars, entities=["e1", "e2"], materialise_graph=few_entities_graph
        )
        # chunk = max(1, 12 // 2) = 6 -> all 6 StatVars fit in one POST.
        assert few_entities_graph.calls["observation_facets_batch_many"] == 1

        many_entities_graph = FakeGraphReader(facets={})
        observation_facets_via_materialise(
            stat_vars=stat_vars,
            entities=["e1", "e2", "e3", "e4"],
            materialise_graph=many_entities_graph,
        )
        # chunk = max(1, 12 // 4) = 3 -> 6 StatVars / 3 per chunk = 2 POSTs.
        assert many_entities_graph.calls["observation_facets_batch_many"] == 2

    def test_chunked_merge_matches_the_per_stat_var_assembly(self, monkeypatch):
        """Chunked output matches independent per-StatVar facet reads."""
        monkeypatch.setattr(materialise_bridge, "_FACET_PROBE_MAX_PAIRS_PER_POST", 4)
        facets = {
            ("sv/A", "country/USA"): [
                MaterialiseFacet(
                    facet_id="f1",
                    provenance_id="dc/base/Src",
                    measurement_method="Meth",
                    unit="Unit",
                    obs_count=3,
                    earliest_date="2020",
                    latest_date="2022",
                    scaling_factor="1",
                )
            ],
            ("sv/A", "country/FRA"): [],
            ("sv/B", "country/USA"): [
                MaterialiseFacet(
                    facet_id="f2",
                    provenance_id="dc/base/Src2",
                    measurement_method=None,
                    unit=None,
                    obs_count=1,
                    earliest_date="2021",
                    latest_date="2021",
                )
            ],
            ("sv/B", "country/FRA"): [],
            ("sv/C", "country/USA"): [],
            ("sv/C", "country/FRA"): [],
        }
        graph = FakeGraphReader(facets=facets)
        stat_vars = ["sv/A", "sv/B", "sv/C"]
        entities = ["country/USA", "country/FRA"]
        # chunk = max(1, 4 // 2) = 2 -> 3 StatVars split across 2 POSTs, so
        # this genuinely exercises the merge across chunk boundaries.
        chunked = observation_facets_via_materialise(
            stat_vars=stat_vars, entities=entities, materialise_graph=graph
        )
        assert graph.calls["observation_facets_batch_many"] == 2

        expected = {
            sv: graph.variable_facets_batch(stat_var=sv, entities=entities) for sv in stat_vars
        }
        assert chunked == expected

    def test_probed_facet_carries_no_per_observation_dates_or_values(self):
        """Probed facets carry summary metadata and empty observation data."""
        graph = FakeGraphReader(
            facets={
                ("sv/A", "country/USA"): [
                    MaterialiseFacet(
                        facet_id="f1",
                        provenance_id="dc/base/Src",
                        measurement_method=None,
                        unit=None,
                        obs_count=3,
                        earliest_date="2020",
                        latest_date="2022",
                    )
                ]
            }
        )
        result = observation_facets_via_materialise(
            stat_vars=["sv/A"], entities=["country/USA"], materialise_graph=graph
        )
        facet = result["sv/A"]["country/USA"][0]
        assert facet.dates == ()
        assert facet.values == ()
        assert facet.obs_count == 3
        assert facet.provenance_id == "dc/base/Src"

    def test_graph_read_error_translates_to_graph_infra_error(self):
        """Translate `GraphReadError` to `GraphInfraError` and preserve `upstream_status`."""

        class _RaisingGraphReader(FakeGraphReader):
            def observation_facets_batch_many(
                self,
                *,
                stat_vars,
                entities,
                needs_dates: bool = False,
                deadline: float | None = None,
            ):
                raise GraphReadError("upstream 502", upstream_status=502)

        with pytest.raises(GraphInfraError) as exc_info:
            observation_facets_via_materialise(
                stat_vars=["sv/A"],
                entities=["country/USA"],
                materialise_graph=_RaisingGraphReader(),
            )
        assert exc_info.value.upstream_status == 502


class TestFacetProbeSemaphore:
    """Pins the process-wide POST limit with rendezvous tests instead of elapsed-time guesses.

    The module attribute is patched because `BoundedSemaphore` has no public resize."""

    def test_concurrent_chunk_posts_bounded_by_semaphore_capacity(
        self, monkeypatch, search_settings
    ):
        """With pool concurrency 8 and semaphore capacity 2, six chunks reach the limit.
        No more than two POSTs run concurrently."""
        cap = 2
        search_settings(facet_probe_total_concurrency=8)
        monkeypatch.setattr(
            materialise_bridge, "_FACET_PROBE_SEMAPHORE", threading.BoundedSemaphore(cap)
        )
        monkeypatch.setattr(materialise_bridge, "_FACET_PROBE_MAX_PAIRS_PER_POST", 1)

        lock = threading.Lock()
        in_flight = 0
        max_in_flight = 0
        never_set = threading.Event()

        class _BlockingGraphReader(FakeGraphReader):
            def observation_facets_batch_many(
                self, *, stat_vars, entities, needs_dates: bool = False, deadline=None
            ):
                nonlocal in_flight, max_in_flight
                with lock:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                never_set.wait(timeout=0.3)
                with lock:
                    in_flight -= 1
                return super().observation_facets_batch_many(
                    stat_vars=stat_vars, entities=entities, deadline=deadline
                )

        stat_vars = [f"sv/{i}" for i in range(6)]
        graph = _BlockingGraphReader(facets={})
        # chunk_size = max(1, 1 // 1) = 1 -> 6 StatVars split into 6 chunks.
        result = observation_facets_via_materialise(
            stat_vars=stat_vars, entities=["country/USA"], materialise_graph=graph
        )

        assert graph.calls["observation_facets_batch_many"] == 6
        assert set(result) == set(stat_vars)
        assert max_in_flight == cap

    def test_exhausted_semaphore_raises_graph_infra_error_past_timeout(
        self, monkeypatch, search_settings
    ):
        """With the sole permit held, a 0.05-second wait raises `GraphInfraError` before the POST."""
        search_settings(graph_timeout_s=0.05)
        held = threading.BoundedSemaphore(1)
        held.acquire()  # the process-wide permit is already held elsewhere
        monkeypatch.setattr(materialise_bridge, "_FACET_PROBE_SEMAPHORE", held)

        class _UnreachedGraphReader(FakeGraphReader):
            def observation_facets_batch_many(
                self, *, stat_vars, entities, needs_dates: bool = False, deadline=None
            ):
                raise AssertionError(
                    "observation_facets_batch_many must not run: the permit wait times out first"
                )

        with pytest.raises(GraphInfraError) as exc_info:
            observation_facets_via_materialise(
                stat_vars=["sv/A"],
                entities=["country/USA"],
                materialise_graph=_UnreachedGraphReader(),
            )
        assert exc_info.value.upstream_status is None

    def test_permit_wait_bounds_on_the_caller_deadline_not_the_configured_timeout(
        self, monkeypatch, search_settings
    ):
        """A deadline 0.2 seconds away overrides the 30-second configured timeout."""
        search_settings(graph_timeout_s=30.0)
        held = threading.BoundedSemaphore(1)
        held.acquire()  # the process-wide permit is already held elsewhere
        monkeypatch.setattr(materialise_bridge, "_FACET_PROBE_SEMAPHORE", held)

        class _UnreachedGraphReader(FakeGraphReader):
            def observation_facets_batch_many(
                self, *, stat_vars, entities, needs_dates: bool = False, deadline=None
            ):
                raise AssertionError(
                    "observation_facets_batch_many must not run: the permit wait times out first"
                )

        deadline = time.monotonic() + 0.2
        start = time.monotonic()
        with pytest.raises(GraphInfraError):
            observation_facets_via_materialise(
                stat_vars=["sv/A"],
                entities=["country/USA"],
                materialise_graph=_UnreachedGraphReader(),
                deadline=deadline,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 2.0

    def test_already_expired_deadline_raises_without_attempting_the_acquire(self, search_settings):
        """An expired deadline raises `GraphInfraError` before acquiring an available permit."""
        search_settings(graph_timeout_s=30.0)

        class _UnreachedGraphReader(FakeGraphReader):
            def observation_facets_batch_many(
                self, *, stat_vars, entities, needs_dates: bool = False, deadline=None
            ):
                raise AssertionError(
                    "observation_facets_batch_many must not run: the deadline has already passed"
                )

        deadline = time.monotonic() - 1.0
        start = time.monotonic()
        with pytest.raises(GraphInfraError):
            observation_facets_via_materialise(
                stat_vars=["sv/A"],
                entities=["country/USA"],
                materialise_graph=_UnreachedGraphReader(),
                deadline=deadline,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
