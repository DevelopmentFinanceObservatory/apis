"""Tests client ownership, app lifespan cleanup, offline guards, and generation wiring."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import search.engine.clients.graph_client as graph_module
import skills_index.shared as skills_shared_module
from graph_generation import FakeGeneration
from materialise.rate_tables import RateTableCache
from search.engine.app import create_app, search_surface
from search.engine.assemble import build_unsupported_part
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.sdmx import LiveSdmxClient
from search.engine.core import resolve, resolve_async
from search.engine.envelope import make_diagnostics, make_query_echo
from search.engine.errors import LLMInfraError
from search.engine.spec_build import build_unsupported_definition
from search.models import RawTextInput, Resolution, ResolveRequest
from tests.conftest import OfflineContractViolation
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import EMPTY_SDMX
from tests.engine._options_fixtures import _options_request
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM

_FAKE_LLM = FakeLLM()


# These queries need the dev-finance fixtures to exercise the data path.
_DEFAULT_SDMX = FakeSdmxClient()


class TestLiveGraphClientToken:
    def test_no_token_sends_no_header(self):
        client = LiveGraphClient(base="http://localhost:9999", token="")
        try:
            assert "x-graph-token" not in client._client._ours._client.headers
        finally:
            client.close()

    def test_token_sent_as_header(self):
        client = LiveGraphClient(base="http://localhost:9999", token="sekrit")
        try:
            assert client._client._ours._client.headers["x-graph-token"] == "sekrit"
        finally:
            client.close()


class TestLiveGraphClientClose:
    def test_close_closes_httpx_client(self):
        client = LiveGraphClient(base="http://localhost:9999")
        httpx_client = client._client._ours._client
        assert not httpx_client.is_closed
        client.close()
        assert httpx_client.is_closed

    def test_close_is_idempotent(self):
        client = LiveGraphClient(base="http://localhost:9999")
        client.close()
        client.close()


class _TrackingGraph(FakeGraph):
    """FakeGraph that records close() calls."""

    def __init__(self):
        super().__init__()
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _run(query: str, graph=None, sdmx=_DEFAULT_SDMX):
    req = ResolveRequest(input=RawTextInput(query=query))
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        return asyncio.run(resolve_async(req, graph=graph, llm=_FAKE_LLM, sdmx=sdmx))


class _TrackingSdmx(FakeSdmxClient):
    """FakeSdmxClient that records close() calls."""

    def __init__(self):
        super().__init__(observations={}, availability={})
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class TestResolveAsyncLifecycle:
    def test_injected_graph_is_not_closed(self):
        tracking = _TrackingGraph()
        _run("health ODA grants from USA to Ethiopia", graph=tracking)
        assert tracking.close_count == 0

    def test_self_built_graph_is_closed(self):
        closed_instances: list[bool] = []

        class _SpyClient(FakeGraph):
            def __init__(self_inner):
                super().__init__()

            def close(self_inner):
                closed_instances.append(True)

        class _FakeLiveGraphClientClass:
            @staticmethod
            def from_search_settings(*_args, **_kwargs):
                return _SpyClient()

        with patch("search.engine.core.LiveGraphClient", new=_FakeLiveGraphClientClass):
            _run("health ODA grants from USA to Ethiopia", graph=None)

        assert len(closed_instances) == 1

    def test_injected_sdmx_is_not_closed(self):
        tracking = _TrackingSdmx()
        _run("health ODA grants from USA to Ethiopia", graph=FakeGraph(), sdmx=tracking)
        assert tracking.close_count == 0

    def test_self_built_sdmx_is_closed(self):
        closed_instances: list[bool] = []

        class _SpyClient(FakeSdmxClient):
            def __init__(self_inner):
                super().__init__(observations={}, availability={})

            def close(self_inner):
                closed_instances.append(True)

        with patch("search.engine.core.LiveSdmxClient", return_value=_SpyClient()):
            _run("health ODA grants from USA to Ethiopia", graph=FakeGraph(), sdmx=None)

        assert len(closed_instances) == 1

    def test_both_self_built_clients_close_once_when_pipeline_raises(self, monkeypatch):
        """Cleanup when both clients are self-built and the pipeline raises:
        the branch a reordering of the two close() calls, or an early return
        added ahead of them, would silently break."""
        graph_closes: list[bool] = []
        sdmx_closes: list[bool] = []

        class _SpyGraph(FakeGraph):
            def close(self):
                graph_closes.append(True)

        class _SpySdmx(FakeSdmxClient):
            def __init__(self):
                super().__init__(observations={}, availability={})

            def close(self):
                sdmx_closes.append(True)

        async def _raising_pipeline(request, **_kwargs):
            raise RuntimeError("simulated pipeline failure")
            yield  # pragma: no cover — makes this an async generator function

        monkeypatch.setattr("search.engine.core._resolve_pipeline_events", _raising_pipeline)

        class _FakeLiveGraphClientClass:
            @staticmethod
            def from_search_settings(*_args, **_kwargs):
                return _SpyGraph()

        with (
            patch("search.engine.core.LiveGraphClient", new=_FakeLiveGraphClientClass),
            patch("search.engine.core.LiveSdmxClient", return_value=_SpySdmx()),
            pytest.raises(RuntimeError, match="simulated pipeline failure"),
        ):
            _run("health ODA grants from USA to Ethiopia", graph=None, sdmx=None)

        assert graph_closes == [True]
        assert sdmx_closes == [True]


class TestLifespanLogger:
    """Lifespan attaches a StreamHandler to the search logger; the guard is idempotent."""

    def test_logger_handler_attached_on_startup(self):
        # Clear any handlers that earlier tests may have attached.
        search_log = logging.getLogger("search")
        search_log.handlers.clear()

        app = create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)
        with TestClient(app):
            # Inside the lifespan the handler must be attached.
            assert any(isinstance(h, logging.StreamHandler) for h in search_log.handlers)

    def test_logger_handler_idempotent(self):
        # Running the lifespan twice must not double-add handlers.
        search_log = logging.getLogger("search")
        search_log.handlers.clear()

        app = create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)
        # First lifespan run
        with TestClient(app):
            pass
        count_after_first = len(search_log.handlers)

        app2 = create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)
        # Second lifespan run — should not add another handler.
        with TestClient(app2):
            assert len(search_log.handlers) == count_after_first


class TestLifespanWarm:
    """Lifespan calls llm.warm() when no LLM is injected; re-raises on failure."""

    def test_warm_called_when_no_llm_injected(self):
        app = create_app(graph=FakeGraph(), sdmx=EMPTY_SDMX)  # llm=None triggers warm
        # The lifespan builds the process's own LLM(memo=...) right after
        # warm(); this test asserts on warm, so the constructor is stubbed
        # as well.
        with (
            patch("search.engine.clients.llm.warm") as mock_warm,
            patch("search.engine.app.LLM") as mock_llm,
            TestClient(app),
        ):
            mock_warm.assert_called_once()
        mock_llm.assert_called_once()

    def test_warm_not_called_when_llm_injected(self):
        app = create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)
        with patch("search.engine.clients.llm.warm") as mock_warm, TestClient(app):
            mock_warm.assert_not_called()

    def test_boot_fails_on_missing_gemini_key(self):
        app = create_app(graph=FakeGraph(), sdmx=EMPTY_SDMX)  # llm=None triggers warm
        with (
            patch(
                "search.engine.clients.llm.warm",
                side_effect=LLMInfraError("GEMINI_API_KEY is required"),
            ),
            pytest.raises(LLMInfraError),
            TestClient(app, raise_server_exceptions=True),
        ):
            pass  # Startup should raise before reaching here


class TestLifespanClientOwnership:
    """The lifespan builds and closes only clients the caller does not inject."""

    def test_self_built_sdmx_is_built_once_and_closed_once(self):
        built: list[bool] = []
        closed: list[bool] = []

        class _SpySdmx:
            def __init__(self):
                built.append(True)

            def close(self):
                closed.append(True)

        with (
            patch("search.engine.app.LiveSdmxClient", side_effect=_SpySdmx),
            TestClient(create_app(graph=FakeGraph(), llm=FakeLLM())),
        ):
            pass

        assert built == [True]
        assert closed == [True]

    def test_injected_sdmx_is_never_built_or_closed(self):
        with (
            patch("search.engine.app.LiveSdmxClient") as mock_sdmx,
            TestClient(create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)),
        ):
            pass

        mock_sdmx.assert_not_called()

    def test_self_built_graph_is_built_once_and_closed_once(self):
        built: list[bool] = []
        closed: list[bool] = []

        class _SpyGraph:
            def __init__(self, **kwargs):
                built.append(True)

            def close(self):
                closed.append(True)

            def stats(self):
                return {}

            def clear(self):
                pass

        class _FakeLiveGraphClientClass:
            """Stands in for the real `LiveGraphClient` name, exposing only
            the one classmethod `search.engine.app`'s lifespan calls, so a
            `patch(..., side_effect=...)` on the bare constructor (never
            called directly) cannot intercept it."""

            @staticmethod
            def from_search_settings(*_args, **_kwargs):
                return _SpyGraph()

        with (
            patch("search.engine.app.LiveGraphClient", new=_FakeLiveGraphClientClass),
            TestClient(create_app(llm=FakeLLM(), sdmx=EMPTY_SDMX)),
        ):
            pass

        assert built == [True]
        assert closed == [True]

    def test_injected_graph_is_never_built_or_closed(self):
        with (
            patch("search.engine.app.LiveGraphClient") as mock_graph,
            TestClient(create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)),
        ):
            pass

        mock_graph.assert_not_called()

    def test_self_built_rate_cache_is_built_once_when_not_injected(self):
        built: list[bool] = []

        class _SpyRateCache:
            def __init__(self):
                built.append(True)

        with (
            patch("search.engine.app.RateTableCache", side_effect=_SpyRateCache),
            TestClient(create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)),
        ):
            pass

        assert built == [True]

    def test_injected_rate_cache_is_never_built(self):
        with (
            patch("search.engine.app.RateTableCache") as mock_rate_cache,
            TestClient(
                create_app(
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=EMPTY_SDMX,
                    rate_cache=RateTableCache(),
                )
            ),
        ):
            pass

        mock_rate_cache.assert_not_called()


class TestSkillsIndexLifespanOwnership:
    """The lifespan releases its shared skills index and refresh task; injected indexes stay caller-owned.

    Each test resets the process-wide singleton to isolate its reference count."""

    @pytest.fixture(autouse=True)
    def _isolated_singleton(self):
        skills_shared_module._singleton.force_reset()

    def test_self_acquired_skills_index_releases_its_refresh_task_on_shutdown(self):
        import search.engine.app as app_module
        from skills_index.shared import acquire_shared_index as real_acquire_shared_index

        captured: dict = {}

        async def _spy_acquire_shared_index(*, graph, **kwargs):
            shared = await real_acquire_shared_index(graph=graph, **kwargs)
            captured["shared"] = shared
            return shared

        with (
            patch.object(app_module, "acquire_shared_index", _spy_acquire_shared_index),
            TestClient(create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)),
        ):
            assert "shared" in captured

        # release_shared_index (the lifespan's own `finally`) cancelled and
        # awaited the background refresh task before this block exits -- it
        # must not be left pending after shutdown.
        assert captured["shared"]._task.done()

    def test_warm_failure_still_releases_the_skills_index(self):
        """A lifespan whose warm() raises still runs the `finally` that
        releases every resource acquired before the failure, the skills
        index included. Asserts on the released state directly (the
        singleton back at refcount 0, its background task done) rather than
        on a call count, since a release call that raced the task teardown
        would still leave a pending task undetected.
        """
        import search.engine.app as app_module
        from skills_index.shared import acquire_shared_index as real_acquire_shared_index

        captured: dict = {}

        async def _spy_acquire_shared_index(*, graph, **kwargs):
            shared = await real_acquire_shared_index(graph=graph, **kwargs)
            captured["shared"] = shared
            return shared

        app = create_app(graph=FakeGraph(), sdmx=EMPTY_SDMX)  # llm=None triggers warm
        with (
            patch.object(app_module, "acquire_shared_index", _spy_acquire_shared_index),
            patch(
                "search.engine.clients.llm.warm",
                side_effect=LLMInfraError("GEMINI_API_KEY is required"),
            ),
            pytest.raises(LLMInfraError),
            TestClient(app, raise_server_exceptions=True),
        ):
            pass  # Startup should raise before reaching here

        assert "shared" in captured
        assert captured["shared"]._task.done()
        assert skills_shared_module._singleton._instance is None
        assert skills_shared_module._singleton._refcount == 0

    def test_injected_skills_index_is_never_acquired_or_released(self):
        import search.engine.app as app_module

        def _fail_acquire(*_args, **_kwargs):
            raise AssertionError("acquire_shared_index must not run when skills_index is injected")

        def _fail_release(*_args, **_kwargs):
            raise AssertionError("release_shared_index must not run when skills_index is injected")

        sentinel = object()
        with (
            patch.object(app_module, "acquire_shared_index", _fail_acquire),
            patch.object(app_module, "release_shared_index", _fail_release),
            TestClient(
                create_app(
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=EMPTY_SDMX,
                    skills_index=sentinel,  # type: ignore[arg-type]
                )
            ),
        ):
            pass


class TestResolveLoopSafe:
    """Pins sync `resolve()` both outside and inside an event loop."""

    def _patch_async(self, monkeypatch, sentinel):
        async def fake_resolve_async(request):
            return sentinel

        monkeypatch.setattr("search.engine.core.resolve_async", fake_resolve_async)

    def test_standalone_call(self, monkeypatch):
        sentinel = object()
        self._patch_async(monkeypatch, sentinel)
        req = ResolveRequest(input=RawTextInput(query="x"))
        assert resolve(req) is sentinel

    def test_call_from_running_loop(self, monkeypatch):
        sentinel = object()
        self._patch_async(monkeypatch, sentinel)
        req = ResolveRequest(input=RawTextInput(query="x"))

        async def driver():
            # Mimics Langfuse calling the sync task from inside its loop.
            return resolve(req)

        assert asyncio.run(driver()) is sentinel


class TestOfflineContractGuardFires:
    """Control tests prove the offline guard raises when a live client is missing."""

    def test_missing_sdmx_injection_raises_offline_contract_violation(self):
        req = ResolveRequest(input=RawTextInput(query="health ODA grants from USA to Ethiopia"))
        with pytest.raises(OfflineContractViolation, match="LiveSdmxClient"):
            asyncio.run(resolve_async(req, graph=FakeGraph(), llm=_FAKE_LLM))

    def test_transport_guard_blocks_outbound_http(self):
        transport = httpx.HTTPTransport()
        with pytest.raises(OfflineContractViolation):
            transport.handle_request(httpx.Request("GET", "http://127.0.0.1:9/"))


class TestGenerationLifespanWiring:
    """The lifespan passes one generation provider to eight caches and unregisters them."""

    _EXPECTED_REGISTERED_NAMES = (
        "search.admission",
        "search.conversion_catalogue",
        "search.default_place_scope",
        "search.entity_resolver",
        "search.graph_client",
        "search.materialise_sdmx",
        "search.options_population_memo",
        "search.options_response_cache",
        "search.response_cache",
        "search.response_cache_memo",
        "search.sdmx",
    )

    def test_injected_generation_reaches_every_lifespan_built_cache_and_registers_it(
        self, monkeypatch
    ):
        """Seven caches, the response cache and its part memo share the injected generation
        provider, and the default place scope's process-wide memo and the conversion
        catalogue's reload register on it. All eleven registrations clear when the lifespan
        exits."""
        fake = FakeGeneration(token="gen-a")
        captured: dict[str, object] = {}

        def _spy(name: str, real_cls: type):
            class _Spy(real_cls):  # type: ignore[misc,valid-type]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    captured[name] = self

            return _Spy

        import search.engine.app as app_module

        monkeypatch.setattr(
            app_module, "LiveGraphClient", _spy("graph_client", graph_module.LiveGraphClient)
        )
        # CachingSdmxClient wraps LiveSdmxClient, so this overrides it from
        # the offline-guard's raiser back to the real class, letting the
        # wrapper it feeds actually construct.
        monkeypatch.setattr(app_module, "LiveSdmxClient", LiveSdmxClient)
        monkeypatch.setattr(
            app_module, "CachingSdmxClient", _spy("sdmx", app_module.CachingSdmxClient)
        )
        monkeypatch.setattr(
            app_module,
            "OptionsResponseCache",
            _spy("options_response_cache", app_module.OptionsResponseCache),
        )
        monkeypatch.setattr(
            app_module,
            "OptionsPopulationMemo",
            _spy("options_population_memo", app_module.OptionsPopulationMemo),
        )
        monkeypatch.setattr(
            app_module, "EntityResolver", _spy("entity_resolver", app_module.EntityResolver)
        )
        monkeypatch.setattr(app_module, "Admission", _spy("admission", app_module.Admission))
        build_materialise_sdmx_client = app_module.build_materialise_sdmx_client

        def _spy_materialise_sdmx(**kwargs):
            captured["materialise_sdmx"] = built = build_materialise_sdmx_client(**kwargs)
            return built

        monkeypatch.setattr(app_module, "build_materialise_sdmx_client", _spy_materialise_sdmx)

        surface = search_surface(
            llm=FakeLLM(),
            skills_index=object(),  # type: ignore[arg-type], never acquired/built, see below
            generation=fake,
        )
        lifespan = surface.lifespan
        assert lifespan is not None  # search_surface() always sets one

        async def _body() -> None:
            async with lifespan(FastAPI()):
                assert set(captured) == {
                    "graph_client",
                    "sdmx",
                    "materialise_sdmx",
                    "options_response_cache",
                    "options_population_memo",
                    "entity_resolver",
                    "admission",
                }
                for name, built in captured.items():
                    if name == "sdmx":
                        # CachingSdmxClient itself holds no `_generation`.
                        # It threads the provider into its own inner
                        # `SingleFlightCache` store instead (sdmx_cache.py's
                        # own docstring).
                        assert built._series_cache._generation is fake, name
                        continue
                    if name in ("options_response_cache", "options_population_memo"):
                        # Same shape as sdmx above: the provider lives on
                        # the inner `SingleFlightCache` store.
                        assert built._cache._generation is fake, name
                        continue
                    assert built._generation is fake, name
                assert fake.registered_names() == self._EXPECTED_REGISTERED_NAMES
            # Every registration is closed on lifespan exit. A forgotten
            # unregister would leave a later, unrelated lifespan colliding
            # on these same stable names against this same shared fake.
            assert fake.registered_names() == ()

        asyncio.run(_body())

    def test_unconfigured_generation_environment_still_serves_normally(self, monkeypatch):
        """Neither `DFO_GRAPH_GENERATION` nor `DFO_GRAPH_GENERATION_SERVICE`
        set (the common case before dfo-infra's own grant lands) must never
        fail a request. The real, unmocked shared provider degrades to
        "unknown" and the surface serves normally, mirroring graph_api.app's
        own identical test for the resolve door.
        """
        for var in ("DFO_GRAPH_GENERATION", "DFO_GRAPH_GENERATION_SERVICE"):
            monkeypatch.delenv(var, raising=False)

        app = create_app(graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX)
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get("/api/search/healthz")
        assert resp.status_code == 200, resp.text


class TestGenerationRefreshBeforeCacheLookup:
    """`resolve_endpoint`/`options_endpoint` both call `deps.generation.
    refresh_if_stale()` once at their own entry, ahead of any cache lookup.
    Asserted here as call order against a spy provider, not merely that it
    is called at all.
    """

    def test_resolve_endpoint_refreshes_before_resolve_async_runs(self, monkeypatch):
        order: list[str] = []

        class _SpyGeneration(FakeGeneration):
            async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
                order.append("refresh_if_stale")

        async def _fake_resolve_async(request, **_kwargs):
            order.append("resolve_async")
            echo = make_query_echo("x", extract_skipped=True)
            diag = make_diagnostics("test-build", [], {}, 0)
            shape = {"kind": "statvar_schema", "population_type": "sh1"}
            definition = build_unsupported_definition(
                shape, [], entities=None, requested_window=None
            )
            part = build_unsupported_part(
                "unit_request_unsupported",
                "/identity/facets/unit",
                "not reached: this response is a stand-in for ordering only",
                definition,
                variable_text="x",
                warnings=[],
            )
            return Resolution(query_echo=echo, diagnostics=diag, parts=[part])

        app = create_app(
            graph=FakeGraph(), llm=FakeLLM(), sdmx=EMPTY_SDMX, generation=_SpyGeneration()
        )
        with TestClient(app, raise_server_exceptions=True) as client:
            monkeypatch.setattr("search.engine.app.resolve_async", _fake_resolve_async)
            resp = client.post(
                "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "x"}}
            )

        assert resp.status_code == 200, resp.text
        assert order == ["refresh_if_stale", "resolve_async"]

    def test_options_endpoint_refreshes_before_resolve_options_target_runs(self, monkeypatch):
        order: list[str] = []

        class _SpyGeneration(FakeGeneration):
            async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
                order.append("refresh_if_stale")

        def _fake_resolve_options_target(*_args, **_kwargs):
            # Short-circuits before any real population derivation. This
            # test is only about call ORDER, not about building a valid
            # options target (tests/engine/test_options_response_cache.py
            # and test_options_surface.py already cover that pipeline).
            order.append("resolve_options_target")
            raise RuntimeError("stand-in: this test asserts call order, never a real response")

        monkeypatch.setattr(
            "search.engine.app.resolve_options_target", _fake_resolve_options_target
        )
        app = create_app(
            graph=FakeGraph(),
            llm=FakeLLM(),
            sdmx=EMPTY_SDMX,
            generation=_SpyGeneration(),
            # options_endpoint refuses without a materialise_graph, so this call-order-only test injects one to reach
            # resolve_options_target rather than refusing earlier at that guard.
            materialise_graph=FakeGraph(),
        )
        body = _options_request(entities={"observationAbout": ["country/KEN"]}).model_dump(
            by_alias=True, mode="json"
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/api/search/options", json=body)

        assert order == ["refresh_if_stale", "resolve_options_target"]
