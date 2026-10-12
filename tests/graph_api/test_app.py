"""Construction, lifespan ownership, and OpenAPI schema tests for `graph_surface`."""

from __future__ import annotations

import asyncio
import logging
import sys

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from dfo_apis.app import all_surfaces
from dfo_platform.app import create_app
from dfo_platform.build import build_commit
from graph_api import app as app_module
from graph_api.app import graph_surface
from graph_api.capabilities import build_capabilities
from graph_api.deps import GRAPH_DEPS_SLOT
from graph_api.node_models import NodeResponse
from graph_api.resolve_models import ResolveResponse
from graph_api.version import GRAPH_CAPABILITY_VERSION
from graph_generation import FakeGeneration

from .conftest import FakeGraphNodeReader, make_client

_GENERATION_ENV_VARS = ("DFO_GRAPH_GENERATION", "DFO_GRAPH_GENERATION_SERVICE")

_GRAPH_ENV_VARS = (
    "GRAPH_DCP_BASE_URL",
    "GRAPH_DCP_TOKEN",
    "GRAPH_NODE_POOL_SIZE",
    "GRAPH_RESOLVE_TIMEOUT_SECONDS",
    "GRAPH_NODE_TIMEOUT_SECONDS",
)


def test_graph_surface_constructs_under_an_empty_environment(monkeypatch: pytest.MonkeyPatch):
    """Surface construction needs no `GRAPH_*` settings; the lifespan reads them."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    surface = graph_surface(graph=FakeGraphNodeReader())

    assert surface.name == "graph"
    assert surface.prefix == "/api/graph"


def test_request_before_the_lifespan_refuses_503():
    """Before startup, dependency validation returns retryable 503 for every door."""
    client = TestClient(
        create_app([graph_surface(graph=FakeGraphNodeReader())]), raise_server_exceptions=False
    )

    for resp in (
        client.post("/api/graph/resolve", json={"texts": []}),
        client.post("/api/graph/resolve", json={"texts": ["Kenya"]}),
        client.get("/api/graph/node/country/KEN"),
    ):
        assert resp.status_code == 503
        body = resp.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True


@pytest.mark.parametrize(
    "send",
    [
        pytest.param(lambda client: client.get("/api/graph/node/country/MOZ"), id="node"),
        # No other test resolves this text, so the resolver's first read is its graph
        # confirmation read, which the resolver submits to the executor itself.
        pytest.param(
            lambda client: client.post("/api/graph/resolve", json={"texts": ["Mozambique"]}),
            id="resolve-confirmation",
        ),
    ],
)
def test_a_read_after_the_dcp_executor_shuts_down_refuses_retryable_503(send):
    graph = FakeGraphNodeReader(nodes={"country/MOZ": {"typeOf": ("Country",)}})
    client = make_client(graph=graph)

    with client:
        client.app.state.graph_deps.dcp_executor.shutdown()
        resp = send(client)

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "service_unavailable"
    assert resp.json()["retryable"] is True
    assert sum(graph.calls.values()) == 0


def test_lifespan_closes_owned_reader_when_dcp_executor_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """If executor construction fails, the lifespan closes its owned reader.

    `node_pool_size=0` raises `ValueError` after reader construction and
    before resolver acquisition or executor creation.
    """
    monkeypatch.setenv("GRAPH_DCP_BASE_URL", "https://graph.example.org/core/api")
    close_calls: list[None] = []
    acquire_calls: list[None] = []

    class _SpyGraphClient:
        def __init__(self, **_options: object) -> None:
            pass

        def close(self) -> None:
            close_calls.append(None)

    def _spy_acquire_shared():
        acquire_calls.append(None)
        raise AssertionError("acquire_shared must not run when dcp_executor construction failed")

    monkeypatch.setattr(app_module, "build_graph_client", _SpyGraphClient)
    monkeypatch.setattr(app_module, "acquire_shared", _spy_acquire_shared)

    surface = graph_surface(node_pool_size=0)
    app = create_app([surface])

    with pytest.raises(ValueError), TestClient(app, raise_server_exceptions=True):
        pass  # startup should raise before reaching here

    assert close_calls == [None]
    assert acquire_calls == []


def test_lifespan_shuts_down_dcp_executor_and_closes_reader_when_resolver_acquisition_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """Resolver acquisition failure shuts down the executor, then closes the reader."""
    monkeypatch.setenv("GRAPH_DCP_BASE_URL", "https://graph.example.org/core/api")
    reader_close_calls: list[None] = []
    built_executors: list[object] = []

    class _SpyGraphClient:
        def __init__(self, **_options: object) -> None:
            pass

        def close(self) -> None:
            reader_close_calls.append(None)

    def _raising_acquire_shared():
        raise RuntimeError("shared resolver acquisition failed")

    real_executor_cls = app_module.DcpReadExecutor

    class _SpyThreadPoolExecutor(real_executor_cls):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.shutdown_called = False
            built_executors.append(self)

        def shutdown(self, *args: object, **kwargs: object) -> None:
            self.shutdown_called = True
            super().shutdown(*args, **kwargs)

    monkeypatch.setattr(app_module, "build_graph_client", _SpyGraphClient)
    monkeypatch.setattr(app_module, "acquire_shared", _raising_acquire_shared)
    monkeypatch.setattr(app_module, "DcpReadExecutor", _SpyThreadPoolExecutor)

    surface = graph_surface()
    app = create_app([surface])

    with (
        pytest.raises(RuntimeError, match="shared resolver acquisition failed"),
        TestClient(app, raise_server_exceptions=True),
    ):
        pass  # startup should raise before reaching here

    assert reader_close_calls == [None]
    assert len(built_executors) == 1
    assert built_executors[0].shutdown_called is True


class _InertResolver:
    """Stand-in for an injected resolver, which the lifespan does not own."""


def test_lifespan_reentry_with_injected_deps_keeps_serving(monkeypatch: pytest.MonkeyPatch):
    """Re-entering the lifespan leaves injected dependencies open and serving."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    graph = FakeGraphNodeReader()
    app = create_app([graph_surface(graph=graph, resolver=_InertResolver())])

    with TestClient(app) as client:
        first = client.get("/api/graph/healthz")
    with TestClient(app) as client:
        second = client.get("/api/graph/healthz")

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text


def test_lifespan_reentry_rebuilds_and_closes_owned_deps(monkeypatch: pytest.MonkeyPatch):
    """Each lifespan entry builds and closes a reader and acquires and releases
    the shared resolver once. Spies avoid rebuilding resolvekit here.
    """
    monkeypatch.setenv("GRAPH_DCP_BASE_URL", "https://graph.example.org/core/api")

    reader_constructions: list[None] = []
    reader_closes: list[None] = []

    class _SpyGraphClient:
        def __init__(self, **_options: object) -> None:
            reader_constructions.append(None)

        def close(self) -> None:
            reader_closes.append(None)

    acquire_calls: list[None] = []
    release_calls: list[object] = []
    sentinel_shared = object()

    def _spy_acquire_shared():
        acquire_calls.append(None)
        return sentinel_shared

    def _spy_release_shared(shared: object) -> None:
        assert shared is sentinel_shared
        release_calls.append(shared)

    monkeypatch.setattr(app_module, "build_graph_client", _SpyGraphClient)
    monkeypatch.setattr(app_module, "acquire_shared", _spy_acquire_shared)
    monkeypatch.setattr(app_module, "release_shared", _spy_release_shared)

    app = create_app([graph_surface()])

    with TestClient(app, raise_server_exceptions=True) as client:
        first = client.get("/api/graph/healthz")
    assert first.status_code == 200, first.text
    assert len(reader_constructions) == 1
    assert len(reader_closes) == 1
    assert len(acquire_calls) == 1
    assert len(release_calls) == 1

    with TestClient(app, raise_server_exceptions=True) as client:
        second = client.get("/api/graph/healthz")
    assert second.status_code == 200, second.text
    assert len(reader_constructions) == 2
    assert len(reader_closes) == 2
    assert len(acquire_calls) == 2
    assert len(release_calls) == 2


def test_lifespan_builds_the_real_shared_resolver_covering_both_packs(
    monkeypatch: pytest.MonkeyPatch,
):
    """The lifespan resolver resolves Kenya from the geo pack and AfDB from `dfo_graph`."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    graph = FakeGraphNodeReader(
        nodes={
            "country/KEN": {"typeOf": ("Country",)},
            "AfricanDevelopmentBank": {"typeOf": ("Organization",)},
        }
    )
    app = create_app([graph_surface(graph=graph)])

    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya", "AfDB"]})

    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert results[0]["status"] == "resolved"
    assert results[0]["entity"]["dcid"] == "country/KEN"
    assert results[1]["status"] == "resolved"
    assert results[1]["entity"]["dcid"] == "AfricanDevelopmentBank"


def test_lifespan_passes_the_shared_generation_provider_to_the_entity_resolver_and_registers_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """The lifespan registers the resolver cache and the source document cache with its shared
    generation provider."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in _GENERATION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    sentinel_generation = FakeGeneration(token="gen-a")
    acquire_calls: list[str | None] = []
    release_calls: list[object] = []

    def _spy_acquire_shared_generation(*, catalogue, expected_graph_base):
        acquire_calls.append(expected_graph_base)
        return sentinel_generation

    def _spy_release_shared_generation(shared):
        assert shared is sentinel_generation
        release_calls.append(shared)

    monkeypatch.setattr(app_module, "acquire_shared_generation", _spy_acquire_shared_generation)
    monkeypatch.setattr(app_module, "release_shared_generation", _spy_release_shared_generation)

    real_entity_resolver_cls = app_module.EntityResolver
    captured_resolvers: list[object] = []

    class _SpyEntityResolver(real_entity_resolver_cls):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            captured_resolvers.append(self)

    monkeypatch.setattr(app_module, "EntityResolver", _SpyEntityResolver)

    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
    app = create_app([graph_surface(graph=graph)])

    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.get("/api/graph/healthz")
        assert resp.status_code == 200, resp.text

        assert len(captured_resolvers) == 1
        resolver = captured_resolvers[0]
        assert resolver._generation is sentinel_generation
        assert sentinel_generation.registered_names() == (
            "graph_api.entity_resolver",
            "graph_api.provenance_docs",
        )

    # `graph` is injected here, so there is no real DCP base URL to
    # cross-check the generation source's own service urls against.
    assert acquire_calls == [None]
    assert release_calls == [sentinel_generation]
    assert sentinel_generation.registered_names() == ()


def test_unconfigured_generation_environment_still_serves_resolve_normally(
    monkeypatch: pytest.MonkeyPatch,
):
    """With neither generation setting, the provider is unknown and resolve serves."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in _GENERATION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
    app = create_app([graph_surface(graph=graph)])

    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "resolved"
    assert result["entity"]["dcid"] == "country/KEN"


def test_openapi_titles_the_graph_resolve_request_schema_distinctly_from_search(
    monkeypatch: pytest.MonkeyPatch,
):
    """The composed app gives graph and search request models distinct schema titles."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    app = create_app(all_surfaces())
    spec = app.openapi()

    schemas = spec["components"]["schemas"]
    graph_schema = schemas["graph_api__resolve_models__ResolveRequest"]
    search_schema = schemas["search__models__request__ResolveRequest"]
    assert graph_schema["title"] == "GraphResolveRequest"
    assert search_schema["title"] != graph_schema["title"]


def test_openapi_titles_the_graph_resolve_response_schema_distinctly_from_search(
    monkeypatch: pytest.MonkeyPatch,
):
    """The composed app gives graph and search response models distinct schema titles."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    app = create_app(all_surfaces())
    spec = app.openapi()

    schemas = spec["components"]["schemas"]
    graph_schema = schemas["ResolveResponse"]
    assert graph_schema["title"] == "GraphResolveResponse"
    search_titles = {
        schemas[name]["title"] for name in ("Resolution", "CatalogueAnswer", "DeclinedAnswer")
    }
    assert graph_schema["title"] not in search_titles


class TestOpenAPIResponseModels:
    """OpenAPI lists each route's response model; wire-byte tests verify the
    served JSON matches that model's serialised form.
    """

    def test_kenya_resolve_wire_body_validates_against_resolve_response(self):
        graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
        client = make_client(graph=graph)

        with client:
            resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

        assert resp.status_code == 200, resp.text
        ResolveResponse.model_validate(resp.json())

        spec = client.app.openapi()
        op = spec["paths"]["/api/graph/resolve"]["post"]
        assert op["responses"]["200"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ResolveResponse"
        }

    def test_kenya_resolve_wire_bytes_match_a_resolveresponse_rebuild(self):
        graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
        client = make_client(graph=graph)

        with client:
            resp = client.post(
                "/api/graph/resolve",
                json={"texts": ["Kenya"]},
                headers={"accept-encoding": "identity"},
            )

        assert resp.status_code == 200, resp.text
        rebuilt = ResolveResponse.model_validate(resp.json()).model_dump(mode="json")
        assert resp.content == JSONResponse(rebuilt).body

    def test_statvar_node_wire_body_validates_against_node_response(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/BilateralFlow_Grants": {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("Bilateral flow (grants)",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/base/BilateralFlow_Grants")

        assert resp.status_code == 200, resp.text
        TypeAdapter(NodeResponse).validate_python(resp.json())

        spec = client.app.openapi()
        op = spec["paths"]["/api/graph/node/{dcid}"]["get"]
        body_schema = op["responses"]["200"]["content"]["application/json"]["schema"]
        assert body_schema["discriminator"]["propertyName"] == "shape"
        assert set(body_schema["discriminator"]["mapping"]) == {
            "stat_var",
            "source",
            "entity",
            "node",
        }

    def test_statvar_node_wire_bytes_match_a_noderesponse_rebuild(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/base/BilateralFlow_Grants": {
                    "typeOf": ("StatisticalVariable",),
                    "name": ("Bilateral flow (grants)",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get(
                "/api/graph/node/dc/base/BilateralFlow_Grants",
                headers={"accept-encoding": "identity"},
            )

        assert resp.status_code == 200, resp.text
        adapter = TypeAdapter(NodeResponse)
        rebuilt = adapter.dump_python(adapter.validate_python(resp.json()), mode="json")
        assert resp.content == JSONResponse(rebuilt).body

    def test_capabilities_operation_declares_a_non_empty_200_schema(self):
        client = make_client()
        spec = client.app.openapi()
        op = spec["paths"]["/api/graph/capabilities"]["get"]
        body_schema = op["responses"]["200"]["content"]["application/json"]["schema"]
        assert body_schema == {"$ref": "#/components/schemas/Capabilities"}

    def test_capabilities_wire_bytes_match_a_fresh_build_capabilities(self):
        client = make_client()

        resp = client.get("/api/graph/capabilities", headers={"accept-encoding": "identity"})

        assert resp.status_code == 200, resp.text
        rebuilt = build_capabilities(
            capability_version=GRAPH_CAPABILITY_VERSION, build=build_commit()
        ).model_dump(mode="json")
        assert resp.content == JSONResponse(rebuilt).body


class TestCoverageAcquisition:
    def test_returns_the_process_wide_service(self, monkeypatch: pytest.MonkeyPatch):
        import series_coverage.shared as coverage_shared

        sentinel = object()
        monkeypatch.setattr(coverage_shared, "get_coverage_service", lambda: sentinel)

        assert asyncio.run(app_module._acquire_coverage()) is sentinel

    def test_unset_uri_returns_none_at_info(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        import series_coverage.shared as coverage_shared

        def _missing():
            raise coverage_shared.CoverageArtifactUriMissingError("unset")

        monkeypatch.setattr(coverage_shared, "get_coverage_service", _missing)

        with caplog.at_level(logging.INFO, logger=app_module.__name__):
            assert asyncio.run(app_module._acquire_coverage()) is None
        assert [r.levelno for r in caplog.records] == [logging.INFO]

    def test_unreadable_artifact_returns_none_at_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        import series_coverage.shared as coverage_shared

        def _unreadable():
            raise OSError("bucket unreachable")

        monkeypatch.setattr(coverage_shared, "get_coverage_service", _unreadable)

        with caplog.at_level(logging.INFO, logger=app_module.__name__):
            assert asyncio.run(app_module._acquire_coverage()) is None
        assert [r.levelno for r in caplog.records] == [logging.WARNING]

    def test_missing_coverage_extra_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        # A `None` entry makes the import raise `ImportError`, as a graph-only install does.
        monkeypatch.setitem(sys.modules, "series_coverage.shared", None)

        assert asyncio.run(app_module._acquire_coverage()) is None

    def test_live_lifespan_publishes_the_acquired_service(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GRAPH_DCP_BASE_URL", "https://graph.example.org/core/api")
        sentinel = object()

        class _SpyGraphClient:
            def __init__(self, **_options: object) -> None:
                pass

            def close(self) -> None:
                pass

        async def _fake_acquire_coverage():
            return sentinel

        monkeypatch.setattr(app_module, "build_graph_client", _SpyGraphClient)
        monkeypatch.setattr(app_module, "acquire_shared", lambda: object())
        monkeypatch.setattr(app_module, "release_shared", lambda shared: None)
        monkeypatch.setattr(app_module, "_acquire_coverage", _fake_acquire_coverage)

        app = create_app([graph_surface()])
        with TestClient(app):
            assert getattr(app.state, GRAPH_DEPS_SLOT.name).coverage is sentinel

    def test_injected_graph_skips_acquisition(self, monkeypatch: pytest.MonkeyPatch):
        async def _refuse():
            raise AssertionError("an injected graph must not acquire coverage")

        monkeypatch.setattr(app_module, "_acquire_coverage", _refuse)

        app = create_app([graph_surface(graph=FakeGraphNodeReader(), resolver=_InertResolver())])
        with TestClient(app):
            assert getattr(app.state, GRAPH_DEPS_SLOT.name).coverage is None
