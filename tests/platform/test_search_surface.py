"""Checks Search and an independent sibling surface in the shared platform app."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.testclient import TestClient

from dfo_platform.app import Surface, create_app
from dfo_platform.problem import ProblemSpec
from search.engine.app import search_surface
from search.engine.config import settings
from tests.engine._search_app import set_search_deps


class _Dummy:
    """Stands in for the graph/llm clients; search_surface's lifespan skips
    building live ones when both are injected, and this test never exercises
    a route that reaches either client.
    """

    def __getattr__(self, name):
        raise AssertionError(f"this test must not reach the {name!r} client")


class _ForeignError(Exception):
    """A surface-specific exception type belonging to the foreign surface
    below, not to search -- proves per-surface exception routing picks the
    right handler with search mounted alongside it.
    """


def _foreign_surface() -> Surface:
    """A surface independent of search.*, mounted beside search's own
    surface below to exercise composition against an arbitrary sibling API
    rather than only the real search surface.
    """
    router = APIRouter()

    @router.get("/api/foreign/ping")
    async def ping():
        return {"surface": "foreign"}

    @router.get("/api/foreign/boom")
    async def boom():
        raise _ForeignError("foreign boom")

    catalogue = {"foreign_code": ("Foreign Title", "foreign default detail")}

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ForeignError):
            return ProblemSpec(status=400, code="foreign_code", retryable=False)
        return None

    return Surface(
        name="foreign",
        prefix="/api/foreign",
        router=router,
        capability_version="vforeign",
        codes=catalogue,
        classify=classify,
        exception_types=(_ForeignError,),
    )


def _make_client() -> TestClient:
    app = create_app([search_surface(graph=_Dummy(), llm=_Dummy()), _foreign_surface()])
    set_search_deps(app, graph=_Dummy(), resolver=_Dummy(), llm=_Dummy(), sdmx=_Dummy())
    return TestClient(app, raise_server_exceptions=False)


def test_searchs_own_routes_still_answer_beside_a_foreign_surface():
    client = _make_client()
    resp = client.get("/api/search/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_foreign_surfaces_routes_answer_too():
    client = _make_client()
    resp = client.get("/api/foreign/ping")
    assert resp.status_code == 200
    assert resp.json() == {"surface": "foreign"}


def test_each_surfaces_refusal_carries_its_own_capability_version():
    client = _make_client()

    # search's own refusal -- an over-long raw_text query, which resolve
    # rejects before any LLM or graph call, so the (dummy) clients stay
    # untouched.
    search_resp = client.post(
        "/api/search/resolve",
        json={
            "schema_version": 1,
            "input": {"kind": "raw_text", "query": "x" * (settings().max_query_chars + 1)},
        },
    )
    assert search_resp.status_code == 422
    search_body = search_resp.json()
    assert search_body["code"] == "query_too_long"
    assert search_body["capability_version"] == settings().capability_version

    foreign_resp = client.get("/api/foreign/boom")
    assert foreign_resp.status_code == 400
    foreign_body = foreign_resp.json()
    assert foreign_body["code"] == "foreign_code"
    assert foreign_body["capability_version"] == "vforeign"


def test_every_route_in_searchs_own_router_starts_with_its_prefix():
    surface = search_surface(graph=_Dummy(), llm=_Dummy())
    assert surface.prefix == "/api/search"
    for route in surface.router.routes:
        assert route.path.startswith("/api/search")


def test_search_surface_builds_with_zero_arguments():
    # A standalone Search app and every existing caller that omits
    # `rate_cache` must keep building -- construction only, no lifespan
    # (which would need live GEMINI_API_KEY/graph credentials).
    surface = search_surface()
    assert surface.name == "search"
