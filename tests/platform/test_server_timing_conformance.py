"""Checks the `total` Server-Timing metric across success and validation responses on all surfaces.

The composed app enables every surface. Graph dependencies are injected because validation runs
before the resolve handler.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from dfo_platform.app import SERVER_TIMING_HEADER
from entity_resolution import Resolution
from graph_api.deps import GRAPH_DEPS_SLOT, GraphDeps
from graph_api.doors.sources import new_provenance_doc_cache
from graph_generation import UNKNOWN_GENERATION
from tests.engine._search_app import with_fake_search_deps
from tests.fixtures import FakeGraph

_ENV_VARS = (
    "DFO_DATASETS_SURFACE",
    "DFO_GRAPH_SURFACE",
    "DFO_PRESENTATION_SURFACE",
    "DFO_SKILLS_SURFACE",
)

# Every mounted surface has a healthz route, so this list covers all five and
# backs the success-response assertion below.
_HEALTHZ_PROBES = [
    ("search", "/api/search/healthz"),
    ("datasets", "/api/datasets/healthz"),
    ("graph", "/api/graph/healthz"),
    ("presentation", "/api/presentation/healthz"),
    ("skills", "/api/skills/healthz"),
]

# Skills serves three GET-only routes and no POST, so it has no request body
# that can fail validation and is excluded from this table; the other four
# surfaces each expose a POST route whose body fails validation before the
# handler runs.
_VALIDATION_PROBES = [
    ("search", "/api/search/resolve"),
    ("datasets", "/api/datasets"),
    ("graph", "/api/graph/resolve"),
    ("presentation", "/api/presentation/components"),
]


class _UnreadResolver:
    """An `EntityResolver`-shaped stand-in for `client`'s own `GraphDeps`,
    never called: every probe below fails FastAPI's own request validation
    before `resolve_endpoint`'s body runs.
    """

    async def resolve(self, texts: list[str], *, deadline: float | None = None) -> list[Resolution]:
        raise NotImplementedError


@pytest.fixture(scope="module")
def client() -> TestClient:
    import dfo_apis.app as dfo_apis_app

    for env_var in _ENV_VARS:
        os.environ[env_var] = "1"
    try:
        app = dfo_apis_app.create_app()
    finally:
        for env_var in _ENV_VARS:
            del os.environ[env_var]

    assert [s.name for s in dfo_apis_app.all_surfaces()] == [
        "search",
        "datasets",
        "graph",
        "presentation",
        "skills",
    ]
    with_fake_search_deps(app)
    executor = ThreadPoolExecutor(max_workers=1)
    GRAPH_DEPS_SLOT.set(
        app,
        GraphDeps(
            graph=FakeGraph(),
            resolver=_UnreadResolver(),
            dcp_executor=executor,
            generation=UNKNOWN_GENERATION,
            provenance_docs=new_provenance_doc_cache(
                executor=executor, generation=UNKNOWN_GENERATION
            ),
        ),
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("name,healthz_path", _HEALTHZ_PROBES, ids=lambda v: v)
def test_success_probe_carries_a_total_server_timing_metric(
    client: TestClient, name: str, healthz_path: str
):
    resp = client.get(healthz_path)

    assert resp.status_code == 200, f"{name}'s own healthz probe did not answer 200: {resp.text}"
    assert "total" in resp.headers.get(SERVER_TIMING_HEADER, "")


@pytest.mark.parametrize("name,post_path", _VALIDATION_PROBES, ids=lambda v: v)
def test_validation_refusal_probe_carries_a_total_server_timing_metric(
    client: TestClient, name: str, post_path: str
):
    resp = client.post(post_path, json={})

    assert resp.status_code in (400, 422), (
        f"{name}'s own validation probe did not answer a request-validation refusal: {resp.text}"
    )
    assert "total" in resp.headers.get(SERVER_TIMING_HEADER, "")
