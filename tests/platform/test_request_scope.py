"""A surface's `request_scope` binds the origin catalogue for every path a request takes."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter
from fastapi.testclient import TestClient

from dfo_platform.app import Surface, create_app
from dfo_platform.executor import submit_with_context
from origin_catalogue import bound_catalogue, current_catalogue
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue

_SNAPSHOT = restrictive_catalogue(build_id="request-scope-build")


def _app() -> TestClient:
    router = APIRouter()

    @router.get("/api/t/async")
    async def read_async() -> dict[str, str]:
        return {"build_id": current_catalogue().build_id}

    @router.get("/api/t/sync")
    def read_sync() -> dict[str, str]:
        return {"build_id": current_catalogue().build_id}

    @router.get("/api/t/to-thread")
    async def read_to_thread() -> dict[str, str]:
        return {"build_id": await asyncio.to_thread(lambda: current_catalogue().build_id)}

    @router.get("/api/t/pool")
    def read_pool() -> dict[str, str]:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = submit_with_context(pool, lambda: current_catalogue().build_id)
            return {"build_id": future.result()}

    surface = Surface(
        name="t",
        prefix="/api/t",
        router=router,
        capability_version="v-t",
        codes={},
        request_scope=lambda: bound_catalogue(_SNAPSHOT),
    )
    return TestClient(create_app([surface]))


def test_every_request_path_reads_the_catalogue_its_surface_bound() -> None:
    client = _app()
    for path in ("/api/t/async", "/api/t/sync", "/api/t/to-thread", "/api/t/pool"):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
        assert response.json() == {"build_id": "request-scope-build"}, path
