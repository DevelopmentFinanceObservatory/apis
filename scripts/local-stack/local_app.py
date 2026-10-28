"""The dfo-apis app for the local stack: `dfo_apis.app.create_app` without Search, without
Cloud Tasks, and with download links for the GCS emulator.

`dfo_apis.app.create_app` always mounts Search, whose startup checks the Gemini key against
Gemini and reads the live graph, so this factory mounts Datasets and Presentation alone, the
latter from `dfo_apis.app.build_presentation_surface` over the Datasets surface's dependency
cell, so live share slots check for newer data in process. Serve it with
`uvicorn --factory local_app:create_app --app-dir scripts/local-stack`.

The Datasets surface reads its graph, SDMX endpoint and coverage index from the doubles
`upstream.py` builds over `fixtures/readings.json`, and admits every provenance through a fake
origin catalogue. It takes a `NullTaskClient`, because the live client needs Google credentials
for Cloud Tasks and no worker runs locally. A mint too large to answer in the request refuses
when it would enqueue.

With `STORAGE_EMULATOR_HOST` set, as `local.env` sets it, the storage client talks to
fake-gcs-server and `artifact_store.signing.sign_get_url` returns the object's unsigned media
link. `local.env` sets `STORAGE_EMULATOR_LINK_BASE` to this server's origin, so the files
routes answer 302 on the API's own origin, as clients require, and `download` serves the
emulator's media path there: it streams the object from the emulator under the content type
and disposition the link names, as a signed URL's overrides set them.

Two local-only routes stand in for upstreams the Presentation surface reads over HTTP:
`download`, and `graph_node`, the `/v2/node` read of the graph client it builds from
`DATASETS_GRAPH_BASE_URL`, answered from the fixture (`upstream.node_reply`), so a card names
places and sources and its citation read does not wait on a refused connection.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from upstream import coverage, graph_nodes, graph_reader, load_fixture, node_reply, sdmx_client

from artifact_store.signing import STORAGE_EMULATOR_ENV, emulator_media_path
from datasets_api.app import datasets_surface
from datasets_api.deps import DepsCell
from datasets_api.operations.task_client import NullTaskClient
from dfo_apis.app import build_presentation_surface
from dfo_platform.app import create_app as platform_create_app
from dfo_platform.config import parse_cors_origins
from graph_release.context import bound_served_release
from graph_release.service import get_release_service
from materialise.rate_tables import RateTableCache
from tests.origin_catalogue.fake_catalogue import fake_catalogue_service


def create_app() -> FastAPI:
    """Mount Datasets and Presentation over the stores local.env names."""
    if not os.environ.get(STORAGE_EMULATOR_ENV):
        raise RuntimeError(
            f"{STORAGE_EMULATOR_ENV} is unset; the local stack serves the GCS emulator"
        )
    rate_cache = RateTableCache()
    datasets_cell = DepsCell()
    fixture = load_fixture()
    # The coverage index reads the code index of the release the stack serves.
    with bound_served_release(get_release_service()):
        fixture_coverage = coverage(fixture)
    surfaces = [
        datasets_surface(
            graph=graph_reader(fixture),
            sdmx=sdmx_client(fixture),
            coverage=fixture_coverage,
            catalogue=fake_catalogue_service(),
            rate_cache=rate_cache,
            task_client=NullTaskClient(),
            deps_cell=datasets_cell,
        ),
        build_presentation_surface(rate_cache, datasets_cell=datasets_cell),
    ]
    app = platform_create_app(
        surfaces, cors_origins=parse_cors_origins(), title="DFO APIs (local stack)", version="1.0"
    )
    nodes = graph_nodes(fixture)

    async def graph_node(request: Request) -> JSONResponse:
        payload: Mapping[str, object] = await request.json()
        return JSONResponse(node_reply(nodes, payload))

    app.add_api_route(
        "/local-stack/graph/core/api/v2/node",
        graph_node,
        methods=["POST"],
        include_in_schema=False,
    )
    app.add_api_route(
        "/download/storage/v1/b/{bucket}/o/{name}",
        download,
        methods=["GET"],
        include_in_schema=False,
    )
    return app


async def download(bucket: str, name: str, request: Request) -> Response:
    """Stream `name` in `bucket` from the emulator, typed and disposed as the link's
    `response-content-type` and `response-content-disposition` say. A refusal from the
    emulator passes through."""
    emulator = os.environ[STORAGE_EMULATOR_ENV].rstrip("/")
    client = httpx.AsyncClient()
    upstream = await client.send(
        client.build_request(
            "GET", emulator + emulator_media_path(bucket=bucket, object_name=name)
        ),
        stream=True,
    )
    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(
            body,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )
    params = request.query_params
    headers = {
        "content-type": params.get("response-content-type")
        or upstream.headers.get("content-type", "application/octet-stream")
    }
    if disposition := params.get("response-content-disposition"):
        headers["content-disposition"] = disposition

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(chunks(), headers=headers)
