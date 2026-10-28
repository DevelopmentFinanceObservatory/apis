"""The dfo-apis app for the local stack: `dfo_apis.app.create_app` without Search, without
Cloud Tasks, and with download links for the GCS emulator.

`dfo_apis.app.create_app` always mounts Search, whose startup checks the Gemini key against
Gemini and reads the live graph, so this factory mounts Datasets and Presentation alone, the
latter from `dfo_apis.app.build_presentation_surface` over the Datasets surface's dependency
cell, so live share slots check for newer data in process. Serve it with
`uvicorn --factory local_app:create_app --app-dir scripts/local-stack`.

The Datasets surface reads its graph, SDMX endpoint and coverage index from the doubles
`upstream.py` builds over `fixtures/readings.json`, and admits every provenance through a fake
origin catalogue. It takes a `NullTaskClient`: the live client needs Google credentials for
Cloud Tasks, and no worker runs locally, so a mint too large to answer in the request refuses
when it would enqueue.

With `STORAGE_EMULATOR_HOST` set, as `local.env` sets it, the storage client talks to
fake-gcs-server and `artifact_store.signing.sign_get_url` returns the object's unsigned media
link on it, so the files routes run unchanged and answer 302 at the emulator.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from upstream import coverage, graph_reader, load_fixture, sdmx_client

from artifact_store.signing import STORAGE_EMULATOR_ENV
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
    return platform_create_app(
        surfaces, cors_origins=parse_cors_origins(), title="DFO APIs (local stack)", version="1.0"
    )
