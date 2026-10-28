"""The dfo-apis app for the local stack: `dfo_apis.app.create_app` without Search, without
Cloud Tasks, and with download links for the GCS emulator.

`dfo_apis.app.create_app` always mounts Search, whose startup checks the Gemini key against
Gemini and reads the live graph, so this factory mounts Datasets and Presentation alone, the
latter from `dfo_apis.app`'s own factory. Serve it with
`uvicorn --factory local_app:create_app --app-dir scripts/local-stack`.

The Datasets surface reads its graph, SDMX endpoint and coverage index from the doubles
`upstream.py` builds over `fixtures/readings.json`, and admits every provenance through a fake
origin catalogue. It takes a `NullTaskClient`: the live client needs Google credentials for
Cloud Tasks, and no worker runs locally, so a mint too large to answer in the request refuses
when it would enqueue.

The files routes answer 302 at a V4 URL that `artifact_store.signing.sign_get_url` signs
through IAM `signBlob`, which needs Google credentials. fake-gcs-server checks no signature,
so with `STORAGE_EMULATOR_HOST` set this factory replaces that one function with
`emulator_url`, which names the object's media link on the emulator. Every other step of the
route (version read, revocation, retention, renewal, the 302 and its headers) runs unchanged.
"""

from __future__ import annotations

import os
from datetime import timedelta
from urllib.parse import quote

from fastapi import FastAPI
from google.cloud import storage
from upstream import coverage, graph_reader, load_fixture, sdmx_client

import artifact_store.store as artifact_store_module
from datasets_api.app import datasets_surface
from datasets_api.operations.task_client import NullTaskClient
from dfo_apis.app import _presentation_factory
from dfo_platform.app import create_app as platform_create_app
from dfo_platform.config import parse_cors_origins
from graph_release.context import bound_served_release
from graph_release.service import get_release_service
from materialise.rate_tables import RateTableCache
from tests.origin_catalogue.fake_catalogue import fake_catalogue_service

EMULATOR_ENV = "STORAGE_EMULATOR_HOST"


def emulator_url(
    *,
    client: storage.Client,
    bucket: str,
    object_name: str,
    digest: str,
    ttl: timedelta,
    response_type: str,
    response_disposition: str,
) -> str:
    """The emulator's media URL for `object_name` in `bucket`, in place of a signed URL. Takes
    `sign_get_url`'s arguments; the emulator serves the object to any caller."""
    host = os.environ[EMULATOR_ENV].rstrip("/")
    return f"{host}/download/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}?alt=media"


def create_app() -> FastAPI:
    """Mount Datasets and Presentation over the stores local.env names."""
    if not os.environ.get(EMULATOR_ENV):
        raise RuntimeError(f"{EMULATOR_ENV} is unset; the local stack serves the GCS emulator")
    artifact_store_module.sign_get_url = emulator_url
    rate_cache = RateTableCache()
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
        ),
        _presentation_factory(rate_cache),
    ]
    return platform_create_app(
        surfaces, cors_origins=parse_cors_origins(), title="DFO APIs (local stack)", version="1.0"
    )
