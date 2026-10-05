"""Offline fixtures for the Presentation API, backed by assembler fakes and a pinned catalogue."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from assembler.catalogue import CatalogueRelease, load_release
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import FrameRef
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
    malaria_ethiopia_rows,
)
from dfo_platform.app import create_app
from presentation_api.app import presentation_surface
from tests.spec_kit_bundle import require_catalogue_dir


@pytest.fixture(scope="session")
def catalogue_dir() -> Path:
    """Resolve the catalogue bundle once per test session."""
    return require_catalogue_dir()


@pytest.fixture(scope="session")
def release(catalogue_dir: Path) -> CatalogueRelease:
    """Load the immutable catalogue release shared by tests."""
    return load_release(catalogue_dir=catalogue_dir)


@pytest.fixture
def frame_ref() -> FrameRef:
    return malaria_ethiopia_frame_ref()


@pytest.fixture
def frame_store(frame_ref: FrameRef) -> FakeFrameStore:
    store = FakeFrameStore()
    store.put(resolved_frame=malaria_ethiopia_resolved_frame())
    return store


@pytest.fixture
def revision_store() -> FakeRevisionStore:
    return FakeRevisionStore()


@pytest.fixture
def release_registry(release: CatalogueRelease) -> FakeReleaseRegistry:
    return FakeReleaseRegistry(release=release)


@pytest.fixture
def app(
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
) -> FastAPI:
    surface = presentation_surface(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    return create_app([surface])


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def multi_provenance_frame_ref() -> FrameRef:
    return FrameRef(
        spec_id="spec_" + "a" * 32,
        content_version="cv_" + "b" * 32,
    )


def build_multi_provenance_resolved_frame():
    """Build a frame with two provenance values to trigger the unbound-facet refusal."""
    rows = tuple(malaria_ethiopia_rows())
    extra = dict(rows[0])
    extra["time_period"] = "2025"
    extra["provenance"] = "src_other"
    return build_resolved_frame(
        frame_ref=multi_provenance_frame_ref(),
        role_keys=("recipient",),
        rows=(*rows, extra),
        labels={"country/ETH": "Ethiopia"},
    )
