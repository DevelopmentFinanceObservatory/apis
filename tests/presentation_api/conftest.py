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
from presentation_api.newest_frames import NewestPending
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


BARE_SPEC_ID = "spec_" + "c" * 32
"""A dataset id a bare `frame_ref` names, which `FakeNewestSpecs` resolves by default to the
malaria frame under its own served id."""


class FakeNewestSpecs:
    """`NewestFrames` resolving bare dataset ids: `answer` is the frame every id resolves to, a
    `NewestPending`, or an exception to raise. Records every id asked about."""

    def __init__(self) -> None:
        self.answer: FrameRef | NewestPending | Exception = malaria_ethiopia_frame_ref()
        self.asked: list[str] = []

    def newest(self, frame: FrameRef, *, rendering: object) -> FrameRef:
        raise AssertionError("no live share slot is checked here")

    def newest_of_spec(self, spec_id: str) -> FrameRef | NewestPending:
        self.asked.append(spec_id)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def newest_specs() -> FakeNewestSpecs:
    return FakeNewestSpecs()


@pytest.fixture
def newest_client(
    frame_store: FakeFrameStore,
    revision_store: FakeRevisionStore,
    release_registry: FakeReleaseRegistry,
    newest_specs: FakeNewestSpecs,
) -> Iterator[TestClient]:
    """A client of a surface that resolves bare dataset ids through `newest_specs`."""
    surface = presentation_surface(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        newest_frames=newest_specs,
    )
    with TestClient(create_app([surface]), raise_server_exceptions=False) as test_client:
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
