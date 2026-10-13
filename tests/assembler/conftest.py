"""Resolve the spec-kit catalogue from `DFO_SPEC_KIT_DIR` or a sibling checkout."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from assembler.catalogue import CatalogueRelease, load_release
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore

from ._catalogue import resolved_catalogue_dir


@pytest.fixture(scope="session")
def catalogue_dir() -> Path:
    """Return the catalogue directory once per session.

    Missing bundles skip users unless `DFO_SPEC_KIT_REQUIRED` is set, which
    turns the missing checkout into a collection error.
    """
    directory = resolved_catalogue_dir()
    if not (directory / "manifest.json").is_file():
        if os.environ.get("DFO_SPEC_KIT_REQUIRED"):
            raise RuntimeError(
                "DFO_SPEC_KIT_REQUIRED is set but no presentation catalogue bundle was found "
                f"under {directory} -- the spec-kit checkout is missing, or the bundle's own "
                "path within it has moved"
            )
        pytest.skip(
            "presentation catalogue bundle not found -- set DFO_SPEC_KIT_DIR or check out "
            "dfo-spec-kit as a sibling of dfo-apis"
        )
    return directory


@pytest.fixture(scope="session")
def release(catalogue_dir: Path) -> CatalogueRelease:
    """Load the resolved bundle once for read-only tests."""
    return load_release(catalogue_dir=catalogue_dir)


@pytest.fixture
def frame_store():
    return FakeFrameStore()


@pytest.fixture
def revision_store():
    return FakeRevisionStore()


@pytest.fixture
def release_registry(release):
    return FakeReleaseRegistry(release=release)
