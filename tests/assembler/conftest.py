"""Resolve the spec-kit catalogue from `DFO_SPEC_KIT_DIR` or a sibling checkout."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from assembler.catalogue import CatalogueRelease, load_release
from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore

_BUNDLE_RELATIVE_PATH = Path("python/dfo-spec-kit-components/src/dfo_spec_kit_components/catalogue")


def _spec_kit_dir() -> Path:
    env_dir = os.environ.get("DFO_SPEC_KIT_DIR")
    if env_dir:
        return Path(env_dir)
    # Sibling checkout: <repo root>/../dfo-spec-kit, resolved from this file's
    # own path so it works regardless of the invoking cwd.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "dfo-spec-kit"


def _resolved_catalogue_dir() -> Path:
    return _spec_kit_dir() / _BUNDLE_RELATIVE_PATH


@pytest.fixture(scope="session")
def catalogue_dir() -> Path:
    """Return the catalogue directory once per session.

    Missing bundles skip users unless `DFO_SPEC_KIT_REQUIRED` is set, which
    turns the missing checkout into a collection error.
    """
    directory = _resolved_catalogue_dir()
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
