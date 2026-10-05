"""Resolve the presentation catalogue from dfo-spec-kit for tests that need a real release.

Each package wraps `require_catalogue_dir` in its own fixture so skip and failure reporting stays
local to that package.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

__all__ = ["require_catalogue_dir", "resolved_catalogue_dir", "spec_kit_dir"]

_BUNDLE_RELATIVE_PATH = Path("python/dfo-spec-kit-components/src/dfo_spec_kit_components/catalogue")


def spec_kit_dir() -> Path:
    env_dir = os.environ.get("DFO_SPEC_KIT_DIR")
    if env_dir:
        return Path(env_dir)
    # Resolve the sibling checkout at <repo root>/../dfo-spec-kit from this file's path,
    # independent of the invoking cwd.
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "dfo-spec-kit"


def resolved_catalogue_dir() -> Path:
    return spec_kit_dir() / _BUNDLE_RELATIVE_PATH


def require_catalogue_dir() -> Path:
    """Return the path. Missing bundles skip by default and raise when `DFO_SPEC_KIT_REQUIRED` is
    set.
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
