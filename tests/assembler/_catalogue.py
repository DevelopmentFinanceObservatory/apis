"""Where the tests find the spec-kit catalogue: `DFO_SPEC_KIT_DIR`, else a sibling checkout."""

from __future__ import annotations

import os
from pathlib import Path

_BUNDLE_RELATIVE_PATH = Path("python/dfo-spec-kit-components/src/dfo_spec_kit_components/catalogue")


def _spec_kit_dir() -> Path:
    env_dir = os.environ.get("DFO_SPEC_KIT_DIR")
    if env_dir:
        return Path(env_dir)
    # Sibling checkout: <repo root>/../dfo-spec-kit, resolved from this file's
    # own path so it works regardless of the invoking cwd.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "dfo-spec-kit"


def resolved_catalogue_dir() -> Path:
    return _spec_kit_dir() / _BUNDLE_RELATIVE_PATH
