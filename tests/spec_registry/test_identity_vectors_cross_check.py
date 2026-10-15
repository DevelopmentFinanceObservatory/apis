"""Cross-check positive `spec-kit` vectors through the registry adapter and write path.

`tests/engine/test_identity_vectors.py` covers the kit itself. Set `DFO_SPEC_KIT_DIR` in a
worktree because the sibling fallback resolves beside it. `DFO_SPEC_KIT_REQUIRED` makes missing
vectors an error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dfo_spec_kit.identity import parse_identity_bytes

from spec_registry.identity import recompute_identity
from spec_registry.registry import LiveSpecRegistry

from .conftest import FakeDatasetStore


def _spec_kit_dir() -> Path:
    env_dir = os.environ.get("DFO_SPEC_KIT_DIR")
    if env_dir:
        return Path(env_dir)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "dfo-spec-kit"


def _vector_paths() -> list[Path]:
    spec_kit_dir = _spec_kit_dir()
    if not spec_kit_dir.is_dir():
        return []
    return sorted((spec_kit_dir / "vectors" / "identity").glob("*.json"))


_VECTOR_PATHS = _vector_paths()

if not _VECTOR_PATHS:
    if os.environ.get("DFO_SPEC_KIT_REQUIRED"):
        raise RuntimeError(
            "DFO_SPEC_KIT_REQUIRED is set but no identity vectors were found "
            f"under {_spec_kit_dir()} -- the vector suite checkout is missing or has moved"
        )
    pytest.skip(
        "dfo-spec-kit identity vector suite not found -- set DFO_SPEC_KIT_DIR "
        "or check out dfo-spec-kit as a sibling of dfo-apis",
        allow_module_level=True,
    )

_ALL_VECTORS = [json.loads(path.read_bytes()) for path in _VECTOR_PATHS]
_POSITIVE_VECTORS = [v for v in _ALL_VECTORS if v["classification"] == "positive"]


@pytest.mark.parametrize("vector", _POSITIVE_VECTORS, ids=[v["name"] for v in _POSITIVE_VECTORS])
def test_registering_a_vector_recomputes_its_pinned_spec_id(vector: dict) -> None:
    if "input_bytes" in vector:
        identity_definition = parse_identity_bytes(bytes.fromhex(vector["input_bytes"]))
    else:
        identity_definition = vector["input"]

    recomputed = recompute_identity(identity_definition)
    assert recomputed.spec_id == vector["spec_id"]

    registry = LiveSpecRegistry(store=FakeDatasetStore())
    result = registry.register_definition(
        identity_definition=identity_definition,
        schema_version=1,
        supplied_spec_id=vector["spec_id"],
    )
    assert result.spec_id == vector["spec_id"]
    assert result.inserted is True
