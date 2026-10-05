"""Checks currency-role identity vectors against the public `BaseIdentity` wire model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dfo_spec_kit.identity import spec_id_from_value
from pydantic import ValidationError

from search.models import BaseIdentity


def _spec_kit_dir() -> Path | None:
    env_dir = os.environ.get("DFO_SPEC_KIT_DIR")
    if env_dir:
        return Path(env_dir)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "dfo-spec-kit"


def _vector(name: str) -> dict | None:
    spec_kit_dir = _spec_kit_dir()
    if spec_kit_dir is None or not spec_kit_dir.is_dir():
        return None
    path = spec_kit_dir / "vectors" / "identity" / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_bytes())


_POSITIVE_VECTOR_NAMES = [
    "unit-request-currency-role-destination-current",
    "unit-request-currency-role-observation-about-current",
    "unit-request-currency-role-origin-constant",
]
_NEGATIVE_VECTOR_NAMES = [
    "unit-request-currency-role-extra-member-rejected",
    "unit-request-currency-role-lowercase-rejected",
    "unit-request-currency-role-unrecognised-rejected",
]

_POSITIVE_VECTORS = [_vector(name) for name in _POSITIVE_VECTOR_NAMES]
_NEGATIVE_VECTORS = [_vector(name) for name in _NEGATIVE_VECTOR_NAMES]

if any(v is None for v in _POSITIVE_VECTORS + _NEGATIVE_VECTORS):
    if os.environ.get("DFO_SPEC_KIT_REQUIRED"):
        raise RuntimeError(
            "DFO_SPEC_KIT_REQUIRED is set but one or more currency-role identity "
            f"vectors were not found under {_spec_kit_dir()} -- the vector suite "
            "checkout is missing, has moved, or dropped a vector this test names"
        )
    pytest.skip(
        "dfo-spec-kit currency-role identity vectors not found -- set "
        "DFO_SPEC_KIT_DIR or check out dfo-spec-kit as a sibling of dfo-apis",
        allow_module_level=True,
    )


@pytest.mark.parametrize("vector", _POSITIVE_VECTORS, ids=_POSITIVE_VECTOR_NAMES)
def test_positive_currency_role_vector_matches_base_identity(vector: dict) -> None:
    model = BaseIdentity.model_validate(vector["input"])
    dumped = model.model_dump(exclude_none=True)
    assert dumped == vector["expected_projection"]
    assert spec_id_from_value(dumped) == vector["spec_id"]


@pytest.mark.parametrize("vector", _NEGATIVE_VECTORS, ids=_NEGATIVE_VECTOR_NAMES)
def test_negative_currency_role_vector_raises_validation_error(vector: dict) -> None:
    with pytest.raises(ValidationError):
        BaseIdentity.model_validate(vector["input"])
