"""Scan JSON fixtures for Google, bearer, OpenAI-style, and OAuth secrets."""

import json
import re
from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).parent

_FIXTURE_FILES = sorted(p.name for p in _FIXTURES_DIR.glob("*.json"))

assert _FIXTURE_FILES, f"no fixture files found in {_FIXTURES_DIR}"

_SECRET_PATTERNS = [
    re.compile(r"AIza"),
    re.compile(r"Bearer "),
    re.compile(r"\bsk-"),
    re.compile(r"ya29\."),
]


def _check_value(val, path: str, errors: list[str]) -> None:
    """Recursively scan JSON string values for secret patterns."""
    if isinstance(val, str):
        errors.extend(
            f"{path}: matched pattern {pat.pattern!r} in {val[:80]!r}"
            for pat in _SECRET_PATTERNS
            if pat.search(val)
        )
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_value(v, f"{path}.{k}", errors)
    elif isinstance(val, list):
        for i, item in enumerate(val):
            _check_value(item, f"{path}[{i}]", errors)


@pytest.mark.parametrize("filename", _FIXTURE_FILES)
def test_no_secrets_in_fixture(filename: str) -> None:
    path = _FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"Fixture file not found: {filename}")

    with open(path) as f:
        data = json.load(f)

    errors: list[str] = []
    _check_value(data, filename, errors)

    if errors:
        pytest.fail(f"Secret pattern(s) found in {filename}:\n" + "\n".join(errors))
