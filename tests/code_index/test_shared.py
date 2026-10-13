"""Test process-wide `get_index` caching and the `reset_index` rebuild hook.
The autouse fixture runs each test on an empty index and restores the process's own after it,
so a later test in the worker does not parse the index again inside its own deadline."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from code_index import CodeIndex, IndexFormatError
from code_index import shared as shared_module
from code_index.shared import get_index, reset_index

from .conftest import FIXTURE_PATH


@pytest.fixture(autouse=True)
def _isolated_index() -> Iterator[None]:
    with shared_module._singleton.isolated():
        yield


def test_returns_the_same_object_on_repeated_calls():
    first = get_index()
    second = get_index()
    assert first is second
    assert isinstance(first, CodeIndex)


def test_reset_makes_the_next_call_rebuild():
    first = get_index()
    reset_index()
    second = get_index()
    assert first is not second
    # The same packaged file yields a fresh object with an equal stamp.
    assert first.stamp == second.stamp


def test_a_missing_packaged_index_raises_naming_the_path(monkeypatch, tmp_path: Path):
    missing = tmp_path / "does-not-exist" / "index.json.gz"
    monkeypatch.setattr(shared_module, "default_index_path", lambda: missing)

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        get_index()


def test_a_failed_first_load_leaves_the_module_retryable(monkeypatch, tmp_path: Path):
    # A failed load leaves `_instance` empty, so restoring the path allows a later retry.
    missing = tmp_path / "does-not-exist" / "index.json.gz"
    real_default_index_path = shared_module.default_index_path
    monkeypatch.setattr(shared_module, "default_index_path", lambda: missing)

    with pytest.raises(FileNotFoundError):
        get_index()

    monkeypatch.setattr(shared_module, "default_index_path", real_default_index_path)
    first = get_index()
    assert isinstance(first, CodeIndex)

    second = get_index()
    assert second is first


def test_a_malformed_packaged_index_raises_index_format_error(monkeypatch, tmp_path: Path):
    payload = json.loads(FIXTURE_PATH.read_text())
    del payload["stamp"]["content_hash"]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload))
    monkeypatch.setattr(shared_module, "default_index_path", lambda: broken)

    with pytest.raises(IndexFormatError):
        get_index()
