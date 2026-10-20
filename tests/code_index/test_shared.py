"""Test process-wide `get_index` caching, the `reset_index` rebuild hook, the `per_index`
memo, and the bound release's indexes, which serve whenever the release carries views.
The autouse fixture runs each test on an empty index and restores the process's own after it,
so a later test in the worker does not parse the index again inside its own deadline. Tests run
with no release bound, so they read the packaged index unless they bind one."""

from __future__ import annotations

import dataclasses
import gc
import json
import re
import weakref
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import pytest

from code_index import CodeIndex, IndexFormatError
from code_index import shared as shared_module
from code_index.loader import load_index
from code_index.shared import get_index, get_measure_index, per_index, reset_index
from graph_release.context import ReleaseNotBound, ReleaseViewsNotBuilt, bound_release
from graph_release.service import Release

from .conftest import FIXTURE_PATH

pytestmark = pytest.mark.unbound_release


@pytest.fixture(autouse=True)
def _isolated_index(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Runs each test on the packaged index, with `GRAPH_RELEASE_COVERAGE` off unless the test
    sets it."""
    monkeypatch.delenv("GRAPH_RELEASE_COVERAGE", raising=False)
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


def test_per_index_builds_once_per_index():
    built: list[int] = []

    @per_index
    def stamp_hash(index: CodeIndex) -> str:
        built.append(id(index))
        return index.stamp.content_hash

    first, second = load_index(FIXTURE_PATH), load_index(FIXTURE_PATH)
    for index in (first, first, second, first, second):
        assert stamp_hash(index) == index.stamp.content_hash

    assert built == [id(first), id(second)]


def test_per_index_keeps_no_replaced_index_alive():
    @per_index
    def stamp_hash(index: CodeIndex) -> str:
        return index.stamp.content_hash

    replaced = load_index(FIXTURE_PATH)
    stamp_hash(replaced)
    held = weakref.ref(replaced)
    del replaced
    gc.collect()

    assert held() is None


@pytest.fixture
def served(request: pytest.FixtureRequest) -> Release:
    """The suite's session release, which carries its views (`tests.conftest`)."""
    return request.getfixturevalue("_packaged_index_release")


@pytest.mark.parametrize("value", [None, "0", "1", " yes "])
def test_a_bound_release_with_views_serves_its_indexes_whatever_the_flag(
    monkeypatch: pytest.MonkeyPatch, served: Release, value: str | None
):
    if value is not None:
        monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", value)

    assert served.views is not None
    with bound_release(served):
        assert get_index() is served.views.code_index
        assert get_measure_index() is served.views.measure_index


@pytest.mark.parametrize("bind_viewless", [False, True], ids=["unbound", "viewless"])
def test_with_the_flag_off_and_no_views_bound_the_indexes_are_the_packaged_ones(
    served: Release, bind_viewless: bool
):
    with bound_release(dataclasses.replace(served, views=None)) if bind_viewless else nullcontext():
        assert get_index() is shared_module._singleton.acquire()
        assert get_measure_index() is shared_module._measure_singleton.acquire()


@pytest.mark.parametrize("read", [get_index, get_measure_index])
def test_with_the_flag_on_an_unbound_read_raises(monkeypatch: pytest.MonkeyPatch, read):
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1")

    with pytest.raises(ReleaseNotBound):
        read()


def test_with_the_flag_on_a_release_without_views_raises(
    monkeypatch: pytest.MonkeyPatch, served: Release
):
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1")

    with (
        bound_release(dataclasses.replace(served, views=None)),
        pytest.raises(ReleaseViewsNotBuilt),
    ):
        get_index()
