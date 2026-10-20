"""Test the `per_index` memo and the bound release's indexes, which `get_index` and
`get_measure_index` serve. Tests run with no release bound unless they bind one."""

from __future__ import annotations

import gc
import weakref

import pytest

from code_index import CodeIndex
from code_index.loader import load_index
from code_index.shared import get_index, get_measure_index, per_index
from graph_release.context import ReleaseNotBound, bound_release
from graph_release.service import Release

from .conftest import FIXTURE_PATH

pytestmark = pytest.mark.unbound_release


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


def test_a_bound_release_serves_its_indexes(served: Release):
    with bound_release(served):
        assert get_index() is served.views.code_index
        assert get_measure_index() is served.views.measure_index


@pytest.mark.parametrize("read", [get_index, get_measure_index])
def test_an_unbound_read_raises(read):
    with pytest.raises(ReleaseNotBound):
        read()
