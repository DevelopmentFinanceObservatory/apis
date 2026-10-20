"""The code and measure indexes a request reads, and `per_index`, the memo for values derived
from one code index.

When the graph release bound to the current context carries views (`graph_release.service`
builds them under `GRAPH_RELEASE_COVERAGE`), they are its views, so a request reads its
release's indexes and none outlives the release. With no such views and the flag off, they are
process-wide, parsed once from the packaged export. With no such views and the flag on, reading
them raises.

The cached values need no cleanup. Reset functions exist for tests only.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

from code_index.loader import default_index_path, load_index, load_measure_axes
from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from dfo_platform.singleton import RefCountedSingleton
from graph_release.context import bound_release_or_none, coverage_views_enabled, current_views

if TYPE_CHECKING:
    # Annotation-only, so reading an index imports no part reader.
    from graph_release.service import ReleaseViews

_singleton: RefCountedSingleton[CodeIndex] = RefCountedSingleton(
    build=lambda: load_index(default_index_path()), label="code index"
)

_measure_singleton: RefCountedSingleton[MeasureIndex] = RefCountedSingleton(
    build=lambda: load_measure_axes(default_index_path()), label="measure index"
)


def _bound_views() -> ReleaseViews | None:
    """The bound release's views, or `None` when none carries any and `GRAPH_RELEASE_COVERAGE` is
    off. With the flag on, raises where `graph_release.context.current_views` does."""
    release = bound_release_or_none()
    if release is not None and release.views is not None:
        return release.views
    return current_views() if coverage_views_enabled() else None


def get_index() -> CodeIndex:
    """Return the bound release's code index when it carries views, else, with
    `GRAPH_RELEASE_COVERAGE` off, the process-wide index, parsing `default_index_path()` on
    first call.

    With the flag on, raises `graph_release.context.ReleaseNotBound` when no release is bound
    and `ReleaseViewsNotBuilt` when the bound one carries no views. With it off, missing files
    raise `FileNotFoundError` and malformed exports raise `IndexFormatError`. Each propagates
    to fail app startup.
    """
    views = _bound_views()
    return views.code_index if views is not None else _singleton.acquire()


def reset_index() -> None:
    """Test-only: clear the cache so the next call reloads `default_index_path()`."""
    _singleton.force_reset()


def get_measure_index() -> MeasureIndex:
    """Return the bound release's measure index when it carries views, else, with
    `GRAPH_RELEASE_COVERAGE` off, the process-wide one, parsed from `default_index_path()` on
    first call. Raises as `get_index` does.

    A deployment with no measure family has an empty `axes` tuple.
    """
    views = _bound_views()
    return views.measure_index if views is not None else _measure_singleton.acquire()


def reset_measure_index() -> None:
    """Test-only: clear the cache so the next call reloads `default_index_path()`."""
    _measure_singleton.force_reset()


def per_index[T](build: Callable[[CodeIndex], T]) -> Callable[[CodeIndex], T]:
    """Memoise `build` on the code index it is called with, for as long as that index lives.

    The index a request reads can change between requests, so a value derived from it is
    cached per index. Entries are keyed on the index object, as
    `code_index.members` keys its views, since in-memory indexes share placeholder content
    hashes, and each entry is dropped when its index is collected. `build` must return a value
    that does not hold its index, or the entry would keep the index alive.
    """
    values: dict[int, T] = {}

    def cached(index: CodeIndex) -> T:
        key = id(index)
        if key not in values:
            values[key] = build(index)
            weakref.finalize(index, values.pop, key, None)
        return values[key]

    return cached


__all__ = ["get_index", "get_measure_index", "per_index", "reset_index", "reset_measure_index"]
