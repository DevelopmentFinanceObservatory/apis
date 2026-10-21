"""The code and measure indexes a request reads, and `per_index`, the memo for values derived
from one code index.

Both indexes are the serving views of the graph release bound to the current context
(`graph_release.service.ReleaseViews`), so a request reads its release's indexes and none
outlives the release. A producer or script that reads the packaged export calls
`code_index.loader.load_index(default_index_path())` itself.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from graph_release.context import current_views


def get_index() -> CodeIndex:
    """Return the bound release's code index. Raises `graph_release.context.ReleaseNotBound`
    when no release is bound."""
    return current_views().code_index


def get_measure_index() -> MeasureIndex:
    """Return the bound release's measure index. Raises as `get_index` does.

    A deployment with no measure family has an empty `axes` tuple.
    """
    return current_views().measure_index


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


__all__ = ["get_index", "get_measure_index", "per_index"]
