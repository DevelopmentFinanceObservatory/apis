"""Process-wide code and measure indexes, parsed once from the packaged export.

The cached values need no cleanup. Reset functions exist for tests only.
"""

from __future__ import annotations

from code_index.loader import default_index_path, load_index, load_measure_axes
from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from dfo_platform.singleton import RefCountedSingleton

_singleton: RefCountedSingleton[CodeIndex] = RefCountedSingleton(
    build=lambda: load_index(default_index_path()), label="code index"
)

_measure_singleton: RefCountedSingleton[MeasureIndex] = RefCountedSingleton(
    build=lambda: load_measure_axes(default_index_path()), label="measure index"
)


def get_index() -> CodeIndex:
    """Return the process-wide index, parsing `default_index_path()` on first call.

    Missing files raise `FileNotFoundError`; malformed exports raise
    `IndexFormatError`. Both propagate to fail app startup.
    """
    return _singleton.acquire()


def reset_index() -> None:
    """Test-only: clear the cache so the next call reloads `default_index_path()`."""
    _singleton.force_reset()


def get_measure_index() -> MeasureIndex:
    """Return the process-wide measure index, parsed from `default_index_path()` on first call.

    A deployment with no measure family has an empty `axes` tuple.
    """
    return _measure_singleton.acquire()


def reset_measure_index() -> None:
    """Test-only: clear the cache so the next call reloads `default_index_path()`."""
    _measure_singleton.force_reset()


__all__ = ["get_index", "get_measure_index", "reset_index", "reset_measure_index"]
