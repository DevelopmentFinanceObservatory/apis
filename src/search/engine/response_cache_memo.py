"""Bounded in-process memo of resubmitted identities' regions, the response cache's part memo.

A `dataset_resubmit` request re-resolves every part of an answer, so on a website edit each
untouched identity part repeats the resolution it ran on the request before. The memo holds
each identity entry's region under `response_cache_keys.part_memo_key`, and a hit costs a
lookup. Stored regions hold no task and no relaunch closure
(`promote_relaunch.without_relaunches`). A hit binds the closures over the request it serves.
The memo keeps the in-memory objects, so private attributes on a Spec survive a hit.

Each entry is weighed once, when it is stored, by the heap its region holds (`region_bytes`),
and entries expire with the generation stamp they were resolved under. The store runs on the
event loop only.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

import cachetools

import search.engine.config as config  # module-attribute access for runtime-resolved values
from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from graph_generation.stamp import ttu_from_stamp_key
from search.engine.heap_weight import heap_bytes
from search.engine.region.result import RegionResult
from series_coverage.index import CoverageIndex

# Objects a region refers to that the process holds apart from it: classes, modules and
# functions, and the process-wide code, measure and coverage indexes. A region reaches the
# coverage index through `RegionResult.served_filter`. The memo key carries the coverage
# stamp, so a hit reads the same index.
_SHARED = (
    type,
    ModuleType,
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    CodeIndex,
    MeasureIndex,
    CoverageIndex,
)


def region_bytes(region: RegionResult, *, limit: int | None = None) -> int:
    """The heap `region` holds, taken as `sys.getsizeof` summed over every object reachable from
    it once, through `gc.get_referents`, apart from the objects the process holds without it
    (`_SHARED`). It counts the Specs and everything kept beside them for a hit and a
    same-shape fold, such as the served members, read plans, member population and
    `promote_relaunch`. Any other object the region shares, such as a string an index also
    holds, counts in full, so the weight errs high. The walk stops once the total passes
    `limit`, returning a total over it, so weighing a region the memo refuses costs no more
    than weighing one it keeps."""
    return heap_bytes(region, shared=_SHARED, limit=limit)


@dataclass(frozen=True, slots=True)
class _Entry:
    region: RegionResult
    size: int


def _entry_bytes(entry: _Entry) -> int:
    return entry.size


class PartMemo:
    """Least-recently-used memo of stored regions, bounded by their weight in bytes.

    `max_bytes` bounds the bytes held in total and defaults to
    `SEARCH_RESPONSE_CACHE_MEMO_MAX_BYTES`. `0` turns the memo off. A region over
    `max_entry_bytes` (`SEARCH_RESPONSE_CACHE_MEMO_MAX_ENTRY_BYTES`) is served and not kept, so
    one large part cannot evict the rest. Keys are `(stamp key, memo key)`, so each entry
    expires with its generation stamp's lifetime (`ttu_from_stamp_key`)."""

    def __init__(
        self,
        *,
        max_bytes: int | None = None,
        max_entry_bytes: int | None = None,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        settings = config.settings()
        self._max_bytes = settings.response_cache_memo_max_bytes if max_bytes is None else max_bytes
        self._max_entry_bytes = min(
            settings.response_cache_memo_max_entry_bytes
            if max_entry_bytes is None
            else max_entry_bytes,
            self._max_bytes,
        )
        self._store: cachetools.TLRUCache[tuple[str, str], _Entry] = cachetools.TLRUCache(
            maxsize=self._max_bytes, ttu=ttu_from_stamp_key, timer=timer, getsizeof=_entry_bytes
        )
        self._hits = 0
        self._misses = 0
        self._oversized = 0

    @property
    def enabled(self) -> bool:
        return self._max_bytes > 0

    def get(self, stamp_key: str, key: str) -> RegionResult | None:
        """Return the live region under `key` or `None`. A hit becomes most recently used."""
        entry = self._store.get((stamp_key, key))
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        return entry.region

    def put(self, stamp_key: str, key: str, region: RegionResult) -> None:
        """Store `region`, weighed once by `region_bytes`, unless it exceeds the per-entry
        cap."""
        size = region_bytes(region, limit=self._max_entry_bytes)
        if size > self._max_entry_bytes:
            self._oversized += 1
            return
        self._store[(stamp_key, key)] = _Entry(region=region, size=size)

    def stats(self) -> Mapping[str, int]:
        """Hit, miss and oversized counts, the entry count and the bytes held."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "oversized": self._oversized,
            "entries": len(self._store),
            "bytes": int(self._store.currsize),
        }

    def clear(self) -> None:
        """Drop every entry. Run on the event loop, where a refresh thread schedules it."""
        self._store.clear()
