"""Request-scoped TTL cache for resolved StatVar member groups.

Graph revisions are unavailable, so each door request gets its own cache; sharing one across
requests can serve a group enumerated before a graph write. Entries
are keyed by query, expire after 60 seconds, and are swept on writes. The lock
protects the cache only. Concurrent misses may enumerate the same key twice."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence

from statvar_members.enumerate import MemberGroup, SlotBinding, _enumerate_resolved
from statvar_members.types import EMPTY_READINGS, AbsentReading, FiveTuple, MemberArcReader

_MEMBER_GROUP_TTL_SECONDS = 60.0
"""Expire entries after 60 seconds, including within long-running requests."""

_MEMBER_GROUP_CACHE_MAX_ENTRIES = 256


class MemberEnumerator:
    """Owns one request's member-group cache."""

    def __init__(
        self,
        *,
        graph: MemberArcReader,
        clock: Callable[[], float] = time.monotonic,
        member_group_ttl: float = _MEMBER_GROUP_TTL_SECONDS,
        member_group_cache_max_entries: int = _MEMBER_GROUP_CACHE_MAX_ENTRIES,
    ) -> None:
        self._graph = graph
        self._clock = clock
        self._member_group_ttl = member_group_ttl
        self._member_group_cache_max_entries = member_group_cache_max_entries
        self._member_group_lock = threading.Lock()
        self._member_groups: dict[
            tuple[FiveTuple, frozenset, frozenset, int, tuple[str, ...], frozenset[str]],
            tuple[float, MemberGroup],
        ] = {}

    def enumerate(
        self,
        *,
        five_tuple: FiveTuple,
        slots: Mapping[str, SlotBinding],
        readings: Mapping[str, AbsentReading] = EMPTY_READINGS,
        member_cap: int = 700,
        extra_candidates: Sequence[str] = (),
        role_keys: Collection[str] = (),
    ) -> MemberGroup:
        """Return a cached group when its full query key matches. Enumeration errors
        propagate uncached. The key includes readings, cap, slots, extra candidates, and
        role keys."""
        cache_key = (
            five_tuple,
            frozenset(slots.items()),
            frozenset(readings.items()),
            member_cap,
            tuple(extra_candidates),
            frozenset(role_keys),
        )

        cached = self._cached_member_group(cache_key)
        if cached is not None:
            return cached

        group = _enumerate_resolved(
            five_tuple,
            slots=slots,
            graph=self._graph,
            readings=readings,
            member_cap=member_cap,
            extra_candidates=extra_candidates,
            role_keys=role_keys,
        )
        self._store_member_group(cache_key, group)
        return group

    def _cached_member_group(
        self,
        cache_key: tuple[FiveTuple, frozenset, frozenset, int, tuple[str, ...], frozenset[str]],
    ) -> MemberGroup | None:
        with self._member_group_lock:
            entry = self._member_groups.get(cache_key)
        if entry is None:
            return None
        cached_at, group = entry
        if self._clock() - cached_at >= self._member_group_ttl:
            return None
        return group

    def _store_member_group(
        self,
        cache_key: tuple[FiveTuple, frozenset, frozenset, int, tuple[str, ...], frozenset[str]],
        group: MemberGroup,
    ) -> None:
        now = self._clock()
        with self._member_group_lock:
            self._member_groups[cache_key] = (now, group)
            stale_before = now - self._member_group_ttl
            for key, (cached_at, _) in list(self._member_groups.items()):
                if cached_at <= stale_before:
                    del self._member_groups[key]
            if len(self._member_groups) > self._member_group_cache_max_entries:
                oldest_key = min(self._member_groups, key=lambda key: self._member_groups[key][0])
                del self._member_groups[oldest_key]
