"""Process-level cache of resolved StatVar member groups.

One `MemberEnumerator` serves every surface in a process (`shared_enumerator`), so the
Datasets doors and Search's member walk read one group for one query, and a mint warms
the resubmit that follows it. The instance holds no graph: each call passes its reader
and the key of the generation stamp its reads belong to (`GenerationStamp.key`, which
folds in the catalogue build). Entries key on that stamp key and the query, and expire
by `graph_generation.ttu_from_stamp_key`, so a new graph generation or catalogue build
walks afresh, and an unknown generation keeps its entries for the shorter unknown
lifetime.

With `GRAPH_RELEASE_ORIGIN` on, a call reads the request's graph release
(`graph_release.context.served_origin`), and raises when none is bound. A family the release
holds (`graph_release.origin.ReleaseOrigin.holds_family`) walks the release's graph reader in
place of the caller's, and every entry's key also holds the release id, so a swap walks afresh.
An enumerator following the process's release service (`MemberEnumerator.follow_swaps`, which
`graph_release.service` sets up for `shared_enumerator`) drops the entries of every other
release once a swap serves a new one, and stores none for a replaced release after it.

Each entry weighs one plus its family size, the dcids of every confirmed candidate
before antichain reduction, since a deep root's family runs to about 14,000 dcids.
Concurrent misses on one key run one enumeration. The others wait, each bounded by its
own deadline, then take the group that walk returned, stored or not, or, when it raised,
walk themselves. Only settled `MemberGroup` values are stored for later calls, so a
refusal, a read failure, a deadline crossing or a group with undismissed candidates
stays uncached.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cachetools

from statvar_members.enumerate import (
    CANDIDATE_CAP,
    DEFAULT_MEMBER_CAP,
    MemberGroup,
    MemberReadDeadline,
    SlotBinding,
    _enumerate_resolved,
)
from statvar_members.types import EMPTY_READINGS, AbsentReading, FiveTuple, MemberArcReader

if TYPE_CHECKING:
    from graph_release.service import Release, ReleaseService

logger = logging.getLogger(__name__)

MEMBER_GROUP_CACHE_MAX_WEIGHT = 200_000
"""Total weight the cache holds, one per entry plus one per family dcid. A family dcid
costs about 100 bytes (a 40-character string and its tuple slot), so a full cache holds
about 20 MB of families, 2% of the 1 GiB the service shares with Search's other caches.
That is fourteen deep-root families of 13,800 dcids, or thousands of the median walk's."""

type _CacheKey = tuple[
    str,
    str | None,
    FiveTuple,
    frozenset[tuple[str, SlotBinding]],
    frozenset[tuple[str, AbsentReading]],
    int | None,
    tuple[str, ...],
    frozenset[str],
    frozenset[str],
]


def _weight(group: MemberGroup) -> int:
    return 1 + len(group.family)


def _ttu(key: _CacheKey, group: MemberGroup, now: float) -> float:
    # Importing `graph_generation` loads httpx, which only the `sdmx` and `generation` extras
    # install. A walk reads a live graph client, which needs httpx too, so the import waits
    # for the first stored group and `statvar_members` imports without extras.
    from graph_generation.stamp import ttu_from_stamp_key

    return ttu_from_stamp_key(key, group, now)


@dataclass
class _Flight:
    """One key's walk lock, the calls holding or waiting for it, and the group the last walk
    under it returned, which the calls still waiting read whether or not it was stored."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0
    group: MemberGroup | None = None


class MemberEnumerator:
    """Resolves member groups through a weighted, generation-keyed, single-flight cache.

    `member_cap` is the one reduced-member cap every caller walks under."""

    def __init__(
        self,
        *,
        member_cap: int = DEFAULT_MEMBER_CAP,
        max_weight: int = MEMBER_GROUP_CACHE_MAX_WEIGHT,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self.member_cap = member_cap
        self._lock = threading.Lock()
        self._groups: cachetools.TLRUCache[_CacheKey, MemberGroup] = cachetools.TLRUCache(
            maxsize=max_weight, ttu=_ttu, timer=timer, getsizeof=_weight
        )
        self._flights: dict[_CacheKey, _Flight] = {}
        # The release the followed service serves, `None` while following none.
        self._served_release: str | None = None
        self._unfollow: Callable[[], None] | None = None

    def follow_swaps(self, service: ReleaseService) -> None:
        """Drop every group keyed on a release `service` no longer serves once it swaps, and
        store none for such a release after, in place of any service followed before. `close`
        stops following."""
        self.close()
        with self._lock:
            self._served_release = service.snapshot().release_id
        self._unfollow = service.on_swap(self._swapped)

    def close(self) -> None:
        """Stop following the service `follow_swaps` named."""
        if self._unfollow is not None:
            self._unfollow()
            self._unfollow = None
        with self._lock:
            self._served_release = None

    def _swapped(self, release: Release) -> None:
        with self._lock:
            self._served_release = release.release_id
            for key in [key for key in self._groups if not self._storable(key)]:
                del self._groups[key]

    def _storable(self, key: _CacheKey) -> bool:
        """Whether `key` names no release, or the one the followed service serves. Called
        under `_lock`."""
        release = key[1]
        return release is None or self._served_release in (None, release)

    def enumerate(
        self,
        *,
        graph: MemberArcReader,
        stamp_key: str,
        five_tuple: FiveTuple,
        slots: Mapping[str, SlotBinding],
        readings: Mapping[str, AbsentReading] = EMPTY_READINGS,
        extra_candidates: Sequence[str] = (),
        role_keys: Collection[str] = (),
        open_role_keys: Collection[str] = (),
        subject_ceiling: int | None = None,
        deadline: float | None = None,
        full_read_deadline: float | None = None,
        spec_id: str | None = None,
    ) -> MemberGroup:
        """Return the group `statvar_members.enumerate_members` resolves for the query,
        reading `graph` on a miss.

        `stamp_key` is the `GenerationStamp.key` of `graph`'s reads. The cache key holds it, the
        release id under `GRAPH_RELEASE_ORIGIN` (`_served_walk`), the five-tuple, slots,
        readings, `subject_ceiling`, extra candidates, and bound and open role keys. `deadline`
        bounds the walk's first page and `full_read_deadline`, which defaults to `deadline` and
        stays out of the key, its full read. The later of the two bounds any wait for another
        call's walk of the same key. A read or wait past its bound raises `MemberReadDeadline`.
        `spec_id` names the walk in its log line. Enumeration errors propagate uncached. A group
        with undismissed candidates is returned uncached, so a call after its walk's flight ends
        reads them again; calls waiting on that flight take the group it returned. A group whose
        inconsistent candidates a confirmed root dismisses is settled and stored: its members
        cannot change, and a fresh walk leaves the same candidates past the same re-probe
        budget."""
        graph, release = _served_walk(graph, five_tuple=five_tuple)
        key: _CacheKey = (
            stamp_key,
            release,
            five_tuple,
            frozenset(slots.items()),
            frozenset(readings.items()),
            subject_ceiling,
            tuple(extra_candidates),
            frozenset(role_keys),
            frozenset(open_role_keys),
        )
        cached = self._cached(key)
        if cached is not None:
            return cached
        window = subject_ceiling if subject_ceiling is not None else CANDIDATE_CAP
        wait_until = _later(deadline, full_read_deadline)
        with self._flight(key, five_tuple=five_tuple, window=window, deadline=wait_until) as flight:
            cached = self._cached(key)
            if cached is not None:
                return cached
            if flight.group is not None:
                return flight.group
            group = self._walk(
                graph,
                five_tuple=five_tuple,
                slots=slots,
                readings=readings,
                extra_candidates=extra_candidates,
                role_keys=role_keys,
                open_role_keys=open_role_keys,
                subject_ceiling=subject_ceiling,
                deadline=deadline,
                full_read_deadline=full_read_deadline,
                spec_id=spec_id,
            )
            flight.group = group
            if not group.undismissed_candidates:
                with self._lock:
                    if _weight(group) <= self._groups.maxsize and self._storable(key):
                        self._groups[key] = group
            return group

    def clear(self) -> None:
        """Drop every stored group. A walk in flight still stores its own."""
        with self._lock:
            self._groups.clear()

    def _cached(self, key: _CacheKey) -> MemberGroup | None:
        with self._lock:
            return self._groups.get(key)

    @contextmanager
    def _flight(
        self, key: _CacheKey, *, five_tuple: FiveTuple, window: int, deadline: float | None
    ) -> Iterator[_Flight]:
        """Hold `key`'s walk lock, waiting at most until `deadline` for another call's walk,
        and yield its flight."""
        with self._lock:
            flight = self._flights.setdefault(key, _Flight())
            flight.users += 1
        try:
            timeout = -1.0 if deadline is None else max(0.0, deadline - time.monotonic())
            if not flight.lock.acquire(timeout=timeout):
                raise MemberReadDeadline(
                    five_tuple=five_tuple, candidates_considered=0, candidate_cap=window
                )
            try:
                yield flight
            finally:
                flight.lock.release()
        finally:
            with self._lock:
                flight.users -= 1
                if flight.users == 0:
                    del self._flights[key]

    def _walk(
        self,
        graph: MemberArcReader,
        *,
        five_tuple: FiveTuple,
        slots: Mapping[str, SlotBinding],
        readings: Mapping[str, AbsentReading],
        extra_candidates: Sequence[str],
        role_keys: Collection[str],
        open_role_keys: Collection[str],
        subject_ceiling: int | None,
        deadline: float | None,
        full_read_deadline: float | None,
        spec_id: str | None,
    ) -> MemberGroup:
        started = time.monotonic()
        try:
            group = _enumerate_resolved(
                five_tuple,
                slots=slots,
                graph=graph,
                readings=readings,
                member_cap=self.member_cap,
                extra_candidates=extra_candidates,
                role_keys=role_keys,
                open_role_keys=open_role_keys,
                subject_ceiling=subject_ceiling,
                deadline=deadline,
                full_read_deadline=full_read_deadline,
            )
        except Exception as exc:
            logger.info(
                "statvar_members.walk spec_id=%s five_tuple=%s outcome=%s elapsed_s=%.2f",
                spec_id,
                _five_tuple_label(five_tuple),
                type(exc).__name__,
                time.monotonic() - started,
            )
            raise
        logger.info(
            "statvar_members.walk spec_id=%s five_tuple=%s truncated=%s proven_complete=%s "
            "candidates_considered=%d full_read=%s inconsistent=%d undismissed=%d "
            "outcome=%s elapsed_s=%.2f",
            spec_id,
            _five_tuple_label(five_tuple),
            group.truncated,
            group.proven_complete,
            group.candidates_considered,
            group.full_read,
            len(group.inconsistent_candidates),
            len(group.undismissed_candidates),
            _outcome(group),
            time.monotonic() - started,
        )
        return group


def _served_walk(
    graph: MemberArcReader, *, five_tuple: FiveTuple
) -> tuple[MemberArcReader, str | None]:
    """The reader `five_tuple`'s walk reads and the release its group keys on: `graph` and
    `None` with `GRAPH_RELEASE_ORIGIN` off. With it on, the bound release's id, and a family
    the release holds reads the release's graph reader. Raises
    `graph_release.context.ReleaseNotBound` when no release is bound."""
    # Imported here, as `_ttu` imports `graph_generation`, so `statvar_members` imports
    # without the release and catalogue packages.
    from graph_release.context import current_release, served_origin

    origin = served_origin()
    if origin is None:
        return graph, None
    from origin_catalogue.context import current_catalogue

    release = current_release().release_id
    if origin.holds_family(five_tuple, catalogue=current_catalogue()):
        return origin.reader, release
    return graph, release


def _later(deadline: float | None, full_read_deadline: float | None) -> float | None:
    """The bound a walk with these deadlines can run to, `None` when either is unbounded."""
    if deadline is None:
        return None
    return deadline if full_read_deadline is None else max(deadline, full_read_deadline)


def _five_tuple_label(five_tuple: FiveTuple) -> str:
    """The five arcs joined by `/`, `-` for an absent arc and `?` for an unresolved
    denominator."""
    denominator = (
        "?" if five_tuple.denominator_unresolved else five_tuple.measurement_denominator or "-"
    )
    return "/".join(
        (
            five_tuple.population_type or "-",
            five_tuple.measured_property or "-",
            five_tuple.stat_type or "-",
            five_tuple.measurement_qualifier or "-",
            denominator,
        )
    )


def _outcome(group: MemberGroup) -> str:
    """How a caller reads `group`: `candidate_window` for a truncated page without proof,
    `undismissed` for candidates no read settled, `empty` for no member, else `settled`."""
    if group.truncated and not group.proven_complete:
        return "candidate_window"
    if group.undismissed_candidates:
        return "undismissed"
    if not group.members:
        return "empty"
    return "settled"


_SHARED = MemberEnumerator()


def shared_enumerator() -> MemberEnumerator:
    """The process's one enumerator, which every surface's member walk reads through."""
    return _SHARED
