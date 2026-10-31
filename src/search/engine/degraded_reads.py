"""Record the reads one region's resolution lost, so the response cache stores none of it.

A read that fails, runs past its deadline, or could not start inside a resolution often leaves
the answer standing on less than it would have read, with no warning to say so. So does a
decision that consulted a StatVar's default scope when that scope was a countries fallback
this request alone took (`note_unsettled_scope`, called by `place.ConsultedPresence`). Two
examples of a lost read are the dated
read the unit gate checks a sparse series with and the re-read that names the provenances a
no-data answer is withheld under. `recording_degraded_reads` binds a record for one region's
resolution (`pipeline.resolve_variable`, `resubmit.resolve_spec_resubmit`). Any site in its
extent notes what it lost with `note_degraded_read`, and the resolution marks its region
`RegionResult.degraded`. The record travels in the context, so the worker threads a
resolution starts with `asyncio.to_thread` or `submit_with_context` note into it. A record
bound inside another passes what it holds to the outer one when it closes, unless it is
detached: the UN leg's reads (`phase_intake._un_shapes`) bear on the answer only when the UN
pass runs, so `pipeline._fill_gap` notes them then, and a place-options scope's reads
(`place_options.build_place_options_events`) mark only its options, which the response bundle
does not hold.

The record spans every pass of a resolution, so a fact noted in a pass reaches the region
whichever pass answers. A reading the part may set aside records into a record of its own,
and the outcome that compares readings hands each record on with `carry` under one rule: its
lost reads always, and the scopes it consulted only when the outcome's rule compared that
reading's result. A party reading withdrawn for want of data hands on both
(`pipeline.resolve_variable`), and one withdrawn for another reason its lost reads alone. A
re-election stand-in hands on both when the choice compared it, the winner and the equivalents
ahead of it (`phase_place_materialise._reelect_zero_observation_sibling`), and every
alternate admission reads hands on both (`candidate_admission`).

Only a read an answer depends on is noted. A fact left unknown because no read was asked for
is not: a one-shape bind menu skips the availability read over a named set's places
(`phase_rank_bind._phase_prune_and_rank`), so that shape's availability there is unknown,
and its region is not degraded.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class DegradedReads:
    """The reads a resolution lost, by name, in the order they were noted, and the StatVars
    whose unsettled default scope one of its decisions consulted."""

    reads: list[str] = field(default_factory=list)
    unsettled: set[str] = field(default_factory=set)

    @property
    def degraded(self) -> bool:
        return bool(self.reads or self.unsettled)

    def merge_into(self, other: DegradedReads) -> None:
        """Add what this record holds to `other`."""
        other.reads.extend(self.reads)
        other.unsettled.update(self.unsettled)


_RECORD: ContextVar[DegradedReads | None] = ContextVar("search_degraded_reads", default=None)


@contextmanager
def recording_degraded_reads(
    *, detached: bool | None = None, into: DegradedReads | None = None
) -> Iterator[DegradedReads]:
    """Collect, for the body of the `with` block, the reads it swallows. A `detached` record
    keeps them from the outer record when it closes. Given `into`, the block records into
    that record, which belongs to one reading recorded over several blocks, and is detached:
    its owner carries it (`carry`), so `into` with `detached=False` raises `TypeError`."""
    if into is not None and detached is False:
        raise TypeError("a record given as `into` is detached; its owner carries it")
    outer = _RECORD.get()
    record = DegradedReads() if into is None else into
    detached = into is not None or bool(detached)
    token = _RECORD.set(record)
    try:
        yield record
    finally:
        _RECORD.reset(token)
        if outer is not None and not detached:
            record.merge_into(outer)


def note_degraded_read(read: str) -> None:
    """Note that `read` was lost or could not start, and the answer stands without it.
    Outside a record it notes nothing."""
    record = _RECORD.get()
    if record is not None:
        record.reads.append(read)


def note_unsettled_scope(stat_var: str) -> None:
    """Note that a decision consulted `stat_var`'s default scope while it was a countries
    fallback this request alone took, which a later request may read over every place with
    data. Outside a record it notes nothing."""
    record = _RECORD.get()
    if record is not None:
        record.unsettled.add(stat_var)


def carry(record: DegradedReads, *, consulted: bool = True) -> None:
    """Note in the bound record what a detached `record` held: its lost reads, and the scopes
    it consulted when `consulted`, as a reading the answer stands on does. A reading the
    answer passed over hands on its lost reads alone. Outside a record it notes nothing."""
    outer = _RECORD.get()
    if outer is None:
        return
    if consulted:
        record.merge_into(outer)
    else:
        outer.reads.extend(record.reads)
