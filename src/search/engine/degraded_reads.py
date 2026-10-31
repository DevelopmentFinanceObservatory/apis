"""Record the reads one region's resolution swallowed, so the response cache stores none of it.

A read that fails or runs past its deadline inside a resolution often leaves the answer
standing on less than it would have read, with no warning to say so. Two examples are the dated
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

Only a read that was made and lost is noted. A fact left unknown because no read was made for
it is not: a one-shape bind menu skips the availability read over a named set's places
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
    """The reads a resolution swallowed, by name, in the order they were noted."""

    reads: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.reads)


_RECORD: ContextVar[DegradedReads | None] = ContextVar("search_degraded_reads", default=None)


@contextmanager
def recording_degraded_reads(*, detached: bool = False) -> Iterator[DegradedReads]:
    """Collect, for the body of the `with` block, the reads it swallows. A `detached` record
    keeps them from the outer record when it closes."""
    outer = _RECORD.get()
    record = DegradedReads()
    token = _RECORD.set(record)
    try:
        yield record
    finally:
        _RECORD.reset(token)
        if outer is not None and not detached:
            outer.reads.extend(record.reads)


def note_degraded_read(read: str) -> None:
    """Note that `read` failed or ran past its deadline and the answer stands without it.
    Outside a record it notes nothing."""
    record = _RECORD.get()
    if record is not None:
        record.reads.append(read)
