"""Revision snapshots and one retry when an upstream revision changes.

`read_generation` calls `generate` at most twice and `reader.read` at most
three times. Supplying `before` skips the initial read, reducing the limit to
two. The retry uses the first attempt's `after` as its `before`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from materialise.errors import UpstreamVersionChangedError


@dataclass(frozen=True)
class SourceRevision:
    source: str
    revision: str


@dataclass(frozen=True)
class RevisionSnapshot:
    graph_revision: str
    source_revisions: tuple[SourceRevision, ...]

    def normalised(self) -> RevisionSnapshot:
        """Sorts by source for equality checks. The endpoint may vary its
        ordering between calls. `mint_content_version` uses a separate sort.
        """
        return RevisionSnapshot(
            graph_revision=self.graph_revision,
            source_revisions=tuple(sorted(self.source_revisions, key=lambda sr: sr.source)),
        )


class RevisionReader(Protocol):
    def read(self) -> RevisionSnapshot: ...


def read_generation[T](
    reader: RevisionReader, generate: Callable[[], T], *, before: RevisionSnapshot | None = None
) -> tuple[T, RevisionSnapshot]:
    """Runs `generate` against a revision snapshot and retries once on drift.

    The first attempt reads before and after generation. A mismatch retries
    using the first `after` as `before`. A second mismatch raises
    `UpstreamVersionChangedError`. `generate` runs at most twice and
    `reader.read()` at most three times. A supplied `before` skips the first
    read, useful when the caller captured it immediately before its graph call.
    """
    if before is None:
        before = reader.read()
    result = generate()
    after = reader.read()
    if before.normalised() == after.normalised():
        return result, after

    result2 = generate()
    after2 = reader.read()
    if after.normalised() == after2.normalised():
        return result2, after2

    raise UpstreamVersionChangedError(
        "the upstream source refreshed mid-materialisation and the retry did not converge"
    )
