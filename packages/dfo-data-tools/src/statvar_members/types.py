"""Query and graph-read types for StatVar member enumeration.

`FiveTuple` mirrors the five identity-shape fields. Missing fields mean absent
arcs. `measured_property` anchors the reverse query. The frozen tuple also keys
the member-group cache. Graph clients satisfy the `MemberArcReader` and `SubjectPage`
protocols structurally, and a reader whose reads of some nodes drop no arc also satisfies
`CompleteArcReads`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, runtime_checkable

AbsentReading = Literal["not_declared", "aggregate"]
"""`not_declared` treats a missing constraint as distinct. `aggregate` covers every
value. Unspecified properties default to `aggregate`. Defined here to avoid an
import dependency on `code_index`."""

EMPTY_READINGS: Final[Mapping[str, AbsentReading]] = MappingProxyType({})
"""Default readings treat every unspecified property as `aggregate`. The immutable
mapping is safe as a shared function default. Callers with `not_declared`
properties must pass an explicit registry-backed mapping."""

UNRESOLVED_NODE: Final[str] = "\x00unresolved-node"
"""Marker for an unresolvable arc target. The NUL prefix is reserved outside valid
DCIDs and arc values. This package defines it locally.
`dc_graph.GraphClient` returns the same marker."""


@dataclass(frozen=True)
class FiveTuple:
    population_type: str | None = None
    measured_property: str | None = None
    stat_type: str | None = None
    measurement_qualifier: str | None = None
    measurement_denominator: str | None = None
    denominator_unresolved: bool = False
    """True when the ask carries an unresolved measurement-denominator arc --
    a distinct state from `measurement_denominator is None`, which means the
    ask carries no denominator arc at all."""


class SubjectPage(Protocol):
    # Properties keep the protocol read-only, so frozen dataclass fields satisfy it.
    @property
    def dcids(self) -> tuple[str, ...]: ...
    @property
    def truncated(self) -> bool: ...


class MemberArcReader(Protocol):
    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        """Return distinct subjects in first-occurrence order. `max_results` and
        `truncated` count subjects after duplicate provenance arcs collapse.
        `filters` matches each subject's out-arcs. One absolute `time.monotonic()`
        `deadline` bounds every page. Transport errors raise."""
        ...

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Fetch requested out-arcs in one batch. Missing nodes map to `None`. Held nodes
        include every requested property, with distinct values in first-seen order.
        An unresolvable target contributes `UNRESOLVED_NODE`, a deduplicated
        presence signal distinct from an absent arc. Transport errors raise.

        `individual_probe_budget` limits follow-up reads for dcids unresolved after
        a batch retry. `None` probes all of them. Unprobed dcids map to `None`."""
        ...


@runtime_checkable
class CompleteArcReads(Protocol):
    """A `MemberArcReader` that knows which nodes it reads in full."""

    def arcs_complete(self, dcid: str) -> bool:
        """Whether every read of `dcid`'s out-arcs returns each arc the node holds. A batch
        read of a live graph can drop an arc, so the walk re-reads a candidate it rejects for
        an absent arc. An absent arc in a complete read is a verdict, and the walk trusts it."""
        ...
