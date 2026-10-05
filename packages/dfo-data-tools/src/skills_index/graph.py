"""Structural graph protocols and result types, independent of `search.*`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class SubjectsCoverage(StrEnum):
    """The three mutually-exclusive end states of one lead dcid's subject
    collection in `SkillsGraph.node_subjects_many`. Exactly one applies
    to a given `SubjectsPage`: a lead that reached its cap did not run
    out of page budget, and one cut off by the page budget never reached
    its cap.

    Structurally identical to
    `search.engine.clients.graph.SubjectsCoverage` by design -- both
    describe the same wire contract -- but declared independently here
    rather than imported, per this module's own docstring. A `SubjectsPage`
    arriving from a real `SkillsGraph` implementation carries that other
    declaration's member, a distinct class from this one, so every
    comparison against a `SubjectsCoverage` member in this package is by
    value (`==`/`!=`), never by identity (`is`/`is not`).
    """

    complete = "complete"
    """The lead's distinct subject set is fully collected: either it was
    read to exhaustion under `max_results`, or the batch's page budget
    ran out with this lead's own set already known-exhausted. `dcids` is
    the real subject set."""

    capped = "capped"
    """A `(max_results + 1)`th distinct subject was confirmed for this
    dcid: `dcids` holds only the first `max_results` of them, a
    known-partial prefix of a subject set confirmed to hold more than
    the cap."""

    budget_exhausted = "budget_exhausted"
    """The batch's page budget ran out before this dcid either capped or
    had its subject set confirmed exhausted. `dcids` holds whatever was
    collected so far, a lower bound on the true set rather than the set
    itself; an empty `dcids` in this state proves nothing, unlike an
    empty `dcids` under `complete`."""


@dataclass(frozen=True, slots=True)
class SubjectsPage:
    """Distinct subject dcids in first-occurrence order and their coverage state."""

    dcids: tuple[str, ...]
    coverage: SubjectsCoverage


@runtime_checkable
class SkillsGraph(Protocol):
    """The two graph methods this package reads through."""

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Fetch named out-arcs. Missing nodes map to `None`. Held nodes include every
        requested property, with `()` for absent arcs."""
        ...

    def node_subjects_many(
        self, dcids: Sequence[str], *, property: str, max_results: int
    ) -> dict[str, SubjectsPage]:
        """Fetch each dcid's incoming subjects separately. Every requested dcid appears
        in the result. Absent nodes or arcs have an empty, complete page. Results
        retain first-occurrence order up to `max_results`. `capped` confirms an
        additional subject exists. `budget_exhausted` leaves only a lower bound,
        even when `dcids` is empty."""
        ...
