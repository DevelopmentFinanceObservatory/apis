"""The `withheld_by_registry` warning on a no-data answer.

A series can come back empty because every facet holding data for it sits under a provenance
the origin catalogue does not admit. Telling that apart from no data at all costs one facet read
with no domain filter, so each request runs it at most once. It runs for the first part that
ends `no_observations` because no member of its elected subject-only shape has admitted data
at the places it names (`NO_MEMBER_HAS_DATA_AT_PLACE`), and it covers that shape's elected
StatVars at those places. Every other no-data reason leaves the answer as it is, since admitted data may exist
there. Role-bearing shapes are skipped because their series are DFO's and always admitted. A
failed read, or one that cannot finish by the request deadline, leaves the answer as it is
and notes the lost read (`degraded_reads.note_degraded_read`), so no cache stores it.

The warning carries an `entity_unavailable` departure on the places, which names the withheld
sources as every surface names a provenance (`refs.provenance_labels`). The name read runs after
the re-read, under the same deadline. One that cannot finish by then leaves the sources out of
the departure, and one that fails names the sources the catalogue lists alone. Either keeps the
warning and notes the lost read.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

from dfo_platform.named import named_capped
from materialise.provenance_doc import ProvenanceNameReader
from search.engine.concurrency import current_scope
from search.engine.config import settings
from search.engine.deadline import _await_bounded
from search.engine.degraded_reads import fail_open_reads
from search.engine.departures import unavailable_entity_facts, warning_key
from search.engine.region.codes import NO_MEMBER_HAS_DATA_AT_PLACE, WITHHELD_BY_REGISTRY
from search.engine.region.refs import provenance_labels
from search.engine.region.result import RegionResult, with_departure_facts
from search.models import Warning

_SUBJECT_ROLE: Final = "observationAbout"

# The re-read ends this long before the request deadline. The fan-out cancels a leg still
# running at the deadline and serves a placeholder in place of its answer, so the leg keeps
# this time to queue the answer the re-read would only have annotated.
_HANDOFF_RESERVE_S: Final = 0.25


@runtime_checkable
class WithheldReader(ProvenanceNameReader, Protocol):
    """The re-read behind `withheld_by_registry` and the labels of what it finds, which
    `LiveGraphClient` implements."""

    def withheld_provenances(
        self, variables: Sequence[str], entities: Sequence[str], *, deadline: float | None = None
    ) -> tuple[str, ...]: ...


async def _source_names(
    graph: WithheldReader, withheld: Sequence[str], *, deadline: float | None
) -> tuple[str, ...]:
    """The names of the `withheld` provenances a name read finds by `deadline` or the
    fail-open budget (`degraded_reads.fail_open_reads`), whichever ends first, empty when that
    read fails or cannot finish by then (`refs.provenance_labels`)."""
    labels: dict[str, str] = {}
    with fail_open_reads(
        "withheld source labels",
        budget_s=settings().fail_open_read_budget_s,
        about={"withheld": withheld},
    ) as bound:
        read_bound = _earlier(bound, deadline)
        read = asyncio.to_thread(provenance_labels, withheld, graph, deadline=read_bound)
        labels = await _await_bounded(read, deadline=read_bound)
    return tuple(dict.fromkeys(labels[dcid] for dcid in withheld if dcid in labels))


def _earlier(bound: float, deadline: float | None) -> float:
    return bound if deadline is None else min(bound, deadline)


def elected_series(result: RegionResult) -> tuple[list[str], list[str]] | None:
    """The elected StatVars and requested places of a subject-only answer that has no data
    because no elected member has admitted data at the places."""
    if (
        result.status != "no_data"
        or result.no_data_reason != "no_observations"
        or result.gate != NO_MEMBER_HAS_DATA_AT_PLACE
        or not result.elected_stat_vars
        or not result.flagged_match
    ):
        return None
    entities = result.flagged_match[0].resolved.entities
    if any(entity.role != _SUBJECT_ROLE for entity in entities):
        return None
    places = list(dict.fromkeys(entity.ref for entity in entities))
    if not places:
        return None
    return [ref.ref for ref in result.elected_stat_vars], places


async def with_withheld_warning(
    result: RegionResult, graph: object, *, deadline: float | None
) -> RegionResult:
    """`result` with `withheld_by_registry` naming the provenances the catalogue withholds at
    its elected series, when it is the request's first eligible no-data answer and there are
    any. A `graph` without the re-read reads nothing. The re-read ends `_HANDOFF_RESERVE_S`
    before `deadline`, the request's, or the fail-open budget after it starts
    (`degraded_reads.fail_open_reads`), whichever comes first, and `result` stands unwarned
    when it cannot finish by then."""
    if not isinstance(graph, WithheldReader):
        return result
    series = elected_series(result)
    if series is None or not current_scope().claim_withheld_read():
        return result
    variables, places = series
    bound = None if deadline is None else deadline - _HANDOFF_RESERVE_S
    withheld: tuple[str, ...] = ()
    with fail_open_reads(
        "withheld re-read",
        budget_s=settings().fail_open_read_budget_s,
        about={"variables": variables, "places": places},
    ) as scope_bound:
        read_bound = _earlier(scope_bound, bound)
        read = asyncio.to_thread(graph.withheld_provenances, variables, places, deadline=read_bound)
        withheld = await _await_bounded(read, deadline=read_bound)
    if not withheld:
        return result
    warning = Warning(
        code=WITHHELD_BY_REGISTRY,
        severity="warn",
        message=(
            f"No admitted source holds data for {named_capped(variables)} at {named_capped(places)}. "
            f"Sources the origin registry does not admit hold data there: {named_capped(withheld)}."
        ),
    )
    warned = dataclasses.replace(result, warnings=(*result.warnings, warning))
    assert result.flagged_match, "an eligible answer has a flagged match"
    facts = unavailable_entity_facts(
        result.flagged_match[0], unserved={(_SUBJECT_ROLE, place) for place in places}
    )
    if not facts:
        return warned
    sources = await _source_names(graph, withheld, deadline=bound)
    # A subject-only answer names its places on one role, so one fact states them.
    (fact,) = facts
    return with_departure_facts(
        warned, {warning_key(warning): dataclasses.replace(fact, sources=sources)}
    )


__all__ = ["WithheldReader", "with_withheld_warning"]
