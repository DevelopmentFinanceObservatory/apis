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
import logging
from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

import dc_graph
from dfo_platform.named import named_capped
from materialise.provenance_doc import ProvenanceNameReader
from search.engine.concurrency import current_scope
from search.engine.deadline import _await_bounded
from search.engine.degraded_reads import note_degraded_read
from search.engine.departures import unavailable_entity_facts, warning_key
from search.engine.errors import GraphInfraError
from search.engine.region.codes import NO_MEMBER_HAS_DATA_AT_PLACE, WITHHELD_BY_REGISTRY
from search.engine.region.refs import provenance_labels
from search.engine.region.result import RegionResult, with_departure_facts
from search.models import Warning

logger = logging.getLogger(__name__)

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
    """The names of the `withheld` provenances a name read finds by `deadline`, empty when
    that read cannot finish by then (`refs.provenance_labels`)."""
    read = asyncio.to_thread(provenance_labels, withheld, graph, deadline=deadline)
    try:
        labels = await (read if deadline is None else _await_bounded(read, deadline=deadline))
    except TimeoutError:
        logger.info(
            "withheld_by_registry source labels skipped: they could not finish by the request "
            "deadline for %r",
            withheld,
        )
        note_degraded_read("withheld source labels")
        return ()
    except GraphInfraError:
        logger.warning("withheld_by_registry source labels failed for %r", withheld, exc_info=True)
        note_degraded_read("withheld source labels")
        return ()
    return tuple(dict.fromkeys(labels[dcid] for dcid in withheld if dcid in labels))


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
    before `deadline`, the request's, and `result` stands unwarned when it cannot finish by
    then."""
    if not isinstance(graph, WithheldReader):
        return result
    series = elected_series(result)
    if series is None or not current_scope().claim_withheld_read():
        return result
    variables, places = series
    bound = None if deadline is None else deadline - _HANDOFF_RESERVE_S
    read = asyncio.to_thread(graph.withheld_provenances, variables, places, deadline=bound)
    try:
        withheld = await (read if bound is None else _await_bounded(read, deadline=bound))
    except (TimeoutError, dc_graph.GraphDeadlineExceededError):
        logger.info(
            "withheld_by_registry re-read skipped: it could not finish by the request deadline "
            "for variables=%r places=%r",
            variables,
            places,
        )
        note_degraded_read("withheld re-read")
        return result
    except GraphInfraError:
        logger.warning(
            "withheld_by_registry re-read failed for variables=%r places=%r",
            variables,
            places,
            exc_info=True,
        )
        note_degraded_read("withheld re-read")
        return result
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
