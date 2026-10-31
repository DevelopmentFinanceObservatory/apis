"""Checks an example against the search engine.

Each run submits the template filled with a place's name, with the
observations preview on, and passes when the answer is a resolution of one
definite part whose resolved StatVars include the example's StatVar, whose
resolved entities in the place's role are that place alone, and whose
preview has points, every series with points observing that place in that
role, and whose answer departs from the sentence in no way that changes what
it shows. The pin catches a wrong variable with real data. The scope checks
catch data for another place or the reverse direction. The points check
catches an answer with nothing to show. The departure check catches an answer
for a different question than the sentence asked. A run the engine
raised on, cut short at its soft deadline, or answered with a preview whose
read failed is tried once more, since that failure says nothing about the
example. A run the engine refused as input fails the example: the refusal is
deterministic, so a retry would draw it again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from search.catalogue.examples import Example
from search.models import (
    DefinitePart,
    DepartureCode,
    EntityRoleKey,
    ObservationsOmissionReason,
    Resolution,
    ResolveRequest,
    ResolveResponse,
    SpecObservations,
)
from series_coverage.coverage_types import Role

ROLE_KEYS: Final[Mapping[Role | None, EntityRoleKey]] = {
    None: "observationAbout",
    "donor": "origin",
    "recipient": "destination",
}
"""The entity role an example's place fills on the wire, by its coverage
role. An example with no role names the place its data is about."""

TEST_PLACES: Final[tuple[str, ...]] = ("country/TUV", "country/GEO")
"""A small country and one whose name is ambiguous. An example is run at
each of them where its StatVar has data."""

type Resolve = Callable[[ResolveRequest], Awaitable[ResolveResponse]]
"""One buffered resolve, as `search.engine.resolve_async` answers it."""

type RunFailure = Literal[
    "engine_error",
    "engine_refused",
    "timed_out",
    "place_unnamed",
    "not_a_resolution",
    "multiple_parts",
    "no_definite_part",
    "stat_var_not_elected",
    "scope_mismatch",
    "departed",
    "preview_failed",
    "preview_empty",
    "preview_scope_mismatch",
]


DATA_DESCRIBING_DEPARTURES: Final[frozenset[DepartureCode]] = frozenset(
    {
        "modelled_values",
        "group_read_as_members",
        "whole_total_read",
        "whole_total_unpublished",
        "direction_inferred",
        "scope_limited",
    }
)
"""Departures that describe the data the answer shows without changing what
was asked: the values are modelled, a named group was read as its members, a
flow side was read as the total its source publishes for its whole population, or as
every party its source holds there where it publishes no such total, a place named
without a direction was read in the one direction holding data there, a check covered
only some values. An example may carry them."""

REFUSING_DEPARTURES: Final[frozenset[DepartureCode]] = frozenset(
    {
        "closest_match",
        "measure_not_applied",
        "operation_not_applied",
        "breakdown_not_applied",
        "entity_unavailable",
        "member_unavailable",
        "year_unavailable",
        "direction_discarded",
    }
)
"""Departures that say the answer is for a different question than the
sentence asked: another measure, no operation, no breakdown, a place, variable or
year without values, a dropped direction. An answer carrying one fails the example
(`departed`). Together with `DATA_DESCRIBING_DEPARTURES` this is every
`DepartureCode`, so a code added to the vocabulary fails a test until it is
put on a side."""

INFRA_FAILURES: Final[frozenset[RunFailure]] = frozenset(
    {"engine_error", "timed_out", "preview_failed"}
)
"""Failures of the engine rather than of the example: the resolve raised
anything but an input refusal, the answer is a partial cut short at the soft
deadline, or its preview read failed."""

EMPTY_PREVIEW_REASONS: Final[frozenset[ObservationsOmissionReason]] = frozenset(
    {
        "observations_no_points",
        "observations_nothing_to_probe",
        "observations_no_matching_source",
        "observations_no_dispatchable_facet",
        "observations_windowed_scope_unsupported",
        "observations_converted_scope_unsupported",
        "observations_preview_too_large",
    }
)
"""Omission reasons that say the answer's selection has no preview to show,
the same on every try: `preview_empty`. Every other reason, including
any added to the vocabulary, is a read that failed and counts as
`preview_failed`. Reading a failed read as an empty preview drops the line
without a retry and can record a refusal that blocks the next builds; reading
an empty preview as a failed read fails the job loudly until someone looks."""


@dataclass(frozen=True, slots=True)
class FailedRun:
    """One run that failed: where, the text sent, why, and the StatVars the
    answer's definite parts resolved, for the alert."""

    place: str
    text: str | None
    reason: RunFailure
    stat_vars: tuple[str, ...] = ()
    detail: str | None = None


def check_places(example: Example, places_with_data: frozenset[str]) -> tuple[str, ...]:
    """The default place, then each test place where the StatVar has data."""
    tests = (place for place in TEST_PLACES if place in places_with_data)
    return tuple(dict.fromkeys((example.place, *tests)))


async def check_example(
    example: Example, *, places: tuple[str, ...], names: Mapping[str, str], resolve: Resolve
) -> list[FailedRun]:
    """Runs `example` at each of `places`, concurrently, and returns the
    failed runs in `places` order. An empty list means the example passed."""
    results = await asyncio.gather(
        *(_run(example, place=place, names=names, resolve=resolve) for place in places)
    )
    return [failure for failure in results if failure is not None]


async def _run(
    example: Example, *, place: str, names: Mapping[str, str], resolve: Resolve
) -> FailedRun | None:
    name = names.get(place)
    if name is None:
        return FailedRun(place=place, text=None, reason="place_unnamed")
    text = example.text(name)
    request = ResolveRequest.model_validate(
        {
            "input": {"kind": "raw_text", "query": text},
            "response_options": {"include_observations": True},
        }
    )
    failure = await _attempt(example, place=place, text=text, request=request, resolve=resolve)
    if failure is not None and failure.reason in INFRA_FAILURES:
        failure = await _attempt(example, place=place, text=text, request=request, resolve=resolve)
    return failure


async def _attempt(
    example: Example, *, place: str, text: str, request: ResolveRequest, resolve: Resolve
) -> FailedRun | None:
    try:
        response = await resolve(request)
    except Exception as exc:
        # The engine's imports are heavy and the job loads them only for a
        # live run, which has loaded them by the time a resolve raises.
        from search.engine.errors import EngineInputError

        if isinstance(exc, EngineInputError):
            return FailedRun(
                place=place, text=text, reason="engine_refused", detail=exc.code or repr(exc)
            )
        # One engine failure fails this run and leaves the others to finish.
        return FailedRun(place=place, text=text, reason="engine_error", detail=repr(exc))
    if response.diagnostics.timed_out:
        return FailedRun(place=place, text=text, reason="timed_out")
    reason, stat_vars, detail = judge(
        response, stat_var=example.stat_var, place=place, role=ROLE_KEYS[example.role]
    )
    if reason is None:
        return None
    return FailedRun(place=place, text=text, reason=reason, stat_vars=stat_vars, detail=detail)


def judge(
    response: ResolveResponse, *, stat_var: str, place: str, role: EntityRoleKey
) -> tuple[RunFailure | None, tuple[str, ...], str | None]:
    """Why `response` fails the check for `stat_var` at `place` in `role`,
    or `None` when it passes, with the StatVars its definite parts resolved
    and a detail naming the departure codes of a `departed` failure.
    An example's sentence asks for one variable at one place, so an answer
    split into several parts fails, as does one resolving another place in
    `role` (`scope_mismatch`), one departing from the sentence in any way in
    `REFUSING_DEPARTURES` (`departed`), or one previewing a series that does
    not observe `place` in `role` (`preview_scope_mismatch`)."""
    if not isinstance(response, Resolution):
        return "not_a_resolution", (), None
    definite = [part for part in response.parts if isinstance(part, DefinitePart)]
    elected = tuple(
        dict.fromkeys(sv.ref.ref for part in definite for sv in part.spec.resolved.members)
    )
    if len(response.parts) > 1:
        return "multiple_parts", elected, None
    if not definite:
        return "no_definite_part", elected, None
    if stat_var not in elected:
        return "stat_var_not_elected", elected, None
    resolved = definite[0].spec.resolved
    if {entity.ref for entity in resolved.entities if entity.role == role} != {place}:
        return "scope_mismatch", elected, None
    departed = dict.fromkeys(
        departure.code
        for departure in definite[0].departures
        if departure.code in REFUSING_DEPARTURES
    )
    if departed:
        return "departed", elected, ", ".join(departed)
    return _preview_failure(resolved.observations, place=place, role=role), elected, None


def _preview_failure(
    observations: SpecObservations | None, *, place: str, role: EntityRoleKey
) -> RunFailure | None:
    if observations is None:
        return "preview_empty"
    if observations.omitted is not None:
        if observations.omitted.reason in EMPTY_PREVIEW_REASONS:
            return "preview_empty"
        return "preview_failed"
    shown = [series for series in observations.series if series.points]
    if not shown:
        return "preview_empty"
    if not all(
        any(entity.role == role and entity.ref == place for entity in series.entities)
        for series in shown
    ):
        return "preview_scope_mismatch"
    return None


__all__ = [
    "DATA_DESCRIBING_DEPARTURES",
    "EMPTY_PREVIEW_REASONS",
    "INFRA_FAILURES",
    "REFUSING_DEPARTURES",
    "ROLE_KEYS",
    "TEST_PLACES",
    "FailedRun",
    "Resolve",
    "RunFailure",
    "check_example",
    "check_places",
    "judge",
]
