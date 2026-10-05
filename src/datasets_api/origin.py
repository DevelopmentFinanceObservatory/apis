"""What the origin catalogue adds to a mint or preview: the build a manifest's trace records, the
`origin_catalogue_stale`, `withheld_by_registry` and `un_system_unavailable` warnings, and the
`excluded_by_registry` refusal.

Each request binds one catalogue (`origin_catalogue.current_catalogue`), so a manifest's trace,
its citation sources and its warnings name one build.

The graph withholds a facet whose provenance the catalogue does not admit. Finding what it
withheld costs one facet read with no domain filter, so the re-read runs at most once per mint or
preview, only when a requested series left an entity without rows, and under the request's
deadline and `WITHHELD_READ_MAX_BYTES`. It names an entity only where the catalogue admits none of the
series' facets there (`WithheldReader.withheld_at`). A failed re-read is logged and leaves the
answer as it was, since the answer itself was read in full.

- When one member's facet probe is empty, materialisation refuses `default_facet_undefined`
  naming that member. `origin_reads` re-reads its variable at its entities and, when a withheld
  provenance holds data there, refuses `excluded_by_registry` naming it instead.
- When some subject entities came back with no rows, `withheld_warning` re-reads the first such
  request's variable at those entities and warns `withheld_by_registry`.

A subject-only request that binds no entity covers its default place scope, every place where it
holds data or its countries with data (`materialise.place_scope`). The warning takes those places
from the materialisation's result, which carries the ones its facet probe read. The refusal has
no result and reads the scope again (`_refusal_scope`).

`origin_reads` also records what UN observation reads could not serve: places the UN front door
refuses, and UN reads that failed inside a read that served our variables. A facet probe made
before the record opened carries its own record, and the materialisation that reuses the probe
adds it to this one, so the record covers the probe at every door. `un_unavailable_warnings`
names those places as `un_system_unavailable`. A UN member with no candidate at such a place
refuses `upstream_error` (`UnSystemUnavailable`), since the empty probe says nothing about data
there. A member that binds no entity covers every country, including the few the UN front door
refuses on every read, so a retry would meet the same refusal. It refuses `upstream_error` only
when a UN read failed, and otherwise `default_facet_undefined` with a detail naming the refused
places its warning names.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Final, Protocol, runtime_checkable

from datasets_api.codes_materialise import ExcludedByRegistry, UnSystemUnavailable, named_capped
from datasets_api.manifest_models import WarningModel
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dc_graph import GraphCapExceededError, GraphReadError
from dc_sdmx import ContainedIn
from materialise import MaterialisationRequest
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from materialise.facets import binds_no_place
from materialise.graph import GraphReader
from materialise.place_scope import (
    DEFAULT_SCOPE_PARENT,
    AvailabilityReader,
    earth_countries,
    read_default_presence,
)
from materialise.window import Coverage
from origin_catalogue import UnBlockedPlaces, current_catalogue, recording_un_blocked_places
from origin_catalogue.graph import is_un_dcid

logger = logging.getLogger(__name__)

_SUBJECT_ROLE: Final[str] = "observationAbout"
_STAT_VAR_REF_PREFIX: Final[str] = "dcid:"

WITHHELD_READ_MAX_BYTES: Final[int] = 16 * 1024 * 1024
"""Caps the unfiltered re-read's response. It reads one variable's facets, without dates, at
the entities of one request."""


@runtime_checkable
class WithheldReader(Protocol):
    """The re-read behind `withheld_by_registry` and `excluded_by_registry`, which
    `origin_catalogue.graph.OriginGraphClient` implements."""

    def withheld_at(
        self,
        variables: Sequence[str],
        entities: Sequence[str],
        *,
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> Mapping[str, tuple[str, ...]]:
        """Per entity, the provenances the catalogue does not admit that hold data for a
        variable there, for entities where it admits none of that variable's facets."""
        ...


def catalogue_build_id() -> str:
    """The build id of the catalogue bound to this request, recorded in each new manifest's trace."""
    return current_catalogue().build_id


def stale_warnings() -> tuple[WarningModel, ...]:
    """`origin_catalogue_stale` when the bound catalogue is older than `origin_catalogue.STALE_AFTER`."""
    catalogue = current_catalogue()
    if not catalogue.is_stale():
        return ()
    _title, detail, severity = DATASETS_WARNING_CATALOGUE["origin_catalogue_stale"]
    return (
        WarningModel(
            code="origin_catalogue_stale",
            severity=severity,
            message=f"{detail} This answer admits data by build {catalogue.build_id!r}, built "
            f"{catalogue.built_at.date().isoformat()}.",
        ),
    )


@contextmanager
def origin_reads(
    graph: GraphReader, *, availability: AvailabilityReader, deadline: float | None
) -> Iterator[UnBlockedPlaces]:
    """Wrap one materialisation. Yields the record of what UN reads could not serve, for
    `un_unavailable_warnings`.

    A `default_facet_undefined` refusal naming a subject-only member is resolved against that
    member alone, at its bound entities or, when it binds none, at the places of its default
    place scope (`_refusal_scope`).
    For a UN variable the UN could not serve at some of those places it becomes
    `UnSystemUnavailable`: a refused place or a failed read counts for a member that names its
    places, and a failed read naming its variable alone for a member that binds none, so that
    check reads no scope.
    Otherwise it becomes `excluded_by_registry` when a withheld provenance holds data for the
    member's variable there. A member that binds none and stays `default_facet_undefined` has
    the places the UN front door refused during the materialisation added to the refusal's
    detail. A `facet_not_available` refusal from a group whose members all came back
    empty (`FacetNotAvailableError.empty_members`) becomes `UnSystemUnavailable` by the same
    rule for its first member the UN could not serve, and otherwise passes through. Any other
    refusal passes through.

    A member that binds none re-reads its scope through `availability` for the last two
    checks. A failed re-read raises the read's error, which the doors refuse as a
    retryable `upstream_error`."""
    with recording_un_blocked_places() as blocked:
        try:
            yield blocked
        except DefaultFacetUndefinedError as exc:
            request = exc.request
            if request is None:
                raise
            unbound = binds_no_place(request.role_bindings, request.declared_roles)
            if unavailable := _un_unavailable(blocked, request):
                raise unavailable from exc
            scope = _refusal_scope(request, graph, availability=availability, deadline=deadline)
            if scope is None:
                raise
            variable, entities = scope
            if withheld := _withheld_at(graph, variable, entities, deadline=deadline):
                provenances = tuple(
                    dict.fromkeys(provenance for held in withheld.values() for provenance in held)
                )
                raise ExcludedByRegistry(variables=(variable,), provenances=provenances) from exc
            if unbound and (refused := _un_unserved(blocked, variable=variable, failed=False)):
                raise DefaultFacetUndefinedError(
                    f"{exc}. No UN data was read for {named_capped(refused)}, which the UN "
                    "System Data Commons front door refuses",
                    pointer=exc.pointer,
                    issues=exc.issues,
                    request=request,
                ) from exc
            raise
        except FacetNotAvailableError as exc:
            for request in exc.empty_members:
                if unavailable := _un_unavailable(blocked, request):
                    raise unavailable from exc
            raise


def _un_unavailable(
    blocked: UnBlockedPlaces, request: MaterialisationRequest
) -> UnSystemUnavailable | None:
    """`UnSystemUnavailable` for a member with no candidate when the UN could not serve it:
    a failed read naming its variable for a member that binds no place, and a refused place or
    a failed read among its entities for one bound to a plain entity list. `None` otherwise,
    and for a role-bearing member or a `ContainedIn` scope."""
    variable = request.stat_var.removeprefix(_STAT_VAR_REF_PREFIX)
    if binds_no_place(request.role_bindings, request.declared_roles):
        places = _un_unserved(blocked, variable=variable, refused=False)
    elif (bound := _subject_binding(request)) is not None:
        places = _un_unserved(blocked, variable=variable, entities=bound[1])
    else:
        places = ()
    return UnSystemUnavailable(variable=variable, places=places) if places else None


def un_unavailable_warnings(
    blocked: UnBlockedPlaces, *, request: MaterialisationRequest | None = None
) -> tuple[WarningModel, ...]:
    """`un_system_unavailable` naming the places `origin_reads` recorded, when there are any.

    With `request`, the warning is that member's alone: none for a variable outside the UN, and
    for a plain-bound member only the recorded places among its entities."""
    if request is None:
        places = _un_unserved(blocked)
    else:
        bound = _subject_binding(request)
        places = _un_unserved(
            blocked,
            variable=request.stat_var.removeprefix(_STAT_VAR_REF_PREFIX),
            entities=None if bound is None else bound[1],
        )
    if not places:
        return ()
    _title, detail, severity = DATASETS_WARNING_CATALOGUE["un_system_unavailable"]
    return (
        WarningModel(
            code="un_system_unavailable",
            severity=severity,
            message=f"{detail} No UN data was read for {named_capped(places)}.",
        ),
    )


def withheld_warning(
    graph: object,
    served: Sequence[tuple[MaterialisationRequest, Coverage]],
    *,
    default_scope: Sequence[str] = (),
    deadline: float | None,
) -> tuple[int, WarningModel] | None:
    """`withheld_by_registry` for the first request in `served` that left a subject entity
    without rows, with that request's index, when that entity has no admitted data and the
    catalogue withholds a provenance holding data there. At most one facet read runs.

    `served` pairs each request with the coverage of the result that served it. A request that
    binds no entity is checked at `default_scope`, the places of the default place scope its
    result carries (`MaterialisationResult.default_scope`), so this reads no scope. A role-bearing
    request and a `ContainedIn` scope are skipped: role-bearing series are DFO's and always
    admitted, and a scope's members are unknown here. A `graph` without the re-read reads
    nothing.
    """
    if not isinstance(graph, WithheldReader):
        return None
    for index, (request, coverage) in enumerate(served):
        if binds_no_place(request.role_bindings, request.declared_roles):
            scope = (request.stat_var.removeprefix(_STAT_VAR_REF_PREFIX), list(default_scope))
        else:
            scope = _subject_binding(request)
        if scope is None:
            continue
        variable, entities = scope
        present = coverage.entity_refs.get(_SUBJECT_ROLE, frozenset())
        unserved = [entity for entity in entities if entity not in present]
        if not unserved:
            continue
        withheld = _withheld_at(graph, variable, unserved, deadline=deadline)
        return (index, _withheld_warning(variable, withheld)) if withheld else None
    return None


def withheld_warnings(
    graph: object,
    served: Sequence[tuple[MaterialisationRequest, Coverage]],
    *,
    default_scope: Sequence[str] = (),
    deadline: float | None,
) -> tuple[WarningModel, ...]:
    """`withheld_warning`'s warning alone, for a manifest that covers every request in `served`."""
    found = withheld_warning(graph, served, default_scope=default_scope, deadline=deadline)
    return () if found is None else (found[1],)


def _withheld_at(
    graph: object, variable: str, entities: Sequence[str], *, deadline: float | None
) -> Mapping[str, tuple[str, ...]]:
    """The re-read for one variable, or nothing when `graph` has no re-read or the read fails."""
    if not isinstance(graph, WithheldReader):
        return {}
    try:
        return graph.withheld_at(
            [variable], entities, deadline=deadline, max_bytes=WITHHELD_READ_MAX_BYTES
        )
    except (GraphReadError, GraphCapExceededError) as exc:
        logger.warning(
            "Withheld-provenance re-read for %s at %d entities failed; the answer carries no "
            "registry explanation: %s",
            variable,
            len(entities),
            exc,
        )
        return {}


def _subject_binding(request: MaterialisationRequest) -> tuple[str, list[str]] | None:
    """The bare variable and distinct bound entities of a subject-only request bound to a plain
    entity list, or `None` for a role-bearing request or a `ContainedIn` scope."""
    bound = request.role_bindings.get(_SUBJECT_ROLE)
    if (
        request.declared_roles != (_SUBJECT_ROLE,)
        or bound is None
        or isinstance(bound, ContainedIn)
    ):
        return None
    return request.stat_var.removeprefix(_STAT_VAR_REF_PREFIX), list(dict.fromkeys(bound))


def _refusal_scope(
    request: MaterialisationRequest,
    graph: GraphReader,
    *,
    availability: AvailabilityReader,
    deadline: float | None,
) -> tuple[str, list[str]] | None:
    """The bare variable and the entities a refused subject-only request covers: its plain
    bound list, or for a request that binds none the places its default place scope could
    hold. `None` for a role-bearing request and a `ContainedIn` scope.

    A refusal returns no result to carry the places its probe read, so this reads the scope
    again. Every place with data counts every provenance, so it holds the withheld data's
    places. The countries fallback counts admitted data only, so a variable whose data is all
    withheld has an empty one, and the re-read then covers Earth's countries and Earth."""
    if not binds_no_place(request.role_bindings, request.declared_roles):
        return _subject_binding(request)
    variable = request.stat_var.removeprefix(_STAT_VAR_REF_PREFIX)
    presence = read_default_presence(
        graph=graph,
        availability=availability,
        stat_vars=[variable],
        deadline=deadline,
        settle=True,
    )
    scope = presence.scope(variable)
    if scope.every_place:
        return variable, list(scope.places)
    # A memoised scope comes without the countries read.
    countries = presence.countries or earth_countries(graph, deadline=deadline)
    return variable, [*countries, DEFAULT_SCOPE_PARENT]


def _un_unserved(
    blocked: UnBlockedPlaces,
    *,
    variable: str | None = None,
    entities: Sequence[str] | None = None,
    refused: bool = True,
    failed: bool = True,
) -> tuple[str, ...]:
    """The places the UN could not serve, in first-seen order: those its front door refused,
    then those of failed UN reads. With `variable`, none unless it is a UN variable, and only
    the failed reads that named it; with `entities`, only places among them. Without `refused`
    the refused places are left out, and without `failed` the failed reads."""
    if variable is not None and not is_un_dcid(variable):
        return ()
    places = dict.fromkeys(blocked.places() if refused else ())
    for failure in blocked.failures() if failed else ():
        if variable is None or variable in failure.variables:
            places.update(dict.fromkeys(failure.places))
    if entities is not None:
        wanted = set(entities)
        return tuple(place for place in places if place in wanted)
    return tuple(places)


def _withheld_warning(variable: str, withheld: Mapping[str, tuple[str, ...]]) -> WarningModel:
    _title, detail, severity = DATASETS_WARNING_CATALOGUE["withheld_by_registry"]
    provenances = dict.fromkeys(provenance for held in withheld.values() for provenance in held)
    return WarningModel(
        code="withheld_by_registry",
        severity=severity,
        message=f"{detail} {variable} served no observations for {named_capped(tuple(withheld))}; "
        f"withheld provenances holding data there: {', '.join(provenances)}.",
    )


__all__ = [
    "WITHHELD_READ_MAX_BYTES",
    "WithheldReader",
    "catalogue_build_id",
    "origin_reads",
    "stale_warnings",
    "un_unavailable_warnings",
    "withheld_warning",
    "withheld_warnings",
]
