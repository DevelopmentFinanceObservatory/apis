"""What the origin catalogue adds to a mint: the build a manifest's trace records, the
`origin_catalogue_stale`, `withheld_by_registry` and `un_system_unavailable` warnings, and the
`excluded_by_registry` refusal.

Each request binds one catalogue (`origin_catalogue.current_catalogue`), so a manifest's trace,
its citation sources and its warnings name one build.

The graph withholds a facet whose provenance the catalogue does not admit. Finding what it
withheld costs a facet read with no domain filter, so a mint re-reads only when its answer left
an entity without rows, and under the request's deadline and `WITHHELD_READ_MAX_BYTES`. A
re-read names an entity only where the catalogue admits none of the series' facets there
(`WithheldReader.withheld_at`).

- When no member's facet probe returns a candidate, materialisation returns the empty result
  (`MaterialisationResult.probe_empty`). `refuse_unserved_empty_probe` re-reads every
  subject-only member's variable at its entities, one read per distinct list of entities,
  and refuses `excluded_by_registry` when a withheld provenance holds data there. The empty
  result says the places hold no data, so a re-read that fails or passes its byte cap refuses
  the mint as a retryable upstream failure.
- When one member's candidates leave an unbound facet axis without a default
  (`DefaultFacetUndefinedError.request`), `refuse_unserved_selection` makes the same re-read
  for that member and refuses `excluded_by_registry` in place of `default_facet_undefined`.
- When some subject entities came back with no rows, `withheld_finding` re-reads the first such
  request's variable at those entities and warns `withheld_by_registry`.

In the last two a failed re-read is logged and leaves the answer as it was, since the answer
itself was read in full.

A subject-only request that binds no entity covers its default place scope, every place where it
holds data or its countries with data (`materialise.place_scope`). The warning takes those places
from the materialisation's result, which carries the ones its facet probe read. The refusal reads
the scope again (`_refusal_scopes`), to cover every place a countries fallback could hold.

A mint materialises inside `origin_catalogue.recording_un_blocked_places`, whose record holds
what UN observation reads could not serve. These are places the UN front door refuses, and UN
reads that failed inside a read that served our variables. A facet probe made
before the record opened carries its own record, and the materialisation that reuses the probe
adds it to this one, so the record covers the probe at every door. `un_unavailable_warnings`
names those places as `un_system_unavailable`. Each warning has an `OriginGap` beside it, the
places it names as the manifest's trace records them. A UN member with no candidate, or with no
default on a facet axis, refuses `upstream_error` (`UnSystemUnavailable`) at such a place,
since its probe says nothing about data there. A member that binds no entity covers every
country, including the few the UN front door refuses on every read, so a retry would meet the
same refusal. It refuses `upstream_error` only
when a UN read failed, and otherwise serves the empty result with the `un_system_unavailable`
warning naming the refused places.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, runtime_checkable

from datasets_api.codes_materialise import ExcludedByRegistry, UnSystemUnavailable
from datasets_api.manifest_models import WarningModel
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dc_graph import GraphCapExceededError, GraphReadError
from dc_sdmx import ContainedIn
from dfo_platform.named import named_capped
from materialise import MaterialisationRequest
from materialise.facets import binds_no_place
from materialise.graph import GraphReader
from materialise.place_scope import (
    DEFAULT_SCOPE_PARENT,
    AvailabilityReader,
    earth_countries,
    read_default_presence,
)
from materialise.window import Coverage
from origin_catalogue import UnBlockedPlaces, current_catalogue
from origin_catalogue.graph import is_un_dcid

logger = logging.getLogger(__name__)

_SUBJECT_ROLE: Final[str] = "observationAbout"
_STAT_VAR_REF_PREFIX: Final[str] = "dcid:"

WITHHELD_READ_MAX_BYTES: Final[int] = 16 * 1024 * 1024
"""Caps the unfiltered re-read's response. It reads one variable's facets, without dates, at
the entities of one request."""


@dataclass(frozen=True)
class OriginGap:
    """The places an answer holds no data at for a reason the origin reads found, as the
    manifest's trace records them (`ManifestTrace.origin_gaps`): for `withheld_by_registry`,
    the variable, the places and the withheld provenances holding data there; for
    `un_system_unavailable`, the places the UN could not serve."""

    code: Literal["withheld_by_registry", "un_system_unavailable"]
    entities: tuple[str, ...]
    variable: str | None = None
    provenances: tuple[str, ...] = ()

    def as_trace_entry(self) -> dict[str, object]:
        """`code`, `variable` and `provenances` where set, and `entities`."""
        entry: dict[str, object] = {"code": self.code}
        if self.variable is not None:
            entry["variable"] = self.variable
        entry["entities"] = list(self.entities)
        if self.provenances:
            entry["provenances"] = list(self.provenances)
        return entry


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


def refuse_unserved_empty_probe(
    blocked: UnBlockedPlaces,
    requests: Sequence[MaterialisationRequest],
    graph: GraphReader,
    *,
    availability: AvailabilityReader,
    deadline: float | None,
) -> None:
    """Refuse a materialisation whose every member probed no candidate
    (`MaterialisationResult.probe_empty`) when the empty probe does not mean the places hold
    no data. Call it inside the `recording_un_blocked_places` block the materialisation ran
    in, with that block's record.

    Each subject-only member is checked at its bound entities or, when it binds none, at the
    places of its default place scope (`_refusal_scopes`). The first member with a UN variable
    the UN could not serve at some of those places raises `UnSystemUnavailable`. A refused
    place or a failed read counts for a member that names its places, and a failed read
    naming its variable alone for a member that binds none, so that check reads no scope.
    Otherwise every member is re-read, the members that share a list of entities in one read,
    and the first read where a withheld provenance holds data raises `ExcludedByRegistry`
    naming that read's variables.

    Returning means every check completed and found nothing. The members that bind no entity
    re-read their scopes through `availability` in one batch. A failed scope read or re-read
    raises the read's error, and a re-read past `WITHHELD_READ_MAX_BYTES` raises
    `GraphReadError`. The doors refuse each as a retryable upstream failure."""
    for request in requests:
        if unavailable := _un_unavailable(blocked, request):
            raise unavailable
    scopes = _refusal_scopes(requests, graph, availability=availability, deadline=deadline)
    for entities, variables in scopes.items():
        _refuse_withheld(variables, _withheld_at(graph, variables, entities, deadline=deadline))


def refuse_unserved_selection(
    blocked: UnBlockedPlaces,
    request: MaterialisationRequest,
    graph: GraphReader,
    *,
    availability: AvailabilityReader,
    deadline: float | None,
) -> None:
    """Refuse for the member whose candidates left an unbound facet axis without a default
    (`DefaultFacetUndefinedError.request`) when the UN could not serve it or the registry
    withholds its data, by `refuse_unserved_empty_probe`'s two checks over that member alone.
    Call it with the record of the block the materialisation ran in. Returning leaves the
    caller's `default_facet_undefined` refusal standing.

    A failed scope read raises the read's error. A failed re-read is logged and refuses
    nothing, since the caller's refusal holds for the candidates either way."""
    if unavailable := _un_unavailable(blocked, request):
        raise unavailable
    scopes = _refusal_scopes([request], graph, availability=availability, deadline=deadline)
    for entities, variables in scopes.items():
        _refuse_withheld(
            variables, _withheld_or_nothing(graph, variables, entities, deadline=deadline)
        )


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
    """`un_system_unavailable` naming the places `blocked` recorded, when there are any.

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


def un_unavailable_gaps(blocked: UnBlockedPlaces) -> tuple[OriginGap, ...]:
    """The places `un_unavailable_warnings` names without a `request`, as an `OriginGap`."""
    places = _un_unserved(blocked)
    return (OriginGap(code="un_system_unavailable", entities=places),) if places else ()


def withheld_warning(
    graph: object,
    served: Sequence[tuple[MaterialisationRequest, Coverage]],
    *,
    default_scope: Sequence[str] = (),
    deadline: float | None,
) -> tuple[int, WarningModel] | None:
    """`withheld_finding`'s request index and warning."""
    found = withheld_finding(graph, served, default_scope=default_scope, deadline=deadline)
    return None if found is None else (found[0], found[1])


def withheld_finding(
    graph: object,
    served: Sequence[tuple[MaterialisationRequest, Coverage]],
    *,
    default_scope: Sequence[str] = (),
    deadline: float | None,
) -> tuple[int, WarningModel, OriginGap] | None:
    """`withheld_by_registry` for the first request in `served` that left a subject entity
    without rows, with that request's index and the places it names, when that entity has no
    admitted data and the catalogue withholds a provenance holding data there. At most one
    facet read runs.

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
        withheld = _withheld_or_nothing(graph, [variable], unserved, deadline=deadline)
        if not withheld:
            return None
        gap = OriginGap(
            code="withheld_by_registry",
            entities=tuple(withheld),
            variable=variable,
            provenances=tuple(
                dict.fromkeys(provenance for held in withheld.values() for provenance in held)
            ),
        )
        return index, _withheld_warning(variable, withheld), gap
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
    graph: object, variables: Sequence[str], entities: Sequence[str], *, deadline: float | None
) -> Mapping[str, tuple[str, ...]]:
    """The re-read of `variables` at `entities`, or nothing when `graph` has no re-read and so
    withholds nothing. A failed read raises its `GraphReadError`, and one past
    `WITHHELD_READ_MAX_BYTES` raises `GraphReadError` from the cap error."""
    if not isinstance(graph, WithheldReader):
        return {}
    try:
        return graph.withheld_at(
            variables, entities, deadline=deadline, max_bytes=WITHHELD_READ_MAX_BYTES
        )
    except GraphCapExceededError as exc:
        raise GraphReadError(
            f"the withheld-provenance re-read of {len(variables)} variables at "
            f"{len(entities)} entities passed its byte cap: {exc}"
        ) from exc


def _withheld_or_nothing(
    graph: object, variables: Sequence[str], entities: Sequence[str], *, deadline: float | None
) -> Mapping[str, tuple[str, ...]]:
    """`_withheld_at`, or nothing when the read fails, for an answer that stands without it."""
    try:
        return _withheld_at(graph, variables, entities, deadline=deadline)
    except GraphReadError as exc:
        logger.warning(
            "Withheld-provenance re-read for %s at %d entities failed; the answer carries no "
            "registry explanation: %s",
            ", ".join(variables),
            len(entities),
            exc,
        )
        return {}


def _refuse_withheld(variables: Sequence[str], withheld: Mapping[str, tuple[str, ...]]) -> None:
    """Raise `ExcludedByRegistry` for `variables` when their re-read found `withheld` data."""
    if withheld:
        provenances = tuple(
            dict.fromkeys(provenance for held in withheld.values() for provenance in held)
        )
        raise ExcludedByRegistry(variables=tuple(variables), provenances=provenances)


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


def _refusal_scopes(
    requests: Sequence[MaterialisationRequest],
    graph: GraphReader,
    *,
    availability: AvailabilityReader,
    deadline: float | None,
) -> dict[tuple[str, ...], list[str]]:
    """The entities each subject-only request among `requests` covers, as each distinct list
    of entities with the bare variables that cover it, in first-seen order. A request covers
    its plain bound list, or when it binds none the places its default place scope could
    hold. A role-bearing request and a `ContainedIn` scope are left out.

    The requests that bind none have their scopes read again, in one batch. Every place with
    data counts every provenance, so it holds the withheld data's places. The countries
    fallback counts admitted data only, so a variable whose data is all withheld has an empty
    one, and its entities are then Earth's countries and Earth."""
    scopes: dict[tuple[str, ...], dict[str, None]] = {}
    unbound: dict[str, None] = {}
    for request in requests:
        if binds_no_place(request.role_bindings, request.declared_roles):
            unbound[request.stat_var.removeprefix(_STAT_VAR_REF_PREFIX)] = None
        elif (bound := _subject_binding(request)) is not None:
            variable, entities = bound
            scopes.setdefault(tuple(entities), {})[variable] = None
    if unbound:
        presence = read_default_presence(
            graph=graph,
            availability=availability,
            stat_vars=unbound,
            deadline=deadline,
            settle=True,
        )
        fallback: tuple[str, ...] | None = None
        for variable in unbound:
            scope = presence.scope(variable)
            if scope.every_place:
                places = tuple(scope.places)
            else:
                if fallback is None:
                    # A memoised scope comes without the countries read.
                    countries = presence.countries or earth_countries(graph, deadline=deadline)
                    fallback = (*countries, DEFAULT_SCOPE_PARENT)
                places = fallback
            scopes.setdefault(places, {})[variable] = None
    return {entities: list(variables) for entities, variables in scopes.items()}


def _un_unserved(
    blocked: UnBlockedPlaces,
    *,
    variable: str | None = None,
    entities: Sequence[str] | None = None,
    refused: bool = True,
) -> tuple[str, ...]:
    """The places the UN could not serve, in first-seen order: those its front door refused,
    then those of failed UN reads. With `variable`, none unless it is a UN variable, and only
    the failed reads that named it; with `entities`, only places among them. Without `refused`
    the refused places are left out."""
    if variable is not None and not is_un_dcid(variable):
        return ()
    places = dict.fromkeys(blocked.places() if refused else ())
    for failure in blocked.failures():
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
    "OriginGap",
    "WithheldReader",
    "catalogue_build_id",
    "refuse_unserved_empty_probe",
    "refuse_unserved_selection",
    "stale_warnings",
    "un_unavailable_gaps",
    "un_unavailable_warnings",
    "withheld_finding",
    "withheld_warning",
    "withheld_warnings",
]
