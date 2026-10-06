"""The `un_system_unavailable` warning for UN observation reads that could not serve.

A UN observation read names places. The UN System Data Commons front door refuses some of
them, and the origin client answers those empty and notes them in the caller's
`recording_un_blocked_places` record. A UN read that fails inside a read also naming our
variables joins the record as an `UnReadFailure` while our part serves. Each read that binds
a record reports what it holds with this warning, so neither reads as no data.

The warning's `params` hold the refused place dcids (`places`), the variables of the
failed reads (`variables`) and their graph names (`variable_labels`), so a client tells the two
apart and names the variables without reading the message. The names come from the members of
the Specs the emitter holds (`member_labels`); the warning makes no read, since the UN reads
it reports have just failed.

On a region's answer the warning carries an `entity_unavailable` departure naming the refused
places the answer names as subjects, when every StatVar the answer reads is a UN one. Those
places then hold no values in it (`refused_places_fact`). A streamed answer reads its
observations after its part is sent, so the departure for places refused there arrives on
`done` (`enrichment.late_departures`). A failed read names variables and no role, so it
departs nowhere and stays with the warning.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final

from origin_catalogue import UnBlockedPlaces
from origin_catalogue.graph import is_un_dcid
from search.engine.departures import (
    NO_ENTITY_WORDS,
    unavailable_entity_facts,
    warning_key,
)
from search.engine.region.codes import UN_SYSTEM_UNAVAILABLE
from search.engine.region.result import RegionResult, with_departure_facts
from search.heading.departure_fact import DepartureFact
from search.models import Spec, Warning

_MAX_NAMED: Final = 5
_SUBJECT_ROLE: Final = "observationAbout"


# The labels an emitter holding no Spec passes. Every variable is named `""`.
NO_LABELS: Final[Mapping[str, str]] = MappingProxyType({})


def member_labels(specs: Iterable[Spec]) -> dict[str, str]:
    """Each served member's graph name over `specs`, by dcid, `""` for a member whose label
    echoes its dcid because the graph holds no name for it."""
    return {
        member.ref.dcid: member.ref.label if member.ref.label != member.ref.dcid else ""
        for spec in specs
        for member in spec.resolved.members
    }


def _named(items: Sequence[str]) -> str:
    named = ", ".join(items[:_MAX_NAMED])
    if len(items) > _MAX_NAMED:
        named = f"{named} and {len(items) - _MAX_NAMED} more"
    return named


def un_unavailable_warning(record: UnBlockedPlaces, *, labels: Mapping[str, str]) -> Warning | None:
    """The warning naming the refused places and failed reads in `record`, or `None` when it
    holds neither. Its `params` list the refused place dcids as `places`, the variables of
    the failed reads as `variables`, each empty when the record holds none, and each
    variable's name from `labels`, `""` where it holds none, as `variable_labels`."""
    places = record.places()
    failures = record.failures()
    sentences: list[str] = []
    if places:
        sentences.append(
            f"The UN System Data Commons refused reads for {', '.join(places)}; "
            "the answer holds no UN data there."
        )
    sentences.extend(
        f"The UN System Data Commons read for {_named(failure.variables)} at "
        f"{_named(failure.places)} failed; the answer holds no UN data for them."
        for failure in failures
    )
    if not sentences:
        return None
    variables = list(
        dict.fromkeys(variable for failure in failures for variable in failure.variables)
    )
    return Warning(
        code=UN_SYSTEM_UNAVAILABLE,
        severity="warn",
        message=" ".join(sentences),
        params={
            "places": list(places),
            "variables": variables,
            "variable_labels": [labels.get(variable, "") for variable in variables],
        },
    )


def refused_places(warning: Warning) -> frozenset[str]:
    """The place dcids an `un_system_unavailable` warning says the front door refused, read
    from its `params`; none for any other warning, or one without them."""
    if warning.code != UN_SYSTEM_UNAVAILABLE or warning.params is None:
        return frozenset()
    places = warning.params.get("places", [])
    return frozenset(places) if isinstance(places, list) else frozenset()


def with_un_unavailable_warning(result: RegionResult, record: UnBlockedPlaces) -> RegionResult:
    """`result` with the warning for what `record` holds, when it holds anything, its
    variables named by the members of `result`'s Specs, and the `entity_unavailable`
    departure on the refused places its answer names (`refused_places_fact`)."""
    specs = result.specs or result.flagged_match or ()
    warning = un_unavailable_warning(record, labels=member_labels(specs))
    if warning is None:
        return result
    warned = dataclasses.replace(result, warnings=(*result.warnings, warning))
    fact = refused_places_fact(specs, refused=refused_places(warning))
    if fact is None:
        return warned
    return with_departure_facts(warned, {warning_key(warning): fact})


def refused_places_fact(
    specs: Sequence[Spec],
    *,
    refused: frozenset[str],
    entity_words: Mapping[str, str] = NO_ENTITY_WORDS,
) -> DepartureFact | None:
    """`entity_unavailable` on the `refused` places the subject role of an answer showing
    `specs` names: its Spec, every reading of a candidates answer, or the flagged match of a
    no-data one. UN series bind places as subjects, so the subject role is the one role. A
    group on that role is asked in `entity_words`. `None` when the answer names none of
    them, or reads a StatVar outside the UN, which may hold values there."""
    if not refused or not specs or not all(_reads_only_un(spec) for spec in specs):
        return None
    facts = unavailable_entity_facts(
        specs[0],
        unserved={
            (entity.role, entity.ref)
            for entity in specs[0].resolved.entities
            if entity.role == _SUBJECT_ROLE and entity.ref in refused
        },
        entity_words=entity_words,
    )
    return facts[0] if facts else None


def _reads_only_un(spec: Spec) -> bool:
    """Whether every StatVar `spec` reads lives on the UN deployment."""
    return all(is_un_dcid(stat_var.ref.dcid) for stat_var in spec.resolved.members)


__all__ = [
    "NO_LABELS",
    "member_labels",
    "refused_places",
    "refused_places_fact",
    "un_unavailable_warning",
    "with_un_unavailable_warning",
]
