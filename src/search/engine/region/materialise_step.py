"""Shared materialisation call and outcome classification for both region doors.

Each door builds its own terminal result from the shared classification.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from materialise.place_scope import DefaultPresence
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.closest_match import substituted_measure_fact
from search.engine.extract import DateRequest
from search.engine.region.codes import FAMILY_TOTAL_ABSENT, MEMBERSHIP_UNVERIFIED
from search.engine.region.grounding import (
    EntityRoles,
    carrier_flag_warnings,
    role_scope_defaulted_warnings,
)
from search.engine.retrieval.retrieve import (
    FamilyTotalAbsent,
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
    materialise,
)
from search.engine.shape import ShapeDraft
from search.heading.departure_fact import DepartureFact
from search.models import Spec, Warning
from search.render.heading import heading_for
from series_coverage.shared import CoverageSnapshot

__all__ = [
    "Departed",
    "NoDataMaterialise",
    "UnsupportedMaterialise",
    "family_total_absent_departure",
    "family_total_absent_departures",
    "interpret_materialised",
    "resubmit_family_total_absent_departure",
    "run_materialise",
]


def run_materialise(
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    entities: Mapping[str, list[str]],
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    date_request: DateRequest | None,
    deadline: float | None,
    place_dcids: list[str] | None,
    default_presence: DefaultPresence | None = None,
    coverage: CoverageSnapshot,
) -> Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft:
    """Materialise the chosen shape with scope discovery enabled. `default_presence` derives
    a default place scope in place of `place_dcids` (`standard_resolve`)."""
    return materialise(
        shape_draft,
        bindings,
        entities,
        graph=graph,
        sdmx=sdmx,
        date_request=date_request,
        deadline=deadline,
        place_dcids=place_dcids,
        default_presence=default_presence,
        discover_scope=True,
        coverage=coverage,
    )


@dataclass(frozen=True, slots=True)
class UnsupportedMaterialise:
    """Scope refusal details for the caller's unsupported terminal."""

    code: str
    pointer: str
    detail: str


@dataclass(frozen=True, slots=True)
class NoDataMaterialise:
    """A no-data outcome. Each caller decides whether to return or defer it."""

    reason: str


type Departed = tuple[Warning, DepartureFact]
"""A warning and the departure fact it discloses."""


def _family_total_absent_message(
    fact: FamilyTotalAbsent, *, label_of: Callable[[Sequence[str]], Sequence[str]]
) -> tuple[str, str]:
    """The warning's message and the label of the member shown. The message names the
    member shown and, when the family holds a total, the total that had no data where the
    read plan read: at the places, and under a year range in the years asked, since the
    total can hold data there in other years."""
    if fact.total is None:
        (shown,) = label_of([fact.shown])
        return (
            f"No total for this family of variables was found, so the answer shows {shown}.",
            shown,
        )
    # The message names a lone place and counts any other number, so only a lone place is
    # labelled. A place-less answer probes up to every country.
    lone_place = fact.places if len(fact.places) == 1 else ()
    total, shown, *named = label_of([fact.total, fact.shown, *lone_place])
    where = named[0] if named else f"any of the {len(fact.places)} places"
    when = " in the years asked" if fact.in_years_asked else ""
    return f"No data for {total} was found for {where}{when}, so the answer shows {shown}.", shown


def family_total_absent_departure(
    fact: FamilyTotalAbsent,
    *,
    spec: Spec | None,
    variable_text: str,
    label_of: Callable[[Sequence[str]], Sequence[str]],
) -> Departed:
    """The `family_total_absent` warning for `fact` and its `closest_match` departure
    (`substituted_measure_fact`), which sets the request's words `variable_text` against the
    member shown. `spec` is the Spec the part holds that shows the member; the warning carries its
    `spec_id`, so a candidates part, which states only what holds for every reading, leaves
    the departure out, and the departure names the measure noun of its heading. An answer
    holding no Spec, one for data outside the years asked, names the member by its label."""
    warning, departure = _family_total_absent(
        fact, spec=spec, variable_text=variable_text, label_of=label_of
    )
    if spec is not None:
        warning = warning.model_copy(update={"spec_id": spec.spec_id})
    return warning, departure


def family_total_absent_departures(
    specs: Iterable[Spec],
    facts: Mapping[str, FamilyTotalAbsent],
    *,
    variable_text: str,
    label_of: Callable[[Sequence[str]], Sequence[str]],
) -> list[Departed]:
    """`family_total_absent_departure` for each Spec in `specs` whose single StatVar is not
    its family's total, reading `facts` by `spec_id`."""
    return [
        family_total_absent_departure(
            facts[spec.spec_id], spec=spec, variable_text=variable_text, label_of=label_of
        )
        for spec in specs
        if spec.spec_id in facts
    ]


def resubmit_family_total_absent_departure(
    fact: FamilyTotalAbsent,
    *,
    spec: Spec | None,
    label_of: Callable[[Sequence[str]], Sequence[str]],
) -> Departed:
    """The `family_total_absent` warning for `fact` and its `closest_match` departure on a
    resubmit, whose door holds no words for the measure. The departure states only the
    member shown, by the measure noun of `spec`'s heading, else by its label. The warning
    carries no `spec_id`, and a resubmit stamps its ground warnings once its answer is
    definite (`_PromoteState.definite_warnings`)."""
    return _family_total_absent(fact, spec=spec, variable_text=None, label_of=label_of)


def _family_total_absent(
    fact: FamilyTotalAbsent,
    *,
    spec: Spec | None,
    variable_text: str | None,
    label_of: Callable[[Sequence[str]], Sequence[str]],
) -> Departed:
    """The unstamped `family_total_absent` warning for `fact` and its `closest_match`
    departure, naming the measure noun of `spec`'s heading, else the member's label."""
    message, shown = _family_total_absent_message(fact, label_of=label_of)
    warning = Warning(code=FAMILY_TOTAL_ABSENT, severity="warn", message=message)
    noun = heading_for(spec).heading.noun.inline_text if spec is not None else shown
    return warning, substituted_measure_fact(variable_text=variable_text, measure_noun=noun)


def interpret_materialised(
    mat_result: Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft,
    *,
    boundary_roles: EntityRoles,
    label_of: Callable[[Sequence[str]], Sequence[str]],
    warnings: list[Warning],
) -> UnsupportedMaterialise | NoDataMaterialise | None:
    """Classify unsupported and no-data outcomes, then append carrier warnings.

    Returns ``UnsupportedMaterialise`` for a scope refusal, ``NoDataMaterialise`` for
    ``NoDataDraft``, and ``None`` for materialised outcomes. A ``Materialised`` also adds
    its scope-defaulted, role-declaration, and membership warnings. Its
    ``family_total_absent`` fact describes one Spec, so the terminal that holds that
    Spec raises it.
    """
    if isinstance(mat_result, UnsupportedDraft):
        return UnsupportedMaterialise(mat_result.code, mat_result.pointer, mat_result.detail)
    if isinstance(mat_result, NoDataDraft):
        return NoDataMaterialise(mat_result.reason)
    if isinstance(mat_result, Materialised):
        warnings.extend(
            role_scope_defaulted_warnings(
                mat_result, boundary_roles=boundary_roles, label_of=label_of
            )
        )
        warnings.extend(
            carrier_flag_warnings(
                role_declaration_unverified=mat_result.role_declaration_unverified,
                open_origin_defaulted=False,
                # The elected shape's own unsettled membership already warned at binding.
                membership_unverified=mat_result.membership_unverified
                and all(warning.code != MEMBERSHIP_UNVERIFIED for warning in warnings),
            )
        )
    return None
