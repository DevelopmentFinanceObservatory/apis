"""Each place's one national currency, read from the graph's `nationalCurrencyOf` arcs.

Search's conversion catalogue loads the map once per graph generation, the Graph API's
variable card folds a facet's national-currency units through it, and a mint whose unit
binds each place's national currency (`facets.NationalCurrency`) narrows its probe through
it (`narrow_to_national_currency`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from materialise.errors import FacetBindingUnsupportedError, UpstreamResponseError
from materialise.facets import (
    MemberProbe,
    NationalCurrency,
    PlaceAdmission,
    currency_native_matches,
    native_currency_iso,
    subject_probe,
)
from materialise.graph import Facet, SubjectPage

_NATIONAL_CURRENCY_OF = "nationalCurrencyOf"
_CURRENCY_UNIT = "CurrencyUnit"
_MAX_CURRENCY_UNITS = 5000
"""The most `CurrencyUnit` nodes the reverse read lists. A truncated read fails."""
_SUBJECT_ROLE = "observationAbout"

type GapReason = Literal[
    "no_national_currency", "several_national_currencies", "no_facet_in_national_currency"
]
"""Why a place serves no row under a national-currency binding: no currency names it
through `nationalCurrencyOf`, several do, or it holds no facet in the one that does."""

_GAP_REASONS: tuple[GapReason, ...] = (
    "no_national_currency",
    "several_national_currencies",
    "no_facet_in_national_currency",
)


class NationalCurrencyReader(Protocol):
    """The two graph reads `read_national_currency_codes` takes. `materialise.graph.GraphReader`
    and the Graph API's `graph_api.reader.GraphNodeReader` both satisfy it."""

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage: ...

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]: ...


@dataclass(frozen=True)
class CoverageGap:
    """The places of one role a national-currency binding serves no row at, for one
    `reason`, in dcid order. A mint's trace carries them (`ManifestTrace.coverage_gaps`)."""

    role: str
    reason: GapReason
    entities: tuple[str, ...]

    def as_trace_entry(self) -> dict[str, object]:
        """The trace's wire shape: `role`, `reason` and `entities`."""
        return {"role": self.role, "reason": self.reason, "entities": list(self.entities)}


def read_national_currency_codes(
    reader: NationalCurrencyReader, *, deadline: float | None
) -> dict[str, frozenset[str]]:
    """The ISO codes of every currency whose `nationalCurrencyOf` names each place: the
    `CurrencyUnit` nodes' arcs, inverted. A place no currency names is absent. One reverse
    read lists the units and one node read reads their arcs. Raises `UpstreamResponseError`
    when the listing is truncated."""
    page = reader.node_subjects(
        _CURRENCY_UNIT, property="typeOf", max_results=_MAX_CURRENCY_UNITS, deadline=deadline
    )
    if page.truncated:
        raise UpstreamResponseError(
            f"more than {_MAX_CURRENCY_UNITS} nodes name {_CURRENCY_UNIT!r} by typeOf"
        )
    units = page.dcids
    # `typeOf` keeps every node from coming back with no arcs, which the read would retry.
    arcs = reader.node_arcs_many(
        list(units), properties=(_NATIONAL_CURRENCY_OF, "typeOf"), deadline=deadline
    )
    named: dict[str, set[str]] = {}
    for unit in units:
        code = native_currency_iso(unit)
        if code is None:
            continue
        for place in (arcs.get(unit) or {}).get(_NATIONAL_CURRENCY_OF, ()):
            named.setdefault(place, set()).add(code)
    return {place: frozenset(codes) for place, codes in named.items()}


def read_national_currencies(
    reader: NationalCurrencyReader, *, deadline: float | None
) -> dict[str, str]:
    """Each place's one national currency, by ISO code (`read_national_currency_codes`). A
    place two currencies name has none, as `materialise.recipes.resolve_entity_currencies`
    leaves it unmapped."""
    return {
        place: next(iter(codes))
        for place, codes in read_national_currency_codes(reader, deadline=deadline).items()
        if len(codes) == 1
    }


def narrow_to_national_currency(
    probe: MemberProbe,
    national_currency: NationalCurrency,
    *,
    reader: NationalCurrencyReader,
    deadline: float | None,
) -> tuple[MemberProbe, tuple[CoverageGap, ...]]:
    """`probe` holding, at each probed place, only its facets in the one currency the graph
    names for it (`read_national_currency_codes`), `Currency<ISO>` units first
    (`facets.currency_native_matches`), and the places left with none, by reason. The
    candidates, admission and series breadth are rebuilt from the narrowed places
    (`facets.subject_probe`), so a row serves only at a place whose currency its facet
    carries, and the election ranks each facet by the places it serves.

    Raises `FacetBindingUnsupportedError`, before any read, for a role other than
    `observationAbout` or a role-bearing probe, which holds no facets by place."""
    if national_currency.role != _SUBJECT_ROLE or not isinstance(probe.admission, PlaceAdmission):
        raise FacetBindingUnsupportedError(
            "a national-currency unit binding serves a subject-only variable at its "
            f"{_SUBJECT_ROLE} places; this request binds role {national_currency.role!r} "
            "or a variable with other roles",
            facet_pointer="/facets/unit",
        )
    if not probe.by_place:
        return probe, ()
    codes = read_national_currency_codes(reader, deadline=deadline)
    narrowed: dict[str, list[Facet]] = {}
    left_out: dict[GapReason, list[str]] = {}
    for place, facets in probe.by_place.items():
        held = codes.get(place, frozenset())
        if len(held) != 1:
            reason: GapReason = (
                "no_national_currency" if not held else "several_national_currencies"
            )
            left_out.setdefault(reason, []).append(place)
            continue
        (code,) = held
        own = currency_native_matches(facets, "unit", code)
        if not own:
            left_out.setdefault("no_facet_in_national_currency", []).append(place)
        narrowed[place] = own
    gaps = tuple(
        CoverageGap(role=_SUBJECT_ROLE, reason=reason, entities=tuple(sorted(left_out[reason])))
        for reason in _GAP_REASONS
        if reason in left_out
    )
    rebuilt = replace(
        subject_probe(narrowed, list(probe.by_place)),
        places=probe.places,
        every_place=probe.every_place,
        variable_breadth=probe.variable_breadth,
    )
    return rebuilt, gaps
