"""A variable card's `availability`: where DFO holds observations of the variable, folded into a
summary, the default a request naming nothing receives, and one entry per source with its series.

Two readers supply the facets, each admitted per facet by the request's catalogue:

- A role-bearing DFO variable, one with coverage role names, reads the coverage index's facet
  summary (`CoverageIndex.facet_summary`). The summary covers every row of the build, so a facet
  whose provenance the catalogue does not admit (`Catalogue.admit_series`) is left out. Its
  entities are parties per graph role, split into parties, totals and residual entries
  (`VariableSummary.party_counts`).
- Every other variable, base, UN and the roleless DFO variables, reads its facet profile
  (`materialise.facet_profile.FacetProfile`): the facets the graph admitted at its default place
  scope and the places each was admitted at. Its entities are places counted by kind.

The default is the profile's `default`, the facets `materialise.facets.served_facets` elects for
a request naming no place, years or facet, so the card and that request cannot disagree. For a
role-bearing variable it is the coverage election the same probe gives.

Within a source, facets group into series by measurement method, period and scaling. A unit
folds into a series' `national_currency` when it is a currency and the graph's
`nationalCurrencyOf` (`materialise.national_currency.read_national_currencies`) names it for
every place its facet covers. A place with no mapping keeps the facet in the plain fold. Spans
are unions over places and claim no continuity at any one place.

A request reads each variable's admitted facets once (`held_facets`), and its read targets
(`HeldFacets.targets`), card (`build_availability`) and `facets` list items (`facet_items`)
come from that one read. A profile miss is `availability: null` with a transient gap
(`UNAVAILABLE`), so the response is not cached. A surface that reads no profiles by
configuration, with no profile cache or no coverage index configured, leaves the permanent
gap `NOT_SERVED`. A default place scope that fell back to countries is a permanent gap that
names no cause.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from code_index.types import PartyRow
from dfo_platform.wire import ResponseModel
from graph_api.refs import DOCUMENTATION_PATH, NodeRef
from materialise.facet_profile import FacetProfile
from materialise.facets import native_currency_iso
from origin_catalogue import Catalogue
from unit_words.currency_nouns import currency_noun

if TYPE_CHECKING:
    # Keep coverage annotation-only. Importing its modules adds NumPy to the graph install.
    from series_coverage.index import CoverageIndex
    from series_coverage.schema import RoleNames
    from series_coverage.summary import PartyCounts, VariableSummary

AVAILABILITY_GAP = "stat_var.availability"
"""The gap of a card whose availability could not be read. It is transient."""

NOT_SERVED_GAP = "stat_var.availability.not_served"
"""The gap of a card on a surface that reads no facet profiles, with no profile cache or no
coverage index configured. It is permanent."""

COUNTRIES_ONLY_GAP = "stat_var.availability.countries_only"
"""The gap of a card whose default place scope is the countries with data, since the full list
of places was unavailable or too long. It is permanent and names no cause."""

MAX_SOURCES = 10
"""The most sources a card lists; `more_sources` counts the rest."""

_LEADING_UNITS = 2


class AvailabilitySource(ResponseModel):
    """A source that holds the variable: its provenance, labelled with the catalogue's name."""

    model_config = ConfigDict(title="GraphAvailabilitySource")

    ref: str
    label: str | None
    documentation: str = Field(description="the path of the provenance's source document")


class EntityCounts(ResponseModel):
    """The distinct entities in one role, split by their kind in the code index."""

    model_config = ConfigDict(title="GraphAvailabilityEntityCounts")

    parties: int = Field(description="entities with figures of their own")
    private: int = Field(description="the parties whose sector is private, a subset of parties")
    totals: int = Field(description="entities that hold other entities' figures")
    residual: int = Field(description="entities holding figures no other entity does")
    unclassified: int = Field(description="entities the code index does not list")


class UnitCount(ResponseModel):
    """One unit of a series. `ref` is `null` for facets that state no unit."""

    model_config = ConfigDict(title="GraphAvailabilityUnit")

    ref: str | None
    label: str | None
    places: int | None = Field(
        description="the places its facets cover; null for a role-bearing variable"
    )


class NationalCurrency(ResponseModel):
    """The units that are each covered place's own currency."""

    model_config = ConfigDict(title="GraphAvailabilityNationalCurrency")

    currencies: int
    places: int


class UnitsFold(ResponseModel):
    """A series' units: how many, the two covering the most places, and how many more."""

    model_config = ConfigDict(title="GraphAvailabilityUnits")

    count: int
    leading: list[UnitCount]
    more: int
    national_currency: NationalCurrency | None = Field(
        description="the units that are their places' national currency; null when none is"
    )


class Binding(ResponseModel):
    """The `scope.facets` values that request a series from Datasets, in its leading unit."""

    model_config = ConfigDict(title="GraphAvailabilityBinding")

    provenance: str
    # The `scope.facets` axis name, so a binding posts as it reads.
    measurementMethod: str | None
    unit: str | None


class Series(ResponseModel):
    """A source's facets sharing a measurement method, period and scaling."""

    model_config = ConfigDict(title="GraphAvailabilitySeries")

    measurement_method: NodeRef | None = Field(
        description="null when no facet states a method: no stated method"
    )
    observation_period: str | None
    scaling: str | None
    first_year: str | None
    last_year: str | None
    places: int | None = Field(description="null for a role-bearing variable")
    observations: int
    units: UnitsFold
    binding: Binding


class SourceAvailability(ResponseModel):
    """One source's facets of the variable, and their series."""

    model_config = ConfigDict(title="GraphAvailabilitySourceEntry")

    source: AvailabilitySource
    places: dict[str, int] | None = Field(
        description="the places covered by kind; null for a role-bearing variable, and while the "
        "places' kinds are unread (a transient gap)"
    )
    entities: dict[str, EntityCounts] | None = Field(
        description="the entities covered per graph role; null for any other variable"
    )
    first_year: str | None
    last_year: str | None
    facets: int
    series: list[Series]


class AvailabilityDefault(ResponseModel):
    """What a data request naming no place, years or facet receives."""

    model_config = ConfigDict(title="GraphAvailabilityDefault")

    facets: list[str] = Field(description="the facet ids the request is served, in order")
    source: AvailabilitySource | None
    unit: NodeRef | None
    measurement_method: NodeRef | None
    observation_period: str | None
    first_year: str | None
    last_year: str | None
    places: int | None = Field(description="null for a role-bearing variable")
    without_data: int = Field(
        description="the places, or parties, holding the variable that the default leaves "
        "without data"
    )


class AvailabilitySummary(ResponseModel):
    """The variable across every source."""

    model_config = ConfigDict(title="GraphAvailabilitySummary")

    places: dict[str, int] | None = Field(
        description="the places covered by kind; null for a role-bearing variable, and while the "
        "places' kinds are unread (a transient gap)"
    )
    entities: dict[str, EntityCounts] | None = Field(
        description="the entities covered per graph role; null for any other variable"
    )
    first_year: str | None
    last_year: str | None
    sources: int
    facets: int


class Availability(ResponseModel):
    """Where DFO holds observations of a variable. `sources: []` means it verifiably holds
    none."""

    model_config = ConfigDict(title="GraphAvailability")

    scope: Literal["dfo"] = "dfo"
    summary: AvailabilitySummary
    default: AvailabilityDefault | None = Field(
        description="null when no facet dispatches a request naming nothing"
    )
    sources: list[SourceAvailability]
    more_sources: int = Field(description="the sources past the first ten")


class FacetItem(ResponseModel):
    """One facet of a variable, as the `facets` list relation serves it. `ref` is the facet id,
    which hashes the facet's metadata."""

    model_config = ConfigDict(title="GraphFacetItem")

    ref: str
    label: None = None
    source: AvailabilitySource
    unit: NodeRef | None
    measurement_method: NodeRef | None
    observation_period: str | None
    scaling: str | None
    first_year: str | None
    last_year: str | None
    observations: int
    places: int | None = Field(description="null for a role-bearing variable")


@dataclass(frozen=True, slots=True)
class AvailabilityRead:
    """A card's availability and the gaps it leaves. `transient` is true when a gap comes from
    a failed read, so the response is not cached."""

    availability: Availability | None
    gaps: tuple[str, ...] = ()
    transient: bool = False


UNAVAILABLE = AvailabilityRead(availability=None, gaps=(AVAILABILITY_GAP,), transient=True)
"""A card whose profile missed, or whose coverage index failed to load."""

NOT_SERVED = AvailabilityRead(availability=None, gaps=(NOT_SERVED_GAP,))
"""A card on a surface that reads no facet profiles by configuration."""


class NoMethod(Enum):
    """The `measurement_method` filter value that selects facets stating no method."""

    NO_METHOD = auto()


NO_METHOD = NoMethod.NO_METHOD


@dataclass(frozen=True, slots=True)
class FacetFilter:
    """The `facets` relation's filters. `None` leaves an axis open, and `NO_METHOD` selects
    the facets with no stated method."""

    source: str | None = None
    unit: str | None = None
    measurement_method: str | NoMethod | None = None

    def keeps(self, held: _Held) -> bool:
        method = self.measurement_method
        return (
            (self.source is None or held.provenance == self.source)
            and (self.unit is None or held.unit == self.unit)
            and (
                method is None
                or held.measurement_method == (None if method is NO_METHOD else method)
            )
        )


_OPEN = FacetFilter()


@dataclass(frozen=True, slots=True)
class ReadTargets:
    """What `build_availability` reads beside the profile: the units that are not currencies
    and the measurement methods it names (`labels`), and the places it counts by kind
    (`place_kind`), empty for a role-bearing variable."""

    named: frozenset[str]
    places: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Held:
    """One admitted facet. `reach` holds the places it covers, or for a role-bearing variable
    the parties it carries in any role."""

    facet_id: str
    provenance: str
    measurement_method: str | None
    unit: str | None
    scaling: str | None
    period: str | None
    first_year: int | None
    last_year: int | None
    observations: int
    reach: frozenset[str]


@dataclass(frozen=True, slots=True)
class HeldFacets:
    """A variable's profile and its admitted facets, ordered by facet id (`held_facets`).
    `role_names` and `summary` are a role-bearing variable's graph roles and coverage facet
    summary, which its entity counts read; both are `None` for any other variable, and
    `summary` is `None` too for a role-bearing variable the summary does not hold."""

    profile: FacetProfile
    facets: tuple[_Held, ...]
    role_names: RoleNames | None = None
    summary: VariableSummary | None = None

    @property
    def role_bearing(self) -> bool:
        return self.role_names is not None

    @property
    def targets(self) -> ReadTargets:
        """The dcids `build_availability` takes names and kinds for."""
        units = {
            h.unit
            for h in self.facets
            if h.unit is not None and native_currency_iso(h.unit) is None
        }
        methods = {h.measurement_method for h in self.facets if h.measurement_method is not None}
        return ReadTargets(
            named=frozenset(units | methods),
            places=frozenset() if self.role_bearing else _union(self.facets),
        )


@dataclass(frozen=True, slots=True)
class _Fold:
    """A variable's admitted facets, ordered by facet id, and how its entities are counted."""

    facets: tuple[_Held, ...]
    role_bearing: bool
    places_by_kind: Callable[[Iterable[_Held]], dict[str, int]] | None
    entities: Callable[[Iterable[_Held]], dict[str, EntityCounts]] | None


def build_availability(
    *,
    held: HeldFacets,
    catalogue: Catalogue,
    parties: Mapping[str, PartyRow],
    national_currencies: Mapping[str, str],
    place_kind: Callable[[str], str] | None,
    labels: Mapping[str, str],
) -> AvailabilityRead:
    """A variable's availability from its admitted facets (`held_facets`).

    `catalogue` is the one the profile was keyed on. `parties` is the code index's party rows
    (`CodeIndex.parties`), `national_currencies` each place's ISO currency
    (`read_national_currencies`; an empty map folds nothing), `place_kind` a place's wire kind,
    `None` while the kinds are unread, which leaves `summary.places` and each
    `sources[].places` null, and `labels` the graph names of units and methods. A currency unit
    reads as its noun (`unit_words.currency_noun`)."""
    profile = held.profile
    fold = _counted(held, parties=parties, place_kind=place_kind)
    names = _Names(catalogue=catalogue, labels=labels)
    currencies = {} if fold.role_bearing else national_currencies
    by_source = _grouped(fold.facets, key=lambda held: held.provenance)
    ordered = sorted(by_source.items(), key=lambda item: (-_reach(item[1]), item[0]))
    availability = Availability(
        summary=AvailabilitySummary(
            places=_places(fold, fold.facets),
            entities=_entities(fold, fold.facets),
            first_year=_first_year(fold.facets),
            last_year=_last_year(fold.facets),
            sources=len(by_source),
            facets=len(fold.facets),
        ),
        default=_default(fold, profile, names),
        sources=[
            _source(fold, provenance, held, names, currencies)
            for provenance, held in ordered[:MAX_SOURCES]
        ],
        more_sources=max(len(ordered) - MAX_SOURCES, 0),
    )
    if profile.countries_fallback:
        return AvailabilityRead(availability=availability, gaps=(COUNTRIES_ONLY_GAP,))
    return AvailabilityRead(availability=availability)


def facet_items(
    *,
    held: HeldFacets,
    catalogue: Catalogue,
    labels: Mapping[str, str],
    where: FacetFilter = _OPEN,
) -> list[FacetItem]:
    """The admitted facets in `held` that `where` keeps, ordered by facet id: the items of the
    `facets` list relation, read as `build_availability` reads them."""
    names = _Names(catalogue=catalogue, labels=labels)
    return [
        FacetItem(
            ref=facet.facet_id,
            source=names.source(facet.provenance),
            unit=names.unit(facet.unit),
            measurement_method=names.method(facet.measurement_method),
            observation_period=facet.period,
            scaling=facet.scaling,
            first_year=_year_text(facet.first_year),
            last_year=_year_text(facet.last_year),
            observations=facet.observations,
            places=None if held.role_bearing else len(facet.reach),
        )
        for facet in held.facets
        if where.keeps(facet)
    ]


# --- reading the facets ------------------------------------------------------------------------


def held_facets(
    *, stat_var: str, profile: FacetProfile, coverage: CoverageIndex, catalogue: Catalogue
) -> HeldFacets:
    """`stat_var`'s admitted facets: a role-bearing variable's from `coverage`'s facet summary,
    each admitted by `catalogue` (`Catalogue.admit_series`), any other's from `profile`.
    `coverage` is the index the profile was read from and `catalogue` the one it was keyed on."""
    role_names = coverage.role_names.get(stat_var)
    if role_names is None:
        return HeldFacets(profile=profile, facets=_profile_facets(profile))
    summary = coverage.facet_summary.variable(stat_var)
    return HeldFacets(
        profile=profile,
        facets=_coverage_facets(summary, catalogue) if summary is not None else (),
        role_names=role_names,
        summary=summary,
    )


def _profile_facets(profile: FacetProfile) -> tuple[_Held, ...]:
    """The profile's candidates that carry an id and a provenance, which the graph read
    admitted at the places `places_by_facet` names."""
    held = [
        _Held(
            facet_id=facet.facet_id,
            provenance=facet.provenance_id,
            measurement_method=facet.measurement_method,
            unit=facet.unit,
            scaling=facet.scaling_factor,
            period=facet.observation_period,
            first_year=_year(facet.earliest_date),
            last_year=_year(facet.latest_date),
            observations=facet.obs_count,
            reach=profile.places_by_facet.get(facet.facet_id, frozenset()),
        )
        for facet in profile.candidates
        if facet.facet_id is not None and facet.provenance_id is not None
    ]
    return tuple(sorted(held, key=lambda h: h.facet_id))


def _coverage_facets(summary: VariableSummary, catalogue: Catalogue) -> tuple[_Held, ...]:
    """The summary's facets whose provenance `catalogue` admits, in facet id order."""
    return tuple(
        _Held(
            facet_id=facet.facet_id,
            provenance=provenance,
            measurement_method=facet.metadata.measurement_method,
            unit=facet.metadata.unit,
            scaling=facet.metadata.scaling_factor,
            period=facet.metadata.observation_period,
            first_year=facet.first_year,
            last_year=facet.last_year,
            observations=facet.observations,
            reach=frozenset().union(*facet.parties.values()),
        )
        for facet in summary.facets
        if (provenance := facet.metadata.provenance_id) is not None
        and provenance in catalogue.admit_series([provenance]).admitted
    )


# --- folding -----------------------------------------------------------------------------------


def _counted(
    held: HeldFacets,
    *,
    parties: Mapping[str, PartyRow],
    place_kind: Callable[[str], str] | None,
) -> _Fold:
    """`held`'s facets with how a card counts their entities: places by kind, or parties per
    graph role."""
    role_names, summary = held.role_names, held.summary
    if role_names is None:
        return _Fold(
            facets=held.facets,
            role_bearing=False,
            places_by_kind=(
                (lambda facets: _count_kinds(_union(facets), place_kind))
                if place_kind is not None
                else None
            ),
            entities=None,
        )
    graph_roles = {"donor": role_names.donor, "recipient": role_names.recipient}

    def entities(facets: Iterable[_Held]) -> dict[str, EntityCounts]:
        if summary is None:
            return {}
        counts = summary.party_counts(parties=parties, facet_ids={h.facet_id for h in facets})
        return {
            role_name: _entity_counts(found)
            for role, found in counts.items()
            if (role_name := graph_roles[role]) is not None
        }

    return _Fold(facets=held.facets, role_bearing=True, places_by_kind=None, entities=entities)


@dataclass(frozen=True, slots=True)
class _Names:
    catalogue: Catalogue
    labels: Mapping[str, str]

    def source(self, provenance: str) -> AvailabilitySource:
        row = self.catalogue.provenance(provenance)
        return AvailabilitySource(
            ref=provenance,
            label=row.name if row is not None else None,
            documentation=DOCUMENTATION_PATH.format(ref=provenance),
        )

    def unit_label(self, unit: str | None) -> str | None:
        if unit is None:
            return None
        iso = native_currency_iso(unit)
        if iso is not None:
            return currency_noun(iso, label=self.labels.get(unit))
        return self.labels.get(unit)

    def unit(self, unit: str | None) -> NodeRef | None:
        return NodeRef(ref=unit, label=self.unit_label(unit)) if unit is not None else None

    def method(self, method: str | None) -> NodeRef | None:
        return NodeRef(ref=method, label=self.labels.get(method)) if method is not None else None


def _default(fold: _Fold, profile: FacetProfile, names: _Names) -> AvailabilityDefault | None:
    by_id = {held.facet_id: held for held in fold.facets}
    served = [by_id[f.facet_id] for f in profile.default if f.facet_id in by_id]
    if not served:
        return None
    top = served[0]
    reached = _union(served)
    return AvailabilityDefault(
        facets=[held.facet_id for held in served],
        source=names.source(top.provenance),
        unit=names.unit(top.unit),
        measurement_method=names.method(top.measurement_method),
        observation_period=top.period,
        first_year=_first_year(served),
        last_year=_last_year(served),
        places=None if fold.role_bearing else len(reached),
        without_data=len(_union(fold.facets) - reached),
    )


def _source(
    fold: _Fold,
    provenance: str,
    held: Sequence[_Held],
    names: _Names,
    national_currencies: Mapping[str, str],
) -> SourceAvailability:
    by_series = _grouped(
        held, key=lambda h: (h.measurement_method or "", h.period or "", h.scaling or "")
    )
    series = sorted(
        by_series.values(),
        key=lambda facets: (
            -_reach(facets),
            -sum(f.observations for f in facets),
            facets[0].measurement_method or "",
            facets[0].period or "",
            facets[0].scaling or "",
        ),
    )
    return SourceAvailability(
        source=names.source(provenance),
        places=_places(fold, held),
        entities=_entities(fold, held),
        first_year=_first_year(held),
        last_year=_last_year(held),
        facets=len(held),
        series=[_series(fold, provenance, facets, names, national_currencies) for facets in series],
    )


def _series(
    fold: _Fold,
    provenance: str,
    held: Sequence[_Held],
    names: _Names,
    national_currencies: Mapping[str, str],
) -> Series:
    top = held[0]
    units = _units(fold, held, names, national_currencies)
    return Series(
        measurement_method=names.method(top.measurement_method),
        observation_period=top.period,
        scaling=top.scaling,
        first_year=_first_year(held),
        last_year=_last_year(held),
        places=None if fold.role_bearing else len(_union(held)),
        observations=sum(h.observations for h in held),
        units=units,
        binding=Binding(
            provenance=provenance,
            measurementMethod=top.measurement_method,
            unit=units.leading[0].ref,
        ),
    )


def _units(
    fold: _Fold,
    held: Sequence[_Held],
    names: _Names,
    national_currencies: Mapping[str, str],
) -> UnitsFold:
    """Every unit in the series, the two covering the most places leading, and the units that
    are each covered place's national currency."""
    by_unit = _grouped(held, key=lambda h: h.unit or "")
    ranked = sorted(by_unit.values(), key=lambda facets: (-_reach(facets), facets[0].unit or ""))
    national = [h for h in held if _is_national(h, national_currencies)]
    return UnitsFold(
        count=len(ranked),
        leading=[
            UnitCount(
                ref=facets[0].unit,
                label=names.unit_label(facets[0].unit),
                places=None if fold.role_bearing else _reach(facets),
            )
            for facets in ranked[:_LEADING_UNITS]
        ],
        more=max(len(ranked) - _LEADING_UNITS, 0),
        national_currency=(
            NationalCurrency(
                currencies=len({h.unit for h in national}), places=len(_union(national))
            )
            if national
            else None
        ),
    )


def _is_national(held: _Held, national_currencies: Mapping[str, str]) -> bool:
    """Whether `held`'s unit is a currency the graph names as the national currency of every
    place it covers."""
    iso = native_currency_iso(held.unit)
    return (
        iso is not None
        and bool(held.reach)
        and all(national_currencies.get(place) == iso for place in held.reach)
    )


def _places(fold: _Fold, held: Iterable[_Held]) -> dict[str, int] | None:
    return fold.places_by_kind(held) if fold.places_by_kind is not None else None


def _entities(fold: _Fold, held: Iterable[_Held]) -> dict[str, EntityCounts] | None:
    return fold.entities(held) if fold.entities is not None else None


def _entity_counts(counts: PartyCounts) -> EntityCounts:
    return EntityCounts(
        parties=counts.parties,
        private=counts.private,
        totals=counts.totals,
        residual=counts.residual,
        unclassified=counts.unclassified,
    )


def _count_kinds(places: frozenset[str], place_kind: Callable[[str], str]) -> dict[str, int]:
    return dict(sorted(Counter(place_kind(place) for place in places).items()))


def _grouped[K](held: Iterable[_Held], *, key: Callable[[_Held], K]) -> dict[K, list[_Held]]:
    """`held` grouped by `key`, each group keeping `held`'s order."""
    groups: dict[K, list[_Held]] = {}
    for item in held:
        groups.setdefault(key(item), []).append(item)
    return groups


def _union(held: Iterable[_Held]) -> frozenset[str]:
    return frozenset().union(*(h.reach for h in held))


def _reach(held: Iterable[_Held]) -> int:
    return len(_union(held))


def _first_year(held: Iterable[_Held]) -> str | None:
    return _year_text(min((h.first_year for h in held if h.first_year is not None), default=None))


def _last_year(held: Iterable[_Held]) -> str | None:
    return _year_text(max((h.last_year for h in held if h.last_year is not None), default=None))


def _year(date: str | None) -> int | None:
    return int(date[:4]) if date else None


def _year_text(year: int | None) -> str | None:
    return f"{year:04d}" if year is not None else None
