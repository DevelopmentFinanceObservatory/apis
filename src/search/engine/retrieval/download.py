"""State the file a mint of an answer's Spec writes before the first download
(`ResolvedSpec.download`).

The counts are what the answer's read plan serves over the identity's whole window, the
window a mint reads: every year when the identity names none, the plan's range for a range,
and each member's elected year for `latest` (`coverage.latest_election`). A preview
reads `BulkPlan.served_years` under `latest`, one year, and the counts ignore it.

- A role-bearing plan counts from the coverage index (`CoverageIndex.served_extent`): the
  rows it counts (`coverage.counted_rows`) less the rows a conversion leaves out, the rows
  its headline counts in one year.
- A standard plan, which binds no roles, counts from facet metadata: each served series' points
  (`read_plan.series_points`) and dates (`read_plan.series_dates`), as coverage does.

Under a unit request each series counts only the years its conversion keeps
(`unit.conversion.CountedConversion`): at constant prices those its entity's deflator
holds, none for an entity whose deflator lacks the base year, and none for an entity no
currency names under a conversion into each entity's own currency. The rates leave no row
out, since the period a unit request shows already holds them.

Rows are null when a member the plan reads is missing from the release's coverage view, since
the mint reads it and the count would understate. They are an upper bound (`rows_exact`
false) under a conversion the count cannot follow in memory
(`unit.conversion.counted_exact`), and when a series' facets carry no complete date list
and their observation counts can overlap or run past the range (`_points_exact`), or a
conversion keeps only some of the years they span. The window is null when a counted series' facets leave its first or
last year unknown, since the file's years could then run past either bound. The licence is
the one served provenance's source-index entry's, as the origin catalogue holds it
(`origin_catalogue.CardFacts`), so stating it reads nothing over the network.

The years with data (`ResolvedSpec.years_with_data`, `answer_years`) read the same sources
over every date of the series the plan holds, with no opt-in, so they reach past the
identity's window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from origin_catalogue import current_catalogue, entry_ref
from search.engine.retrieval.coverage import (
    counted_rows,
    coverage_exclusions,
    held_rows,
    latest_election,
)
from search.engine.retrieval.read_plan import (
    PairPlan,
    PlanSeries,
    PlanYears,
    ReadPlan,
    dated_without_complete_dates,
    facet_dates,
    kept_points,
    opt_in,
    series_dates,
    series_point_dates,
    series_points,
    year_of,
)
from search.engine.unit.conversion import (
    UNDECIDED,
    CountConversion,
    CountedConversion,
    counted_exact,
)
from search.models import DownloadFacts, DownloadLicence, TimeWindow, YearsWithData
from series_coverage.schema import ABSENT_ENTITY, MASK_WORD_BITS

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from search.engine.clients.graph import Facet
    from search.models import RequestedWindow
    from series_coverage.index import CoverageIndex, KeptYears
    from series_coverage.index import Role as CoverageRole


@dataclass(frozen=True)
class Extent:
    """The rows, series and years a read serves, and whether `rows` is exact or an upper
    bound."""

    rows: int
    series: int
    window: TimeWindow | None
    exact: bool


def download_facts(
    plan: ReadPlan,
    coverage_index: CoverageIndex,
    *,
    requested_window: RequestedWindow | None,
    conversion: CountConversion,
    provenance: str | None,
) -> DownloadFacts:
    """The facts of the default file a mint of the answer's Spec writes.

    Args:
        plan: The answer's read plan, the plan its coverage and headline count.
        coverage_index: The request's coverage index, which a role-bearing plan counts from.
        requested_window: The identity's window: None for every year, `"latest"`, or a
            range, whose bounds the plan's years carry.
        conversion: How the identity's unit request changes the rows
            (`unit.conversion.counted_conversion`).
        provenance: The one provenance the answer serves, or None when it serves several.
    """
    years: PlanYears = (
        plan.years
        if isinstance(plan.years, TimeWindow)
        else ("latest" if requested_window == "latest" else None)
    )
    kept = conversion if isinstance(conversion, CountedConversion) else None
    extent = (
        pair_extent(plan, years, kept)
        if isinstance(plan, PairPlan) and plan.roles is None
        else _coverage_extent(plan, coverage_index, years, kept)
    )
    licence = provenance_licence(provenance) if provenance is not None else None
    if extent is None:
        return DownloadFacts(rows=None, rows_exact=False, series=None, window=None, licence=licence)
    return DownloadFacts(
        rows=extent.rows,
        rows_exact=counted_exact(extent.exact, conversion),
        series=extent.series,
        window=extent.window,
        licence=licence,
    )


def pair_extent(
    plan: PairPlan, years: PlanYears, conversion: CountedConversion | None = None
) -> Extent:
    """A standard plan's rows, series and years from its served series' facet metadata,
    read under `years`, each series keeping the years `conversion` keeps. The coverage
    probe counts a slot option's plan here too (`standard.standard_option_coverage`)."""
    rows = series = 0
    exact = True
    found: list[int | None] = []
    for s in plan.served:
        points, points_exact = series_rows(
            s, years, conversion.kept_years(dict(s.role_tuple)) if conversion else None
        )
        exact = exact and points_exact
        if not points:
            continue
        rows += points
        series += 1
        # Each facet's own bounds, since a series' hull drops a bound one facet leaves unknown.
        spans = (
            [series_dates(s, years)]
            if years == "latest"
            else [facet_dates(f, years) for f in s.group if f.obs_count > 0]
        )
        found.extend(year_of(date) for span in spans if span is not None for date in span)
    known = [year for year in found if year is not None]
    window = (
        TimeWindow(start_year=min(known), end_year=max(known))
        if known and len(known) == len(found)
        else None
    )
    return Extent(rows=rows, series=series, window=window, exact=exact)


def series_rows(
    series: PlanSeries, years: PlanYears, kept: frozenset[int] | None = None
) -> tuple[int, bool]:
    """The rows one served series of a standard plan writes under `years`, in the years
    `kept` names (every year when None), and whether the count is exact: its latest point
    alone under `latest`, else its points (`series_points`), exact when `_points_exact`
    holds. A series without a complete date list counts every point it holds, exact only
    when `kept` holds every year of its span."""
    if years == "latest":
        latest = series_dates(series, years)
        held = latest is not None and (kept is None or year_of(latest[0]) in kept)
        return (1 if held else 0), True
    read = years if isinstance(years, TimeWindow) else None
    if kept is None:
        return series_points(series, read), _points_exact(series, read)
    if not dated_without_complete_dates(series.group):
        dates = series_point_dates(series, read)
        return sum(1 for d in dates if year_of(d) in kept), True
    span = series_dates(series, years)
    first, last = (year_of(span[0]), year_of(span[1])) if span is not None else (None, None)
    whole = first is not None and last is not None and kept.issuperset(range(first, last + 1))
    return series_points(series, read), _points_exact(series, read) and whole


def _points_exact(series: PlanSeries, read: TimeWindow | None) -> bool:
    """Whether `series_points` counts `series` exactly. A group without a complete date
    list counts its facets' whole observation counts: exact for one data-bearing facet
    whose span lies inside `read`, an upper bound for several, whose dates can coincide,
    or for a span `read` clips."""
    facets = [f for f in series.group if f.obs_count > 0]
    if not dated_without_complete_dates(facets):
        return True
    if len(facets) > 1:
        return False
    if read is None:
        return True
    first, last = year_of(facets[0].earliest_date), year_of(facets[0].latest_date)
    return (
        first is not None
        and last is not None
        and (read.start_year is None or first >= read.start_year)
        and (read.end_year is None or last <= read.end_year)
    )


def _coverage_extent(
    plan: ReadPlan,
    index: CoverageIndex,
    years: PlanYears,
    conversion: CountedConversion | None,
) -> Extent | None:
    """A role-bearing plan's rows, series and years from the coverage index, each series
    keeping the years `conversion` keeps; None when the index lacks a member the plan
    reads."""
    if plan.roles is None or not all(index.covers(sv) for sv in plan.members):
        return None
    read = years if isinstance(years, TimeWindow) else None
    latest = latest_election(plan, years)
    extent = index.served_extent(
        variables=plan.members,
        bound=plan.roles.bound,
        served=counted_rows(plan, latest),
        excluded=coverage_exclusions(plan),
        first_year=read.start_year if read is not None else None,
        last_year=read.end_year if read is not None else None,
        latest=latest,
        kept_years=(
            coverage_kept_years(conversion, plan.roles.role_names)
            if conversion is not None
            else None
        ),
        include=opt_in(years),
    )
    if extent is None:
        return Extent(rows=0, series=0, window=None, exact=True)
    return Extent(
        rows=extent.rows,
        series=extent.series,
        window=TimeWindow(start_year=extent.first_year, end_year=extent.last_year),
        exact=True,
    )


def answer_years(
    plan: ReadPlan, coverage_index: CoverageIndex | None, *, conversion: CountConversion
) -> YearsWithData | None:
    """The years the series `plan` holds have data in (`YearsWithData`): the series it holds
    under its window before a `latest` window reduces them (`PairPlan.held`), each read under
    every date, unclipped by the window and with no opt-in, so a source's partial period and
    projected facets hold no year. Under a unit request each series keeps the years
    `conversion` keeps, as the download counts them.

    - A role-bearing plan reads the coverage index (`CoverageIndex.series_years`) over the
      rows it holds (`coverage.held_rows`) less those a conversion leaves out
      (`coverage.coverage_exclusions`).
    - A standard plan reads its held series' facet metadata alone (`_facet_years`), so it
      needs no coverage index.

    None when the conversion's effect cannot be read in memory (`UNDECIDED`), for a
    role-bearing plan with no coverage index or one lacking a member the plan reads, for a
    standard plan with a data-bearing facet whose first or last date is unknown, and when no
    series holds a year."""
    if conversion == UNDECIDED:
        return None
    kept = conversion if isinstance(conversion, CountedConversion) else None
    if isinstance(plan, PairPlan) and plan.roles is None:
        years: set[int] = set()
        for s in plan.held:
            allowed = kept.kept_years(dict(s.role_tuple)) if kept is not None else None
            for facet in s.group:
                held = _facet_years(facet)
                if held is None:
                    return None
                years.update(held if allowed is None else held & allowed)
        return _years_with_data(years)
    if (
        plan.roles is None
        or coverage_index is None
        or not all(coverage_index.covers(sv) for sv in plan.members)
    ):
        return None
    found = coverage_index.series_years(
        variables=plan.members,
        bound=plan.roles.bound,
        served=held_rows(plan),
        excluded=coverage_exclusions(plan),
        kept_years=(coverage_kept_years(kept, plan.roles.role_names) if kept is not None else None),
    )
    if found is None:
        return None
    epoch = coverage_index.horizon.epoch_year
    held_words = np.bitwise_or.reduce(found.masks, axis=0).tolist()
    return _years_with_data(
        {
            epoch + MASK_WORD_BITS * word + bit
            for word, value in enumerate(held_words)
            for bit in range(int(value).bit_length())
            if int(value) >> bit & 1
        }
    )


def _facet_years(facet: Facet) -> frozenset[int] | None:
    """The years `facet` holds under every date with no opt-in: those of the dates it keeps
    (`kept_points`) when it lists them all, else every year of its kept span (`facet_dates`).
    Empty for a facet holding no point; None when its span has an unknown bound."""
    if facet.obs_count <= 0:
        return frozenset()
    if facet.dates and facet.dates_complete:
        return frozenset(
            year
            for d in kept_points(facet, facet.dates, include=frozenset())
            if (year := year_of(d)) is not None
        )
    span = facet_dates(facet, None)
    if span is None:
        return frozenset()
    first, last = year_of(span[0]), year_of(span[1])
    if first is None or last is None:
        return None
    return frozenset(range(first, last + 1))


def _years_with_data(years: set[int]) -> YearsWithData | None:
    """`years` as their first and last year and the years between them missing from it."""
    if not years:
        return None
    first, last = min(years), max(years)
    gaps = sorted(set(range(first, last + 1)) - years)
    return YearsWithData(start_year=first, end_year=last, gap_years=gaps or None)


@dataclass(frozen=True)
class EntityYears:
    """The years an entity's served rows hold that a conversion leaves out (`lost`) and
    keeps (`kept`)."""

    lost: frozenset[int]
    kept: frozenset[int]


def deflator_omissions(
    plan: ReadPlan, coverage_index: CoverageIndex | None, conversion: CountedConversion
) -> dict[str, EntityYears]:
    """Each entity on `conversion`'s deflator role whose rows `plan` serves and the
    conversion leaves out for a missing deflator, with the years of those rows and of the
    rows it keeps, from the counts' own series years: a role-bearing plan's from the
    coverage index (`CoverageIndex.series_years`), a standard plan's from its facets' dates,
    or every year of their span without a complete date list. Empty at current prices."""
    role = conversion.deflator_role
    if role is None:
        return {}
    served: dict[str, set[int]] = {}
    for entity, years in _served_entity_years(plan, coverage_index, role):
        served.setdefault(entity, set()).update(years)
    found: dict[str, EntityYears] = {}
    for entity, years in sorted(served.items()):
        kept = conversion.kept_years({role: entity})
        if kept is not None and (lost := frozenset(years) - kept):
            found[entity] = EntityYears(lost=lost, kept=frozenset(years) & kept)
    return found


def _served_entity_years(
    plan: ReadPlan, index: CoverageIndex | None, role: str
) -> Iterator[tuple[str, frozenset[int]]]:
    """Each served series' entity on graph role `role` with the years it serves."""
    years: PlanYears = plan.years
    if isinstance(plan, PairPlan) and plan.roles is None:
        for s in plan.served:
            entity = dict(s.role_tuple).get(role)
            if entity is not None:
                yield entity, _series_years(s, years)
        return
    if plan.roles is None or index is None:
        return
    coverage_role = next(
        (cov for cov, graph in plan.roles.role_names.items() if graph == role), None
    )
    read = years if isinstance(years, TimeWindow) else None
    latest = latest_election(plan, years)
    found = index.series_years(
        variables=plan.members,
        bound=plan.roles.bound,
        served=counted_rows(plan, latest),
        excluded=coverage_exclusions(plan),
        first_year=read.start_year if read is not None else None,
        last_year=read.end_year if read is not None else None,
        latest=latest,
        include=opt_in(years),
    )
    if found is None or coverage_role is None:
        return
    epoch = index.horizon.epoch_year
    for (_sv, recipient, donor), bits in zip(found.keys(), found.year_bits(), strict=True):
        entity = recipient if coverage_role == "recipient" else donor
        if entity != ABSENT_ENTITY:
            yield entity, frozenset(epoch + i for i in range(bits.bit_length()) if bits >> i & 1)


def _series_years(series: PlanSeries, years: PlanYears) -> frozenset[int]:
    """The years a standard series serves under `years`: its kept dates' years when its
    facets carry them all (`series_point_dates`), else every year of its span."""
    if years != "latest" and not dated_without_complete_dates(series.group):
        read = years if isinstance(years, TimeWindow) else None
        return frozenset(
            year for d in series_point_dates(series, read) if (year := year_of(d)) is not None
        )
    span = series_dates(series, years)
    first, last = (year_of(span[0]), year_of(span[1])) if span is not None else (None, None)
    return (
        frozenset(range(first, last + 1)) if first is not None and last is not None else frozenset()
    )


def coverage_kept_years(
    conversion: CountedConversion, role_names: Mapping[CoverageRole, str]
) -> KeptYears:
    """`conversion`'s kept years for a coverage series, given its recipient and donor,
    each placed on the graph role `role_names` names for it."""
    recipient_role = role_names.get("recipient")
    donor_role = role_names.get("donor")

    def kept(recipient: str, donor: str) -> frozenset[int] | None:
        entities = {
            role: dcid
            for role, dcid in ((recipient_role, recipient), (donor_role, donor))
            if role is not None and dcid != ABSENT_ENTITY
        }
        return conversion.kept_years(entities)

    return kept


def provenance_licence(provenance: str) -> DownloadLicence | None:
    """The licence of `provenance`'s source-index entry (`origin_catalogue.entry_ref`), or
    None when the catalogue does not list the provenance or its entry's provenances name no
    single licence."""
    catalogue = current_catalogue()
    row = catalogue.provenance(provenance)
    facts = catalogue.dataset_facts(entry_ref(row)) if row is not None else None
    if facts is None or facts.license_type is None:
        return None
    return DownloadLicence(label=facts.license_type)
