"""The runs of held-back years a read's sources hold, read from facet metadata alone, so a
read states them whatever its window and Search and the mint state the same runs.

A read holds back two kinds of years unless its window's `include` names them
(`materialise.held_back`):

- `partial`: a source's partly reported period (`origin_catalogue.partial_period_of`). The
  run starts at that period's year and ends at the newest year any served actual facet of the
  source holds (`Facet.latest_date`). A source whose served facets end before the period holds
  no partial run.
- `projected`: the years a projection facet (`method_status.is_projection`) continues a
  served series past its actual years, by the rule the mint continues a series by
  (`facets.continuation`, `facets.continues_series`). The run starts at the earliest year a
  continuing projection facet holds after the earliest year a continued series ends, and ends
  at the newest year a continuing projection facet holds. A read serving only projection
  facets, as one binding the projection method, is projected over their years.

A served facet is one the read dispatches. The read dispatches no projection facet it holds
back, so the projected runs read every candidate of the series.

A read's runs are `read_runs` over each member's candidates and the facet ids it
dispatches, merged across places as a mint probes them. The mint states them on its result
(`generation`), Search on a per-series answer (`search.engine.retrieval.held_back`) and on a
composite answer from the component plans it settles
(`search.engine.region.composite_answer.read_composite_plans`), each from the coverage probe
a mint reads.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from materialise.facets import at_cadence, cadences_by_member, continues_series
from materialise.graph import Facet
from materialise.held_back import HeldKind, cut_catalogue
from materialise.method_status import is_projection
from materialise.partial_period import start_of
from origin_catalogue import Catalogue
from origin_catalogue.catalogue import Undocumented, partial_period_of

__all__ = [
    "DatedFacet",
    "HeldRun",
    "merged_runs",
    "partial_runs",
    "projected_runs",
    "read_runs",
]


@dataclass(frozen=True, slots=True)
class HeldRun:
    """One run of held-back years of `kind` from `source`: `start_year` to `end_year`."""

    kind: HeldKind
    source: str
    start_year: int
    end_year: int


class DatedFacet(Protocol):
    """The facet metadata a partial run is read from."""

    @property
    def provenance_id(self) -> str | None: ...

    @property
    def measurement_method(self) -> str | None: ...

    @property
    def latest_date(self) -> str | None: ...


def _year_of(date: str | None) -> int | None:
    try:
        return int(str(date)[:4])
    except (ValueError, TypeError):
        return None


def _is_actual(facet: DatedFacet) -> bool:
    return not is_projection(facet.provenance_id, facet.measurement_method)


def _partial_start(catalogue: Catalogue, source: str) -> int | None:
    """The first year of the period `catalogue` marks `source`'s import as holding only in
    part, None when it marks none, documents none, or the mark cannot be read."""
    try:
        mark = partial_period_of(catalogue, source)
    except Undocumented:
        return None
    if mark is None:
        return None
    try:
        return int(start_of(mark)[:4])
    except ValueError:
        return None


def partial_runs(served: Iterable[DatedFacet], *, catalogue: Catalogue) -> Iterator[HeldRun]:
    """A partial run per source of the actual facets of `served` whose newest year reaches
    the period `catalogue` marks partial, by source."""
    newest: dict[str, int] = {}
    for f in served:
        year = _year_of(f.latest_date)
        if f.provenance_id is not None and year is not None and _is_actual(f):
            newest[f.provenance_id] = max(year, newest.get(f.provenance_id, year))
    for source, end in sorted(newest.items()):
        start = _partial_start(catalogue, source)
        if start is not None and start <= end:
            yield HeldRun(kind="partial", source=source, start_year=start, end_year=end)


def projected_runs(series: Iterable[tuple[Sequence[Facet], Sequence[Facet]]]) -> Iterator[HeldRun]:
    """A projected run per source whose projection facets continue a series of `series`, each
    given as its served facets and its candidates, past the year its actual facets end.

    A series' actual served facets are continued as the mint continues them
    (`facets.continuation`): by a projection candidate under the provenance of the first of
    them, in a unit any of them states, at its cadence, reaching past the earliest year any
    series they hold ends (`Facet.earliest_series_end`). A series whose served facets are all
    projection facets, as under a request binding the projection method, is projected over
    their years. A source's run starts at the earliest year a continuing facet holds past
    such an end, the year after it at the soonest, and ends at the newest year a continuing
    or served projection facet holds."""
    start: dict[str, int] = {}
    end: dict[str, int] = {}

    def held(source: str, first: int, last: int) -> None:
        start[source] = min(first, start.get(source, first))
        end[source] = max(last, end.get(source, last))

    for served, candidates in series:
        actual = [f for f in served if _is_actual(f)]
        if not actual:
            for f in served:
                first, last = _year_of(f.earliest_date), _year_of(f.latest_date)
                if f.provenance_id is not None and first is not None and last is not None:
                    held(f.provenance_id, first, last)
            continue
        lead = actual[0]
        ends = [year for f in actual if (year := _year_of(f.earliest_series_end)) is not None]
        if not ends:
            continue
        actual_end = min(ends)
        (cadence_of,) = cadences_by_member([candidates])
        serves = partial(
            at_cadence, cadence=cadence_of.get(lead.facet_id or ""), cadence_of=cadence_of
        )
        units = {f.unit for f in actual}
        for c in candidates:
            if c.provenance_id is None or _is_actual(c):
                continue
            last = _year_of(c.latest_date)
            if last is not None and continues_series(
                c, provenance=lead.provenance_id, units=units, after=actual_end, serves=serves
            ):
                first = max(actual_end + 1, _year_of(c.earliest_date) or actual_end + 1)
                held(c.provenance_id, first, last)
    for source in sorted(end):
        if start[source] <= end[source]:
            yield HeldRun(
                kind="projected", source=source, start_year=start[source], end_year=end[source]
            )


def merged_runs(runs: Iterable[HeldRun]) -> tuple[HeldRun, ...]:
    """`runs` merged to one per kind and source, from the earliest start to the newest end,
    partial runs first, then by source."""
    by_key: dict[tuple[HeldKind, str], HeldRun] = {}
    for run in runs:
        known = by_key.get((run.kind, run.source))
        by_key[(run.kind, run.source)] = (
            run
            if known is None
            else HeldRun(
                kind=run.kind,
                source=run.source,
                start_year=min(run.start_year, known.start_year),
                end_year=max(run.end_year, known.end_year),
            )
        )
    return tuple(by_key[key] for key in sorted(by_key, key=lambda k: (k[0] != "partial", k[1])))


def read_runs(members: Iterable[tuple[Sequence[Facet], Collection[str]]]) -> tuple[HeldRun, ...]:
    """The runs of partial and projected years the sources of `members`, each its candidates
    and the facet ids the read dispatches of them, hold past their served actual years,
    merged per kind and source (`merged_runs`). They are read from facet metadata, so a read
    states them whatever its window and whatever rows it cut.

    Raises:
        CatalogueNotBound: no origin catalogue is bound, so partial marks cannot be read
            (`held_back.cut_catalogue`).
    """
    catalogue = cut_catalogue()
    series = [
        ([c for c in candidates if c.facet_id in dispatched], candidates)
        for candidates, dispatched in members
    ]
    return merged_runs(
        [
            *partial_runs((f for served, _ in series for f in served), catalogue=catalogue),
            *projected_runs(series),
        ]
    )
