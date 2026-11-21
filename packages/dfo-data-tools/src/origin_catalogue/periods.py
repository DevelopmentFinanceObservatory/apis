"""The last complete year of each family of a DFO provenance's data, and the sources behind a
frame's rows that a partial final year is judged against.

The bound graph release computes a year for each family under each provenance
(`graph_release.origin.ReleaseOrigin.complete_years`), and a `[[last_complete_period]]` entry
in the registry declares one for every family of a provenance, which wins. A frame's rows of
one family under one source are partial past that family's year.

`source_periods` names, for each provenance a frame's rows carry, the source and family each
group of its rows comes from, with the family's last complete year and the last year those rows
reach. A version records it when it is minted (`datasets_api.manifest.last_complete_years`),
and a frame minted before versions recorded it reads it from the release bound to the load,
with no last year. `frame_periods` turns either into the `FramePeriod`s of one frame's rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from graph_release.context import bound_release_or_none
from origin_catalogue.families import FamilyKey
from origin_catalogue.registry import load_registry

type SourceRows = tuple[str, Sequence[str]]
"""A source of a frame's rows: its provenance and the bare dcids of the variables it serves."""


@dataclass(frozen=True, slots=True)
class FramePeriod:
    """The rows of one family under one source in a frame: `row_ref`, the provenance the rows
    carry (a composite's own for a composite's rows), `provenance`, the source they come from,
    `variables`, the frame's `stat_var` cells of the family, `last_complete_year`, the year the
    source held the family complete through, and `last_year`, the last year those rows reached
    when the version was minted, `None` when the version recorded none."""

    row_ref: str
    provenance: str
    variables: frozenset[str]
    last_complete_year: int
    last_year: int | None

    def holds(self, *, stat_var: object, ref: object) -> bool:
        """Whether a row of `stat_var` carrying provenance `ref` is one of these rows. A row
        carrying no provenance is matched by its variable alone."""
        return stat_var in self.variables and (
            ref is None or ref == self.row_ref or ref == self.provenance
        )

    def partial(self, reach: int) -> bool:
        """Whether rows of this family reaching `reach` hold a year past the last complete one.
        They reach no later than the version's own rows did."""
        last = reach if self.last_year is None else min(reach, self.last_year)
        return last > self.last_complete_year


@cache
def declared_periods() -> Mapping[str, int]:
    """The registry's declared last complete year of each provenance, read once."""
    return load_registry().declared_periods()


def counts_periods(provenances: Iterable[str]) -> bool:
    """Whether the bound release counts complete years (`ReleaseOrigin.complete_years`) or the
    registry declares one for any of `provenances`, so a source no year is given for is one
    neither decides, and not one nothing counted."""
    release = bound_release_or_none()
    if release is not None and release.origin.complete_years:
        return True
    declared = declared_periods()
    return any(provenance in declared for provenance in provenances)


def family_of(variable: str) -> FamilyKey | None:
    """The family the bound release files `variable` (a bare dcid) under, `None` with no
    release bound or none listing it."""
    release = bound_release_or_none()
    return None if release is None else release.origin.family_of(variable)


def complete_year(provenance: str, family: FamilyKey | None) -> int | None:
    """The last complete year of `family` under `provenance`: the registry's declared year,
    which wins for every family, else the bound release's year for the family. `None` when
    neither gives one, and for an unknown family the registry declares nothing for."""
    declared = declared_periods().get(provenance)
    if declared is not None:
        return declared
    release = bound_release_or_none()
    if release is None or family is None:
        return None
    return release.origin.complete_years.get(provenance, {}).get(family)


def source_periods(
    *,
    row_refs: Sequence[str],
    sources: Sequence[SourceRows],
    ends: Mapping[tuple[str, str], int] | None,
) -> dict[str, list[dict[str, object]]]:
    """For each of `row_refs`, the provenances a frame's rows carry, one entry per source and
    family of `sources`' variables with a known last complete year (`complete_year`): its
    `provenance`, its `family` (the `FamilyKey` members, `None` for variables no family lists),
    `last_complete_year` and `last_year`, the latest of `ends[(row ref, variable)]` over the
    family's variables. With `ends`, a variable no row reaches is left out; with `ends` `None`
    every variable counts and `last_year` is `None`. A row ref none of whose sources gives a
    year is left out."""
    periods: dict[str, list[dict[str, object]]] = {}
    for row_ref in row_refs:
        found: list[dict[str, object]] = []
        for provenance, variables in sources:
            lasts: dict[FamilyKey | None, int | None] = {}
            for variable in variables:
                last = None if ends is None else ends.get((row_ref, variable))
                if ends is not None and last is None:
                    continue
                family = family_of(variable)
                held = lasts.get(family)
                lasts[family] = (
                    held if held is not None and (last is None or held >= last) else last
                )
            for family, last in lasts.items():
                year = complete_year(provenance, family)
                if year is not None:
                    found.append(
                        {
                            "provenance": provenance,
                            "family": None if family is None else list(family),
                            "last_complete_year": year,
                            "last_year": last,
                        }
                    )
        if found:
            periods[row_ref] = found
    return periods


def frame_periods(
    recorded: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    cells: Iterable[tuple[str, str | None]],
) -> tuple[FramePeriod, ...]:
    """The `FramePeriod` of each `recorded` entry (`source_periods`' shape) holding a row of
    the frame whose `(stat_var, provenance)` cells are `cells`. A row is of an entry when it
    carries the entry's row ref or source, or no provenance, and the bound release files its
    variable under the entry's family; an entry whose family is `None` holds a row of any
    family. With no release bound, no row is of a named family."""
    held = sorted(set(cells), key=lambda cell: (cell[0], cell[1] or ""))
    families = {stat_var: family_of(stat_var.removeprefix("dcid:")) for stat_var, _ in held}
    periods: list[FramePeriod] = []
    for row_ref, entries in recorded.items():
        for entry in entries:
            provenance, complete = entry.get("provenance"), entry.get("last_complete_year")
            last, family = entry.get("last_year"), entry.get("family")
            if not isinstance(provenance, str) or not isinstance(complete, int):
                continue
            key = FamilyKey._make(family) if isinstance(family, list) else None
            variables = frozenset(
                stat_var
                for stat_var, ref in held
                if (ref is None or ref in (row_ref, provenance))
                and (key is None or families[stat_var] == key)
            )
            if variables:
                periods.append(
                    FramePeriod(
                        row_ref=row_ref,
                        provenance=provenance,
                        variables=variables,
                        last_complete_year=complete,
                        last_year=last if isinstance(last, int) else None,
                    )
                )
    return tuple(periods)


__all__ = [
    "FramePeriod",
    "SourceRows",
    "complete_year",
    "counts_periods",
    "declared_periods",
    "family_of",
    "frame_periods",
    "source_periods",
]
