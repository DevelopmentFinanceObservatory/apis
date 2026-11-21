"""The last complete year of a DFO provenance's data for one variable or definition shape.

The bound graph release computes a year for each family under each provenance
(`graph_release.origin.ReleaseOrigin.complete_years`), and a `[[last_complete_period]]` entry
in the registry declares one for every family of a provenance, which wins. A frame whose last
period is later than the year holds a partial final period.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING, Final

from graph_release.context import bound_release_or_none
from origin_catalogue.families import FamilyKey
from origin_catalogue.registry import load_registry

if TYPE_CHECKING:
    from description.facts import VariableFacts

# The `FamilyKey` member each definition shape property names.
_SHAPE_MEMBERS: Final[Mapping[str, str]] = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "statType": "stat_type",
    "measurementQualifier": "measurement_qualifier",
}


@cache
def declared_periods() -> Mapping[str, int]:
    """The registry's declared last complete year of each provenance, read once."""
    return load_registry().declared_periods()


def last_complete_year(
    provenance: str,
    *,
    variables: Collection[str] = (),
    shape: Mapping[str, str] | None = None,
) -> int | None:
    """The last complete year of `provenance`'s data for `variables`, or else for the variables
    a definition `shape` names, `None` when nothing says.

    A year the registry declares for `provenance` wins. Otherwise the bound release answers
    with the earliest year of the families of `variables` it holds under `provenance`.
    Without one, `shape` maps graph properties (`populationType`, `measuredProperty`,
    `statType`, `measurementQualifier`) to the values a definition fixes, and the earliest
    year of the provenance's families whose members match every one answers. An empty
    `shape`, no bound release and a provenance the release holds no family of answer
    `None`."""
    declared = declared_periods().get(provenance)
    if declared is not None:
        return declared
    release = bound_release_or_none()
    if release is None:
        return None
    years = release.origin.complete_years.get(provenance)
    if not years:
        return None
    held = [
        years[family]
        for variable in variables
        if (family := release.origin.family_of(variable)) is not None and family in years
    ]
    if held:
        return min(held)
    if not shape:
        return None
    matched = [year for key, year in years.items() if _matches(key, shape)]
    return min(matched, default=None)


def version_last_complete_period(
    provenance: str, facts: VariableFacts | None, *, resolved: Sequence[str] = ()
) -> str | None:
    """`last_complete_year` of `provenance` for a `spec_` version, as a year: over `resolved`,
    the bare dcids of the variables its definition resolved to, then the one variable its
    record `facts` name, then the shape they record. `None` when nothing says."""
    variables = [*resolved]
    if facts is not None and facts.variable is not None:
        variables.append(facts.variable)
    year = last_complete_year(
        provenance,
        variables=variables,
        shape=None if facts is None else {fixed.property: fixed.value for fixed in facts.shape},
    )
    return None if year is None else str(year)


def _matches(key: FamilyKey, shape: Mapping[str, str]) -> bool:
    """Whether family `key` holds the value `shape` gives each property it names. A property
    that is not one of the five-tuple's matches no family."""
    return all(
        (member := _SHAPE_MEMBERS.get(prop)) is not None and getattr(key, member) == value
        for prop, value in shape.items()
    )


__all__ = ["declared_periods", "last_complete_year", "version_last_complete_period"]
