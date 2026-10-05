"""The countries where an example's StatVar has data, which are the places a
catalogue answer may fill its template with.

A StatVar the coverage index holds takes its places from the coverage rows,
on the role the example's place fills. Any other StatVar takes them from
the origin catalogue's base variable table. A StatVar with neither source,
such as a DFO variable outside the coverage index or a UN System variable,
has no known places, so the example is refused.
"""

from __future__ import annotations

from typing import Literal

from origin_catalogue.catalogue import Catalogue
from search.catalogue.counts import COUNTRY_PREFIX
from search.catalogue.examples import Example
from series_coverage.index import CoverageIndex

type PresenceReason = Literal[
    "no_presence_source", "role_required", "role_without_coverage", "default_place_without_data"
]


class PresenceRefusal(Exception):
    """An example whose places cannot be known, or whose default place has
    no data."""

    def __init__(self, reason: PresenceReason, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason: PresenceReason = reason
        self.detail = detail


def example_places(
    example: Example, *, catalogue: Catalogue, coverage: CoverageIndex
) -> frozenset[str]:
    """The `country/` dcids where `example.stat_var` has data, its default
    place among them. Raises `PresenceRefusal` otherwise."""
    stat_var = example.stat_var
    in_coverage = stat_var in coverage.role_names
    if example.role is not None:
        if not in_coverage:
            raise PresenceRefusal(
                "role_without_coverage",
                f"{stat_var} names role {example.role!r}, but the coverage index holds no rows "
                "for it",
            )
        places = coverage.open_axis(variables=[stat_var], axis=example.role)
    elif in_coverage:
        raise PresenceRefusal(
            "role_required",
            f"the coverage index holds {stat_var}, so the example must name the role its place "
            "fills",
        )
    else:
        base = catalogue.base_places(stat_var)
        if base is None:
            raise PresenceRefusal(
                "no_presence_source",
                f"{stat_var} is neither in the coverage index nor a base variable in the origin "
                "catalogue",
            )
        places = base
    countries = frozenset(place for place in places if place.startswith(COUNTRY_PREFIX))
    if example.place not in countries:
        raise PresenceRefusal(
            "default_place_without_data", f"{stat_var} has no data at {example.place}"
        )
    return countries


__all__ = ["PresenceReason", "PresenceRefusal", "example_places"]
