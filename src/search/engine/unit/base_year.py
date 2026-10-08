"""Read the deflator base year the live conversion tables declare.

A constant-price ask that names no year completes to the year in which every
conversion table's deflator series equals 100. Each table provenance named by
`RECIPE_BODIES[*].table_provenance` declares it in `deflatorBaseYear`, and one
`node_arcs_many` read fetches it from all of them with their
`lastDataRefreshDate`.

A provenance qualifies under the same rule the rate-table loader applies: it
must exist and carry `lastDataRefreshDate`, since a table without one cannot
load. It must also declare exactly one base year, and every provenance must
declare the same one. Any other state raises `TableBaseYearError`, which Search
serves as 501 `capability_unavailable`. A transport failure raises the graph
client's own error.

The first year read is kept for the life of the graph client, which is the
life of the process in production. A failed read is not kept, so the next ask
reads again. Rebasing leaves every ratio of deflator values unchanged, so a
completed request converts correctly against later tables whatever year they
declare.
"""

from __future__ import annotations

import re
import threading
import weakref

from dfo_platform.named import named_capped
from materialise.rate_tables import RateTableInvalidError
from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.clients.graph import EngineGraphClient
from search.engine.unit.lexicon import MAX_YEAR, MIN_YEAR

BASE_YEAR_PROPERTY = "deflatorBaseYear"
_REFRESH_PROPERTY = "lastDataRefreshDate"

TABLE_PROVENANCES: tuple[str, ...] = tuple(
    sorted({recipe.table_provenance for recipe in RECIPE_BODIES.values()})
)
"""Every conversion-table provenance this deployment's recipes read."""

# The graph serves the integer as its decimal text ("2024").
_YEAR_TEXT_RE = re.compile(r"[0-9]{4}")


class TableBaseYearError(RateTableInvalidError):
    """501 `capability_unavailable` when the conversion-table provenances do not
    establish one base year: a provenance is absent, carries no
    `lastDataRefreshDate`, declares no base year, several, or one that is not a
    four-digit year in range, or two provenances declare different years.
    """


_lock = threading.Lock()
_years: weakref.WeakKeyDictionary[EngineGraphClient, int] = weakref.WeakKeyDictionary()


def declared_base_year(graph: EngineGraphClient, *, deadline: float | None = None) -> int:
    """The base year every conversion-table provenance declares.

    Reads the graph once per `graph` client and keeps the first year that
    qualifies. `deadline` bounds the read. Raises `TableBaseYearError` when
    the provenances do not establish one year.
    """
    with _lock:
        cached = _years.get(graph)
    if cached is not None:
        return cached
    year = _read_base_year(graph, deadline=deadline)
    with _lock:
        return _years.setdefault(graph, year)


def _read_base_year(graph: EngineGraphClient, *, deadline: float | None) -> int:
    arcs_by_id = graph.node_arcs_many(
        TABLE_PROVENANCES, properties=(_REFRESH_PROPERTY, BASE_YEAR_PROPERTY), deadline=deadline
    )
    years: dict[str, int] = {}
    for provenance_id in TABLE_PROVENANCES:
        arcs = arcs_by_id.get(provenance_id)
        if arcs is None:
            raise TableBaseYearError(
                f"conversion-table provenance {provenance_id!r} is absent from the graph"
            )
        if not arcs.get(_REFRESH_PROPERTY):
            raise TableBaseYearError(
                f"conversion-table provenance {provenance_id!r} carries no "
                f"{_REFRESH_PROPERTY}, so its tables cannot load"
            )
        years[provenance_id] = _parse_year(provenance_id, arcs.get(BASE_YEAR_PROPERTY, ()))
    if len(set(years.values())) != 1:
        declared = ", ".join(f"{pid}={year}" for pid, year in years.items())
        raise TableBaseYearError(
            f"conversion-table provenances declare different {BASE_YEAR_PROPERTY} values "
            f"({declared}), so a constant-price ask has no single base year to complete to"
        )
    return next(iter(years.values()))


def _parse_year(provenance_id: str, values: tuple[str, ...]) -> int:
    """The one four-digit year in `values`, within the binder's year bounds."""
    if len(values) != 1:
        raise TableBaseYearError(
            f"conversion-table provenance {provenance_id!r} declares {len(values)} "
            f"{BASE_YEAR_PROPERTY} values {named_capped(values)}; exactly one is required"
        )
    (text,) = values
    if _YEAR_TEXT_RE.fullmatch(text) is None or not MIN_YEAR <= int(text) <= MAX_YEAR:
        raise TableBaseYearError(
            f"conversion-table provenance {provenance_id!r} declares {BASE_YEAR_PROPERTY} "
            f"{text!r}, which is not a year in [{MIN_YEAR}, {MAX_YEAR}]"
        )
    return int(text)
