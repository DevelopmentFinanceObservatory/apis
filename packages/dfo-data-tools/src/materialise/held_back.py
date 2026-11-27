"""Whether a frame row is one a read holds back unless its window opts in to it
(`window.RequestedWindow.include`): a row of the period its source's import holds only part
of, or a projected row.

A row is judged by its source, its `measurementMethod` and its `time_period` cell:

- `projected`: the `(source, measurementMethod)` pair is declared a projection pair
  (`method_status.projection_methods`), whatever period the row holds. An estimate under an
  undeclared pair is an actual row.
- `partial`: the row reaches the period its source's import holds only part of
  (`origin_catalogue.partial_period_of`, `partial_period.reaches`). A declared dataset's
  rows carry its id, and they reach a period any of its member provenances marks
  (`partial_mark`).
- `unknown`: the catalogue documents no such source, the row names none, or the frame grammar
  cannot read the mark or the period. Nothing says, so the row is served.

A row's source is its `provenance` cell, and in a composite's frame the component serving its
variable (`source_of`), since every row of a composite carries the composite's provenance. The
mint cuts held-back rows (`columnar.held_back_mask`) and records what an opted-in frame holds
(`datasets_api.manifest.incomplete_after`) by the one `judge`, so the cut and the record agree.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache
from typing import Final, Literal

from graph_release.context import bound_release_or_none
from materialise.method_status import is_projection
from materialise.partial_period import reaches, start_of
from origin_catalogue import Catalogue, CatalogueNotBound, current_catalogue
from origin_catalogue.catalogue import Undocumented, partial_period_of

__all__ = [
    "HELD_KINDS",
    "HeldKind",
    "Judgement",
    "cut_catalogue",
    "held_back",
    "judge",
    "last_period_before",
    "last_year_before",
    "partial_mark",
    "partial_marks",
    "source_of",
]

_log = logging.getLogger(__name__)

type HeldKind = Literal["partial", "projected"]
"""A class of rows a read holds back unless its window's `include` names it."""

HELD_KINDS: Final[frozenset[HeldKind]] = frozenset({"partial", "projected"})


@dataclass(frozen=True, slots=True)
class Judgement:
    """What a row is, other than a complete actual one. `partial_period` is the source's mark
    on a `partial` judgement, `None` on the others."""

    status: Literal["partial", "projected", "unknown"]
    partial_period: str | None = None


_UNKNOWN: Final = Judgement(status="unknown")
_PROJECTED: Final = Judgement(status="projected")


def source_of(provenance: str, stat_var: str, *, component_of: Mapping[str, str]) -> str:
    """The source a row with `provenance` and `stat_var` cells is judged by: the provenance of
    the composite component serving `stat_var` (`component_of`, by bare StatVar dcid), else
    `provenance` itself."""
    return component_of.get(stat_var.removeprefix("dcid:"), provenance)


def judge(
    *, source: str | None, method: str | None, period: str, catalogue: Catalogue
) -> Judgement | None:
    """The judgement of a row of `source` under `method` at the `time_period` cell `period`,
    against `catalogue`'s partial marks (see the module docstring). `None` for a complete
    actual row."""
    if is_projection(source, method):
        return _PROJECTED
    if source is None:
        return _UNKNOWN
    try:
        mark = partial_mark(catalogue, source)
    except Undocumented:
        _warn_undocumented(source)
        return _UNKNOWN
    if mark is None:
        return None
    reached = reaches(period, mark)
    if reached is None:
        return _UNKNOWN
    return Judgement(status="partial", partial_period=mark) if reached else None


def partial_marks(catalogue: Catalogue, source: str) -> tuple[str, ...]:
    """The distinct partial marks, sorted, of the provenances whose rows carry `source`:
    `source`'s own (`partial_period_of`), or for a declared dataset's id its member
    provenances' in the bound release's code index (`CodeIndex.provenances_of`), a member the
    catalogue does not document adding none. Empty where none is marked. With no release
    bound, `source` reads as a plain table.

    Raises:
        Undocumented: `catalogue` documents none of those provenances."""
    release = bound_release_or_none()
    provenances = (source,) if release is None else release.views.code_index.provenances_of(source)
    marks: set[str] = set()
    documented = False
    for provenance in provenances:
        try:
            mark = partial_period_of(catalogue, provenance)
        except Undocumented:
            continue
        documented = True
        if mark is not None:
            marks.add(mark)
    if not documented:
        raise Undocumented(source)
    return tuple(sorted(marks))


def partial_mark(catalogue: Catalogue, source: str) -> str | None:
    """The partial mark `source`'s rows are judged by: the one of `partial_marks` that starts
    first (`partial_period.start_of`), since it reaches every period a later one reaches, so a
    declared dataset holds back each period any member marks. `None` where none is marked.

    Raises:
        Undocumented: as `partial_marks` does."""
    return min(partial_marks(catalogue, source), key=start_of, default=None)


@cache
def _warn_undocumented(source: str) -> None:
    """Logs, once per process, that rows of `source` are served unjudged: the catalogue
    documents no such source, so its partial period is unknown."""
    _log.warning(
        "held-back cut: the origin catalogue documents no source %s; its rows are served unjudged",
        source,
    )


def held_back(judgement: Judgement | None, *, include: frozenset[str]) -> bool:
    """Whether a read whose window opts in to `include` holds back a row judged `judgement`:
    a partial or projected row whose kind `include` does not name."""
    return (
        judgement is not None and judgement.status in HELD_KINDS and judgement.status not in include
    )


def last_year_before(judgement: Judgement) -> int | None:
    """The newest calendar year before the period a `partial` judgement's mark starts in, the
    newest year whose rows the mark does not reach. `None` for any other judgement, and for a
    mark the frame grammar cannot read."""
    if judgement.partial_period is None:
        return None
    try:
        return int(start_of(judgement.partial_period)[:4]) - 1
    except ValueError:
        return None


def last_period_before(judgement: Judgement, *, period: str) -> str | None:
    """The newest period at the grain of the `time_period` cell `period` (a year, `2025-09`
    or `2025-09-30`) that a `partial` judgement's mark does not reach: the period before the
    one holding the mark's start. A `2025-06` mark leaves a year read through `2024` and a
    month read through `2025-05`. `None` for any other judgement, and for a mark or a grain
    the frame grammar or this rule cannot read."""
    if judgement.partial_period is None:
        return None
    try:
        start = date.fromisoformat(_padded(start_of(judgement.partial_period)))
    except ValueError:
        return None
    match len(period):
        case 4:
            return str(start.year - 1)
        case 7:
            return (
                f"{start.year - 1}-12"
                if start.month == 1
                else f"{start.year}-{start.month - 1:02d}"
            )
        case 10:
            return (start - timedelta(days=1)).isoformat()
        case _:
            return None


def _padded(start: str) -> str:
    """`start`, a period's first day at its own grain, as a full date."""
    return {4: f"{start}-01-01", 7: f"{start}-01"}.get(len(start), start)


def cut_catalogue() -> Catalogue:
    """The origin catalogue bound to the read, which the partial cut judges rows against.

    Raises:
        CatalogueNotBound: no catalogue is bound, so the read cannot tell which periods its
            sources hold only part of and refuses.
    """
    try:
        return current_catalogue()
    except CatalogueNotBound as error:
        raise CatalogueNotBound(
            "a read that does not opt in to partial periods judges its rows against the "
            "origin catalogue, and none is bound to this context"
        ) from error
