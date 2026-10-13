"""The series a frame's estimates continue, and the one rule that groups a continuing row
with the series it continues.

A dataset version records its continuation in its trace: `trace.continuations` names the
estimated method continuing the elected series (`facets.FacetSelection.continued_by`), and
`trace.facets` names the selections that series was held under. `continued_series_of_trace`
reads both as `ContinuedSeries`. A computed version stores the entries of the versions it was
computed from (`ContinuedSeries.as_entry`, `datasets_api.lineage_continuations`), which
`continued_series_of_entries` reads back.

`continuation_leads` is the grouping every series counter applies: Presentation's
`assembler.frame_view.series_keys` and the statistics fold
(`statistics.SeriesStatisticsSink`), so a card and a description count the same series.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

__all__ = [
    "ContinuedSeries",
    "continuation_leads",
    "continued_series_of_entries",
    "continued_series_of_trace",
]

_METHOD_AXIS = "measurementMethod"
_PROVENANCE_AXIS = "provenance"


@dataclass(frozen=True)
class ContinuedSeries:
    """Rows under `method` continue the series held under the measurement method `continues`
    (`None` for a null method) from the source `provenance`. `provenance` is `None` when the
    root version selected no single source, and then scopes nothing."""

    method: str
    continues: str | None
    provenance: str | None

    def as_entry(self) -> dict[str, object]:
        """The stored form a computed version's `continuations` holds."""
        return {
            "axis": _METHOD_AXIS,
            "method": self.method,
            "continues": self.continues,
            "provenance": self.provenance,
        }

    def admits(self, *, method: object, provenance: object) -> bool:
        """Whether a row under `method` from `provenance` is one of the continuing rows."""
        return method == self.method and self.provenance in (None, provenance)


def continued_series_of_trace(trace: Mapping[str, object]) -> tuple[ContinuedSeries, ...]:
    """The continuations a dataset version's trace records, each scoped to the
    `measurementMethod` and `provenance` selections its `facets` name. An entry naming no
    method string is skipped."""
    selected: dict[object, object] = {}
    for facet in _mappings(trace.get("facets")):
        selected.setdefault(facet.get("axis"), facet.get("ref"))
    continues, provenance = selected.get(_METHOD_AXIS), selected.get(_PROVENANCE_AXIS)
    return tuple(
        dict.fromkeys(
            ContinuedSeries(
                method=method,
                continues=continues if isinstance(continues, str) else None,
                provenance=provenance if isinstance(provenance, str) else None,
            )
            for entry in _mappings(trace.get("continuations"))
            if isinstance(method := entry.get("method"), str) and method
        )
    )


def continued_series_of_entries(entries: object) -> tuple[ContinuedSeries, ...]:
    """The `ContinuedSeries` a computed version's stored `continuations` hold. An entry not in
    the `as_entry` shape is skipped, so no stored value fails a read."""
    return tuple(
        dict.fromkeys(
            ContinuedSeries(method=method, continues=continues, provenance=provenance)
            for entry in _mappings(entries)
            if isinstance(method := entry.get("method"), str)
            and method
            and isinstance(continues := entry.get("continues"), str | None)
            and isinstance(provenance := entry.get("provenance"), str | None)
        )
    )


def continuation_leads(
    continued: Sequence[ContinuedSeries],
    *,
    identities: Sequence[Hashable],
    methods: Sequence[object],
    provenances: Sequence[object] | None,
) -> list[int | None]:
    """For each row, the index of the row whose series it continues, or `None` when it
    continues none.

    `identities` holds each row's series identity without its method. A row that one of
    `continued` admits (`ContinuedSeries.admits`) continues the first row of its identity that
    no entry admits and that is held under that entry's `continues` method; with no such row
    it is a series of its own. `provenances` is `None` for rows that carry no source, which
    then scope nothing."""
    admitting = [
        [
            series
            for series in continued
            if series.admits(
                method=method,
                provenance=series.provenance if provenances is None else provenances[position],
            )
        ]
        for position, method in enumerate(methods)
    ]
    first: dict[tuple[Hashable, object], int] = {}
    for position, (identity, method) in enumerate(zip(identities, methods, strict=True)):
        if not admitting[position]:
            first.setdefault((identity, method), position)
    return [
        next(
            (
                lead
                for series in admitting[position]
                if (lead := first.get((identity, series.continues))) is not None
            ),
            None,
        )
        for position, identity in enumerate(identities)
    ]


def _mappings(value: object) -> Iterable[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return (cast("Mapping[str, object]", item) for item in value if isinstance(item, Mapping))
