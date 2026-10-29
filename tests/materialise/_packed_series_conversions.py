"""The packed `series_conversions` object, which `decode_series_conversion_entries` reads beside
the per-series entry list `encode_series_conversions` writes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from materialise.series_conversion import (
    RoleTuple,
    SeriesConversion,
    SeriesKey,
    _record_entry,
)


def pack_series_conversions(
    conversions: Mapping[SeriesKey, SeriesConversion],
) -> dict[str, object]:
    """`conversions` as the packed object `decode_series_conversion_entries` reads: `stat_vars`,
    `roles` and `records` list each distinct StatVar, role tuple and record in order of first
    appearance over the series sorted by key, and `series` holds one `[stat_var, roles, record]`
    triple of list positions per series, sorted by key."""
    positions: dict[str, dict[object, int]] = {"stat_vars": {}, "roles": {}, "records": {}}

    def position(kind: str, value: object) -> int:
        return positions[kind].setdefault(value, len(positions[kind]))

    series = [
        [position("stat_vars", stat_var), position("roles", roles), position("records", record)]
        for (stat_var, roles), record in sorted(conversions.items())
    ]
    return {
        "stat_vars": list(positions["stat_vars"]),
        "roles": [[list(pair) for pair in cast(RoleTuple, roles)] for roles in positions["roles"]],
        "records": [
            _record_entry(cast(SeriesConversion, record)) for record in positions["records"]
        ],
        "series": series,
    }
