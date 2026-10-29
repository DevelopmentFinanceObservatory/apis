"""The per-series entry list versions stored in `series_conversions` before records were packed
(`materialise.series_conversion.pack_series_conversions`)."""

from __future__ import annotations

from collections.abc import Mapping

from materialise.series_conversion import SeriesConversion, SeriesKey


def legacy_series_conversion_entries(
    conversions: Mapping[SeriesKey, SeriesConversion],
) -> list[dict[str, object]]:
    """One JSON entry per series, sorted by key, every record field repeated."""
    return [
        {
            "stat_var": stat_var,
            "roles": [list(pair) for pair in roles],
            "recipe_id": conversion.recipe_id,
            "table_provenance_id": conversion.table_provenance_id,
            "exchange_rate_table_version": conversion.exchange_rate_table_version,
            "deflator_table_version": conversion.deflator_table_version,
            "fallback_count": conversion.fallback_count,
            "fallback_examples": [list(pair) for pair in conversion.fallback_examples],
            "projected_count": conversion.projected_count,
            "projected_examples": list(conversion.projected_examples),
        }
        for (stat_var, roles), conversion in sorted(conversions.items())
    ]
