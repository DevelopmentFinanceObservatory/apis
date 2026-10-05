"""Per-series conversion records a mint stores in its version body.

A converted mint converts a member in one call that keeps each series apart
(`generation._convert_by_series`), giving each series the record a preview read with
`series_roles` gives it, so a series' record counts that series' rows alone. A
series is a bare StatVar dcid and its role tuple: `(role property, entity dcid)` pairs sorted
by role property. `SeriesConversion` keeps the fields a reader of one series reports, which is
`ConversionRecord` without the member-level `native_currency`, `native_unit_ref`,
`table_refresh_date`, and `omitted_entities`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from materialise.convert import ConversionRecord

type RoleTuple = tuple[tuple[str, str], ...]
"""A series' `(role property, entity dcid)` pairs, sorted by role property."""

type SeriesKey = tuple[str, RoleTuple]
"""A series' bare StatVar dcid and its `RoleTuple`."""


@dataclass(frozen=True)
class SeriesConversion:
    """The recipe, tables, and fallback and projection markers behind one converted series.
    Counts are rows of that series; examples are sorted and capped as in `ConversionRecord`."""

    recipe_id: str
    table_provenance_id: str
    exchange_rate_table_version: str
    deflator_table_version: str | None
    fallback_count: int
    fallback_examples: tuple[tuple[str, str], ...]
    projected_count: int
    projected_examples: tuple[str, ...]

    @classmethod
    def from_record(cls, record: ConversionRecord) -> SeriesConversion:
        """`record` narrowed to the fields one series reports."""
        return cls(
            recipe_id=record.recipe_id,
            table_provenance_id=record.table_provenance_id,
            exchange_rate_table_version=record.exchange_rate_table_version,
            deflator_table_version=record.deflator_table_version,
            fallback_count=record.fallback_count,
            fallback_examples=record.fallback_examples,
            projected_count=record.projected_count,
            projected_examples=record.projected_examples,
        )


def series_key(stat_var: str, roles: Sequence[str], dcids: Sequence[str]) -> SeriesKey:
    """The key of the series binding `dcids` on `roles` of the bare StatVar `stat_var`."""
    return stat_var, tuple(sorted(zip(roles, dcids, strict=True)))


def encode_series_conversions(
    conversions: Mapping[SeriesKey, SeriesConversion],
) -> list[dict[str, object]]:
    """JSON entries for `conversions`, one per series, sorted by key. Role tuples and fallback
    pairs are JSON lists."""
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


def _pairs(value: object) -> tuple[tuple[str, str], ...]:
    return tuple((str(first), str(second)) for first, second in cast(list[list[str]], value))


def decode_series_conversion_entries(
    entries: Sequence[Mapping[str, object]],
) -> dict[SeriesKey, SeriesConversion]:
    """`encode_series_conversions`' inverse."""
    decoded: dict[SeriesKey, SeriesConversion] = {}
    for entry in entries:
        deflator_version = entry["deflator_table_version"]
        decoded[(str(entry["stat_var"]), _pairs(entry["roles"]))] = SeriesConversion(
            recipe_id=str(entry["recipe_id"]),
            table_provenance_id=str(entry["table_provenance_id"]),
            exchange_rate_table_version=str(entry["exchange_rate_table_version"]),
            deflator_table_version=None if deflator_version is None else str(deflator_version),
            fallback_count=cast(int, entry["fallback_count"]),
            fallback_examples=_pairs(entry["fallback_examples"]),
            projected_count=cast(int, entry["projected_count"]),
            projected_examples=tuple(str(e) for e in cast(list[str], entry["projected_examples"])),
        )
    return decoded
