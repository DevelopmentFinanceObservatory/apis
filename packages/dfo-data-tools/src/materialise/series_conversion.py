"""Per-series conversion records a mint stores with its version.

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


def pack_series_conversions(
    conversions: Mapping[SeriesKey, SeriesConversion],
) -> dict[str, object]:
    """`conversions` as one JSON object that names each distinct StatVar, role tuple and
    conversion record once. `stat_vars`, `roles` and `records` list them in order of first
    appearance over the series sorted by key, a record as a JSON object of its fields. `series`
    holds one `[stat_var, roles, record]` triple of list positions per series, sorted by key.

    A mint's series mostly share one record, so the object grows by a few bytes a series. The
    per-series entry list that earlier versions stored repeated every field.
    `decode_series_conversion_entries` reads both forms."""
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


def _record_entry(conversion: SeriesConversion) -> dict[str, object]:
    return {
        "recipe_id": conversion.recipe_id,
        "table_provenance_id": conversion.table_provenance_id,
        "exchange_rate_table_version": conversion.exchange_rate_table_version,
        "deflator_table_version": conversion.deflator_table_version,
        "fallback_count": conversion.fallback_count,
        "fallback_examples": [list(pair) for pair in conversion.fallback_examples],
        "projected_count": conversion.projected_count,
        "projected_examples": list(conversion.projected_examples),
    }


def _pairs(value: object) -> tuple[tuple[str, str], ...]:
    return tuple((str(first), str(second)) for first, second in cast(list[list[str]], value))


def _record(entry: Mapping[str, object]) -> SeriesConversion:
    deflator_version = entry["deflator_table_version"]
    return SeriesConversion(
        recipe_id=str(entry["recipe_id"]),
        table_provenance_id=str(entry["table_provenance_id"]),
        exchange_rate_table_version=str(entry["exchange_rate_table_version"]),
        deflator_table_version=None if deflator_version is None else str(deflator_version),
        fallback_count=cast(int, entry["fallback_count"]),
        fallback_examples=_pairs(entry["fallback_examples"]),
        projected_count=cast(int, entry["projected_count"]),
        projected_examples=tuple(str(e) for e in cast(list[str], entry["projected_examples"])),
    )


def decode_series_conversion_entries(
    entries: list[Mapping[str, object]] | dict[str, object],
) -> dict[SeriesKey, SeriesConversion]:
    """The inverse of `pack_series_conversions` for its object, and the records of a list of
    per-series entries, the form earlier versions stored. Each entry holds a record's fields
    with the series' `stat_var` and `roles`."""
    if isinstance(entries, list):
        return {
            (str(entry["stat_var"]), _pairs(entry["roles"])): _record(entry) for entry in entries
        }
    stat_vars = [str(stat_var) for stat_var in cast(list[str], entries["stat_vars"])]
    roles = [_pairs(pairs) for pairs in cast(list[object], entries["roles"])]
    records = [_record(entry) for entry in cast(list[Mapping[str, object]], entries["records"])]
    return {
        (stat_vars[stat_var], roles[role]): records[record]
        for stat_var, role, record in cast(list[list[int]], entries["series"])
    }
