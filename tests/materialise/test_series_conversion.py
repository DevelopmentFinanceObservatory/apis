"""Reads per-series conversion records in both stored forms: the per-series entry list this
image writes, and the packed object that names each distinct record once."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from materialise.series_conversion import (
    RoleTuple,
    SeriesConversion,
    SeriesKey,
    decode_series_conversion_entries,
    encode_series_conversions,
    series_key,
)

_SHARED = SeriesConversion(
    recipe_id="ConversionRecipeDAC",
    table_provenance_id="dc/base/IMF",
    exchange_rate_table_version="sha256:rates",
    deflator_table_version="sha256:deflators",
    fallback_count=0,
    fallback_examples=(),
    projected_count=0,
    projected_examples=(),
)
_FALLBACK = SeriesConversion(
    recipe_id="ConversionRecipeDAC",
    table_provenance_id="dc/base/IMF",
    exchange_rate_table_version="sha256:rates",
    deflator_table_version=None,
    fallback_count=2,
    fallback_examples=(("country/KEN", "2015"), ("country/KEN", "2016")),
    projected_count=1,
    projected_examples=("2024",),
)

# 40 StatVars x 2 providers x 25 recipients, all on one record but one series.
_CONVERSIONS = {
    series_key(
        f"Amount_{stat_var}",
        ("financingProvider", "financingRecipient"),
        (f"country/P{provider}", f"country/R{recipient:02d}"),
    ): _SHARED
    for stat_var in range(40)
    for provider in range(2)
    for recipient in range(25)
}
_CONVERSIONS[series_key("Amount_0", ("financingRecipient",), ("country/KEN",))] = _FALLBACK


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


def _packed(conversions: Mapping[SeriesKey, SeriesConversion]) -> dict[str, object]:
    """`conversions` as the packed object: each distinct StatVar, role tuple and record once,
    in order of first appearance over the series sorted by key, and one `[stat_var, roles,
    record]` triple of list positions per series, sorted by key."""
    stat_vars: dict[str, int] = {}
    roles: dict[RoleTuple, int] = {}
    records: dict[SeriesConversion, int] = {}
    series = [
        [
            stat_vars.setdefault(stat_var, len(stat_vars)),
            roles.setdefault(pairs, len(roles)),
            records.setdefault(record, len(records)),
        ]
        for (stat_var, pairs), record in sorted(conversions.items())
    ]
    return {
        "stat_vars": list(stat_vars),
        "roles": [[list(pair) for pair in pairs] for pairs in roles],
        "records": [_record_entry(record) for record in records],
        "series": series,
    }


def _stored(value: object) -> list[Mapping[str, object]] | dict[str, object]:
    """`value` as a jsonb column returns it."""
    return json.loads(json.dumps(value))


def test_packed_records_read_back_as_every_series_record():
    packed = _packed(_CONVERSIONS)

    assert decode_series_conversion_entries(_stored(packed)) == _CONVERSIONS
    assert len(cast(list, packed["records"])) == 2


def test_the_entry_list_this_image_writes_reads_back():
    entries = encode_series_conversions(_CONVERSIONS)

    assert decode_series_conversion_entries(_stored(entries)) == _CONVERSIONS
