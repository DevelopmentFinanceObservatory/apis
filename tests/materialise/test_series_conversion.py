"""Stores per-series conversion records once per distinct record and reads both stored forms."""

from __future__ import annotations

import json

from materialise.series_conversion import (
    SeriesConversion,
    decode_series_conversion_entries,
    pack_series_conversions,
    series_key,
)

from ._legacy_series_conversions import legacy_series_conversion_entries

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


def _stored(value: object) -> object:
    """`value` as a jsonb column returns it."""
    return json.loads(json.dumps(value))


def test_packed_records_read_back_as_every_series_record():
    packed = pack_series_conversions(_CONVERSIONS)

    assert decode_series_conversion_entries(_stored(packed)) == _CONVERSIONS
    assert len(packed["records"]) == 2
    assert len(packed["stat_vars"]) == 40
    assert len(packed["roles"]) == 2 * 25 + 1


def test_a_version_stored_before_packing_reads_its_entry_list():
    entries = legacy_series_conversion_entries(_CONVERSIONS)

    assert decode_series_conversion_entries(_stored(entries)) == _CONVERSIONS


def test_packing_stores_a_few_bytes_a_series():
    """The 2,001 series take about 15 bytes each packed, against about 375 as entries."""
    packed = len(json.dumps(pack_series_conversions(_CONVERSIONS)))
    entries = len(json.dumps(legacy_series_conversion_entries(_CONVERSIONS)))

    assert packed * 10 < entries
