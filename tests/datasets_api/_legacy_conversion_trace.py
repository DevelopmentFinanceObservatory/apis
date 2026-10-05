"""A stored `trace.conversion` in the single-object shape and the list it migrates to.

Migrations `0017_trace_conversion_entries.sql` and
`0020_trace_conversion_entries_after_deploy.sql` in `src/dataset_store/migrations/` rewrite
`LEGACY_CONVERSION` into `MIGRATED_CONVERSION`. The live migration test in
`tests/dataset_store/test_live.py` asserts that mapping against Postgres, and the
definition-door test serves `MIGRATED_CONVERSION` from a stored row.

The `native_facets` keys sort differently in byte order (`ODA_Total` first) and
under a linguistic collation such as `en_US.utf8` (`oda_Grants` first).
"""

from __future__ import annotations

_RATE_TABLE_VERSION = "ab" * 32
_DEFLATOR_TABLE_VERSION = "cd" * 32
_FALLBACK = {"count": 2, "examples": [["KEN", "2020"]], "truncated": False}
_PROJECTED = {"count": 1, "examples": ["2024"], "truncated": False}

LEGACY_CONVERSION: dict[str, object] = {
    "recipe": "ConversionRecipeDAC",
    "rate_table_version": _RATE_TABLE_VERSION,
    "native_facets": {"oda_Grants": "CurrencyUSD", "ODA_Total": "CurrencyEUR"},
    "fallback": _FALLBACK,
    "projected": _PROJECTED,
}

MIGRATED_CONVERSION: list[dict[str, object]] = [
    {
        "recipe": "ConversionRecipeDAC",
        "rate_table_version": _RATE_TABLE_VERSION,
        "members": [
            {"stat_var": "ODA_Total", "native_unit": "CurrencyEUR"},
            {"stat_var": "oda_Grants", "native_unit": "CurrencyUSD"},
        ],
        "fallback": _FALLBACK,
        "projected": _PROJECTED,
    }
]

LEGACY_DEFLATED_CONVERSION: dict[str, object] = {
    **LEGACY_CONVERSION,
    "deflator_table_version": _DEFLATOR_TABLE_VERSION,
}

MIGRATED_DEFLATED_CONVERSION: list[dict[str, object]] = [
    {**MIGRATED_CONVERSION[0], "deflator_table_version": _DEFLATOR_TABLE_VERSION}
]
