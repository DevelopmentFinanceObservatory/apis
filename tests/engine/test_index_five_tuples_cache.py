"""The five-tuple cache drops entries on index collection, preventing stale `id(index)` reuse."""

from __future__ import annotations

import gc

from code_index import CodeIndex, IndexStamp, StatVarRow
from search.engine.retrieval.bound_values import _index_five_tuples, _index_five_tuples_cache

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test-hash",
)


def _row(dcid: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties={},
    )


def _index(*dcids: str) -> CodeIndex:
    return CodeIndex(stamp=_STAMP, declarations=(), codes=(), statvars=[_row(d) for d in dcids])


def test_collected_index_leaves_no_cache_entry_behind() -> None:
    """After collection, the index's cache key is absent and safe for address reuse."""
    index = _index("sv_a")
    _index_five_tuples(index)
    key = id(index)
    assert key in _index_five_tuples_cache

    del index
    gc.collect()

    assert key not in _index_five_tuples_cache
