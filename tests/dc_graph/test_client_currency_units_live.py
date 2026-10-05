"""Live smoke test for `GraphClient.currency_units`. Skipped unless `SEARCH_GRAPH_BASE` is set.
Set `SEARCH_GRAPH_TOKEN` when the deployment uses a WAF. Run with:
    uv run --env-file .env pytest -m live tests/dc_graph/test_client_currency_units_live.py
"""

from __future__ import annotations

import os

import pytest

from dc_graph import GraphClient

_GRAPH_BASE = os.environ.get("SEARCH_GRAPH_BASE")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _GRAPH_BASE, reason="set SEARCH_GRAPH_BASE to run the currency_units live test"
    ),
]


def test_currency_units_dev_returns_a_large_verified_roster():
    client = GraphClient(
        base_url=f"{_GRAPH_BASE}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN")
    )
    try:
        result = client.currency_units()
    finally:
        client.close()

    print(f"currency_units() returned {len(result)} currency units")
    assert len(result) >= 270
    assert result.get("CurrencyUSD") == "USD"
    assert all(len(code) == 3 and code.isalpha() and code.isupper() for code in result.values()), (
        "every ISO code must be three upper-case letters"
    )
