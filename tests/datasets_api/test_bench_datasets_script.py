"""`scripts/bench_datasets.py` runs every request kind against an in-process surface and reads
each response's counts from its `Server-Timing` header."""

from __future__ import annotations

import contextvars
import random
from typing import cast

import httpx

from scripts.bench_datasets import Shape, run_bench, window_variants

from ._frame_doubles import _body
from .test_store_budget import surface  # noqa: F401 -- the shared fixture

_SERVE_COUNTS = {
    "sdmx_requests",
    "graph_requests",
    "db_transactions",
    "sign_blob_calls",
    "permit_wait_ms",
}


class _ContextClient:
    """`client` with each request run in a copy of the context it was built in, as a request
    from that thread runs, so a burst the bench sends from its pool reads the graph release the
    test binds."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._context = contextvars.copy_context()
        self.base_url = client.base_url

    def get(self, *args, **kwargs) -> httpx.Response:
        return self._context.copy().run(self._client.get, *args, **kwargs)

    def post(self, *args, **kwargs) -> httpx.Response:
        return self._context.copy().run(self._client.post, *args, **kwargs)


def test_the_bench_reports_every_request_kind_with_its_store_counts(surface):  # noqa: F811
    minted = surface.client.post("/api/datasets", json=_body(format="csv", bundle=True))
    assert minted.status_code == 200, minted.text
    definition = minted.json()["definition"]
    shape = Shape(
        name="fixture",
        description="one pinned StatVar",
        definition=definition,
        variants=window_variants(definition, rng=random.Random(3)),
    )

    report = run_bench(
        cast(httpx.Client, _ContextClient(surface.client)), [shape], samples=2, seed=3
    )

    requests = report["shapes"]["fixture"]["requests"]
    assert set(requests) == {
        "cold_mint",
        "warm_mint",
        "cold_get",
        "warm_get",
        "snapshot",
        "concurrent_warm_get",
        "concurrent_cold_mint_3",
        "concurrent_cold_mint_6",
    }
    for kind, summary in requests.items():
        # A burst past the read or work executor's size refuses its excess with a retryable 503.
        allowed = {200, 503} if kind.startswith("concurrent") else {200}
        assert 200 in summary["statuses"], (kind, summary["samples"])
        assert set(summary["statuses"]) <= allowed, (kind, summary["samples"])
        assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"] > 0
    for kind in ("cold_mint", "warm_mint", "cold_get", "warm_get", "concurrent_cold_mint_6"):
        assert set(requests[kind]["counts"]) == _SERVE_COUNTS, kind
    assert requests["cold_mint"]["served_from_head"] == 0
    assert requests["cold_get"]["served_from_head"] == 0
    assert requests["warm_get"]["served_from_head"] == 2
    assert requests["warm_get"]["counts"]["db_transactions"]["max"] > 0
    assert report["skipped"]
