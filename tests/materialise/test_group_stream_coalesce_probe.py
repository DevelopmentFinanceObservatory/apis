"""Streaming group output matches a single-chunk run under out-of-order fetch completion.

Four members form three chunks across two entity scopes. An event makes the `Stat_B` chunk
finish before the earlier `Stat_A`/`Stat_C` chunk; `Stat_C` must wait for `Stat_B` in dcid order.
The caller supplies members in `C, A, D, B` order to check dispatch order is internal.
"""

from __future__ import annotations

import threading

import httpx

import materialise.generation as generation_module
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, materialise_group
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._group_support import (
    _BOUND_FACETS,
    _CSV_HEADER,
    _ENTITY,
    _OTHER_ENTITY,
    _ROLE,
    _SHARED_FACET,
    _request,
)
from .conftest import FakeGraphReader

# B is the only member bound to `_OTHER_ENTITY`, so it forms its own chunk.
_MEMBERS = (
    ("Stat_A", _ENTITY, "100"),
    ("Stat_B", _OTHER_ENTITY, "200"),
    ("Stat_C", _ENTITY, "300"),
    ("Stat_D", _ENTITY, "400"),
)


def _row(*, bare_stat_var: str, entity: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare_stat_var},{entity},UNIT1,METH1,"
        f"NotApplicable,PROV1,2015,{value},,facet1\r\n"
    )


def _bodies() -> dict[str, str]:
    return {
        bare: _row(bare_stat_var=bare, entity=entity, value=value)
        for bare, entity, value in _MEMBERS
    }


def _coverage() -> CoverageIndex:
    role_names = {bare: RoleNames(donor="donor", recipient=_ROLE) for bare, _, _ in _MEMBERS}
    facet_metadata = {
        "facet1": FacetMetadata(provenance_id="PROV1", measurement_method="METH1", unit="UNIT1")
    }
    return build_coverage_index(
        [(bare, "facet1", "country/DONOR", entity, [2015]) for bare, entity, _ in _MEMBERS],
        role_names=role_names,
        facet_metadata=facet_metadata,
    )


def _graph() -> FakeGraphReader:
    return FakeGraphReader(facets={(bare, entity): [_SHARED_FACET] for bare, entity, _ in _MEMBERS})


def _requests_in_shuffled_order() -> list:
    by_bare = {
        bare: _request(stat_var=f"dcid:{bare}", facet_bindings=_BOUND_FACETS, entity=entity)
        for bare, entity, _ in _MEMBERS
    }
    return [by_bare["Stat_C"], by_bare["Stat_A"], by_bare["Stat_D"], by_bare["Stat_B"]]


def _sdmx_client_plain() -> SdmxClient:
    """Builds a client whose requests return without a barrier."""
    bodies = _bodies()

    def handler(request: httpx.Request) -> httpx.Response:
        stat_vars = str(request.url.params.get("c[variableMeasured]")).split(",")
        rows = "".join(bodies[sv] for sv in stat_vars)
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )


def _sdmx_client_with_inversion() -> tuple[SdmxClient, threading.Event]:
    """Makes the `Stat_B` response finish before the earlier `Stat_A`/`Stat_C` chunk."""
    bodies = _bodies()
    b_answered = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        stat_vars = str(request.url.params.get("c[variableMeasured]")).split(",")
        if "Stat_A" in stat_vars:
            assert b_answered.wait(timeout=5.0), (
                "Stat_B's own fetch never answered ahead of Stat_A/Stat_C's"
            )
        rows = "".join(bodies[sv] for sv in stat_vars)
        response = httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
        if stat_vars == ["Stat_B"]:
            b_answered.set()
        return response

    client = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    return client, b_answered


def _run(
    *, sdmx: SdmxClient, member_concurrency: int, target_rows: int | None, monkeypatch
) -> tuple[str, tuple]:
    if target_rows is not None:
        monkeypatch.setattr(generation_module, "_FRAME_CHUNK_TARGET_ROWS", target_rows)
    result = materialise_group(
        _requests_in_shuffled_order(),
        sdmx=sdmx,
        graph=_graph(),
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=member_concurrency,
        coverage=_coverage(),
    )
    chunks = list(result.frame_chunks())
    rows = tuple(row for chunk in chunks for row in chunk.rows)
    return result.content_version, rows


def test_out_of_order_completion_with_small_chunks_matches_the_whole_frame_case(monkeypatch):
    whole_version, whole_rows = _run(
        sdmx=_sdmx_client_plain(), member_concurrency=1, target_rows=None, monkeypatch=monkeypatch
    )

    streamed_sdmx, b_answered = _sdmx_client_with_inversion()
    streamed_version, streamed_rows = _run(
        sdmx=streamed_sdmx, member_concurrency=2, target_rows=1, monkeypatch=monkeypatch
    )

    assert b_answered.is_set(), "the barrier was never reached; the probe exercised no inversion"
    assert streamed_version == whole_version
    assert streamed_rows == whole_rows
    assert len(whole_rows) == len(_MEMBERS)
