"""Group streaming limits, latest-window filtering, and aggregate row ceilings."""

from __future__ import annotations

import threading

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, materialise_group, materialise_spec
from materialise.errors import OverCeilingError
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _BOUND_FACETS,
    _CSV_HEADER,
    _ENTITY,
    _OTHER_ENTITY,
    _SHARED_FACET,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from .conftest import FakeGraphReader


def _coverage_for(entries: list[tuple[str, str, str]]) -> CoverageIndex:
    """Build coverage for each StatVar/facet/entity triple with matching
    PROV1/METH1/UNIT1 metadata.
    """
    bare_stat_vars = {bare for bare, _, _ in entries}
    role_names = {bare: RoleNames(donor="donor", recipient="recipient") for bare in bare_stat_vars}
    facet_ids = {facet_id for _, facet_id, _ in entries}
    facet_metadata = {
        facet_id: FacetMetadata(provenance_id="PROV1", measurement_method="METH1", unit="UNIT1")
        for facet_id in facet_ids
    }
    rows = [(bare, facet_id, "country/DONOR", entity, [2015]) for bare, facet_id, entity in entries]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


class _RoutedStreamingTransport(httpx.BaseTransport):
    """Route requests by `c[variableMeasured]` and yield response bytes in
    chunks. `requested` signals arrival before body production. Tests can
    gate one stream on another member's request.
    """

    def __init__(
        self,
        bodies_by_bare_stat_var: dict[str, bytes],
        *,
        chunk_size: int = 32,
        block_until_requested: dict[str, str] | None = None,
    ) -> None:
        self._bodies = bodies_by_bare_stat_var
        self._chunk_size = chunk_size
        self.requests: list[httpx.Request] = []
        self.chunks_yielded: dict[str, int] = dict.fromkeys(bodies_by_bare_stat_var, 0)
        self.requested: dict[str, threading.Event] = {
            stat_var: threading.Event() for stat_var in bodies_by_bare_stat_var
        }
        self._block_until_requested = block_until_requested or {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        stat_var = str(request.url.params.get("c[variableMeasured]"))
        self.requested[stat_var].set()
        return httpx.Response(
            200,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            content=self._iter_chunks(stat_var),
        )

    def _iter_chunks(self, stat_var: str):
        wait_for = self._block_until_requested.get(stat_var)
        if wait_for is not None and not self.requested[wait_for].wait(timeout=5.0):
            raise AssertionError(
                f"{wait_for!r} was never requested while streaming {stat_var!r}; "
                "the bounded pool should dispatch both members' fetches concurrently"
            )
        body = self._bodies[stat_var]
        for i in range(0, len(body), self._chunk_size):
            self.chunks_yielded[stat_var] += 1
            yield body[i : i + self._chunk_size]

    def total_chunks(self, stat_var: str) -> int:
        body = self._bodies[stat_var]
        return -(-len(body) // self._chunk_size) if body else 0


# ---------------------------------------------------------------------------
# Per-member streamed fetch
# ---------------------------------------------------------------------------


def test_group_member_oversized_stream_aborts_mid_stream_before_reading_every_chunk():
    # Every row in A's 80-read-block response survives, crossing its 5-row
    # ceiling before the stream ends. B uses a distinct entity and chunk. The
    # event gate makes B's request arrive before A yields bytes, proving the
    # abort leaves A's stream unread while B's concurrent fetch completes.
    from dc_sdmx.client import _STREAM_BATCH_BLOCK_SIZE

    ceiling = 5
    one_row = _csv_row(
        stat_var=_STAT_VAR_A,
        year="2000",
        value="1000",
        unit="UNIT1",
        method="METH1",
        provenance="PROV1",
        facet_id="facet1",
    )
    n_rows = -(-80 * _STREAM_BATCH_BLOCK_SIZE // len(one_row))
    body_a = (
        _CSV_HEADER
        + "".join(
            _csv_row(
                stat_var=_STAT_VAR_A,
                year=str(2000 + i),
                value="1000",
                unit="UNIT1",
                method="METH1",
                provenance="PROV1",
                facet_id="facet1",
            )
            for i in range(n_rows)
        )
    ).encode()
    body_b = (
        _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
    ).encode()
    transport = _RoutedStreamingTransport(
        {_BARE_A: body_a, _BARE_B: body_b},
        chunk_size=4096,
        block_until_requested={_BARE_A: _BARE_B},
    )
    sdmx = SdmxClient(base_url="https://datacommons.one.org/core/api", transport=transport)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _OTHER_ENTITY): [_SHARED_FACET],
        }
    )
    coverage = _coverage_for([(_BARE_A, "facet1", _ENTITY), (_BARE_B, "facet1", _OTHER_ENTITY)])
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS, entity=_OTHER_ENTITY),
    ]

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
            ceiling=ceiling,
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    # The refusal is member A's alone: same code, and `detail` names A's own
    # stat_var, regardless of how many of B's requests the pool issued in
    # the meantime.
    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.detail is not None
    assert "was not measured" in excinfo.value.detail
    assert _STAT_VAR_A in excinfo.value.detail
    assert _STAT_VAR_B not in excinfo.value.detail
    assert transport.chunks_yielded[_BARE_A] < transport.total_chunks(_BARE_A)
    # Coverage supplies candidates without graph probes. The event gate
    # starts B before A yields bytes. Shutdown waits for B's started fetch
    # after A aborts. With concurrency 1, only A's fetch is submitted.
    assert len(transport.requests) == 2


def test_group_latest_window_over_a_huge_raw_response_does_not_early_refuse_and_succeeds():
    # Each member returns 90 periods, but latest filtering keeps one row.
    # Both share a chunk, so 180 raw rows draw on the shared 20 x ceiling
    # budget. Ceiling 20 gives a 400-row budget and avoids that guard.
    ceiling = 20
    bodies = {
        bare: _CSV_HEADER
        + "".join(
            _csv_row(
                stat_var=stat_var,
                year=str(2000 + i),
                value="1000",
                unit="UNIT1",
                method="METH1",
                provenance="PROV1",
                facet_id="facet1",
            )
            for i in range(90)
        )
        for bare, stat_var in ((_BARE_A, _STAT_VAR_A), (_BARE_B, _STAT_VAR_B))
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    coverage = _coverage_for([(_BARE_A, "facet1", _ENTITY), (_BARE_B, "facet1", _ENTITY)])
    requests = [
        _request(
            stat_var=_STAT_VAR_A,
            facet_bindings=_BOUND_FACETS,
            window=RequestedWindow(kind="latest"),
        ),
        _request(
            stat_var=_STAT_VAR_B,
            facet_bindings=_BOUND_FACETS,
            window=RequestedWindow(kind="latest"),
        ),
    ]

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
        ceiling=ceiling,
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert result.coverage.rows == 2  # one latest-year row per member
    assert result.trace.applied_window["kind"] == "latest"


def test_group_shared_raw_budget_refuses_a_combined_total_neither_member_alone_would_trip():
    # Both latest-window members share one chunk at `member_concurrency=1`
    # (`ceil(2 / 1) == 2`). With ceiling 5, the shared budget is 100 rows.
    # Each 60-row response fits alone, but 120 combined rows exceed it.
    # The multi-member error names both StatVars.
    ceiling = 5
    bodies = {
        bare: _CSV_HEADER
        + "".join(
            _csv_row(
                stat_var=stat_var,
                year=str(2000 + i),
                value="1000",
                unit="UNIT1",
                method="METH1",
                provenance="PROV1",
                facet_id="facet1",
            )
            for i in range(60)
        )
        for bare, stat_var in ((_BARE_A, _STAT_VAR_A), (_BARE_B, _STAT_VAR_B))
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    coverage = _coverage_for([(_BARE_A, "facet1", _ENTITY), (_BARE_B, "facet1", _ENTITY)])
    requests = [
        _request(
            stat_var=_STAT_VAR_A,
            facet_bindings=_BOUND_FACETS,
            window=RequestedWindow(kind="latest"),
        ),
        _request(
            stat_var=_STAT_VAR_B,
            facet_bindings=_BOUND_FACETS,
            window=RequestedWindow(kind="latest"),
        ),
    ]

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
            ceiling=ceiling,
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=1,
            coverage=coverage,
        )

    assert excinfo.value.ceiling_kind == "rows"
    assert excinfo.value.cap == 20 * ceiling
    assert excinfo.value.count > 20 * ceiling
    # Both stat_vars are named. The raw guard trips on the shared stream
    # before either member's own rows are attributed to it alone.
    assert excinfo.value.detail is not None
    assert _STAT_VAR_A in excinfo.value.detail
    assert _STAT_VAR_B in excinfo.value.detail


def test_group_latest_window_spans_members_with_different_latest_years():
    # Each member keeps its own latest year, 2024 for A and 2022 for B.
    # Coverage must span 2022-2024 and count two years.
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2020",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2024",
            value="20",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2019",
            value="30",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2022",
            value="40",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    coverage = _coverage_for([(_BARE_A, "facet1", _ENTITY), (_BARE_B, "facet1", _ENTITY)])
    requests = [
        _request(
            stat_var=_STAT_VAR_A,
            facet_bindings=_BOUND_FACETS,
            window=RequestedWindow(kind="latest"),
        ),
        _request(
            stat_var=_STAT_VAR_B,
            facet_bindings=_BOUND_FACETS,
            window=RequestedWindow(kind="latest"),
        ),
    ]

    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
        ceiling=10,
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    # Each member contributes its own latest year: 2022 for B and 2024 for A.
    assert result.coverage.rows == 2
    years = next(d for d in result.coverage.dimensions if d["label"] == "years")
    assert years["count"] == 2
    assert result.trace.applied_window == {
        "kind": "latest",
        "start_year": 2022,
        "end_year": 2024,
        "origin": "query",
    }


# ---------------------------------------------------------------------------
# Aggregate ceiling
# ---------------------------------------------------------------------------


def test_aggregate_ceiling_trips_on_the_sum_not_per_member():
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    bodies = {
        _BARE_A: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        + _csv_row(
            stat_var=_STAT_VAR_A,
            year="2016",
            value="20",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
        _BARE_B: _CSV_HEADER
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2015",
            value="10",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        + _csv_row(
            stat_var=_STAT_VAR_B,
            year="2016",
            value="20",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        ),
    }
    sdmx, _requests = _sdmx_client_for(bodies)
    coverage = _coverage_for([(_BARE_A, "facet1", _ENTITY), (_BARE_B, "facet1", _ENTITY)])
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS),
    ]

    # Each member alone carries 2 rows, individually under a ceiling of 3.
    sdmx_solo, _r = _sdmx_client_for(bodies)
    graph_solo = FakeGraphReader(facets={(_BARE_A, _ENTITY): [_SHARED_FACET]})
    coverage_solo = _coverage_for([(_BARE_A, "facet1", _ENTITY)])
    solo_result = materialise_spec(
        requests[0],
        sdmx=sdmx_solo,
        graph=graph_solo,
        enrichment=EnrichmentLookup(graph_solo, type_kinds={}, parties={}),
        ceiling=3,
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_solo,
    )
    assert solo_result.coverage.rows == 2

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
            ceiling=3,
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    assert excinfo.value.detail is not None
    assert "4" in excinfo.value.detail
    assert "3" in excinfo.value.detail


# ---------------------------------------------------------------------------
# Mid-stream ceiling inside a genuinely shared chunk
# ---------------------------------------------------------------------------


class _SharedChunkStreamingTransport(httpx.BaseTransport):
    """Stream one combined body for a shared chunk in `chunk_size` pieces."""

    def __init__(self, body: bytes, *, chunk_size: int) -> None:
        self._body = body
        self._chunk_size = chunk_size
        self.requests: list[httpx.Request] = []
        self.chunks_yielded = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            content=self._iter_chunks(),
        )

    def _iter_chunks(self):
        for i in range(0, len(self._body), self._chunk_size):
            self.chunks_yielded += 1
            yield self._body[i : i + self._chunk_size]

    def total_chunks(self) -> int:
        return -(-len(self._body) // self._chunk_size) if self._body else 0


def test_shared_chunk_mid_stream_ceiling_names_a_member_of_the_chunk_not_a_specific_one():
    # Both members share a recipient and the all-window scope. With
    # `member_concurrency=1`, `ceil(2 / 1) == 2` keeps them in one chunk.
    # Every row survives. The first batch crosses ceiling 40 for either
    # member, so the named StatVar is nondeterministic. The batch has a few
    # hundred rows, below the 800-row shared raw budget. The body spans many
    # read blocks, leaving unread chunks when the per-member ceiling fires.
    from dc_sdmx.client import _STREAM_BATCH_BLOCK_SIZE

    ceiling = 40

    def _row(stat_var: str, year: int) -> str:
        return _csv_row(
            stat_var=stat_var,
            year=str(year),
            value="1000",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )

    one_pair = _row(_BARE_A, 2000) + _row(_BARE_B, 2000)
    n_pairs = -(-80 * _STREAM_BATCH_BLOCK_SIZE // len(one_pair))
    body = (
        _CSV_HEADER
        + "".join(_row(_BARE_A, 2000 + i) + _row(_BARE_B, 2000 + i) for i in range(n_pairs))
    ).encode()

    transport = _SharedChunkStreamingTransport(body, chunk_size=4096)
    sdmx = SdmxClient(base_url="https://datacommons.one.org/core/api", transport=transport)
    graph = FakeGraphReader(
        facets={
            (_BARE_A, _ENTITY): [_SHARED_FACET],
            (_BARE_B, _ENTITY): [_SHARED_FACET],
        }
    )
    coverage = _coverage_for([(_BARE_A, "facet1", _ENTITY), (_BARE_B, "facet1", _ENTITY)])
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS),
        _request(stat_var=_STAT_VAR_B, facet_bindings=_BOUND_FACETS),
    ]

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
            ceiling=ceiling,
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=1,
            coverage=coverage,
        )

    assert excinfo.value.ceiling_kind == "rows"
    # `cap == ceiling` (not `20 * ceiling`) is what distinguishes this from
    # the shared raw-row budget tripping instead. Both guards report
    # `ceiling_kind == "rows"`.
    assert excinfo.value.cap == ceiling
    assert excinfo.value.detail is not None
    assert (
        "fetched together with 1 other member(s) sharing this member's own "
        "entity scope, in one combined SDMX request"
    ) in excinfo.value.detail
    named_a = _STAT_VAR_A in excinfo.value.detail
    named_b = _STAT_VAR_B in excinfo.value.detail
    assert named_a != named_b  # Exactly one chunk member is named.
    assert transport.chunks_yielded < transport.total_chunks()
