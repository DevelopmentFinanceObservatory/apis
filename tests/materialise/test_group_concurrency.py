"""Group fetch concurrency, error ordering, and serial dispatch at concurrency 1."""

from __future__ import annotations

import threading
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from concurrent.futures import Future
from concurrent.futures import wait as futures_wait

import httpx
import pytest

from dc_sdmx import SdmxClient, SdmxResponseError
from materialise import EnrichmentLookup, generation, materialise_group
from materialise.errors import OverCeilingError
from materialise.generation import _DISPATCH_LOOK_AHEAD_POOLS, _GroupRowStreamer
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _BARE_C,
    _BOUND_FACETS,
    _CSV_HEADER,
    _ENTITY,
    _OTHER_ENTITY,
    _ROLE,
    _SHARED_FACET,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _csv_row,
    _request,
)
from .conftest import FakeGraphReader


def _coverage_for(entries: list[tuple[str, str, str]]) -> CoverageIndex:
    """Build coverage for each StatVar/facet/entity triple with bound facet metadata."""
    bare_stat_vars = {bare for bare, _, _ in entries}
    role_names = {bare: RoleNames(donor="donor", recipient="recipient") for bare in bare_stat_vars}
    facet_ids = {facet_id for _, facet_id, _ in entries}
    facet_metadata = {
        facet_id: FacetMetadata(provenance_id="PROV1", measurement_method="METH1", unit="UNIT1")
        for facet_id in facet_ids
    }
    rows = [(bare, facet_id, "country/DONOR", entity, [2015]) for bare, facet_id, entity in entries]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _is_data_fetch(request: httpx.Request) -> bool:
    """Data fetches carry `c[facetId]`. Role-bearing probes use coverage."""
    return "c[facetId]" in request.url.params


# ---------------------------------------------------------------------------
# Barrier proof: members fetch concurrently
# ---------------------------------------------------------------------------


def test_member_concurrency_dispatches_every_group_member_at_once_barrier_proof():
    """All three fetches must reach the barrier within 5 seconds.
    Serial dispatch raises `BrokenBarrierError`.
    """
    entities = ("country/MEMBER_A", "country/MEMBER_B", "country/MEMBER_C")
    members = (
        (_BARE_A, _STAT_VAR_A, entities[0]),
        (_BARE_B, _STAT_VAR_B, entities[1]),
        (_BARE_C, _STAT_VAR_C, entities[2]),
    )
    barrier = threading.Barrier(len(members))
    bodies = {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=stat_var,
            year="2015",
            value="100",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        for bare, stat_var, _entity in members
    }

    def handler(request: httpx.Request) -> httpx.Response:
        bare_stat_var = str(request.url.params.get("c[variableMeasured]"))
        if _is_data_fetch(request):
            barrier.wait(timeout=5.0)
        return httpx.Response(
            200,
            text=bodies[bare_stat_var],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    graph = FakeGraphReader(
        facets={(bare, entity): [_SHARED_FACET] for bare, _stat_var, entity in members}
    )
    coverage = _coverage_for([(bare, "facet1", entity) for bare, _stat_var, entity in members])

    result = materialise_group(
        [
            _request(stat_var=stat_var, facet_bindings=_BOUND_FACETS, entity=entity)
            for _bare, stat_var, entity in members
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=len(members),
        coverage=coverage,
    )

    assert result.trace.resolved_stat_vars == (
        {"ref": _STAT_VAR_A, "roles": [_ROLE], "observation_count": 1},
        {"ref": _STAT_VAR_B, "roles": [_ROLE], "observation_count": 1},
        {"ref": _STAT_VAR_C, "roles": [_ROLE], "observation_count": 1},
    )


# ---------------------------------------------------------------------------
# Dcid-order-vs-completion-order exception precedence
# ---------------------------------------------------------------------------


def test_group_dcid_order_first_member_error_wins_over_a_wall_clock_faster_sibling():
    """Both fetches raise `SdmxResponseError` because each response omits a
    declared role. An event makes B finish first. `_generate_group` reports
    A's error because it drains futures in dcid order. Both requests complete
    before the error surfaces.
    """
    b_answered = threading.Event()
    transport_requests: list[httpx.Request] = []
    bodies = {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=stat_var,
            year="2015",
            value="100",
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        for bare, stat_var in ((_BARE_A, _STAT_VAR_A), (_BARE_B, _STAT_VAR_B))
    }

    def handler(request: httpx.Request) -> httpx.Response:
        transport_requests.append(request)
        bare_stat_var = str(request.url.params.get("c[variableMeasured]"))
        if _is_data_fetch(request):
            if bare_stat_var == _BARE_B:
                pass  # answers immediately, below
            else:
                assert b_answered.wait(timeout=5.0), (
                    "member B's own fetch never answered ahead of member A's"
                )
        response = httpx.Response(
            200,
            text=bodies[bare_stat_var],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
        if _is_data_fetch(request) and bare_stat_var == _BARE_B:
            b_answered.set()
        return response

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    graph = FakeGraphReader(
        facets={(_BARE_A, _ENTITY): [_SHARED_FACET], (_BARE_B, _OTHER_ENTITY): [_SHARED_FACET]}
    )
    coverage = _coverage_for([(_BARE_A, "facet1", _ENTITY), (_BARE_B, "facet1", _OTHER_ENTITY)])
    declared_roles = (_ROLE, "a_role_the_response_never_reports")
    requests = [
        _request(stat_var=_STAT_VAR_A, facet_bindings=_BOUND_FACETS, declared_roles=declared_roles),
        _request(
            stat_var=_STAT_VAR_B,
            facet_bindings=_BOUND_FACETS,
            declared_roles=declared_roles,
            entity=_OTHER_ENTITY,
        ),
    ]

    with pytest.raises(SdmxResponseError) as excinfo:
        materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=2,
            coverage=coverage,
        )

    assert _BARE_A in str(excinfo.value)
    assert _BARE_B not in str(excinfo.value)
    fetch_requests_by_member = [
        str(r.url.params.get("c[variableMeasured]"))
        for r in transport_requests
        if _is_data_fetch(r)
    ]
    # Both members' own fetches were issued: B's despite losing the race to
    # surface its own error, A's despite blocking on B first. The pool
    # shutdown let both complete.
    assert sorted(fetch_requests_by_member) == [_BARE_A, _BARE_B]


# ---------------------------------------------------------------------------
# member_concurrency=1 reproduces the old serial loop exactly
# ---------------------------------------------------------------------------


def test_group_member_concurrency_one_reproduces_serial_behaviour_exactly():
    """With concurrency 1, member A's ceiling error surfaces before member
    B's separate chunk is submitted. Coverage supplies candidates, so only
    A's aborted fetch reaches the transport.
    """
    ceiling = 5
    n_rows = 20
    body_a = _CSV_HEADER + "".join(
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
    body_b = _CSV_HEADER + _csv_row(
        stat_var=_STAT_VAR_B,
        year="2015",
        value="10",
        unit="UNIT1",
        method="METH1",
        provenance="PROV1",
        facet_id="facet1",
    )
    bodies = {_BARE_A: body_a, _BARE_B: body_b}
    transport_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_requests.append(request)
        bare_stat_var = str(request.url.params.get("c[variableMeasured]"))
        return httpx.Response(
            200,
            text=bodies[bare_stat_var],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    graph = FakeGraphReader(
        facets={(_BARE_A, _ENTITY): [_SHARED_FACET], (_BARE_B, _OTHER_ENTITY): [_SHARED_FACET]}
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
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            ceiling=ceiling,
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=1,
            coverage=coverage,
        )

    assert _STAT_VAR_A in str(excinfo.value)
    assert _STAT_VAR_B not in str(excinfo.value)
    assert len(transport_requests) == 1
    fetch_requests_by_member = [
        str(r.url.params.get("c[variableMeasured]"))
        for r in transport_requests
        if _is_data_fetch(r)
    ]
    # Member B's own fetch is skipped entirely under member_concurrency=1:
    # only A's aborted fetch reaches the transport.
    assert fetch_requests_by_member == [_BARE_A]


# ---------------------------------------------------------------------------
# A slow chunk holds one worker and the rest of the pool keeps fetching
# ---------------------------------------------------------------------------


def test_a_slow_chunk_leaves_every_other_worker_fetching_and_the_rows_in_dcid_order():
    """Six members land in six chunks on four workers. Each fast fetch answers only once the
    pool is full, or holds every chunk not yet answered, and the first chunk answers only once
    the other five have. That can happen only if chunks five and six are submitted while the
    first is still pending. The frame matches a serial read's, row for row."""
    members = [(f"Stat_{letter}", f"country/{letter}") for letter in "ABCDEF"]
    slow_bare = members[0][0]
    workers = 4
    bodies = {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=bare,
            year="2015",
            value=str(index + 1),
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        for index, (bare, _entity) in enumerate(members)
    }
    changed = threading.Condition()
    in_flight = 0
    answered = 0
    pool_full: list[bool] = []
    slow_released: list[bool] = []

    def respond(bare: str) -> httpx.Response:
        return httpx.Response(
            200,
            text=bodies[bare],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    def stalling(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, answered
        bare = str(request.url.params.get("c[variableMeasured]"))
        if not _is_data_fetch(request):
            return respond(bare)
        with changed:
            in_flight += 1
            changed.notify_all()
            if bare == slow_bare:
                slow_released.append(
                    changed.wait_for(lambda: answered == len(members) - 1, timeout=2.0)
                )
            else:
                pool_full.append(
                    changed.wait_for(
                        lambda: in_flight == min(workers, len(members) - answered), timeout=2.0
                    )
                )
            in_flight -= 1
            answered += 1
            changed.notify_all()
        return respond(bare)

    def read(handler, member_concurrency: int):
        sdmx = SdmxClient(
            base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
        )
        graph = FakeGraphReader(
            facets={(bare, entity): [_SHARED_FACET] for bare, entity in members}
        )
        return materialise_group(
            [
                _request(stat_var=f"dcid:{bare}", facet_bindings=_BOUND_FACETS, entity=entity)
                for bare, entity in members
            ],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=member_concurrency,
            coverage=_coverage_for([(bare, "facet1", entity) for bare, entity in members]),
        )

    pooled = read(stalling, workers)
    serial = read(lambda request: respond(str(request.url.params.get("c[variableMeasured]"))), 1)

    assert slow_released == [True]
    assert pool_full == [True] * (len(members) - 1)
    pooled_rows = [row for frame in pooled.frame_chunks() for row in frame.table.to_pylist()]
    serial_rows = [row for frame in serial.frame_chunks() for row in frame.table.to_pylist()]
    assert pooled_rows == serial_rows
    assert [row["stat_var"] for row in pooled_rows] == [f"dcid:{bare}" for bare, _entity in members]
    assert pooled.content_version == serial.content_version


# ---------------------------------------------------------------------------
# A stalled head chunk bounds how far dispatch runs ahead of it
# ---------------------------------------------------------------------------


def test_a_stalled_first_chunk_lets_dispatch_run_at_most_the_look_ahead():
    """Twelve members land in twelve chunks on two workers. The first chunk stalls until the
    others stop arriving: three answered, then 0.2 s with no fourth. Only chunks within
    `_DISPATCH_LOOK_AHEAD_POOLS * member_concurrency` of the first one are sent while it is
    pending, so the
    completed chunks waiting on it stay bounded. The frame matches a serial read's."""
    members = [(f"Stat_{index:02d}", f"country/M{index:02d}") for index in range(12)]
    slow_bare = members[0][0]
    workers = 2
    look_ahead = _DISPATCH_LOOK_AHEAD_POOLS * workers
    bodies = {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=bare,
            year="2015",
            value=str(index + 1),
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        for index, (bare, _entity) in enumerate(members)
    }
    changed = threading.Condition()
    answered_before_head: list[int] = []
    answered = 0

    def respond(bare: str) -> httpx.Response:
        return httpx.Response(
            200,
            text=bodies[bare],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    def stalling(request: httpx.Request) -> httpx.Response:
        nonlocal answered
        bare = str(request.url.params.get("c[variableMeasured]"))
        if not _is_data_fetch(request):
            return respond(bare)
        with changed:
            if bare == slow_bare:
                changed.wait_for(lambda: answered >= look_ahead - 1, timeout=2.0)
                changed.wait_for(lambda: answered >= look_ahead, timeout=0.2)
                answered_before_head.append(answered)
            else:
                answered += 1
                changed.notify_all()
        return respond(bare)

    def read(handler, member_concurrency: int):
        sdmx = SdmxClient(
            base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
        )
        graph = FakeGraphReader(
            facets={(bare, entity): [_SHARED_FACET] for bare, entity in members}
        )
        return materialise_group(
            [
                _request(stat_var=f"dcid:{bare}", facet_bindings=_BOUND_FACETS, entity=entity)
                for bare, entity in members
            ],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            member_concurrency=member_concurrency,
            coverage=_coverage_for([(bare, "facet1", entity) for bare, entity in members]),
        )

    pooled = read(stalling, workers)
    serial = read(lambda request: respond(str(request.url.params.get("c[variableMeasured]"))), 1)

    assert answered_before_head == [look_ahead - 1]
    pooled_rows = [row for frame in pooled.frame_chunks() for row in frame.table.to_pylist()]
    serial_rows = [row for frame in serial.frame_chunks() for row in frame.table.to_pylist()]
    assert pooled_rows == serial_rows
    assert pooled.content_version == serial.content_version


# ---------------------------------------------------------------------------
# `stream_rows` learns of finished fetches without waiting for the head
# ---------------------------------------------------------------------------


class _StagedChunks:
    """Holds chosen chunks of a group read, fails some of them, and logs what `stream_rows`
    submits and blocks on, so a test can time a chunk's end against the loop's own steps.

    Chunk `n` is the group's `n`th member, which sits alone in its chunk. A held chunk waits
    for `release`, and a failing chunk raises once released. `on_submit[n]` runs on the calling
    thread right after chunk `n` is submitted. `on_collected[n]` runs right after a `wait`
    returns chunk `n`'s future. Each hook fires once."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        held: AbstractSet[int],
        failing: AbstractSet[int] = frozenset(),
    ) -> None:
        self.log: list[str] = []
        self.on_submit: dict[int, Callable[[], None]] = {}
        self.on_collected: dict[int, Callable[[], None]] = {}
        self._gates = {chunk: threading.Event() for chunk in held}
        self._futures: dict[int, Future] = {}
        real_generate = _GroupRowStreamer._generate_one_chunk
        real_submit = generation.submit_with_context
        real_wait = generation.wait

        def generate(streamer, chunk):
            index = chunk.member_indices[0]
            if gate := self._gates.get(index):
                assert gate.wait(timeout=5.0), f"chunk {index} was never released"
            if index in failing:
                raise RuntimeError(f"chunk {index} failed")
            return real_generate(streamer, chunk)

        def submit(pool, fn, chunk):
            index = chunk.member_indices[0]
            future = real_submit(pool, fn, chunk)
            self._futures[index] = future
            self.log.append(f"submit {index}")
            if hook := self.on_submit.pop(index, None):
                hook()
            return future

        def wait(in_flight, **kwargs):
            if kwargs.get("timeout") is None:
                self.log.append("block")
            done, pending = real_wait(in_flight, **kwargs)
            for future in done:
                if hook := self.on_collected.pop(in_flight[future], None):
                    hook()
            return done, pending

        monkeypatch.setattr(_GroupRowStreamer, "_generate_one_chunk", generate)
        monkeypatch.setattr(generation, "submit_with_context", submit)
        monkeypatch.setattr(generation, "wait", wait)

    def release(self, *chunks: int) -> None:
        for chunk in chunks:
            self._gates[chunk].set()

    def finish(self, *chunks: int) -> None:
        """Releases the chunks and returns once each one's future is done."""
        self.release(*chunks)
        _done, not_done = futures_wait([self._futures[chunk] for chunk in chunks], timeout=5.0)
        assert not not_done, "a released chunk never finished"

    def submitted(self) -> list[int]:
        return [int(entry.split()[1]) for entry in self.log if entry.startswith("submit")]

    def close(self) -> None:
        self.release(*self._gates)


def _read_group_of(count: int, *, workers: int):
    """Reads `count` single-member chunks through `materialise_group` on `workers` workers."""
    members = [(f"Stat_{index:02d}", f"country/M{index:02d}") for index in range(count)]
    bodies = {
        bare: _CSV_HEADER
        + _csv_row(
            stat_var=bare,
            year="2015",
            value=str(index + 1),
            unit="UNIT1",
            method="METH1",
            provenance="PROV1",
            facet_id="facet1",
        )
        for index, (bare, _entity) in enumerate(members)
    }

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=bodies[str(request.url.params.get("c[variableMeasured]"))],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    graph = FakeGraphReader(facets={(bare, entity): [_SHARED_FACET] for bare, entity in members})
    return materialise_group(
        [
            _request(stat_var=f"dcid:{bare}", facet_bindings=_BOUND_FACETS, entity=entity)
            for bare, entity in members
        ],
        sdmx=SdmxClient(
            base_url="https://datacommons.one.org/core/api",
            transport=httpx.MockTransport(respond),
        ),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=workers,
        coverage=_coverage_for([(bare, "facet1", entity) for bare, entity in members]),
    )


def test_no_chunk_is_submitted_after_a_failure_that_ended_while_the_head_was_ready(monkeypatch):
    """Two workers, chunk 0 slow, chunk 2 failing. Chunk 2 fails right after chunk 0 is
    collected, while chunk 0 is the head and chunk 1 is already waiting in `ready`. The loop
    submits again before its next blocking wait, and the failure is still in `in_flight`
    uncollected, so it must be seen first: nothing past chunk 2 is submitted."""
    staged = _StagedChunks(monkeypatch, held={0, 2}, failing={2})
    staged.on_submit[2] = lambda: staged.release(0)
    staged.on_collected[0] = lambda: staged.finish(2)
    try:
        with pytest.raises(RuntimeError, match="chunk 2 failed"):
            _read_group_of(8, workers=2)
    finally:
        staged.close()

    assert staged.submitted() == [0, 1, 2]


def test_finished_fetches_free_their_workers_before_the_next_chunks_are_submitted(monkeypatch):
    """Three workers, chunks 2 and 3 held until chunk 0 is collected, then both finish.
    With the head ready, the loop refills before blocking: chunks 4 and 5 go out back to back
    with no wait between them, and no blocking wait collected chunks 2 and 3."""
    staged = _StagedChunks(monkeypatch, held={0, 2, 3})
    staged.on_submit[3] = lambda: staged.release(0)
    staged.on_collected[0] = lambda: staged.finish(2, 3)
    try:
        _read_group_of(10, workers=3)
    finally:
        staged.close()

    assert staged.log[staged.log.index("submit 4") + 1] == "submit 5"
