"""Group fetch concurrency, error ordering, and serial dispatch at concurrency 1."""

from __future__ import annotations

import threading

import httpx
import pytest

from dc_sdmx import SdmxClient, SdmxResponseError
from materialise import EnrichmentLookup, materialise_group
from materialise.errors import OverCeilingError
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
    _SNAPSHOT,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _csv_row,
    _request,
)
from .conftest import FakeGraphReader, FakeRevisionReader


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
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
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
            enrichment=EnrichmentLookup(graph),
            revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
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
            enrichment=EnrichmentLookup(graph),
            revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
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
