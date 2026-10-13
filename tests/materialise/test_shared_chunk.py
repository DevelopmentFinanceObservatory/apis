"""One shared SDMX chunk for several group members: its survivor ceiling and a fetch the
mixer rejects."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import httpx
import pyarrow as pa
import pytest

from dc_sdmx import SdmxClient, SdmxIncompatibleVariablesError
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_group
from materialise import generation as generation_module
from materialise.errors import MembersNotHeldError, OverCeilingError
from materialise.facets import CoverageAdmission, FacetBinding, FacetSelection
from materialise.graph import Facet, GraphReadError
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._table_support import table_from_cells
from .conftest import FakeGraphReader

_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
    "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
    "unit": FacetBinding(state="bound", ref="USD"),
}


def _facet(facet_id: str) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )


def _coverage_for(entries: list[tuple[str, str, str]]) -> CoverageIndex:
    """Build coverage for each StatVar/recipient/facet triple with metadata
    matching `_facet` and `_BOUND_FACETS`.
    """
    bare_stat_vars = {bare for bare, _, _ in entries}
    role_names = {bare: RoleNames(donor="donor", recipient="recipient") for bare in bare_stat_vars}
    facet_ids = {facet_id for _, _, facet_id in entries}
    facet_metadata = {
        facet_id: FacetMetadata(
            provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
        )
        for facet_id in facet_ids
    }
    rows = [
        (bare, facet_id, "country/DONOR", recipient, [2015])
        for bare, recipient, facet_id in entries
    ]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _request_for(*, stat_var: str, recipient: str) -> MaterialisationRequest:
    """One member bound to `recipient`. Two members with the same `recipient` share one chunk
    (`_plan_collection_chunks`) and stay distinguishable by `stat_var`."""
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )


def _sdmx_client_for(bodies_by_entity: dict[str, str]) -> tuple[SdmxClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        query = str(request.url.params.get("c[recipient]"))
        return httpx.Response(
            200,
            text=bodies_by_entity[query],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    ), requests


def test_generate_chunk_survivor_ceiling_overshoot_bounded_to_one_batch():
    """With ceiling 30, the 500 surviving rows are checked in 20-row batches,
    bounding overshoot at 20. Two members share the chunk, so the error
    includes its multi-member clause.
    """
    stat_var_0 = "dcid:ChunkOvershootBoundMemberA"
    stat_var_1 = "dcid:ChunkOvershootBoundMemberB"
    bare_0 = stat_var_0.removeprefix("dcid:")
    bare_1 = stat_var_1.removeprefix("dcid:")
    recipient = "country/KEN"
    ceiling = 30
    batch_size = 20

    def _cell(*, stat_var: str, year: str, facet_id: str) -> dict[str, str | None]:
        return {
            "variableMeasured": stat_var,
            "recipient": recipient,
            "unit": "USD",
            "measurementMethod": "GrantEquivalent",
            "observationPeriod": None,
            "provenance": "OECD_DAC2A",
            "TIME_PERIOD": year,
            "OBS_VALUE": "1000",
            "scalingFactor": None,
            "facetId": facet_id,
        }

    cells = [_cell(stat_var=bare_0, year="2010", facet_id="facet1")]
    cells += [_cell(stat_var=bare_1, year=str(2000 + i), facet_id="facet2") for i in range(500)]
    table = table_from_cells(cells, roles=("recipient",))

    consumed: list[int] = []

    def _batches() -> Iterator[pa.RecordBatch]:
        for batch in table.to_batches(max_chunksize=batch_size):
            consumed.append(batch.num_rows)
            yield batch

    @contextmanager
    def fake_stream_data_batches(
        **kwargs: object,
    ) -> Iterator[tuple[tuple[str, ...], Iterator[pa.RecordBatch]]]:
        yield ("recipient",), _batches()

    sdmx, _requests = _sdmx_client_for({})
    sdmx.stream_data_batches = fake_stream_data_batches  # type: ignore[method-assign]

    selections = {
        axis: FacetSelection(
            axis=axis,
            ref=ref,
            basis="bound",
            default_policy=None,
            default_policy_version=None,
            selection_reason=None,
            dispatch_facet_id="facet1",
        )
        for axis, ref in (
            ("provenance", "OECD_DAC2A"),
            ("measurementMethod", "GrantEquivalent"),
            ("unit", "USD"),
        )
    }
    requests = [
        ("0", _request_for(stat_var=stat_var_0, recipient=recipient)),
        ("1", _request_for(stat_var=stat_var_1, recipient=recipient)),
    ]
    plans = [
        generation_module._MemberFetchPlan(
            selections=selections,
            dispatch_ids=("facet1",),
            admission=CoverageAdmission(),
            conversion_plan=None,
        ),
        generation_module._MemberFetchPlan(
            selections=selections,
            dispatch_ids=("facet2",),
            admission=CoverageAdmission(),
            conversion_plan=None,
        ),
    ]
    chunk = generation_module._ShapeChunk(
        member_indices=(0, 1),
        bare_stat_vars=(bare_0, bare_1),
        role_bindings=requests[0][1].role_bindings,
        index_by_stat_var={bare_0: 0, bare_1: 1},
    )

    with pytest.raises(OverCeilingError) as excinfo:
        generation_module._generate_chunk(
            chunk,
            requests,
            plans,
            sdmx=sdmx,
            graph=FakeGraphReader(),
            ceiling=ceiling,
            raw_budget=generation_module._RawRowBudget(ceiling),
        )

    message = str(excinfo.value)
    assert stat_var_1 in message
    assert stat_var_0 not in message
    assert (
        "fetched together with 1 other member(s) sharing this member's own "
        "entity scope, in one combined SDMX request"
    ) in message

    member_1_survivors_at_abort = sum(consumed) - 1  # member 0's own single row
    assert member_1_survivors_at_abort > ceiling
    assert member_1_survivors_at_abort <= ceiling + batch_size


_REJECTED_CHUNK_STAT_VARS = ("SharedChunkAbsentMember", "SharedChunkHeldMember")


_REJECTED_CHUNK_RECIPIENT = "country/KEN"


def _rejected_chunk_graph(*, held: tuple[str, ...]) -> FakeGraphReader:
    """Facets for both shared-chunk members; a graph node only for `held`."""
    return FakeGraphReader(
        facets={
            (bare, _REJECTED_CHUNK_RECIPIENT): [_facet("facet1")]
            for bare in _REJECTED_CHUNK_STAT_VARS
        },
        arcs={bare: {"typeOf": ("StatisticalVariable",)} for bare in held},
    )


def _materialise_rejected_chunk(graph: FakeGraphReader) -> None:
    """Materialises two members in one chunk (one lane) whose SDMX fetch the
    mixer rejects for incompatible observation properties."""
    rejection = json.dumps(
        {
            "message": "resolveSdmxEntityShape: incompatible observationProperties for "
            'stat var "SharedChunkAbsentMember": got [observationAbout], want [recipient]',
            "code": 3,
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=rejection, headers={"content-type": "application/json"})

    materialise_group(
        [
            _request_for(stat_var=f"dcid:{bare}", recipient=_REJECTED_CHUNK_RECIPIENT)
            for bare in _REJECTED_CHUNK_STAT_VARS
        ],
        sdmx=SdmxClient(
            base_url="https://datacommons.one.org/core/api",
            transport=httpx.MockTransport(handler),
        ),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=_coverage_for(
            [(bare, _REJECTED_CHUNK_RECIPIENT, "facet1") for bare in _REJECTED_CHUNK_STAT_VARS]
        ),
    )


def test_rejected_chunk_names_the_stat_var_with_no_graph_node():
    graph = _rejected_chunk_graph(held=("SharedChunkHeldMember",))

    with pytest.raises(MembersNotHeldError) as excinfo:
        _materialise_rejected_chunk(graph)

    assert excinfo.value.stat_vars == ("SharedChunkAbsentMember",)
    assert isinstance(excinfo.value.__cause__, SdmxIncompatibleVariablesError)
    assert graph.node_arcs_many_calls[-1] == (_REJECTED_CHUNK_STAT_VARS, ("typeOf",))
    # A budgeted probe maps unprobed StatVars to `None`, the same as absent ones.
    assert graph.last_individual_probe_budget is None


def test_rejected_chunk_with_every_stat_var_held_keeps_the_sdmx_error():
    graph = _rejected_chunk_graph(held=_REJECTED_CHUNK_STAT_VARS)

    with pytest.raises(SdmxIncompatibleVariablesError):
        _materialise_rejected_chunk(graph)


class _TypeOfProbeFailsGraphReader(FakeGraphReader):
    """Fails the `typeOf` probe the way a graph outage would."""

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        if tuple(properties) == ("typeOf",):
            raise GraphReadError("graph unavailable")
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


def test_rejected_chunk_whose_probe_fails_keeps_the_sdmx_error():
    graph = _TypeOfProbeFailsGraphReader(
        facets={
            (bare, _REJECTED_CHUNK_RECIPIENT): [_facet("facet1")]
            for bare in _REJECTED_CHUNK_STAT_VARS
        },
    )

    with pytest.raises(SdmxIncompatibleVariablesError):
        _materialise_rejected_chunk(graph)
