"""Tests the raw-fetch guard for one member and aggregate materialisation limits for collections.

The composed mint path uses real registry and materialisation code with fake graph and SDMX
responses. Tests lower `_MATERIALISE_RAW_FETCH_GUARD_ROWS` to exercise early refusal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

import datasets_api.app as app_module
from datasets_api.config import DatasetsBuildSettings
from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxCapExceededError, SdmxClient
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_spec
from materialise.errors import OverCeilingError
from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.revision import RevisionSnapshot
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise.conftest import FakeGraphReader, FakeRevisionReader

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2000",
    latest_date="2099",
)

_SCOPE = {
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


def _csv_row(*, stat_var: str, year: str) -> str:
    """Writes the bare dcid to `variableMeasured`, matching SDMX queries and response values."""
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var.removeprefix('dcid:')},{_RECIPIENT},USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,{year},1000,,facet1\r\n"
    )


def _csv_body(*, stat_var: str, num_rows: int) -> str:
    rows = "".join(_csv_row(stat_var=stat_var, year=str(2000 + i)) for i in range(num_rows))
    return _CSV_HEADER + rows


def _graph_for(stat_vars: list[str]) -> FakeMintGraphReader:
    arcs = {
        stat_var.removeprefix("dcid:"): {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        }
        for stat_var in stat_vars
    }
    # The graph facet probe keys on the bare dcid, same as `arcs` above.
    facets = {(stat_var.removeprefix("dcid:"), _RECIPIENT): [_FACET] for stat_var in stat_vars}
    return FakeMintGraphReader(
        arcs=arcs, facets=facets, labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"}
    )


def _sdmx_transport(
    bodies_by_stat_var: dict[str, str], requests: list[httpx.Request]
) -> httpx.MockTransport:
    """`bodies_by_stat_var` is keyed on the bare dcid. `sdmx.data` sends
    `c[variableMeasured]` bare, so that is what a real request's own query
    value matches against. `c[variableMeasured]` may name more than one
    stat_var at once (`materialise.generation._generate_chunk`'s combined
    SDMX request for a collection whose members share an identical entity
    scope, `_SCOPE` below). Each requested stat_var's own data rows are
    concatenated under one header, splitting the combined query value back
    into its individual stat_vars first.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        requested_vars = str(request.url.params.get("c[variableMeasured]")).split(",")
        rows = "".join(
            bodies_by_stat_var[stat_var].removeprefix(_CSV_HEADER)
            for stat_var in requested_vars
            if stat_var in bodies_by_stat_var
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _csv_body_for(*, stat_var: str, recipient: str, num_rows: int) -> str:
    """Builds SDMX rows for a member and recipient, preserving the entity filter."""
    rows = "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var.removeprefix('dcid:')},{recipient},USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,{2000 + i},1000,,facet1\r\n"
        for i in range(num_rows)
    )
    return _CSV_HEADER + rows


def _aggregate_ceiling_spec_id(stat_var: str) -> str:
    """Returns a deterministic spec id in the required `spec_` plus 32-hex form."""
    return "spec_" + hashlib.sha256(stat_var.encode()).hexdigest()[:32]


def _aggregate_ceiling_request(*, stat_var: str, recipient: str) -> MaterialisationRequest:
    """Builds a pinned request for direct tests with distinct recipient scopes."""
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(state="bound", ref="USD"),
        },
        window=RequestedWindow(kind="all"),
    )


def _mint_body(*, statvars: list[str]) -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": statvars, "scope": _SCOPE},
        "options": {"bundle": False},
    }


def _coverage_index(members: list[tuple[str, str]]) -> CoverageIndex:
    """Builds a one-facet-per-member index matching `_FACET` for each StatVar and recipient.

    One in-range year supplies each candidate and confirms the bound facet refs.
    """
    bare = [(stat_var.removeprefix("dcid:"), recipient) for stat_var, recipient in members]
    return build_coverage_index(
        [(stat_var, "facet1", "country/USA", recipient, [2020]) for stat_var, recipient in bare],
        role_names={
            stat_var: RoleNames(donor="provider", recipient="recipient") for stat_var, _ in bare
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )


def _coverage_service(members: list[tuple[str, str]]) -> FakeCoverageService:
    return FakeCoverageService(_coverage_index(members))


def _client(
    *,
    monkeypatch: pytest.MonkeyPatch,
    ceiling: int,
    graph,
    sdmx,
    coverage: FakeCoverageService,
    member_concurrency: int | None = None,
):
    monkeypatch.setattr(app_module, "_MATERIALISE_RAW_FETCH_GUARD_ROWS", ceiling)
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        settings=(
            DatasetsBuildSettings(mint_member_max_concurrency=member_concurrency)
            if member_concurrency is not None
            else None
        ),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage,
    )


def test_single_member_far_over_raw_guard_refuses_over_ceiling_without_reading_the_full_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """With a ceiling of 5, the raw-fetch guard is 100 rows. A 2025 window keeps one row from
    the response, so the raw guard stops the multi-block read before it reaches the body end.
    """
    stat_var = "dcid:BoundsTestFarOverCeiling"
    ceiling = 5
    raw_guard = 20 * ceiling
    from dc_sdmx.client import _STREAM_BATCH_BLOCK_SIZE

    one_row = _csv_row(stat_var=stat_var, year="2000")
    num_rows = -(-10 * _STREAM_BATCH_BLOCK_SIZE // len(one_row))
    requests: list[httpx.Request] = []

    sdmx = SdmxClient(
        base_url=_BASE_URL,
        transport=_sdmx_transport(
            {stat_var.removeprefix("dcid:"): _csv_body(stat_var=stat_var, num_rows=num_rows)},
            requests,
        ),
    )
    client = _client(
        monkeypatch=monkeypatch,
        ceiling=ceiling,
        graph=_graph_for([stat_var]),
        sdmx=sdmx,
        coverage=_coverage_service([(stat_var, _RECIPIENT)]),
    )
    scope = {**_SCOPE, "window": {"kind": "range", "start_year": 2025, "end_year": 2025}}
    body = _mint_body(statvars=[stat_var])
    body["input"]["scope"] = scope

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "over_ceiling"
    # `stream_data_batches` checks `max_rows` on each batch's cumulative
    # row count, so the cap trips once `raw_guard + 1` rows have been
    # parsed, plus up to one batch's worth of overshoot, well short of the
    # upstream body's own full row count. This proves the read stopped
    # early.
    assert str(num_rows) not in problem["detail"]
    # The detail identifies the raw-fetch guard as the limit that tripped.
    assert f"raw fetch guard of {raw_guard} rows" in problem["detail"]
    # Coverage supplies the facet candidate, so the transport sees one fetch.
    # The cap aborts that fetch without a retry.
    assert len(requests) == 1


def test_counted_early_refusal_wire_shape_matches_the_raw_guard_refusal(
    monkeypatch: pytest.MonkeyPatch,
):
    """All rows match the bound facet, so the survivor count crosses 5 before the 100-row raw
    guard. The 80-block body exceeds pyarrow's read-ahead window and lets the test verify an
    early stop. Coverage supplies the facet, so SDMX handles one fetch.
    """
    from dc_sdmx.client import _STREAM_BATCH_BLOCK_SIZE
    from tests.dc_sdmx.conftest import StreamingTransport

    stat_var = "dcid:BoundsTestCountedEarlyRefusal"
    ceiling = 5
    one_row = _csv_row(stat_var=stat_var, year="2000")
    num_rows = -(-80 * _STREAM_BATCH_BLOCK_SIZE // len(one_row))  # well over the read-ahead window
    csv_body = _csv_body(stat_var=stat_var, num_rows=num_rows).encode()
    transport = StreamingTransport(body=csv_body, chunk_size=4096)
    sdmx = SdmxClient(base_url=_BASE_URL, transport=transport)
    client = _client(
        monkeypatch=monkeypatch,
        ceiling=ceiling,
        graph=_graph_for([stat_var]),
        sdmx=sdmx,
        coverage=_coverage_service([(stat_var, _RECIPIENT)]),
    )

    resp = client.post("/api/datasets", json=_mint_body(statvars=[stat_var]))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "over_ceiling"
    assert problem["detail"] is not None
    assert "was not measured" in problem["detail"]
    # The read stopped well short of every chunk of the body.
    assert transport.chunks_yielded < transport.total_chunks


@pytest.mark.parametrize("member_concurrency", [1, 4, 8])
def test_collection_aggregate_over_ceiling_refuses_before_every_member_is_read(
    member_concurrency: int,
):
    """Three members contribute 2 rows each, exceeding the ceiling of 3 with 6 rows.
    Distinct recipients create separate SDMX chunks, so this test calls `materialise_collection`
    directly. The mint door uses a shared scope.

    Concurrency 1 reads two members. Concurrency 4 or 8 may fetch extras. The refusal reports
    the same request-order prefix and row count.
    """
    members = [
        ("dcid:BoundsTestAggregateA", "country/KEN"),
        ("dcid:BoundsTestAggregateB", "country/UGA"),
        ("dcid:BoundsTestAggregateC", "country/TZA"),
    ]
    ceiling = 3
    requests: list[httpx.Request] = []
    from materialise import materialise_collection

    # Each member alone carries 2 rows, comfortably under the ceiling of
    # 3, but any two members together already carry 4, over it.
    bodies = {
        stat_var.removeprefix("dcid:"): _csv_body_for(
            stat_var=stat_var, recipient=recipient, num_rows=2
        )
        for stat_var, recipient in members
    }
    graph = FakeGraphReader(
        facets={
            (stat_var.removeprefix("dcid:"), recipient): [_FACET] for stat_var, recipient in members
        }
    )
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, requests))
    snapshot = RevisionSnapshot(graph_revision="rev-1", source_revisions=())
    reader = FakeRevisionReader([snapshot, snapshot])

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_collection(
            [
                (
                    _aggregate_ceiling_spec_id(stat_var),
                    _aggregate_ceiling_request(stat_var=stat_var, recipient=recipient),
                )
                for stat_var, recipient in members
            ],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=reader,
            ceiling=ceiling,
            materialiser_build="test-build",
            capability_version="cap-v1",
            member_concurrency=member_concurrency,
            coverage=_coverage_index(members),
        )

    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.detail == (
        "collection's aggregate materialised coverage of 4 rows across 2 member(s) "
        f"exceeds the sync materialisation ceiling of {ceiling} rows"
    )
    # Coverage supplies each distinct-recipient facet candidate.
    # Data fetches reach the transport.
    if member_concurrency == 1:
        # The second member crosses the ceiling, so only two SDMX reads run.
        assert len(requests) == 2
    else:
        assert len(requests) <= 3


def test_collection_aggregate_over_ceiling_shared_scope_fetches_one_combined_chunk(
    monkeypatch: pytest.MonkeyPatch,
):
    """All three members share `_SCOPE`, so one SDMX request returns 6 rows across 3 members.
    The coverage estimate is 3 rows. With a raw-guard floor of 3, the effective ceiling is
    `ceil(3 * 1.5) = 5`.

    Concurrency is pinned to 1 to keep the members in one combined chunk.
    """
    stat_vars = [
        "dcid:BoundsTestAggregateA",
        "dcid:BoundsTestAggregateB",
        "dcid:BoundsTestAggregateC",
    ]
    ceiling = 3
    effective_ceiling = 5
    requests: list[httpx.Request] = []

    # Each member carries 2 rows, and the shared scope puts all 3 in one chunk.
    # Their 6 rows exceed the effective ceiling of 5.
    bodies = {
        stat_var.removeprefix("dcid:"): _csv_body(stat_var=stat_var, num_rows=2)
        for stat_var in stat_vars
    }
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, requests))
    client = _client(
        monkeypatch=monkeypatch,
        ceiling=ceiling,
        graph=_graph_for(stat_vars),
        sdmx=sdmx,
        coverage=_coverage_service([(stat_var, _RECIPIENT) for stat_var in stat_vars]),
        member_concurrency=1,
    )

    resp = client.post("/api/datasets", json=_mint_body(statvars=stat_vars))

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "over_ceiling"
    assert problem["detail"] == (
        "collection's aggregate materialised coverage of 6 rows across 3 member(s) "
        f"exceeds the sync materialisation ceiling of {effective_ceiling} rows"
    )
    # The shared scope produces one combined fetch. Coverage supplies the facet candidates.
    assert len(requests) == 1


def test_collection_aggregate_exactly_at_ceiling_succeeds():
    """Two members contribute 2 rows each, exactly meeting the ceiling of 4. Their distinct
    recipient scopes produce separate SDMX chunks.
    """
    members = [
        ("dcid:BoundsTestExactCeilingA", "country/KEN"),
        ("dcid:BoundsTestExactCeilingB", "country/UGA"),
    ]
    ceiling = 4
    requests: list[httpx.Request] = []
    from materialise import materialise_collection

    bodies = {
        stat_var.removeprefix("dcid:"): _csv_body_for(
            stat_var=stat_var, recipient=recipient, num_rows=2
        )
        for stat_var, recipient in members
    }
    graph = FakeGraphReader(
        facets={
            (stat_var.removeprefix("dcid:"), recipient): [_FACET] for stat_var, recipient in members
        }
    )
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, requests))
    snapshot = RevisionSnapshot(graph_revision="rev-1", source_revisions=())
    reader = FakeRevisionReader([snapshot, snapshot])

    result = materialise_collection(
        [
            (
                _aggregate_ceiling_spec_id(stat_var),
                _aggregate_ceiling_request(stat_var=stat_var, recipient=recipient),
            )
            for stat_var, recipient in members
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        ceiling=ceiling,
        materialiser_build="test-build",
        capability_version="cap-v1",
        coverage=_coverage_index(members),
    )

    assert len(result.members) == 2
    # Each distinct-recipient group produces one fetch. Coverage supplies each facet candidate.
    assert len(requests) == 2


class _AlwaysCapExceededSdmxClient:
    """A minimal SDMX client double whose `stream_data_batches()` raises a chosen cap error.

    Coverage supplies facet candidates, so tests call only the fetch method.
    """

    def __init__(self, *, cap_kind: str, cap: int, observed: int) -> None:
        self._cap_kind = cap_kind
        self._cap = cap
        self._observed = observed

    @contextmanager
    def stream_data_batches(self, **_kwargs: object) -> Iterator[object]:
        raise SdmxCapExceededError(
            f"SDMX /data read exceeded the {self._cap_kind} cap of {self._cap}",
            cap_kind=self._cap_kind,
            cap=self._cap,
            observed=self._observed,
        )
        yield  # pragma: no cover, unreachable; makes this a generator function


def test_byte_cap_exceeded_labels_the_detail_in_bytes_not_rows():
    """A byte-cap refusal labels `observed` in bytes."""
    stat_var = "dcid:BoundsTestByteCap"
    facet = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2000",
        latest_date="2099",
    )
    # The graph facet probe keys on the bare dcid.
    graph = FakeGraphReader(facets={(stat_var.removeprefix("dcid:"), _RECIPIENT): [facet]})
    request = MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [_RECIPIENT]},
        declared_roles=("recipient",),
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(state="bound", ref="USD"),
        },
        window=RequestedWindow(kind="all"),
    )
    sdmx = _AlwaysCapExceededSdmxClient(cap_kind="bytes", cap=1_000_000, observed=1_048_576)
    snapshot = RevisionSnapshot(graph_revision="rev-1", source_revisions=())
    reader = FakeRevisionReader([snapshot, snapshot])

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=reader,
            ceiling=5,
            materialiser_build="test-build",
            capability_version="cap-v1",
            coverage=_coverage_index([(stat_var, _RECIPIENT)]),
        )

    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.detail is not None
    assert "1048576 bytes" in excinfo.value.detail
    assert "1048576 rows" not in excinfo.value.detail
