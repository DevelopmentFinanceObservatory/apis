"""Collection generation, concurrency, row accounting, cancellation, and resource release."""

from __future__ import annotations

import gc
import json
import threading
import time
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import httpx
import pyarrow as pa
import pytest
from dfo_frame_grammar import canonical_frame_bytes

from dc_sdmx import SdmxClient, SdmxIncompatibleVariablesError, SdmxResponseError
from materialise import (
    EnrichmentLookup,
    MaterialisationCancelled,
    MaterialisationRequest,
    RowProgress,
    materialise_collection,
    materialise_spec,
)
from materialise import generation as generation_module
from materialise.content_version import mint_collection_version
from materialise.errors import MaterialisationUsageError, MembersNotHeldError, OverCeilingError
from materialise.facets import CoverageAdmission, FacetBinding, FacetSelection
from materialise.graph import Facet, GraphReadError
from materialise.revision import RevisionSnapshot, SourceRevision
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._table_support import table_from_cells
from .conftest import FakeGraphReader, FakeRevisionReader, only_chunk

_STAT_VAR = "dcid:MalariaGrantDisbursements"
# Graph requests use bare dcids. The fake fixtures use that wire form.
_BARE_STAT_VAR = "MalariaGrantDisbursements"

_SPEC_ID_1 = "spec_00000000000000000000000000000001"
_SPEC_ID_2 = "spec_00000000000000000000000000000002"
_SPEC_ID_3 = "spec_00000000000000000000000000000003"

_SNAPSHOT = RevisionSnapshot(
    graph_revision="graph-rev-2024-01",
    source_revisions=(SourceRevision(source="OECD_DAC_CRS", revision="rev-7"),),
)

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


def _csv_body(*, recipient: str, facet_id: str) -> str:
    # SDMX reports bare dcids in `variableMeasured`. The chunk router uses that form.
    return (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
        f"dataflow,DC:DF_OBS(1.0.0),I,{_BARE_STAT_VAR},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,1000,,{facet_id}\r\n"
    )


def _multi_row_csv_body(*, recipient: str, facet_id: str, periods: tuple[str, ...]) -> str:
    header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )
    rows = "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{_BARE_STAT_VAR},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{period},1000,,{facet_id}\r\n"
        for period in periods
    )
    return header + rows


def _request(*, recipient: str) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )


def _request_for(*, stat_var: str, recipient: str) -> MaterialisationRequest:
    """Like `_request`, but takes `stat_var` as its own parameter. `_request`
    itself pulls `stat_var` from the module-wide `_STAT_VAR`. This lets a
    test give two members the same `recipient` (so `_plan_collection_chunks`
    puts them in one shared chunk) while keeping them distinguishable by
    `stat_var`.
    """
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )


def _request_latest_for(*, stat_var: str, recipient: str) -> MaterialisationRequest:
    """Like `_request_for`, but `window.kind="latest"`, for the tests
    exercising `_generate_chunk`'s `latest_indices` carve-out (raw,
    unfiltered per-member buffering) and its shared raw-row budget.
    """
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="latest"),
    )


def _is_data_fetch(request: httpx.Request) -> bool:
    """Data fetches include `c[facetId]`. LATEST probes omit the parameter."""
    return "c[facetId]" in request.url.params


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


def _two_member_requests() -> list[tuple[str, MaterialisationRequest]]:
    return [
        (_SPEC_ID_1, _request(recipient="country/TGO")),
        (_SPEC_ID_2, _request(recipient="country/KEN")),
    ]


def _graph_and_sdmx() -> tuple[FakeGraphReader, SdmxClient, list[httpx.Request], CoverageIndex]:
    graph = FakeGraphReader(
        facets={
            (_BARE_STAT_VAR, "country/TGO"): [_facet("facet1")],
            (_BARE_STAT_VAR, "country/KEN"): [_facet("facet2")],
        }
    )
    sdmx, requests = _sdmx_client_for(
        {
            "country/TGO": _csv_body(recipient="country/TGO", facet_id="facet1"),
            "country/KEN": _csv_body(recipient="country/KEN", facet_id="facet2"),
        }
    )
    coverage = _coverage_for(
        [
            (_BARE_STAT_VAR, "country/TGO", "facet1"),
            (_BARE_STAT_VAR, "country/KEN", "facet2"),
        ]
    )
    return graph, sdmx, requests, coverage


def test_generation_spans_every_member_read_called_exactly_twice_for_whole_batch():
    graph, sdmx, _requests, coverage = _graph_and_sdmx()
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])
    deadlines: list[float | None] = []
    original_stream_data_batches = sdmx.stream_data_batches

    def recording_stream_data_batches(**kwargs):
        deadlines.append(kwargs.get("deadline"))
        return original_stream_data_batches(**kwargs)

    sdmx.stream_data_batches = recording_stream_data_batches  # type: ignore[method-assign]
    deadline = time.monotonic() + 30

    result = materialise_collection(
        _two_member_requests(),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        deadline=deadline,
        coverage=coverage,
    )

    assert reader.read_count == 2
    assert len(result.members) == 2
    assert deadlines == [deadline, deadline]

    expected_cv = mint_collection_version(
        [(spec_id, member.content_version) for spec_id, member in result.members]
    )
    assert result.content_version == expected_cv


def test_progress_counter_ends_equal_to_the_collections_total_row_count():
    """Progress equals the sum of member row counts after all outcomes are absorbed."""
    graph, sdmx, _requests, coverage = _graph_and_sdmx()
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])
    progress = RowProgress()

    result = materialise_collection(
        _two_member_requests(),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
        progress=progress,
    )

    assert progress.value == sum(member.coverage.rows for _spec_id, member in result.members)
    assert progress.value > 0


class _CancelAfterFirstChunk(RowProgress):
    """Cancels itself once the first member's rows are absorbed, the shape
    of a heartbeat losing its fence partway through a collection."""

    def add(self, rows: int) -> None:
        super().add(rows)
        self.cancel()


def test_a_progress_cancelled_mid_collection_stops_at_the_next_member_boundary():
    graph, sdmx, _requests, coverage = _graph_and_sdmx()
    progress = _CancelAfterFirstChunk()

    with pytest.raises(MaterialisationCancelled):
        materialise_collection(
            _two_member_requests(),
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
            progress=progress,
        )
    assert progress.cancelled
    assert progress.value > 0


def test_reverse_order_input_mints_identical_collection_content_version():
    graph_forward, sdmx_forward, _r1, coverage_forward = _graph_and_sdmx()
    result_forward = materialise_collection(
        _two_member_requests(),
        sdmx=sdmx_forward,
        graph=graph_forward,
        enrichment=EnrichmentLookup(graph_forward),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_forward,
    )

    graph_reversed, sdmx_reversed, _r2, coverage_reversed = _graph_and_sdmx()
    result_reversed = materialise_collection(
        list(reversed(_two_member_requests())),
        sdmx=sdmx_reversed,
        graph=graph_reversed,
        enrichment=EnrichmentLookup(graph_reversed),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_reversed,
    )

    assert result_forward.content_version == result_reversed.content_version


def test_duplicate_member_spec_id_raises_before_any_network_call():
    graph, sdmx, requests, _coverage = _graph_and_sdmx()
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])
    duplicate_requests = [
        (_SPEC_ID_1, _request(recipient="country/TGO")),
        (_SPEC_ID_1, _request(recipient="country/KEN")),
    ]

    with pytest.raises(MaterialisationUsageError):
        materialise_collection(
            duplicate_requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=reader,
            materialiser_build="build-123",
            capability_version="cap-v1",
        )

    assert requests == []
    assert reader.read_count == 0
    assert sum(graph.calls.values()) == 0


def _three_member_requests() -> list[tuple[str, MaterialisationRequest]]:
    return [
        (_SPEC_ID_1, _request(recipient="country/TGO")),
        (_SPEC_ID_2, _request(recipient="country/KEN")),
        (_SPEC_ID_3, _request(recipient="country/ETH")),
    ]


def _graph_and_sdmx_for(
    recipients: list[str], *, handler
) -> tuple[FakeGraphReader, SdmxClient, CoverageIndex]:
    graph = FakeGraphReader(
        facets={
            (_BARE_STAT_VAR, recipient): [_facet(f"facet-{recipient}")] for recipient in recipients
        }
    )
    coverage = _coverage_for(
        [(_BARE_STAT_VAR, recipient, f"facet-{recipient}") for recipient in recipients]
    )
    return (
        graph,
        SdmxClient(
            base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
        ),
        coverage,
    )


def test_member_concurrency_dispatches_every_member_at_once_barrier_proof():
    """All three fetches must reach a `threading.Barrier` within 5 seconds.
    Serial dispatch raises `BrokenBarrierError`.
    """
    recipients = ["country/TGO", "country/KEN", "country/ETH"]
    barrier = threading.Barrier(len(recipients))
    bodies = {
        recipient: _csv_body(recipient=recipient, facet_id=f"facet-{recipient}")
        for recipient in recipients
    }

    def handler(request: httpx.Request) -> httpx.Response:
        recipient = str(request.url.params.get("c[recipient]"))
        # Probes run sequentially before fetch chunks are submitted.
        # The barrier therefore covers only the concurrent data fetches.
        if _is_data_fetch(request):
            barrier.wait(timeout=5.0)
        return httpx.Response(
            200,
            text=bodies[recipient],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    graph, sdmx, coverage = _graph_and_sdmx_for(recipients, handler=handler)
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])

    result = materialise_collection(
        _three_member_requests(),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=len(recipients),
        coverage=coverage,
    )

    assert len(result.members) == len(recipients)


def test_outcomes_stay_in_request_order_under_jittered_fake_completion():
    """An event makes the first member finish last. Results retain request order."""
    recipients = ["country/TGO", "country/KEN", "country/ETH"]
    bodies = {
        recipient: _csv_body(recipient=recipient, facet_id=f"facet-{recipient}")
        for recipient in recipients
    }
    later_members_answered = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        recipient = str(request.url.params.get("c[recipient]"))
        response = httpx.Response(
            200,
            text=bodies[recipient],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
        # `probe_collection_facets` probes every member sequentially before
        # any fetch chunk is submitted, so the jittered-completion rendezvous
        # below only makes sense for the real fetch calls.
        if not _is_data_fetch(request):
            return response
        if recipient == recipients[0]:
            assert later_members_answered.wait(timeout=5.0), (
                "the later members never answered ahead of the first"
            )
        if recipient == recipients[-1]:
            later_members_answered.set()
        return response

    graph, sdmx, coverage = _graph_and_sdmx_for(recipients, handler=handler)
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])

    result = materialise_collection(
        _three_member_requests(),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=len(recipients),
        coverage=coverage,
    )

    assert [spec_id for spec_id, _member in result.members] == [
        spec_id for spec_id, _request in _three_member_requests()
    ]


def _diverging_request(*, stat_var: str, recipient: str) -> MaterialisationRequest:
    """Declare one role absent from the SDMX response to trigger
    `SdmxResponseError` before any batch is yielded.
    """
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient", "a_role_the_response_never_reports"),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )


@pytest.mark.parametrize(
    "failure_path",
    [
        pytest.param("request-order-error", id="request-order-error"),
        pytest.param("declared-role-divergence", id="declared-role-divergence"),
        pytest.param("post-fetch-ceiling", id="post-fetch-ceiling"),
    ],
)
def test_collection_failure_waits_for_in_flight_sibling_before_it_surfaces(
    failure_path: str,
):
    """Keep request-order errors and in-flight-sibling shutdown as named rows."""
    recipient_0 = "country/TGO"
    recipient_1 = "country/KEN"
    stat_var_0 = (
        "dcid:CollectionFailFastMemberA"
        if failure_path == "request-order-error"
        else "dcid:CollectionErrorPathMemberA"
    )
    stat_var_1 = (
        "dcid:CollectionFailFastMemberB"
        if failure_path == "request-order-error"
        else "dcid:CollectionErrorPathMemberB"
    )
    bare_stat_var_0 = stat_var_0.removeprefix("dcid:")
    bare_stat_var_1 = stat_var_1.removeprefix("dcid:")
    member_1_started = threading.Event()
    release_member_1 = threading.Event()
    second_member_answered = threading.Event()
    completion_order: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if failure_path in {"request-order-error", "declared-role-divergence"}:
            stat_var = str(request.url.params.get("c[variableMeasured]"))
            recipient = recipient_1 if stat_var == bare_stat_var_1 else recipient_0
            body = _csv_body(recipient=recipient, facet_id="facet1").replace(
                _BARE_STAT_VAR, stat_var
            )
            member_1_fetch = stat_var == bare_stat_var_1
        else:
            recipient = str(request.url.params.get("c[recipient]"))
            stat_var = _BARE_STAT_VAR
            member_1_fetch = recipient == recipient_1
            if recipient == recipient_0:
                body = _multi_row_csv_body(
                    recipient=recipient_0, facet_id="facet1", periods=("2015", "2016")
                )
            else:
                body = _csv_body(recipient=recipient_1, facet_id="facet2")
        response = httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )
        # `probe_collection_facets` probes every member sequentially before
        # any fetch chunk is submitted, so the rendezvous below only makes
        # sense for the real fetch calls.
        if not _is_data_fetch(request):
            return response
        if failure_path == "request-order-error":
            if stat_var == bare_stat_var_0:
                assert second_member_answered.wait(timeout=5.0), (
                    "the second member never answered ahead of the first"
                )
            else:
                second_member_answered.set()
            return response
        if member_1_fetch:
            member_1_started.set()
            assert release_member_1.wait(timeout=5.0), "member 1's fetch was never released"
        else:
            # Wait for the sibling to start before this member fails. The
            # test then covers shutdown of an in-flight fetch.
            assert member_1_started.wait(timeout=5.0), "member 1's fetch never started"
        if member_1_fetch:
            completion_order.append("member_1_fetch_returned")
        return response

    if failure_path in {"request-order-error", "declared-role-divergence"}:
        graph = FakeGraphReader(
            facets={
                (stat_var_0.removeprefix("dcid:"), recipient_0): [_facet("facet1")],
                (stat_var_1.removeprefix("dcid:"), recipient_1): [_facet("facet1")],
            }
        )
        coverage = _coverage_for(
            [
                (bare_stat_var_0, recipient_0, "facet1"),
                (bare_stat_var_1, recipient_1, "facet1"),
            ]
        )
        requests = [
            (_SPEC_ID_1, _diverging_request(stat_var=stat_var_0, recipient=recipient_0)),
            (_SPEC_ID_2, _diverging_request(stat_var=stat_var_1, recipient=recipient_1)),
        ]
    else:
        graph = FakeGraphReader(
            facets={
                (_BARE_STAT_VAR, recipient_0): [_facet("facet1")],
                (_BARE_STAT_VAR, recipient_1): [_facet("facet2")],
            }
        )
        coverage = _coverage_for(
            [
                (_BARE_STAT_VAR, recipient_0, "facet1"),
                (_BARE_STAT_VAR, recipient_1, "facet2"),
            ]
        )
        requests = [
            (_SPEC_ID_1, _request(recipient=recipient_0)),
            (_SPEC_ID_2, _request(recipient=recipient_1)),
        ]
        ceiling = 1

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])
    outcome: dict[str, BaseException | None] = {}

    def run() -> None:
        try:
            materialise_collection(
                requests,
                sdmx=sdmx,
                graph=graph,
                enrichment=EnrichmentLookup(graph),
                revision_reader=reader,
                materialiser_build="build-123",
                capability_version="cap-v1",
                member_concurrency=2,
                coverage=coverage,
                **({"ceiling": ceiling} if failure_path == "post-fetch-ceiling" else {}),
            )
        # captured across the thread boundary
        except BaseException as exc:
            outcome["exc"] = exc
        else:
            outcome["exc"] = None
        completion_order.append("materialise_collection_returned")

    thread = threading.Thread(target=run)
    thread.start()
    if failure_path == "request-order-error":
        thread.join(timeout=5.0)
    else:
        try:
            assert member_1_started.wait(timeout=5.0), "member 1's fetch never started"
            thread.join(timeout=0.5)
            assert thread.is_alive(), (
                "materialise_collection returned before member 1's in-flight fetch finished"
            )
        finally:
            release_member_1.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    if failure_path in {"request-order-error", "declared-role-divergence"}:
        assert isinstance(outcome["exc"], SdmxResponseError)
        assert bare_stat_var_0 in str(outcome["exc"])
        assert bare_stat_var_1 not in str(outcome["exc"])
    else:
        assert isinstance(outcome["exc"], OverCeilingError)
        assert outcome["exc"].code == "over_ceiling"
    if failure_path == "request-order-error":
        assert completion_order == ["materialise_collection_returned"]
    else:
        assert completion_order == ["member_1_fetch_returned", "materialise_collection_returned"]


def test_a_finished_members_rows_are_freed_before_the_next_member_finishes_generating(
    monkeypatch: pytest.MonkeyPatch,
):
    """`_MemberOutcome` stores coverage and entities, so its table can be
    collected when `_finish_member_fetch` returns. A weak reference in
    `columnar.coverage` checks this before the second fetch. Concurrency 1
    serializes the generations.
    """
    captured: list[weakref.ReferenceType] = []
    checked_while_second_member_was_fetching: list[bool] = []
    real_coverage = generation_module.columnar.coverage

    def _capturing_coverage(table, **kwargs):
        coverage = real_coverage(table, **kwargs)
        if not captured and table.num_rows:
            captured.append(weakref.ref(table))
        return coverage

    monkeypatch.setattr(generation_module.columnar, "coverage", _capturing_coverage)

    recipients = ("country/TGO", "country/KEN")
    bodies = {
        recipient: _csv_body(recipient=recipient, facet_id=f"facet-{recipient}")
        for recipient in recipients
    }

    def handler(request: httpx.Request) -> httpx.Response:
        recipient = str(request.url.params.get("c[recipient]"))
        # Probes run sequentially before fetches. KEN's handler checks table
        # collection while its fetch runs.
        if recipient == "country/KEN" and _is_data_fetch(request):
            gc.collect()
            assert captured, "the first member's own rows were never captured"
            checked_while_second_member_was_fetching.append(captured[0]() is None)
        return httpx.Response(
            200,
            text=bodies[recipient],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    graph, sdmx, coverage = _graph_and_sdmx_for(list(recipients), handler=handler)
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])

    result = materialise_collection(
        [
            (_SPEC_ID_1, _request(recipient="country/TGO")),
            (_SPEC_ID_2, _request(recipient="country/KEN")),
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        member_concurrency=1,
        coverage=coverage,
    )

    assert len(result.members) == 2
    assert checked_while_second_member_was_fetching == [True], (
        "the first member's own observation table was still alive once "
        "the second member began generating"
    )


def test_mid_stream_ceiling_breach_inside_a_shared_chunk_names_the_chunk_clause():
    """`member_concurrency=1` puts both same-recipient members in one chunk
    (`ceil(2 / 1) == 2`). The first batch crosses one member's ceiling of
    40. The member is nondeterministic. The batch has a few hundred rows,
    below the shared budget of 800. The body spans 80 read blocks, leaving
    unread chunks when the per-member ceiling fires. The error carries the
    multi-member clause.
    """
    from dc_sdmx.client import _STREAM_BATCH_BLOCK_SIZE
    from tests.dc_sdmx.conftest import StreamingTransport

    stat_var_0 = "dcid:CollectionSharedChunkMemberA"
    stat_var_1 = "dcid:CollectionSharedChunkMemberB"
    bare_0 = stat_var_0.removeprefix("dcid:")
    bare_1 = stat_var_1.removeprefix("dcid:")
    recipient = "country/KEN"
    ceiling = 5

    header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )

    def _member_1_row(period: str) -> str:
        return (
            f"dataflow,DC:DF_OBS(1.0.0),I,{bare_1},{recipient},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,{period},1000,,facet2\r\n"
        )

    member_0_row = (
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare_0},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2010,1000,,facet1\r\n"
    )
    # The body spans 80 read blocks, beyond Arrow's read-ahead window.
    n_rows = -(-80 * _STREAM_BATCH_BLOCK_SIZE // len(_member_1_row("2000")))
    body = (
        header + member_0_row + "".join(_member_1_row(str(2000 + i)) for i in range(n_rows))
    ).encode()

    stream = StreamingTransport(body=body, chunk_size=4096)
    sdmx = SdmxClient(base_url="https://datacommons.one.org/core/api", transport=stream)
    graph = FakeGraphReader(
        facets={
            (bare_0, recipient): [_facet("facet1")],
            (bare_1, recipient): [_facet("facet2")],
        }
    )
    coverage = _coverage_for([(bare_0, recipient, "facet1"), (bare_1, recipient, "facet2")])
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_collection(
            [
                (_SPEC_ID_1, _request_for(stat_var=stat_var_0, recipient=recipient)),
                (_SPEC_ID_2, _request_for(stat_var=stat_var_1, recipient=recipient)),
            ],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=reader,
            materialiser_build="build-123",
            capability_version="cap-v1",
            ceiling=ceiling,
            # Pinned to 1: `_plan_collection_chunks`'s own lane target is
            # `ceil(len(group) / member_concurrency)`, and the default (4)
            # would already keep these 2 same-scope members in their own
            # singleton chunks (`ceil(2 / 4) == 1`), which is exactly the
            # shape this test means to rule out -- see this test's own
            # docstring.
            member_concurrency=1,
            coverage=coverage,
        )

    message = str(excinfo.value)
    assert stat_var_1 in message
    assert stat_var_0 not in message
    assert (
        "fetched together with 1 other member(s) sharing this member's own "
        "entity scope, in one combined SDMX request"
    ) in message
    assert stream.chunks_yielded < stream.total_chunks


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
        (_SPEC_ID_1, _request_for(stat_var=stat_var_0, recipient=recipient)),
        (_SPEC_ID_2, _request_for(stat_var=stat_var_1, recipient=recipient)),
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

    materialise_collection(
        [
            (spec_id, _request_for(stat_var=f"dcid:{bare}", recipient=_REJECTED_CHUNK_RECIPIENT))
            for spec_id, bare in zip(
                (_SPEC_ID_1, _SPEC_ID_2), _REJECTED_CHUNK_STAT_VARS, strict=True
            )
        ],
        sdmx=SdmxClient(
            base_url="https://datacommons.one.org/core/api",
            transport=httpx.MockTransport(handler),
        ),
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
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


def test_shared_chunk_output_matches_the_per_member_path():
    """A shared two-member fetch must match two separate `materialise_spec`
    calls in each member's frame bytes and content version. Both paths use
    the same rows and revision snapshot.
    """
    stat_var_0 = "dcid:CollectionChunkEquivalenceMemberA"
    stat_var_1 = "dcid:CollectionChunkEquivalenceMemberB"
    bare_0 = stat_var_0.removeprefix("dcid:")
    bare_1 = stat_var_1.removeprefix("dcid:")
    recipient = "country/KEN"

    header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )

    def _row(*, stat_var: str, period: str, facet_id: str) -> str:
        return (
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{recipient},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,{period},1000,,{facet_id}\r\n"
        )

    combined_body = (
        header
        + _row(stat_var=bare_0, period="2015", facet_id="facet1")
        + _row(stat_var=bare_1, period="2016", facet_id="facet1")
    )
    body_0 = header + _row(stat_var=bare_0, period="2015", facet_id="facet1")
    body_1 = header + _row(stat_var=bare_1, period="2016", facet_id="facet1")

    def _combined_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=combined_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx_collection = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=httpx.MockTransport(_combined_handler),
    )
    graph_collection = FakeGraphReader(
        facets={
            (bare_0, recipient): [_facet("facet1")],
            (bare_1, recipient): [_facet("facet1")],
        }
    )
    coverage_collection = _coverage_for(
        [(bare_0, recipient, "facet1"), (bare_1, recipient, "facet1")]
    )
    collection_result = materialise_collection(
        [
            (_SPEC_ID_1, _request_for(stat_var=stat_var_0, recipient=recipient)),
            (_SPEC_ID_2, _request_for(stat_var=stat_var_1, recipient=recipient)),
        ],
        sdmx=sdmx_collection,
        graph=graph_collection,
        enrichment=EnrichmentLookup(graph_collection),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_collection,
    )

    def _per_member_handler(body: str):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text=body,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )

        return handler

    sdmx_0 = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=httpx.MockTransport(_per_member_handler(body_0)),
    )
    graph_0 = FakeGraphReader(facets={(bare_0, recipient): [_facet("facet1")]})
    coverage_0 = _coverage_for([(bare_0, recipient, "facet1")])
    member_0_alone = materialise_spec(
        _request_for(stat_var=stat_var_0, recipient=recipient),
        sdmx=sdmx_0,
        graph=graph_0,
        enrichment=EnrichmentLookup(graph_0),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_0,
    )

    sdmx_1 = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=httpx.MockTransport(_per_member_handler(body_1)),
    )
    graph_1 = FakeGraphReader(facets={(bare_1, recipient): [_facet("facet1")]})
    coverage_1 = _coverage_for([(bare_1, recipient, "facet1")])
    member_1_alone = materialise_spec(
        _request_for(stat_var=stat_var_1, recipient=recipient),
        sdmx=sdmx_1,
        graph=graph_1,
        enrichment=EnrichmentLookup(graph_1),
        revision_reader=FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage_1,
    )

    (_spec_id_0, member_0_chunked), (_spec_id_1, member_1_chunked) = collection_result.members

    assert member_0_chunked.content_version == member_0_alone.content_version
    assert member_1_chunked.content_version == member_1_alone.content_version
    assert canonical_frame_bytes(only_chunk(member_0_chunked)) == canonical_frame_bytes(
        only_chunk(member_0_alone)
    )
    assert canonical_frame_bytes(only_chunk(member_1_chunked)) == canonical_frame_bytes(
        only_chunk(member_1_alone)
    )


def test_shared_chunk_raw_guard_is_a_fixed_collection_wide_total():
    """Eight latest-window members share a recipient and one chunk because
    `member_concurrency=1` gives `dispatch_lanes(1, 8) == 1`. The single
    `max_rows` limit must equal the collection-wide budget multiplier times `ceiling`,
    regardless of the chunk's member count.
    """
    n_members = 8
    recipient = "country/TGO"
    stat_vars = [f"CollectionRawGuardMember{i}" for i in range(n_members)]
    ceiling = 1000

    header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )
    rows = "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{sv},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
        for sv in stat_vars
    )
    body = header + rows

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    captured_max_rows: list[int | None] = []
    original_stream_data_batches = sdmx.stream_data_batches

    def recording_stream_data_batches(**kwargs):
        captured_max_rows.append(kwargs.get("max_rows"))
        return original_stream_data_batches(**kwargs)

    sdmx.stream_data_batches = recording_stream_data_batches  # type: ignore[method-assign]

    graph = FakeGraphReader(facets={(sv, recipient): [_facet("facet1")] for sv in stat_vars})
    coverage = _coverage_for([(sv, recipient, "facet1") for sv in stat_vars])
    requests = [
        (f"spec_{i:032d}", _request_latest_for(stat_var=f"dcid:{sv}", recipient=recipient))
        for i, sv in enumerate(stat_vars, start=1)
    ]
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])

    result = materialise_collection(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        ceiling=ceiling,
        member_concurrency=1,
        coverage=coverage,
    )

    assert len(result.members) == n_members
    # One shared chunk (one shared SDMX request), so exactly one
    # `stream_data_batches` call captures the chunk-wide raw guard, and it
    # must equal the fixed collection-wide budget, not `n_members` times it.
    assert captured_max_rows == [generation_module._RAW_FETCH_ROWS_GUARD_MULTIPLIER * ceiling]


def test_shared_raw_budget_catches_two_concurrent_chunks_whose_combined_raw_rows_exceed_it():
    """Two 11-member latest-window groups share a raw budget with
    `member_concurrency=2`. The 22-member collection uses 3 lanes, so each
    group forms three chunks of 4, 4, and 3 members. With ceiling 22, each
    group's 11 x ceiling rows fit under the shared 20 x ceiling budget, but
    the combined 22 x ceiling rows exceed it. Each member contributes one latest row, so
    aggregate coverage equals the ceiling. The fake transport returns each
    group's full body. Chunk routing limits each chunk's contribution.
    """
    n_members_per_chunk = 11
    ceiling = 2 * n_members_per_chunk
    raw_budget_total = generation_module._RAW_FETCH_ROWS_GUARD_MULTIPLIER * ceiling
    assert n_members_per_chunk * ceiling < raw_budget_total
    assert 2 * n_members_per_chunk * ceiling > raw_budget_total

    recipients = ("country/KEN", "country/TGO")
    header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )

    def _stat_vars_for(recipient: str) -> list[str]:
        suffix = recipient.split("/")[-1]
        return [f"CollectionSharedBudgetMember{suffix}{i}" for i in range(n_members_per_chunk)]

    bodies_by_entity: dict[str, str] = {}
    facets: dict[tuple[str, str], list[Facet]] = {}
    coverage_entries: list[tuple[str, str, str]] = []
    requests: list[tuple[str, MaterialisationRequest]] = []
    spec_index = 1
    for recipient in recipients:
        stat_vars = _stat_vars_for(recipient)
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{sv},{recipient},USD,GrantEquivalent,"
            f"NotApplicable,OECD_DAC2A,{2000 + period},1000,,facet1\r\n"
            for sv in stat_vars
            for period in range(ceiling)
        )
        bodies_by_entity[recipient] = header + rows
        for sv in stat_vars:
            facets[(sv, recipient)] = [_facet("facet1")]
            coverage_entries.append((sv, recipient, "facet1"))
            requests.append(
                (
                    f"spec_{spec_index:032d}",
                    _request_latest_for(stat_var=f"dcid:{sv}", recipient=recipient),
                )
            )
            spec_index += 1

    graph = FakeGraphReader(facets=facets)
    coverage = _coverage_for(coverage_entries)
    sdmx, _requests = _sdmx_client_for(bodies_by_entity)
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_collection(
            requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=reader,
            materialiser_build="build-123",
            capability_version="cap-v1",
            ceiling=ceiling,
            member_concurrency=2,
            coverage=coverage,
        )

    assert excinfo.value.code == "over_ceiling"
    assert str(raw_budget_total) in (excinfo.value.detail or "")


def test_latest_member_raw_rows_over_ceiling_reduce_to_one_period_and_succeed():
    """The 10 raw periods exceed ceiling 3, but latest filtering leaves one
    row, so materialisation succeeds.
    """
    recipient = "country/KEN"
    ceiling = 3
    periods = tuple(str(2010 + i) for i in range(10))

    csv_body = _multi_row_csv_body(recipient=recipient, facet_id="facet1", periods=periods)
    sdmx, _requests = _sdmx_client_for({recipient: csv_body})
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, recipient): [_facet("facet1")]})
    coverage = _coverage_for([(_BARE_STAT_VAR, recipient, "facet1")])
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])

    result = materialise_collection(
        [(_SPEC_ID_1, _request_latest_for(stat_var=_STAT_VAR, recipient=recipient))],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        revision_reader=reader,
        materialiser_build="build-123",
        capability_version="cap-v1",
        ceiling=ceiling,
        coverage=coverage,
    )

    ((_spec_id, member),) = result.members
    assert member.coverage.rows == 1
    assert member.trace.applied_window["kind"] == "latest"


def test_latest_member_reduced_period_alone_exceeds_ceiling_raises_after_reduction():
    """Three recipients produce 9 raw rows across 2018-2020. Latest filtering
    leaves 3 rows, which exceeds ceiling 2 and triggers the post-reduction
    coverage check.
    """
    stat_var = "dcid:CollectionLatestReducedPeriodOverCeilingMember"
    bare = stat_var.removeprefix("dcid:")
    recipients = ("country/AAA", "country/BBB", "country/CCC")
    ceiling = 2

    header = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    )
    rows = "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{period},1000,,facet1\r\n"
        for recipient in recipients
        for period in ("2018", "2019", "2020")
    )
    body = header + rows

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    graph = FakeGraphReader(
        facets={(bare, recipient): [_facet("facet1")] for recipient in recipients}
    )
    coverage = _coverage_for([(bare, recipient, "facet1") for recipient in recipients])
    reader = FakeRevisionReader([_SNAPSHOT, _SNAPSHOT])
    request = MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": list(recipients)},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="latest"),
    )

    with pytest.raises(OverCeilingError) as excinfo:
        materialise_collection(
            [(_SPEC_ID_1, request)],
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph),
            revision_reader=reader,
            materialiser_build="build-123",
            capability_version="cap-v1",
            ceiling=ceiling,
            coverage=coverage,
        )

    assert excinfo.value.code == "over_ceiling"
    message = str(excinfo.value)
    assert (
        "materialised coverage of 3 rows exceeds the sync materialisation ceiling of 2 rows"
        in message
    )
    assert "aborted mid-stream" not in message
