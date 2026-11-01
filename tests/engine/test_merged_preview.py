"""A same-shape merged answer previews as a resubmit of its identity previews it: one read of
the merged identity, streamed in one event under the merged spec_id or embedded when
buffered, in place of the previews its members read for themselves."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from typing import Any, cast

import pytest

from origin_catalogue import current_catalogue
from search.engine import same_shape
from search.engine.concurrency import create_owned_task, scoped_concurrency
from search.engine.config import settings
from search.engine.deadline import _cancel_and_await
from search.engine.enrichment import _reconcile_after_collapse, _stream_slot_options
from search.engine.entry import dataset_resubmit
from search.engine.errors import EngineInputError, GraphDeadlineFailure, GraphInfraError
from search.engine.events import DoneEvent, ObservationsEvent, ResponseEvent
from search.engine.region import composite_answer
from search.engine.region.codes import (
    DEFLATOR_PROJECTED,
    OBSERVATIONS_CLIENTS_UNAVAILABLE,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_SCOPE_PARTIAL,
    OBSERVATIONS_TIMED_OUT,
    _ObsProbeResult,
)
from search.engine.region.composite_answer import ComponentReads
from search.engine.region.result import RegionResult
from search.models import (
    DatasetResubmitInput,
    DefinitePart,
    ObservationPoint,
    ObservationSeries,
    ObservationSeriesConversion,
    ProvenanceRef,
    Resolution,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityPayload,
    SeriesEntityRef,
    SeriesFacetRef,
    SeriesStatVarRef,
    Spec,
    SpecObservations,
    Warning,
)
from tests.conftest import minimal_spec
from tests.engine import test_dataset_resubmit as resubmit
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_events

_PRIVATE = "dc/private"
_SHARED = "dc/shared"


def _series(
    place: str, source: str, years: tuple[int, ...], *, projected: int = 0
) -> ObservationSeries:
    conversion = (
        ObservationSeriesConversion(
            recipe_id="r",
            table_provenance=SeriesFacetRef(ref="dc/rates"),
            exchange_rate_table_version="v1",
            deflator_table_version=None,
            fallback_count=0,
            fallback_examples=[],
            projected_count=projected,
            projected_examples=[f"{place}@{years[-1]}"],
        )
        if projected
        else None
    )
    return ObservationSeries(
        stat_var=SeriesStatVarRef(ref="oda"),
        entities=[SeriesEntityRef(role="observationAbout", ref=place)],
        points=[ObservationPoint(date=str(year), value=1.0) for year in years],
        total_obs=len(years),
        source_refs=[ProvenanceRef(ref=source, label=source)],
        conversion=conversion,
    )


def _preview(*series: ObservationSeries) -> SpecObservations:
    return SpecObservations(
        series=list(series), total_obs=sum(each.total_obs for each in series), truncated=False
    )


# Each member elects for itself: Kenya its private source and both years, Uganda the shared
# source and 2021. The merged identity's read elects the shared source and the latest year
# at both places, so a union of the members' previews differs from it in source and period.
_KEN_OWN = _preview(_series("country/KEN", _PRIVATE, (2020, 2021)))
_UGA_OWN = _preview(_series("country/UGA", _SHARED, (2021,)))
_MERGED_READ = _preview(
    _series("country/KEN", _SHARED, (2021,), projected=1),
    _series("country/UGA", _SHARED, (2021,)),
)


class _Resubmits:
    """`resolve_spec_resubmit`: records each payload and answers it with a region of
    `status` embedding `_MERGED_READ`, `degraded` when set, and warning under a preview code
    and a resolve code, or raises `failure`."""

    def __init__(self) -> None:
        self.calls: list[ResubmitIdentityPayload] = []
        self.failure: Exception | None = None
        self.status = "definite"
        self.degraded = False

    def __call__(self, *, inp: ResubmitIdentityPayload, **_clients: object) -> RegionResult:
        self.calls.append(inp)
        if self.failure is not None:
            raise self.failure
        return RegionResult(
            variable_text=None,
            status=cast(Any, self.status),
            specs=(_spec("resubmitted", _MERGED_READ),),
            no_data_reason=None,
            warnings=(
                Warning(code=OBSERVATIONS_SCOPE_PARTIAL, severity="warn", message="partial"),
                Warning(code="role_scope_defaulted", severity="info", message="defaulted"),
            ),
            timing_by_step={},
            degraded=self.degraded,
        )


@pytest.fixture
def resubmits(monkeypatch: pytest.MonkeyPatch) -> _Resubmits:
    fake = _Resubmits()
    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", fake)
    return fake


def _reads() -> ComponentReads:
    return ComponentReads(
        graph=cast(Any, None),
        sdmx=cast(Any, None),
        coverage=cast(Any, None),
        materialise_graph=None,
        materialise_sdmx=None,
        rate_cache=None,
        deadline=None,
        catalogue=current_catalogue(),
    )


def _spec(spec_id: str, observations: SpecObservations | None = None) -> Spec:
    spec = Spec.model_validate(minimal_spec(spec_id))
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"observations": observations})}
    )


def _relaunch(observations: SpecObservations):
    """A member's own relaunch: a task sending `observations` under its spec_id."""

    def launch(spec_id: str) -> asyncio.Task[_ObsProbeResult]:
        async def events() -> _ObsProbeResult:
            return (ObservationsEvent(spec_id=spec_id, observations=observations),), ()

        return create_owned_task(events())

    return launch


def _member(name: str, index: int, own: SpecObservations, *, streamed: bool) -> RegionResult:
    """A member that streams or embeds `own`, its own preview; an embedded one comes with
    that preview's warnings keyed to the member."""
    own_warnings = (
        Warning(code=DEFLATOR_PROJECTED, severity="warn", message=name, spec_id=name),
        Warning(code=OBSERVATIONS_PROBE_FAILED, severity="warn", message=name, spec_id=name),
    )
    return RegionResult(
        variable_text=name,
        status="definite",
        specs=(_spec(name, None if streamed else own),),
        no_data_reason=None,
        warnings=() if streamed else own_warnings,
        timing_by_step={},
        earliest_index=index,
        observations_relaunch=_relaunch(own) if streamed else None,
    )


def _merged(*members: RegionResult, reads: ComponentReads | None) -> RegionResult:
    return same_shape._finalize_merged_region(
        list(members),
        _spec("merged"),
        preview_reads=reads,
        facet_probe=None,
        facet_warning=None,
        measure_partition=None,
        measure_siblings=(),
        open_axis_facts=(),
        option_axis_facts=(),
        role_axis_facts=(),
        served_filter=None,
        served_members=(),
        served_plans=(),
    )


def _preview_of(*, streamed: bool) -> tuple[SpecObservations, list[tuple[str, str | None]]]:
    """The preview of Kenya's and Uganda's merged answer and its warnings by
    `(code, spec_id)`: streamed, the event and warnings the task reconciliation launches
    returns, each member's own running read cancelled as stale; buffered, its embedded
    preview and its warnings."""
    members = [
        _member("kenya", 0, _KEN_OWN, streamed=streamed),
        _member("uganda", 1, _UGA_OWN, streamed=streamed),
    ]
    if not streamed:
        (merged,) = same_shape._read_merged_previews([_merged(*members, reads=_reads())])
        assert merged.observations_relaunch is None
        observations = merged.spec.resolved.observations
        assert observations is not None
        return observations, [(w.code, w.spec_id) for w in merged.warnings]

    async def run() -> tuple[ObservationsEvent, tuple[Warning, ...]]:
        async with scoped_concurrency():
            running = [
                dataclasses.replace(member, observations_task=launch(member.spec.spec_id))
                for member in members
                if (launch := member.observations_relaunch) is not None
            ]
            (streaming,), stale = _reconcile_after_collapse(
                running, [_merged(*running, reads=_reads())]
            )
            assert set(stale) == {member.observations_task for member in running}
            await asyncio.gather(*(_cancel_and_await(task) for task in stale))
            assert streaming.observations_task is not None
            (event,), warnings = await streaming.observations_task
            assert isinstance(event, ObservationsEvent)
            return event, warnings

    event, warnings = asyncio.run(run())
    assert event.spec_id == "merged"
    return event.observations, [(w.code, w.spec_id) for w in warnings]


@pytest.mark.parametrize("streamed", [True, False])
def test_a_merged_answer_previews_the_read_of_its_identity(resubmits, streamed):
    """The merged answer reads its identity once and shows what that read embeds: the shared
    source and the latest year at both places, where its members elected otherwise. Its
    preview warnings are that read's, keyed to it, with one `deflator_projected` over it. The
    members' own preview warnings and the read's resolve warnings stay out."""
    observations, warnings = _preview_of(streamed=streamed)

    assert observations == _MERGED_READ
    assert [payload.identity for payload in resubmits.calls] == [
        _spec("merged").definition.identity
    ]
    assert warnings == [(OBSERVATIONS_SCOPE_PARTIAL, "merged"), (DEFLATOR_PROJECTED, "merged")]


@pytest.mark.parametrize("streamed", [True, False])
@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        pytest.param(GraphInfraError("down"), "definite", OBSERVATIONS_PROBE_FAILED, id="infra"),
        pytest.param(
            GraphDeadlineFailure("late"), "definite", OBSERVATIONS_TIMED_OUT, id="deadline"
        ),
        pytest.param(TimeoutError(), "definite", OBSERVATIONS_TIMED_OUT, id="timeout"),
        pytest.param(
            EngineInputError("refused"), "definite", OBSERVATIONS_PROBE_FAILED, id="refused"
        ),
        pytest.param(None, "no_data", OBSERVATIONS_PROBE_FAILED, id="no-data"),
    ],
)
def test_a_lost_merged_read_omits_the_preview_and_warns(resubmits, streamed, failure, status, code):
    resubmits.failure = failure
    resubmits.status = status

    observations, warnings = _preview_of(streamed=streamed)

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == code
    assert warnings == [(code, "merged")]


def test_a_merged_read_failing_on_a_code_fault_raises(resubmits):
    resubmits.failure = ValueError("a fault")

    with pytest.raises(ValueError, match="a fault"):
        _preview_of(streamed=False)


@pytest.mark.parametrize("streamed", [True, False])
def test_a_merge_of_previewing_members_without_the_request_s_clients_raises(resubmits, streamed):
    members = (
        _member("kenya", 0, _KEN_OWN, streamed=streamed),
        _member("uganda", 1, _UGA_OWN, streamed=streamed),
    )

    with pytest.raises(ValueError, match="preview reads"):
        _merged(*members, reads=None)
    assert resubmits.calls == []


def test_members_previewing_nothing_merge_into_an_answer_that_reads_nothing(resubmits):
    member = RegionResult(
        variable_text="kenya",
        status="definite",
        specs=(_spec("kenya"),),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
    )

    merged = _merged(member, member, reads=None)

    assert merged.observations_relaunch is None
    assert merged.spec.resolved.observations is None
    assert resubmits.calls == []


@pytest.mark.parametrize(
    ("outcome", "degraded_leg"), [("whole", False), ("degraded", True), ("lost", True)]
)
def test_a_degraded_buffered_merged_read_marks_the_leg_degraded(monkeypatch, outcome, degraded_leg):
    """Two resubmitted identities fold on the dataset_resubmit door, buffered. Their merged
    read is a resubmit of the merged identity, through the door's clients. When that read
    stands on a read it swallowed, or is itself lost, `done` reports a degraded leg, so no
    cache stores the answer; with a whole read the leg is whole."""
    real = composite_answer.resolve_spec_resubmit
    merged_reads: list[ResubmitIdentityPayload] = []

    def merged_read(**kwargs: Any) -> RegionResult:
        merged_reads.append(kwargs["inp"])
        if outcome == "lost":
            raise GraphInfraError("down")
        return dataclasses.replace(real(**kwargs), degraded=outcome == "degraded")

    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", merged_read)
    request = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                resubmit._gender_entry("Female", resubmit._FEMALE_SV, window=None),
                resubmit._gender_entry("Male", resubmit._MALE_SV, window=None),
            ]
        ),
        response_options=ResolveOptions(include_observations=True),
    )

    events = offline_events(
        request,
        graph=resubmit._gender_fold_graph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    (response,) = [e.response for e in events if isinstance(e, ResponseEvent)]
    assert isinstance(response, Resolution)
    first, second = response.parts
    assert isinstance(first, DefinitePart) and isinstance(second, DefinitePart)
    assert first.spec.spec_id == second.spec.spec_id
    assert first.spec.resolved.observations is not None
    assert len(merged_reads) == 1
    (done,) = [e for e in events if isinstance(e, DoneEvent)]
    assert done.degraded_leg is degraded_leg


def test_buffered_merges_read_their_previews_together(monkeypatch):
    """Two merge groups' reads run at once: each read waits until both have started, which
    reads one after the other would never reach."""
    both = threading.Barrier(2, timeout=5)

    def waiting(*, inp: ResubmitIdentityPayload, **_clients: object) -> RegionResult:
        both.wait()
        return RegionResult(
            variable_text=None,
            status="definite",
            specs=(_spec("resubmitted", _MERGED_READ),),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )

    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", waiting)
    groups = [
        _merged(
            _member("kenya", 0, _KEN_OWN, streamed=False),
            _member("uganda", 1, _UGA_OWN, streamed=False),
            reads=_reads(),
        )
        for _ in range(2)
    ]

    read = same_shape._read_merged_previews(groups)

    assert [region.spec.resolved.observations for region in read] == [_MERGED_READ] * 2


@pytest.mark.parametrize("degraded", [False, True])
def test_a_streamed_merged_read_that_swallowed_a_read_leaves_the_stream_incomplete(
    resubmits, degraded
):
    """The merged answer's trailing read states the read its resubmit swallowed, and the
    stream reports the preview incomplete, so no cache stores the bundle."""
    resubmits.degraded = degraded
    incomplete: list[str] = []

    async def run() -> None:
        async with scoped_concurrency():
            members = [
                _member("kenya", 0, _KEN_OWN, streamed=True),
                _member("uganda", 1, _UGA_OWN, streamed=True),
            ]
            (streaming,), _stale = _reconcile_after_collapse(
                members, [_merged(*members, reads=_reads())]
            )
            async for _event in _stream_slot_options(
                [streaming],
                deadline=time.monotonic() + 5,
                warnings_out=[],
                late_departures_out=[],
                incomplete_out=incomplete,
                response=None,
                answered=(),
            ):
                pass

    asyncio.run(run())

    assert incomplete == (["observations"] if degraded else [])


@pytest.mark.parametrize(
    ("deadline_s", "cut"),
    [pytest.param(5.0, False, id="within"), pytest.param(0.4, True, id="past")],
)
def test_a_buffered_merged_read_runs_to_its_deadline_past_the_fail_open_budget(
    monkeypatch, resubmits, deadline_s, cut
):
    """The merged preview is the answer's own preview, so a buffered one reads to the reads'
    deadline less the combine's tail, as a single answer's does, however short the fail-open
    budget of label and type reads. A read slower than that budget embeds its preview whole.
    One still running at the deadline is cut: the preview is omitted
    `observations_timed_out`, and the region is degraded, so no cache stores it. The read
    takes 0.3 s on the real clock."""
    monkeypatch.setattr(settings(), "fail_open_read_budget_s", 0.05)

    def slow(**kwargs: Any) -> RegionResult:
        time.sleep(0.3)
        return resubmits(**kwargs)

    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", slow)
    reads = dataclasses.replace(_reads(), deadline=time.monotonic() + deadline_s)
    members = [
        _member("kenya", 0, _KEN_OWN, streamed=False),
        _member("uganda", 1, _UGA_OWN, streamed=False),
    ]

    (merged,) = same_shape._read_merged_previews([_merged(*members, reads=reads)])

    observations = merged.spec.resolved.observations
    assert observations is not None
    assert merged.degraded is cut
    if cut:
        assert observations.series == []
        assert observations.omitted is not None
        assert observations.omitted.reason == OBSERVATIONS_TIMED_OUT
        assert [(w.code, w.spec_id) for w in merged.warnings] == [
            (OBSERVATIONS_TIMED_OUT, "merged")
        ]
    else:
        assert observations == _MERGED_READ


def test_a_buffered_merged_read_after_a_full_read_runs_to_the_extended_deadline(
    monkeypatch, search_settings
):
    """Two resubmitted identities fold on the dataset_resubmit door, buffered, and their
    resubmits run the full read past the request's deadline but within the full read's own.
    The door combines under the full read's deadline, and the merged read runs under it too,
    so the answer embeds the merged read's own outcome, where a read bound by the request's
    deadline would be cut `observations_timed_out` before it starts. The resubmits outlast a 1 s request deadline on
    the real clock, which every read checks, so the test takes about a second."""
    search_settings(request_timeout_s=1.0, resubmit_full_read_timeout_s=10.0)
    real = dataset_resubmit.resolve_spec_resubmit

    def full_read(*, full_read_deadline: float, **kwargs: Any) -> RegionResult:
        region = real(full_read_deadline=full_read_deadline, **kwargs)
        request_deadline = full_read_deadline - 9.0
        time.sleep(max(0.0, request_deadline + 0.1 - time.monotonic()))
        return dataclasses.replace(region, reads_deadline=full_read_deadline)

    monkeypatch.setattr(dataset_resubmit, "resolve_spec_resubmit", full_read)
    request = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                resubmit._gender_entry("Female", resubmit._FEMALE_SV, window=None),
                resubmit._gender_entry("Male", resubmit._MALE_SV, window=None),
            ]
        ),
        response_options=ResolveOptions(include_observations=True),
    )

    events = offline_events(
        request,
        graph=resubmit._gender_fold_graph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    (response,) = [e.response for e in events if isinstance(e, ResponseEvent)]
    assert isinstance(response, Resolution)
    first, second = response.parts
    assert isinstance(first, DefinitePart) and isinstance(second, DefinitePart)
    assert first.spec.spec_id == second.spec.spec_id
    # The fixture supplies no materialise clients, so the merged read, once it runs, states
    # that it fetched none.
    observations = first.spec.resolved.observations
    assert observations is not None
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert OBSERVATIONS_TIMED_OUT not in {w.code for w in first.warnings}
