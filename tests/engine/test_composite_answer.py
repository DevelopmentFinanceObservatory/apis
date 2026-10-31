"""A composite answer previews its observations from one read of each component, composed
under the composite's spec_id: streamed after the `response` event, embedded when buffered,
and read again for a table identity the part memo held."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import pytest

import dc_graph
from origin_catalogue import current_catalogue, scoped_reads
from search.engine import response_cache_flow, same_shape
from search.engine.concurrency import scoped_concurrency
from search.engine.entry import dataset_resubmit
from search.engine.errors import (
    EngineInfraError,
    EngineInputError,
    GraphDeadlineFailure,
    GraphInfraError,
)
from search.engine.events import DoneEvent, Event, ObservationsEvent, ResponseEvent
from search.engine.region import composite_answer, pipeline
from search.engine.region.codes import _ObsProbeResult
from search.engine.region.composite_answer import (
    ComponentReads,
    CompositeRelaunch,
    composite_served,
    read_composite_preview,
)
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions, ResolveState
from search.engine.response_cache_memo import PartMemo
from search.models import (
    BaseIdentity,
    CoverageExact,
    DatasetResubmitInput,
    DefinitePart,
    IdentityBindingValue,
    ObservationPoint,
    ObservationSeries,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Resolution,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
    SeriesStatVarRef,
    Spec,
    SpecObservations,
    Warning,
    omitted_observations,
)
from search.scope import composite
from search.scope.composite import ServedComponent, compose_observations
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_events, offline_resolve
from tests.scope import test_provider as provider
from tests.scope.test_basis import _graph, _PickingLLM

_WINDOW = range(2015, 2021)
# The bilateral component binds no provenance and reads the CRS family the table identity
# names. The imputed one binds its table.
_STAT_VARS = {None: "bilateral", provider._IMPUTED: "imputed"}
_BOTH = {name: [str(year) for year in _WINDOW] for name in _STAT_VARS.values()}
_PARTIAL = Warning(code="observations_scope_partial", severity="warn", message="partial")
_DEFAULTED = Warning(code="role_scope_defaulted", severity="info", message="defaulted")
_UGA = "country/UGA"
_QUESTION = "total health aid from the United States to Kenya, 2015 to 2020"


def _provenance(identity: BaseIdentity) -> str | None:
    binding = (identity.facets or {}).get("provenance")
    assert binding is None or isinstance(binding, IdentityBindingValue)
    return None if binding is None else binding.ref


class _Components:
    """`resolve_spec_resubmit` for the inclusive composite's components: each answers
    `template` with one series named for its table, holding a point in each year of `_WINDOW`
    and estimates where `estimated`, or raises its entry of `failures`, and answers degraded
    where named in `degraded`. The bilateral component warns once under a preview code and
    once under another. Each call records the source scope of the catalogue it reads
    through."""

    def __init__(self, template: Spec, *, estimated: bool) -> None:
        self.template = template
        self.estimated = estimated
        self.failures: dict[str, Exception] = {}
        self.degraded: set[str] = set()
        self.calls: list[tuple[ResubmitIdentityPayload, EngineOptions]] = []
        self.scopes: list[frozenset[str] | None] = []

    def __call__(
        self,
        *,
        inp: ResubmitIdentityPayload,
        graph: object,
        sdmx: object,
        coverage: object,
        options: EngineOptions,
        full_read_deadline: float | None = None,
    ) -> RegionResult:
        self.calls.append((inp, options))
        self.scopes.append(current_catalogue().scope)
        name = _STAT_VARS[_provenance(inp.identity)]
        if name in self.failures:
            raise self.failures[name]
        series = ObservationSeries(
            stat_var=SeriesStatVarRef(ref=name),
            entities=[],
            points=[ObservationPoint(date=str(year), value=1.0) for year in _WINDOW],
            total_obs=len(_WINDOW),
        ).stating_estimates(self.estimated)
        observations = SpecObservations(series=[series], total_obs=len(_WINDOW), truncated=False)
        spec = self.template.model_copy(
            update={
                "spec_id": f"component-{name}",
                "resolved": self.template.resolved.model_copy(
                    update={"observations": observations}
                ),
            }
        )
        warned = (_PARTIAL, _DEFAULTED) if name == "bilateral" else ()
        return RegionResult(
            variable_text="health aid",
            status="definite",
            specs=(spec,),
            no_data_reason=None,
            warnings=tuple(w.model_copy(update={"spec_id": spec.spec_id}) for w in warned),
            timing_by_step={},
            degraded=name in self.degraded,
        )


def _components(
    monkeypatch: pytest.MonkeyPatch, *, estimated: bool = False
) -> tuple[Spec, _Components]:
    """The composite a fresh search of `_QUESTION` answers, with no preview asked for, and
    its components answered by `_Components` from here on."""
    provider._cover(monkeypatch, provider._ALL_YEARS)
    first = provider._ask(_QUESTION, time="2015-2020")
    components = _Components(first, estimated=estimated)
    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", components)
    return first, components


def _fresh() -> ResolveRequest:
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="health aid",
                    entities=[
                        ParsedEntity(text="United States", role_text="from"),
                        ParsedEntity(text="Kenya", role_text="to"),
                    ],
                    time_text="2015-2020",
                )
            ],
            user_message=_QUESTION,
        ),
        response_options=ResolveOptions(include_observations=True),
    )


def _resubmit(spec: Spec) -> ResolveRequest:
    return ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitIdentityEntry(variable_text="health aid", identity=spec.definition.identity)
            ]
        ),
        response_options=ResolveOptions(include_observations=True),
    )


def _request(door: str, first: Spec) -> ResolveRequest:
    return _fresh() if door == "fresh" else _resubmit(first)


def _clients() -> dict[str, Any]:
    return {
        "graph": _graph("health aid", provider._CRS_MEMBERS),
        "llm": _PickingLLM(),
        "sdmx": FakeSdmxClient(observations={}, availability={}),
    }


def _stream(request: ResolveRequest, *, part_memo: PartMemo | None = None) -> list[Event]:
    return offline_events(request, stream_slot_options=True, part_memo=part_memo, **_clients())


def _answered(events: Sequence[Event]) -> Spec:
    (response,) = [e for e in events if isinstance(e, ResponseEvent)]
    resolution = response.response
    assert isinstance(resolution, Resolution)
    (part,) = resolution.parts
    assert isinstance(part, DefinitePart)
    return part.spec


def _observed(events: Sequence[Event]) -> ObservationsEvent:
    (event,) = [e for e in events if isinstance(e, ObservationsEvent)]
    return event


def _done(events: Sequence[Event]) -> DoneEvent:
    (done,) = [e for e in events if isinstance(e, DoneEvent)]
    return done


def _dates(observations: SpecObservations | None) -> dict[str, list[str]]:
    assert observations is not None
    return {s.stat_var.ref: [p.date for p in s.points] for s in observations.series}


@pytest.mark.parametrize("door", ["fresh", "resubmit"])
def test_a_streamed_composite_answer_previews_both_components_after_the_response(monkeypatch, door):
    """The preview streams once, after the `response` event, under the composite's spec_id,
    with each component's series in the years it serves. Each component embeds its preview
    and reads no options, and only its preview warnings reach `done`, keyed to the
    composite."""
    first, components = _components(monkeypatch)

    events = _stream(_request(door, first))

    spec = _answered(events)
    assert spec.spec_id == first.spec_id
    event = _observed(events)
    assert event.spec_id == spec.spec_id
    assert _dates(event.observations) == _BOTH
    kinds = [e.kind for e in events]
    assert kinds.index("response") < kinds.index("observations")
    assert sorted(_STAT_VARS[_provenance(inp.identity)] for inp, _ in components.calls) == [
        "bilateral",
        "imputed",
    ]
    assert {
        (o.include_observations, o.stream_observations, o.include_option_coverage)
        for _, o in components.calls
    } == {(True, False, False)}
    assert [(w.code, w.spec_id) for w in _done(events).warnings] == [(_PARTIAL.code, spec.spec_id)]


@pytest.mark.parametrize("door", ["fresh", "resubmit"])
def test_a_buffered_composite_answer_embeds_the_composed_preview(monkeypatch, door):
    first, _ = _components(monkeypatch)

    response = offline_resolve(_request(door, first), **_clients())

    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    spec = part.spec
    assert spec.spec_id == first.spec_id
    assert _dates(spec.resolved.observations) == _BOTH
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.evidence == "observed"
    keyed = [(w.code, w.spec_id) for w in part.warnings if w.code == _PARTIAL.code]
    assert keyed == [(_PARTIAL.code, spec.spec_id)]


def test_a_composite_resubmit_reads_its_table_identity_with_no_preview(monkeypatch):
    first, _ = _components(monkeypatch)
    real = dataset_resubmit.resolve_spec_resubmit
    seen: list[EngineOptions] = []

    def recording(**kwargs: Any) -> RegionResult:
        seen.append(kwargs["options"])
        return real(**kwargs)

    monkeypatch.setattr(dataset_resubmit, "resolve_spec_resubmit", recording)

    _stream(_resubmit(first))

    assert [(o.include_observations, o.stream_observations) for o in seen] == [(False, False)]


def test_a_memo_hit_on_the_table_identity_still_previews_and_its_components_hit_too(
    monkeypatch,
):
    """A repeated composite resubmit takes its table identity and both components from the
    part memo and still streams the composed preview, which the hit's closure reads."""
    first, components = _components(monkeypatch)
    # The fixture graph holds none of the table identity's members, and the memo refuses a
    # region warning `members_absent_from_graph`.
    monkeypatch.setattr(response_cache_flow, "is_region_storable", lambda *_a, **_k: True)
    memo = PartMemo()
    _stream(_resubmit(first), part_memo=memo)
    components.calls.clear()

    events = _stream(_resubmit(first), part_memo=memo)

    event = _observed(events)
    assert event.spec_id == _answered(events).spec_id
    assert _dates(event.observations) == _BOTH
    assert components.calls == []
    assert memo.stats()["hits"] == 3


def test_a_composite_preview_keeps_each_component_s_estimates_for_the_late_departures(
    monkeypatch,
):
    """Component series holding estimates reach the composed event as copies, so the part's
    late departures name the modelled values."""
    _components(monkeypatch, estimated=True)

    events = _stream(_fresh())

    (late,) = _done(events).late_departures
    assert late.spec_id == _answered(events).spec_id
    assert "modelled_values" in {d.code for d in late.departures}


def _coverage(monkeypatch: pytest.MonkeyPatch) -> CoverageSnapshot:
    """A snapshot of the coverage `_components` patches in."""
    rows, extra = provider._coverage_rows(provider._ALL_YEARS)
    return FakeCoverageService(patch_coverage(monkeypatch, rows, **extra)).snapshot()


def _reads(coverage: Any = None) -> ComponentReads:
    return ComponentReads(
        graph=cast(Any, None),
        sdmx=cast(Any, None),
        coverage=coverage,
        materialise_graph=None,
        materialise_sdmx=None,
        rate_cache=None,
        deadline=None,
        catalogue=current_catalogue(),
    )


@pytest.mark.parametrize(
    ("mode", "failure", "code"),
    [
        pytest.param(
            "streamed", GraphInfraError("down"), "observations_probe_failed", id="streamed-graph"
        ),
        pytest.param(
            "streamed",
            GraphDeadlineFailure("late"),
            "observations_timed_out",
            id="streamed-graph-deadline",
        ),
        pytest.param(
            "streamed", EngineInputError("refused"), "observations_probe_failed", id="refused"
        ),
        pytest.param(
            "buffered-fresh", GraphInfraError("down"), "observations_probe_failed", id="fresh"
        ),
        pytest.param(
            "buffered-fresh",
            dc_graph.GraphReadError("read"),
            "observations_probe_failed",
            id="fresh-upstream",
        ),
        pytest.param(
            "buffered-fresh",
            EngineInputError("refused"),
            "observations_probe_failed",
            id="fresh-refused",
        ),
        pytest.param(
            "buffered-resubmit",
            GraphInfraError("down"),
            "observations_probe_failed",
            id="resubmit",
        ),
        pytest.param(
            "buffered-resubmit", TimeoutError(), "observations_timed_out", id="resubmit-timeout"
        ),
        pytest.param(
            "buffered-fresh",
            dc_graph.GraphCapExceededError("cap", cap_kind="bytes", cap=1, observed=2),
            "observations_probe_failed",
            id="fresh-cap-exceeded",
        ),
        pytest.param(
            "buffered-resubmit",
            EngineInfraError("down"),
            "observations_probe_failed",
            id="resubmit-engine-infra",
        ),
    ],
)
def test_a_component_with_no_read_leaves_its_years_out_and_names_its_role(
    monkeypatch, mode, failure, code
):
    """A component read that fails, crosses the deadline or is refused leaves the answer
    definite, and its preview the other component's series, with one warning keyed to the
    composite."""
    first, components = _components(monkeypatch)
    components.failures["imputed"] = failure

    if mode == "streamed":
        events = _stream(_fresh())
        spec = _answered(events)
        observations = _observed(events).observations
        warnings = _done(events).warnings
    else:
        door = mode.removeprefix("buffered-")
        response = offline_resolve(_request(door, first), **_clients())
        assert isinstance(response, Resolution)
        (part,) = response.parts
        assert isinstance(part, DefinitePart)
        spec, observations, warnings = part.spec, part.spec.resolved.observations, part.warnings

    assert spec.spec_id == first.spec_id
    assert _dates(observations) == {"bilateral": _BOTH["bilateral"]}
    assert [w.spec_id for w in warnings if w.code == code] == [spec.spec_id]


@pytest.mark.parametrize("door", ["fresh", "resubmit"])
def test_a_component_read_failing_on_a_code_fault_fails_the_request(monkeypatch, door):
    first, components = _components(monkeypatch)
    components.failures["imputed"] = ValueError("a fault")

    with pytest.raises(ValueError, match="a fault"):
        offline_resolve(_request(door, first), **_clients())


@pytest.mark.parametrize("door", ["fresh", "resubmit"])
@pytest.mark.parametrize("degraded", [set(), {"imputed"}])
def test_a_degraded_component_marks_the_buffered_composite_answer_degraded(
    monkeypatch, door, degraded
):
    """A component answer standing on a read its resolve swallowed marks the composite's
    leg degraded, so no cache stores it; with no such component the leg is whole."""
    first, components = _components(monkeypatch)
    components.degraded.update(degraded)

    events = offline_events(_request(door, first), **_clients())

    assert _dates(_answered(events).resolved.observations) == _BOTH
    assert _done(events).degraded_leg is bool(degraded)


def test_a_composite_whose_every_component_read_fails_streams_its_omission(monkeypatch):
    _, components = _components(monkeypatch)
    components.failures.update(bilateral=GraphInfraError("down"), imputed=GraphInfraError("down"))

    events = _stream(_fresh())

    observations = _observed(events).observations
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == "observations_probe_failed"
    missing = [w.spec_id for w in _done(events).warnings if w.code == "observations_probe_failed"]
    assert missing == [_answered(events).spec_id] * 2


def test_a_component_reads_under_its_own_pin_whatever_scope_the_part_read_under(monkeypatch):
    """A part whose source scope leaves out the imputed table still previews it: each
    component reads through the request's catalogue, which only its own pin narrows."""
    spec, components = _components(monkeypatch)
    relaunch = CompositeRelaunch(spec=spec, reads=_reads(_coverage(monkeypatch)))

    async def run() -> _ObsProbeResult:
        async with scoped_concurrency():
            with scoped_reads(frozenset({provider._CRS})):
                task = relaunch(spec.spec_id)
            return await task

    (event,), _ = asyncio.run(run())

    assert isinstance(event, ObservationsEvent)
    assert _dates(event.observations) == _BOTH
    assert components.scopes == [None, None]


def test_the_answer_s_composite_facts_read_the_catalogue_the_preview_reads(monkeypatch):
    """The part's source scope narrows the catalogue its binders read, here to a catalogue
    in which the bilateral table holds no 2020 series, and the composite they settle is
    restated under the request's catalogue, as its preview reads: the component years the
    answer states are the years the preview filters by."""
    first, _ = _components(monkeypatch)
    kept: list[tuple[RegionResult, ResolveState]] = []
    real = pipeline._with_scope_choices

    def capture(result: RegionResult, state: ResolveState) -> RegionResult:
        kept.append((result, state))
        return real(result, state)

    monkeypatch.setattr(pipeline, "_with_scope_choices", capture)
    previewed = _dates(_observed(_stream(_fresh())).observations)
    (result, state), *_ = kept
    scopes: list[frozenset[str] | None] = []
    assemble = pipeline.assemble_composite

    def record(*args: Any, **kwargs: Any) -> Any:
        scopes.append(current_catalogue().scope)
        return assemble(*args, **kwargs)

    monkeypatch.setattr(pipeline, "assemble_composite", record)
    held = composite._component_years

    def narrowed(identity: Any, *, facts: Any) -> composite.YearCounts:
        years = held(identity, facts=facts)
        if (
            current_catalogue().scope is None
            or _provenance(BaseIdentity.model_validate(identity)) is not None
        ):
            return years
        return composite.YearCounts(counts={**years.counts, 2020: 0}, latest=years.latest)

    monkeypatch.setattr(composite, "_component_years", narrowed)

    with scoped_reads(frozenset({provider._CRS, provider._IMPUTED})):
        restated = real(result, state)

    assert restated.scope_choices is not None
    assert restated.scope_choices.compiled.spec_id == first.spec_id
    assert state.request_catalogue is not None
    assert scopes == [state.request_catalogue.scope]
    stated = {
        _STAT_VARS[_provenance(component.identity)]: sorted(
            str(year) for start, end in component.periods or () for year in range(start, end + 1)
        )
        for component in restated.scope_choices.compiled.components or ()
    }
    assert stated == previewed
    relaunch = restated.observations_relaunch
    assert isinstance(relaunch, CompositeRelaunch)
    assert relaunch.served is not None
    assert {
        _STAT_VARS[_provenance(BaseIdentity.model_validate(item.component.identity))]: sorted(
            str(year) for year in item.served
        )
        for item in relaunch.served
    } == stated


def test_a_component_serving_no_year_is_not_read_and_warns_nothing(monkeypatch):
    """The imputed component serves no year here, and its read would fail: it is not read,
    so the preview holds the bilateral series and the bilateral component's warning alone."""
    spec, components = _components(monkeypatch)
    components.failures["imputed"] = GraphInfraError("down")
    reads = _reads(_coverage(monkeypatch))
    served = [
        item
        if _provenance(BaseIdentity.model_validate(item.component.identity)) != provider._IMPUTED
        else dataclasses.replace(item, served=frozenset())
        for item in composite_served(spec.definition.identity, reads=reads)
    ]

    observations, warnings = asyncio.run(
        read_composite_preview(served=served, entity_groups=None, reads=reads, semaphore=None)
    )

    assert [_STAT_VARS[_provenance(inp.identity)] for inp, _ in components.calls] == ["bilateral"]
    assert _dates(observations) == {"bilateral": _BOTH["bilateral"]}
    assert [w.code for w in warnings] == [_PARTIAL.code]


def test_a_component_serving_no_year_leaves_the_omission_to_those_that_serve_one():
    """One component serves 2020 and found no point in it; the other serves no year and has
    no read. The composite omits its preview for want of points."""
    serving, idle = (
        ServedComponent(
            component=cast(Any, None), years=cast(Any, None), served=years, members=None
        )
        for years in (frozenset({2020}), frozenset[int]())
    )
    empty = omitted_observations("observations_no_points", "none", dispatched=1, empty=1)

    composed = compose_observations([serving, idle], [empty, None])

    assert composed.omitted is not None
    assert composed.omitted.reason == "observations_no_points"
    assert composed.omitted.dispatched == 1


def test_cancelling_the_composite_task_cancels_every_component_read(monkeypatch):
    spec, _ = _components(monkeypatch)
    served = [
        ServedComponent(
            component=cast(Any, index),
            years=cast(Any, None),
            served=frozenset({2020}),
            members=None,
        )
        for index in range(3)
    ]
    monkeypatch.setattr(composite_answer, "served_components", lambda *_a, **_k: served)
    started: list[int] = []
    cancelled: list[int] = []

    async def read(item: ServedComponent, _groups: object, **_kwargs: object) -> object:
        index = cast(int, item.component)
        started.append(index)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(index)
            raise
        raise AssertionError("a component read ended on its own")

    monkeypatch.setattr(composite_answer, "_read_component", read)

    async def run() -> tuple[BaseException | None, list[int]]:
        async with scoped_concurrency():
            task = CompositeRelaunch(spec=spec, reads=_reads())(spec.spec_id)
            while len(started) < len(served):
                await asyncio.sleep(0)
            task.cancel()
            ended: BaseException | None = None
            try:
                await task
            except asyncio.CancelledError as exc:
                ended = exc
            # Let the cancelled reads run their handlers, and read them before the loop's
            # shutdown cancels whatever still runs.
            for _ in range(3):
                await asyncio.sleep(0)
            return ended, sorted(cancelled)

    ended, cancelled_reads = asyncio.run(run())

    assert isinstance(ended, asyncio.CancelledError)
    assert cancelled_reads == [0, 1, 2]


def test_the_spliced_headline_under_latest_serves_each_component_in_its_own_periods(
    monkeypatch,
):
    """Under a `latest` window both headline components read their own latest year. The net
    component's lies past its periods (to 2017), so the composite serves the grant-equivalent
    year alone."""
    facts = provider._dac1_facts(provider._DAC1_TOTAL_ODA)
    identity = provider._dac1_identity(
        provider._DAC1_TOTAL_ODA, (provider._USA,), window=None, headline=True
    )
    windows: list[object] = []

    async def read(item: ServedComponent, _groups: object, **_kwargs: object) -> object:
        windows.append(item.component.identity["requested_window"])
        series = ObservationSeries(
            stat_var=SeriesStatVarRef(ref=item.component.role),
            entities=[],
            points=[ObservationPoint(date="2024", value=1.0)],
            total_obs=1,
        )
        return SpecObservations(series=[series], total_obs=1, truncated=False), ()

    monkeypatch.setattr(composite_answer, "_read_component", read)

    reads = _reads(facts.coverage)
    observations, _ = asyncio.run(
        read_composite_preview(
            served=composite_served(identity, reads=reads),
            entity_groups=None,
            reads=reads,
            semaphore=None,
        )
    )

    assert windows == [{"kind": "latest"}, {"kind": "latest"}]
    assert _dates(observations) == {"headline_ge": ["2024"]}


def test_a_merged_composite_previews_every_member_s_places(monkeypatch):
    """A same-shape merge rebuilds the composite's relaunch over the merged Spec, whose
    identity holds both members' recipients, so each component reads them both."""
    spec, components = _components(monkeypatch)
    identity = spec.definition.identity
    entities = {**(identity.entities or {}), "destination": [provider._KEN, _UGA]}
    merged = spec.model_copy(
        update={
            "spec_id": "merged",
            "definition": spec.definition.model_copy(
                update={"identity": identity.model_copy(update={"entities": entities})}
            ),
        }
    )
    member = RegionResult(
        variable_text="health aid",
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        observations_relaunch=CompositeRelaunch(spec=spec, reads=_reads(_coverage(monkeypatch))),
    )

    region = same_shape._finalize_merged_region(
        [member, dataclasses.replace(member, earliest_index=1)],
        merged,
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
    relaunch = region.observations_relaunch
    assert relaunch is not None

    async def run() -> _ObsProbeResult:
        async with scoped_concurrency():
            return await relaunch("merged")

    asyncio.run(run())

    assert [(inp.identity.entities or {}).get("destination") for inp, _ in components.calls] == [
        [provider._KEN, _UGA],
        [provider._KEN, _UGA],
    ]
