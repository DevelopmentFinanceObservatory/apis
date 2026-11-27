"""A composite answer previews its observations from one read of each component, composed
under the composite's spec_id: streamed after the `response` event, embedded when buffered,
and read again for a table identity the part memo held."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

import dc_graph
from code_index.shared import get_index
from datasets_api.adapter import composite_components
from materialise.held_runs import HeldRun
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
from search.engine.events import (
    DoneEvent,
    Event,
    ObservationsEvent,
    ResponseEvent,
)
from search.engine.region import (
    composite_answer,
    phase_answer,
    pipeline,
    place_options,
    resubmit,
)
from search.engine.region.codes import (
    MEMBERS_ABSENT_FROM_GRAPH,
    _ObsProbeResult,
)
from search.engine.region.composite_answer import (
    ComponentReads,
    CompositeRelaunch,
    composite_served,
    read_composite_preview,
)
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions, ResolveState
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_memo import PartMemo
from search.engine.unit.conversion import ConversionPeriod
from search.models import (
    BaseIdentity,
    CoverageExact,
    DatasetResubmitInput,
    DefinitePart,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityShapeSchema,
    IdentityWindowLatest,
    IdentityWindowRange,
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
from series_coverage.schema import FacetMetadata
from series_coverage.shared import CoverageSnapshot
from tests.engine._assemble_fixtures import _make_spec
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_events, offline_resolve
from tests.scope import test_basis as basis
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
    where named in `degraded`, and serves a continuation where named in `continued`. The
    bilateral component warns once under a preview code and
    once under another, and each component adds its entry of `warned`. Each call records the
    source scope of the catalogue it reads through."""

    def __init__(self, template: Spec, *, estimated: bool) -> None:
        self.template = template
        self.estimated = estimated
        # The components whose answer serves a continuation in the years it previews
        # (`ResolvedSpec.continued`).
        self.continued: set[str] = set()
        self.failures: dict[str, Exception] = {}
        self.degraded: set[str] = set()
        self.warned: dict[str, tuple[Warning, ...]] = {}
        # The components answering a status other than definite.
        self.statuses: dict[str, str] = {}
        self.calls: list[tuple[ResubmitIdentityPayload, EngineOptions]] = []
        self.scopes: list[frozenset[str] | None] = []
        # Each component's held years, `_WINDOW` where unnamed, and the latest points a
        # preview keeps of the years its window asks for and its options keep
        # (`EngineOptions.preview_years`), every point where None.
        self.held: dict[str, range] = {}
        self.limit: int | None = None

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
        window = inp.identity.requested_window
        years = [
            year
            for year in self.held.get(name, _WINDOW)
            if not isinstance(window, IdentityWindowRange)
            or (window.start_year or year) <= year <= (window.end_year or year)
        ]
        if options.preview_years is not None:
            years = [year for year in years if year in options.preview_years]
        shown = years if self.limit is None else years[-self.limit :]
        series = ObservationSeries(
            stat_var=SeriesStatVarRef(ref=name),
            entities=[],
            points=[ObservationPoint(date=str(year), value=1.0) for year in shown],
            total_obs=len(years),
        ).stating_estimates(self.estimated)
        observations = SpecObservations(
            series=[series], total_obs=len(years), truncated=len(shown) < len(years)
        )
        spec = self.template.model_copy(
            update={
                "spec_id": f"component-{name}",
                "resolved": self.template.resolved.model_copy(
                    update={"observations": observations}
                ).stating_continued(name in self.continued),
            }
        )
        warned = ((_PARTIAL, _DEFAULTED) if name == "bilateral" else ()) + self.warned.get(name, ())
        return RegionResult(
            variable_text="health aid",
            status=cast(Any, self.statuses.get(name, "definite")),
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


def _part(events: Sequence[Event]) -> DefinitePart:
    (response,) = [e for e in events if isinstance(e, ResponseEvent)]
    resolution = response.response
    assert isinstance(resolution, Resolution)
    (part,) = resolution.parts
    assert isinstance(part, DefinitePart)
    return part


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
    composite. Before the `response` event the imputed component, whose preview resolves no
    table the answer read, resolves its plan with no preview, for the runs it holds."""
    first, components = _components(monkeypatch)

    events = _stream(_request(door, first))

    spec = _answered(events)
    assert spec.spec_id == first.spec_id
    event = _observed(events)
    assert event.spec_id == spec.spec_id
    assert _dates(event.observations) == _BOTH
    kinds = [e.kind for e in events]
    assert kinds.index("response") < kinds.index("observations")
    assert sorted(
        (_STAT_VARS[_provenance(inp.identity)], o.include_observations)
        for inp, o in components.calls
    ) == [("bilateral", True), ("imputed", False), ("imputed", True)]
    assert {(o.stream_observations, o.include_option_coverage) for _, o in components.calls} == {
        (False, False)
    }
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
    # The table identity's series are one component's, so the composite states no years.
    assert spec.resolved.years_with_data is None


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
    part memo, with the imputed component's plan, and still streams the composed preview,
    which the hit's closure reads."""
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
    assert memo.stats()["hits"] == 4


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


@pytest.mark.parametrize(
    ("continued", "lost"),
    [
        pytest.param(set(), None, id="none"),
        pytest.param({"imputed"}, None, id="imputed"),
        # The imputed read is lost, and the bilateral one states no continuation.
        pytest.param(set(), "imputed", id="lost-beside-none"),
        # The imputed read is lost, and the bilateral one states a continuation.
        pytest.param({"bilateral"}, "imputed", id="lost-beside-a-continuation"),
    ],
)
@pytest.mark.parametrize("door", ["fresh", "resubmit"])
@pytest.mark.parametrize("mode", ["streamed", "buffered", "unread"])
def test_a_composite_states_modelled_values_exactly_when_a_component_read_continues(
    monkeypatch, mode, door, continued, lost
):
    """No series of either component holds an estimate, and the composite's own Spec states
    no continuation. The part names modelled values exactly when a component's answer it
    read serves a continuation in the years it previews: streamed in its late departures
    once its preview is read, buffered in its own, and, where the request reads no preview,
    in its own. There the bilateral component, whose preview resolves the table the answer
    read, states the answer's own plan, which serves none, and the imputed one is resolved
    with no preview in the years it serves. A lost read states none; a lost preview read
    warns of it."""
    first, components = _components(monkeypatch)
    components.continued = continued
    if lost is not None:
        components.failures[lost] = GraphInfraError("down")
    request = _request(door, first)

    if mode == "streamed":
        events = _stream(request)
        assert "modelled_values" not in {d.code for d in _part(events).departures}
        late = _done(events).late_departures
        departures = late[0].departures if late else _part(events).departures
        warnings = _done(events).warnings
    else:
        if mode == "unread":
            request = request.model_copy(update={"response_options": ResolveOptions()})
        response = offline_resolve(request, **_clients())
        assert isinstance(response, Resolution)
        (part,) = response.parts
        assert isinstance(part, DefinitePart)
        assert part.spec.spec_id == first.spec_id
        departures, warnings = part.departures, part.warnings

    claimed = "imputed" in continued if mode == "unread" else bool(continued)
    assert ("modelled_values" in {d.code for d in departures}) is claimed
    # One plan resolve of the imputed component on every path, none of the table the answer
    # read, and one preview resolve per component where a preview is read.
    read = sorted(
        (_STAT_VARS[_provenance(inp.identity)], options.include_observations, options.preview_years)
        for inp, options in components.calls
    )
    previewed = [] if mode == "unread" else sorted(_STAT_VARS.values())
    assert read == sorted(
        [("imputed", False, frozenset(_WINDOW))]
        + [(name, True, frozenset(_WINDOW)) for name in previewed]
    )
    preview_lost = lost is not None and mode != "unread"
    assert ("observations_probe_failed" in {w.code for w in warnings}) is preview_lost


def _narrowed_to(start: int, end: int) -> Any:
    """`phase_answer._over_conversion_period` narrowing every answer to `start` to `end`, as a
    unit request whose conversion tables cover those years alone narrows it
    (`_narrowed_to_period`)."""

    def over_period(
        state: Any, mat_result: Any, filtered: Any, admission: Any, *, settled: Any
    ) -> Any:
        period = ConversionPeriod(
            served=(_WINDOW[0], _WINDOW[-1]), years=(start, end), limited_by="exchange_rates"
        )
        mat_result, filtered, _shown = phase_answer._narrowed_to_period(
            state, mat_result, admission, period, requested=None
        )
        return mat_result, filtered

    return over_period


@pytest.mark.parametrize(
    ("narrowing", "resolved"),
    [
        # A resubmit binding US dollars leaves the table's euro facet out of its plan.
        pytest.param("unit-filter", ["bilateral", "imputed"], id="facet-filtered"),
        # The same binding over a table holding US dollars alone leaves nothing out.
        pytest.param("unit-binding", ["imputed"], id="facet-binding-leaving-nothing-out"),
        pytest.param("conversion-period", ["bilateral", "imputed"], id="conversion-period"),
    ],
)
def test_an_unread_composite_resolves_the_component_its_narrowed_plan_cannot_state(
    monkeypatch, narrowing, resolved
):
    """Asked with no preview, the bilateral component's preview resolves the table the
    answer read, so its continuation is stated from the answer's own plan. A plan the
    request's own facet filter or a conversion's period narrowed elected over inputs the
    identity's hash does not hold, so the component is resolved; a filter leaving no facet
    out lends the plan."""
    rows, extra = provider._coverage_rows(provider._ALL_YEARS)
    if narrowing == "unit-filter":
        rows += [
            (dcid, "crs-eur", provider._USA, provider._KEN, list(provider._BILATERAL_YEARS))
            for dcid in provider._CRS_MEMBERS
        ]
        extra["facet_metadata"]["crs-eur"] = FacetMetadata(provenance_id=provider._CRS, unit="EUR")
    patch_coverage(monkeypatch, rows, **extra)
    first = provider._ask(_QUESTION, time="2015-2020")
    components = _Components(first, estimated=False)
    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", components)
    if narrowing == "conversion-period":
        monkeypatch.setattr(phase_answer, "_over_conversion_period", _narrowed_to(2016, 2019))
        request = _fresh()
    else:
        document = first.definition.identity.model_dump(mode="json", exclude_none=True)
        document["facets"]["unit"] = {"state": "value", "ref": "USD"}
        identity = BaseIdentity.model_validate(document)
        request = _resubmit(
            first.model_copy(
                update={"definition": first.definition.model_copy(update={"identity": identity})}
            )
        )
    request = request.model_copy(update={"response_options": ResolveOptions()})

    response = offline_resolve(request, **_clients())

    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert sorted(_STAT_VARS[_provenance(inp.identity)] for inp, _ in components.calls) == resolved
    assert not any(options.include_observations for _, options in components.calls)


class _Spliced:
    """`resolve_spec_resubmit` for the OECD headline's components, named by the measured
    property each reads: each answers a definite Spec serving a continuation where named in
    `continued`, or raises where named `lost`, after waiting on `together` where given. Each
    call records the measured property, whether it embeds a preview and its preview years."""

    def __init__(
        self,
        *,
        continued: set[str] = frozenset(),
        lost: str | None = None,
        together: threading.Barrier | None = None,
    ) -> None:
        self.continued = continued
        self.lost = lost
        self.together = together
        self.template = _make_spec(
            sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/USA"
        )
        self.calls: list[tuple[str, bool, frozenset[int] | None]] = []

    def __call__(
        self, *, inp: ResubmitIdentityPayload, options: EngineOptions, **_: object
    ) -> RegionResult:
        shape = inp.identity.shape
        assert isinstance(shape, IdentityShapeSchema)
        measured = shape.measured_property
        self.calls.append((measured, options.include_observations, options.preview_years))
        if self.together is not None:
            self.together.wait()
        if measured == self.lost:
            raise GraphInfraError("down")
        resolved = self.template.resolved.stating_continued(measured in self.continued)
        return RegionResult(
            variable_text="aid",
            status="definite",
            specs=(self.template.model_copy(update={"resolved": resolved}),),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
        )


@pytest.mark.parametrize(
    ("continued", "lost"),
    [
        pytest.param(set(), None, id="none"),
        pytest.param({"amount"}, None, id="net"),
        pytest.param({"grantEquivalent"}, None, id="grant-equivalent"),
        # The grant-equivalent resolve is lost, and the net one serves no continuation.
        pytest.param(set(), "grantEquivalent", id="lost-beside-none"),
        # The grant-equivalent resolve is lost, and the net one serves a continuation.
        pytest.param({"amount"}, "grantEquivalent", id="lost-beside-a-continuation"),
    ],
)
def test_an_unread_spliced_composite_states_modelled_values_from_its_component_plans(
    monkeypatch, continued, lost
):
    """A donor's aid over 2015 to 2020 is the OECD headline, net flows to 2017 spliced with
    grant equivalents from 2018, and the request reads no preview. The answer read the net
    identity over the whole window, which the net component shares but over 2015 to 2017
    alone, so neither component reads the table the answer read: each is resolved once with
    no preview in the years it serves, and the part names modelled values exactly when one
    of them serves a continuation there. A lost resolve states none."""
    basis._cover(monkeypatch, basis._DAC1_MEMBERS)
    components = _Spliced(continued=continued, lost=lost)
    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", components)

    part = basis._resolve_part(_headline(), "aid")

    assert part.spec.resolved.facets["provenance"].ref == basis._HEADLINE
    assert part.spec.resolved.observations is None
    assert sorted(components.calls) == [
        ("amount", False, frozenset(range(2015, 2018))),
        ("grantEquivalent", False, frozenset(range(2018, 2021))),
    ]
    assert ("modelled_values" in {d.code for d in part.departures}) is bool(continued)


@pytest.mark.parametrize("held", [True, False])
def test_a_component_the_answer_read_states_its_own_plan_and_is_not_resolved(monkeypatch, held):
    """The answer's Spec states the continuation its own plan serves for the
    grant-equivalent component, whose preview resolves the read the answer made. Settling
    the composite's continuation resolves the net component alone, which serves none, and
    takes the grant-equivalent's from the Spec."""
    facts = provider._dac1_facts(provider._DAC1_TOTAL_ODA)
    identity = provider._dac1_identity(
        provider._DAC1_TOTAL_ODA, (provider._USA,), window=(2010, 2024), headline=True
    )
    reads = _reads(facts.coverage)
    net, grant_equivalent = composite_served(identity, reads=reads)
    components = _Spliced()
    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", components)
    resolved = components.template.resolved.stating_component_continued(
        [
            (
                composite.component_spec_id(grant_equivalent),
                grant_equivalent.preview_years,
                held,
            )
        ]
    )

    continued = asyncio.run(
        composite_answer.read_composite_continuation(
            served=[net, grant_equivalent], resolved=resolved, reads=reads, semaphore=None
        )
    )

    assert continued is held
    assert [measured for measured, _, _ in components.calls] == ["amount"]


def test_an_unread_composite_resolves_its_components_together(monkeypatch):
    """Each of the OECD headline's component resolves waits until the other's has started,
    so the request answers only when they run together."""
    basis._cover(monkeypatch, basis._DAC1_MEMBERS)
    components = _Spliced(together=threading.Barrier(2, timeout=10))
    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", components)

    basis._resolve_part(_headline(), "aid")

    assert len(components.calls) == 2


def test_an_unread_composite_s_component_resolves_take_the_request_s_continuation_permits(
    monkeypatch, search_settings
):
    """With one permit per request, the OECD headline's two component resolves run one at a
    time inside the part leg, which holds the request's one leg permit, and the request
    answers."""
    search_settings(max_variables=1, max_variable_concurrency=1)
    basis._cover(monkeypatch, basis._DAC1_MEMBERS)
    components = _Spliced()
    lock = threading.Lock()
    running = [0]
    peak = [0]

    def counted(**kwargs: Any) -> RegionResult:
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        try:
            # Long enough for the other resolve's thread to start if no permit held it.
            time.sleep(0.05)
            return components(**kwargs)
        finally:
            with lock:
                running[0] -= 1

    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", counted)

    basis._resolve_part(_headline(), "aid")

    assert len(components.calls) == 2
    assert peak[0] == 1


def test_an_unread_composite_resolves_each_component_plan_once_and_reads_no_row(monkeypatch):
    """The OECD headline over 2015 to 2020, asked with no preview, resolves each of its two
    components once with no preview, and nothing reads SDMX."""
    basis._cover(monkeypatch, basis._DAC1_MEMBERS)
    real = composite_answer.resolve_spec_resubmit
    calls: list[tuple[bool, frozenset[int] | None]] = []

    def counted(**kwargs: Any) -> RegionResult:
        options = kwargs["options"]
        calls.append((options.include_observations, options.preview_years))
        return real(**kwargs)

    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", counted)
    sdmx = FakeSdmxClient(observations={}, availability={})

    response = offline_resolve(
        _headline(),
        graph=basis._graph("aid", basis._ALL_MEMBERS),
        llm=_PickingLLM(),
        sdmx=sdmx,
    )

    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert part.spec.resolved.facets["provenance"].ref == basis._HEADLINE
    assert sorted(calls, key=lambda call: min(call[1] or ())) == [
        (False, frozenset(range(2015, 2018))),
        (False, frozenset(range(2018, 2021))),
    ]
    assert (sdmx.series_calls, sdmx.availability_calls, sdmx.subject_observations_calls) == (
        [],
        [],
        [],
    )


def _headline() -> ResolveRequest:
    """A donor's aid over 2015 to 2020, which the OECD headline serves, with no preview."""
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text="aid",
                    entities=[ParsedEntity(text="United States", role_text="from")],
                    time_text="2015-2020",
                )
            ]
        )
    )


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
@pytest.mark.parametrize(
    ("fault", "degraded", "memo_entries"),
    [
        pytest.param("lost", True, 1, id="lost"),
        pytest.param("refused", False, 1, id="refused"),
        pytest.param("no_data", False, 2, id="no_data"),
    ],
)
def test_a_continuation_resolve_that_states_nothing_still_answers_the_part(
    monkeypatch, caplog, door, fault, degraded, memo_entries
):
    """Asked with no preview, the imputed component's resolve is lost to a graph failure,
    refused, or answers no definite Spec. The part still answers the composite and states no
    continuation, and one INFO line names the component's role, except for a lost read, which
    `_resubmit` already warned. A lost read is transient,
    so the answer is degraded and no cache stores it; a refusal or a no-data answer is
    deterministic, so the answer stays whole and is cached."""
    first, components = _components(monkeypatch)
    if fault == "lost":
        components.failures["imputed"] = GraphInfraError("down")
    elif fault == "refused":
        components.failures["imputed"] = EngineInputError("refused")
    else:
        components.statuses["imputed"] = "no_data"
    components.continued.add("imputed")
    # The fixture graph holds none of the table identity's members, and the memo refuses a
    # region warning `members_absent_from_graph`.
    monkeypatch.setattr(response_cache_flow, "is_region_storable", lambda *_a, **_k: True)
    cache = ResponseCache()
    memo = PartMemo()
    request = _request(door, first).model_copy(update={"response_options": ResolveOptions()})

    with caplog.at_level(logging.INFO, logger=composite_answer.__name__):
        events = offline_events(request, response_cache=cache, part_memo=memo, **_clients())

    assert _answered(events).spec_id == first.spec_id
    assert "modelled_values" not in {d.code for d in _part(events).departures}
    assert _done(events).degraded_leg is degraded
    stated_none = [
        record.json_fields["component_role"]
        for record in caplog.records
        if record.levelname == "INFO" and "continuation_cause" in getattr(record, "json_fields", {})
    ]
    assert stated_none == ([] if fault == "lost" else ["imputed"])
    if door == "fresh":
        assert (cache.stats().entries > 0) is not degraded
    else:
        # A resubmit's part memo holds the table identity's part, stored before its
        # components resolve, so a repeat resolves them again, and the imputed component's
        # answer only where it answered: a no-data answer is stored, a lost or refused
        # resolve leaves nothing.
        assert memo.stats()["entries"] == memo_entries


@pytest.mark.parametrize("door", ["fresh", "resubmit"])
def test_a_continuation_resolve_failing_on_a_code_fault_fails_the_request(monkeypatch, door):
    """Asked with no preview, a fault in the code resolving the imputed component's
    continuation propagates, as it does on the preview path."""
    first, components = _components(monkeypatch)
    components.failures["imputed"] = ValueError("a fault")
    request = _request(door, first).model_copy(update={"response_options": ResolveOptions()})

    with pytest.raises(ValueError, match="a fault"):
        offline_resolve(request, **_clients())


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


_ABSENT = Warning(code=MEMBERS_ABSENT_FROM_GRAPH, severity="warn", message="absent")


@pytest.mark.parametrize("mode", ["streamed", "buffered", "unread"])
@pytest.mark.parametrize("fault", [None, "degraded", "members_absent", "lost"])
def test_a_component_the_cache_would_refuse_keeps_the_composite_answer_out_of_it(
    monkeypatch, mode, fault
):
    """A component answer the response cache would not store, one standing on a read its
    resolve swallowed or one warning `members_absent_from_graph`, or one whose resolve was
    lost, keeps the composite answer out of the cache too, so a repeat request reads the
    components again; a whole component lets the answer be stored and replayed. The streamed
    preview reads its components after the part's resolution has ended, and a request
    reading no preview resolves them with none. The composite's preview shows only the
    component's preview warnings."""
    _, components = _components(monkeypatch)
    if fault == "degraded":
        components.degraded.add("imputed")
    elif fault == "members_absent":
        components.warned["imputed"] = (_ABSENT,)
    elif fault == "lost":
        components.failures["imputed"] = GraphInfraError("down")
    cache = ResponseCache()
    request = _fresh()
    if mode == "unread":
        request = request.model_copy(update={"response_options": ResolveOptions()})

    def ask() -> list[Event]:
        return offline_events(
            request, stream_slot_options=mode == "streamed", response_cache=cache, **_clients()
        )

    events = ask()
    reads = len(components.calls)
    ask()

    if mode == "unread":
        assert _answered(events).resolved.observations is None
    else:
        shown = _observed(events).observations if mode == "streamed" else None
        expected = {"bilateral": _BOTH["bilateral"]} if fault == "lost" else _BOTH
        assert _dates(shown or _answered(events).resolved.observations) == expected
    assert MEMBERS_ABSENT_FROM_GRAPH not in {w.code for w in _done(events).warnings}
    assert (cache.stats().entries > 0) is (fault is None)
    assert len(components.calls) == (reads if fault is None else 2 * reads)


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
    minted under the request's catalogue (`resubmit.composite_mint`), as its preview reads:
    the component years the answer states are the years the preview filters by."""
    first, _ = _components(monkeypatch)
    kept: list[tuple[RegionResult, ResolveState]] = []
    preview = pipeline._with_composite_preview

    async def capture(result: RegionResult, state: ResolveState) -> RegionResult:
        kept.append((result, state))
        return await preview(result, state)

    monkeypatch.setattr(pipeline, "_with_composite_preview", capture)
    scopes: list[frozenset[str] | None] = []
    mint = resubmit.composite_mint

    def recording(identity: BaseIdentity, *, coverage: Any, **kwargs: Any) -> Any:
        scopes.append(current_catalogue().scope)
        return mint(identity, coverage=coverage, **kwargs)

    monkeypatch.setattr(resubmit, "composite_mint", recording)
    decide = phase_answer._decide_answer_scope

    def narrowed_decide(*args: Any, **kwargs: Any) -> Any:
        with scoped_reads(frozenset({provider._CRS, provider._IMPUTED})):
            return decide(*args, **kwargs)

    monkeypatch.setattr(phase_answer, "_decide_answer_scope", narrowed_decide)
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

    previewed = _dates(_observed(_stream(_fresh())).observations)
    (stated, state), *_ = kept

    assert stated.scope_choices is not None
    assert stated.scope_choices.compiled.spec_id == first.spec_id
    assert state.request_catalogue is not None
    assert scopes == [state.request_catalogue.scope]
    components = {
        _STAT_VARS[_provenance(component.identity)]: sorted(
            str(year) for start, end in component.periods or () for year in range(start, end + 1)
        )
        for component in stated.scope_choices.compiled.components or ()
    }
    assert components == previewed


def test_a_fresh_composite_answer_launches_none_of_its_table_s_trailing_tasks(monkeypatch):
    """A fresh answer minted as the composite reaches `_with_scope_choices` holding no slot
    options, place options or preview task of the table it read, nor a closure relaunching
    them, so none streams under the composite's spec_id. Its preview reads each component
    (`composite_answer.CompositeRelaunch`)."""
    first, _ = _components(monkeypatch)
    arrived: list[RegionResult] = []
    real = pipeline._with_scope_choices

    def capture(result: RegionResult) -> RegionResult:
        arrived.append(result)
        return real(result)

    monkeypatch.setattr(pipeline, "_with_scope_choices", capture)

    _observed(_stream(_fresh()))

    (region, *_) = arrived
    assert region.spec.spec_id == first.spec_id
    assert (region.slot_options_task, region.place_options_task, region.observations_task) == (
        None,
        None,
        None,
    )
    assert (region.slot_options_relaunch, region.place_options_relaunch) == (None, None)
    assert isinstance(region.observations_relaunch, CompositeRelaunch)


@pytest.mark.parametrize(
    "unread",
    [
        pytest.param({"served": frozenset()}, id="serves-no-year"),
        # Under `latest` a component previews the elected year alone, which it does not serve.
        pytest.param({"elected": 2030}, id="misses-the-elected-year"),
    ],
)
def test_a_component_previewing_no_year_is_not_read_and_warns_nothing(monkeypatch, unread):
    """The imputed component previews no year here, and its read would fail: it is not read,
    so the preview holds the bilateral series and the bilateral component's warning alone."""
    spec, components = _components(monkeypatch)
    components.failures["imputed"] = GraphInfraError("down")
    reads = _reads(_coverage(monkeypatch))
    served = [
        item
        if _provenance(BaseIdentity.model_validate(item.component.identity)) != provider._IMPUTED
        else dataclasses.replace(item, **unread)
        for item in composite_served(spec.definition.identity, reads=reads)
    ]

    observations, warnings = asyncio.run(
        read_composite_preview(served=served, entity_groups=None, reads=reads, semaphore=None)
    )

    assert [_STAT_VARS[_provenance(inp.identity)] for inp, _ in components.calls] == ["bilateral"]
    assert _dates(observations) == {"bilateral": _BOTH["bilateral"]}
    assert [w.code for w in warnings] == [_PARTIAL.code]


@pytest.mark.parametrize(
    ("served_years", "shown"),
    [
        pytest.param(frozenset(range(2015, 2019)), ["2017", "2018"], id="tail-cut"),
        pytest.param(frozenset({2015, 2016, 2017, 2020}), ["2017", "2020"], id="interior-gap"),
    ],
)
def test_a_preview_limit_keeps_each_component_s_latest_points_in_the_years_it_serves(
    monkeypatch, served_years, shown
):
    """The bilateral component holds years past the composite's last served year, and each
    preview keeps its latest two points. Each component reads under the window a Datasets
    mint reads it under, and its preview keeps the years it serves before the limit cuts, so
    each line shows two served points, past an unserved tail or an unserved gap alike."""
    spec, components = _components(monkeypatch)
    components.held["bilateral"] = range(2015, 2025)
    components.limit = 2
    reads = _reads(_coverage(monkeypatch))
    served = [
        dataclasses.replace(item, served=served_years)
        for item in composite_served(spec.definition.identity, reads=reads)
    ]

    observations, _ = asyncio.run(
        read_composite_preview(served=served, entity_groups=None, reads=reads, semaphore=None)
    )

    assert _dates(observations) == {"bilateral": shown, "imputed": shown}
    minted = composite_components(
        spec.definition.identity.model_dump(mode="json", exclude_none=True),
        code_index=get_index(),
    )
    assert minted is not None
    assert sorted(
        inp.identity.model_dump(mode="json", exclude_none=True)["requested_window"].items()
        for inp, _ in components.calls
    ) == sorted(
        BaseIdentity.model_validate(component.identity)
        .model_dump(mode="json", exclude_none=True)["requested_window"]
        .items()
        for component in minted
    )
    assert {o.preview_years for _, o in components.calls} == {served_years}


def test_a_stacked_composite_under_latest_previews_the_year_it_elects_on_every_line(
    monkeypatch,
):
    """Under a `latest` window the bilateral component's own latest year (2024) lies past the
    last year the imputed table covers, and so the composite serves (2020). Each component
    reads the composite's elected year alone, so both lines show it."""
    spec, components = _components(monkeypatch)
    components.held["bilateral"] = range(2015, 2025)
    rows, extra = provider._coverage_rows(range(2005, 2021))
    reads = _reads(FakeCoverageService(patch_coverage(monkeypatch, rows, **extra)).snapshot())
    latest = spec.definition.identity.model_copy(
        update={"requested_window": IdentityWindowLatest()}
    )
    served = composite_served(latest, reads=reads)
    assert [(max(item.served), item.elected) for item in served] == [(2020, 2020)] * 2

    observations, _ = asyncio.run(
        read_composite_preview(served=served, entity_groups=None, reads=reads, semaphore=None)
    )

    assert _dates(observations) == {"bilateral": ["2020"], "imputed": ["2020"]}


def _unperiodised(index: int = 0) -> Any:
    """A stand-in component numbered `index` whose recipe holds every year."""
    return SimpleNamespace(index=index, recipe=SimpleNamespace(periods=None))


def test_a_component_serving_no_year_leaves_the_omission_to_those_that_serve_one():
    """One component serves 2020 and found no point in it; the other serves no year and has
    no read. The composite omits its preview for want of points."""
    serving, idle = (
        ServedComponent(
            component=_unperiodised(), years=cast(Any, None), served=years, members=None
        )
        for years in (frozenset({2020}), frozenset[int]())
    )
    empty = omitted_observations("observations_no_points", "none", dispatched=1, empty=1)

    composed = compose_observations([serving, idle], [empty, None])

    assert composed.omitted is not None
    assert composed.omitted.reason == "observations_no_points"
    assert composed.omitted.dispatched == 1


@pytest.mark.parametrize(
    ("stated", "continued"),
    [
        pytest.param((False, False), False, id="every-read-serves-none"),
        pytest.param((True, None), True, id="a-read-continues-beside-a-lost-one"),
        pytest.param((False, None), None, id="a-read-is-lost"),
    ],
)
def test_a_composite_preview_states_a_continuation_only_as_its_reads_settle_it(stated, continued):
    """Two components serve 2020, each read stating whether it serves a continuation there,
    None for a lost read. The composed preview serves one when a read does, serves none when
    every read serves none, and states nothing when a lost read could hold one."""
    items = [
        ServedComponent(
            component=_unperiodised(index),
            years=cast(Any, None),
            served=frozenset({2020}),
            members=None,
        )
        for index in range(2)
    ]
    reads = [
        None
        if one is None
        else omitted_observations(
            "observations_no_points", "none", dispatched=1, empty=1
        ).stating_continued(one)
        for one in stated
    ]

    assert compose_observations(items, reads).continued is continued


def test_cancelling_the_composite_task_cancels_every_component_read(monkeypatch):
    spec, _ = _components(monkeypatch)
    served = [
        ServedComponent(
            component=_unperiodised(index),
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
        index = cast(Any, item.component).index
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


def test_the_spliced_headline_under_latest_reads_and_states_each_component_in_its_periods(
    monkeypatch,
):
    """Under a `latest` window the net component's identity passes unclipped and its series
    run to 2024, past its periods, which end in 2017. The composite elects 2024, the latest
    year a component serves within its periods: only the grant-equivalent component is
    read, over 2024 alone, and the net component is never requested. The Spec's components,
    coverage and headline count the net series in 2017 and before alone."""
    facts = provider._dac1_facts(provider._DAC1_TOTAL_ODA)
    identity = provider._dac1_identity(
        provider._DAC1_TOTAL_ODA, (provider._USA,), window=None, headline=True
    )
    requested: list[tuple[object, frozenset[int] | None]] = []

    async def resubmit(
        read_identity: BaseIdentity, _groups: object, **kwargs: Any
    ) -> tuple[SpecObservations, tuple[()]]:
        requested.append((read_identity.requested_window, kwargs["years"]))
        series = ObservationSeries(
            stat_var=SeriesStatVarRef(ref="headline_ge"),
            entities=[],
            points=[ObservationPoint(date="2024", value=1.0)],
            total_obs=1,
        )
        return SpecObservations(series=[series], total_obs=1, truncated=False), ()

    monkeypatch.setattr(composite_answer, "_resubmit_preview", resubmit)

    reads = _reads(facts.coverage)
    served = composite_served(identity, reads=reads)
    observations, _ = asyncio.run(
        read_composite_preview(served=served, entity_groups=None, reads=reads, semaphore=None)
    )

    assert requested == [(IdentityWindowRange(start_year=2024, end_year=2024), frozenset({2024}))]
    assert _dates(observations) == {"headline_ge": ["2024"]}
    net, grant_equivalent = served
    assert max(net.served) == 2017 and min(grant_equivalent.served) == 2018
    mint = composite.CompositeMint(
        identity=identity, spec_id="spec", served=tuple(served), facts=facts
    )
    periods = {component.role: component.periods for component in mint.components()}
    assert periods == {
        net.component.role: [(2005, 2017)],
        grant_equivalent.component.role: [(2018, 2024)],
    }
    counted = sum(
        item.years.counts.get(year, 0)
        for item, first, last in ((net, 2005, 2017), (grant_equivalent, 2018, 2024))
        for year in range(first, last + 1)
    )
    assert mint.coverage().observation_count == counted
    headline = mint.headline(None)
    assert headline is not None
    assert (headline.year, headline.observations) == (2024, grant_equivalent.years.counts[2024])


_PURPOSES = ("CRSPurposeCode120", "CRSPurposeCode130")


def _purpose(identity: BaseIdentity, *codes: str) -> BaseIdentity:
    """`identity` binding the CRS purposes `codes`, a value for one and a set for more."""
    binding = (
        IdentityBindingValue(ref=codes[0])
        if len(codes) == 1
        else IdentityBindingSet(refs=list(codes))
    )
    slots = [
        slot.model_copy(update={"binding": binding}) if slot.property == "crsPurpose" else slot
        for slot in identity.slots
    ]
    return identity.model_copy(update={"slots": slots})


def _entry(identity: BaseIdentity) -> ResubmitIdentityEntry:
    """An entry posting `identity` with no scope entries, so its scope compiles."""
    return ResubmitIdentityEntry(variable_text="health aid", identity=identity, scope_entries=[])


def _entries(*identities: BaseIdentity) -> ResolveRequest:
    return ResolveRequest(
        input=DatasetResubmitInput(entries=[_entry(identity) for identity in identities]),
        response_options=ResolveOptions(include_observations=True),
    )


def _composite_facts(spec: Spec) -> tuple[object, ...]:
    """What a composite answer states of the composite: its spec_id, identity, components,
    years, counts and download facts."""
    resolved = spec.resolved
    scope = resolved.scope_choices
    assert scope is not None
    return (
        spec.spec_id,
        spec.definition.identity,
        scope.compiled.spec_id,
        scope.compiled.components,
        resolved.facets["provenance"],
        resolved.data_years,
        resolved.coverage,
        resolved.headline,
        resolved.stat_vars,
        resolved.download,
    )


def _answer(request: ResolveRequest, mode: str) -> tuple[list[Spec], SpecObservations | None]:
    """The Specs `request` answers in `mode`, and the first one's preview: its streamed
    event, once the stream completed, or the one it embeds."""
    if mode == "streamed":
        events = _stream(request)
        assert _done(events).timed_out is False
        resolution = next(e.response for e in events if isinstance(e, ResponseEvent))
        specs = [part.spec for part in _answered_parts(resolution)]
        event = _observed(events)
        assert event.spec_id == specs[0].spec_id
        return specs, event.observations
    specs = [part.spec for part in _answered_parts(offline_resolve(request, **_clients()))]
    return specs, specs[0].resolved.observations


_PARTIAL_2020 = HeldRun(
    kind="partial", source="provenance/OECDCreditorReportingSystem", start_year=2020, end_year=2020
)


@pytest.mark.parametrize("runs", [(), (_PARTIAL_2020,)], ids=["no-runs", "partial-run"])
@pytest.mark.parametrize("mode", ["streamed", "buffered"])
def test_two_merged_composites_answer_the_composite_over_both_members(monkeypatch, mode, runs):
    """Two composite entries the same-shape merge folds, one per purpose, answer the composite
    over both purposes as one entry posting it does, the held-back runs its components'
    plans settle included, and its preview holds every component in every year each serves,
    each component read over both purposes."""
    first, components = _components(monkeypatch)
    settled = CompositeRelaunch.plans

    async def plans(self, *, semaphore):
        found = await settled(self, semaphore=semaphore)
        return dataclasses.replace(found, runs=runs)

    monkeypatch.setattr(CompositeRelaunch, "plans", plans)
    identity = first.definition.identity

    specs, observations = _answer(_entries(*(_purpose(identity, code) for code in _PURPOSES)), mode)
    read = {
        (_STAT_VARS[_provenance(inp.identity)], _purposes(inp.identity))
        for inp, _ in components.calls
    }
    (expected,), _ = _answer(_entries(_purpose(identity, *_PURPOSES)), mode)

    assert [_composite_facts(spec) for spec in specs] == [_composite_facts(expected)] * 2
    assert [
        (h.kind, h.served, h.source, h.start_year, h.end_year)
        for h in expected.resolved.coverage.held_back or ()
    ] == [(run.kind, False, run.source, run.start_year, run.end_year) for run in runs]
    assert [spec.resolved.stat_vars.sums_components for spec in specs] == [True, True]
    assert _dates(observations) == _BOTH
    assert {("bilateral", _PURPOSES), ("imputed", _PURPOSES)} <= read


def _purposes(identity: BaseIdentity) -> tuple[str, ...]:
    (slot,) = (slot for slot in identity.slots if slot.property == "crsPurpose")
    binding = slot.binding
    if isinstance(binding, IdentityBindingValue):
        return (binding.ref,)
    assert isinstance(binding, IdentityBindingSet)
    return tuple(binding.refs)


def _answered_parts(resolution: object) -> list[DefinitePart]:
    assert isinstance(resolution, Resolution)
    parts = [part for part in resolution.parts if isinstance(part, DefinitePart)]
    assert len(parts) == len(resolution.parts)
    return parts


def test_a_composite_entry_and_a_single_table_entry_stay_apart(monkeypatch):
    """A composite answer and a single-table answer the merge would otherwise fold keep their
    own Specs: the composite previews its components, and the table answer claims no
    composite."""
    first, components = _components(monkeypatch)
    identity = first.definition.identity
    table = identity.model_copy(update={"facets": None})

    events = _stream(_entries(_purpose(identity, _PURPOSES[0]), _purpose(table, _PURPOSES[1])))

    resolution = next(e.response for e in events if isinstance(e, ResponseEvent))
    composite_part, table_part = _answered_parts(resolution)
    assert _provenance(composite_part.spec.definition.identity) == provider._INCLUSIVE_SOURCE
    assert _provenance(table_part.spec.definition.identity) is None
    assert composite_part.spec.spec_id != table_part.spec.spec_id
    previews = {e.spec_id: e.observations for e in events if isinstance(e, ObservationsEvent)}
    assert _dates(previews[composite_part.spec.spec_id]) == _BOTH
    assert {_purposes(inp.identity) for inp, _ in components.calls} == {_PURPOSES[:1]}


@pytest.mark.parametrize("merged", [False, True], ids=["one-entry", "merged"])
@pytest.mark.parametrize("mode", ["streamed", "buffered"])
def test_a_composite_entry_whose_walk_ran_the_full_read_previews_to_its_deadline(
    monkeypatch, mode, merged
):
    """A composite entry whose member walk ran the full read, past the request deadline,
    reads its components to the full read's deadline, so its preview holds both and warns
    no timeout. A component read bound by the request deadline finds it passed. A merge
    whose second member alone ran the full read reads to that member's deadline."""
    first, components = _components(monkeypatch)
    identity = first.definition.identity
    real = dataset_resubmit.resolve_spec_resubmit
    request_deadlines: set[float | None] = set()

    def full_read(**kwargs: Any) -> RegionResult:
        request_deadlines.add(kwargs["options"].deadline)
        region = real(**kwargs)
        if _purposes(kwargs["inp"].identity) == _PURPOSES[:1] and merged:
            return region
        return dataclasses.replace(region, reads_deadline=kwargs["full_read_deadline"])

    def component(**kwargs: Any) -> RegionResult:
        (request_deadline,) = request_deadlines
        assert request_deadline is not None
        if kwargs["options"].deadline <= request_deadline:
            raise TimeoutError
        return components(**kwargs)

    monkeypatch.setattr(dataset_resubmit, "resolve_spec_resubmit", full_read)
    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", component)
    request = (
        _entries(*(_purpose(identity, code) for code in _PURPOSES)) if merged else _resubmit(first)
    )

    if mode == "streamed":
        events = _stream(request)
        observations = _observed(events).observations
        warnings = _done(events).warnings
    else:
        part, *_ = _answered_parts(offline_resolve(request, **_clients()))
        observations, warnings = part.spec.resolved.observations, part.warnings

    assert _dates(observations) == _BOTH
    assert "observations_timed_out" not in {w.code for w in warnings}


@pytest.mark.parametrize("streamed", [True, False])
def test_two_merged_composites_preview_both_members_places(monkeypatch, streamed):
    """A same-shape merge of two composite answers, one per recipient, previews as a resubmit
    of the merged identity previews it: each component is read once, over both recipients,
    under the merged spec_id, streamed or embedded."""
    spec, components = _components(monkeypatch)
    identity = spec.definition.identity
    reads = _reads(_coverage(monkeypatch))

    def with_destinations(source: Spec, spec_id: str, *destinations: str) -> Spec:
        entities = {**(identity.entities or {}), "destination": list(destinations)}
        return source.model_copy(
            update={
                "spec_id": spec_id,
                "definition": source.definition.model_copy(
                    update={"identity": identity.model_copy(update={"entities": entities})}
                ),
            }
        )

    def member(destination: str, index: int) -> RegionResult:
        own = with_destinations(spec, destination, destination)
        if not streamed:
            embedded = SpecObservations(series=[], total_obs=0, truncated=False)
            own = own.model_copy(
                update={"resolved": own.resolved.model_copy(update={"observations": embedded})}
            )
        return RegionResult(
            variable_text="health aid",
            status="definite",
            specs=(own,),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
            earliest_index=index,
            observations_relaunch=CompositeRelaunch(spec=own, reads=reads) if streamed else None,
        )

    region = same_shape._finalize_merged_region(
        [member(provider._KEN, 0), member(_UGA, 1)],
        with_destinations(spec, "merged", provider._KEN, _UGA),
        preview_reads=reads,
        preview_read=None,
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
    if streamed:
        relaunch = region.observations_relaunch
        assert isinstance(relaunch, CompositeRelaunch)

        async def run() -> _ObsProbeResult:
            async with scoped_concurrency():
                return await relaunch("merged")

        (event,), _ = asyncio.run(run())
        assert isinstance(event, ObservationsEvent)
        assert event.spec_id == "merged"
        observations = event.observations
    else:
        (region,) = same_shape._read_merged_previews([region])
        assert region.observations_relaunch is None
        observations = region.spec.resolved.observations

    assert sorted(_STAT_VARS[_provenance(inp.identity)] for inp, _ in components.calls) == [
        "bilateral",
        "imputed",
    ]
    assert [(inp.identity.entities or {}).get("destination") for inp, _ in components.calls] == [
        [provider._KEN, _UGA]
    ] * 2
    assert _dates(observations) == _BOTH


def test_a_cache_hit_launches_no_place_options_for_a_composite_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    """A response cache hit launches the place options a fresh run launches: one task per
    definite table answer, none for a composite answer, whose place options a fresh run
    launches none of."""
    composite_part = provider._posted_dac2a(monkeypatch, provider._DAC2A_BILATERAL, provider._KEN)
    table_part = provider._resubmitted_part(
        provider._dac2a_identity("Gross", destination=provider._DEVELOPING), None
    )
    assert provider._source(composite_part.spec) == provider._DAC2A_BILATERAL
    assert provider._source(table_part.spec) is None
    launched: list[str] = []

    def recorded(spec: Spec, **_kwargs: Any) -> Any:
        launched.append(spec.spec_id)
        return spec.spec_id

    monkeypatch.setattr(place_options, "place_options_task", recorded)
    launch = place_options.replayed_place_options(
        stream=True,
        graph=cast(Any, None),
        sdmx=cast(Any, None),
        materialise_graph=cast(Any, object()),
        materialise_sdmx=None,
        pool=cast(Any, object()),
        coverage=cast(Any, None),
        population_memo=None,
    )
    assert launch is not None

    launch(cast(Any, SimpleNamespace(kind="resolution", parts=[composite_part, table_part])))

    assert launched == [table_part.spec.spec_id]


def test_a_merge_of_composite_parts_with_no_axis_states_the_composite_scope(
    monkeypatch: pytest.MonkeyPatch,
):
    """A DAC2a ranking over every donor has no axis, and its single part states a scope with
    no choice whose compiled identity is its bilateral destinations. A same-shape merge of
    such parts states that scope too, as the field's description says, never none."""
    facts = provider._dac2a_facts(provider._ALL_YEARS)
    identity = provider._dac2a_identity("Net", donor=None, destination=None)
    single = provider._fresh_scope(identity, facts)
    assert single is not None and single.choices == []
    assert provider._provenance(single.compiled.identity) == provider._DAC2A_BILATERAL
    monkeypatch.setattr(pipeline, "build_part_scope", lambda **_: single)
    monkeypatch.setattr(pipeline, "scope_facts", lambda *_, **__: facts)
    spec = SimpleNamespace(definition=SimpleNamespace(identity=identity))
    merged = cast(RegionResult, SimpleNamespace(spec=spec, operations=()))
    group = [cast(RegionResult, SimpleNamespace(scope_choices=single))] * 2

    assert pipeline.merged_scope_choices(merged, group, coverage=facts.coverage) == single
