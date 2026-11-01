"""Each read an answer swallows notes the lost read (`degraded_reads.note_degraded_read`), so
the region is marked degraded and no response cache stores it. One row per site, each driven
to the failure it swallows. A site bounded by `degraded_reads.fail_open_reads` swallows every
way a read is lost under its budget, and its reads, in whichever thread they run, see the
budget as the request deadline."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pydantic
import pytest

import search.engine.place as place
from dc_graph import GraphCapExceededError, GraphDeadlineExceededError
from dfo_platform.executor import submit_with_context
from dfo_platform.request_deadline import bounded_deadline, request_deadline
from dfo_platform.single_flight_cache import WaitTimeout
from origin_catalogue.catalogue import Catalogue
from origin_catalogue.families import FamilyKey
from search.engine import degraded_reads
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.config import settings
from search.engine.degraded_reads import (
    DegradedReads,
    ReadBudget,
    carry,
    fail_open_reads,
    recording_degraded_reads,
)
from search.engine.errors import (
    EngineInfraError,
    GraphDeadlineFailure,
    GraphInfraError,
    LLMInfraError,
    RoleSignatureError,
)
from search.engine.region import convertibility, grounding, related_read
from search.engine.region import phase_place_materialise as materialise
from search.engine.region import state as region_state
from search.engine.region.availability import read_coverage
from search.engine.region.codes import OBSERVATIONS_TIMED_OUT
from search.engine.region.phase_rank_bind import _stray_requirements
from search.engine.region.related_read import PendingRelated
from search.engine.region.state import ResolveState
from search.engine.retrieval import bound_values, derive
from search.engine.retrieval.read_plan import PairPlan
from search.engine.unit import refs
from search.models import EntityRoleKey, Spec, Warning, omitted_observations
from series_coverage.index import CoverageIndex
from tests.engine import _fail_open_deadline
from tests.engine._fail_open_deadline import FAIL_OPEN_REQUESTS, assert_fail_open_deadline

_LOST_READS = {
    "graph": GraphInfraError("the read failed"),
    "deadline": GraphDeadlineExceededError("the read ran past its deadline"),
    "timeout": TimeoutError("the await ran past its deadline"),
    "wait": WaitTimeout("the wait on a shared flight ran past its deadline"),
    "engine": EngineInfraError("the client's infrastructure failed"),
    "llm": LLMInfraError("the model's transport failed"),
    "cap": GraphCapExceededError(
        "the response ran past its cap", cap_kind="bytes", cap=1, observed=2
    ),
}
"""Each way a read is lost under a fail-open budget, by id."""


def _validation_error() -> pydantic.ValidationError:
    try:
        pydantic.TypeAdapter(int).validate_python("not a number")
    except pydantic.ValidationError as exc:
        return exc
    raise AssertionError("the value validated")


def _raising(error: Exception) -> Callable[..., object]:
    def read(*_args, **_kwargs):
        raise error

    return read


def _raise_graph_failure(*_args, **_kwargs):
    raise GraphInfraError("the read failed")


def _availability_probe(error: Exception) -> None:
    probe = derive._probe_place_availability(
        {"Count_Person": {}},
        place_dcids=["country/KEN"],
        full_facets_max_candidates=None,
        graph=SimpleNamespace(observation_facets_batch=_raising(error)),
        deadline=None,
        coverage=None,
    )
    assert probe is None


def _menu_availability(error: Exception) -> None:
    coverage = read_coverage(
        ["Count_Person"],
        graph=cast(EngineGraphClient, SimpleNamespace(observation_facets_batch=_raising(error))),
        places=["country/KEN"],
        deadline=None,
        read="menu availability",
    )
    assert coverage is None


def _currency_units() -> None:
    refs._graph_currency(
        SimpleNamespace(currency_units=_raise_graph_failure), "Kenya_Shilling", deadline=None
    )


def _relaxation_ctx(**changes: object) -> SimpleNamespace:
    roles = SimpleNamespace(resolved_entities={}, origin_dcids=[], roles=[], open_axes=frozenset())
    return SimpleNamespace(
        **{
            "graph": None,
            "sdmx": None,
            "shape_draft": SimpleNamespace(sv_arc_facts={"Count_Person": {}}),
            "place": SimpleNamespace(
                roles=roles,
                default_scope="countries",
                probe_places=["country/KEN"],
                presence_for=lambda presence: presence,
            ),
            "unsettled": None,
            "presence": SimpleNamespace(covers=lambda _stat_var: False),
            "date_request": None,
            "deadline": None,
            "role_class_walks": object(),
            **changes,
        }
    )


def _no_role_keys(*_args, **_kwargs) -> SimpleNamespace:
    return SimpleNamespace(bound=None, open=None)


def _relaxation_walk(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(grounding, "shape_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grounding, "entity_role_keys", _no_role_keys)
    monkeypatch.setattr(grounding, "elected_role_class_shape", _raising(error))
    assert grounding._walked_relaxations(_relaxation_ctx(), [[]]) == [None]


def _relaxations_presence(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(region_state, "presence_stat_vars", lambda *_args: ["Count_Person"])
    monkeypatch.setattr(grounding, "entity_role_keys", _no_role_keys)
    monkeypatch.setattr(place, "read_default_presence", _raising(error))
    ctx = _relaxation_ctx()
    assert grounding._relaxations_presence(ctx, [(ctx.shape_draft, [])]) is ctx.presence


def _relaxed_suggestion(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(grounding, "is_role_bearing", lambda _shape: False)
    monkeypatch.setattr(grounding, "standard_resolve", _raising(error))
    ctx = _relaxation_ctx()
    suggestion = grounding._relaxed_binding_suggestion(ctx, ctx.shape_draft, [], None)
    assert suggestion == (None, None, None, None)


def _stray_state(label_dcids: Callable[..., object]) -> SimpleNamespace:
    return SimpleNamespace(
        capped_shapes=[],
        label_dcids=label_dcids,
        admitted_qualifiers=(),
        query_qualifiers=[],
        variable="population",
        query_text=None,
        dropped_qualifier_phrases=(),
    )


_STRAY = SlotBindingDraft(
    axis="how", property_dcid="gender", kind="value", value_dcids=["dc/g/SexFemale"]
)


def _stray_value_labels(error: Exception) -> None:
    """A stray value no menu shape or code index labels, named in words from its dcid when
    the graph's label read fails, so no dcid reaches the prose that discloses it."""
    state = _stray_state(_raising(error))
    (requirement,) = asyncio.run(_stray_requirements(cast(ResolveState, state), [_STRAY]))
    assert requirement.term.text == "sex female"
    assert requirement.departure is not None
    assert requirement.departure[1].asked == "sex female"


_SCOPED_SITES: dict[str, tuple[Callable[[pytest.MonkeyPatch, Exception], None], str]] = {
    "availability": (lambda _mp, error: _availability_probe(error), "place-availability probe"),
    "menu-availability": (lambda _mp, error: _menu_availability(error), "menu availability"),
    "walk": (_relaxation_walk, "nearest-real relaxation walk"),
    "presence": (_relaxations_presence, "nearest-real presence"),
    "suggestion": (_relaxed_suggestion, "nearest-real suggestion"),
    "stray-labels": (lambda _mp, error: _stray_value_labels(error), "stray value labels"),
}


@pytest.mark.parametrize(
    ("swallow", "read", "error"),
    [
        pytest.param(swallow, read, error, id=f"{site}-{lost}")
        for site, (swallow, read) in _SCOPED_SITES.items()
        for lost, error in _LOST_READS.items()
    ]
    + [
        pytest.param(
            lambda _mp, _error: _currency_units(),
            "currency units",
            _LOST_READS["graph"],
            id="currency-graph",
        )
    ],
)
def test_a_swallowed_read_is_noted(
    swallow: Callable[[pytest.MonkeyPatch, Exception], None],
    read: str,
    error: Exception,
    monkeypatch,
) -> None:
    """Each site answers with its fallback, which the row asserts, and notes the read."""
    with recording_degraded_reads() as lost:
        swallow(monkeypatch, error)
    assert lost.reads == [read]


@pytest.mark.parametrize(
    ("site", "about"),
    [
        ("availability", {"candidates": 1, "place_dcids": ["country/KEN"]}),
        ("menu-availability", {"stat_vars": 1, "places": 1}),
        ("stray-labels", {"unlabelled": ["dc/g/SexFemale"]}),
    ],
)
def test_a_lost_read_is_logged_with_what_it_covered(
    site: str, about: dict[str, object], monkeypatch, caplog
) -> None:
    """The warning's arguments hold the identifiers the read was to cover, so a degraded
    region can be traced to the variables, places or values it lost."""
    swallow, read = _SCOPED_SITES[site]
    with caplog.at_level(logging.WARNING, logger="search.engine.degraded_reads"):
        swallow(monkeypatch, _LOST_READS["graph"])
    (record,) = [r for r in caplog.records if r.name == "search.engine.degraded_reads"]
    assert record.args is not None
    assert read in record.args
    assert about in record.args


@pytest.mark.parametrize("error", list(_LOST_READS.values()), ids=list(_LOST_READS))
def test_a_lost_read_keeps_its_traceback_unless_it_ran_past_a_deadline(
    error: Exception, caplog
) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="search.engine.degraded_reads"),
        fail_open_reads("the read", budget_s=2.0),
    ):
        raise error
    (record,) = caplog.records
    ran_past_deadline = isinstance(error, TimeoutError | GraphDeadlineExceededError)
    assert (not record.exc_info) is ran_past_deadline


def test_a_refused_relaxation_walk_notes_nothing(monkeypatch) -> None:
    """A walk the mint refuses is deterministic, so the answer without it can be stored."""

    def _refused(*_args, **_kwargs):
        raise RoleSignatureError("role sets the election cannot order")

    monkeypatch.setattr(grounding, "shape_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grounding, "entity_role_keys", _no_role_keys)
    monkeypatch.setattr(grounding, "elected_role_class_shape", _refused)
    with recording_degraded_reads() as lost:
        assert grounding._walked_relaxations(_relaxation_ctx(), [[]]) == [None]
    assert lost.reads == []


class _FacetGraph:
    """Records the deadline each facet read it serves is bound by, and answers none."""

    def __init__(self) -> None:
        self.deadlines: list[float | None] = []

    def observation_facets_batch(self, svs, places, _needs_dates, *, deadline=None):
        self.deadlines.append(bounded_deadline(deadline))
        return {sv: {place: [] for place in places} for sv in svs}


@FAIL_OPEN_REQUESTS
def test_the_place_availability_probe_reads_under_the_fail_open_budget(request_in_s) -> None:
    graph = _FacetGraph()
    requested = _fail_open_deadline.request_deadline(request_in_s)
    started = time.monotonic()

    derive._probe_place_availability(
        {"Count_Person": {}},
        place_dcids=["country/KEN"],
        full_facets_max_candidates=None,
        graph=cast(EngineGraphClient, graph),
        deadline=requested,
        coverage=None,
    )

    (deadline,) = graph.deadlines
    assert_fail_open_deadline(deadline, requested=requested, started=started)


def test_a_probe_read_past_its_deadline_is_a_lost_read_and_no_request_timeout() -> None:
    """The client's deadline error is an `Exception` the probe catches. Its class sits outside
    `TimeoutError`, which the request-level classifiers read as the request's 504."""
    assert not issubclass(GraphDeadlineFailure, TimeoutError)

    def expired(*_args, **_kwargs):
        raise GraphDeadlineFailure("the read ran past its deadline")

    with recording_degraded_reads() as lost:
        read = derive._graph_facets_at(
            ["Count_Person"],
            ["country/KEN"],
            full_facets_max_candidates=None,
            graph=cast(EngineGraphClient, SimpleNamespace(observation_facets_batch=expired)),
            deadline=None,
        )

    assert read is None
    assert lost.reads == ["place-availability probe"]


# ---------------------------------------------------------------------------
# The scope itself


def _within_budget(bound: float | None, *, started: float, budget_s: float) -> bool:
    """`bound` is `budget_s` from a moment between `started` and now."""
    return bound is not None and started + budget_s <= bound <= time.monotonic() + budget_s


def test_a_scope_bounds_its_reads_by_its_budget_and_never_widens_the_request_deadline() -> None:
    """Under a far request deadline the budget binds; under a near one, and inside a scope
    already bound, the nearer bound stands. The bound ends with the block."""
    started = time.monotonic()
    with request_deadline(started + 60), fail_open_reads("read", budget_s=2.0) as bound:
        assert bounded_deadline(None) == bound
        assert _within_budget(bound, started=started, budget_s=2.0)
        with fail_open_reads("inner read", budget_s=5.0) as inner:
            assert inner == bound
    near = time.monotonic() + 0.5
    with request_deadline(near), fail_open_reads("read", budget_s=2.0) as bound:
        assert bound == near
        assert bounded_deadline(None) == near
    assert bounded_deadline(None) is None


@pytest.mark.parametrize("error", list(_LOST_READS.values()), ids=list(_LOST_READS))
def test_a_scope_swallows_a_lost_read_and_notes_it(error: Exception) -> None:
    with recording_degraded_reads() as lost, fail_open_reads("the read", budget_s=2.0):
        raise error
    assert lost.reads == ["the read"]


@pytest.mark.parametrize(
    "error",
    [ValueError("a malformed value"), TypeError("a wrong argument"), _validation_error()],
    ids=["value", "type", "validation"],
)
def test_a_scope_lets_a_fault_in_the_code_through_unnoted(error: Exception) -> None:
    with (
        recording_degraded_reads() as lost,
        pytest.raises(type(error)),
        fail_open_reads("the read", budget_s=2.0),
    ):
        raise error
    assert lost.reads == []


@pytest.mark.parametrize(
    "error",
    [_LOST_READS["deadline"], _LOST_READS["timeout"], _LOST_READS["wait"]],
    ids=["deadline", "timeout", "wait"],
)
def test_a_scope_whose_own_budget_ran_out_notes_the_read_cut(error: Exception) -> None:
    """A read that is healthy but slower than the budget leaves an answer that is not
    degraded, so the user is told nothing, and that no cache stores. The scope's deadline, a
    budget earlier than the request's, has passed when it raises."""
    with (
        request_deadline(time.monotonic() + 60),
        recording_degraded_reads() as lost,
        fail_open_reads("the read", budget_s=0.0),
    ):
        raise error
    assert (lost.reads, lost.cut) == ([], ["the read"])
    assert not lost.degraded
    assert not lost.storable


def test_a_scope_inside_a_spent_scope_is_cut_by_a_budget_too() -> None:
    """The inner scope's deadline is the outer scope's budget, not the request's."""
    with (
        request_deadline(time.monotonic() + 60),
        recording_degraded_reads() as lost,
        fail_open_reads("outer read", budget_s=0.0),
        fail_open_reads("inner read", budget_s=2.0),
    ):
        raise TimeoutError("the read ran past the outer budget")
    assert (lost.reads, lost.cut) == ([], ["inner read"])


def _cut(read: str) -> None:
    with fail_open_reads(read, budget_s=0.0):
        raise TimeoutError(f"{read} ran past its budget")


def test_a_note_made_after_its_record_closed_reaches_the_record_around_it() -> None:
    """A trailing preview starts inside its region's record and reads after the region's
    resolution closed it, so its notes pass on as they are made."""
    with recording_degraded_reads() as request:
        with recording_degraded_reads() as region:
            started = contextvars.copy_context()
        started.run(_cut, "the trailing read")
    assert region.cut == request.cut == ["the trailing read"]


def test_a_detached_record_keeps_its_cut_reads_and_carry_hands_them_on() -> None:
    """A cut read follows a lost read's rule: a detached record keeps it, and its owner
    hands it on whether or not the outcome compared the reading."""
    passed_over = DegradedReads()
    with recording_degraded_reads() as request:
        with recording_degraded_reads(detached=True):
            _cut("an off-answer read")
        with recording_degraded_reads(into=passed_over):
            _cut("a reading's read")
        assert request.cut == []
        with recording_degraded_reads() as region:
            carry(passed_over, consulted=False)
    assert region.cut == request.cut == ["a reading's read"]


@pytest.mark.parametrize(
    ("request_in_s", "budget_s", "error"),
    [
        pytest.param(0.0, 2.0, _LOST_READS["deadline"], id="request-deadline"),
        pytest.param(None, 2.0, _LOST_READS["timeout"], id="within-budget"),
        pytest.param(60.0, 0.0, _LOST_READS["graph"], id="failure-past-budget"),
    ],
)
def test_a_scope_notes_a_read_lost_any_other_way(
    request_in_s: float | None, budget_s: float, error: Exception
) -> None:
    """A deadline at the request's own bound, a deadline error raised before the budget
    passed, and a failure, even once the budget has passed, each note the read."""
    requested = _fail_open_deadline.request_deadline(request_in_s)
    with (
        request_deadline(requested) if requested is not None else nullcontext(),
        recording_degraded_reads() as lost,
        fail_open_reads("the read", budget_s=budget_s),
    ):
        raise error
    assert (lost.reads, lost.cut) == (["the read"], [])


def test_a_wait_on_another_request_s_held_family_decode_ends_at_the_scope_s_bound(
    monkeypatch,
) -> None:
    """A reader joining a held family's decode another request runs waits until the ambient
    deadline a fail-open scope binds at most, and the scope drops it once its budget ran
    out."""
    key = (
        "build",
        "origin",
        FamilyKey("Person", "count", "measuredValue", None, None, False, False),
    )
    decode: Future[object] = Future()
    monkeypatch.setitem(bound_values._held_flights, key, decode)
    # The decode ends after 2 s, so a wait the scope does not bound still returns.
    ends = threading.Timer(2.0, decode.set_result, args=(None,))
    ends.start()
    started = time.monotonic()

    try:
        with (
            request_deadline(started + 60),
            recording_degraded_reads() as lost,
            fail_open_reads("held family", budget_s=0.05),
        ):
            bound_values._shared_held_members(key, cast(Catalogue, None))
        waited = time.monotonic() - started
    finally:
        ends.cancel()

    assert waited < 1.0
    assert lost.reads == []


def _availability_probe_past_its_deadline() -> None:
    covered = SimpleNamespace(role_names={"X": SimpleNamespace(donor="d", recipient="r")})
    probe = derive._probe_place_availability(
        {"X": {}},
        place_dcids=["country/KEN"],
        full_facets_max_candidates=None,
        graph=cast(EngineGraphClient, None),
        deadline=time.monotonic() - 1,
        coverage=cast(CoverageIndex, covered),
    )
    assert probe is None


def _dated_plan(monkeypatch: pytest.MonkeyPatch, *, deadline: float | None) -> None:
    monkeypatch.setattr(convertibility, "dated_without_complete_dates", lambda _group: True)
    served = SimpleNamespace(
        role_tuple=(("observationAbout", "country/KEN"),), group=None, stat_var="X"
    )
    reader = SimpleNamespace(
        observation_facets_batch=_raising(GraphDeadlineFailure("the dated read timed out"))
    )
    plan = convertibility._dated_plan(
        cast(PairPlan, SimpleNamespace(served=[served])),
        cast(convertibility.DatedFacetReader, reader),
        deadline=deadline,
    )
    assert plan is None


def _related_join_timed_out(budget: ReadBudget) -> None:
    """A resubmit door's related read still running when its budget's deadline passes, so
    the join itself times out (`related_read.joined_related`)."""
    pending = PendingRelated(stat_var="X", future=Future(), budget=budget)
    assert related_read.joined_related(pending) is None


_BUDGETED_SITES: dict[str, tuple[Callable[[pytest.MonkeyPatch, ReadBudget], None], str]] = {
    "availability-past-deadline": (
        lambda _mp, _budget: _availability_probe_past_its_deadline(),
        "place-availability probe",
    ),
    "dated-read-past-deadline": (
        lambda mp, _budget: _dated_plan(mp, deadline=time.monotonic() - 1),
        "unit coverage dates",
    ),
    "dated-read-timed-out": (
        lambda mp, _budget: _dated_plan(mp, deadline=None),
        "unit coverage dates",
    ),
    "related-join-timeout": (
        lambda _mp, budget: _related_join_timed_out(budget),
        "related measures",
    ),
}


@pytest.mark.parametrize("site", list(_BUDGETED_SITES))
def test_a_site_whose_budget_ran_out_notes_the_read_cut(site: str, monkeypatch) -> None:
    """Each site that notes a read lost to a deadline notes it through
    `degraded_reads.note_lost_read`, so a read cut by its budget, a fail-open scope's or the
    related read's own, with the request's deadline still ahead, leaves an answer that is
    not degraded and that no cache stores."""
    swallow, read = _BUDGETED_SITES[site]
    now = time.monotonic()
    spent = ReadBudget(deadline=now, request_deadline=now + 60)
    with (
        request_deadline(now + 60),
        recording_degraded_reads() as lost,
        fail_open_reads("the enclosing read", budget_s=0.0),
    ):
        swallow(monkeypatch, spent)
    assert (lost.reads, lost.cut) == ([], [read])


class _RelatedClock:
    """The clock `degraded_reads` weighs a related read's budget by. It starts at the real
    time and passes the budget's deadline as the read raises when the budget cut the read,
    else only once the answer joins the read."""

    def __init__(self, *, cut_by_budget: bool) -> None:
        self.now = time.monotonic()
        self.cut_by_budget = cut_by_budget

    def monotonic(self) -> float:
        return self.now

    def read(self, **_kwargs: object) -> object:
        if self.cut_by_budget:
            self._pass_the_budget()
        raise TimeoutError("the related read timed out")

    def joined(self) -> None:
        if not self.cut_by_budget:
            self._pass_the_budget()

    def _pass_the_budget(self) -> None:
        self.now += settings().related_measures_timeout_s + 1


_SPEC = cast(Spec, "the spec")
_PLACES: list[tuple[str, EntityRoleKey]] = [("country/KEN", "observationAbout")]


def _related_task(clock: _RelatedClock) -> None:
    async def joined() -> Spec:
        state = SimpleNamespace(deadline=None, date_request=None, graph=None, related_reads=[])
        read = related_read.start_related_read(
            cast(ResolveState, state),
            stat_vars=["X"],
            measure_partition=None,
            places=_PLACES,
            arcs=None,
        )
        assert read is not None
        await asyncio.wait([read.task])
        clock.joined()
        return await related_read.with_related(_SPEC, read)

    assert asyncio.run(joined()) is _SPEC


def _related_thread(clock: _RelatedClock) -> None:
    pending = related_read.submit_related_read(
        stat_vars=["X"],
        measure_partition=None,
        places=_PLACES,
        arcs=None,
        window=None,
        graph=cast(EngineGraphClient, None),
        request_deadline=None,
    )
    clock.joined()
    assert related_read.joined_related(pending) is None


@pytest.mark.parametrize("door", [_related_task, _related_thread], ids=["task", "thread"])
@pytest.mark.parametrize("cut_by_budget", [False, True], ids=["early", "cut"])
def test_a_related_read_is_classed_by_when_it_ended_not_when_it_was_joined(
    door: Callable[[_RelatedClock], None], cut_by_budget: bool, monkeypatch
) -> None:
    """A related read that failed on its own deadline error before its budget ran out is a
    lost read, however late the answer joins it, and one that ended once its budget ran out,
    with no request deadline, is cut: the budget is weighed as the read ends
    (`related_read.LostRelated`)."""
    clock = _RelatedClock(cut_by_budget=cut_by_budget)
    monkeypatch.setattr(degraded_reads, "time", clock)
    monkeypatch.setattr(related_read, "read_related", clock.read)

    with recording_degraded_reads() as lost:
        door(clock)

    noted = ["related measures"]
    assert (lost.reads, lost.cut) == (([], noted) if cut_by_budget else (noted, []))


# ---------------------------------------------------------------------------
# Where each new site enters the scope, relative to the threads its reads run in


def test_the_stray_value_label_read_sees_the_fail_open_budget_in_its_thread() -> None:
    """The label read takes no deadline and runs in a worker thread, so the budget reaches it
    only as the ambient request deadline the scope binds before the thread starts."""
    seen: list[float | None] = []

    def label_dcids(dcids: list[str]) -> list[str]:
        seen.append(bounded_deadline(None))
        return ["female"] * len(dcids)

    started = time.monotonic()
    (requirement,) = asyncio.run(
        _stray_requirements(cast(ResolveState, _stray_state(label_dcids)), [_STRAY])
    )

    assert requirement.term.text == "female"
    (bound,) = seen
    assert _within_budget(bound, started=started, budget_s=settings().fail_open_read_budget_s)


class _NearestRealRecorder:
    """Stands in for the nearest-real probe and the previews of its suggestions. Each records
    the deadline its reads would be bound by, in the probe's thread, in a pool thread the
    probe starts as a relaxation walk does, and in each preview's thread."""

    def __init__(
        self,
        *,
        preview_error: Exception | None = None,
        preview: tuple[object, Warning | None] | None = None,
    ) -> None:
        self.probe: list[float | None] = []
        self.walk: list[float | None] = []
        self.previews: list[float | None] = []
        self.attached: list[object] = []
        self._preview_error = preview_error
        self._preview = preview

    def suggest(self, **_kwargs):
        self.probe.append(bounded_deadline(None))
        with ThreadPoolExecutor(max_workers=1) as pool:
            submit_with_context(pool, lambda: self.walk.append(bounded_deadline(None))).result()
        spec = SimpleNamespace(spec_id="near", resolved=SimpleNamespace(members=[]))
        place_ctx = SimpleNamespace(
            probe_places=["country/KEN"],
            roles=SimpleNamespace(obs_destination_dcid=None, origin_dcids=[]),
        )
        read = SimpleNamespace(place=place_ctx, plan=None, member_facts=None)
        return [spec], [], {"near": read}, None

    def attach(self, spec, result, _warnings):
        self.attached.append(result)
        return spec

    def build_observations(self, *_args, **_kwargs):
        self.previews.append(bounded_deadline(None))
        if self._preview_error is not None:
            raise self._preview_error
        return self._preview or (SimpleNamespace(omitted=None), None)


def _nearest_real(monkeypatch: pytest.MonkeyPatch, recorder: _NearestRealRecorder):
    monkeypatch.setattr(materialise, "_suggest_nearest_real", recorder.suggest)
    monkeypatch.setattr(materialise, "_build_observations", recorder.build_observations)
    monkeypatch.setattr(materialise, "request_named_groups", lambda *_args: ())
    monkeypatch.setattr(materialise, "stamp_offered_place_warnings", lambda *_a, **_k: None)
    monkeypatch.setattr(
        materialise,
        "_observation_entities",
        lambda *_args, **_kwargs: ({"observationAbout": ["country/KEN"]}, None),
    )
    monkeypatch.setattr(materialise, "_attach_observations", recorder.attach)
    state = SimpleNamespace(
        graph=None,
        sdmx=None,
        shape_draft=None,
        bindings=[],
        place=SimpleNamespace(roles=None),
        default_presence=None,
        date_request=None,
        deadline=None,
        include_option_coverage=False,
        materialise_graph=None,
        materialise_sdmx=None,
        role_class_walks=None,
        entity_label_text_by_dcid={},
        text_recognized_properties=frozenset(),
        grouped_properties=frozenset(),
        measure_elected_properties=frozenset(),
        defaulted_properties=frozenset(),
        coverage=None,
        unsettled_place=None,
        pin_served_provenance=False,
        include_observations=True,
        stream_observations=False,
        rate_cache=None,
        warnings=[],
        departure_facts={},
    )
    return materialise._nearest_real_for(cast(ResolveState, state))


def test_the_nearest_real_probe_and_each_preview_read_under_their_own_budget(
    monkeypatch,
) -> None:
    """The probe runs in a worker thread and walks in pool threads, all under the nearest-real
    budget taken as the probe starts. A buffered request's previews run after it, in threads
    of their own, under the fail-open budget taken as each starts."""
    recorder = _NearestRealRecorder()
    started = time.monotonic()

    with recording_degraded_reads() as lost:
        near = asyncio.run(_nearest_real(monkeypatch, recorder))

    assert near is not None and [spec.spec_id for spec in near] == ["near"]
    assert lost.reads == []
    (bound,) = recorder.probe
    assert _within_budget(bound, started=started, budget_s=settings().nearest_real_budget_s)
    assert recorder.walk == [bound]
    (preview_bound,) = recorder.previews
    assert _within_budget(
        preview_bound, started=started, budget_s=settings().fail_open_read_budget_s
    )


def _timed_out_preview() -> tuple[object, Warning]:
    message = "the observation fetch did not finish before the request deadline"
    return (
        omitted_observations(OBSERVATIONS_TIMED_OUT, message),
        Warning(code=OBSERVATIONS_TIMED_OUT, severity="warn", message=message),
    )


@pytest.mark.parametrize(
    ("budget_s", "noted", "cut"),
    [
        pytest.param(2.0, ["nearest-real preview"], [], id="within-budget"),
        pytest.param(0.0, [], ["nearest-real preview"], id="budget-ran-out"),
    ],
)
@pytest.mark.parametrize("lost", ["raised", "omitted"])
def test_a_timed_out_preview_is_dropped_with_its_warning(
    lost: str, budget_s: float, noted: list[str], cut: list[str], monkeypatch
) -> None:
    """A preview read that raises past a deadline, or that `_build_observations` omits as
    `observations_timed_out`, leaves its suggestion without a preview or warning. It is noted
    lost, or cut when the preview's own budget ran out first, which leaves an answer that is
    not degraded and that no cache stores."""
    monkeypatch.setattr(settings(), "fail_open_read_budget_s", budget_s)
    recorder = (
        _NearestRealRecorder(preview_error=TimeoutError("the preview ran out"))
        if lost == "raised"
        else _NearestRealRecorder(preview=_timed_out_preview())
    )

    with recording_degraded_reads() as record:
        near = asyncio.run(_nearest_real(monkeypatch, recorder))

    assert near is not None and [spec.spec_id for spec in near] == ["near"]
    assert recorder.attached == []
    assert (record.reads, record.cut) == (noted, cut)
