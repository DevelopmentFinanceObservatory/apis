"""The fixture world of the answer-invariant grid (`test_answer_invariants`).

One row set is a table of `(place, facet, years)` holdings. A role route serves it as debt
flows from one creditor to the places, through the coverage index, the bulk SDMX double and
the materialise doubles; the standard route serves it as an amount at the places, through
the graph's observation facets and the same materialise doubles. Every double is built from
the row set alone, so a check can compare what an answer states with what the world holds.

Each door runs under a recorder that logs every request a double serves, with a logical
tick: a request ends one tick after the control flow that made it last waited, and a flow
that waits for work it handed to a thread or a task resumes at the latest tick that work
reached. The largest tick of an answer is its serial depth, with no wall-clock time. The
recorder also logs how many requests preceded each phase of the door.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import threading
import weakref
from collections import Counter
from collections.abc import Coroutine, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform import executor as platform_executor
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import GraphReadError, SubjectPage
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.conjoin import assemble_region
from search.engine.errors import EngineInputError
from search.engine.events import SentenceEvent
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    grounding,
    phase_answer,
    phase_place_materialise,
    pipeline,
    promote_ground,
    resolve_spec_resubmit,
    resubmit_promote,
)
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.models import BaseIdentity, ResubmitIdentityPayload, Spec, TimeWindow
from search.models.response import CandidatesPart, DefinitePart, NoDataPart, Part
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from statvar_members import shared_enumerator
from tests.engine._binders import elected_reply
from tests.engine._facet_option_builds import wrap_facet_option_builds
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient, scope_availability_of
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    door_roles_for_test_query,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import conversion_arcs, recipe_table_bodies
from tests.materialise.conftest import FakeGraphReader

ALAND, BLAND, CREDITOR = "country/AAA", "country/BBB", "country/CRA"
PLACE_NAMES = {ALAND: "Aland", BLAND: "Bland"}
IDS, ALT, PROJ = "dc/base/Ids", "dc/base/Alt", "dc/base/Proj"
SOURCE_LABELS = {IDS: "IDS", ALT: "Alternative", PROJ: "Projections"}
USD, EUR = "CurrencyUSD", "CurrencyEUR"
_RECIPE = RECIPE_BODIES["ConversionRecipeDAC"]
_RATES, _RATE_TABLE = _RECIPE.exchange_rate_variable, _RECIPE.table_provenance
_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
# The world's variables are financial flows, a population type the DAC recipe converts. The
# code index holds real flow families measured as `amount`, so the world's variables carry a
# measured property of their own and resolve to themselves alone.
_AMOUNT = "answerInvariantsAmount"
PURPOSE, _PURPOSE_ASKED, _PURPOSE_HELD = (
    "answerInvariantsPurpose",
    "AnswerInvariantsPurposeAsked",
    "AnswerInvariantsPurposeHeld",
)

# Each facet's provenance and unit. `f-none` carries no provenance, and `f-eur` is the
# alternative source reporting euros.
FACETS: dict[str, tuple[str | None, str]] = {
    "f-ids": (IDS, USD),
    "f-alt": (ALT, USD),
    "f-eur": (ALT, EUR),
    "f-proj": (PROJ, USD),
    "f-none": (None, USD),
}


def _years(*spans: tuple[int, int]) -> tuple[int, ...]:
    return tuple(year for first, last in spans for year in range(first, last + 1))


type Holdings = Mapping[str, tuple[tuple[str, tuple[int, ...]], ...]]

# The pinned clock reads 2026. In `overlap` the long series holds more points over its life
# and the short one more from 2019 on; in `gap` the long series' span covers years it holds
# no point in.
ROW_SETS: dict[str, Holdings] = {
    "overlap": {
        ALAND: (("f-ids", _years((2000, 2021))), ("f-alt", _years((2019, 2023)))),
        BLAND: (("f-ids", _years((2000, 2021))),),
    },
    "projection": {
        ALAND: (("f-ids", _years((2010, 2020))), ("f-proj", _years((2021, 2030)))),
        BLAND: (("f-ids", _years((2010, 2020))), ("f-proj", _years((2021, 2030)))),
    },
    "unsourced": {
        ALAND: (("f-none", _years((2010, 2020))), ("f-alt", _years((2016, 2018)))),
        BLAND: (("f-none", _years((2010, 2020))),),
    },
    "other_unit": {
        ALAND: (("f-ids", _years((2010, 2020))), ("f-eur", _years((2012, 2023)))),
        BLAND: (("f-ids", _years((2010, 2020))),),
    },
    "gap": {
        ALAND: (("f-ids", _years((2010, 2015), (2027, 2030))), ("f-alt", _years((2021, 2022)))),
        BLAND: (("f-ids", _years((2010, 2015), (2027, 2030))),),
    },
    "one_empty": {ALAND: (("f-ids", _years((2010, 2020))),), BLAND: ()},
}

# The row set whose worlds' member walk fails, so every door of their cells answers from the
# members it holds and warns `membership_unverified`.
UNSETTLED_ROW_SET = "one_empty"

# `dateless` is a query naming no date.
DATES: dict[str, DateRequest | None] = {
    "dateless": None,
    "latest": DateRequest(window=None, latest=True),
    "year": DateRequest(window=TimeWindow(start_year=2018, end_year=2018), latest=False),
    "clip_start": DateRequest(window=TimeWindow(start_year=1995, end_year=2012), latest=False),
    "clip_end": DateRequest(window=TimeWindow(start_year=2019, end_year=2024), latest=False),
    "since": DateRequest(window=TimeWindow(start_year=2016, end_year=None), latest=False),
    "future": DateRequest(window=TimeWindow(start_year=2027, end_year=2029), latest=False),
    "unreached": DateRequest(window=TimeWindow(start_year=1950, end_year=1955), latest=False),
}

ROUTES = ("standard", "named_pair", "named_bulk", "discovered_pair", "discovered_bulk")
KINDS = ("definite", "candidates", "collapse", "nearest_real")
UNIT_TEXT = {"none": None, "native": "USD", "converting": "EUR"}


# ---------------------------------------------------------------------------
# The recorder and its logical clock.
# ---------------------------------------------------------------------------

_CLOCK: contextvars.ContextVar[int] = contextvars.ContextVar("answer_invariants_clock", default=0)
# Set while a nearest-real relaxation's member walk runs (`log_relaxation_walks`).
_RELAXATION_WALK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "answer_invariants_relaxation_walk", default=False
)
# Set while an alternate's or a re-election sibling's member walk runs
# (`log_alternate_walks`).
_ALTERNATE_WALK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "answer_invariants_alternate_walk", default=False
)
# Set while a Spec's facet options are built (`log_facet_options`).
_FACET_OPTIONS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "answer_invariants_facet_options", default=False
)
RATE_TABLE_READ = " of the rate table"
# A candidate cap above the two candidates a world holds, so a part keeps every Spec.
_MAX_CANDIDATES = 10
# The reads a client answers with no wire call (`dc_graph.GraphClient`): `un_availability`
# hands back the UN deployment's availability client, whose reads are the wire calls.
_NO_WIRE = frozenset(
    {
        "default_facet_policy",
        "generation_stamp",
        "observation_read_groups",
        "scope_availability",
        "source_key",
        "un_availability",
        "un_observation_reader",
    }
)


@dataclass
class Recorder:
    """What one answer did: each request as `(client, method, tick)`, each sentence event and
    each phase as the number of requests logged before it, and each unit decision the engine
    made. The requests a nearest-real relaxation's member walk made are also logged apart,
    with each walk's first and last tick, and so are the requests the member walks of the
    alternates and re-election siblings made, and so are the requests each Spec's facet
    options made (`log_facet_options`), with each build's first and last tick."""

    requests: list[tuple[str, str, int]] = field(default_factory=list)
    sentences: list[int] = field(default_factory=list)
    phases: list[tuple[str, int]] = field(default_factory=list)
    unit_decisions: list[Mapping[str, tuple[Mapping[str, Any], ...]]] = field(default_factory=list)
    relaxation_requests: list[str] = field(default_factory=list)
    relaxation_walks: list[tuple[int, int]] = field(default_factory=list)
    alternate_requests: list[str] = field(default_factory=list)
    facet_requests: list[str] = field(default_factory=list)
    facet_builds: list[tuple[int, int]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request(self, client: str, method: str) -> None:
        tick = _CLOCK.get() + 1
        _CLOCK.set(tick)
        with self._lock:
            self.requests.append((client, method, tick))
            if _RELAXATION_WALK.get():
                self.relaxation_requests.append(f"{client}.{method}")
            if _ALTERNATE_WALK.get():
                self.alternate_requests.append(f"{client}.{method}")
            if _FACET_OPTIONS.get():
                self.facet_requests.append(f"{client}.{method}")

    def relaxation_walk(self) -> dict[str, Any] | None:
        """The requests the relaxations' member walks made, and the serial steps of the
        deepest walk; None when no relaxation walked."""
        if not self.relaxation_walks:
            return None
        return {
            "requests": dict(sorted(Counter(self.relaxation_requests).items())),
            "depth": max(end - start for start, end in self.relaxation_walks),
        }

    def alternate_walk(self) -> dict[str, Any] | None:
        """The requests the member walks of the alternates and re-election siblings made;
        None when none walked."""
        if not self.alternate_requests:
            return None
        return {"requests": dict(sorted(Counter(self.alternate_requests).items()))}

    def facet_options(self) -> dict[str, Any] | None:
        """The requests the facet-option builds made, and the serial steps of the deepest
        build; None when no build made a request."""
        if not self.facet_requests:
            return None
        return {
            "requests": dict(sorted(Counter(self.facet_requests).items())),
            "depth": max(end - start for start, end in self.facet_builds),
        }

    def counts(self) -> Counter[str]:
        """Requests by client and method. Reads of the rate table count once: previews that
        convert in parallel each load it when they reach the request's rate cache together,
        so their number varies from run to run."""
        counts = Counter(f"{client}.{method}" for client, method, _tick in self.requests)
        for request in counts:
            if request.endswith(RATE_TABLE_READ):
                counts[request] = 1
        return counts

    def depth(self) -> int:
        return max((tick for _client, _method, tick in self.requests), default=0)

    def requests_from(self, phase: str) -> list[str]:
        """The requests logged once `phase` began, none when it never ran."""
        began = next((n for name, n in self.phases if name == phase), len(self.requests))
        return [f"{client}.{method}" for client, method, _tick in self.requests[began:]]


_ACTIVE: list[Recorder] = []


def _record(client: str, method: str) -> None:
    if _ACTIVE:
        _ACTIVE[-1].request(client, method)


_INSIDE = threading.local()


class _Recording:
    """A double that logs each public wire read made from outside it under `_client`. A read
    the double serves through another of its own methods counts once."""

    _client: str

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        if name.startswith("_") or name in _NO_WIRE or not callable(attr):
            return attr
        client = super().__getattribute__("_client")

        def call(*args, **kwargs):
            if getattr(_INSIDE, client, 0) == 0:
                rates = _RATE_TABLE in str(args) + str(kwargs)
                _record(client, name + RATE_TABLE_READ if rates else name)
            setattr(_INSIDE, client, getattr(_INSIDE, client, 0) + 1)
            try:
                return attr(*args, **kwargs)
            finally:
                setattr(_INSIDE, client, getattr(_INSIDE, client) - 1)

        return call


class _Graph(_Recording, FakeGraph):
    _client = "graph"


class _Sdmx(_Recording, FakeSdmxClient):
    _client = "sdmx"


class _MaterialiseGraph(_Recording, FakeGraphReader):
    """A materialise graph whose reverse query keeps the subjects its filters match, as the
    graph's does."""

    _client = "materialise_graph"

    def node_subjects(
        self, dcid: str, *, filters: Mapping[str, str] | None = None, **kwargs
    ) -> SubjectPage:
        page = super().node_subjects(dcid, filters=filters, **kwargs)
        node_arcs = super().node_arcs_many(page.dcids, properties=tuple(filters or ()))
        return SubjectPage(
            dcids=tuple(
                subject
                for subject in page.dcids
                if all(
                    value in (node_arcs[subject] or {}).get(prop, ())
                    for prop, value in (filters or {}).items()
                )
            ),
            truncated=page.truncated,
        )


class _UnwalkableGraph(_MaterialiseGraph):
    """A materialise graph whose member walk fails: the reverse query that starts it raises,
    as a graph read past its bound or a failed one does."""

    def node_subjects(self, dcid: str, **kwargs) -> SubjectPage:
        raise GraphReadError(f"the reverse query on {dcid!r} failed")


def _join(end: int) -> None:
    if end > _CLOCK.get():
        _CLOCK.set(end)


class _ClockTask(asyncio.Task):
    """A task whose awaiter resumes at the tick the task reached."""

    def __await__(self):
        try:
            return (yield from super().__await__())
        finally:
            if self.done():
                _join(self.get_context().get(_CLOCK, 0))

    __iter__ = __await__


def carry_logical_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Carry the recorder's tick across each hand-off the engine makes: a pool thread and a
    task start at their submitter's tick, and whoever waits for one resumes at the later of
    its own tick and the tick that work reached.

    `submit_with_context` runs its job inside a second copy of the submitter's context, where
    a tick is out of the submitting pool's sight. The pool's `submit` here already runs each
    job under a copy of the submitter's context, so every module's `submit_with_context`
    submits to it directly."""
    real_submit, real_result = ThreadPoolExecutor.submit, Future.result
    real_to_thread, real_gather = asyncio.to_thread, asyncio.gather
    ends: weakref.WeakKeyDictionary[Future, list[int]] = weakref.WeakKeyDictionary()

    def submit(self, fn, /, *args, **kwargs):
        context, end = contextvars.copy_context(), [0]

        def run():
            try:
                return fn(*args, **kwargs)
            finally:
                end[0] = _CLOCK.get()

        future = real_submit(self, context.run, run)
        ends[future] = end
        return future

    def result(self, timeout=None):
        try:
            return real_result(self, timeout)
        finally:
            _join(ends.get(self, [0])[0])

    async def to_thread(func, /, *args, **kwargs):
        end = [0]

        def run():
            try:
                return func(*args, **kwargs)
            finally:
                end[0] = _CLOCK.get()

        try:
            return await real_to_thread(run)
        finally:
            _join(end[0])

    async def gather(*aws, return_exceptions=False):
        tasks = [asyncio.ensure_future(aw) for aw in aws]
        try:
            return await real_gather(*tasks, return_exceptions=return_exceptions)
        finally:
            for task in tasks:
                if isinstance(task, asyncio.Task) and task.done():
                    _join(task.get_context().get(_CLOCK, 0))

    def submit_in_context(executor, fn, /, *args, **kwargs):
        return executor.submit(fn, *args, **kwargs)

    real_submit_with_context = platform_executor.submit_with_context
    for module in list(sys.modules.values()):
        if getattr(module, "submit_with_context", None) is real_submit_with_context:
            monkeypatch.setattr(module, "submit_with_context", submit_in_context)

    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    monkeypatch.setattr(Future, "result", result)
    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    monkeypatch.setattr(asyncio, "gather", gather)


class _DoorExecutor(ThreadPoolExecutor):
    """The door loop's thread pool, which can wait until every job it was given has ended."""

    def __init__(self) -> None:
        super().__init__()
        self._jobs: set[Future] = set()
        self._jobs_lock = threading.Lock()

    def submit(self, fn, /, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        with self._jobs_lock:
            self._jobs.add(future)
        future.add_done_callback(self._ended)
        return future

    def _ended(self, future: Future) -> None:
        with self._jobs_lock:
            self._jobs.discard(future)

    def drain(self) -> None:
        with self._jobs_lock:
            jobs = list(self._jobs)
        wait(jobs)


_DOOR_LOOP: list[tuple[asyncio.AbstractEventLoop, _DoorExecutor]] = []


@contextmanager
def door_loop() -> Iterator[None]:
    """One event loop and thread pool for every first door of a module, as a server keeps
    one, so no answer pays to start and join its threads."""
    loop, executor = asyncio.new_event_loop(), _DoorExecutor()
    loop.set_default_executor(executor)
    loop.set_task_factory(lambda loop, coro, **kwargs: _ClockTask(coro, loop=loop, **kwargs))
    _DOOR_LOOP.append((loop, executor))
    try:
        yield
    finally:
        _DOOR_LOOP.pop()
        executor.shutdown()
        loop.close()


def _run_to_completion[T](main: Coroutine[Any, Any, T]) -> T:
    """Run `main` on the door loop and return, as `asyncio.run` does, only once it has left
    no task and no thread job behind."""
    loop, executor = _DOOR_LOOP[-1]
    try:
        return loop.run_until_complete(main)
    finally:
        leftover = asyncio.all_tasks(loop)
        for task in leftover:
            task.cancel()
        if leftover:
            loop.run_until_complete(asyncio.wait(leftover))
        executor.drain()


def log_unit_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log every unit decision either door makes (`convertibility.decide_part_unit`) on the
    answer's recorder."""
    for module in (phase_answer, promote_ground):
        real = module.decide_part_unit

        def decide(*args, _real=real, **kwargs):
            decided = _real(*args, **kwargs)
            if _ACTIVE:
                _ACTIVE[-1].unit_decisions.append(decided)
            return decided

        monkeypatch.setattr(module, "decide_part_unit", decide)


def log_relaxation_walks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log the requests each nearest-real relaxation's member walk makes
    (`grounding._walked_relaxations`), and the ticks it starts and ends at."""
    walk = getattr(grounding, "elected_role_class_shape", None)
    if walk is None:
        # A base tree for the I9 fixture, whose relaxations walk no members.
        return

    def logged(*args, **kwargs):
        token, start = _RELAXATION_WALK.set(True), _CLOCK.get()
        try:
            return walk(*args, **kwargs)
        finally:
            _RELAXATION_WALK.reset(token)
            if _ACTIVE:
                with _ACTIVE[-1]._lock:
                    _ACTIVE[-1].relaxation_walks.append((start, _CLOCK.get()))

    monkeypatch.setattr(grounding, "elected_role_class_shape", logged)


def log_alternate_walks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log the requests the member walk of each alternate and re-election sibling makes
    (`phase_place_materialise.walk_alternate`)."""
    walk = getattr(phase_place_materialise, "walk_alternate", None)
    if walk is None:
        # A base tree for the I9 fixture, whose alternates walk no members ahead.
        return

    def logged(*args, **kwargs):
        token = _ALTERNATE_WALK.set(True)
        try:
            return walk(*args, **kwargs)
        finally:
            _ALTERNATE_WALK.reset(token)

    monkeypatch.setattr(phase_place_materialise, "walk_alternate", logged)


def log_facet_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log the requests each build of a Spec's facet options makes
    (`option_coverage._maybe_build_facet_options`), and the ticks it starts and ends at."""

    def logged(build):
        def run(*args, **kwargs):
            token, start = _FACET_OPTIONS.set(True), _CLOCK.get()
            try:
                return build(*args, **kwargs)
            finally:
                _FACET_OPTIONS.reset(token)
                if _ACTIVE:
                    with _ACTIVE[-1]._lock:
                        _ACTIVE[-1].facet_builds.append((start, _CLOCK.get()))

        return run

    wrap_facet_option_builds(monkeypatch, logged)


def log_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log, as each phase of either door begins, how many requests the answer has made
    (`Recorder.phases`), and as the first door's grounding of its elected shape begins
    (`phase_answer._ground_elected_answer`), a step of its outcome stage."""

    def logged(phase):
        def begin() -> None:
            if _ACTIVE:
                _ACTIVE[-1].phases.append((phase.__name__, len(_ACTIVE[-1].requests)))

        if asyncio.iscoroutinefunction(phase):

            async def run(state):
                begin()
                return await phase(state)

        else:

            def run(state):
                begin()
                return phase(state)

        return run

    monkeypatch.setattr(pipeline, "_PHASES", tuple(logged(p) for p in pipeline._PHASES))
    monkeypatch.setattr(
        phase_answer, "_ground_elected_answer", logged(phase_answer._ground_elected_answer)
    )
    monkeypatch.setattr(
        resubmit_promote,
        "_PROMOTE_PHASES",
        tuple(logged(p) for p in resubmit_promote._PROMOTE_PHASES),
    )


@contextmanager
def _recorded() -> Iterator[Recorder]:
    # Each answer walks members cold, so the reads it records are its own and two answers
    # the grid compares make the same walk.
    shared_enumerator().clear()
    recorder = Recorder()
    token = _CLOCK.set(0)
    _ACTIVE.append(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.pop()
        _CLOCK.reset(token)


# ---------------------------------------------------------------------------
# The world.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """A series of `stat_var` under `facet_id` at `place`, from `creditor` on a role route.

    `years` are the years the coverage index and the graph state. `sdmx_years`, when given,
    are the years SDMX returns instead, for an index ahead of or behind SDMX. SDMX returns
    the rows of `blank_years` with an empty value."""

    stat_var: str
    facet_id: str
    place: str
    years: tuple[int, ...]
    creditor: str = CREDITOR
    sdmx_years: tuple[int, ...] | None = None
    blank_years: tuple[int, ...] = ()

    @property
    def read_years(self) -> tuple[int, ...]:
        """The years an SDMX read of this series returns."""
        return self.years if self.sdmx_years is None else self.sdmx_years


@dataclass(frozen=True)
class Answer:
    """One door's outcome: the region result, the code of the input refusal it raised or
    any other error it raised, and what the recorder logged while it ran."""

    result: RegionResult | None
    refusal: str | None
    recorder: Recorder
    error: str | None = None

    @cached_property
    def part(self) -> Part | None:
        """The answer as the wire states it: the one response part either door assembles
        from its region with sentences on (`conjoin.assemble_region`). None for a refusal."""
        if self.result is None:
            return None
        response = assemble_region(
            self.result,
            extra_warnings=[],
            start_ms=0,
            engine_build="answer-invariants",
            max_candidates=_MAX_CANDIDATES,
        )
        return response.parts[0]

    @cached_property
    def specs(self) -> tuple[Spec, ...]:
        """The Specs the wire part states, each carrying its `rendered_sentence`: the
        answer's own, or its nearest-real suggestions. Candidates keep the region's order,
        the elected Spec first, where the part lists the broadest first."""
        part = self.part
        if isinstance(part, DefinitePart):
            return (part.spec,)
        if isinstance(part, CandidatesPart) and self.result is not None:
            stated = {spec.spec_id: spec for spec in part.candidates.specs}
            return tuple(stated[spec.spec_id] for spec in self.result.specs)
        if isinstance(part, NoDataPart):
            return tuple(part.no_data.nearest_real or ())
        return ()


class _Bind:
    """Elects the first shape, binds `bindings` and keeps `also_plausible` shapes as
    candidates."""

    def __init__(self, also_plausible: Sequence[int], bindings: Sequence[SlotBindingDraft]):
        self._also_plausible, self._bindings = list(also_plausible), list(bindings)

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return (
            elected_reply(0, bindings=self._bindings, also_plausible=self._also_plausible),
            None,
        )


def _fixture(facet_id: str) -> ObsFacetFixture:
    provenance, unit = FACETS[facet_id]
    return ObsFacetFixture(facet_id=facet_id, provenance_id=provenance, unit=unit)


_YEARS = range(1990, 2031)
# Exchange rates for every year a row set holds, keyed by currency.
_RATE_BODY = recipe_table_bodies(
    _RECIPE.recipe_id,
    rates={USD: dict.fromkeys(_YEARS, 1.0), EUR: dict.fromkeys(_YEARS, 0.9)},
)[_RATES]
# The rate table's `/availability` listing: the currencies its rows sit at.
_RATE_AVAILABILITY = json.dumps(
    {
        "data": {
            "dataConstraints": [
                {"cubeRegions": [{"keyValues": [{"id": "observationAbout", "values": [USD, EUR]}]}]}
            ]
        }
    }
)


@dataclass(frozen=True, eq=False)
class World:
    """One route over one set of rows: the doubles each door reads, built from `rows`.

    `stat_vars` maps each variable the query detects to its measured property and, for a
    nearest-real world, the purpose it is constrained to. `asked_purpose` is the purpose the
    query binds. The mint's member walk over the materialise graph settles each variable's
    family when `walk_settles`, and fails otherwise."""

    route: str
    rows: tuple[Row, ...]
    stat_vars: Mapping[str, tuple[str, str | None]]
    also_plausible: tuple[int, ...] = ()
    asked_purpose: str | None = None
    walk_settles: bool = True

    @property
    def role(self) -> bool:
        return self.route != "standard"

    @property
    def discovered(self) -> bool:
        return self.route.startswith("discovered")

    def lacks_total(self, stat_var: str) -> bool:
        """Whether the family an answer naming `stat_var` elects from holds no total: the
        query binds no purpose, and the variables of `stat_var`'s measured property each
        assert a purpose and do not all assert the same one, so none narrows the family on
        nothing. Only the standard route elects a family by its total."""
        if self.role:
            return False
        measured = self.stat_vars[stat_var][0]
        purposes = [p for m, p in self.stat_vars.values() if m == measured]
        return self.asked_purpose is None and None not in purposes and len(set(purposes)) > 1

    @property
    def label(self) -> str:
        """Every variable's label."""
        return "Test flow" if self.role else "Test amount"

    @property
    def bulk(self) -> bool:
        return self.route.endswith("bulk")

    @property
    def declared_roles(self) -> tuple[str, ...]:
        return ("creditor", "debtor") if self.role else ("observationAbout",)

    @cached_property
    def index(self) -> CoverageIndex:
        if not self.role:
            return build_coverage_index([], role_names={})
        return build_coverage_index(
            [(r.stat_var, r.facet_id, r.creditor, r.place, list(r.years)) for r in self.rows],
            role_names=dict.fromkeys(self.stat_vars, _ROLE_NAMES),
            facet_metadata={
                facet_id: FacetMetadata(provenance_id=provenance, unit=unit)
                for facet_id, (provenance, unit) in FACETS.items()
            },
        )

    @cached_property
    def coverage(self) -> CoverageSnapshot:
        return FakeCoverageService(self.index).snapshot()

    def _stat_var_arcs(self, stat_var: str) -> dict[str, tuple[str, ...]]:
        """`stat_var`'s arcs, which the graph and the materialise graph both serve: a nominal
        financial flow, its roles on a role route, and its purpose if it has
        one."""
        measured_property, purpose = self.stat_vars[stat_var]
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("FinancialFlow",),
            "measuredProperty": (measured_property,),
            "statType": ("measuredValue",),
            "measurementQualifier": ("Nominal",),
            **({"observationProperties": ("creditor", "debtor")} if self.role else {}),
            **({"constraintProperties": (PURPOSE,), PURPOSE: (purpose,)} if purpose else {}),
        }

    @cached_property
    def _nodes(self) -> dict:
        def arcs(stat_var: str) -> dict:
            return {
                prop: {"nodes": [{"dcid": value} for value in values]}
                for prop, values in self._stat_var_arcs(stat_var).items()
            }

        return {
            **{dcid: {"label": name, "type": "Country"} for dcid, name in PLACE_NAMES.items()},
            CREDITOR: {"label": "Credland", "type": "Country"},
            **{
                row.creditor: {"label": row.creditor, "type": "Country"}
                for row in self.rows
                if row.creditor != CREDITOR
            },
            **{
                stat_var: {
                    "label": "Test flow" if self.role else "Test amount",
                    "type": "StatisticalVariable",
                    "arcs": arcs(stat_var),
                }
                for stat_var in self.stat_vars
            },
            **{dcid: {"label": label} for dcid, label in SOURCE_LABELS.items()},
            _PURPOSE_ASKED: {"label": "Asked purpose"},
            _PURPOSE_HELD: {"label": "Held purpose"},
        }

    @cached_property
    def _resolved_places(self) -> dict:
        """Each place name's resolution, made once as a request resolves its entities before
        it resolves any variable (`resolve_variable`'s `pre_resolved`)."""
        names = list(PLACE_NAMES.values())
        places = {name: dcid for dcid, name in PLACE_NAMES.items()}
        resolver = make_resolver(FakeGraph(nodes=self._nodes, resolve=places))
        return dict(zip(names, asyncio.run(resolver.resolve(names)), strict=True))

    @cached_property
    def _graph_obs(self) -> dict[str, list[dict]]:
        """The graph's observation facets, which the standard route alone reads."""
        obs: dict[str, list[dict]] = {}
        for row in self.rows if not self.role else ():
            provenance, unit = FACETS[row.facet_id]
            obs.setdefault(f"{row.stat_var}|{row.place}", []).append(
                {
                    "facetId": row.facet_id,
                    **({"provenanceId": provenance} if provenance else {}),
                    "unit": unit,
                    "earliestDate": str(row.years[0]),
                    "latestDate": str(row.years[-1]),
                    "obsCount": len(row.years),
                    "dates": [str(y) for y in row.years],
                    "values": [1.0] * len(row.years),
                }
            )
        return obs

    @cached_property
    def _graphs(self) -> dict[str | None, FakeGraph]:
        return {}

    def graph(self, query: str | None = None) -> FakeGraph:
        """The graph a door reads, detecting the world's variables for `query`. One graph
        serves every answer to the same query."""
        if query not in self._graphs:
            scores = [0.9 - 0.05 * i for i in range(len(self.stat_vars))]
            detect = {query: {"svs": list(self.stat_vars), "cosine_scores": scores}}
            self._graphs[query] = _Graph(
                nodes=self._nodes,
                obs=self._graph_obs,
                detect=detect if query else {},
                resolve={name: dcid for dcid, name in PLACE_NAMES.items()},
            )
        return self._graphs[query]

    @cached_property
    def sdmx(self) -> FakeSdmxClient:
        """The bulk role read's rows, which a role route above the pair cap reads, and the
        default place scope's availability, every place the world's graph holds data at."""
        observations: dict = {}
        for row in self.rows if self.role else ():
            provenance, unit = FACETS[row.facet_id]
            dates = tuple(str(y) for y in row.read_years)
            observations.setdefault(row.stat_var, {}).setdefault(
                (("creditor", row.creditor), ("debtor", row.place)), []
            ).append(
                Facet(
                    facet_id=row.facet_id,
                    provenance_id=provenance,
                    unit=unit,
                    earliest_date=dates[0],
                    latest_date=dates[-1],
                    obs_count=len(dates),
                    dates=dates,
                    values=tuple(
                        None if year in row.blank_years else 1.0 for year in row.read_years
                    ),
                )
            )
        return _Sdmx(
            observations=observations,
            availability={},
            scope_availability=scope_availability_of(self._graph_obs),
        )

    def _materialise_rows(self, request: httpx.Request) -> httpx.Response:
        """Serve the rows a materialise read names: its variable, bound roles and facet
        ids, each series' latest row alone under `LATEST` and its rows at the periods the
        read names."""
        params = request.url.params
        stat_var = params.get("c[variableMeasured]", "").removeprefix("dcid:")
        _record("materialise_sdmx", "GET" + RATE_TABLE_READ if stat_var == _RATES else "GET")
        if stat_var == _RATES and "/availability/" in request.url.path:
            return httpx.Response(
                200,
                text=_RATE_AVAILABILITY,
                headers={"content-type": "application/vnd.sdmx.structure+json;version=2.0.0"},
            )
        body = _RATE_BODY if stat_var == _RATES else _csv_header(self.declared_roles)
        place_key = "c[debtor]" if self.role else "c[observationAbout]"
        places = params[place_key].split(",") if place_key in params else None
        facet_ids = params["c[facetId]"].split(",") if "c[facetId]" in params else None
        period = params.get("c[TIME_PERIOD]")
        for row in self.rows:
            if row.stat_var != stat_var or (places is not None and row.place not in places):
                continue
            if facet_ids is not None and row.facet_id not in facet_ids:
                continue
            years = row.read_years[-1:] if period == "LATEST" else row.read_years
            if period not in (None, "LATEST"):
                years = tuple(year for year in years if str(year) in period.split(","))
            for year in years:
                body += _csv_row(
                    stat_var=row.stat_var,
                    role_dcids=[row.creditor, row.place] if self.role else [row.place],
                    fixture=_fixture(row.facet_id),
                    date_str=str(year),
                    value=None if year in row.blank_years else 1.0,
                )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    @cached_property
    def materialise_graph(self) -> FakeGraphReader:
        """What materialise reads: a standard pair's facets, with spans and no dates as the
        graph's facet probe returns them, each variable's arcs and the variables of each
        measured property, which the mint's member walk reads, and the arcs that decide a
        unit request (each source declaring the conversion recipe, each currency naming the
        entity its rates sit at)."""
        facets: dict[tuple[str, str], list[MaterialiseFacet]] = {}
        for row in self.rows if not self.role else ():
            provenance, unit = FACETS[row.facet_id]
            facets.setdefault((row.stat_var, row.place), []).append(
                MaterialiseFacet(
                    facet_id=row.facet_id,
                    provenance_id=provenance,
                    measurement_method=None,
                    unit=unit,
                    obs_count=len(row.years),
                    earliest_date=str(row.years[0]),
                    latest_date=str(row.years[-1]),
                    scaling_factor=None,
                )
            )
        arcs = {
            **{stat_var: self._stat_var_arcs(stat_var) for stat_var in self.stat_vars},
            **{
                node: node_arcs
                for source in SOURCE_LABELS
                for node, node_arcs in conversion_arcs(
                    recipe_id=_RECIPE.recipe_id, provenance_id=source, currencies=[USD, EUR]
                ).items()
            },
        }
        subjects: dict[tuple[str, str], list[str]] = {}
        for stat_var, (measured_property, _purpose) in self.stat_vars.items():
            subjects.setdefault((measured_property, "measuredProperty"), []).append(stat_var)
        return (_MaterialiseGraph if self.walk_settles else _UnwalkableGraph)(
            facets=facets,
            arcs=arcs,
            subjects={
                key: SubjectPage(dcids=tuple(dcids), truncated=False)
                for key, dcids in subjects.items()
            },
        )

    @cached_property
    def materialise_sdmx(self) -> MaterialiseSdmxClient:
        return MaterialiseSdmxClient(
            base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(self._materialise_rows)
        )

    def _options(self, *, preview: bool) -> EngineOptions:
        return EngineOptions(
            include_observations=preview,
            materialise_graph=self.materialise_graph,
            materialise_sdmx=self.materialise_sdmx,
            rate_cache=RateTableCache(),
        )

    def first(
        self,
        places: Sequence[str],
        date_request: DateRequest | None,
        *,
        preview: bool,
        unit_text: str | None,
    ) -> Answer:
        """The first door: a query naming `places` in that order, or none when the route
        discovers its destinations."""
        names = [] if self.discovered else [PLACE_NAMES[p] for p in places]
        if self.role:
            query = "debt flows" + (f" to {' and '.join(names)}" if names else "")
        else:
            query = "amount in " + " and ".join(names)
        graph = self.graph(query)
        pre_resolved = self._resolved_places
        bindings = (
            [
                SlotBindingDraft(
                    axis="what",
                    property_dcid=PURPOSE,
                    kind="value",
                    value_dcids=[self.asked_purpose],
                )
            ]
            if self.asked_purpose
            else []
        )

        async def run(recorder: Recorder) -> RegionResult:
            async def emit(event) -> None:
                if isinstance(event, SentenceEvent):
                    recorder.sentences.append(len(recorder.requests))

            return await resolve_variable(
                "amount",
                entities=names,
                date_request=date_request,
                detect_query=query,
                role_query=query,
                entity_roles=door_roles_for_test_query(query, names) if self.role else None,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                pre_resolved=pre_resolved,
                llm=_Bind(self.also_plausible, bindings),
                sdmx=self.sdmx,
                base_timing={},
                emit=emit,
                unit_text=unit_text,
                coverage=self.coverage,
                options=self._options(preview=preview),
            )

        with _recorded() as recorder:
            try:
                return Answer(_run_to_completion(run(recorder)), None, recorder)
            except EngineInputError as refusal:
                return Answer(None, refusal.code, recorder)
            except Exception as error:
                return Answer(None, None, recorder, f"{type(error).__name__}: {error}")

    def resubmit(
        self,
        spec: Spec,
        *,
        preview: bool,
        facets: Mapping[str, object] | None = None,
        window: Mapping[str, object] | None = None,
    ) -> Answer:
        """The second door: `spec`'s identity posted back, unchanged or with `facets` added
        to its facet bindings or `window` replacing its requested window."""
        identity = spec.definition.identity
        if facets is not None or window is not None:
            payload = identity.model_dump(mode="python")
            if facets is not None:
                payload["facets"] = {**(payload.get("facets") or {}), **facets}
            if window is not None:
                payload["requested_window"] = window
            identity = BaseIdentity.model_validate(payload)
        with _recorded() as recorder:
            try:
                result = resolve_spec_resubmit(
                    inp=ResubmitIdentityPayload(
                        identity=identity,
                    ),
                    graph=self.graph(),
                    sdmx=self.sdmx,
                    coverage=self.coverage,
                    options=self._options(preview=preview),
                )
            except EngineInputError as refusal:
                return Answer(None, refusal.code, recorder)
            except Exception as error:
                return Answer(None, None, recorder, f"{type(error).__name__}: {error}")
            return Answer(result, None, recorder)

    def held(self, stat_var: str, place: str) -> dict[str, tuple[int, ...]]:
        """Each facet's years for `stat_var` at `place`."""
        return {
            row.facet_id: row.years
            for row in self.rows
            if row.stat_var == stat_var and row.place == place
        }


def _stat_var(route: str, name: str) -> str:
    return f"TESTFIXTURE/AnswerInvariants/{'Amount' if route == 'standard' else 'Flow'}{name}"


def world(route: str, row_set: str, kind: str, *, relaxed: bool = False) -> World:
    """`route` over `row_set`.

    A `definite` world holds every row on one variable. A `candidates` world keeps a second
    variable as also plausible: the first holds Aland's rows alone and the second every row,
    so the two serve different places. A `collapse` world's second variable holds no row. A
    `nearest_real` world's query binds a purpose whose variable holds no row, while a
    variable of the other purpose holds every row; `relaxed` is the same world asked with
    no purpose bound. The member walk fails in an `UNSETTLED_ROW_SET` world."""
    held = [
        (place, facet_id, years)
        for place, facets in ROW_SETS[row_set].items()
        for facet_id, years in facets
    ]
    first, second = _stat_var(route, ""), _stat_var(route, "Alt")
    walk_settles = row_set != UNSETTLED_ROW_SET
    if kind == "nearest_real":
        return World(
            route=route,
            rows=tuple(Row(second, facet_id, place, years) for place, facet_id, years in held),
            stat_vars={first: (_AMOUNT, _PURPOSE_ASKED), second: (_AMOUNT, _PURPOSE_HELD)},
            asked_purpose=None if relaxed else _PURPOSE_ASKED,
            walk_settles=walk_settles,
        )
    rows = [
        Row(first, facet_id, place, years)
        for place, facet_id, years in held
        if kind != "candidates" or place == ALAND
    ]
    if kind == "candidates":
        rows += [Row(second, facet_id, place, years) for place, facet_id, years in held]
    if kind == "definite":
        return World(
            route=route,
            rows=tuple(rows),
            stat_vars={first: (_AMOUNT, None)},
            walk_settles=walk_settles,
        )
    return World(
        route=route,
        rows=tuple(rows),
        stat_vars={first: (_AMOUNT, None), second: ("amountAlt", None)},
        also_plausible=(1,),
        walk_settles=walk_settles,
    )
