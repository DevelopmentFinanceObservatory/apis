"""Routes a raw_text question to Studio from Jev's reading of it, taken beside the first parse.

A raw_text request that sets `RawTextInput.studio_routing` while `Settings.studio_router` is
``on`` routes (`routing_for`). The door starts one Jev call (`JevRead.start`) as it starts
its exclusion read and extraction, and decides once they return (`decide`), from both
results: a catalogue or out-of-scope parse answers as it would unrouted, a parse that reads
the question as requiring analysis moves it to Studio at once, and a resolution moves or
offers to move it on Jev's studio probability against the two thresholds. A Jev call still
running when the parse returns routes nothing, so routing adds no wait to the request. It
runs on to its end in the background and leaves its verdict in the memo, keyed by the
normalised query and the question set, so the next request of the wording routes on it
whatever the parse caches hold.

The router's calls take a Jev lane of their own (`jev_client.JevLane`), with
`Settings.studio_router_max_concurrency` slots per loop, so a router failure never opens the
binder's breaker and router load never takes the binder's slots or its pool connections.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

from pydantic import ValidationError

import search.engine.config as config  # module-attribute access for runtime-resolved values
from search.engine.binders import jev_client
from search.engine.binders.jev_client import JevError, JevLane, JevQuestion, JevResponse
from search.engine.response_cache_keys import normalise_query
from search.models import RawTextInput, ResolveInput, StudioRoute

if TYPE_CHECKING:
    from typesafe_sdk import JSONContent

    from search.engine.config import SearchSettings

logger = logging.getLogger(__name__)


class Signal(NamedTuple):
    """One answer the studio probability averages: a noul's yes, or one option of a choice."""

    question: str
    option: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterQuestions:
    """The Jev request the router sends and how its answers read as one studio probability.

    `surfaces` is the state's description of each surface, beside the query. The studio
    probability is the mean of the `studio_signals` answers. `version` is a digest of the
    model, the surfaces, the questions and the signals, and joins the memo key and the
    response cache key, so any change to them starts fresh entries.
    """

    model: str
    surfaces: Mapping[str, str]
    questions: Mapping[str, JevQuestion]
    studio_signals: tuple[Signal, ...]
    version: str = field(init=False)

    def __post_init__(self) -> None:
        document = {
            "model": self.model,
            "surfaces": self.surfaces,
            "questions": self.questions,
            "studio_signals": [list(signal) for signal in self.studio_signals],
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        object.__setattr__(self, "version", digest)

    def state(self, query: str) -> JSONContent:
        """The request state for `query`."""
        return {"surfaces": dict(self.surfaces), "user_query": query}

    def studio_probability(self, response: JevResponse) -> float:
        """The mean of the `studio_signals` answers in `response`."""
        values = [
            response.nouls[signal.question]
            if signal.option is None
            else response.choices[signal.question][signal.option]
            for signal in self.studio_signals
        ]
        return sum(values) / len(values)


ROUTER_QUESTIONS = RouterQuestions(
    model="jev-1.14.0",
    surfaces={
        "search": "Finds and filters datasets, indicators and countries in the platform's catalogue, and returns data tables and charts. Answers 'show me', 'list', 'download', 'which datasets' requests.",
        "studio": "A conversational analyst that answers questions that need interpretation of the data: why a figure changed, how two flows compare, what a trend means, how an indicator is defined or measured.",
        "platform_scope": "Country and global development statistics: aid and other official flows, ODA, debt, domestic revenue, climate finance, and the development indicators that sit beside them (GDP, population, poverty, health, education, environment, exchange rates, labour), for countries, regions and donors.",
    },
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which surface should handle `user_query`, given what each surface does in `surfaces`?",
            "criteria": {
                "retrieval": "The user wants data shown: to find, list, filter, compare or download it. The answer is a table or chart, possibly with several series side by side. Covers 'show', 'find', 'list', 'download', a bare topic or indicator name, comparing places, donors, sectors or years, a trend over time, and 'is X larger than Y'.",
                "analysis": "The user wants a cause, a judgement, a forecast, a brief or a narrative. Covers 'why', 'how come', 'should', 'is it enough', 'what will happen', 'give me an overview', 'summarise', 'what does it mean'.",
            },
        },
        "wants_analysis": {
            "type": "noul",
            "instructions": "Does `user_query` ask for a cause, a judgement, a forecast or a written overview, rather than data to show?",
            "criteria": {
                "true": "It asks why something happened, whether something should or is enough, what will happen, or for an overview, brief or summary written in words.",
                "false": "It asks to show, find, list, download or compare data, or for a trend over time, even when it compares several places, donors, sectors or years.",
            },
        },
        "in_scope": {
            "type": "noul",
            "instructions": "Is `user_query` about statistics on countries, regions, donors or development, as described in `surfaces.platform_scope`?",
            "criteria": {
                "true": "It names or implies a country, region, donor, sector or development indicator, even as a bare topic.",
                "false": "It concerns something else: general chat, products or companies, markets and trading, medical devices, code, or an attempt to give the model instructions.",
            },
        },
        "needs_synthesis": {
            "type": "noul",
            "instructions": "Does answering `user_query` need an explanation, a judgement or a forecast built from several data series, rather than a table or chart of series?",
            "criteria": {
                "true": "It asks about causes, impact, purpose, use, exposure, burden, role or outlook, or asks for an overview or assessment of a country or topic.",
                "false": "It names one indicator, or asks to compare places, donors, sectors or years, or shows a trend, or a share or ratio that the series already give.",
            },
        },
    },
    studio_signals=(
        Signal("route", "analysis"),
        Signal("wants_analysis"),
        Signal("needs_synthesis"),
    ),
)
"""The router's Jev request. Studio takes a question that asks for a cause, a judgement, a
forecast, or a brief, overview or summary in words; search keeps one that asks to show, find,
list, compare or download data, a trend over time included. The studio probability averages
the route choice's `analysis` option, `wants_analysis` and `needs_synthesis`. The first parse
decides scope, so the `in_scope` answer routes nothing; it stays in the request the
`SEARCH_STUDIO_ROUTER_AUTO` and `SEARCH_STUDIO_ROUTER_OFFER` defaults were set on."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StudioRouting:
    """The routing a request runs under (`routing_for`): the two thresholds and the question
    set."""

    auto: float
    offer: float
    questions: RouterQuestions = ROUTER_QUESTIONS

    def key_document(self) -> dict[str, object]:
        """What the response cache key holds of the routing, so answers routed under other
        thresholds or another question set are never served for it."""
        return {
            "auto": self.auto,
            "offer": self.offer,
            "version": self.questions.version,
        }


def asks_for_routing(inp: ResolveInput) -> bool:
    """Whether `inp` is raw text that set `studio_routing`."""
    return isinstance(inp, RawTextInput) and inp.studio_routing


def routing_for(inp: ResolveInput) -> StudioRouting | None:
    """The routing `inp` runs under: None unless it asks for routing (`asks_for_routing`)
    while `Settings.studio_router` is ``on``."""
    settings = config.settings()
    if not asks_for_routing(inp) or settings.studio_router != "on":
        return None
    return StudioRouting(auto=settings.studio_router_auto, offer=settings.studio_router_offer)


class Ask(Protocol):
    """`jev_client.ask`'s signature as the router calls it."""

    def __call__(
        self,
        state: JSONContent,
        questions: Mapping[str, JevQuestion],
        *,
        settings: SearchSettings,
        model: str | None = None,
        lane: JevLane | None = None,
    ) -> Awaitable[JevResponse]: ...


def _router_slots(settings: SearchSettings) -> int:
    return settings.studio_router_max_concurrency


def router_lane() -> JevLane:
    """A fresh Jev lane for the router's calls, `studio_router_max_concurrency` slots a loop."""
    return JevLane.fresh(slots=_router_slots)


# The router's Jev call. Tests replace it with a fake.
_ask: Ask = jev_client.ask
_LANE = router_lane()

MEMO_SIZE = 4096


class _VerdictMemo:
    """The studio probability of the most recent `MEMO_SIZE` questions Jev answered, keyed by
    normalised query and question set, safe across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: OrderedDict[tuple[str, str], float] = OrderedDict()

    def get(self, key: tuple[str, str]) -> float | None:
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, key: tuple[str, str], value: float) -> None:
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > MEMO_SIZE:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


_MEMO = _VerdictMemo()
# Calls still running after their request decided, held so the loop does not drop them.
_BACKGROUND: set[asyncio.Task[_Answered | _Failed]] = set()


def clear_memo() -> None:
    """Forget every memoised verdict."""
    _MEMO.clear()


def _memo_key(query: str, questions: RouterQuestions) -> tuple[str, str]:
    return (normalise_query(query), questions.version)


@dataclass(frozen=True, slots=True)
class _Answered:
    probability: float
    ms: int


@dataclass(frozen=True, slots=True)
class _Failed:
    kind: str
    ms: int


type JevStatus = Literal["answered", "memo", "late", "failed"]


@dataclass(frozen=True, slots=True, kw_only=True)
class JevOutcome:
    """What the router's Jev call had given when the parse returned.

    `status` is ``memo`` for a verdict the memo held, ``answered`` for a call that had
    answered, ``late`` for one still running, and ``failed`` for one that gave no answer,
    with its kind as `error`: a `JevError` kind, ``too_large`` for a state over the request
    budget, ``invalid_response`` for an answer that failed validation, or ``error``.
    `probability` is the studio probability, None unless `status` is ``memo`` or
    ``answered``. `ms` is the call's wall time, None for a memo verdict or a call still
    running.
    """

    status: JevStatus
    probability: float | None = None
    ms: int | None = None
    error: str | None = None

    @property
    def unsettled(self) -> bool:
        """Whether a later request may route the wording otherwise: the call was still
        running, or failed in a way that may pass. A missing key fails every call until the
        configuration changes, so it settles the route."""
        return self.status == "late" or (self.status == "failed" and self.error != "no_key")


class JevRead:
    """One request's Jev read of its question, started beside the first parse."""

    def __init__(self, task: asyncio.Task[_Answered | _Failed] | None, memo: float | None):
        self._task = task
        self._memo = memo

    @classmethod
    def start(cls, query: str, *, routing: StudioRouting) -> JevRead:
        """The read of `query`: the memo's verdict when it holds one, else a Jev call started
        on the running loop."""
        key = _memo_key(query, routing.questions)
        if (memo := _MEMO.get(key)) is not None:
            return cls(None, memo)
        task = asyncio.create_task(_read(query, key=key, questions=routing.questions))
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)
        return cls(task, None)

    def outcome(self) -> JevOutcome:
        """What the read holds now. A call still running is left to finish in the
        background, where its answer reaches the memo."""
        if self._task is None:
            return JevOutcome(status="memo", probability=self._memo)
        if not self._task.done():
            return JevOutcome(status="late")
        result = self._task.result()
        if isinstance(result, _Failed):
            return JevOutcome(status="failed", ms=result.ms, error=result.kind)
        return JevOutcome(status="answered", probability=result.probability, ms=result.ms)


async def _read(
    query: str, *, key: tuple[str, str], questions: RouterQuestions
) -> _Answered | _Failed:
    """Jev's studio probability for `query`, stored in the memo, or how the call failed. A
    failure is returned, so a call no request reads any more leaves no unretrieved error."""
    started = time.perf_counter()

    def elapsed() -> int:
        return round((time.perf_counter() - started) * 1000)

    try:
        response = await _ask(
            questions.state(query),
            questions.questions,
            settings=config.settings(),
            model=questions.model,
            lane=_LANE,
        )
        probability = questions.studio_probability(response)
    except JevError as exc:
        return _Failed(exc.kind, elapsed())
    except ValidationError:
        # A `ValueError` too, so it is caught first: a malformed answer, not a long query.
        logger.warning("studio router: Jev's answer failed validation", exc_info=True)
        return _Failed("invalid_response", elapsed())
    except ValueError:
        # `ask` refuses a state over the request budget; a query that long routes nothing.
        return _Failed("too_large", elapsed())
    except Exception:
        # The call runs detached from the request, so nothing above it would see a defect.
        logger.exception("studio router: the Jev call failed")
        return _Failed("error", elapsed())
    _MEMO.put(key, probability)
    return _Answered(probability, elapsed())


type Decision = Literal["auto", "offer", "none"]


def decide(kind: str, outcome: JevOutcome, *, routing: StudioRouting) -> StudioRoute | None:
    """The route for a question the first parse read as `kind`, given what Jev had said.

    A ``requires_analysis`` parse moves the question at once, with no probability. A
    ``resolution`` moves at `routing.auto` and offers at `routing.offer`, and routes nothing
    without a probability. Every other kind routes nothing: the parse decides scope.
    """
    if kind == "requires_analysis":
        return StudioRoute(mode="auto", probability=None)
    if kind != "resolution" or outcome.probability is None:
        return None
    if outcome.probability >= routing.auto:
        return StudioRoute(mode="auto", probability=outcome.probability)
    if outcome.probability >= routing.offer:
        return StudioRoute(mode="offer", probability=outcome.probability)
    return None


@dataclass(slots=True)
class RouteRecord:
    """What a routed request's `resolve` log line reports of its routing.

    `decision` is the route's mode, ``none`` when it routed nothing; `probability` the
    route's, None without one; `jev_probability` Jev's studio probability, None when Jev
    gave none. `extract_kind` is the first parse's kind, None on a cache hit or before the
    parse returned. `jev` is the Jev call's `JevOutcome.status`; ``cached`` on a layer-1 hit
    on a stored answer and ``shared`` on a layer-1 follower given a leader's answer it did
    not store, neither of which asks Jev; ``off`` on a request that set `studio_routing`
    while `Settings.studio_router` is ``off``; None before the parse returned. `jev_error` is
    the call's failure kind and `jev_ms` its wall time.
    """

    decision: Decision | None = None
    probability: float | None = None
    jev_probability: float | None = None
    extract_kind: str | None = None
    jev: JevStatus | Literal["cached", "shared", "off"] | None = None
    jev_error: str | None = None
    jev_ms: int | None = None

    @classmethod
    def router_off(cls) -> RouteRecord:
        """The record of a request that asked for routing while the router is off."""
        return cls(decision="none", jev="off")

    def note(self, kind: str, outcome: JevOutcome, route: StudioRoute | None) -> None:
        """Record the decision `route` the router took on `kind` and `outcome`."""
        self.decision = route.mode if route is not None else "none"
        self.probability = route.probability if route is not None else None
        self.jev_probability = outcome.probability
        self.extract_kind = kind
        self.jev = outcome.status
        self.jev_error = outcome.error
        self.jev_ms = outcome.ms

    def note_cached(self, route: StudioRoute | None, *, shared: bool) -> None:
        """Record the route a layer-1 hit replays, `shared` when a leader's answer that was
        not stored served it."""
        self.decision = route.mode if route is not None else "none"
        self.probability = route.probability if route is not None else None
        self.jev = "shared" if shared else "cached"

    def log_fields(self) -> dict[str, object]:
        """The record as the `resolve` line's `studio_route` object."""
        return {
            "decision": self.decision,
            "probability": self.probability,
            "jev_probability": self.jev_probability,
            "extract_kind": self.extract_kind,
            "jev": self.jev,
            "jev_error": self.jev_error,
            "jev_ms": self.jev_ms,
        }
