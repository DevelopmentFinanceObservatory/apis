"""Pool order: our pass elects over our candidates, and the UN System Data Commons'
shapes, derived beside it, reach a menu on the UN pass that runs when ours ends `no_data`
(`region.pipeline._fill_gap`). With `SEARCH_UN_ON_FIRST_MENU` they also reach our pass's
menu beside ours (`region.phase_intake._merge_un_pool`). Each pool meets its own score
floor, and a part waits for the UN resolve and its shape derivation at most
`SEARCH_UN_FIRST_PASS_WAIT_S` from the start of derivation: when its own pass ends
`no_data`, or under the flag when its best shape is weak.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import cast

import pytest

from dfo_platform.timings import record_elapsed
from materialise.errors import UpstreamVersionChangedError
from origin_catalogue import note_un_blocked_places
from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import GeminiBinder
from search.engine.clients.graph import DetectedSV, Detection
from search.engine.errors import DeclaredDefaultUnsupportedError, GraphInfraError
from search.engine.events import Event, EventSink, FrameEvent, SentenceEvent, StepEvent
from search.engine.qualifier_match import CodedQualifier, QualifierMatch
from search.engine.quantity_read import NONE_OF_THESE, QuantityOutput
from search.engine.region import phase_intake, pipeline
from search.engine.region.phase_rank_bind import _dataset_candidates
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions, ResolveState, UnLeg, UnPool
from search.engine.shape import Pool, ShapeDraft
from search.engine.source_scope import SourceScope
from search.models import Spec, Warning
from tests.engine._binders import nominee_ratings
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_ids_index_sourcing import _crs_shape
from tests.fixtures import FakeGraph, make_resolver

_KEN = "country/KEN"
_TZA = "country/TZA"
_OURS = "Percent_Person_Literate"
_UN = "undata/unicef/SERIES-ED_LR"
_UNESCO = "undata/unesco/SERIES-LR_AG15T24"
_UNFPA = "undata/unfpa/PCT_LITERATE"
_POPULATION = "Count_Person"
_FEMALE_POPULATION = "Count_Person_Female"
_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_UN_GDP_PER_CAPITA = "undata/wb/SERIES-NY_GDP_PCAP_CD"
_QUERY = "literacy rate"
_PER_CAPITA_QUERY = "gdp per capita"


def _arc(*dcids: str) -> dict:
    return {"nodes": [{"dcid": dcid} for dcid in dcids]}


def _sv_node(
    dcid: str,
    *,
    population_type: str = "Person",
    meas_prop: str,
    constraints: dict[str, str] | None = None,
    denominator: str | None = None,
) -> dict:
    constraints = constraints or {}
    arcs = {
        "typeOf": _arc("StatisticalVariable"),
        "name": {"nodes": [{"value": dcid}]},
        "populationType": _arc(population_type),
        "measuredProperty": _arc(meas_prop),
        "statType": _arc("measuredValue"),
        "constraintProperties": _arc(*constraints),
        **{prop: _arc(value) for prop, value in constraints.items()},
    }
    if denominator is not None:
        arcs["measurementDenominator"] = _arc(denominator)
    return {"label": dcid, "type": "StatisticalVariable", "arcs": arcs}


_NODES = {
    _KEN: {"label": "Kenya", "type": "Country"},
    _TZA: {"label": "Tanzania", "type": "Country"},
    _OURS: _sv_node(_OURS, meas_prop="literacyRate"),
    # A UN series has its own population type, as every UN System Data Commons variable does.
    _UN: _sv_node(_UN, population_type="undata/unicef/SERIES-ED_LR", meas_prop="value"),
    _UNESCO: _sv_node(_UNESCO, population_type=_UNESCO, meas_prop="value"),
    _UNFPA: _sv_node(_UNFPA, population_type=_UNFPA, meas_prop="value"),
    _POPULATION: _sv_node(_POPULATION, meas_prop="count"),
    _FEMALE_POPULATION: _sv_node(
        _FEMALE_POPULATION, meas_prop="count", constraints={"gender": "Female"}
    ),
    _GDP: _sv_node(_GDP, population_type="EconomicActivity", meas_prop="amount"),
    _UN_GDP_PER_CAPITA: _sv_node(
        _UN_GDP_PER_CAPITA, population_type="undata/wb/SERIES-NY_GDP_PCAP_CD", meas_prop="value"
    ),
}

_FACET = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}


def _leg(*scored: tuple[str, float]) -> dict:
    return {"svs": [dcid for dcid, _ in scored], "cosine_scores": [score for _, score in scored]}


def _graph(
    *,
    ours: tuple[tuple[str, float], ...] = ((_OURS, 0.9),),
    un: tuple[tuple[str, float], ...] | None = ((_UN, 0.85),),
    data: tuple[str, ...] = (_UN,),
    query: str = _QUERY,
    cls: type[FakeGraph] = FakeGraph,
    nodes: dict[str, dict] = _NODES,
    **kwargs,
) -> FakeGraph:
    """Our resolve returns `ours` and the UN's `un` (`None` fails it), over `nodes`; the
    variables in `data` have observations at Kenya."""
    return cls(
        nodes=nodes,
        obs={f"{dcid}|{_KEN}": [_FACET] for dcid in data},
        detect={query: _leg(*ours)},
        un_detect={query: None if un is None else _leg(*un)},
        resolve={"Kenya": _KEN, "Tanzania": _TZA},
        **kwargs,
    )


def _menu_blocks(prompt: str) -> list[str]:
    """Each shape's block of the bind menu, in menu order."""
    shapes = prompt.split("CANDIDATE SHAPES", 1)[1]
    return re.split(r"\nShape \d+:\n", shapes)[1:]


class _ElectLLM:
    """Elects the menu shape whose sample members name `dcid`, else the first, or with
    `ask_without` asks when no shape names `dcid`. It binds nothing but `bindings`, on the
    first bind only, and nominates the shapes naming `nominate` (`also_plausible`). Its
    quantity read chooses the denominator `denominator` names, when given, on the phrase's
    words `base_words`. Records each bind menu it saw."""

    def __init__(
        self,
        dcid: str | None = None,
        *,
        nominate: tuple[str, ...] = (),
        bindings: tuple[SlotBindingDraft, ...] = (),
        ask_without: bool = False,
        denominator: str | None = None,
        base_words: str | None = None,
    ) -> None:
        self.dcid = dcid
        self.nominate = nominate
        self.bindings = bindings
        self.ask_without = ask_without
        self.denominator = denominator
        self.base_words = base_words
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            quantity = QuantityOutput(
                base_words=self.base_words, denominator_index=self._denominator_index(prompt)
            )
            return quantity, None
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        blocks = _menu_blocks(prompt)
        chosen = next(
            (i for i, block in enumerate(blocks) if self.dcid is not None and self.dcid in block),
            None,
        )
        usage = {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}
        nominated = [
            i for i, block in enumerate(blocks) if any(dcid in block for dcid in self.nominate)
        ]
        output = _UniversalBindOutput(
            chosen_shape_index=chosen or 0,
            bindings=list(self.bindings) if len(self.prompts) == 1 else [],
            also_plausible=nominated or None,
            ask="Which measure do you mean?" if chosen is None and self.ask_without else None,
            measuring=[
                ShapeRating(index=chosen or 0, rating=1.0),
                *nominee_ratings(nominated, rated=(chosen or 0,)),
            ],
        )
        return output, usage

    def _denominator_index(self, prompt: str) -> int | None:
        """The index of the listed denominator `denominator` names, else the index of
        "none of these", as the model answers when the menu's series divide by no such
        base."""
        if self.denominator is None:
            return None
        listed = re.findall(r"\n  (\d+): ([^\n]*)", prompt.split("DENOMINATORS", 1)[1])
        return next(
            (int(number) for number, entry in listed if self.denominator in entry),
            next(int(number) for number, entry in listed if entry == NONE_OF_THESE),
        )


async def _resolving(
    graph: FakeGraph,
    *,
    query: str = _QUERY,
    entities: tuple[str, ...] = ("Kenya",),
    llm: _ElectLLM | None = None,
    emit: EventSink | None = None,
    source_scope: SourceScope | None = None,
    qualifiers: tuple[CodedQualifier, ...] | None = None,
    deadline: float | None = None,
) -> RegionResult:
    return await resolve_variable(
        query,
        entities=list(entities),
        date_request=None,
        detect_query=query,
        role_query=query,
        qualifiers=qualifiers,
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        llm=llm or _ElectLLM(),
        base_timing={},
        emit=emit,
        source_scope=source_scope,
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(deadline=deadline),
    )


def _resolve(graph: FakeGraph, **kwargs) -> RegionResult:
    return asyncio.run(_resolving(graph, **kwargs))


def _elected(result: RegionResult) -> list[str]:
    return [sv.ref.ref for spec in result.specs for sv in spec.resolved.members]


def _codes(result: RegionResult) -> list[str]:
    return [w.code for w in result.warnings]


def _menu(llm: _ElectLLM) -> str:
    (prompt,) = llm.prompts
    return prompt


def _named(llm: _ElectLLM, dcid: str) -> list[bool]:
    """Whether each bind menu `llm` saw, in order, names `dcid`."""
    return [dcid in prompt for prompt in llm.prompts]


def _gap_fill_line(caplog: pytest.LogCaptureFixture) -> str:
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gap_fill ")]
    return line


def _origins(prompt: str) -> list[tuple[str, str]]:
    """Each menu shape's member and origin, in menu order."""
    return [
        (
            next(dcid for dcid in _NODES if dcid in block),
            re.search(r"origin: ([^\n]*)", block).group(1),
        )
        for block in _menu_blocks(prompt)
    ]


class _SlowOurReads(FakeGraph):
    """Holds the first arc read naming our literacy rate for half a second, so the UN leg
    finishes before our derivation does."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.held = False

    def node_arcs_batch(self, dcids):
        if not self.held and _OURS in dcids:
            self.held = True
            time.sleep(0.5)
        return super().node_arcs_batch(dcids)


# -- pool order --


@pytest.mark.parametrize(
    ("on_first_menu", "cls"),
    [
        pytest.param(False, FakeGraph, id="flag-off"),
        pytest.param(True, FakeGraph, id="flag-on"),
        pytest.param(True, _SlowOurReads, id="flag-on-un-leg-first"),
    ],
)
def test_our_pass_answers_and_the_un_pool_reaches_no_menu(
    caplog, search_settings, on_first_menu: bool, cls: type[FakeGraph]
) -> None:
    """Our literacy rate and the UN's series both have data at Kenya, and the model would
    elect the UN's series were it on the menu. Our pass answers from ours alone, so the UN
    reading neither displaces ours nor stands beside it as a rival. Under
    `SEARCH_UN_ON_FIRST_MENU` our strong shape never takes the UN pool, even when the UN leg
    finishes before our slow derivation does, so the answer does not depend on which read
    came first."""
    search_settings(un_on_first_menu=on_first_menu)
    llm = _ElectLLM(_UN, nominate=(_UN,))

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(data=(_OURS, _UN), cls=cls), llm=llm)

    assert (result.status, _elected(result)) == ("definite", [_OURS])
    assert _named(llm, _UN) == [False]
    assert "un_system_unavailable" not in _codes(result)
    assert _gap_fill_line(caplog) == (
        "gap_fill part=0 status=definite/None pool=ours un_pass=skipped:answered un_ms=0"
        " bind_asked=false un_merged=false doubted=false"
    )


def test_a_merged_menu_seats_our_pool_first(caplog, search_settings) -> None:
    """Our literacy rate clears its floor but is weak, so our pass waits for the UN leg,
    whose GDP per capita series, another measure, scores higher on its own resolver's
    scale. The menu holds both, each showing its pool's origin, ours first
    (`seating.seat_menu`)."""
    search_settings(un_on_first_menu=True)
    llm = _ElectLLM(_OURS)
    graph = _graph(ours=((_OURS, 0.68),), un=((_UN_GDP_PER_CAPITA, 0.85),), data=(_OURS,))

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(graph, llm=llm)

    assert _origins(_menu(llm)) == [
        (_OURS, "our Data Commons"),
        (_UN_GDP_PER_CAPITA, "UN System Data Commons"),
    ]
    assert (result.status, _elected(result)) == ("definite", [_OURS])
    assert _gap_fill_line(caplog).endswith(" un_merged=true doubted=false")


@pytest.mark.parametrize(
    ("on_first_menu", "ours_count"),
    [
        pytest.param(False, 1, id="flag-off"),
        pytest.param(True, 1, id="un-seated"),
        pytest.param(True, 2, id="un-seated-after-two"),
        pytest.param(True, 8, id="un-unseated"),
    ],
)
def test_a_doubted_answer_reaches_the_un_shape_its_menu_left_unseated(
    monkeypatch, search_settings, on_first_menu: bool, ours_count: int
) -> None:
    """Our weak literacy shapes take the menu's seats first (`seating.seat_menu`), and the
    binder rates "none of the shapes" above its pick of ours. A UN shape that took a seat
    was elected on our menu; one that took none reached no binder, so the UN pass runs over
    it and elects it."""
    search_settings(un_on_first_menu=on_first_menu, bind_top_k=8)
    ours = [f"{_OURS}_{k}" for k in range(ours_count)]
    nodes = {
        **_NODES,
        **{dcid: _sv_node(dcid, population_type=dcid, meas_prop="literacyRate") for dcid in ours},
    }
    graph = FakeGraph(
        nodes=nodes,
        obs={f"{dcid}|{_KEN}": [_FACET] for dcid in (*ours, _UN)},
        detect={_QUERY: _leg(*((dcid, 0.68) for dcid in ours))},
        un_detect={_QUERY: _leg((_UN, 0.9))},
        resolve={"Kenya": _KEN},
    )
    bind = GeminiBinder.bind

    async def doubted_on_ours(self, request, *, llm):
        result, usage = await bind(self, request, llm=llm)
        return dataclasses.replace(result, none_first=request.pool == "ours"), usage

    monkeypatch.setattr(GeminiBinder, "bind", doubted_on_ours)

    result = _resolve(graph, llm=_ElectLLM(_UN))

    assert (result.status, _elected(result)) == ("definite", [_UN])


@pytest.mark.parametrize("nominated", [True, False], ids=["nominated", "not-nominated"])
def test_one_measure_in_both_pools_stays_two_readings(search_settings, nominated: bool) -> None:
    """Our literacy rate and the UN's both have data at Kenya and seat side by side, neither
    collapsed into the other. A pool apart they are two measures to the status rule, so
    the UN's is listed beside the elected ours only when the model nominates it."""
    search_settings(un_on_first_menu=True)
    llm = _ElectLLM(_OURS, nominate=(_UN,) if nominated else ())
    graph = _graph(ours=((_OURS, 0.68),), un=((_UN, 0.85),), data=(_OURS, _UN))

    result = _resolve(graph, llm=llm)

    assert [dcid for dcid, _ in _origins(_menu(llm))] == [_OURS, _UN]
    assert _elected(result) == ([_OURS, _UN] if nominated else [_OURS])
    assert result.status == ("candidates" if nominated else "definite")


def test_a_part_ours_leaves_without_data_is_answered_by_the_un_pass(caplog) -> None:
    """Our literacy rate has no data at Kenya. The UN pass elects over the UN pool alone,
    with each shape's origin shown, and its answer stands. The stream's frame, shape and
    step events describe our pass, and the answer carries both passes' model usage."""
    llm = _ElectLLM(_UN)
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(), llm=llm, emit=collect)

    assert (result.status, _elected(result)) == ("definite", [_UN])
    (spec,) = result.specs
    assert [origin.id for origin in spec.resolved.shape.data_commons] == ["un_system"]
    assert _named(llm, _OURS) == [True, False]
    assert _named(llm, _UN) == [False, True]
    (un_block,) = _menu_blocks(llm.prompts[1])
    assert re.search(r"origin: ([^\n]*)", un_block).group(1) == "UN System Data Commons"
    assert [e.step for e in events if isinstance(e, StepEvent)].count("bind") == 1
    assert len([e for e in events if isinstance(e, FrameEvent)]) == 1
    assert not [e for e in events if isinstance(e, SentenceEvent)]
    assert result.llm_usage == {"input_tokens": 2, "output_tokens": 2, "cached_tokens": 0}
    assert "un_system_unavailable" not in _codes(result)
    assert _gap_fill_line(caplog).startswith(
        "gap_fill part=0 status=no_data/no_observations pool=ours un_pass=ran:definite/None "
    )


def test_a_un_pass_that_also_ends_no_data_keeps_our_refusal(caplog) -> None:
    llm = _ElectLLM()

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(data=()), llm=llm)

    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
    assert [sv.ref for sv in result.elected_stat_vars or []] == [_OURS]
    assert _named(llm, _UN) == [False, True]
    assert "un_pass=ran:no_data/" in _gap_fill_line(caplog)


def test_a_un_fallback_answers_where_the_bound_member_of_ours_reads_zero() -> None:
    """ "female population of Togo": ours has data where the question asks through its
    total, and Female, bound on it, names a member with none. Our pass ends no_data, and
    the UN pass answers from the UN's reading."""
    female = SlotBindingDraft(
        axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
    )
    llm = _ElectLLM(_POPULATION, bindings=(female,))
    graph = _graph(
        ours=((_POPULATION, 0.9), (_FEMALE_POPULATION, 0.88)),
        data=(_POPULATION, _UN),
        query="female population",
    )

    result = _resolve(graph, query="female population", llm=llm)

    assert (result.status, _elected(result)) == ("definite", [_UN])
    assert _named(llm, _UN) == [False, True]


@pytest.mark.parametrize(
    ("floors", "scores", "on_first_menu", "menus"),
    [
        pytest.param((0.75, 0.65), (0.7, 0.7), False, [[_UN]], id="ours-under-its-floor"),
        pytest.param((0.65, 0.75), (0.7, 0.7), False, [[_OURS]], id="un-under-its-floor"),
        pytest.param((0.65, 0.65), (0.66, 0.6), True, [[_OURS]], id="merged-un-under-its-floor"),
        pytest.param(
            (0.65, 0.65), (0.66, 0.7), True, [[_OURS, _UN]], id="merged-un-over-its-floor"
        ),
    ],
)
def test_each_pool_meets_its_own_score_floor(
    search_settings,
    floors: tuple[float, float],
    scores: tuple[float, float],
    on_first_menu: bool,
    menus: list[list[str]],
) -> None:
    """Neither resolver's candidate has data at Kenya. Each floor drops only its own pool's:
    under ours, our pass holds no shape and the UN pass elects the UN's; under the UN's,
    the UN pass holds no shape and never binds. A UN shape merged onto our weak pass's menu
    (`SEARCH_UN_ON_FIRST_MENU`) meets the UN's floor there too: below it, it reaches
    neither menu; over it, the merged menu elects it, and the UN pass, holding only the
    shape our pass found without data, does not run."""
    ours_floor, un_floor = floors
    ours_score, un_score = scores
    search_settings(
        unknown_score_threshold=ours_floor,
        un_unknown_score_threshold=un_floor,
        un_on_first_menu=on_first_menu,
    )
    llm = _ElectLLM(_UN)

    _resolve(_graph(ours=((_OURS, ours_score),), un=((_UN, un_score),), data=()), llm=llm)

    assert [[dcid for dcid in (_OURS, _UN) if dcid in prompt] for prompt in llm.prompts] == menus


def test_a_per_capita_query_lists_each_pools_shape_without_a_denominator_as_not_electable() -> None:
    """Neither our GDP level nor the UN series, which states no denominator, serves a
    per-person form, so each pass's menu is empty and its pool's shape is not electable. A
    model that names neither as nearest leaves the part refusing on the form, flagging
    nothing."""
    llm = _ElectLLM()
    graph = _graph(
        ours=((_GDP, 0.9),),
        un=((_UN_GDP_PER_CAPITA, 0.85),),
        data=(_GDP, _UN_GDP_PER_CAPITA),
        query=_PER_CAPITA_QUERY,
    )

    result = _resolve(graph, query=_PER_CAPITA_QUERY, llm=llm)

    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    assert not result.flagged_match
    assert ["CANDIDATE SHAPES (0 total)" in prompt for prompt in llm.prompts] == [True, True]
    assert _named(llm, _UN_GDP_PER_CAPITA) == [False, True]


# -- when the UN pass does not run --


@pytest.mark.parametrize(
    ("entities", "reason"),
    [
        pytest.param(("Atlantis",), "entity_not_resolved", id="unresolved-entity"),
        pytest.param(("Kenya", "Tanzania"), "entity_set_too_large", id="places-over-the-bound"),
    ],
)
def test_an_entity_terminal_runs_no_un_pass(caplog, search_settings, entities, reason) -> None:
    """A part that ends on the request's entities is not filled from the UN's candidates,
    which hold data at Kenya. Two named places are over a bound of one."""
    search_settings(max_child_places=1)
    llm = _ElectLLM()

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(), entities=entities, llm=llm)

    assert (result.status, result.no_data_reason) == ("no_data", reason)
    assert not any(_named(llm, _UN))
    assert "un_pass=skipped:entity " in _gap_fill_line(caplog)


def test_a_part_excluding_a_source_keeps_the_tables_the_index_does_not_declare() -> None:
    """A part that excludes a source keeps the CRS shape, a declared table its scope still
    allows, and the UN and base series the index does not declare: the scoped read serves
    none of an excluded source's data for them."""
    crs = _crs_shape()
    un = _shape(_UN, population_type="undata/unicef/SERIES-ED_LR", pool="un_system")
    base = _shape(_OURS, population_type="Person", pool="ours")
    excluding = SourceScope(
        allowed=frozenset({"provenance/OECDCreditorReportingSystem", "dc/base/WDI", "undata/p/X"}),
        named=False,
    )

    kept = _dataset_candidates([crs, un, base], excluding)

    assert [shape.shape_id for shape in kept] == [crs.shape_id, un.shape_id, base.shape_id]


@pytest.mark.parametrize("on_first_menu", [False, True], ids=["flag-off", "flag-on"])
def test_a_scope_allowing_no_un_provenance_never_starts_the_un_leg(
    caplog, search_settings, on_first_menu: bool
) -> None:
    search_settings(un_on_first_menu=on_first_menu)
    leg = _Leg()
    graph = _graph(cls=_PendingUnGraph, leg=leg)
    llm = _ElectLLM()
    named = SourceScope(allowed=frozenset({"dc/base/WDI"}), named=True)

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = asyncio.run(asyncio.wait_for(_resolving(graph, llm=llm, source_scope=named), 5))

    assert leg.waiters == 0
    assert result.status == "no_data"
    assert "un_system_unavailable" not in _codes(result)
    assert not any(_named(llm, _UN))
    assert "un_pass=skipped:source_scope " in _gap_fill_line(caplog)


# -- the UN leg beside our pass --


class _Leg(Future):
    """A running UN resolve that the test settles; counts the waiters it gains."""

    def __init__(self) -> None:
        super().__init__()
        self.set_running_or_notify_cancel()
        self.waiters = 0

    def add_done_callback(self, fn: Callable[[Future], object]) -> None:
        self.waiters += 1
        super().add_done_callback(fn)

    def settle_after(self, seconds: float, un: tuple[DetectedSV, ...] | None) -> None:
        timer = threading.Timer(seconds, self.set_result, args=(un,))
        timer.daemon = True
        timer.start()


class _PendingUnGraph(FakeGraph):
    """Hands out `leg` as every detection's UN leg, as the live client does while the UN
    resolve is still running."""

    def __init__(self, *, leg: _Leg, **kwargs) -> None:
        super().__init__(**kwargs)
        self.leg = leg

    def detect_svs(self, query: str) -> Detection:
        return dataclasses.replace(super().detect_svs(query), un=self.leg)


def _un_leg(dcid: str) -> tuple[DetectedSV, ...]:
    return (DetectedSV(dcid=dcid, score=0.85, type_of="StatisticalVariable", sentence=dcid),)


@pytest.mark.parametrize("on_first_menu", [False, True], ids=["flag-off", "flag-on"])
def test_our_answer_does_not_wait_for_a_running_un_resolve(
    search_settings, on_first_menu: bool
) -> None:
    """The UN resolve never settles. Our pass answers from its strong shape alone without
    waiting out the UN leg's budget, warns nothing, and leaves no task behind."""
    search_settings(un_first_pass_wait_s=3.0, un_on_first_menu=on_first_menu)
    leg = _Leg()
    llm = _ElectLLM()
    graph = _graph(cls=_PendingUnGraph, leg=leg, data=(_OURS,))

    async def answer() -> tuple[RegionResult, set[asyncio.Task]]:
        async with asyncio.timeout(5):
            result = await _resolving(graph, llm=llm)
        await asyncio.sleep(0)
        return result, asyncio.all_tasks() - {asyncio.current_task()}

    started = time.monotonic()
    result, leftover = asyncio.run(answer())

    assert time.monotonic() - started < 2.0
    assert result.status == "definite"
    assert [dcid for dcid, _ in _origins(_menu(llm))] == [_OURS]
    assert "un_system_unavailable" not in _codes(result)
    assert not leg.done() and not leg.cancelled()
    assert leftover == set()


@pytest.mark.parametrize("leg_state", ["late", "failed"])
def test_a_weak_shape_waits_for_the_un_leg_and_an_answer_warns_on_a_miss(
    caplog, search_settings, leg_state: str
) -> None:
    """Under `SEARCH_UN_ON_FIRST_MENU` our pass's only shape is weak, so it waits for the
    UN leg: out to the bound for a resolve that never settles, or not at all for one that
    failed. The menu is ours alone, and the answer it serves carries
    `un_system_unavailable`."""
    wait = 0.3
    search_settings(un_first_pass_wait_s=wait, un_on_first_menu=True)
    leg = _Leg()
    llm = _ElectLLM()
    graph = (
        _graph(cls=_PendingUnGraph, leg=leg, ours=((_OURS, 0.68),), data=(_OURS,))
        if leg_state == "late"
        else _graph(un=None, ours=((_OURS, 0.68),), data=(_OURS,))
    )
    started = time.monotonic()

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = asyncio.run(asyncio.wait_for(_resolving(graph, llm=llm), 5))

    took = time.monotonic() - started
    assert took < 2.0
    if leg_state == "late":
        assert took >= wait
    assert (result.status, _elected(result)) == ("definite", [_OURS])
    assert [dcid for dcid, _ in _origins(_menu(llm))] == [_OURS]
    assert "un_system_unavailable" in _codes(result)
    assert "un_pass=skipped:un_unavailable " in _gap_fill_line(caplog)
    assert _gap_fill_line(caplog).endswith(" un_merged=false doubted=false")


def test_a_un_resolve_in_within_the_wait_fills_a_part_without_data(search_settings) -> None:
    search_settings(un_first_pass_wait_s=2.0)
    leg = _Leg()
    leg.settle_after(0.1, _un_leg(_UN))

    result = _resolve(_graph(cls=_PendingUnGraph, leg=leg), llm=_ElectLLM(_UN))

    assert _elected(result) == [_UN]
    assert "un_system_unavailable" not in _codes(result)


@pytest.mark.parametrize("ours_have_data", [True, False], ids=["answered", "no_data"])
def test_a_slow_un_resolve_warns_only_a_part_without_data(
    search_settings, ours_have_data: bool
) -> None:
    """The UN resolve never settles. Past the wait a part without data keeps our refusal
    and warns; the resolve runs on unawaited."""
    search_settings(un_first_pass_wait_s=0.2)
    leg = _Leg()
    data = (_OURS, _UN) if ours_have_data else (_UN,)
    llm = _ElectLLM(_UN)
    started = time.monotonic()

    result = asyncio.run(
        asyncio.wait_for(_resolving(_graph(cls=_PendingUnGraph, leg=leg, data=data), llm=llm), 5)
    )

    assert time.monotonic() - started < 2.0
    assert result.status == ("definite" if ours_have_data else "no_data")
    assert not any(_named(llm, _UN))
    assert ("un_system_unavailable" in _codes(result)) is not ours_have_data
    assert not leg.done() and not leg.cancelled()


@pytest.mark.parametrize("ours_have_data", [True, False], ids=["answered", "no_data"])
def test_a_failed_un_resolve_warns_only_a_part_without_data(ours_have_data: bool) -> None:
    data = (_OURS, _UN) if ours_have_data else (_UN,)

    result = _resolve(_graph(un=None, data=data))

    assert result.status == ("definite" if ours_have_data else "no_data")
    assert ("un_system_unavailable" in _codes(result)) is not ours_have_data


class _FailingUnReads(FakeGraph):
    """Raises a UN deployment outage from every arc read that names a UN variable."""

    def node_arcs_batch(self, dcids):
        if any(dcid.startswith("undata/") for dcid in dcids):
            raise GraphInfraError("UN deployment unavailable")
        return super().node_arcs_batch(dcids)


class _StalledUnReads(FakeGraph):
    """Holds every arc read that names a UN variable until `release` is set."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.release = threading.Event()

    def node_arcs_batch(self, dcids):
        if any(dcid.startswith("undata/") for dcid in dcids):
            self.release.wait(10)
        return super().node_arcs_batch(dcids)


@pytest.mark.parametrize("cls", [_FailingUnReads, _StalledUnReads], ids=["fails", "stalls"])
def test_a_failed_or_stalled_un_shape_read_keeps_our_refusal_with_a_warning(
    search_settings, cls: type[FakeGraph]
) -> None:
    """The UN resolve is in at once and its shape derivation fails or stalls. One wait
    bounds the resolve and the derivation together, so a stalled read leaves our refusal
    standing inside it."""
    search_settings(un_first_pass_wait_s=0.2)
    llm = _ElectLLM(_UN)
    graph = _graph(cls=cls)

    async def answer() -> tuple[RegionResult, float]:
        # The stalled read is released once the part answers, so the loop's shutdown does
        # not wait on its thread.
        started = time.monotonic()
        try:
            return await asyncio.wait_for(_resolving(graph, llm=llm), 5), time.monotonic() - started
        finally:
            if isinstance(graph, _StalledUnReads):
                graph.release.set()

    result, took = asyncio.run(answer())

    assert took < 2.0
    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
    assert not any(_named(llm, _UN))
    assert "un_system_unavailable" in _codes(result)


class _UnservedUnProbe(FakeGraph):
    """The first availability probe naming a UN variable, the UN leg's, names Kenya as a
    place the UN front door refuses and then fails."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.probed = False

    def observation_facets_batch(self, dcids, *args, **kwargs):
        if not self.probed and any(dcid.startswith("undata/") for dcid in dcids):
            self.probed = True
            note_un_blocked_places([_KEN])
            raise GraphInfraError("UN deployment unavailable")
        return super().observation_facets_batch(dcids, *args, **kwargs)


@pytest.mark.parametrize("ours_have_data", [True, False], ids=["answered", "no_data"])
def test_the_un_legs_unserved_reads_mark_only_an_answer_the_un_pass_ran_for(
    search_settings, ours_have_data: bool
) -> None:
    """The UN leg's probe names a refused place and fails. An answer our pass serves never
    read the UN pool, so it carries neither `un_system_unavailable` nor `degraded`; once our
    pass ends without data the UN pass runs over the leg's shapes and carries both."""
    search_settings(un_first_pass_wait_s=2.0)
    data = (_OURS, _UN) if ours_have_data else (_UN,)

    result = _resolve(_graph(cls=_UnservedUnProbe, data=data), llm=_ElectLLM(_UN))

    assert result.status == "definite"
    assert _elected(result) == [_OURS if ours_have_data else _UN]
    assert ("un_system_unavailable" in _codes(result)) is not ours_have_data
    assert result.degraded is not ours_have_data


_SHAPE_STAGES = ("shape_arcs", "shape_probe", "shape_labels", "shape_groups", "shape_admission")


def test_the_un_legs_derivation_stays_out_of_our_shape_timing(monkeypatch, search_settings) -> None:
    """The UN leg's derivation records an hour of `shape_arcs` before ours starts deriving.
    Each derivation records onto its own collector, so our part's shape stages are ours."""
    search_settings(un_first_pass_wait_s=2.0)
    real_derive = phase_intake.derive_shapes
    un_recorded = threading.Event()

    def derive(**kwargs):
        if any(dcid.startswith("undata/") for dcid in kwargs["confirmed_svs"]):
            record_elapsed("shape_arcs", 3600.0)
            un_recorded.set()
        else:
            assert un_recorded.wait(2)
        return real_derive(**kwargs)

    monkeypatch.setattr(phase_intake, "derive_shapes", derive)

    result = _resolve(_graph(data=(_OURS, _UN)), llm=_ElectLLM(_UN))

    assert _elected(result) == [_OURS]
    assert set(_SHAPE_STAGES) <= set(result.timing_by_step)
    assert result.timing_by_step["shape_arcs"] < 3_600_000


# -- the UN pass's own failures and bounds --


def _failing_un_pass(monkeypatch, error: Exception) -> None:
    """Our pass runs; the UN pass raises `error`."""
    run_phases = pipeline._run_phases

    async def run(state: ResolveState) -> RegionResult:
        if state.pool == "un_system":
            raise error
        return await run_phases(state)

    monkeypatch.setattr(pipeline, "_run_phases", run)


@pytest.mark.parametrize(
    ("error", "warned"),
    [
        pytest.param(GraphInfraError("UN deployment unavailable"), True, id="upstream"),
        pytest.param(RuntimeError("a bug"), False, id="defect"),
    ],
)
def test_a_failed_un_pass_keeps_our_refusal(monkeypatch, caplog, error, warned) -> None:
    """An upstream failure warns that the UN pool could not be read; a defect is logged
    with its traceback and warns nothing."""
    _failing_un_pass(monkeypatch, error)

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph())

    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
    assert [sv.ref for sv in result.elected_stat_vars or []] == [_OURS]
    assert ("un_system_unavailable" in _codes(result)) is warned
    assert f"un_pass=error:{type(error).__name__} " in _gap_fill_line(caplog)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert bool(errors) is not warned


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(UpstreamVersionChangedError("rate table moved"), id="rate_table_moved"),
        pytest.param(DeclaredDefaultUnsupportedError("declared default"), id="declared_default"),
    ],
)
def test_a_un_pass_refusing_the_request_propagates(monkeypatch, error) -> None:
    """A refusal of the whole request on every door raised in the UN pass reaches the
    caller in place of our `no_data`."""
    _failing_un_pass(monkeypatch, error)

    with pytest.raises(type(error)):
        _resolve(_graph())


def test_a_deadline_spent_before_the_un_pass_keeps_our_refusal_with_a_warning(caplog) -> None:
    llm = _ElectLLM()
    deadline = time.monotonic() + pipeline._UN_PASS_RESERVE_S - 0.5

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(), llm=llm, deadline=deadline)

    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
    assert "un_system_unavailable" in _codes(result)
    assert not any(_named(llm, _UN))
    assert "un_pass=skipped:" in _gap_fill_line(caplog)


def _gap_result(
    status: str,
    *,
    reason: str = "no_observations",
    unevaluated: bool = False,
    alternate: Spec | None = None,
    usage: int = 0,
) -> RegionResult:
    warnings = (
        (Warning(code="alternates_unevaluated", severity="warn", message="-"),)
        if unevaluated
        else ()
    )
    return RegionResult(
        variable_text=_QUERY,
        status=status,
        specs=(),
        no_data_reason=reason if status == "no_data" else None,
        warnings=warnings,
        timing_by_step={},
        alternate_match=alternate,
        llm_usage={"input_tokens": usage, "output_tokens": usage, "cached_tokens": 0},
    )


def _fill(
    ours: RegionResult,
    un: RegionResult,
    *,
    ours_doubted: bool = False,
    un_doubted: bool = False,
    party_words_off: bool = True,
    elected: ShapeDraft | None = None,
    missed: str | None = None,
    merged: bool = False,
    seated: bool = True,
    trailing: dict[str, bool] | None = None,
    un_score: float | None = None,
    un_elected: bool | None = None,
) -> tuple[RegionResult, list[ShapeDraft] | None]:
    """`pipeline._fill_gap` over our pass's `ours`, elected on `elected`, with a UN pass
    that returns `un`, and the shapes the UN pass ran over, None when it did not run.
    `ours_doubted` and `un_doubted` say each pass's binder rated "none of the shapes"
    first (`BindResult.none_first`), `missed` is why the UN leg's pool is missing, and
    `merged` puts the UN pool on our pass's menu (`ResolveState.un_pool`), where its shape
    takes a seat when `seated` (`ResolveState.capped_shapes`). Given
    `trailing`, each pass's result holds a pending observations task, and `trailing` maps
    "ours" and "un" to whether `_fill_gap` cancelled it. `un_score` is the measured
    score of the shape the UN pass elects; it elects one when `un_elected`, which defaults
    to whether `un_score` is given."""
    ran: list[list[ShapeDraft]] = []

    async def fill() -> RegionResult:
        nonlocal ours, un
        tasks: dict[str, asyncio.Task[None]] = {}
        if trailing is not None:
            tasks = {side: asyncio.create_task(asyncio.sleep(60)) for side in ("ours", "un")}
            ours = dataclasses.replace(ours, observations_task=tasks["ours"])
            un = dataclasses.replace(un, observations_task=tasks["un"])

        async def shapes() -> UnPool:
            if missed is not None:
                return UnPool(shapes=[], missed=missed)
            return UnPool(shapes=[_shape(_UN, population_type=_UN, pool="un_system")])

        merged_shape = _shape(_UN, population_type=_UN, pool="un_system")
        first = SimpleNamespace(
            shape_draft=elected,
            capped_shapes=[merged_shape] if merged and seated else [],
            scope=None,
            source_code=None,
            scope_settled=False,
            rcl=None,
            variable_index=0,
            bind_result=None,
            bind_decision=SimpleNamespace(none_first=ours_doubted),
            party_words_off=party_words_off,
            un_leg=UnLeg(task=asyncio.create_task(shapes()), ends=time.monotonic() + 1),
            un_pool=UnPool(shapes=[merged_shape]) if merged else None,
        )
        un_state = SimpleNamespace(
            rcl=None,
            shapes_found=[],
            shape_draft=(
                SimpleNamespace(measured_score=un_score)
                if (un_score is not None if un_elected is None else un_elected)
                else None
            ),
            bind_decision=SimpleNamespace(none_first=un_doubted),
        )

        async def run(state: SimpleNamespace) -> RegionResult:
            ran.append(state.shapes_found)
            return un

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pipeline, "_run_phases", run)
            result = await pipeline._fill_gap(
                ours,
                first=cast(ResolveState, first),
                deadline=None,
                un_state=lambda _bound: cast(ResolveState, un_state),
            )
        if trailing is not None:
            trailing.update({side: task.cancelled() for side, task in tasks.items()})
        return result

    result = asyncio.run(fill())
    return result, ran[0] if ran else None


def test_a_gap_filled_answer_states_an_alternate_either_pass_left_unevaluated() -> None:
    """A pass that left an alternate unevaluated states `alternates_unevaluated`; the
    answer that stands carries it from whichever pass left it, so it stays out of the
    cache."""
    filled, _ = _fill(_gap_result("no_data", unevaluated=True), _gap_result("definite"))
    kept, _ = _fill(_gap_result("no_data"), _gap_result("no_data", unevaluated=True))

    assert filled.status == "definite" and filled.alternates_left_out
    assert kept.status == "no_data" and kept.alternates_left_out


@pytest.mark.parametrize("ours_offers", [False, True], ids=["ours_bare", "ours_offers"])
def test_a_un_refusal_offering_an_alternate_stands_over_a_bare_refusal_of_ours(
    ours_offers: bool,
) -> None:
    """ "measles immunization coverage in Pakistan": our pass's measure has no data at
    Pakistan and nominates nothing; the UN pass's pick has none either, and its nominee,
    with data, rides on its refusal. That refusal stands, carrying both passes' model usage
    and an alternate our pass left unevaluated. A refusal of ours that offers its own
    alternate stands."""
    un_alternate = cast(Spec, object())
    our_alternate = cast(Spec, object()) if ours_offers else None
    ours = _gap_result("no_data", unevaluated=True, alternate=our_alternate, usage=1)
    un = _gap_result("no_data", alternate=un_alternate, usage=2)

    chosen, _ = _fill(ours, un)

    assert chosen.alternate_match is (our_alternate if ours_offers else un_alternate)
    assert chosen.alternates_left_out
    assert chosen.llm_usage == {"input_tokens": 3, "output_tokens": 3, "cached_tokens": 0}


@pytest.mark.parametrize(
    ("ours", "un", "ours_doubted", "un_doubted", "party_words_off", "replaced"),
    [
        # "refugee population in Turkey" on Jev: our Person count is elected though Jev
        # rates "none of the shapes" above it; the UN pass answers with refugees by host
        # country, whose own election Jev does not doubt.
        ("definite", "definite", True, False, True, True),
        ("candidates", "candidates", True, False, True, True),
        # Both binders doubt their picks, so the UN answer has no better claim.
        ("definite", "definite", True, True, True, False),
        # A UN refusal, even one offering an alternate, leaves our answer standing.
        ("definite", "no_data", True, False, True, False),
        # An answer its binder does not doubt never waits for the UN leg.
        ("definite", "definite", False, False, True, None),
        # On the resolution with party words a UN answer could bind no party.
        ("definite", "definite", True, False, False, None),
    ],
    ids=[
        "doubted-replaced",
        "doubted-candidates-replaced",
        "both-doubted",
        "un-refuses",
        "undoubted",
        "party-words",
    ],
)
def test_a_doubted_answer_of_ours_gives_way_to_an_undoubted_un_answer(
    caplog: pytest.LogCaptureFixture,
    ours: str,
    un: str,
    ours_doubted: bool,
    un_doubted: bool,
    party_words_off: bool,
    replaced: bool | None,
) -> None:
    """An answer of ours whose binder rated "none of the shapes" above its pick
    (`BindResult.none_first`) runs the UN pass, and the UN answer replaces it when it is
    `definite` or `candidates` and its own binder did not doubt it. `replaced` None marks
    an answer the UN pass never ran for."""
    ours_result = _gap_result(ours, usage=1)
    un_result = _gap_result(un, alternate=cast(Spec, object()), usage=2)

    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        chosen, ran = _fill(
            ours_result,
            un_result,
            ours_doubted=ours_doubted,
            un_doubted=un_doubted,
            party_words_off=party_words_off,
        )

    if replaced is None:
        assert ran is None and chosen is ours_result
        assert "un_pass=skipped:answered" in _gap_fill_line(caplog)
        return
    assert ran is not None
    assert chosen.status == (un if replaced else ours)
    assert (chosen.alternate_match is un_result.alternate_match) == replaced
    assert chosen.llm_usage == {"input_tokens": 3, "output_tokens": 3, "cached_tokens": 0}
    assert _gap_fill_line(caplog).endswith(" doubted=true")


def test_a_doubted_answer_excludes_its_own_series_from_the_un_pool() -> None:
    """Our pass can elect a UN series it recalls under a superseded variable, and the UN
    pass would elect it again and replace our answer with itself, so a doubted answer's
    elected series leaves the UN pool."""
    elected = _shape(_UN, population_type=_UN, pool="ours")
    ours = _gap_result("definite")

    chosen, ran = _fill(ours, _gap_result("definite"), ours_doubted=True, elected=elected)
    kept, ran_undoubted = _fill(ours, _gap_result("definite"), elected=elected)

    assert ran is None and chosen is ours
    assert ran_undoubted is None and kept is ours


def test_a_doubted_answer_whose_un_leg_missed_states_it() -> None:
    """A doubted answer the UN pass could have replaced, whose UN leg failed or was late,
    carries `un_system_unavailable`, which keeps it out of the response cache; an
    undoubted answer does not."""
    ours = _gap_result("definite")

    doubted, _ = _fill(ours, _gap_result("definite"), ours_doubted=True, missed="late")
    undoubted, _ = _fill(ours, _gap_result("definite"), missed="late")

    assert "un_system_unavailable" in _codes(doubted)
    assert "un_system_unavailable" not in _codes(undoubted)


@pytest.mark.parametrize(
    ("ours_reason", "un_reason", "un_elected", "un_score", "un_stands"),
    [
        # "government debt as a share of GDP in Japan": our menu matches nothing, and the
        # UN pass matches the IMF series and finds no Japan values.
        ("variable_not_resolved", "no_observations", True, 0.9, True),
        # "measles incidence rate Kenya": the UN pass matches a burden-of-disease rate only
        # weakly, which says no more than ours matching nothing.
        ("variable_not_resolved", "no_observations", True, 0.7, False),
        # A UN pass that elected no shape, or one with no measured score, matched nothing
        # it can show, whatever its reason.
        ("variable_not_resolved", "no_observations", False, None, False),
        ("variable_not_resolved", "denominator_not_available", False, None, False),
        ("variable_not_resolved", "no_observations", True, None, False),
        ("variable_not_resolved", "variable_not_resolved", True, 0.9, False),
        # Between two refusals that both matched a measure ours stands: D6's
        # '"climate" is not a distinction this data records' over a UN series with no data.
        ("qualifier_not_bound", "no_observations", True, 0.9, False),
        ("no_observations", "qualifier_not_bound", True, 0.9, False),
        # The UN pass mints these on its elected shape's places, so it matched a variable,
        # though `NO_DATA_RESOLUTION_ORDER` ranks both before `variable_not_resolved`.
        ("variable_not_resolved", "entity_set_too_large", True, 0.9, True),
        ("variable_not_resolved", "entity_role_not_admitted", True, 0.9, True),
    ],
)
def test_a_un_refusal_on_a_matched_variable_stands_over_ours_matching_none(
    ours_reason: str, un_reason: str, un_elected: bool, un_score: float | None, un_stands: bool
) -> None:
    """Of two refusals, the UN pass's stands when ours matched no variable and the UN pass
    elected a shape whose measured score is present and not weak
    (`_representative_score_is_weak`), since its reason says what the data lacks."""
    ours = _gap_result("no_data", reason=ours_reason)
    un = _gap_result("no_data", reason=un_reason)

    chosen, _ = _fill(ours, un, un_score=un_score, un_elected=un_elected)

    assert chosen.no_data_reason == (un_reason if un_stands else ours_reason)


def test_a_un_refusal_before_election_leaves_ours_matching_none(caplog) -> None:
    """Our pass recalls nothing for "literacy rate", and the UN pass's one series counts
    people, so it refuses ``denominator_not_available`` with no shape elected. That says no
    more about the data than ours matching nothing, so ours stands."""
    nodes = {**_NODES, _UN: _sv_node(_UN, population_type=_UN, meas_prop="count")}
    graph = _graph(ours=(), un=((_UN, 0.7),), data=(_UN,), nodes=nodes)

    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        result = _resolve(graph, llm=_ElectLLM(ask_without=True))

    assert "un_pass=ran:no_data/denominator_not_available " in _gap_fill_line(caplog)
    assert (result.status, result.no_data_reason) == ("no_data", "variable_not_resolved")


@pytest.mark.parametrize(
    ("ours", "un", "ours_doubted", "un_doubted", "dropped"),
    [
        # Our doubted answer gives way to the UN answer.
        ("definite", "definite", True, False, "ours"),
        # Both binders doubt, so the UN answer is dropped.
        ("definite", "definite", True, True, "un"),
        ("no_data", "definite", False, False, "ours"),
    ],
    ids=["ours-replaced", "both-doubted", "gap-filled"],
)
def test_the_dropped_answer_of_a_gap_fill_cancels_its_trailing_tasks(
    ours: str, un: str, ours_doubted: bool, un_doubted: bool, dropped: str
) -> None:
    """Both passes stream their trailing reads, and the answer that does not stand names a
    Spec the response never shows, so its reads are cancelled and the standing one's run."""
    trailing: dict[str, bool] = {}

    _fill(
        _gap_result(ours),
        _gap_result(un),
        ours_doubted=ours_doubted,
        un_doubted=un_doubted,
        trailing=trailing,
    )

    assert trailing == {"ours": dropped == "ours", "un": dropped == "un"}


@pytest.mark.parametrize("seated", [True, False], ids=["seated", "unseated"])
def test_a_doubted_answer_on_a_merged_menu_runs_the_un_pass_over_its_unseated_shapes(
    caplog: pytest.LogCaptureFixture, seated: bool
) -> None:
    """A menu that seated the UN pool merged onto it already offered the binder the UN
    shapes it rated "none" above, so a UN pass would elect again among them. A UN shape our
    shapes kept off the menu reached no binder, so the UN pass runs over it. A merged
    `no_data` part runs the UN pass over the whole pool either way."""
    ours = _gap_result("definite")

    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        chosen, ran = _fill(
            ours, _gap_result("definite"), ours_doubted=True, merged=True, seated=seated
        )
    _, refused_ran = _fill(
        _gap_result("no_data"), _gap_result("definite"), merged=True, seated=seated
    )

    if seated:
        assert ran is None and chosen is ours
        assert "un_pass=skipped:merged " in _gap_fill_line(caplog)
    else:
        assert [shape.shape_id for shape in ran or ()] == [_UN]
        assert chosen.status == "definite" and chosen is not ours
    assert [shape.shape_id for shape in refused_ran or ()] == [_UN]


def _shape(dcid: str, *, population_type: str, pool: Pool) -> ShapeDraft:
    return ShapeDraft(
        shape_id=dcid,
        label=dcid,
        pop_type_dcid=population_type,
        meas_prop_dcid="value",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={dcid: {}},
        pool=pool,
    )


# -- "Current health expenditure as share of GDP", Kenya --

_HEALTH_QUERY = "Current health expenditure as share of GDP"
_EDUCATION_SHARE = "worldBank/SE_XPD_TOTL_GD_ZS"
_MILITARY_SHARE = "worldBank/MS_MIL_XPND_GD_ZS"
_WHO_HEALTH_SHARE = "undata/who/GHED_SH.HE_CATEGORY--CHE__REF_EXP_CAT--GDP"
_HEALTH = QualifierMatch(
    code_system="expenditureType",
    code_dcid="HealthcareExpenditure",
    label="Healthcare expenditure",
    source="label",
    matched_text="health expenditure",
)


def _health_graph() -> FakeGraph:
    """Our education share of GDP holds `expenditureType`; our military spending share of
    GDP holds no slot; WHO's current health expenditure share states no denominator. All
    three have data at Kenya."""
    nodes = {
        _KEN: {"label": "Kenya", "type": "Country"},
        _GDP: _sv_node(_GDP, population_type="EconomicActivity", meas_prop="amount"),
        _EDUCATION_SHARE: _sv_node(
            _EDUCATION_SHARE,
            population_type="EconomicActivity",
            meas_prop="amount",
            constraints={"expenditureType": "EducationExpenditure"},
            denominator=_GDP,
        ),
        _MILITARY_SHARE: _sv_node(
            _MILITARY_SHARE,
            population_type="MilitaryExpenditure",
            meas_prop="amount",
            denominator=_GDP,
        ),
        _WHO_HEALTH_SHARE: _sv_node(
            _WHO_HEALTH_SHARE,
            population_type="undata/who/GHED_SH",
            meas_prop="value",
        ),
    }
    nodes[_GDP]["arcs"]["name"] = {"nodes": [{"value": "Nominal GDP"}]}
    return FakeGraph(
        nodes=nodes,
        obs={
            f"{dcid}|{_KEN}": [_FACET]
            for dcid in (_EDUCATION_SHARE, _MILITARY_SHARE, _WHO_HEALTH_SHARE)
        },
        detect={_HEALTH_QUERY: _leg((_EDUCATION_SHARE, 0.82), (_MILITARY_SHARE, 0.74))},
        un_detect={_HEALTH_QUERY: _leg((_WHO_HEALTH_SHARE, 0.81))},
        resolve={"Kenya": _KEN},
    )


def test_current_health_expenditure_share_of_gdp_elects_who_on_the_un_pass(caplog) -> None:
    """The education share cannot bind HealthcareExpenditure, so it is refused, and the
    model asks rather than elect military spending for health spending, so our pass ends
    no_data. The model elects WHO's series on the UN pass with GDP as the denominator;
    WHO's series states no denominator, so the answer warns that GDP, in the question's own
    words, is unverified."""
    llm = _ElectLLM(_WHO_HEALTH_SHARE, ask_without=True, denominator=_GDP, base_words="GDP")
    sidecar = CodedQualifier(
        phrase="health expenditure",
        spans=(),
        matches=(_HEALTH,),
    )

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_health_graph(), query=_HEALTH_QUERY, llm=llm, qualifiers=(sidecar,))

    ours, un = llm.prompts
    assert _EDUCATION_SHARE not in ours.split("NOT ELECTABLE")[0]
    assert _WHO_HEALTH_SHARE in un
    assert result.status == "definite"
    assert _elected(result) == [_WHO_HEALTH_SHARE]
    (warning,) = [w for w in result.warnings if w.code == "denominator_unverified"]
    assert 'whether it divides by "GDP"' in warning.message
    # The UN pass's answer stands, and the line still records that our bind asked.
    assert _gap_fill_line(caplog).endswith(" bind_asked=true un_merged=false doubted=false")
