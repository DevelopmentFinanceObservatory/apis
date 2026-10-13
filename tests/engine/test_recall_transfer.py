"""Our recall, UN data: a superseded base variable our resolver recalls, or one of our Topics
expands to, stands in the first pass as its successor UN variable at our rank and score. The
first pass reads our resolver's candidates alone and the second pass the UN resolver's, as it
does without a successor table."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from concurrent.futures import Future

import pytest

from origin_catalogue import BASE, UN_SYSTEM, CatalogueSnapshot
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import DetectedSV, Detection, admit_candidates, standing_dcid
from search.engine.quantity_read import QuantityOutput
from search.engine.region import phase_intake
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.retrieval.topics import expand_topics
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_KEN = "country/KEN"
_QUERY = "literacy rate"
_WDI = "dc/base/WorldDevelopmentIndicators"
_OURS = "Percent_Person_Literate"
_SDG_ROW = "undata/p/SDG"
_WHO_ROW = "undata/p/WHO"
_SDG = "undata/sdg/SE_ADT_LITR"
_WHO = "undata/who/LITERACY"
_SDG_SERIES = "undata/sdg/SERIES-SE_ADT_LITR"
_WHO_SERIES = "undata/who/SERIES-LITERACY"
_BASE = "sdg/SE_ADT_LITR"
_BASE_VARIANT = "sdg/SE_ADT_LITR.SEX--F"
_BASE_ALONE = "sdg/SE_GONE"
_OURS_LOW = "Percent_Person_EnrolledInSchool"
_SDG_FEMALE = "undata/sdg/SE_ADT_LITR.SEX--F"
_TOPIC = "dc/topic/sdg_4.6.1"
_TOPIC_ALONE = "dc/topic/sdg_4.6.2"
_TOPIC_BASE = "dc/topic/sdg_4.6"

_FACET = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}


def _sv_node(dcid: str, *, population_type: str, meas_prop: str) -> dict:
    return {
        "label": dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": dcid}]},
            "populationType": {"nodes": [{"dcid": population_type}]},
            "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": []},
        },
    }


def _topic_node(*leaves: str) -> dict:
    return {
        "label": "Literacy",
        "type": "Topic",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "Topic"}]},
            "relevantVariable": {
                "nodes": [{"dcid": leaf, "types": ["StatisticalVariable"]} for leaf in leaves]
            },
        },
    }


_NODES = {
    _KEN: {"label": "Kenya", "type": "Country"},
    _OURS: _sv_node(_OURS, population_type="Person", meas_prop="literacyRate"),
    _OURS_LOW: _sv_node(_OURS_LOW, population_type="Person", meas_prop="enrollmentRate"),
    _SDG: _sv_node(_SDG, population_type=_SDG_SERIES, meas_prop="value"),
    _SDG_FEMALE: _sv_node(_SDG_FEMALE, population_type=_SDG_SERIES, meas_prop="value"),
    _WHO: _sv_node(_WHO, population_type=_WHO_SERIES, meas_prop="value"),
    _BASE: _sv_node(_BASE, population_type="SDG_SE_ADT_LITR", meas_prop="value"),
    _BASE_VARIANT: _sv_node(_BASE_VARIANT, population_type="SDG_SE_ADT_LITR", meas_prop="value"),
    _BASE_ALONE: _sv_node(_BASE_ALONE, population_type="SDG_SE_GONE", meas_prop="value"),
    _TOPIC: _topic_node(_BASE, _BASE_VARIANT, _BASE_ALONE),
    _TOPIC_ALONE: _topic_node(_BASE_ALONE),
    _TOPIC_BASE: _topic_node(_BASE),
}


def _catalogue(*, un_system_enabled: bool = True, transfer: bool = True) -> CatalogueSnapshot:
    """`_BASE` and its variant are superseded and stand as `_SDG`; `_BASE_ALONE` is
    superseded with no successor. Without `transfer` no variable has a successor."""
    return restrictive_catalogue(
        provenances=[
            provenance_row(_WDI, BASE),
            provenance_row(_SDG_ROW, UN_SYSTEM, segment="sdg"),
            provenance_row(_WHO_ROW, UN_SYSTEM, segment="who"),
        ],
        base_variables={_OURS: ([_WDI], [])},
        withheld_variables=[_BASE, _BASE_VARIANT, _BASE_ALONE],
        topics={
            _TOPIC: (_BASE, _BASE_VARIANT, _BASE_ALONE),
            _TOPIC_ALONE: (_BASE_ALONE,),
            _TOPIC_BASE: (_BASE,),
        },
        successors={_BASE: _SDG, _BASE_VARIANT: _SDG, _BASE_ALONE: None} if transfer else {},
        un_mirror=un_mirror(
            {
                _SDG: un_variable(_SDG_SERIES, name=_SDG),
                _SDG_FEMALE: un_variable(_SDG_SERIES, name=_SDG_FEMALE),
                _WHO: un_variable(_WHO_SERIES, name=_WHO),
            }
        ),
        un_system_enabled=un_system_enabled,
    )


def _leg(*scored: tuple[str, float]) -> dict:
    return {
        "svs": [dcid for dcid, _ in scored],
        "cosine_scores": [score for _, score in scored],
        "types": {dcid: _NODES[dcid]["type"] for dcid, _ in scored},
    }


def _graph(
    *,
    ours: tuple[tuple[str, float], ...] = ((_BASE, 0.9),),
    un: tuple[tuple[str, float], ...] | None = (),
    data: tuple[str, ...] = (_SDG,),
    cls: type[FakeGraph] = FakeGraph,
) -> FakeGraph:
    """Our resolve returns `ours` and the UN's `un` (`None` fails it); the variables in `data`
    have observations at Kenya."""
    return cls(
        nodes=_NODES,
        obs={f"{dcid}|{_KEN}": [_FACET] for dcid in data},
        detect={_QUERY: _leg(*ours)},
        un_detect={_QUERY: None if un is None else _leg(*un)},
        resolve={"Kenya": _KEN},
    )


class _ElectFirstLLM:
    """Elects the first shape on the menu with no bindings, and records each menu it saw."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        usage = {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[], ask=None), usage


async def _resolving(graph: FakeGraph, *, llm: _ElectFirstLLM | None = None) -> RegionResult:
    return await resolve_variable(
        _QUERY,
        entities=["Kenya"],
        date_request=None,
        detect_query=_QUERY,
        role_query=_QUERY,
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        llm=llm or _ElectFirstLLM(),
        base_timing={},
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(deadline=None),
    )


def _resolve(graph: FakeGraph, **kwargs) -> RegionResult:
    return asyncio.run(_resolving(graph, **kwargs))


def _elected(result: RegionResult) -> list[str]:
    return [sv.ref.ref for spec in result.specs for sv in spec.resolved.members]


def _answer(result: RegionResult) -> tuple:
    return (result.status, result.no_data_reason, result.specs, result.warnings)


def _gap_fill_line(caplog: pytest.LogCaptureFixture) -> str:
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gap_fill ")]
    return line


def _names(text: str, dcid: str) -> bool:
    """Whether `text` names `dcid` itself, apart from a UN id that ends with it."""
    return re.search(rf"(?<![\w/]){re.escape(dcid)}", text) is not None


def _candidate(dcid: str, score: float) -> DetectedSV:
    return DetectedSV(dcid=dcid, score=score, type_of="StatisticalVariable", sentence=None)


# -- a superseded variable stands as its successor --


def test_a_recalled_superseded_variable_answers_from_its_successor_at_our_score(
    bind_catalogue,
) -> None:
    bind_catalogue(_catalogue())
    llm = _ElectFirstLLM()

    result = _resolve(_graph(ours=((_BASE, 0.9), (_OURS, 0.8)), data=(_SDG, _BASE, _OURS)), llm=llm)

    assert result.status == "definite"
    assert _elected(result) == [_SDG]
    (menu,) = llm.prompts
    assert menu.index(_SDG) < menu.index(_OURS)
    assert "retrieval_score: 0.900" in menu
    assert not _names(menu, _BASE) and not _names(repr(result), _BASE)


def test_superseded_variants_with_one_successor_make_one_candidate_at_their_best_score() -> None:
    pool = [_candidate(_BASE, 0.7), _candidate(_OURS, 0.8), _candidate(_BASE_VARIANT, 0.95)]

    admitted = admit_candidates(pool, catalogue=_catalogue(), detect_k=3)

    assert [(c.dcid, c.score) for c in admitted] == [(_SDG, 0.95), (_OURS, 0.8)]


def test_supersession_brings_in_no_candidate_the_cut_left_out() -> None:
    """Eighty hits fill the cut where no variable is superseded: 77 admitted ones around a
    replaced variable, a variant merged into it and a superseded one with no successor. The
    three admitted hits ranked after them stay out, as they do without a successor table."""
    kept = [_candidate(f"Kept_{n}", 0.9 - n / 1000) for n in range(77)]
    superseded = [
        _candidate(_BASE, 0.95),
        _candidate(_BASE_VARIANT, 0.85),
        _candidate(_BASE_ALONE, 0.84),
    ]
    below = [_candidate(f"Below_{n}", 0.5 - n / 1000) for n in range(3)]

    admitted = admit_candidates([*below, *kept, *superseded], catalogue=_catalogue(), detect_k=80)

    assert [c.dcid for c in admitted] == [_SDG, *(c.dcid for c in kept)]
    assert admitted[0].score == 0.95


@dataclasses.dataclass(frozen=True)
class _SizeRefusingCatalogue(FakeCatalogue):
    """Refuses `dc/topic/Health` as a catalogue refuses a Topic past the member cap. A leaf
    of it is superseded, and the Topic itself is not: the catalogue would admit it on the
    leaves dev counted only if they fit the cap, and they do not."""

    def admit_candidate(self, dcid: str):
        return None if dcid == "dc/topic/Health" else super().admit_candidate(dcid)


def test_a_topic_refused_for_size_leaves_its_slot_to_the_next_hit() -> None:
    catalogue = _SizeRefusingCatalogue(successors={"sdg/SH_DYN_MORT": "undata/sdg/SH_DYN_MORT"})
    pool = [_candidate("dc/topic/Health", 0.9), _candidate("Count_Person", 0.8)]

    admitted = admit_candidates(pool, catalogue=catalogue, detect_k=1)

    assert [c.dcid for c in admitted] == ["Count_Person"]


def test_a_candidate_refused_for_another_reason_leaves_its_slot() -> None:
    """`undata/who/GONE` is in no UN table and is not superseded, so the next hit takes
    its slot."""
    pool = [_candidate("undata/who/GONE", 0.9), _candidate(_OURS, 0.8), _candidate(_WHO, 0.7)]

    admitted = admit_candidates(pool, catalogue=_catalogue(), detect_k=2)

    assert [c.dcid for c in admitted] == [_OURS, _WHO]


def test_a_topic_of_superseded_leaves_yields_their_successors(bind_catalogue) -> None:
    bind_catalogue(_catalogue())
    graph = FakeGraph(nodes=_NODES, obs={}, detect={}, resolve={})

    assert expand_topics([_TOPIC], graph=graph) == {_SDG: _TOPIC}
    # The cap counts the three superseded leaves the graph holds, merged or dropped, as it
    # counted them when their base provenance was admitted.
    assert expand_topics([_TOPIC], graph=graph, max_members=3) == {_SDG: _TOPIC}
    assert expand_topics([_TOPIC], graph=graph, max_members=2) == {}


def test_a_recalled_topic_of_superseded_leaves_answers_from_their_successor(
    bind_catalogue,
) -> None:
    bind_catalogue(_catalogue())
    llm = _ElectFirstLLM()

    result = _resolve(_graph(ours=((_TOPIC, 0.9), (_TOPIC_ALONE, 0.9))), llm=llm)

    assert _elected(result) == [_SDG]
    (menu,) = llm.prompts
    assert not _names(menu, _BASE) and not _names(menu, _BASE_ALONE)


def test_a_superseded_topic_holds_its_slot_in_the_cut() -> None:
    """Every leaf of `_TOPIC_ALONE` is superseded with no successor, so the Topic is dropped
    and the hit ranked after it stays out, as it does where the Topic is admitted."""
    pool = [_candidate(_TOPIC_ALONE, 0.9), _candidate(_OURS, 0.8)]

    assert admit_candidates(pool, catalogue=_catalogue(), detect_k=1) == ()
    assert [c.dcid for c in admit_candidates(pool, catalogue=_catalogue(), detect_k=2)] == [_OURS]


def test_a_superseded_variable_with_no_successor_is_no_candidate(bind_catalogue) -> None:
    catalogue = _catalogue()
    bind_catalogue(catalogue)
    llm = _ElectFirstLLM()

    result = _resolve(
        _graph(ours=((_BASE_ALONE, 0.95), (_OURS, 0.9)), data=(_BASE_ALONE,)), llm=llm
    )

    assert standing_dcid(_BASE_ALONE, catalogue=catalogue) is None
    assert result.status == "no_data"
    (menu,) = llm.prompts
    assert _OURS in menu and _BASE_ALONE not in menu
    assert _BASE_ALONE not in repr(result)


def test_with_the_un_origin_off_nothing_is_replaced_and_no_second_pass_runs(
    bind_catalogue, caplog
) -> None:
    bind_catalogue(_catalogue(un_system_enabled=False))
    llm = _ElectFirstLLM()

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(_graph(un=((_SDG, 0.9), (_WHO, 0.8)), data=(_SDG, _WHO, _BASE)), llm=llm)

    assert (result.status, result.specs) == ("no_data", ())
    assert llm.prompts == []
    assert "pass2=skipped:no_un_candidates " in _gap_fill_line(caplog)
    assert [dcid for dcid in (_SDG, _WHO, _BASE) if _names(repr(result), dcid)] == []


# -- the passes --


def test_the_first_pass_holds_no_candidate_of_the_un_resolver(bind_catalogue) -> None:
    """The UN resolve returns two candidates that outscore ours, and one is the successor
    our recall also stands a variable as."""
    bind_catalogue(_catalogue())
    llm = _ElectFirstLLM()

    result = _resolve(
        _graph(ours=((_OURS, 0.8),), un=((_WHO, 0.95), (_SDG, 0.9)), data=(_OURS, _WHO, _SDG)),
        llm=llm,
    )

    assert _elected(result) == [_OURS]
    (menu,) = llm.prompts
    assert _WHO not in menu and _SDG not in menu


def test_each_passs_candidates_are_scored_by_one_resolver(bind_catalogue, monkeypatch) -> None:
    """`_SDG` reaches the first pass through our recall at 0.9 and the second through the UN
    resolve at 0.7, and has no data, so both passes run. Each pass's shapes carry the scores
    its own resolver gave."""
    bind_catalogue(_catalogue())
    derived: list[dict[str, float]] = []
    derive = phase_intake.derive_shapes

    def recording(**kwargs):
        shapes = derive(**kwargs)
        derived.append({sv: s.representative_score for s in shapes for sv in s.sv_arc_facts or {}})
        return shapes

    monkeypatch.setattr(phase_intake, "derive_shapes", recording)

    _resolve(_graph(ours=((_BASE, 0.9), (_OURS, 0.8)), un=((_SDG, 0.7), (_WHO, 0.6)), data=(_WHO,)))

    assert derived == [{_SDG: 0.9, _OURS: 0.8}, {_SDG: 0.7, _WHO: 0.6}]


def test_the_second_pass_does_not_elect_the_transferred_shape_the_first_found_empty(
    bind_catalogue,
) -> None:
    """`_SDG` has no data. The first pass elects it through our recall; the UN resolve ranks
    it first again, and the second pass's menu holds `_WHO` alone."""
    bind_catalogue(_catalogue())
    llm = _ElectFirstLLM()

    result = _resolve(_graph(un=((_SDG, 0.9), (_WHO, 0.8)), data=(_WHO,)), llm=llm)

    assert _elected(result) == [_WHO]
    first, second = llm.prompts
    assert _SDG in first
    assert _WHO in second and _SDG not in second


def test_the_first_pass_answers_from_a_held_series_member_recall_did_not_find(
    bind_catalogue,
) -> None:
    """The first pass elects `_SDG`'s shape, which stands for the series the catalogue
    holds, so its population holds `_SDG_FEMALE`, which has data, though only the UN
    resolve recalls it. The first pass answers from it and no second pass runs."""
    bind_catalogue(_catalogue())
    llm = _ElectFirstLLM()

    result = _resolve(_graph(un=((_SDG_FEMALE, 0.9),), data=(_SDG_FEMALE,)), llm=llm)

    assert _elected(result) == [_SDG_FEMALE]
    (menu,) = llm.prompts
    assert _SDG in menu


@pytest.mark.parametrize(
    "ours",
    [
        pytest.param(
            ((_BASE, 0.95), (_BASE_VARIANT, 0.9), (_OURS, 0.8), (_OURS_LOW, 0.7)), id="merged"
        ),
        pytest.param(
            ((_BASE, 0.95), (_BASE_ALONE, 0.9), (_OURS, 0.8), (_OURS_LOW, 0.7)), id="dropped"
        ),
        pytest.param(((_TOPIC, 0.95), (_OURS, 0.9), (_OURS_LOW, 0.7)), id="topic-leaves"),
        pytest.param(
            ((_TOPIC, 0.95), (_TOPIC_BASE, 0.95), (_OURS, 0.9), (_OURS_LOW, 0.7)),
            id="leaf-of-two-roots",
        ),
        pytest.param(
            ((_BASE_VARIANT, 0.95), (_TOPIC_BASE, 0.95), (_OURS, 0.9), (_OURS_LOW, 0.7)),
            id="leaf-standing-as-a-recalled-successor",
        ),
    ],
)
def test_a_superseded_hit_is_read_only_as_its_successor(bind_catalogue, monkeypatch, ours) -> None:
    """A merged or dropped hit, recalled directly or as a Topic leaf, is never read or
    drafted under its own dcid, and every other candidate, `_OURS_LOW` included, is."""
    bind_catalogue(_catalogue())
    llm = _ElectFirstLLM()
    graph = _graph(ours=ours, data=(_SDG, _OURS, _OURS_LOW))
    read: list[str] = []
    arcs = graph.node_arcs_batch
    monkeypatch.setattr(graph, "node_arcs_batch", lambda dcids: read.extend(dcids) or arcs(dcids))
    drafted: list[str] = []
    derive = phase_intake.derive_shapes

    def recording(**kwargs):
        shapes = derive(**kwargs)
        drafted.extend(sv for shape in shapes for sv in shape.sv_arc_facts or {})
        return shapes

    monkeypatch.setattr(phase_intake, "derive_shapes", recording)

    _resolve(graph, llm=llm)

    (menu,) = llm.prompts
    assert _SDG in menu
    assert _OURS_LOW in menu
    superseded = (_BASE, _BASE_VARIANT, _BASE_ALONE)
    assert [d for d in superseded if d in read or d in drafted or _names(menu, d)] == []
    # The control: the reads did reach the confirmed variables.
    assert _SDG in read and _SDG in drafted


class _PendingUnGraph(FakeGraph):
    """Hands out a UN leg that never settles, as a resolve still running does."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.leg: Future = Future()
        self.leg.set_running_or_notify_cancel()

    def detect_svs(self, query: str) -> Detection:
        return dataclasses.replace(super().detect_svs(query), un=self.leg)


def test_a_un_resolve_that_never_answers_leaves_a_transferred_answer_intact(
    bind_catalogue,
) -> None:
    bind_catalogue(_catalogue())
    settled = _resolve(_graph())
    graph = _graph(cls=_PendingUnGraph)

    async def answer() -> RegionResult:
        async with asyncio.timeout(5):
            return await _resolving(graph)

    result = asyncio.run(answer())

    assert _elected(result) == [_SDG]
    assert _answer(result) == _answer(settled)
    assert "un_system_unavailable" not in [w.code for w in result.warnings]
    assert isinstance(graph, _PendingUnGraph) and not graph.leg.done()


def test_the_bind_prompt_is_unchanged_when_recall_holds_no_superseded_variable(
    bind_catalogue,
) -> None:
    """The same recall under a catalogue with a successor table and a UN resolve that returns
    candidates, and under one with neither."""
    prompts = []
    for catalogue, un in (
        (_catalogue(), ((_SDG, 0.95), (_WHO, 0.9))),
        (_catalogue(transfer=False, un_system_enabled=False), ()),
    ):
        bind_catalogue(catalogue)
        llm = _ElectFirstLLM()
        _resolve(_graph(ours=((_OURS, 0.8),), un=un, data=(_OURS,)), llm=llm)
        prompts.append(llm.prompts)

    assert prompts[0] == prompts[1]
    assert len(prompts[0]) == 1
