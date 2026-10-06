"""The per-capita guard distinguishes an unresolved denominator arc from no arc.

An unresolved target preserves a valid per-capita result; a missing arc rejects
it. Both cases run through derive_shapes, bind, and the guard. So does the prune that
leaves a per-capita query's menu for the shapes divided by a count of persons, and the
guard's refusal of a shape divided by anything else.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.region import resolve_variable
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.retrieve import Materialised
from search.models import CoverageBare
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, _load_json, make_resolver

_QUERY = "GDP per capita of Vietnam"
_SV = "Amount_EconomicActivity_GDP_Test_PerCapita"


def _gdp_arcs(*, denominator: dict | None) -> dict:
    # Modelled on the real dfo-dev arc for
    # Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita: its
    # measurementDenominator is a dcid-less ProvisionalNode.
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Nominal gross domestic product per capita"}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if denominator is not None:
        arcs["measurementDenominator"] = denominator
    return arcs


class _ElectSingleShapeLLM:
    """Elects the sole candidate shape (index 0) with no constraint bindings
    and no also_plausible alternates -- the shape has no constraint slots, so
    an empty bindings list is a complete, valid bind() output for it."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[],
                also_plausible=[],
                ask=None,
            ),
            None,
        )


def _run(*, denominator: dict | None, monkeypatch):
    nodes = {
        "country/VNM": {"label": "Vietnam", "type": "Country"},
        _SV: {
            "label": "Nominal gross domestic product per capita",
            "type": "StatisticalVariable",
            "arcs": _gdp_arcs(denominator=denominator),
        },
    }
    detect = {_QUERY: {"svs": [_SV], "cosine_scores": [0.9]}}
    graph = FakeGraph(nodes=nodes, detect=detect, resolve={"Vietnam": "country/VNM"}, obs={})

    # materialise itself is monkeypatched to a guaranteed-successful result --
    # these tests isolate the guard's own precedence decision from graph/sdmx
    # observation resolution, mirroring test_denominator_order.py's pattern.
    monkeypatch.setattr(
        "search.engine.region.materialise_step.materialise",
        lambda *args, **kwargs: Materialised(
            sv_dcids=[_SV],
            has_data=True,
            coverage=CoverageBare(has_data=True, window=None),
            # A real resolve path always populates this (Materialised's own
            # docstring); _phase_assemble_definite reads it unconditionally
            # to observe this Spec's own served measure partition.
            member_facts={_SV: ArcsMemberFacts(arcs=_gdp_arcs(denominator=denominator))},
        ),
    )

    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Vietnam"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectSingleShapeLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_unresolved_denominator_guard_does_not_fire(monkeypatch):
    """An unresolved denominator arc preserves the per-capita result."""
    result = _run(
        denominator={"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]},
        monkeypatch=monkeypatch,
    )
    assert result.status == "definite", (
        f"expected the guard to stand down for an unresolved-but-present denominator, "
        f"got status={result.status!r} reason={result.no_data_reason!r}"
    )
    assert len(result.specs) == 1
    assert result.specs[0].resolved.members[0].ref.dcid == _SV


def test_genuinely_bare_denominator_guard_fires(monkeypatch):
    """A missing denominator rejects the base total for a per-capita query."""
    result = _run(denominator=None, monkeypatch=monkeypatch)
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"


# A per-person request read against the denominator each shape divides by.

_GNI = "Amount_EconomicActivity_GrossNationalIncome_Test"
_ODA_GNI_SHARE = "Amount_Oda_Test_AsFractionOf_" + _GNI
_ODA_AMOUNT = "Amount_Oda_Test"
_ADULTS = "Count_Person_15OrMoreYears"
_ALCOHOL_PER_ADULT = (
    "Amount_Consumption_Alcohol_15OrMoreYears_AsFractionOf_Count_Person_15OrMoreYears"
)
_ALCOHOL_TOTAL = "Amount_Consumption_Alcohol_Test"


def _sv_node(name: str, pop_type: str, meas_prop: str, denominator: str | None = None) -> dict:
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": name}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if denominator is not None:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": denominator}]}
    return {"label": name, "type": "StatisticalVariable", "arcs": arcs}


_ODA_NODES = {
    "country/NOR": {"label": "Norway", "type": "Country"},
    _GNI: _sv_node("Gross national income", "EconomicActivity", "amount"),
    _ODA_GNI_SHARE: _sv_node("ODA as a share of GNI", "Oda", "amount", denominator=_GNI),
    _ODA_AMOUNT: _sv_node("ODA", "Oda", "amount"),
}

_TEACHERS = "Count_Teacher_Test"
_MURDER_PER_CAPITA = "Count_CriminalActivities_Murder_Test_AsFractionOf_Count_Person"
_TEACHER_NODES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _TEACHERS: _sv_node("Number of teachers", "Teacher", "count"),
    _MURDER_PER_CAPITA: _sv_node(
        "Murder cases per capita", "CriminalActivities", "count", denominator="Count_Person"
    ),
}

_FIXTURE_NODES = _load_json("graph_nodes.json")
_ALCOHOL_NODES = {
    "country/VNM": {"label": "Vietnam", "type": "Country"},
    _ADULTS: _FIXTURE_NODES[_ADULTS],
    _ALCOHOL_PER_ADULT: _FIXTURE_NODES[_ALCOHOL_PER_ADULT],
    _ALCOHOL_TOTAL: _sv_node("Alcohol consumption", "Consumption", "amount"),
}


class _ArcsSpyGraph(FakeGraph):
    """Records every `node_arcs_batch` request."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.arc_reads: list[list[str]] = []

    def node_arcs_batch(self, dcids, *, deadline=None):
        self.arc_reads.append(list(dcids))
        return super().node_arcs_batch(dcids, deadline=deadline)


class _MenuLLM(_ElectSingleShapeLLM):
    """Elects index 0 and keeps each bind prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        self.prompts.append(prompt)
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _resolve(query: str, *, nodes: dict, svs: list[str], place: tuple[str, str], monkeypatch):
    """Resolve `query` over `svs`, ranked in that order, with bind electing the first shape
    on its menu and every shape materialising with data."""
    name, dcid = place
    graph = _ArcsSpyGraph(
        nodes=nodes,
        detect={query: {"svs": svs, "cosine_scores": [0.9 - 0.01 * i for i in range(len(svs))]}},
        resolve={name: dcid},
        obs={},
    )
    monkeypatch.setattr(
        "search.engine.region.materialise_step.materialise",
        lambda shape, *args, **kwargs: Materialised(
            sv_dcids=list(shape.sv_arc_facts),
            has_data=True,
            coverage=CoverageBare(has_data=True, window=None),
            member_facts={sv: ArcsMemberFacts(arcs=nodes[sv]["arcs"]) for sv in shape.sv_arc_facts},
        ),
    )
    llm = _MenuLLM()
    result = asyncio.run(
        resolve_variable(
            query,
            entities=[name],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm,
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )
    return result, graph, llm


@pytest.mark.parametrize(
    "svs",
    [[_ODA_GNI_SHARE, _ODA_AMOUNT], [_ODA_AMOUNT, _ODA_GNI_SHARE]],
    ids=["elects-gni-share", "elects-amount"],
)
def test_aid_per_capita_refuses_whichever_aid_shape_bind_elects(svs, monkeypatch):
    """dq-crs-048: aid as a share of GNI and an aid amount are both no per-capita series,
    so either election refuses `denominator_not_available`."""
    result, graph, _ = _resolve(
        "aid per capita from Norway",
        nodes=_ODA_NODES,
        svs=svs,
        place=("Norway", "country/NOR"),
        monkeypatch=monkeypatch,
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    # The denominator's kind is read once, in one batch.
    assert [read for read in graph.arc_reads if _GNI in read] == [[_GNI]]


def test_person_count_denominator_answers_a_per_capita_query(monkeypatch):
    """Alcohol consumption per adult divides by a count of persons, read from its
    denominator's arcs, so its election answers."""
    result, graph, _ = _resolve(
        "alcohol consumption per capita",
        nodes=_ALCOHOL_NODES,
        svs=[_ALCOHOL_PER_ADULT, _ALCOHOL_TOTAL],
        place=("Vietnam", "country/VNM"),
        monkeypatch=monkeypatch,
    )
    assert result.status == "definite", result.no_data_reason
    assert result.specs[0].resolved.members[0].ref.dcid == _ALCOHOL_PER_ADULT
    assert [read for read in graph.arc_reads if _ADULTS in read] == [[_ADULTS]]


def test_off_topic_per_capita_series_stays_beside_the_named_shape(monkeypatch):
    """nd-07: a murder rate per capita on the menu does not displace the teacher count
    "teacher count per capita" names, and the elected count refuses
    `denominator_not_available`."""
    result, _, llm = _resolve(
        "teacher count per capita in Kenya",
        nodes=_TEACHER_NODES,
        svs=[_TEACHERS, _MURDER_PER_CAPITA],
        place=("Kenya", "country/KEN"),
        monkeypatch=monkeypatch,
    )
    (prompt,) = llm.prompts
    assert _TEACHERS in prompt
    assert _MURDER_PER_CAPITA in prompt
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"


def test_no_per_person_request_reads_no_denominator(monkeypatch):
    """ "Germany's total ODA" names no denominator: no denominator is read, and bind sees
    the share of GNI beside the amount."""
    nodes = {**_ODA_NODES, "country/DEU": {"label": "Germany", "type": "Country"}}
    result, graph, llm = _resolve(
        "Germany's total ODA",
        nodes=nodes,
        svs=[_ODA_GNI_SHARE, _ODA_AMOUNT],
        place=("Germany", "country/DEU"),
        monkeypatch=monkeypatch,
    )
    assert result.status in ("definite", "candidates")
    assert not [read for read in graph.arc_reads if _GNI in read]
    (prompt,) = llm.prompts
    assert _ODA_GNI_SHARE in prompt
    assert _ODA_AMOUNT in prompt.replace(_ODA_GNI_SHARE, "")
