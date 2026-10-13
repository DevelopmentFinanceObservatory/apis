"""The materialise quantity gate holds the elected shape to the leg's quantity.

An unresolved per-capita denominator target preserves a valid per-capita result; a missing
arc rejects it. Both cases run through derive_shapes, bind, and the guard. So does the
prune that leaves a per-capita query's menu for the shapes divided by a count of persons,
and the guard's refusal of a shape divided by anything else. Every named form is held,
not only per-person ones, and a denominator the question names is held by the two-case
rule (`quantity.check_denominator`), the refusal carrying the elected shape as its flagged
match, its level reading as nearest_real, and the denominators asked and found.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.conjoin import assemble_region
from search.engine.errors import GraphInfraError, LLMInfraError
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.retrieve import Materialised
from search.engine.role_text import DoorRole
from search.models import CoverageBare, NoDataPart
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
    an empty bindings list is a complete, valid bind() output for it. The quantity
    read finds no form and no denominator."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return QuantityOutput(), None
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
    assert result.specs[0].resolved.members[0].ref.ref == _SV


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
    """Elects index 0 and keeps each bind prompt in `prompts` and each quantity read
    prompt in `quantity_prompts`."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.quantity_prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        (self.quantity_prompts if schema is QuantityOutput else self.prompts).append(prompt)
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _resolve(
    query: str,
    *,
    nodes: dict,
    svs: list[str],
    place: tuple[str, str],
    monkeypatch,
    llm: _MenuLLM | None = None,
    obs: dict | None = None,
    without_data: frozenset[str] = frozenset(),
    graph_cls: type[_ArcsSpyGraph] = _ArcsSpyGraph,
    roles: list[DoorRole] | None = None,
):
    """Resolve `query` over `svs`, ranked in that order, with bind electing the first shape
    on its menu, or as `llm` reads it, and every shape materialising with data unless its
    members are in `without_data`. `roles` are the place's door roles."""
    name, dcid = place
    graph = graph_cls(
        nodes=nodes,
        detect={query: {"svs": svs, "cosine_scores": [0.9 - 0.01 * i for i in range(len(svs))]}},
        resolve={name: dcid},
        obs=obs or {},
    )

    def _materialise(shape, *args, **kwargs) -> Materialised:
        has_data = not set(shape.sv_arc_facts) <= without_data
        return Materialised(
            sv_dcids=list(shape.sv_arc_facts),
            has_data=has_data,
            coverage=CoverageBare(has_data=has_data, window=None),
            available_at_place=None if has_data else False,
            member_facts={sv: ArcsMemberFacts(arcs=nodes[sv]["arcs"]) for sv in shape.sv_arc_facts},
        )

    monkeypatch.setattr("search.engine.region.materialise_step.materialise", _materialise)
    llm = llm or _MenuLLM()
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
            entity_roles={name: roles} if roles else None,
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
    assert result.specs[0].resolved.members[0].ref.ref == _ALCOHOL_PER_ADULT
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


def test_a_request_naming_no_form_lists_the_denominators_for_the_quantity_read(monkeypatch):
    """ "Germany's total ODA" names no form: the share of GNI's denominator is read once,
    for the name the quantity read lists it under, and bind sees the share beside the
    amount and no denominator list."""
    nodes = {**_ODA_NODES, "country/DEU": {"label": "Germany", "type": "Country"}}
    result, graph, llm = _resolve(
        "Germany's total ODA",
        nodes=nodes,
        svs=[_ODA_GNI_SHARE, _ODA_AMOUNT],
        place=("Germany", "country/DEU"),
        monkeypatch=monkeypatch,
    )
    assert result.status in ("definite", "candidates")
    assert [read for read in graph.arc_reads if _GNI in read] == [[_GNI]]
    (prompt,) = llm.prompts
    assert _ODA_GNI_SHARE in prompt
    assert _ODA_AMOUNT in prompt.replace(_ODA_GNI_SHARE, "")
    assert "DENOMINATORS" not in prompt
    (quantity_prompt,) = llm.quantity_prompts
    assert quantity_prompt.startswith("Variable phrase: ")
    assert f"  0: Gross national income ({_GNI})\n  1: none of these" in quantity_prompt


class _DenominatorReadFailsGraph(_ArcsSpyGraph):
    """Fails every arcs read that names the GNI denominator, as a graph read that omits
    it from both the batch and its retry does."""

    def node_arcs_batch(self, dcids, *, deadline=None):
        if _GNI in dcids:
            raise GraphInfraError(f"graph /v2/node omitted {[_GNI]!r}")
        return super().node_arcs_batch(dcids, deadline=deadline)


def test_a_failed_denominator_read_leaves_a_formless_request_answering(monkeypatch):
    """ "Germany's total ODA" names no form, so a failed read of the share of GNI's
    denominator only leaves that denominator off the quantity read's list."""
    nodes = {**_ODA_NODES, "country/DEU": {"label": "Germany", "type": "Country"}}
    result, _, llm = _resolve(
        "Germany's total ODA",
        nodes=nodes,
        svs=[_ODA_GNI_SHARE, _ODA_AMOUNT],
        place=("Germany", "country/DEU"),
        monkeypatch=monkeypatch,
        graph_cls=_DenominatorReadFailsGraph,
    )
    assert result.status in ("definite", "candidates")
    (prompt,) = llm.quantity_prompts
    assert f"({_GNI})" not in prompt


def test_a_failed_denominator_read_fails_a_request_naming_a_form(monkeypatch):
    """ "aid per capita" names a form, whose check needs the denominator's kind."""
    with pytest.raises(GraphInfraError):
        _resolve(
            "aid per capita from Norway",
            nodes=_ODA_NODES,
            svs=[_ODA_GNI_SHARE, _ODA_AMOUNT],
            place=("Norway", "country/NOR"),
            monkeypatch=monkeypatch,
            graph_cls=_DenominatorReadFailsGraph,
        )


@pytest.mark.parametrize(
    ("denominator", "stated_form"),
    [("none of these", "rate"), ("none of these", None), (None, "rate")],
    ids=["none-of-these-and-a-stated-form", "none-of-these", "stated-form"],
)
def test_a_failed_denominator_read_fails_a_request_the_model_reads_a_quantity_in(
    denominator, stated_form, monkeypatch
):
    """ "debt to national income" names no form a phrase reads, so a failed read of the GNI
    denominator leaves the quantity read listing only "none of these". A stated form or a denominator
    index would then be judged on reads that are missing, so the read's error is raised
    where a `denominator_not_available` refusal would blame the series."""
    llm = _ReadingLLM(
        "External debt (% of GNI)",
        denominator=denominator,
        base_words="national income",
        stated_form=stated_form,
    )
    with pytest.raises(GraphInfraError):
        _resolve(
            "debt to national income",
            nodes=_DEBT_NODES,
            svs=_DEBT_MENU,
            place=_GHANA,
            monkeypatch=monkeypatch,
            llm=llm,
            graph_cls=_DenominatorReadFailsGraph,
        )


# Debt menus: a dollar level, a share of GNI, a UN ratio whose name alone states its base,
# and a military share of GDP that puts GDP on the menu's denominator list.

_GHANA = ("Ghana", "country/GHA")
_GDP = "Amount_EconomicActivity_GDP_Test"
_DEBT_LEVEL = "Amount_Debt_Test_Nominal"
_DEBT_GNI = "Amount_Debt_Test_AsFractionOf_" + _GNI
_DEBT_UN = "undata/Debt_Test_GDP_Ratio"
_DEBT_GDP = "Amount_Debt_Test_AsFractionOf_" + _GDP
_MILITARY_GDP = "Amount_Military_Test_AsFractionOf_" + _GDP
_DEBT_PER_CAPITA = "Amount_Debt_Test_PerCapita"
# Debt service split by creditor type, its values named with the graph's code-system
# suffix. The debt nodes hold it; a menu lists it where a test offers it.
_CREDITOR_TYPE = "idsCreditorType"
_PRIVATE_CREDITORS = "IDSCreditorTypePRVT"
_OFFICIAL_CREDITORS = "IDSCreditorTypeOFFT"
_SERVICE_PRIVATE = "Amount_DebtService_Test_" + _PRIVATE_CREDITORS
_SERVICE_OFFICIAL = "Amount_DebtService_Test_" + _OFFICIAL_CREDITORS
_GDP_LINKED = "IDSCreditorTypeGDPL"
_SERVICE_GDP_LINKED = "Amount_DebtService_Test_" + _GDP_LINKED


def _unnamed_sv_node(dcid: str, pop_type: str, meas_prop: str) -> dict:
    """A StatVar node the graph gives no name, labelled by its dcid as a nameless read is."""
    node = _sv_node(dcid, pop_type, meas_prop)
    del node["arcs"]["name"]
    return node


def _creditor_node(name: str, creditor: str) -> dict:
    node = _sv_node(name, "DebtService", "amount")
    node["arcs"]["constraintProperties"] = {"nodes": [{"dcid": _CREDITOR_TYPE}]}
    node["arcs"][_CREDITOR_TYPE] = {"nodes": [{"dcid": creditor}]}
    return node


_DEBT_NODES = {
    "country/GHA": {"label": "Ghana", "type": "Country"},
    _GDP: _sv_node("Gross domestic product", "EconomicActivity", "amount"),
    _GNI: _sv_node("Gross national income", "EconomicActivity", "amount"),
    _DEBT_LEVEL: _sv_node("External debt (current US$)", "Debt", "amount"),
    _DEBT_GNI: _sv_node("External debt (% of GNI)", "Debt", "amount", denominator=_GNI),
    _DEBT_UN: _sv_node("External debt to GDP ratio", "Debt", "value"),
    _DEBT_GDP: _sv_node("External debt (% of GDP)", "Debt", "amount", denominator=_GDP),
    _MILITARY_GDP: _sv_node("Military spending (% of GDP)", "Military", "amount", denominator=_GDP),
    _DEBT_PER_CAPITA: _sv_node(
        "External debt per capita", "Debt", "amount", denominator="Count_Person"
    ),
    _CREDITOR_TYPE: {"label": "IDS creditor type", "type": "Property"},
    _PRIVATE_CREDITORS: {"label": "private creditors (IDS creditor type)", "type": "Enum"},
    _OFFICIAL_CREDITORS: {"label": "official creditors (IDS creditor type)", "type": "Enum"},
    _SERVICE_PRIVATE: _creditor_node("Debt service to private creditors", _PRIVATE_CREDITORS),
    _SERVICE_OFFICIAL: _creditor_node("Debt service to official creditors", _OFFICIAL_CREDITORS),
    _GDP_LINKED: {"label": "holders of GDP-linked bonds (IDS creditor type)", "type": "Enum"},
    _SERVICE_GDP_LINKED: _creditor_node("Debt service to GDP-linked bondholders", _GDP_LINKED),
}
_DEBT_MENU = [_DEBT_LEVEL, _DEBT_GNI, _DEBT_UN, _MILITARY_GDP]
_CREDITOR_MENU = [*_DEBT_MENU, _SERVICE_PRIVATE, _SERVICE_OFFICIAL]
_GDP_LINKED_MENU = [*_DEBT_MENU, _SERVICE_GDP_LINKED]
_NONE_OF_THESE = "none of these"
_OBS = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}


class _ReadingLLM(_MenuLLM):
    """Elects the shape labelled `elect`, nominates the shapes labelled `also`, and names
    the DENOMINATORS entry `denominator` (a label, or "none of these") on the phrase's
    words `base_words`, with `stated_form` as the form it states and `unplaced` as the
    phrases it places nowhere."""

    def __init__(
        self,
        elect: str,
        *,
        denominator: str | int | None = None,
        base_words: str | None = None,
        stated_form: str | None = None,
        also: tuple[str, ...] = (),
        unplaced: tuple[str, ...] = (),
    ) -> None:
        """`denominator` names a DENOMINATORS entry by label, or, as an int, the offset
        past the list's last index (`1` is the first index outside it)."""
        super().__init__()
        self.elect = elect
        self.denominator = denominator
        self.base_words = base_words
        self.stated_form = stated_form
        self.also = also
        self.unplaced = unplaced

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            self.quantity_prompts.append(prompt)
            return QuantityOutput(
                base_words=self.base_words,
                denominator_index=self._denominator_index(prompt),
                stated_form=self.stated_form,
            ), None
        self.prompts.append(prompt)
        labels = re.findall(r"^Shape (\d+):\n  label: (.+)$", prompt, flags=re.MULTILINE)
        chosen = next(int(index) for index, label in labels if label == self.elect)
        output = _UniversalBindOutput(
            chosen_shape_index=chosen,
            bindings=[],
            also_plausible=[int(i) for i, label in labels if label in self.also],
            ask=None,
            unplaced=list(self.unplaced),
        )
        return output, None

    def _denominator_index(self, prompt: str) -> int | None:
        entries = re.findall(r"^  (\d+): (.+?)(?: \(.+\))?$", prompt, flags=re.MULTILINE)
        if self.denominator is None:
            return None
        if isinstance(self.denominator, int):
            return len(entries) - 1 + self.denominator
        return next(int(i) for i, entry in entries if entry == self.denominator)


def _resolve_debt(
    query: str,
    llm: _ReadingLLM,
    monkeypatch,
    *,
    obs: dict | None = None,
    menu: list[str] = _DEBT_MENU,
    without_data: frozenset[str] = frozenset(),
    nodes: dict = _DEBT_NODES,
):
    result, _, _ = _resolve(
        query,
        nodes=nodes,
        svs=menu,
        place=_GHANA,
        monkeypatch=monkeypatch,
        llm=llm,
        obs=obs,
        without_data=without_data,
    )
    return result


def _members(spec) -> list[str]:
    return [sv.ref.ref for sv in spec.resolved.members]


@pytest.mark.parametrize(
    "query,nodes,svs,elect,unplaced",
    [
        ("debt to GDP ratio", _DEBT_NODES, _DEBT_MENU, "External debt (current US$)", ()),
        (
            "debt service as a share of exports",
            _DEBT_NODES,
            _DEBT_MENU,
            "External debt (current US$)",
            (),
        ),
        (
            "aid density",
            _ODA_NODES,
            [_ODA_AMOUNT, _ODA_GNI_SHARE],
            "ODA",
            (),
        ),
        (
            "measles debt to GDP ratio",
            _DEBT_NODES,
            _DEBT_MENU,
            "External debt (current US$)",
            ("measles",),
        ),
    ],
    ids=["rate-level", "share-level", "per-area-level", "rate-level-unplaced-word"],
)
def test_a_form_the_elected_shape_does_not_serve_refuses(
    query, nodes, svs, elect, unplaced, monkeypatch
):
    """A rate, a share or a density the elected level does not serve refuses
    `denominator_not_available`, flagging the level, where the gate once held only
    per-person forms. A level has no level relaxation, so the refusal offers no
    nearest_real and probes no relaxation of the refused series. The flagged level's trace
    records the words the elected reading leaves unplaced, which the refusal does not
    name, and each of them departs once."""
    from search.engine.region import phase_place_materialise

    probes: list[object] = []
    monkeypatch.setattr(
        phase_place_materialise,
        "_nearest_real_for",
        lambda state, **kwargs: probes.append(kwargs),
    )
    place = _GHANA if nodes is _DEBT_NODES else ("Norway", "country/NOR")
    result, _, _ = _resolve(
        query,
        nodes=nodes,
        svs=svs,
        place=place,
        monkeypatch=monkeypatch,
        llm=_ReadingLLM(elect, unplaced=unplaced),
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    (flagged,) = result.flagged_match or ()
    assert _members(flagged) == [svs[0]]
    assert [q.text for q in flagged.resolved.trace.query_qualifiers] == list(unplaced)
    assert result.unbound_qualifiers == ()
    assert [(f.code, f.asked) for f in result.departure_facts.values()] == [
        ("measure_not_applied", word) for word in unplaced
    ]
    assert result.denominator is not None
    assert result.denominator.asked is None and result.denominator.found is None
    assert result.nearest_real is None
    assert probes == []


def test_a_matching_denominator_answers(monkeypatch):
    result = _resolve_debt(
        "debt as a share of national income",
        _ReadingLLM(
            "External debt (% of GNI)",
            denominator="Gross national income",
            base_words="national income",
        ),
        monkeypatch,
    )
    assert result.status == "definite", result.no_data_reason
    assert _members(result.specs[0]) == [_DEBT_GNI]
    assert not [w for w in result.warnings if w.code == "denominator_unverified"]


@pytest.mark.parametrize(
    ("base_words", "status", "reason"),
    [
        ("national income", "no_data", "variable_not_resolved"),
        (None, "definite", None),
        ("export earnings", "definite", None),
    ],
    ids=["grounded", "no-words", "words-the-phrase-lacks"],
)
def test_a_denominator_index_outside_the_list_refuses_only_when_grounded(
    base_words, status, reason, monkeypatch
):
    """A `denominator_index` past "none of these" names no entry. Grounding decides
    first: on words the phrase holds, the quantity read's output is malformed and the part
    refuses `variable_not_resolved`, as an out-of-range shape index does; on no words, or
    words the phrase lacks, the phrase names no base and the share of GNI answers."""
    result = _resolve_debt(
        "debt as a share of national income",
        _ReadingLLM("External debt (% of GNI)", denominator=1, base_words=base_words),
        monkeypatch,
    )
    assert (result.status, result.no_data_reason) == (status, reason)


@pytest.mark.parametrize(
    ("query", "base_words"),
    [
        ("debt-to-GDP ratio", "GDP"),
        ("government debt-to-GDP", "GDP"),
        ("tax-to-GDP ratio", "GDP"),
        ("Debt-to-gdp", "gdp"),
    ],
    ids=["debt-to-GDP", "government-debt-to-GDP", "tax-to-GDP", "lowercase"],
)
def test_a_base_inside_a_hyphenated_ratio_grounds(query, base_words, monkeypatch):
    """The live quantity read quotes "GDP" for "debt-to-GDP ratio", inside what the
    tokeniser holds as one word. The quote grounds, so the share of GNI refuses as it
    does for "debt to GDP ratio"."""
    result = _resolve_debt(
        query,
        _ReadingLLM(
            "External debt (% of GNI)", denominator="Gross domestic product", base_words=base_words
        ),
        monkeypatch,
    )
    assert (result.status, result.no_data_reason) == ("no_data", "denominator_not_available")
    assert result.denominator is not None
    assert result.denominator.asked is not None and result.denominator.asked.ref == _GDP
    assert result.denominator.found is not None and result.denominator.found.ref == _GNI


@pytest.mark.parametrize(
    ("query", "base_words", "menu", "asked"),
    [
        ("debt as a share of GDP", "share of GDP", _DEBT_MENU, _GDP),
        ("debt as a share of GDP", "share", _DEBT_MENU, None),
        ("debt ratio", "ratio", _DEBT_MENU, None),
        ("debt per capita", "capita", _DEBT_MENU, None),
        ("debt to GDP", "GDP", _CREDITOR_MENU, _GDP),
        ("debt to GDP", "GDP", _GDP_LINKED_MENU, _GDP),
        ("debt to GDP of private creditors", "GDP of private creditors", _CREDITOR_MENU, _GDP),
    ],
    ids=[
        "share-of-GDP",
        "share",
        "ratio",
        "capita",
        "GDP-beside-creditor-types",
        "GDP-beside-a-label-holding-it",
        "quote-reaching-past-a-creditor-type",
    ],
)
def test_a_quote_of_form_words_alone_names_no_base(query, base_words, menu, asked, monkeypatch):
    """The live quantity read answers "poverty headcount ratio" with GDP quoted as
    "ratio". A quote names a base only when it holds a word beyond form wording, so
    "share of GDP" names GDP and the share of GNI refuses, while "share", "ratio" and
    "capita", per-head wording on its own, name no base. "GDP" grounds on a menu whose
    slots hold creditor types, since no value's label occurs in the phrase around it;
    "holders of GDP-linked bonds" holds the word but the phrase does not hold the label.
    A quote holding a value's span grounds on the words it leaves outside it: "GDP of
    private creditors" leaves "GDP of" beside the creditor type "private creditors", as
    "total government spending" leaves "total spending" beside the value "Government"."""
    result = _resolve_debt(
        query,
        _ReadingLLM(
            "External debt (% of GNI)", denominator="Gross domestic product", base_words=base_words
        ),
        monkeypatch,
        menu=menu,
    )
    named = result.denominator.asked if result.denominator is not None else None
    assert (named.ref if named is not None else None) == asked
    if asked is not None:
        assert result.no_data_reason == "denominator_not_available"


def test_a_mismatched_denominator_refuses_with_both_denominators(monkeypatch):
    """Ghana's debt to GDP ratio, elected on the share of GNI: the refusal flags the share,
    offers the debt level as nearest_real, and names GDP asked and GNI found."""
    obs = {
        f"{_DEBT_LEVEL}|country/GHA": [
            {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}
        ]
    }
    result = _resolve_debt(
        "debt to GDP ratio",
        _ReadingLLM(
            "External debt (% of GNI)", denominator="Gross domestic product", base_words="GDP"
        ),
        monkeypatch,
        obs=obs,
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    (flagged,) = result.flagged_match or ()
    assert _members(flagged) == [_DEBT_GNI]
    assert result.denominator is not None
    assert result.denominator.asked is not None and result.denominator.asked.ref == _GDP
    assert result.denominator.found is not None and result.denominator.found.ref == _GNI
    (nearest,) = result.nearest_real or ()
    assert _members(nearest) == [_DEBT_LEVEL]
    assert nearest.relaxation.property_dcid == "measurementDenominator"
    assert nearest.relaxation.dropped_value_dcids == [_GNI]
    # The assembled part carries both on the wire.
    (part,) = assemble_region(
        result, extra_warnings=[], start_ms=0, engine_build="test", max_candidates=5
    ).parts
    assert isinstance(part, NoDataPart)
    wire = part.model_dump(mode="json", exclude_none=True)["no_data"]
    assert [s["spec_id"] for s in wire["nearest_real"]] == [nearest.spec_id]
    assert wire["denominator"] == {
        "asked": {"ref": _GDP, "label": "Gross domestic product"},
        "found": {"ref": _GNI, "label": "Gross national income"},
    }


@pytest.mark.parametrize(
    ("elect", "base_words", "also", "status", "unverified"),
    [
        ("External debt (% of GNI)", "exports", (), "definite", ['"Gross national income"']),
        ("External debt (% of GNI)", None, (), "definite", []),
        ("External debt (% of GNI)", "export earnings", (), "definite", []),
        ("External debt (current US$)", "exports", (), "no_data", []),
        (
            "External debt (% of GNI)",
            "exports",
            ("External debt (% of GDP)", "External debt per capita"),
            "candidates",
            ['"Gross national income"', '"Gross domestic product"', '"people"'],
        ),
        ("External debt per capita", "exports", (), "definite", ['"people"']),
    ],
    ids=[
        "quoted",
        "unquoted",
        "words-the-phrase-lacks",
        "quoted-refuses-a-level",
        "candidates-each-warn",
        "per-person-names-people",
    ],
)
def test_none_of_these_leaves_a_shape_with_a_denominator_arc_unverified(
    elect, base_words, also, status, unverified, monkeypatch
):
    """ "debt to exports ratio" names exports, and the read matches it to no DENOMINATORS
    entry. "None of these" is the absence of a match, so it decides nothing about the share
    of GNI's arc: the share answers, warning `denominator_unverified` with the question's
    words for the base beside the arc's graph name, or people for a per-person series. Each
    warning carries the spec_id of the spec it qualifies, a definite answer's one spec or
    each candidate in candidate order. The quoted base still refuses a level, which
    divides by nothing. A read naming "none of these" on no words of the phrase names no
    base, and the share of GNI answers without the warning."""
    result = _resolve_debt(
        "debt to exports ratio",
        _ReadingLLM(elect, denominator=_NONE_OF_THESE, base_words=base_words, also=also),
        monkeypatch,
        menu=[*_DEBT_MENU, _DEBT_GDP, _DEBT_PER_CAPITA],
        obs={f"{sv}|country/GHA": [_OBS] for sv in (_DEBT_GDP, _DEBT_PER_CAPITA)},
    )
    assert result.status == status, result.no_data_reason
    warnings = [w for w in result.warnings if w.code == "denominator_unverified"]
    assert len(warnings) == len(unverified)
    for warning, found in zip(warnings, unverified, strict=True):
        assert '"exports"' in warning.message and found in warning.message
    if warnings:
        assert [w.spec_id for w in warnings] == [spec.spec_id for spec in result.specs]
    if status == "no_data":
        assert result.no_data_reason == "denominator_not_available"
        assert result.denominator is not None
        assert result.denominator.asked is None
        assert result.denominator.found is None


@pytest.mark.parametrize(
    ("denominator", "nodes"),
    [
        ("Gross domestic product", _DEBT_NODES),
        (_NONE_OF_THESE, _DEBT_NODES),
        (_GDP, {**_DEBT_NODES, _GDP: _unnamed_sv_node(_GDP, "EconomicActivity", "amount")}),
    ],
    ids=["listed", "none-of-these", "listed-unnamed"],
)
def test_a_series_stating_its_base_only_in_its_name_answers_unverified(
    denominator, nodes, monkeypatch
):
    """The warning quotes the question's own words for the base, "GDP", whichever entry the
    read named, and so never the dcid a denominator the graph gives no name is listed
    under."""
    result = _resolve_debt(
        "debt to GDP ratio",
        _ReadingLLM("External debt to GDP ratio", denominator=denominator, base_words="GDP"),
        monkeypatch,
        nodes=nodes,
    )
    assert result.status == "definite", result.no_data_reason
    (warning,) = [w for w in result.warnings if w.code == "denominator_unverified"]
    assert '"GDP"' in warning.message
    assert _GDP not in warning.message


@pytest.mark.parametrize(
    "query,elect,base_words,stated_form,status,traced",
    [
        # "debt to GDP" names no form a phrase pattern reads; the model's rate joins
        # beside its quote of the base.
        ("debt to GDP", "External debt (current US$)", "GDP", "rate", "no_data", None),
        ("debt to GDP", "External debt (current US$)", "GDP", None, "definite", None),
        ("debt to GDP", "External debt to GDP ratio", "GDP", "rate", "definite", "rate"),
        # An unquoted rate joins: a share of people quotes no base by design.
        ("debt to GDP", "External debt (current US$)", None, "rate", "no_data", None),
        # "ratio" names a rate, so the model's form is not read.
        ("debt to GDP ratio", "External debt (% of GNI)", None, "per_person", "definite", None),
    ],
    ids=[
        "stated-rate-refuses-a-level",
        "none-stated",
        "stated-rate-traced",
        "unquoted-rate-joins",
        "phrase-form-wins",
    ],
)
def test_a_stated_form_joins_only_when_no_phrase_names_one(
    query, elect, base_words, stated_form, status, traced, monkeypatch
):
    result = _resolve_debt(
        query, _ReadingLLM(elect, base_words=base_words, stated_form=stated_form), monkeypatch
    )
    assert result.status == status, result.no_data_reason
    if status == "definite":
        assert result.specs[0].resolved.trace.stated_form == traced


class _QuantityReadFailsLLM(_ReadingLLM):
    """Elects as `_ReadingLLM` does, and fails every quantity read as a transport error or
    an output that fails validation does."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            raise LLMInfraError("LLM output failed schema validation")
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def test_a_failed_quantity_read_answers_as_a_phrase_naming_no_form(monkeypatch):
    """ "debt to GDP" names no form a phrase reads. A failed quantity read leaves the part
    answering on the elected level, as a read that states no form does
    (`test_a_stated_form_joins_only_when_no_phrase_names_one[none-stated]`), where a stated
    rate would refuse it."""
    result = _resolve_debt(
        "debt to GDP",
        _QuantityReadFailsLLM("External debt (current US$)", stated_form="rate"),
        monkeypatch,
    )
    assert result.status == "definite", result.no_data_reason
    assert result.specs[0].resolved.trace.stated_form is None


def test_an_alternate_dividing_by_another_base_is_not_offered(monkeypatch):
    """Ghana's debt to GDP ratio elects the share of GDP, which holds no data there; the
    nominated share of GNI holds data but divides by another base than the GDP the question
    names, so the alternate guard refuses it as the gate would, and the answer offers no
    alternate."""
    result = _resolve_debt(
        "debt to GDP ratio",
        _ReadingLLM(
            "External debt (% of GDP)",
            denominator="Gross domestic product",
            base_words="GDP",
            also=("External debt (% of GNI)",),
        ),
        monkeypatch,
        menu=[_DEBT_GDP, _DEBT_GNI, _MILITARY_GDP],
        without_data=frozenset({_DEBT_GDP}),
        obs={f"{_DEBT_GNI}|country/GHA": [_OBS]},
    )
    assert result.status == "no_data"
    assert result.alternate_match is None
    assert result.gate != "only_alternate_has_data"


def test_an_elected_level_answers_nothing_when_the_alternate_that_serves_holds_no_data(
    monkeypatch,
):
    """The elected dollar level serves no rate and holds data; the nominated share of GDP
    serves the rate and holds none. With no reading that serves the quantity left, the
    refusal the gate deferred to the fan-out stands, flagging the level."""
    result = _resolve_debt(
        "debt to GDP ratio",
        _ReadingLLM(
            "External debt (current US$)",
            denominator="Gross domestic product",
            base_words="GDP",
            also=("External debt (% of GDP)",),
        ),
        monkeypatch,
        menu=[_DEBT_LEVEL, _DEBT_GDP, _MILITARY_GDP],
        without_data=frozenset({_DEBT_GDP}),
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    (flagged,) = result.flagged_match or ()
    assert _members(flagged) == [_DEBT_LEVEL]


def test_an_unverified_series_without_data_raises_no_unverified_warning(monkeypatch):
    """The warning describes the series an answer serves: a UN ratio elected without data
    serves nothing, so the no-data answer carries no `denominator_unverified`."""
    result = _resolve_debt(
        "debt to GDP ratio",
        _ReadingLLM(
            "External debt to GDP ratio", denominator="Gross domestic product", base_words="GDP"
        ),
        monkeypatch,
        without_data=frozenset({_DEBT_UN}),
    )
    assert result.status == "no_data"
    assert not [w for w in result.warnings if w.code == "denominator_unverified"]


@pytest.mark.parametrize(
    ("base_words", "stated_form", "status"),
    [
        (None, None, "definite"),
        ("Ghana", None, "definite"),
        ("private creditors", None, "definite"),
        ("private creditors", "rate", "definite"),
        ("private", None, "definite"),
        ("to private creditors", None, "definite"),
        ("to private creditors", "rate", "definite"),
        ("to", "rate", "definite"),
        ("Ghana", "rate", "definite"),
        (None, "rate", "no_data"),
        (None, "share", "no_data"),
        (None, "per_head_base", "no_data"),
    ],
    ids=[
        "no-form",
        "base-on-the-place",
        "base-on-the-creditor-type",
        "rate-on-the-creditor-type",
        "base-on-part-of-the-creditor-type",
        "base-on-the-creditor-type-and-its-cue",
        "rate-on-the-creditor-type-and-its-cue",
        "rate-on-a-form-word",
        "rate-on-the-place",
        "unquoted-rate",
        "unquoted-share",
        "stated-per-head-base",
    ],
)
def test_the_quantity_read_reads_no_word_a_place_owns(base_words, stated_form, status, monkeypatch):
    """dq-ids-008 with Ghana left in the variable phrase: Ghana's door role carries its own
    span, so the quantity read reads the phrase without "Ghana", and its role's cue "to"
    stays. A read naming "none of these" on "Ghana" names no base, and the debt service
    level answers. Nor does one on "private creditors", on "private" or on "to private
    creditors": each leaves nothing or a function word outside the span of the phrase that
    the label of a value a menu shape's slot holds occupies, "private creditors (IDS
    creditor type)" read without its property. The live read states a
    rate beside that quote, the form of a division by the creditor type it took for a
    base; a rate or share drops beside a quote that names no base, as it does beside a
    quote of form words alone ("to") or of the place ("Ghana"). An unquoted rate or share
    stands and refuses the level: a share of people quotes no base by design, as
    dq-ext-005's read of "HIV prevalence among young women" (a share, no quote) does. A
    per-head form names its base in its own words, so a stated one stands too."""
    llm = _ReadingLLM(
        "External debt (current US$)",
        denominator=None if base_words is None else _NONE_OF_THESE,
        base_words=base_words,
        stated_form=stated_form,
    )
    result, _graph, _llm = _resolve(
        "Ghana debt service to private creditors",
        nodes=_DEBT_NODES,
        svs=_CREDITOR_MENU,
        place=_GHANA,
        monkeypatch=monkeypatch,
        llm=llm,
        roles=[DoorRole(axis="destination", span="Ghana")],
    )
    assert llm.quantity_prompts[0].splitlines()[0] == (
        "Variable phrase: debt service to private creditors"
    )
    assert result.status == status, result.no_data_reason
