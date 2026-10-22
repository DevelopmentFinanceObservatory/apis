"""`bind_repair.apply_shape_slot_defaults`: an open `medicalStatus` slot on a births shape
binds `LiveBirth`, so "birth rate" reads the crude birth rate."""

from __future__ import annotations

import asyncio
import logging

from code_index.shared import get_index
from search.engine.bind import (
    DraftBindingEvidence,
    DraftBindingKind,
    SlotBindingDraft,
    _UniversalBindOutput,
)
from search.engine.bind_repair import apply_shape_slot_defaults
from search.engine.errors import GraphInfraError
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import SLOT_DEFAULT_NOT_APPLIED
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.shape import ShapeDraft, shape_draft_from
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_ETH = "country/ETH"
_BIRTH_RATE_QUERY = "birth rate"
_BIRTHS_PER_CAPITA = "Count_BirthEvent_AsAFractionOfCount_Person"
_CRUDE_BIRTH_RATE = "Count_BirthEvent_LiveBirth_AsFractionOf_Count_Person"
_FERTILITY_RATE = "FertilityRate_Person_Female"
_UN_TOTAL_FERTILITY_RATE = "undata/unfpa/FERT_RATE_TOTAL"
_FACET = {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}


def _arc(*dcids: str) -> dict:
    return {"nodes": [{"dcid": dcid} for dcid in dcids]}


def _constrained_sv_node(
    dcid: str,
    *,
    population_type: str,
    meas_prop: str,
    constraints: dict[str, str],
    denominator: str | None = None,
) -> dict:
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


def _leg(*scored: tuple[str, float]) -> dict:
    return {"svs": [dcid for dcid, _ in scored], "cosine_scores": [score for _, score in scored]}


def _birth_rate_graph(*, un_score: float) -> FakeGraph:
    """The reads behind "birth rate Ethiopia" on the dev graph. The births-per-capita
    family's total has no data at Ethiopia and dominates the crude birth rate, which has;
    the fertility rate the model nominates as an alternate has data, and so does the UN's
    total fertility rate, recalled at `un_score`."""
    births = {"population_type": "BirthEvent", "meas_prop": "count", "denominator": "Count_Person"}
    return FakeGraph(
        nodes={
            _ETH: {"label": "Ethiopia", "type": "Country"},
            _BIRTHS_PER_CAPITA: _constrained_sv_node(_BIRTHS_PER_CAPITA, constraints={}, **births),
            _CRUDE_BIRTH_RATE: _constrained_sv_node(
                _CRUDE_BIRTH_RATE, constraints={"medicalStatus": "LiveBirth"}, **births
            ),
            _FERTILITY_RATE: _constrained_sv_node(
                _FERTILITY_RATE,
                population_type="Person",
                meas_prop="fertilityRate",
                constraints={"gender": "Female"},
            ),
            _UN_TOTAL_FERTILITY_RATE: _constrained_sv_node(
                _UN_TOTAL_FERTILITY_RATE,
                population_type="undata/unfpa/SERIES-FERT_RATE_TOTAL",
                meas_prop="value",
                constraints={},
            ),
        },
        obs={
            f"{sv}|{_ETH}": [_FACET]
            for sv in (_FERTILITY_RATE, _UN_TOTAL_FERTILITY_RATE, _CRUDE_BIRTH_RATE)
        },
        detect={
            _BIRTH_RATE_QUERY: _leg(
                (_BIRTHS_PER_CAPITA, 1.0), (_CRUDE_BIRTH_RATE, 1.0), (_FERTILITY_RATE, 0.865)
            )
        },
        un_detect={_BIRTH_RATE_QUERY: _leg((_UN_TOTAL_FERTILITY_RATE, un_score))},
        resolve={"Ethiopia": _ETH},
    )


class _RecordedBirthRateLLM:
    """Answers the bind as the live model did for "birth rate Ethiopia": the births shape,
    medicalStatus unbound, and the fertility rate as a plausible alternate. Records each
    menu it saw."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        recorded = _UniversalBindOutput.model_validate_json(
            '{"chosen_shape_index":0,"bindings":[{"axis":"how","property_dcid":"medicalStatus",'
            '"kind":"unbound","value_dcids":[],"evidence":"mentioned",'
            '"requested_all_phrase":null}],"also_plausible":[1],"ask":null,"unplaced":[],'
            '"measuring":[{"index":0,"rating":1.0},{"index":1,"rating":0.9}]}'
        )
        return recorded, {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}


def _resolve(graph: FakeGraph, *, query: str, entities: list[str], llm) -> RegionResult:
    return asyncio.run(
        resolve_variable(
            query,
            entities=entities,
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=llm,
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )


def _shape(
    *, population_type: str = "BirthEvent", taxonomy: tuple[str, ...] = ("LiveBirth",)
) -> ShapeDraft:
    """A shape with one `medicalStatus` slot whose taxonomy holds `taxonomy`."""
    return shape_draft_from(
        shape_id="births_per_capita",
        label="Count Of Birth Event (Per Capita)",
        pop_type_dcid=population_type,
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid="Count_Person",
        constraint_props=["medicalStatus"],
        prop_labels={"medicalStatus": "medicalStatus"},
        prop_observed_values={"medicalStatus": list(taxonomy)},
        sv_arc_facts={},
        slot_taxonomy={"how:medicalStatus": list(taxonomy)},
        place_available=True,
        value_labels=None,
        recalled=False,
    )


def _status(
    kind: DraftBindingKind = "unbound",
    *,
    value: str | None = None,
    evidence: DraftBindingEvidence = "mentioned",
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid="medicalStatus",
        kind=kind,
        value_dcids=[] if value is None else [value],
        evidence=evidence,
    )


def test_the_index_declares_live_births_and_debt_outstanding_as_slot_defaults() -> None:
    defaults = get_index().shape_slot_defaults
    assert [(d.population_type, d.property_dcid, d.value_dcid) for d in defaults] == [
        ("BirthEvent", "medicalStatus", "LiveBirth"),
        ("FinancialPosition", "financialPositionType", "AmountOutstanding"),
    ]


def test_an_unbound_status_on_a_births_shape_binds_live_birth() -> None:
    draft = _status()

    repaired, defaulted, _relaxed = apply_shape_slot_defaults([draft], _shape(), index=get_index())

    assert defaulted == [draft]
    assert [(b.property_dcid, b.kind, b.value_dcids) for b in repaired] == [
        ("medicalStatus", "value", ["LiveBirth"])
    ]


def test_a_status_drafted_absent_binds_live_birth() -> None:
    repaired, defaulted, _relaxed = apply_shape_slot_defaults(
        [_status("absent")], _shape(), index=get_index()
    )

    assert len(defaulted) == 1
    assert repaired[0].kind == "value"
    assert repaired[0].value_dcids == ["LiveBirth"]


def test_a_status_the_binder_bound_keeps_its_value() -> None:
    bound = _status("value", value="Stillbirth")

    repaired, defaulted, _relaxed = apply_shape_slot_defaults(
        [bound], _shape(taxonomy=("LiveBirth", "Stillbirth")), index=get_index()
    )

    assert defaulted == []
    assert repaired == [bound]


def test_a_status_the_query_asked_for_in_full_stays_open() -> None:
    requested = _status(evidence="requested_all")

    repaired, defaulted, _relaxed = apply_shape_slot_defaults(
        [requested], _shape(), index=get_index()
    )

    assert defaulted == []
    assert repaired == [requested]


def test_another_population_type_keeps_its_open_status() -> None:
    draft = _status()

    repaired, defaulted, _relaxed = apply_shape_slot_defaults(
        [draft], _shape(population_type="MortalityEvent"), index=get_index()
    )

    assert defaulted == []
    assert repaired == [draft]


def test_birth_rate_ethiopia_answers_the_crude_birth_rate_beside_the_fertility_rate(
    caplog,
) -> None:
    """The reads and the bind recorded for "birth rate Ethiopia": the model elects births per
    capita with medicalStatus unbound and nominates the fertility rate. Our pass answers, so
    the UN's total fertility rate never reaches a menu."""
    llm = _RecordedBirthRateLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(
            _birth_rate_graph(un_score=0.73), query="birth rate", entities=["Ethiopia"], llm=llm
        )

    assert result.status == "candidates"
    assert [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs] == [
        [_CRUDE_BIRTH_RATE],
        [_FERTILITY_RATE],
    ]
    crude_birth_rate = result.specs[0]
    assert [
        (f.key.property.ref, [ref.ref for ref in f.refs], f.provenance)
        for f in crude_birth_rate.resolved.trace.slot_filters
        if f.key.property is not None
    ] == [("medicalStatus", ["LiveBirth"], "declared_default")]
    (menu,) = llm.prompts
    assert _UN_TOTAL_FERTILITY_RATE not in menu
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gap_fill")]
    assert line.startswith("gap_fill part=0 status=candidates/None pool=ours ")
    assert "un_pass=skipped:answered " in line


class _ProbeFailsGraph(FakeGraph):
    """A graph whose place-availability probe, the first batch read, raises, so the shape's
    taxonomy lists every member and `place_available` is unknown. Later reads succeed."""

    _probed = False

    def observation_facets_batch(self, *args, **kwargs):
        if not self._probed:
            self._probed = True
            raise GraphInfraError("simulated availability probe failure")
        return super().observation_facets_batch(*args, **kwargs)


def test_a_failed_availability_probe_leaves_the_status_open_and_the_total_answers() -> None:
    """Only the births total has data at Ethiopia. With the probe down the taxonomy still
    lists `LiveBirth`, and a default bound from it would read the crude birth rate: no data."""
    births = {"population_type": "BirthEvent", "meas_prop": "count", "denominator": "Count_Person"}
    graph = _ProbeFailsGraph(
        nodes={
            _ETH: {"label": "Ethiopia", "type": "Country"},
            _BIRTHS_PER_CAPITA: _constrained_sv_node(_BIRTHS_PER_CAPITA, constraints={}, **births),
            _CRUDE_BIRTH_RATE: _constrained_sv_node(
                _CRUDE_BIRTH_RATE, constraints={"medicalStatus": "LiveBirth"}, **births
            ),
        },
        obs={f"{_BIRTHS_PER_CAPITA}|{_ETH}": [_FACET]},
        detect={
            _BIRTH_RATE_QUERY: _leg((_BIRTHS_PER_CAPITA, 1.0), (_CRUDE_BIRTH_RATE, 1.0)),
        },
        un_detect={},
        resolve={"Ethiopia": _ETH},
    )

    result = _resolve(
        graph, query=_BIRTH_RATE_QUERY, entities=["Ethiopia"], llm=_RecordedBirthRateLLM()
    )

    assert result.status != "no_data"
    assert [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs] == [
        [_BIRTHS_PER_CAPITA]
    ]


def _births_total_only_graph(graph_class: type[FakeGraph] = FakeGraph) -> FakeGraph:
    """The births-per-capita family at Ethiopia where only its total has data: the crude
    birth rate, the one member reading `LiveBirth`, holds none there."""
    births = {"population_type": "BirthEvent", "meas_prop": "count", "denominator": "Count_Person"}
    return graph_class(
        nodes={
            _ETH: {"label": "Ethiopia", "type": "Country"},
            _BIRTHS_PER_CAPITA: _constrained_sv_node(_BIRTHS_PER_CAPITA, constraints={}, **births),
            _CRUDE_BIRTH_RATE: _constrained_sv_node(
                _CRUDE_BIRTH_RATE, constraints={"medicalStatus": "LiveBirth"}, **births
            ),
        },
        obs={f"{_BIRTHS_PER_CAPITA}|{_ETH}": [_FACET]},
        detect={
            _BIRTH_RATE_QUERY: _leg((_BIRTHS_PER_CAPITA, 1.0), (_CRUDE_BIRTH_RATE, 1.0)),
        },
        un_detect={},
        resolve={"Ethiopia": _ETH},
    )


def test_a_default_no_member_with_data_holds_is_not_applied_and_the_total_answers() -> None:
    """Derive probes both members at Ethiopia and finds data on the total alone. The live-birth
    default would read the crude birth rate, which has none there, so the slot stays open,
    the total answers definite, and a warning says the default was not applied."""
    result = _resolve(
        _births_total_only_graph(),
        query=_BIRTH_RATE_QUERY,
        entities=["Ethiopia"],
        llm=_RecordedBirthRateLLM(),
    )

    assert result.status == "definite"
    assert [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs] == [
        [_BIRTHS_PER_CAPITA]
    ]
    assert [
        f.key.property.ref
        for f in result.specs[0].resolved.trace.slot_filters
        if f.key.property is not None and f.refs
    ] == []
    assert [w.code for w in result.warnings if w.code == SLOT_DEFAULT_NOT_APPLIED] == [
        SLOT_DEFAULT_NOT_APPLIED
    ]
