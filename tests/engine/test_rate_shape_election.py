"""An explicit rate request filters raw-count shapes before ranking and the single bind call."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.config import settings
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.phase_rank_bind import (
    _has_strong_rate_intent,
    _is_unnormalised_raw_count,
    _mentioned_raw_qualifier_props,
    _prune_explicit_rate_raw_counts,
    _raw_count_is_pending_joint_slice,
)
from search.engine.region.result import RegionResult
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import Binding, BindingSet, BindingUnbound, BindingValue
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_KENYA = "country/KEN"


def _arcs(
    label: str,
    *,
    population_type: str = "Person",
    stat_type: str = "measuredValue",
    denominator: str | None = None,
    unresolved_denominator: bool = False,
    constraints: dict[str, str] | None = None,
    roles: tuple[str, ...] = (),
) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": population_type}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": stat_type}]},
    }
    if denominator is not None:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": denominator}]}
    if unresolved_denominator:
        arcs["measurementDenominator"] = {"nodes": [{}]}
    if constraints:
        arcs["constraintProperties"] = {
            "nodes": [{"dcid": property_dcid} for property_dcid in constraints]
        }
        arcs.update(
            {
                property_dcid: {"nodes": [{"dcid": value_dcid}]}
                for property_dcid, value_dcid in constraints.items()
            }
        )
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return arcs


def _node(label: str, **kwargs: Any) -> dict:
    return {"label": label, "type": "StatisticalVariable", "arcs": _arcs(label, **kwargs)}


def _shape_with_members(
    *,
    shape_id: str,
    label: str,
    denominator: str | None,
    properties: tuple[str, ...],
    members: list[dict[str, str]],
    value_labels: dict[str, str],
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=label,
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=denominator,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=property_dcid, label=property_dcid)
            for property_dcid in properties
        ),
        sv_arc_facts={
            f"{shape_id}_{index}": _arcs(label, constraints=constraints)
            for index, constraints in enumerate(members)
        },
        value_labels=value_labels,
    )


class _RecordingBindLLM:
    def __init__(
        self,
        *,
        chosen_shape_index: int = 0,
        bindings: list[SlotBindingDraft] | None = None,
        ask: str | None = None,
        usage: dict | None = None,
    ) -> None:
        self.chosen_shape_index = chosen_shape_index
        self.bindings = [] if bindings is None else bindings
        self.ask = ask
        self.usage = usage
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        return (
            _UniversalBindOutput(
                chosen_shape_index=self.chosen_shape_index,
                bindings=self.bindings,
                ask=self.ask,
            ),
            self.usage,
        )


class _UnreachableLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, **_kwargs):
        self.calls += 1
        raise AssertionError("the pre-bind rate filter must stop before the universal binder")


def _run(
    variable: str,
    *,
    nodes: dict,
    svs: list[str],
    scores: list[float],
    llm,
    with_data: list[str] | None = None,
) -> RegionResult:
    nodes = {"country/KEN": {"label": "Kenya", "type": "Country"}, **nodes}
    graph = FakeGraph(
        nodes=nodes,
        detect={variable: {"svs": svs, "cosine_scores": scores}},
        resolve={"Kenya": _KENYA},
        obs={
            f"{stat_var}|{_KENYA}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
            for stat_var in (with_data or [])
        },
    )
    return asyncio.run(
        resolve_variable(
            variable,
            entities=["Kenya"],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm,
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


def _slots_by_property(result: RegionResult) -> dict[str, Binding]:
    return {
        slot.key.property.ref: slot.binding
        for slot in result.spec.resolved.slots
        if slot.key.property is not None
    }


def test_explicit_rate_filter_precedes_top_k_ranking() -> None:
    raw_stat_vars = [f"RawCount{index}" for index in range(settings().bind_top_k + 1)]
    rate_stat_var = "RateBelowRawTopK"
    raw_nodes = {
        stat_var: _node(
            f"raw count {index}",
            population_type=f"RawPopulation{index}",
        )
        for index, stat_var in enumerate(raw_stat_vars)
    }
    nodes = {
        **raw_nodes,
        rate_stat_var: _node("rate below raw top-k", denominator="Count_Person"),
    }
    llm = _RecordingBindLLM()

    result = _run(
        "illustrative rate",
        nodes=nodes,
        svs=[*raw_stat_vars, rate_stat_var],
        scores=[0.95 - index * 0.01 for index in range(len(raw_stat_vars))] + [0.70],
        llm=llm,
        with_data=[rate_stat_var],
    )

    assert result.status == "definite"
    assert len(llm.prompts) == 1
    assert "CANDIDATE SHAPES (1 total)" in llm.prompts[0]
    assert "rate below raw top-k" in llm.prompts[0]
    assert all(f"raw count {index}" not in llm.prompts[0] for index in range(len(raw_stat_vars)))
    denominator = result.spec.resolved.shape.measurement_denominator
    assert denominator is not None
    assert denominator.ref == "Count_Person"


def test_multiple_rate_candidates_remain_for_the_single_model_election() -> None:
    raw_stat_var = "RawCount"
    first_rate = "RatePerPerson"
    second_rate = "RatePerHousehold"
    llm = _RecordingBindLLM(chosen_shape_index=1)

    result = _run(
        "illustrative rate",
        nodes={
            raw_stat_var: _node("raw count"),
            first_rate: _node("rate per person", denominator="Count_Person"),
            second_rate: _node("rate per household", denominator="Count_Household"),
        },
        svs=[raw_stat_var, first_rate, second_rate],
        scores=[0.95, 0.90, 0.80],
        llm=llm,
        with_data=[first_rate, second_rate],
    )

    assert result.status == "definite"
    assert len(llm.prompts) == 1
    assert "CANDIDATE SHAPES (2 total)" in llm.prompts[0]
    assert "rate per person" in llm.prompts[0]
    assert "rate per household" in llm.prompts[0]
    assert "raw count" not in llm.prompts[0]
    denominator = result.spec.resolved.shape.measurement_denominator
    assert denominator is not None
    assert denominator.ref == "Count_Household"


@pytest.mark.parametrize(
    ("variable", "status"),
    [("illustrative count", "definite"), ("illustrative rate count", "no_data")],
)
def test_count_or_ambiguous_rate_count_keeps_raw_count_candidates(
    variable: str, status: str
) -> None:
    """Bind is offered the raw count either way. "Rate" names a rate, which an elected raw
    count does not serve, so that election refuses at materialise."""
    raw_stat_var = "RawCount"
    rate_stat_var = "RatePerPerson"
    llm = _RecordingBindLLM()

    result = _run(
        variable,
        nodes={
            raw_stat_var: _node("raw count"),
            rate_stat_var: _node("rate per person", denominator="Count_Person"),
        },
        svs=[raw_stat_var, rate_stat_var],
        scores=[0.95, 0.80],
        llm=llm,
        with_data=[raw_stat_var],
    )

    assert result.status == status
    assert len(llm.prompts) == 1
    assert "raw count" in llm.prompts[0]
    answer = result.spec if status == "definite" else result.flagged_match[0]
    assert answer.resolved.shape.measurement_denominator is None


@pytest.mark.parametrize(
    ("label", "node_kwargs"),
    [
        ("population growth", {"stat_type": "growthRate"}),
        ("unresolved denominator", {"unresolved_denominator": True}),
    ],
)
def test_rate_filter_preserves_non_raw_count_shapes(label: str, node_kwargs: dict) -> None:
    raw_stat_var = "RawCount"
    preserved_stat_var = "PreservedShape"
    llm = _RecordingBindLLM()

    _run(
        "illustrative percentage",
        nodes={
            raw_stat_var: _node("raw count"),
            preserved_stat_var: _node(label, **node_kwargs),
        },
        svs=[raw_stat_var, preserved_stat_var],
        scores=[0.95, 0.80],
        llm=llm,
    )

    assert len(llm.prompts) == 1
    assert label in llm.prompts[0]
    assert "raw count" not in llm.prompts[0]


def test_no_rate_eligible_shape_stops_before_the_model() -> None:
    llm = _UnreachableLLM()

    result = _run(
        "illustrative %",
        nodes={"RawCount": _node("raw count")},
        svs=["RawCount"],
        scores=[0.90],
        llm=llm,
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"
    assert llm.calls == 0
    assert "bind" not in result.timing_by_step


def test_structural_rate_filter_keeps_ordinary_normalized_menu_order() -> None:
    llm = _RecordingBindLLM()

    result = _run(
        "incidence rate",
        nodes={
            "RawIncidence": _node("incidence count"),
            "IncidenceRate": _node("incidence rate", denominator="Count_Person"),
            "ImmunizationRate": _node(
                "immunization rate",
                population_type="Immunization",
                denominator="Count_Person",
            ),
        },
        svs=["RawIncidence", "IncidenceRate", "ImmunizationRate"],
        scores=[0.90, 0.85, 0.80],
        llm=llm,
        with_data=["IncidenceRate"],
    )

    assert result.status == "definite"
    assert len(llm.prompts) == 1
    assert "CANDIDATE SHAPES (2 total)" in llm.prompts[0]
    assert "incidence rate" in llm.prompts[0]
    assert "immunization rate" in llm.prompts[0]


class _CountingFixtureLLM:
    def __init__(self) -> None:
        self.delegate = FakeLLM()
        self.bind_prompts: list[str] = []

    @property
    def bind_calls(self) -> int:
        return len(self.bind_prompts)

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "_UniversalBindOutput":
            self.bind_prompts.append(prompt)
        return await self.delegate.generate_structured(prompt=prompt, system=system, schema=schema)


def test_hiv_incidence_retains_raw_menu_and_elects_pinned_sdg_shape() -> None:
    llm = _CountingFixtureLLM()
    response = offline_resolve(make_request("HIV incidence rate Kenya"), llm=llm)

    assert response.status == "definite"
    assert llm.bind_calls == 1
    assert [stat_var.ref.ref for stat_var in response.parts[0].spec.resolved.members] == [
        "sdg/SH_HIV_INCD"
    ]


def test_recorded_measles_menu_retains_raw_count_and_refuses_its_form() -> None:
    """Bind elects the measles case count, which serves no rate, so the answer refuses
    `denominator_not_available`, flagging the one count materialise elected."""
    response = offline_resolve(make_request("measles incidence rate Kenya"), llm=FakeLLM())

    assert response.status == "no_data"
    part = response.parts[0]
    assert part.no_data.reason == "denominator_not_available"
    assert part.no_data.elected_count == 1
    assert part.no_data.flagged_match is not None
    assert [sv.ref.ref for sv in part.no_data.flagged_match.resolved.members] == [
        "Count_MedicalConditionIncident_ConditionMeasles"
    ]


def test_recorded_hiv_prevalence_menu_refuses_the_sdg_incidence_shape() -> None:
    """Recorded "HIV prevalence" reaches SDG 3.3.1, new HIV infections per 1,000
    uninfected, through `dc/topic/HIV` above the floor. Its population type's name names
    incidence, so the shape leaves the menu before bind (#1020)."""
    llm = _CountingFixtureLLM()
    offline_resolve(make_request("HIV prevalence in Nigeria"), llm=llm)

    assert llm.bind_prompts
    assert all("SDG_SH_HIV_INCD" not in prompt for prompt in llm.bind_prompts)


def test_recorded_female_urban_menu_retains_joint_slice_carrier() -> None:
    response = offline_resolve(make_request("female urban literacy rate"), llm=FakeLLM())

    assert response.status == "no_data"
    part = response.parts[0]
    assert part.no_data.reason == "no_matching_slice"
    assert part.no_data.gate == "no_member_asserts_bound_properties"
    assert part.no_data.elected_count == 1


def test_joint_slice_deferral_keeps_data_bearing_alternate_when_raw_is_unavailable() -> None:
    """A fully bound raw member with no data is not a missing joint slice.

    The default place scope contains data for the model-nominated alternate,
    while the elected raw family has a real Female+Urban member that lacks
    observations. Its Materialised carrier is therefore availability=False,
    not the availability=None missing-slice case that may force the
    single-shape path. The raw count serves no rate, but the nominated rate
    alternate does, so the quantity gate lets the fan-out offer it.
    """
    variable = "female urban literacy rate"
    raw_female_urban = "RawFemaleUrbanLiteracy"
    raw_male_rural = "RawMaleRuralLiteracy"
    normalized_literacy = "NormalizedLiteracy"
    data_bearing_neighbor = "DataBearingNeighbor"

    class _FullyBoundRawLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema is QuantityOutput:
                return NO_QUANTITY
            assert schema.__name__ == "_UniversalBindOutput"
            return (
                _UniversalBindOutput(
                    chosen_shape_index=0,
                    bindings=[
                        SlotBindingDraft(
                            axis="how",
                            property_dcid="gender",
                            kind="value",
                            value_dcids=["Female"],
                        ),
                        SlotBindingDraft(
                            axis="how",
                            property_dcid="residence",
                            kind="value",
                            value_dcids=["Urban"],
                        ),
                    ],
                    also_plausible=[1],
                ),
                None,
            )

    nodes = {
        raw_female_urban: _node(
            "Female urban literacy count",
            constraints={"gender": "Female", "residence": "Urban"},
        ),
        raw_male_rural: _node(
            "Male rural literacy count",
            constraints={"gender": "Male", "residence": "Rural"},
        ),
        normalized_literacy: _node("Literacy rate", denominator="Count_Person"),
        data_bearing_neighbor: _node(
            "Female urban unemployment rate",
            population_type="Unemployment",
            denominator="Count_Person",
            constraints={"gender": "Female", "residence": "Urban"},
        ),
        "Female": {"label": "Female", "type": "GenderType"},
        "Male": {"label": "Male", "type": "GenderType"},
        "Urban": {"label": "Urban", "type": "ResidenceType"},
        "Rural": {"label": "Rural", "type": "ResidenceType"},
    }
    graph = FakeGraph(
        nodes=nodes,
        detect={
            variable: {
                "svs": [
                    raw_female_urban,
                    raw_male_rural,
                    data_bearing_neighbor,
                    normalized_literacy,
                ],
                "cosine_scores": [0.90, 0.89, 0.885, 0.85],
            }
        },
        resolve={},
        obs={
            f"{data_bearing_neighbor}|country/A1": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
            ]
        },
        children={"Earth": {"Country": ["country/A1"]}},
    )

    result = asyncio.run(
        resolve_variable(
            variable,
            entities=[],
            date_request=None,
            detect_query=variable,
            role_query=variable,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FullyBoundRawLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "no_data"
    assert result.gate == "only_alternate_has_data"
    assert result.alternate_match is not None
    assert [stat_var.ref.ref for stat_var in result.alternate_match.resolved.members] == [
        data_bearing_neighbor
    ]


def test_single_bind_output_passes_value_set_unbound_and_absent_without_transfer() -> None:
    first_rate = "RateCategoryA"
    second_rate = "RateCategoryB"
    bindings = [
        SlotBindingDraft(
            axis="how", property_dcid="category", kind="set", value_dcids=["CategoryA", "CategoryB"]
        ),
        SlotBindingDraft(axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]),
        SlotBindingDraft(axis="how", property_dcid="age", kind="unbound", value_dcids=[]),
        SlotBindingDraft(axis="how", property_dcid="function", kind="absent", value_dcids=[]),
    ]
    llm = _RecordingBindLLM(bindings=bindings)

    result = _run(
        "illustrative rate",
        nodes={
            first_rate: _node(
                "category a rate",
                denominator="Count_Person",
                constraints={
                    "category": "CategoryA",
                    "gender": "Female",
                    "age": "Years10Onwards",
                    "function": "FunctionA",
                },
            ),
            second_rate: _node(
                "category b rate",
                denominator="Count_Person",
                constraints={"category": "CategoryB", "gender": "Female", "age": "Years25Onwards"},
            ),
        },
        svs=[first_rate, second_rate],
        scores=[0.90, 0.80],
        llm=llm,
        with_data=[first_rate, second_rate],
    )

    assert result.status == "definite"
    assert len(llm.prompts) == 1
    slots = _slots_by_property(result)
    assert isinstance(slots["category"], BindingSet)
    assert [value.ref.ref for value in slots["category"].values if value.ref is not None] == [
        "CategoryA",
        "CategoryB",
    ]
    assert isinstance(slots["gender"], BindingValue)
    assert slots["gender"].value.ref is not None
    assert slots["gender"].value.ref.ref == "Female"
    assert isinstance(slots["age"], BindingUnbound)
    assert isinstance(slots["function"], BindingUnbound)


def test_rate_filter_never_transfers_a_cross_member_raw_binding_combination() -> None:
    raw_female = "RawFemale"
    raw_literate = "RawLiterate"
    rate_literate = "RateLiterate"
    llm = _RecordingBindLLM(
        bindings=[
            SlotBindingDraft(
                axis="how", property_dcid="literacyStatus", kind="value", value_dcids=["Literate"]
            ),
            SlotBindingDraft(
                axis="how", property_dcid="age", kind="value", value_dcids=["Years10Onwards"]
            ),
        ]
    )

    result = _run(
        "female literacy rate",
        nodes={
            raw_female: _node("female count", constraints={"gender": "Female"}),
            raw_literate: _node("literate count", constraints={"literacyStatus": "Literate"}),
            rate_literate: _node(
                "literate rate",
                denominator="Count_Person",
                constraints={"literacyStatus": "Literate", "age": "Years10Onwards"},
            ),
        },
        svs=[raw_female, raw_literate, rate_literate],
        scores=[0.95, 0.90, 0.80],
        llm=llm,
        with_data=[rate_literate],
    )

    assert result.status == "definite"
    assert len(llm.prompts) == 1
    assert "female count" not in llm.prompts[0]
    assert "literate count" not in llm.prompts[0]
    assert [stat_var.ref.ref for stat_var in result.spec.resolved.members] == [rate_literate]
    assert "gender" not in _slots_by_property(result)


def test_partial_raw_set_is_not_transferred_to_a_rate_candidate() -> None:
    raw_a = "RawCategoryA"
    raw_b = "RawCategoryB"
    partial_rate = "RateCategoryA"
    llm = _RecordingBindLLM(ask="which category?")

    result = _run(
        "category rate",
        nodes={
            raw_a: _node("raw category a", constraints={"category": "CategoryA"}),
            raw_b: _node("raw category b", constraints={"category": "CategoryB"}),
            partial_rate: _node(
                "partial category rate",
                denominator="Count_Person",
                constraints={"category": "CategoryA"},
            ),
        },
        svs=[raw_a, raw_b, partial_rate],
        scores=[0.95, 0.90, 0.80],
        llm=llm,
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"
    assert len(llm.prompts) == 1
    assert "partial category rate" in llm.prompts[0]
    assert "raw category a" not in llm.prompts[0]
    assert "raw category b" not in llm.prompts[0]


def test_rate_intent_recognises_percent_forms_and_count_exception() -> None:
    assert _has_strong_rate_intent("population rate")
    assert _has_strong_rate_intent("population percentages")
    assert _has_strong_rate_intent("population %")
    assert _has_strong_rate_intent("renewable energy share")
    assert not _has_strong_rate_intent("shareholder equity")
    assert not _has_strong_rate_intent("population rate count")
    assert not _has_strong_rate_intent("how many population rates")


def test_raw_count_predicate_excludes_only_the_literal_standard_count_shape() -> None:
    from search.engine.shape import ShapeDraft

    raw_count = ShapeDraft(
        shape_id="raw",
        label="raw",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
    )

    assert _is_unnormalised_raw_count(raw_count)
    assert not _is_unnormalised_raw_count(replace(raw_count, stat_type_dcid="growthRate"))
    assert not _is_unnormalised_raw_count(replace(raw_count, meas_denom_unresolved=True))
    assert not _is_unnormalised_raw_count(replace(raw_count, declared_roles=("creditor", "debtor")))


def test_qualifier_detection_counts_female_and_urban_properties_once() -> None:
    raw = _shape_with_members(
        shape_id="raw-literacy",
        label="female literate count",
        denominator=None,
        properties=("gender", "residence", "literacyStatus", "fixed"),
        members=[
            {
                "gender": "Female",
                "residence": "Urban",
                "literacyStatus": "Literate",
                "fixed": "FixedValue",
            },
            {
                "gender": "Male",
                "residence": "Rural",
                "literacyStatus": "Illiterate",
                "fixed": "FixedValue",
            },
            {
                "gender": "Female",
                "residence": "Urban",
                "literacyStatus": "Literate",
                "fixed": "FixedValue",
            },
        ],
        value_labels={
            "Female": "Female",
            "Male": "Male",
            "Urban": "Urban",
            "Rural": "Rural",
            "Literate": "Literate",
            "Illiterate": "Illiterate",
            "FixedValue": "Fixed",
        },
    )

    mentioned, contradictory = _mentioned_raw_qualifier_props(raw, "female urban literacy rate")

    assert mentioned == {"gender", "residence"}
    assert not contradictory


@pytest.mark.parametrize(
    "variable",
    [
        "female literacy rate excluding male",
        "female literacy rate, not male",
        "female literacy rate, female but not female",
    ],
)
def test_negated_or_mixed_qualifier_wording_retains_raw_count(variable: str) -> None:
    raw = _shape_with_members(
        shape_id="raw-literacy",
        label="literacy count",
        denominator=None,
        properties=("gender",),
        members=[{"gender": "Female"}, {"gender": "Male"}],
        value_labels={"Female": "Female", "Male": "Male"},
    )
    normalized = _shape_with_members(
        shape_id="normalized-literacy",
        label="literacy rate",
        denominator="Count_Person",
        properties=(),
        members=[{}],
        value_labels={},
    )

    mentioned, contradictory = _mentioned_raw_qualifier_props(raw, variable)

    assert contradictory
    assert mentioned == set()
    assert raw in _prune_explicit_rate_raw_counts([raw, normalized], variable)


def test_per_candidate_qualifier_loss_never_unions_complementary_candidates() -> None:
    raw = _shape_with_members(
        shape_id="raw-literacy",
        label="literacy count",
        denominator=None,
        properties=("gender", "residence", "age"),
        members=[
            {"gender": "Female", "residence": "Urban", "age": "Adult"},
            {"gender": "Male", "residence": "Rural", "age": "Child"},
        ],
        value_labels={
            "Female": "Female",
            "Male": "Male",
            "Urban": "Urban",
            "Rural": "Rural",
            "Adult": "Adult",
            "Child": "Child",
        },
    )
    gender_rate = _shape_with_members(
        shape_id="gender-rate",
        label="literacy rate by gender",
        denominator="Count_Person",
        properties=("gender",),
        members=[{"gender": "Female"}],
        value_labels={"Female": "Female"},
    )
    residence_rate = _shape_with_members(
        shape_id="residence-rate",
        label="literacy rate by residence",
        denominator="Count_Person",
        properties=("residence",),
        members=[{"residence": "Urban"}],
        value_labels={"Urban": "Urban"},
    )
    shapes = [raw, gender_rate, residence_rate]

    assert raw in _prune_explicit_rate_raw_counts(shapes, "female urban adult literacy rate")
    assert _raw_count_is_pending_joint_slice(raw, shapes, "female urban adult literacy rate")


def test_compatible_candidate_order_does_not_change_raw_removal_before_top_k() -> None:
    raw = _shape_with_members(
        shape_id="raw-literacy",
        label="literacy count",
        denominator=None,
        properties=("gender",),
        members=[{"gender": "Female"}, {"gender": "Male"}],
        value_labels={"Female": "Female", "Male": "Male"},
    )
    normalized = _shape_with_members(
        shape_id="normalized-literacy",
        label="literacy rate",
        denominator="Count_Person",
        properties=(),
        members=[{}],
        value_labels={},
    )
    false_friend = _shape_with_members(
        shape_id="other-rate",
        label="unemployment rate",
        denominator="Count_Person",
        properties=(),
        members=[{}],
        value_labels={},
    )

    first = _prune_explicit_rate_raw_counts([raw, normalized, false_friend], "female literacy rate")
    second = _prune_explicit_rate_raw_counts(
        [false_friend, normalized, raw], "female literacy rate"
    )

    assert raw not in first
    assert raw not in second
    assert [shape.shape_id for shape in first] == ["normalized-literacy", "other-rate"]
    assert [shape.shape_id for shape in second] == ["other-rate", "normalized-literacy"]


def test_replays_female_literacy_fixture_electing_the_normalized_rate() -> None:
    """The literacy rate elected over the raw count declares no gender, so "female" is
    reported unbound."""
    response = offline_resolve(make_request("female literacy rate in Bangladesh"), llm=FakeLLM())

    assert response.status == "no_data"
    part = response.parts[0]
    assert part.no_data.reason == "qualifier_not_bound"
    assert [q.text for q in part.no_data.unbound_qualifiers] == ["female"]
    assert part.no_data.elected_count == 1


class _UnplacedOverridingLLM(FakeLLM):
    """The recorded bank, with each bind answer's `unplaced` replaced by `unplaced`."""

    def __init__(self, unplaced: list[str]) -> None:
        super().__init__()
        self.unplaced = unplaced

    async def generate_structured(self, *, prompt, system, schema):
        output, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if isinstance(output, _UniversalBindOutput):
            output = output.model_copy(update={"unplaced": self.unplaced})
        return output, usage


@pytest.mark.parametrize("unplaced", [[], ["female"]], ids=["none-unplaced", "female-unplaced"])
def test_female_literacy_refuses_on_female_whatever_the_bind_model_leaves_unplaced(
    unplaced: list[str],
) -> None:
    """The recorded female literacy answer elects a literacy rate with no gender slot. The
    request names Female, a value a shape on the bind menu holds, so the part refuses on
    "female" once, whether or not the bind model lists it as unplaced."""
    llm = _UnplacedOverridingLLM(unplaced)
    response = offline_resolve(make_request("female literacy rate in Bangladesh"), llm=llm)

    assert response.status == "no_data"
    part = response.parts[0]
    assert part.no_data.reason == "qualifier_not_bound"
    assert [q.text for q in part.no_data.unbound_qualifiers] == ["female"]
