"""An explicit rate request refuses raw-count shapes on their form before ranking and the
single bind call (`admissibility.judge_pinned_parts`)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.config import settings
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.result import RegionResult
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


class _RecordingBindLLM:
    def __init__(
        self,
        *,
        chosen_shape_index: int = 0,
        bindings: list[SlotBindingDraft] | None = None,
        ask: str | None = None,
        nearest_refused_index: int | None = None,
        usage: dict | None = None,
    ) -> None:
        self.chosen_shape_index = chosen_shape_index
        self.bindings = [] if bindings is None else bindings
        self.ask = ask
        self.nearest_refused_index = nearest_refused_index
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
                nearest_refused_index=self.nearest_refused_index,
            ),
            self.usage,
        )


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


def _menu(prompt: str) -> str:
    """The electable part of a bind prompt: everything before the NOT ELECTABLE list."""
    return prompt.partition("NOT ELECTABLE")[0]


def _not_electable(prompt: str) -> str:
    """The NOT ELECTABLE list of a bind prompt, empty when it has none."""
    return prompt.partition("NOT ELECTABLE")[2].partition("DENOMINATORS")[0]


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
    assert all(
        f"raw count {index}" not in _menu(llm.prompts[0]) for index in range(len(raw_stat_vars))
    )
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
    assert "raw count" not in _menu(llm.prompts[0])
    assert "raw count (fails the form the query asks for)" in _not_electable(llm.prompts[0])
    denominator = result.spec.resolved.shape.measurement_denominator
    assert denominator is not None
    assert denominator.ref == "Count_Household"


@pytest.mark.parametrize(
    ("variable", "status", "denominator"),
    [
        ("illustrative count", "definite", None),
        ("illustrative rate count", "no_data", "Count_Person"),
    ],
)
def test_a_named_rate_refuses_the_raw_count_and_a_count_elects_it(
    variable: str, status: str, denominator: str | None
) -> None:
    """A count question is offered the raw count and elects it. "Rate" names a rate, which
    a raw count does not serve, so the raw count is not electable and bind elects the rate,
    which has no data at the place."""
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
    assert ("raw count" in _menu(llm.prompts[0])) is (denominator is None)
    answer = result.spec if status == "definite" else result.flagged_match[0]
    found = answer.resolved.shape.measurement_denominator
    assert (found.ref if found is not None else None) == denominator


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
    assert label in _menu(llm.prompts[0])
    assert "raw count" not in _menu(llm.prompts[0])


@pytest.mark.parametrize(
    ("nearest_refused_index", "flagged"),
    [(0, ["RawCount"]), (None, [])],
    ids=["names-the-refused-reading", "names-none"],
)
def test_a_part_with_only_refused_readings_binds_over_the_not_electable_list(
    nearest_refused_index: int | None, flagged: list[str]
) -> None:
    """The one reading fails the form "%" names, so no shape is admissible. Bind runs over
    an empty menu with the reading listed as not electable, and the part refuses on that
    form. The reading the model names as nearest is the flagged match, with the
    denominator asked beside the one found; a model that names none flags nothing."""
    llm = _RecordingBindLLM(
        ask="no shape is a percentage", nearest_refused_index=nearest_refused_index
    )

    result = _run(
        "illustrative %",
        nodes={"RawCount": _node("raw count")},
        svs=["RawCount"],
        scores=[0.90],
        llm=llm,
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    assert [
        member.ref.ref for spec in result.flagged_match or () for member in spec.resolved.members
    ] == flagged
    assert (result.denominator is not None) == bool(flagged)
    (prompt,) = llm.prompts
    assert "CANDIDATE SHAPES (0 total)" in prompt
    assert "  0: raw count (fails the form the query asks for)" in prompt


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
    # Each menu shape states whether it has data at Kenya, in menu order.
    assert [
        line.removeprefix("  data at the named places: ")
        for line in llm.prompts[0].splitlines()
        if line.startswith("  data at the named places: ")
    ] == ["yes", "no"]


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


def test_recorded_measles_menu_lists_the_case_count_as_nearest_and_refuses_its_form() -> None:
    """The case count serves no rate, so it is not electable beside the immunization
    coverage, which measures something else. Bind declines the coverage and names the case
    count as nearest, so the answer refuses `denominator_not_available` flagging the measles
    case count alone: the reading the question asked for, settled on the condition it names."""
    response = offline_resolve(make_request("measles incidence rate Kenya"), llm=FakeLLM())

    assert response.status == "no_data"
    part = response.parts[0]
    assert part.no_data.reason == "denominator_not_available"
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


def test_recorded_female_urban_menu_refuses_the_values_the_literacy_rate_lacks() -> None:
    """The female literate population count serves no rate and is not electable. Bind
    elects the literacy rate of people aged ten and over, which has no gender or residence
    slot, so the answer leaves the values the phrase names unmet and refuses
    `qualifier_not_bound` naming female and urban, flagging the rate it elected: a
    refusal names each requirement its flagged match leaves unmet."""
    response = offline_resolve(make_request("female urban literacy rate"), llm=FakeLLM())

    assert response.status == "no_data"
    part = response.parts[0]
    assert part.no_data.reason == "qualifier_not_bound"
    assert [term.text for term in part.no_data.unbound_qualifiers] == ["female", "urban"]
    flagged = part.no_data.flagged_match
    assert flagged is not None
    assert [sv.ref.ref for sv in flagged.resolved.members] == [
        "Count_Person_10OrMoreYears_Literate_AsAFractionOf_Count_Person_10OrMoreYears"
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
    assert "female count" not in _menu(llm.prompts[0])
    assert "literate count" not in _menu(llm.prompts[0])
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
    assert "raw category a" not in _menu(llm.prompts[0])
    assert "raw category b" not in _menu(llm.prompts[0])


def test_replays_female_literacy_fixture_electing_the_normalized_rate() -> None:
    """The female literate count serves no rate and is not electable, so bind elects the
    literacy rate, which has no gender slot, and the part refuses on "female"."""
    llm = _CountingFixtureLLM()
    response = offline_resolve(make_request("female literacy rate in Bangladesh"), llm=llm)

    assert all(
        "Population: Female, Literate (fails the form" in _not_electable(prompt)
        for prompt in llm.bind_prompts
    )
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


class _FemaleUrbanBindLLM:
    """Binds the literacy rate's gender to Female and its residence to Urban."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
                ),
                SlotBindingDraft(
                    axis="how", property_dcid="residence", kind="value", value_dcids=["Urban"]
                ),
            ],
        ), None


def test_a_joint_slice_no_member_holds_does_not_answer_with_one_qualifier_dropped() -> None:
    """ "female urban literacy rate" binds Female and Urban. The rate family holds a female
    rate, an urban rate and a male rural rate, each with data, and no female urban one. The
    answer never serves the female or the urban rate as though it were the slice asked."""
    rates = {
        "FemaleLiteracyRate": {"gender": "Female"},
        "UrbanLiteracyRate": {"residence": "Urban"},
        "MaleRuralLiteracyRate": {"gender": "Male", "residence": "Rural"},
    }
    nodes = {
        dcid: _node(f"Literacy rate {dcid}", denominator="Count_Person", constraints=constraints)
        for dcid, constraints in rates.items()
    }
    nodes |= {
        value: {"label": value, "type": "Thing"} for value in ("Female", "Male", "Urban", "Rural")
    }

    result = _run(
        "female urban literacy rate",
        nodes=nodes,
        svs=list(rates),
        scores=[0.9, 0.89, 0.88],
        llm=_FemaleUrbanBindLLM(),
        with_data=list(rates),
    )

    assert result.status == "no_data"
    assert result.specs == ()
