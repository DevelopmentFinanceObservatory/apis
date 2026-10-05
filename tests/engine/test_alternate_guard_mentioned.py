"""Test the constraint-property axes: an alternate whose members hold another concept value
than the elected reading, or a value on a property the elected reading leaves open, is
refused.

Unit cases build shapes from member arcs. The pipeline pair runs the GDP and GNI per
capita shapes of gdp-per-capita-vietnam-2023 through ``resolve_variable``, once with the
activitySource binding marked ``mentioned`` and once marked ``sole_option``; both refuse GNI.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import DraftBindingEvidence, SlotBindingDraft, _UniversalBindOutput
from search.engine.region import resolve_variable
from search.engine.region.alternate_guard import inadmissible_alternates, query_pins
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_GDP_PC = "Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita"
_GNI_PPP_PC = "Amount_EconomicActivity_GrossNationalIncome_PurchasingPowerParity_PerCapita"


def _constraint_arcs(constraints: dict[str, str]) -> dict:
    arcs: dict = {
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
    }
    for prop, value in constraints.items():
        arcs[prop] = {"nodes": [{"dcid": value}]}
    return arcs


def _shape(label: str, qual: str, members: dict[str, dict[str, str]]) -> ShapeDraft:
    return ShapeDraft(
        shape_id=label,
        label=label,
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=qual,
        meas_denom_dcid=None,
        slot_keys=(),
        meas_denom_unresolved=True,
        sv_arc_facts={sv: _constraint_arcs(c) for sv, c in members.items()},
    )


def _binding(
    prop: str, values: list[str], *, evidence: DraftBindingEvidence = "mentioned"
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid=prop,
        kind="value" if len(values) == 1 else "set",
        value_dcids=values,
        evidence=evidence,
    )


_ELECTED = _shape(
    "Nominal gross domestic product per capita",
    "Nominal",
    {_GDP_PC: {"activitySource": "GrossDomesticProduction"}},
)
_GNI_ALTERNATE = _shape(
    "Gross National Income Based on Purchasing Power Parity Per Capita",
    "PurchasingPowerParity",
    {_GNI_PPP_PC: {"activitySource": "GrossNationalIncome"}},
)
_GDP_MENTIONED = [_binding("activitySource", ["GrossDomesticProduction"])]


def _code(reason: str) -> str:
    return reason.partition(": ")[0]


@pytest.mark.parametrize(
    "variable,elected,alternate,bindings",
    [
        pytest.param(
            "GDP per capita",
            _ELECTED,
            _GNI_ALTERNATE,
            _GDP_MENTIONED,
            id="gdp-elected/gdp-per-capita-vietnam-2023/std-denom-01/02/gdp-per-capita-nominal",
        ),
        pytest.param(
            "GNI per capita PPP",
            _GNI_ALTERNATE,
            _ELECTED,
            [_binding("activitySource", ["GrossNationalIncome"])],
            id="gni-elected/gni-per-capita-ppp-ethiopia",
        ),
    ],
)
def test_alternate_contradicting_the_elected_concept_is_refused(
    variable, elected, alternate, bindings
):
    """GDP and GNI per capita hold different activitySource values, so each is another
    concept beside the other, whether or not the query's words bound activitySource.

    Normalisation cannot separate the pair: both carry the per-capita denominator and
    differ only in qualifier and activitySource.
    """
    for pins in (query_pins(variable), query_pins(variable, bindings=bindings)):
        result = inadmissible_alternates(pins, elected, [(1, alternate)])
        assert result.keys() == {1}
        assert _code(result[1]) == "concept:activitySource", result


def test_alternate_holding_the_mentioned_value_is_kept():
    """A PPP reading of GDP itself shares the mentioned value and stays a reading of the query."""
    gdp_ppp = _shape(
        "Gross domestic product based on purchasing power parity per capita",
        "PurchasingPowerParity",
        {"Amount_GDP_PPP_PerCapita": {"activitySource": "GrossDomesticProduction"}},
    )
    assert (
        inadmissible_alternates(
            query_pins("GDP per capita", bindings=_GDP_MENTIONED), _ELECTED, [(1, gdp_ppp)]
        )
        == {}
    )


def test_alternate_holding_a_property_the_elected_reading_leaves_open_is_refused():
    """A value on a property neither the query nor the elected reading names specialises
    the reading, since GDP of manufacturing is not GDP."""
    gdp_by_sector = _shape(
        "Gross domestic product per capita, manufacturing",
        "PurchasingPowerParity",
        {
            "Amount_GDP_Manufacturing_PerCapita": {
                "activitySource": "GrossDomesticProduction",
                "naics": "NAICS/31-33",
            }
        },
    )
    result = inadmissible_alternates(
        query_pins("GDP per capita", bindings=_GDP_MENTIONED), _ELECTED, [(1, gdp_by_sector)]
    )
    assert _code(result[1]) == "specialisation:naics", result


@pytest.mark.parametrize("evidence", ["mentioned", "sole_option"])
def test_alternate_refused_on_member_values_whatever_the_binding_evidence(evidence):
    """The elected reading's member values decide, so a sole-option binding refuses GNI
    as a mentioned one does."""
    pins = query_pins(
        "GDP per capita",
        bindings=[_binding("activitySource", ["GrossDomesticProduction"], evidence=evidence)],
    )
    result = inadmissible_alternates(pins, _ELECTED, [(1, _GNI_ALTERNATE)])
    assert _code(result[1]) == "concept:activitySource", result


def test_alternate_with_a_member_silent_on_the_mentioned_property_is_kept():
    """One member that asserts no activitySource leaves the alternate a possible reading."""
    mixed = _shape(
        "Gross national income per capita",
        "PurchasingPowerParity",
        {
            _GNI_PPP_PC: {"activitySource": "GrossNationalIncome"},
            "Amount_EconomicActivity_PurchasingPowerParity_PerCapita": {},
        },
    )
    assert (
        inadmissible_alternates(
            query_pins("GDP per capita", bindings=_GDP_MENTIONED), _ELECTED, [(1, mixed)]
        )
        == {}
    )


_GDP_BY_GENDER = _shape(
    "Gross domestic product per capita, women",
    "Nominal",
    {
        "Amount_GDP_Female_PerCapita": {
            "activitySource": "GrossDomesticProduction",
            "gender": "Female",
        }
    },
)


@pytest.mark.parametrize(
    "binding,engine_authored,refused",
    [
        pytest.param(_binding("gender", ["Female"]), frozenset(), False, id="mentioned-value"),
        pytest.param(
            _binding("gender", ["Female", "Male"]), frozenset(), False, id="mentioned-set"
        ),
        pytest.param(
            SlotBindingDraft(
                axis="how", property_dcid="gender", kind="unbound", evidence="requested_all"
            ),
            frozenset(),
            False,
            id="requested-all",
        ),
        pytest.param(
            _binding("gender", ["Female"], evidence="sole_option"),
            frozenset(),
            True,
            id="sole-option",
        ),
        pytest.param(
            SlotBindingDraft(axis="how", property_dcid="gender", kind="unbound"),
            frozenset(),
            True,
            id="unbound",
        ),
        pytest.param(
            _binding("gender", ["Female"]), frozenset({"gender"}), True, id="engine-authored"
        ),
    ],
)
def test_a_property_named_by_the_query_admits_its_specialisation(binding, engine_authored, refused):
    """A mentioned value or set, or a request for every value, names a property.
    A sole-option or unbound draft and a property the engine authored (a measure election or
    a declared default) name nothing, so a member holding it specialises the reading."""
    pins = query_pins("GDP per capita", bindings=[binding], engine_authored=engine_authored)
    result = inadmissible_alternates(pins, _ELECTED, [(1, _GDP_BY_GENDER)])
    assert (_code(result[1]) if result else None) == ("specialisation:gender" if refused else None)


def test_elected_values_widen_to_code_index_descendants():
    """A CRS sector admits an alternate whose members hold one of its sub-sectors."""
    basic_health = _shape(
        "Aid to basic health",
        "Nominal",
        {"Amount_Aid_BasicHealth": {"crsPurposeCode": "CRSPurposeCode122"}},
    )
    health = _shape(
        "Aid to health", "Nominal", {"Amount_Aid_Health": {"crsPurposeCode": "CRSPurposeCode120"}}
    )
    pins = query_pins("health aid", bindings=[_binding("crsPurposeCode", ["CRSPurposeCode120"])])
    assert inadmissible_alternates(pins, health, [(1, basic_health)]) == {}
    assert inadmissible_alternates(query_pins("health aid"), health, [(1, basic_health)]) == {}


# The pipeline pair runs the recorded bind output of gdp-per-capita-vietnam-2023.

_VNM = "country/VNM"


def _node(name: str, qual: str, activity_source: str) -> dict:
    return {
        "label": name,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": name}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementQualifier": {"nodes": [{"dcid": qual}]},
            "measurementDenominator": {
                "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
            },
            **_constraint_arcs({"activitySource": activity_source}),
        },
    }


class _GdpBindLLM:
    """Elect the GDP shape, bind activitySource with the given evidence, nominate GNI."""

    def __init__(self, evidence: DraftBindingEvidence):
        self._evidence = evidence

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                _binding("activitySource", ["GrossDomesticProduction"], evidence=self._evidence)
            ],
            also_plausible=[1],
            ask=None,
        ), None


def _resolve_gdp_per_capita(evidence: DraftBindingEvidence):
    query = "GDP per capita of Vietnam"
    facet = {"earliestDate": "2015", "latestDate": "2023", "obsCount": 9}
    graph = FakeGraph(
        nodes={
            _VNM: {"label": "Vietnam", "type": "Country"},
            _GDP_PC: _node(
                "Nominal gross domestic product per capita", "Nominal", "GrossDomesticProduction"
            ),
            _GNI_PPP_PC: _node(
                "Gross National Income Based on Purchasing Power Parity Per Capita",
                "PurchasingPowerParity",
                "GrossNationalIncome",
            ),
        },
        detect={query: {"svs": [_GDP_PC, _GNI_PPP_PC], "cosine_scores": [0.889, 0.829]}},
        resolve={"Vietnam": _VNM},
        obs={f"{_GDP_PC}|{_VNM}": [facet], f"{_GNI_PPP_PC}|{_VNM}": [facet]},
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Vietnam"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_GdpBindLLM(evidence),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def _served(result) -> list[set[str]]:
    return [{sv.ref for sv in spec.resolved.stat_vars.items} for spec in result.specs]


def test_pipeline_mentioned_gdp_refuses_the_gni_nomination():
    result = _resolve_gdp_per_capita("mentioned")
    assert result.status == "definite"
    assert _served(result) == [{_GDP_PC}]


def test_pipeline_sole_option_gdp_refuses_the_gni_nomination():
    """The elected reading's member values decide, so a sole-option binding refuses GNI as
    a mentioned one does."""
    result = _resolve_gdp_per_capita("sole_option")
    assert result.status == "definite"
    assert _served(result) == [{_GDP_PC}]


def test_a_registry_twin_in_another_vocabulary_survives_a_mentioned_value():
    """The registry pairs vocabularies, and the constraint axes compare value dcids. The
    under-five mortality twins bind age as `YearsUpto4` and `SDG_AgeEnum_Y0T4`, so a twin
    passes the guard whole."""
    elected = _shape(
        "Under-five mortality rate",
        "Nominal",
        {
            "MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth": {
                "age": "YearsUpto4"
            }
        },
    )
    twin = _shape(
        "Under-five mortality rate (SDG)",
        "Nominal",
        {"sdg/SH_DYN_MORT.AGE--Y0T4": {"age": "SDG_AgeEnum_Y0T4"}},
    )

    pins = query_pins("under-five mortality", bindings=[_binding("age", ["YearsUpto4"])])
    assert inadmissible_alternates(pins, elected, [(1, twin)]) == {}
