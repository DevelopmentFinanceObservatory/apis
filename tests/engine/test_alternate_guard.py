"""Test alternate admissibility and its pipeline effects.

Unit cases cover refused and legitimate shape pairs, asserting the refusal's reason code.
Pipeline cases check model-nominated and near-neighbour alternatives against offline graph
fixtures.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.qualifier_match import CodedQualifier
from search.engine.quantity import denominator_kinds, normalisation_class, quantity_requirement
from search.engine.region import NEAR_NEIGHBOR_AMBIGUOUS, resolve_variable
from search.engine.region.alternate_guard import (
    inadmissible_alternates,
    measure_class,
    query_pins,
    shape_measure_class,
)
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

# Unit tests use observed tuples where FakeGraph has arcs.


def _code(reason: str) -> str:
    """The reason code a refusal opens with, such as `pin:normalisation`."""
    return reason.partition(": ")[0]


def _shape(
    label: str,
    pop: str,
    prop: str,
    stat: str,
    qual: str | None,
    denom: str | None,
    *,
    unresolved: bool = False,
    members: tuple[str, ...] = (),
    roles: tuple[str, ...] | None = None,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test",
        label=label,
        pop_type_dcid=pop,
        meas_prop_dcid=prop,
        stat_type_dcid=stat,
        meas_qual_dcid=qual,
        meas_denom_dcid=denom,
        slot_keys=(),
        meas_denom_unresolved=unresolved,
        sv_arc_facts={m: {} for m in members} if members else None,
        declared_roles=roles,
    )


# Corpus reproductions captured from the bind call.

REJECT_CASES = [
    pytest.param(
        "sdg-01",
        "maternal mortality ratio",
        _shape(
            "Maternal mortality ratio [Female]",
            "SDG_SH_STA_MORT",
            "value",
            "measuredValue",
            None,
            None,
            members=("sdg/SH_STA_MORT.SEX--F",),
        ),
        _shape(
            "Number of maternal deaths",
            "MaternalMortalityEvent",
            "count",
            "measuredValue",
            None,
            None,
            members=("Count_MaternalMortalityEvent",),
        ),
        "pin:normalisation",
        id="sdg-01-false-friend-count",
    ),
    pytest.param(
        "sdg-04",
        "HIV incidence rate",
        _shape(
            "Number of new HIV infections per 1,000 uninfected population",
            "SDG_SH_HIV_INCD",
            "value",
            "measuredValue",
            None,
            None,
            members=("sdg/SH_HIV_INCD",),
        ),
        _shape(
            "Count of HIV/Acquired Immune Deficiency Syndrome incidents",
            "MedicalConditionIncident",
            "count",
            "measuredValue",
            None,
            None,
            members=("Count_MedicalConditionIncident_ConditionHIVAIDS",),
        ),
        "pin:normalisation",
        id="sdg-04-false-friend-count",
    ),
    pytest.param(
        "gdp-per-capita-vietnam-2023",
        "GDP per capita",
        _shape(
            "Nominal gross domestic product per capita",
            "EconomicActivity",
            "amount",
            "measuredValue",
            "Nominal",
            None,
            unresolved=True,
            members=("Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita",),
        ),
        _shape(
            "Nominal gross domestic product",
            "EconomicActivity",
            "amount",
            "measuredValue",
            "Nominal",
            None,
            members=("Amount_EconomicActivity_GrossDomesticProduction_Nominal",),
        ),
        "pin:normalisation",
        id="gdp-vietnam-contradicted-denominator",
    ),
    pytest.param(
        "ghg-emissions-bangladesh",
        "greenhouse gas emissions",
        _shape(
            "Annual Amount of Emissions: Greenhouse Gas",
            "Emissions",
            "amount",
            "measuredValue",
            "Annual",
            None,
            members=("Annual_Emissions_GreenhouseGas",),
        ),
        _shape(
            "CO2 Emissions Per Capita",
            "Emissions",
            "amount",
            "measuredValue",
            None,
            None,
            unresolved=True,
            members=("Amount_Emissions_CarbonDioxide_PerCapita",),
        ),
        "specialisation:normalisation",
        id="bangladesh-qual-and-denom-delta",
    ),
    # The offline LLM fixture omits the captured rural and urban variants.
    pytest.param(
        "electricity-access-rwanda",
        "access to electricity",
        _shape(
            "Access to electricity (% of population)",
            "Electricity",
            "worldBank/EG_ELC_ACCS_ZS",
            "measuredValue",
            None,
            None,
            members=("worldBank/EG_ELC_ACCS_ZS",),
        ),
        _shape(
            "Access to electricity, rural (% of rural population)",
            "Electricity",
            "worldBank/EG_ELC_ACCS_RU_ZS",
            "measuredValue",
            None,
            None,
            members=("worldBank/EG_ELC_ACCS_RU_ZS",),
        ),
        "specialisation:residence",
        id="rwanda-rural-narrowing",
    ),
    pytest.param(
        "electricity-access-rwanda",
        "access to electricity",
        _shape(
            "Access to electricity (% of population)",
            "Electricity",
            "worldBank/EG_ELC_ACCS_ZS",
            "measuredValue",
            None,
            None,
            members=("worldBank/EG_ELC_ACCS_ZS",),
        ),
        _shape(
            "Access to electricity, urban (% of urban population)",
            "Electricity",
            "worldBank/EG_ELC_ACCS_UR_ZS",
            "measuredValue",
            None,
            None,
            members=("worldBank/EG_ELC_ACCS_UR_ZS",),
        ),
        "specialisation:residence",
        id="rwanda-urban-narrowing",
    ),
    pytest.param(
        "remittances-inward-kenya",
        "inward remittances to Kenya",
        _shape(
            "Inward Remittances",
            "Remittance",
            "amount",
            "measuredValue",
            None,
            None,
            members=("Amount_Remittance_InwardRemittance",),
        ),
        _shape(
            "Inward Remittances as a Percentage of Nominal Gross Domestic Production",
            "Remittance",
            "amount",
            "measuredValue",
            None,
            "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
            members=(
                "Amount_Remittance_InwardRemittance_AsFractionOf_"
                "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
            ),
        ),
        "specialisation:normalisation",
        id="remittances-level-refuses-the-share-of-gdp",
    ),
    pytest.param(
        "dq-crs-003",
        "aid from Japan",
        _shape(
            "Gross disbursements of official development assistance",
            "FinancialFlow",
            "amount",
            "measuredValue",
            "Nominal",
            None,
            members=("Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1010_Nominal",),
        ),
        _shape(
            "Grant equivalents of official development assistance (% of GNI)",
            "FinancialFlow",
            "grantEquivalent",
            "measuredValue",
            "Nominal",
            "PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal",
            members=(
                "GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1010_Nominal_AsAFractionOf_"
                "PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal",
            ),
        ),
        "specialisation:normalisation",
        id="aid-amount-refuses-the-grant-equivalent-share-of-gni",
    ),
    pytest.param(
        "ids-interest-external-debt-kenya",
        "Kenya interest payments on external debt",
        _shape(
            "Interest payments on external debt",
            "FinancialFlow",
            "amount",
            "measuredValue",
            "Nominal",
            None,
            members=("Amount_FinancialFlow_InterestPayment_Nominal",),
        ),
        _shape(
            "Interest payments on external debt (% of GNI)",
            "FinancialFlow",
            "amount",
            "measuredValue",
            None,
            "Amount_EconomicActivity_GrossNationalIncome_Nominal",
            members=(
                "Amount_FinancialFlow_InterestPayment_AsAFractionOf_"
                "Amount_EconomicActivity_GrossNationalIncome_Nominal",
            ),
        ),
        "specialisation:normalisation",
        id="interest-payments-level-refuses-the-share-of-gni",
    ),
]


@pytest.mark.parametrize("golden_id,variable,elected,alt,expected_code", REJECT_CASES)
def test_repro_nomination_rejected(golden_id, variable, elected, alt, expected_code):
    result = inadmissible_alternates(
        query_pins(variable, quantity=quantity_requirement([variable])), elected, [(0, alt)]
    )
    assert 0 in result, f"{golden_id}: expected rejection, guard admitted the alternate"
    assert _code(result[0]) == expected_code, f"{golden_id}: got reason {result[0]!r}"


def test_rwanda_twin_is_exempt_from_narrowing_check():
    """The registered electricity twin passes before its urbanisation members are judged.

    Its member set includes both registry-cluster and non-cluster variants.
    """
    elected = _shape(
        "Access to electricity (% of population)",
        "Electricity",
        "worldBank/EG_ELC_ACCS_ZS",
        "measuredValue",
        None,
        None,
        members=("worldBank/EG_ELC_ACCS_ZS",),
    )
    twin = _shape(
        "Proportion of population with access to electricity",
        "SDG_EG_ACS_ELEC",
        "value",
        "measuredValue",
        "SDG_ReportingStandard",
        None,
        unresolved=True,
        members=(
            "sdg/EG_ACS_ELEC",
            "sdg/EG_ACS_ELEC.URBANISATION--R",
            "sdg/EG_ACS_ELEC.URBANISATION--U",
        ),
    )
    assert (
        inadmissible_alternates(
            query_pins(
                "access to electricity", quantity=quantity_requirement(["access to electricity"])
            ),
            elected,
            [(0, twin)],
        )
        == {}
    )


# Cross-population false friends.

_CRS_BIODIVERSITY = (
    "Official development assistance grant equivalents for biodiversity "
    "(reimbursable grant, contribution to specific-purpose programme and fund "
    "managed by an international organisation)",
    "FinancialFlow",
    "grantEquivalent",
    "measuredValue",
    "Nominal",
    None,
    ("financingProvider", "financingRecipient"),
)
_SDG_BY_DONOR = (
    "Total official development assistance for biodiversity",
    "SDG_DC_ODA_BDVDL",
    "value",
    "measuredValue",
    None,
    None,
    ("observationAbout",),
)
_SDG_CREDIT_ACCESS = (
    "Proportion of small-scale industries with a loan or line of credit [Manufacturing]",
    "SDG_FC_ACC_SSID",
    "value",
    "measuredValue",
    None,
    None,
    ("observationAbout",),
)


def _r5_shape(spec: tuple, members: tuple[str, ...]) -> ShapeDraft:
    label, pop, prop, stat, qual, denom, roles = spec
    return _shape(
        label,
        pop,
        prop,
        stat,
        qual,
        denom,
        members=members,
        roles=roles,
    )


R5_CASES = [
    pytest.param(
        "biodiversity aid to Zambia since 2018 in grant equivalents",
        _CRS_BIODIVERSITY,
        (
            "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSPurposeCode41030_Nominal",
        ),
        _SDG_BY_DONOR,
        ("sdg/DC_ODA_BDVDL",),
        id="902-biodiversity-by-donor-sdg-variant",
    ),
    pytest.param(
        "grant financing for Mozambique",
        _CRS_BIODIVERSITY,
        (
            "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSPurposeCode12262_Nominal",
        ),
        _SDG_CREDIT_ACCESS,
        ("sdg/FC_ACC_SSID.ACTIVITY--ISIC4_C",),
        id="902-grant-financing-manufacturing-credit-indicator",
    ),
    pytest.param(
        "concessional loans to Sri Lanka",
        (
            "Official development assistance commitments for general budget support "
            "(standard loan)",
            "FinancialObligation",
            "amount",
            "measuredValue",
            "Nominal",
            None,
            ("financingProvider", "financingRecipient"),
        ),
        (
            "Amount_FinancialObligation_Commitment_CRSTypeOfFlow10_CRSTypeOfFinance421_CRSPurposeCode15112_Nominal",
        ),
        _SDG_CREDIT_ACCESS,
        ("sdg/FC_ACC_SSID.ACTIVITY--ISIC4_C",),
        id="902-concessional-loans-manufacturing-credit-indicator",
    ),
]


@pytest.mark.parametrize(
    "query,elected_spec,elected_members,alt_spec,alt_members",
    R5_CASES,
    ids=[p.id for p in R5_CASES],
)
def test_cross_population_false_friend_rejected(
    query, elected_spec, elected_members, alt_spec, alt_members
):
    """The cross-population SDG alternate to the elected CRS flow is refused (#902)."""
    elected = _r5_shape(elected_spec, elected_members)
    alt = _r5_shape(alt_spec, alt_members)
    result = inadmissible_alternates(
        query_pins(query, quantity=quantity_requirement([query])), elected, [(0, alt)]
    )
    assert 0 in result, f"{query}: admitted the cross-population alternate"
    assert _code(result[0]) == "cross_population", result[0]


def test_same_population_sibling_admitted():
    """A same-population CRS sibling with a different measure and qualifier is admitted."""
    elected = _r5_shape(
        _CRS_BIODIVERSITY,
        (
            "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSPurposeCode41030_Nominal",
        ),
    )
    sibling = _r5_shape(
        (
            "Official development assistance net disbursements for biodiversity",
            "FinancialFlow",
            "amount",
            "measuredValue",
            None,
            None,
            ("financingProvider", "financingRecipient"),
        ),
        (
            "Amount_FinancialFlow_Net_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSPurposeCode41030_Nominal",
        ),
    )
    assert (
        inadmissible_alternates(
            query_pins(
                "biodiversity aid to Zambia since 2018 in grant equivalents",
                quantity=quantity_requirement(
                    ["biodiversity aid to Zambia since 2018 in grant equivalents"]
                ),
            ),
            elected,
            [(0, sibling)],
        )
        == {}
    )


def test_cross_population_registry_twin_still_exempt():
    """The registered electricity twin passes despite its cross-population signature."""
    elected = _shape(
        "Access to electricity (% of population)",
        "Electricity",
        "worldBank/EG_ELC_ACCS_ZS",
        "measuredValue",
        None,
        None,
        members=("worldBank/EG_ELC_ACCS_ZS",),
    )
    twin = _shape(
        "Proportion of population with access to electricity",
        "undata/sdg/SERIES-EG_ACS_ELEC",
        "value",
        "measuredValue",
        None,
        None,
        members=("undata/sdg/EG_ACS_ELEC",),
        roles=("unit",),
    )
    assert (
        inadmissible_alternates(
            query_pins(
                "access to electricity", quantity=quantity_requirement(["access to electricity"])
            ),
            elected,
            [(0, twin)],
        )
        == {}
    )


# The GDP and per-capita sibling tuples.

_GDP_PLAIN = (
    "Nominal gross domestic product",
    "EconomicActivity",
    "amount",
    "measuredValue",
    "Nominal",
    None,
)
_GDP_PER_CAPITA = (
    "Nominal gross domestic product per capita",
    "EconomicActivity",
    "amount",
    "measuredValue",
    "Nominal",
    "PerCapita",
)


def _r6_pair() -> tuple[ShapeDraft, ShapeDraft]:
    return (
        _shape(*_GDP_PLAIN, members=("Amount_EconomicActivity_GrossDomesticProduction_Nominal",)),
        _shape(
            *_GDP_PER_CAPITA,
            members=("Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita",),
        ),
    )


def test_per_capita_alternate_rejected_on_bare_phrase():
    """The per-capita sibling specialises a bare GDP question (#692)."""
    elected, per_capita = _r6_pair()
    result = inadmissible_alternates(
        query_pins("GDP", quantity=quantity_requirement(["GDP"])), elected, [(0, per_capita)]
    )
    assert _code(result[0]) == "specialisation:normalisation", result


@pytest.mark.parametrize(
    "variable",
    ["GDP per capita", "GDP per person", "GDP per head", "GDP per inhabitant"],
    ids=["per-capita", "per-person", "per-head", "per-inhabitant"],
)
def test_per_capita_alternate_kept_when_phrase_names_the_framing(variable):
    """The alternate stands when the query names per-capita framing."""
    elected, per_capita = _r6_pair()
    assert (
        inadmissible_alternates(
            query_pins(variable, quantity=quantity_requirement([variable])),
            elected,
            [(0, per_capita)],
        )
        == {}
    )


@pytest.mark.parametrize(
    "variable,expected_code",
    [
        ("GDP per 1,000 live births", "specialisation:normalisation"),
        ("GDP per year", "specialisation:normalisation"),
        ("GDP per cent", "specialisation:normalisation"),
        # `percent` names a share, which an amount per person is not.
        ("GDP percent", "pin:normalisation"),
    ],
    ids=["per-thousand-live-births", "per-year", "per-cent", "percent"],
)
def test_per_capita_alternate_rejected_when_phrase_names_no_per_capita_framing(
    variable, expected_code
):
    """Only phrases naming per-capita framing select that alternate (#954)."""
    elected, per_capita = _r6_pair()
    result = inadmissible_alternates(
        query_pins(variable, quantity=quantity_requirement([variable])), elected, [(0, per_capita)]
    )
    assert 0 in result, f"admitted the per-capita alternate on {variable!r}"
    assert _code(result[0]) == expected_code, result[0]


def test_per_capita_elected_with_plain_alternate_not_rejected():
    """The plain-total sibling stands when the per-capita shape is elected."""
    elected, per_capita = _r6_pair()
    assert (
        inadmissible_alternates(
            query_pins("GDP", quantity=quantity_requirement(["GDP"])), per_capita, [(0, elected)]
        )
        == {}
    )


def test_count_person_denominator_alternate_is_per_capita():
    """A `Count_Person` denominator is per person, so it specialises a bare GDP question."""
    elected, _ = _r6_pair()
    fraction_of_population = _shape(
        "Nominal gross domestic product as a fraction of population",
        "EconomicActivity",
        "amount",
        "measuredValue",
        "Nominal",
        "Count_Person",
        members=("Amount_EconomicActivity_GrossDomesticProduction_AsFractionOf_Count_Person",),
    )
    result = inadmissible_alternates(
        query_pins("GDP", quantity=quantity_requirement(["GDP"])),
        elected,
        [(0, fraction_of_population)],
    )
    assert _code(result[0]) == "specialisation:normalisation", result


# A StatVar denominator classified from its own arcs.

_ADULTS = "Count_Person_15OrMoreYears"
_GNI = "Amount_EconomicActivity_GrossNationalIncome"
_DENOMINATOR_NODES = {
    _ADULTS: {
        "arcs": {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "age": {"nodes": [{"dcid": "Years15Onwards"}]},
        }
    },
    _GNI: {
        "arcs": {
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        }
    },
    "Area_Land": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "area"}]}}},
}


class _CountingGraph(FakeGraph):
    def __init__(self) -> None:
        super().__init__(nodes=_DENOMINATOR_NODES)
        self.batches: list[list[str]] = []

    def node_arcs_batch(self, dcids, *, deadline=None):
        self.batches.append(list(dcids))
        return super().node_arcs_batch(dcids, deadline=deadline)


def test_denominator_kinds_reads_statvar_denominators_in_one_batch():
    """Schema terms are not read; each StatVar denominator is read once."""
    graph = _CountingGraph()
    kinds = denominator_kinds(
        [_ADULTS, "Count_Person", _GNI, "PerCapita", "area", "Area_Land", _ADULTS, "Unloaded"],
        graph=graph,
    )
    assert graph.batches == [[_ADULTS, _GNI, "Area_Land", "Unloaded"]]
    assert kinds == {_ADULTS: "person", _GNI: "other", "Area_Land": "area", "Unloaded": "unknown"}


def test_denominator_kinds_skips_the_read_with_no_statvar_denominator():
    graph = _CountingGraph()
    assert denominator_kinds(["Count_Person", "PerCapita", "area"], graph=graph) == {}
    assert graph.batches == []


@pytest.mark.parametrize(
    "denominator,kinds,expected",
    [
        (_ADULTS, {_ADULTS: "person"}, "per_person"),
        (_GNI, {_GNI: "other"}, "series"),
        ("Area_Land", {"Area_Land": "area"}, "per_area"),
        ("Unloaded", {"Unloaded": "unknown"}, "unknown"),
        # A denominator missing from the map follows the dcid rule.
        (_ADULTS, {}, "series"),
        ("Count_Person", {}, "per_person"),
        ("area", {}, "per_area"),
        # A schema term outranks a map entry.
        ("Count_Person", {"Count_Person": "unknown"}, "per_person"),
    ],
    ids=[
        "person-arcs",
        "gni-arcs",
        "area-arcs",
        "unresolved-arcs",
        "unmapped-statvar",
        "count-person",
        "area",
        "schema-term-wins",
    ],
)
def test_normalisation_class_reads_the_denominator_kind(denominator, kinds, expected):
    shape = _shape("x", "Consumption", "amount", "measuredValue", None, denominator)
    assert normalisation_class(shape, kinds=kinds) == expected


def test_person_count_denominator_alternate_honours_per_capita():
    """Alcohol consumption per adult is per capita once its denominator's arcs are read."""
    elected = _shape(
        "Alcohol consumption",
        "Consumption",
        "amount",
        "measuredValue",
        None,
        None,
        members=("Amount_Consumption_Alcohol",),
    )
    per_adult = _shape(
        "Alcohol consumption per adult",
        "Consumption",
        "amount",
        "measuredValue",
        None,
        _ADULTS,
        members=(f"Amount_Consumption_Alcohol_15OrMoreYears_AsFractionOf_{_ADULTS}",),
    )
    variable = "alcohol consumption per capita"
    unread = inadmissible_alternates(
        query_pins(variable, quantity=quantity_requirement([variable])), elected, [(0, per_adult)]
    )
    assert _code(unread[0]) == "pin:normalisation", unread
    pins = query_pins(
        variable, quantity=quantity_requirement([variable]), denominator_kinds={_ADULTS: "person"}
    )
    assert inadmissible_alternates(pins, elected, [(0, per_adult)]) == {}


def test_alternate_compared_with_the_request_when_the_election_fulfils_it():
    """A share of the population honours "percent of population"; a per-person amount
    of the same class beside it does not, and the pin refuses it."""
    share = _shape(
        "Share of the population that is urban",
        "Person",
        "count",
        "measuredValue",
        None,
        "Count_Person",
        members=("Count_Person_Urban_AsFractionOf_Count_Person",),
    )
    income_per_person = _shape(
        "Income per person",
        "Person",
        "income",
        "measuredValue",
        None,
        "Count_Person",
        members=("Amount_Income_Person_AsFractionOf_Count_Person",),
    )
    pins = query_pins(
        "percent of population", quantity=quantity_requirement(["percent of population"])
    )
    result = inadmissible_alternates(pins, share, [(0, income_per_person)])
    assert _code(result[0]) == "pin:normalisation", result


# An unrequested per-area (density) alternate of a bare count.


def _r7_pair() -> tuple[ShapeDraft, ShapeDraft]:
    return (
        _shape(
            "Total population",
            "Person",
            "count",
            "measuredValue",
            None,
            None,
            members=("Count_Person",),
        ),
        _shape(
            "Population Density",
            "Person",
            "count",
            "measuredValue",
            None,
            "area",
            members=("Count_Person_PerArea",),
        ),
    )


def test_per_area_alternate_rejected_on_bare_count_phrase():
    """The density series specialises "population of Kenya"."""
    elected, density = _r7_pair()
    result = inadmissible_alternates(
        query_pins("population of Kenya", quantity=quantity_requirement(["population of Kenya"])),
        elected,
        [(0, density)],
    )
    assert _code(result[0]) == "specialisation:normalisation", result


@pytest.mark.parametrize(
    "variable",
    [
        # A token that starts with the `dens` stem.
        "population density of Kenya",
        "Population Density of Kenya",
        "population-density",
        "population densities",
        "how densely populated is Kenya",
        "most dense countries",
        "denser countries",
        "densest country",
        "pop. dens.",
        "bone density",
        # A unit that names an area on its own, with any wording around it.
        "population per ha",
        "people/ha",
        "hospitals/ha noi",
        "people per hectare",
        "people per hectares",
        "people per acre",
        "trees per acres",
        "people a hectare",
        "people for every hectare",
        "people in each acre",
        "per-hectare population",
        "people per thousand hectares",
        "people per 1 000 hectares",
        "people per 10 000 ha",
        "people per 1000ha",
        "hectares of forest",
        # A square marker before each length unit.
        "population per sq km",
        "people per sq kms",
        "population per square kilometre",
        "people per square foot",
        "people per square feet",
        "people per sq ft",
        "people per ft2",
        "people per square yard",
        "people per square yards",
        "people per sq yd",
        "people per sq cm",
        "people per square centimetre",
        "people per square centimetres",
        "people per square centimeter",
        "people per square centimeters",
        "population per square kilometres",
        "population per square kilometer",
        "population per square kilometers",
        "population per sq mi",
        "population per square mile",
        "population per square miles",
        "population per sq m",
        "population per square metre",
        "population per square metres",
        "population per square meter",
        "population per square meters",
        "people per squared kilometre",
        # Separators between the marker and the unit.
        "population per sq. km",
        "population per square-kilometre",
        "people per sq_km",
        "people per  square   mile",
        "PEOPLE PER SQ. KM",
        # A square marker with no `per` before it.
        "people a square kilometre",
        "population in each square kilometre",
        "people/sq mi",
        "land area in sq. km",
        # A square marker after the length unit.
        "people per km sq",
        "people per mi sq.",
        "people per kilometre square",
        "population per kilometre squared",
        # A square marker written against its unit.
        "people per sqkm",
        "people per squaredkm",
        "people per sqm",
        "people per squaremile",
        # A length unit then the exponent `2` or `²`.
        "population per km2",
        "People per KM2",
        "population per km²",
        "people / km²",
        "population/mi²",
        "population per mi2",
        "people per m2",
        "people per km^2",
        "people per km**2",
        "people per km 2",
        "people per km ²",
        "people per km-2",
        "people per 1 000 km2",
        "people per thousand km2",
        "people per-km2",
        "population\tper\tkm2",
        "population per\u00a0km2",
        # `area` or `areas`, wherever it stands.
        "population per area",
        "wells per areas",
        "per-area population",
        "population/area",
        "population per unit area",
        "population per unit of area",
        "population per land area",
        "population per land-area",
        "land area of Kenya",
        "people per surface area",
        "people per total area",
        "population by area",
        "population relative to its area",
        "population relative to area",
        "population in urban area",
        "area of Kenya",
        "area per person",
        "people living in area",
    ],
)
def test_per_area_alternate_kept_when_phrase_names_the_framing(variable):
    """The density alternate stands when the phrase holds an area token."""
    elected, density = _r7_pair()
    assert (
        inadmissible_alternates(
            query_pins(variable, quantity=quantity_requirement([variable])), elected, [(0, density)]
        )
        == {}
    )


@pytest.mark.parametrize(
    "variable",
    [
        # No framing at all.
        "population per year",
        "deaths per 100,000",
        # Wording with no area token.
        "how crowded is Kenya",
        "sparsely populated countries",
        # A word that only begins with `area`.
        "spend per areawide programme",
        # A length unit with no square marker or exponent is a distance.
        "road deaths per mile driven",
        "cost per km of road",
        "emissions per km",
        "wells per miles",
        "spend/mi",
        "urban/km",
        "people per m",
        # The exponent token must equal `2` and must follow the unit.
        "water per m3",
        "water per km3",
        "people per km20",
        "people per km12",
        "roads per 2 km",
        # A square marker must sit next to the length unit.
        "population per square",
        "town square markets",
        "town square 2 km away",
        "m/square",
        "people per square nautical mile",
        # A unit spelling inside a longer word.
        "population per kmart store",
        "and/sql users",
        "yield/harvest",
        "population per hamlet",
        "crops per acreage",
        "sqmap users",
        # The `dens` stem inside a longer word, and a shorter stem.
        "condenser production",
        "dental visits",
    ],
)
def test_per_area_alternate_rejected_when_phrase_names_no_density_framing(variable):
    """The density alternate specialises phrases that hold no area token."""
    elected, density = _r7_pair()
    result = inadmissible_alternates(
        query_pins(variable, quantity=quantity_requirement([variable])), elected, [(0, density)]
    )
    assert 0 in result, f"admitted the density alternate on {variable!r}"
    assert _code(result[0]) == "specialisation:normalisation", result[0]


def test_per_area_alternate_rejected_when_phrase_names_a_share():
    """`percent of population` names a share, which a density does not honour."""
    elected, density = _r7_pair()
    result = inadmissible_alternates(
        query_pins(
            "percent of population", quantity=quantity_requirement(["percent of population"])
        ),
        elected,
        [(0, density)],
    )
    assert _code(result[0]) == "pin:normalisation", result


# Legitimate corpus candidate pairs must survive in either election order.


def _T(
    label: str,
    pop: str,
    prop: str,
    stat: str,
    qual: str | None,
    denom: str | None,
    *,
    unresolved: bool = False,
) -> tuple:
    return (label, pop, prop, stat, qual, denom, unresolved)


NON_REJECT_PAIRS = [
    (
        "forest-area-compare/ethiopia",
        "compare forest cover in Kenya and Tanzania",
        # FakeGraph has no arcs here. Use the observed SDG tuple convention.
        _T("Proportion of forest area", "SDG_AG_LND_FRST", "value", "measuredValue", None, None),
        _T("Forest area (national)", "SDG_AG_LND_FRSTN", "value", "measuredValue", None, None),
    ),
    (
        "renewable-energy-share",
        "share of renewable energy in total energy",
        # FakeGraph has no arcs. Use the observed World Bank tuple convention.
        _T(
            "Renewable energy share of total final energy consumption",
            "SDG_EG_FEC_RNEW",
            "value",
            "measuredValue",
            None,
            None,
        ),
        _T(
            "Share of renewable energy in electricity",
            "Energy",
            "worldBank/4_1_SHARE_RE_IN_ELECTRICITY",
            "measuredValue",
            None,
            None,
        ),
    ),
    (
        "synth-poverty-bangladesh-what",
        "poverty in Bangladesh",
        _T(
            "Proportion of population below international poverty line",
            "SDG_SI_POV_DAY1",
            "value",
            "measuredValue",
            None,
            None,
        ),
        _T(
            "Proportion of population living below the national poverty line",
            "SDG_SI_POV_NAHC",
            "value",
            "measuredValue",
            None,
            None,
        ),
    ),
    (
        "sdg-06/govt-education-expenditure",
        "government education spending Kenya",
        _T(
            "Percent of total government expenditure on education",
            "EconomicActivity",
            "amount",
            "measuredValue",
            None,
            "Amount_EconomicActivity_ExpenditureActivity_Government",
        ),
        _T(
            "Percent of nominal GDP spent on education",
            "EconomicActivity",
            "amount",
            "measuredValue",
            None,
            "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
        ),
    ),
    (
        "std-03",
        "birth rate Ethiopia",
        _T("Crude Birth Rate", "BirthEvent", "count", "measuredValue", None, "Count_Person"),
        _T("Fertility rate of women", "Person", "fertilityRate", "measuredValue", None, None),
    ),
    (
        "std-05-a",
        "under-5 child mortality rate",
        _T(
            "Mortality Rate of Children Younger Than 4 Years",
            "Person",
            "",
            "measuredValue",
            None,
            "Count_BirthEvent_LiveBirth",
        ),
        _T(
            "Infant Mortality Rate",
            "MortalityEvent",
            "count",
            "measuredValue",
            None,
            "Count_BirthEvent_LiveBirth",
        ),
    ),
    (
        "std-05-b",
        "under-5 child mortality rate",
        _T(
            "Mortality Rate of Children Younger Than 4 Years",
            "Person",
            "",
            "measuredValue",
            None,
            "Count_BirthEvent_LiveBirth",
        ),
        _T(
            "Under-five mortality rate [under 5 years old]",
            "SDG_SH_DYN_MORT",
            "value",
            "measuredValue",
            None,
            None,
        ),
    ),
    (
        "std-05-c-who-minted",
        "under-5 child mortality rate",
        _T(
            "Mortality Rate of Children Younger Than 4 Years",
            "Person",
            "",
            "measuredValue",
            None,
            "Count_BirthEvent_LiveBirth",
        ),
        _T("Number of Under-Five Deaths", "Person", "who/CM_01", "measuredValue", None, None),
    ),
    (
        "cand-r2",
        "income in California",
        _T("Median Income of a Population", "Person", "income", "medianValue", None, None),
        _T("Median Income for All Households", "Household", "income", "medianValue", None, None),
    ),
    (
        "nn-03/under5-mortality-rate",
        "child mortality rate Ethiopia",
        _T(
            "Infant Mortality Rate",
            "MortalityEvent",
            "count",
            "measuredValue",
            None,
            "Count_BirthEvent_LiveBirth",
        ),
        _T(
            "Mortality Rate of Children Younger Than 4 Years",
            "Person",
            "",
            "measuredValue",
            None,
            "Count_BirthEvent_LiveBirth",
        ),
    ),
    (
        "digital-access-latest",
        "most recent data on internet access",
        _T(
            "Percent of Internet Users",
            "Person",
            "count",
            "measuredValue",
            None,
            None,
            unresolved=True,
        ),
        _T(
            "Proportion of individuals using the Internet",
            "SDG_IT_USE_ii99",
            "value",
            "measuredValue",
            None,
            None,
        ),
    ),
    (
        "synth-ghg-emissions-nigeria",
        "greenhouse gas emissions in Nigeria",
        _T(
            "Annual Amount of Emissions: Greenhouse Gas",
            "Emissions",
            "amount",
            "measuredValue",
            "Annual",
            None,
        ),
        # FakeGraph has no arcs. Use the observed SDG tuple convention.
        _T("Total CO2 emissions", "SDG_EN_ATM_CO2", "value", "measuredValue", None, None),
    ),
    (
        "dq-crs-003",
        "aid from Japan",
        # A grant equivalent is another measure of the same flow on the declared
        # `measuredProperty` axis, and a level of it divides nothing.
        _T(
            "Gross disbursements of ODA",
            "FinancialFlow",
            "amount",
            "measuredValue",
            "Nominal",
            None,
        ),
        _T(
            "Grant equivalents of ODA",
            "FinancialFlow",
            "grantEquivalent",
            "measuredValue",
            "Nominal",
            None,
        ),
    ),
    (
        "subj-education-data",
        "What education data do you have?",
        # A ratio whose numerator is another quantity does not divide the elected count.
        _T(
            "Count Of Civic Structure: Education Facility",
            "CivicStructure",
            "count",
            "measuredValue",
            None,
            None,
        ),
        _T(
            "Percent of nominal GDP spent on education",
            "EconomicActivity",
            "amount",
            "measuredValue",
            None,
            "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
        ),
    ),
]


@pytest.mark.parametrize(
    "golden_ids,variable,a,b",
    NON_REJECT_PAIRS,
    ids=[p[0] for p in NON_REJECT_PAIRS],
)
def test_legitimate_pair_never_rejected(golden_ids, variable, a, b):
    shape_a = _shape(*a[:6], unresolved=a[6], members=("sv_a",))
    shape_b = _shape(*b[:6], unresolved=b[6], members=("sv_b",))
    a_elected = inadmissible_alternates(
        query_pins(variable, quantity=quantity_requirement([variable])), shape_a, [(0, shape_b)]
    )
    b_elected = inadmissible_alternates(
        query_pins(variable, quantity=quantity_requirement([variable])), shape_b, [(0, shape_a)]
    )
    assert a_elected == {}, f"{golden_ids}: rejected b when a elected: {a_elected}"
    assert b_elected == {}, f"{golden_ids}: rejected a when b elected: {b_elected}"


# Measure class, read from query phrases and recorded labels.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HIV prevalence", "prevalence"),
        ("measles incidence rate", "incidence"),
        ("maternal mortality ratio", "mortality"),
        ("HIV", None),
        (
            "3.3.1: Number of new HIV infections per 1,000 uninfected population, "
            "by sex, age and key populations",
            "incidence",
        ),
        (
            "Number Of New Reported Cases Of Human African Trypanosomiasis (T.B. Gambiense)",
            "incidence",
        ),
        ("People living with HIV", "prevalence"),
        ("Deaths among people living with HIV (all causes)", "mortality"),
        ("HIV prevalence and deaths", None),
        ("Cause-specific mortality rate among children", "mortality"),
        (
            "Cause-specific mortality rate among children [Age=Under 5 years old, Global health "
            "estimate cause=Measles]",
            "mortality",
        ),
        ("Count of pediatric mortality Influenza incidents", "mortality"),
        ("Count of HIV/Acquired Immune Deficiency Syndrome incidents", None),
        ("Number of Measles Cases", None),
        ("Number of people with health insurance", None),
    ],
)
def test_measure_class_reads_the_class_a_text_names(text, expected):
    assert measure_class(text) == expected


def _family(pop_type_name: str | None, label: str = "label") -> ShapeDraft:
    return dataclasses.replace(
        _shape(label, "Pop", "value", "measuredValue", None, None), pop_type_name=pop_type_name
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(
            _family("Number of new HIV infections per 1,000 uninfected population, by sex and age"),
            "incidence",
            id="sdg-hiv-incidence",
        ),
        pytest.param(_family("HIV prevalence"), "prevalence", id="unaids-hiv-prevalence"),
        pytest.param(_family("People living with HIV"), "prevalence", id="unaids-plwh"),
        pytest.param(
            _family("Deaths among people living with HIV (all causes)"),
            "mortality",
            id="unaids-plhiv-deaths",
        ),
        pytest.param(
            _family("Cause-specific mortality rate among children"),
            "mortality",
            id="who-under-five-cause-specific-mortality",
        ),
        pytest.param(_family("MortalityEvent"), "mortality", id="camel-case-mortality-event"),
        pytest.param(_family("Death"), None, id="singular-death"),
        # Recall reaching only the influenza pediatric mortality member labels the family
        # with that member's name, which speaks for no other member.
        pytest.param(
            _family(
                "MedicalConditionIncident", label="Count of pediatric mortality Influenza incidents"
            ),
            None,
            id="sampled-member-does-not-decide",
        ),
        pytest.param(
            _family(None, label="Number of new HIV infections per 1,000 uninfected population"),
            None,
            id="no-population-type-name",
        ),
    ],
)
def test_shape_measure_class_reads_the_population_type_name(shape, expected):
    assert shape_measure_class(shape) == expected


# Pipeline tests cover guard rejections and near-neighbour warning outcomes.

_TSTLAND = {"country/TST": {"label": "Testland", "type": "Country"}}


def _arcs(
    *,
    pop_type: str,
    meas_prop: str,
    name: str,
    meas_qual: str | None = None,
    meas_denom: str | None = None,
    meas_denom_unresolved: bool = False,
) -> dict:
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": name}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if meas_qual is not None:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": meas_qual}]}
    if meas_denom is not None:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": meas_denom}]}
    elif meas_denom_unresolved:
        arcs["measurementDenominator"] = {
            "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
        }
    return arcs


def _node(**kwargs) -> dict:
    return {"label": kwargs["name"], "type": "StatisticalVariable", "arcs": _arcs(**kwargs)}


def _facet() -> dict:
    return {
        "earliestDate": "2015",
        "latestDate": "2020",
        "obsCount": 5,
        "provenanceId": "source/Test",
    }


class _FixedBindLLM:
    """Return fixed shape and alternate indices for slot-free candidates."""

    def __init__(self, *, chosen_shape_index: int, also_plausible: list[int] | None):
        self._chosen_shape_index = chosen_shape_index
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen_shape_index,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


def _sv_refs(spec) -> set[str]:
    return {sv.ref.dcid for sv in spec.resolved.members}


def _warning_codes(result) -> set[str]:
    return {w.code for w in result.warnings}


def _run(
    *,
    query: str,
    nodes: dict,
    svs: list[str],
    scores: list[float],
    chosen_shape_index: int,
    also_plausible: list[int] | None,
    obs: dict,
    qualifiers: tuple[CodedQualifier, ...] | None = None,
):
    graph = FakeGraph(
        nodes={**_TSTLAND, **nodes},
        detect={query: {"svs": svs, "cosine_scores": scores}},
        resolve={"Testland": "country/TST"},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Testland"],
            date_request=None,
            detect_query=query,
            role_query=query,
            qualifiers=qualifiers,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(chosen_shape_index=chosen_shape_index, also_plausible=also_plausible),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


# An unresolved per-capita winner beside a plain-total alternate.


# The pin reads the leg's quantity requirement, so a sidecar phrase naming the framing pins
# it as the variable does.
@pytest.mark.parametrize(
    "query,qualifiers",
    [
        ("GDP per capita of Testland", None),
        ("GDP of Testland", (CodedQualifier(phrase="per capita", spans=(), matches=()),)),
    ],
    ids=["variable", "sidecar-phrase"],
)
def test_pipeline_r1_contradicted_denominator_rejected(query, qualifiers):
    elected_sv, alt_sv = "Test_GDP_PerCapita", "Test_GDP_Total"
    result = _run(
        query=query,
        qualifiers=qualifiers,
        nodes={
            elected_sv: _node(
                pop_type="EconomicActivity",
                meas_prop="amount",
                name="Nominal gross domestic product per capita",
                meas_qual="Nominal",
                meas_denom_unresolved=True,
            ),
            alt_sv: _node(
                pop_type="EconomicActivity",
                meas_prop="amount",
                name="Nominal gross domestic product",
                meas_qual="Nominal",
            ),
        },
        svs=[elected_sv, alt_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert _sv_refs(result.specs[0]) == {elected_sv}


# A ratio query beside an alternate raw count: the normalisation pin refuses the count.


def test_pipeline_r2_normalised_vs_raw_count_rejected():
    elected_sv, alt_sv = "Test_MortalityRatio", "Test_MortalityCount"
    result = _run(
        query="mortality ratio in Testland",
        nodes={
            elected_sv: _node(
                pop_type="MortalityCondition",
                meas_prop="value",
                name="Mortality ratio",
            ),
            alt_sv: _node(
                pop_type="MortalityEvent",
                meas_prop="count",
                name="Count of deaths",
            ),
        },
        svs=[elected_sv, alt_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert _sv_refs(result.specs[0]) == {elected_sv}


# A rural alternate that narrows a general query: residence specialisation refuses it.


def test_pipeline_r3_population_narrowing_rejected():
    # Distinct measuredProperty values keep derive_shapes from merging these shapes.
    elected_sv, alt_sv = "Test_Access_National", "Test_Access_Rural"
    result = _run(
        query="access to something in Testland",
        nodes={
            elected_sv: _node(
                pop_type="Electricity",
                meas_prop="access_national",
                name="Access to something (national)",
            ),
            alt_sv: _node(
                pop_type="Electricity",
                meas_prop="access_rural",
                name="Access to something, rural",
            ),
        },
        svs=[elected_sv, alt_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert _sv_refs(result.specs[0]) == {elected_sv}


# An alternate that differs from the winner in both denominator and qualifier.


def test_pipeline_r4_qual_and_denom_delta_rejected():
    elected_sv, alt_sv = "Test_Emissions_Annual", "Test_Emissions_PerCapita"
    result = _run(
        query="emissions in Testland",
        nodes={
            elected_sv: _node(
                pop_type="Emissions",
                meas_prop="amount",
                name="Annual emissions total",
                meas_qual="Annual",
            ),
            alt_sv: _node(
                pop_type="Emissions",
                meas_prop="amount",
                name="Emissions per capita",
                meas_denom_unresolved=True,
            ),
        },
        svs=[elected_sv, alt_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert _sv_refs(result.specs[0]) == {elected_sv}


# A model-nominated density series beside a plain count query: normalisation specialisation drops it.


def test_pipeline_r7_density_nomination_rejected():
    elected_sv, alt_sv = "Test_Count_Person", "Test_Count_Person_PerArea"
    result = _run(
        query="population of Testland",
        nodes={
            elected_sv: _node(pop_type="Person", meas_prop="count", name="Total population"),
            alt_sv: _node(
                pop_type="Person",
                meas_prop="count",
                name="Population Density",
                meas_denom="area",
            ),
        },
        svs=[elected_sv, alt_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert _sv_refs(result.specs[0]) == {elected_sv}


# The same nomination survives when the query names density.


def test_pipeline_r7_density_nomination_kept_when_query_names_density():
    elected_sv, alt_sv = "Test_Count_Person", "Test_Count_Person_PerArea"
    result = _run(
        query="population density of Testland",
        nodes={
            elected_sv: _node(pop_type="Person", meas_prop="count", name="Total population"),
            alt_sv: _node(
                pop_type="Person",
                meas_prop="count",
                name="Population Density",
                meas_denom="area",
            ),
        },
        svs=[elected_sv, alt_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2
    assert {ref for s in result.specs for ref in _sv_refs(s)} == {elected_sv, alt_sv}


# The margin gate injects a raw-count alternate. The normalisation pin refuses it and clears the warning.


def test_pipeline_fired_index_rejected_retracts_warning():
    elected_sv, alt_sv = "Test_Incidence_Alpha", "Test_Incident_Beta"
    result = _run(
        query="sickness rate reporting in Testland",
        nodes={
            elected_sv: _node(
                pop_type="MedicalCondition",
                meas_prop="incidenceRateAlpha",
                name="Disease Incidence Alpha Index",
            ),
            alt_sv: _node(
                pop_type="MedicalConditionEvent",
                meas_prop="count",
                name="Count Of Disease Cases Beta",
            ),
        },
        svs=[elected_sv, alt_sv],
        # A 0.01 score gap is below the 0.025 margin. With 0.0 label similarity,
        # the gate fires on alt_sv without a model nomination.
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=None,
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert _sv_refs(result.specs[0]) == {elected_sv}
    assert NEAR_NEIGHBOR_AMBIGUOUS not in _warning_codes(result)


# The gate-injected ordinary-count alternate is a candidate with its warning.


def test_pipeline_fired_index_untouched_still_surfaces():
    elected_sv, alt_sv = "Test_Widget_Count", "Test_Gadget_Count"
    result = _run(
        query="widget counting in Testland",
        nodes={
            elected_sv: _node(pop_type="Widget", meas_prop="count", name="Population Count Widget"),
            alt_sv: _node(pop_type="Gadget", meas_prop="count", name="Population Count Gadget"),
        },
        svs=[elected_sv, alt_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=None,
        obs={
            f"{elected_sv}|country/TST": [_facet()],
            f"{alt_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "candidates"
    assert len(result.specs) == 2
    assert {ref for s in result.specs for ref in _sv_refs(s)} == {elected_sv, alt_sv}
    assert NEAR_NEIGHBOR_AMBIGUOUS in _warning_codes(result)
