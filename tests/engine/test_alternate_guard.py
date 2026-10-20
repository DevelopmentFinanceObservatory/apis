"""Test alternate admissibility and its pipeline effects.

Unit cases cover refused and legitimate shape pairs, asserting the refusal's reason code.
Pipeline cases check model-nominated and near-neighbour alternatives against offline graph
fixtures.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys

import pytest

from code_index.shared import get_measure_index
from search.engine.bind import _UniversalBindOutput
from search.engine.qualifier_match import CodedQualifier
from search.engine.quantity import (
    DenominatorRead,
    NamedDenominator,
    check_denominator,
    normalisation_class,
    quantity_requirement,
    read_denominators,
    serves_every_form,
    with_stated_form,
)
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.alternate_guard import (
    AlternateJudge,
    QueryPins,
    _price_basis_vocabulary,
    inadmissible_alternates,
    inadmissible_reason,
    measure_class,
    query_pins,
    shape_measure_class,
)
from search.engine.region.phase_rank_bind import judge_value_requirements
from search.engine.region.requirements import judge, meets_stray_values
from search.engine.region.state import Requirement, ValueRule, unapplied_departure
from search.engine.shape import ShapeDraft
from search.models import GraphRef, ResolvedQualifier
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
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
            "name": {"nodes": [{"value": "Gross national income"}]},
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


def test_read_denominators_reads_statvar_denominators_in_one_batch():
    """Schema terms are not read; each StatVar denominator is read once, for its kind and
    its graph name."""
    graph = _CountingGraph()
    reads = read_denominators(
        [_ADULTS, "Count_Person", _GNI, "PerCapita", "area", "Area_Land", _ADULTS, "Unloaded"],
        graph=graph,
    )
    assert graph.batches == [[_ADULTS, _GNI, "Area_Land", "Unloaded"]]
    assert reads == {
        _ADULTS: DenominatorRead(kind="person", name=None),
        _GNI: DenominatorRead(kind="other", name="Gross national income"),
        "Area_Land": DenominatorRead(kind="area", name=None),
        "Unloaded": DenominatorRead(kind="unknown", name=None),
    }


def test_read_denominators_skips_the_read_with_no_statvar_denominator():
    graph = _CountingGraph()
    assert read_denominators(["Count_Person", "PerCapita", "area"], graph=graph) == {}
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


@pytest.mark.parametrize(
    ("denominator", "expected"),
    [("PerCapita", "per_person"), ("Count_Person", "per_person"), (None, "unknown")],
    ids=["per-capita-provisional", "count-person-provisional", "dcid-less"],
)
def test_an_unresolved_denominator_arc_reads_its_schema_term(denominator, expected):
    """Base Data Commons holds `PerCapita` as a ProvisionalNode: its dcid still names the
    per-person base. An arc with no dcid says nothing of its base."""
    shape = _shape("x", "EconomicActivity", "amount", "measuredValue", None, denominator)
    shape = dataclasses.replace(shape, meas_denom_unresolved=True)
    assert normalisation_class(shape) == expected


_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"


@pytest.mark.parametrize(
    "shape,named,expected,asked",
    [
        (
            _shape("x", "Debt", "amount", "measuredValue", None, _GDP),
            NamedDenominator(ref=GraphRef(ref=_GDP, label="GDP"), base_words="GDP"),
            "matches",
            None,
        ),
        (
            _shape("x", "Debt", "amount", "measuredValue", None, _GNI),
            NamedDenominator(ref=GraphRef(ref=_GDP, label="GDP"), base_words="GDP"),
            "refused",
            f"the query names the denominator {_GDP},",
        ),
        # "None of these" is no match, which proves nothing about the arc.
        (
            _shape("x", "Debt", "amount", "measuredValue", None, _GNI),
            NamedDenominator(ref=None, base_words="exports"),
            "unverified",
            None,
        ),
        (
            _shape("x", "Debt", "amount", "measuredValue", None, "Count_Person"),
            NamedDenominator(ref=GraphRef(ref=_GDP, label="GDP"), base_words="GDP"),
            "refused",
            f"the query names the denominator {_GDP},",
        ),
        # A series whose base only its name states passes unverified.
        (
            _shape("x", "Debt", "value", "measuredValue", None, None),
            NamedDenominator(ref=None, base_words="exports"),
            "unverified",
            None,
        ),
        (
            _shape("x", "Debt", "amount", "measuredValue", None, _GNI, unresolved=True),
            NamedDenominator(ref=GraphRef(ref=_GDP, label="GDP"), base_words="GDP"),
            "unverified",
            None,
        ),
        # The ProvisionalNode `PerCapita` states its base as `Count_Person` does.
        (
            _shape("x", "Debt", "amount", "measuredValue", None, "PerCapita", unresolved=True),
            NamedDenominator(ref=GraphRef(ref=_GDP, label="GDP"), base_words="GDP"),
            "refused",
            f"the query names the denominator {_GDP},",
        ),
        # A level divides by nothing, whichever base the question names.
        (
            _shape("x", "Debt", "amount", "measuredValue", None, None),
            NamedDenominator(ref=GraphRef(ref=_GDP, label="GDP"), base_words="GDP"),
            "refused",
            f"the query names the denominator {_GDP},",
        ),
        (
            _shape("x", "Debt", "amount", "measuredValue", None, None),
            NamedDenominator(ref=None, base_words="exports"),
            "refused",
            'the query names the denominator "exports",',
        ),
    ],
    ids=[
        "arc-matches",
        "arc-differs",
        "none-of-these-leaves-an-arc-unverified",
        "schema-term-arc-differs",
        "arcless-other-unverified",
        "unresolved-arc-unverified",
        "provisional-schema-term-differs",
        "level-refused",
        "level-refused-under-none-of-these",
    ],
)
def test_check_denominator_holds_the_shape_to_the_named_denominator(shape, named, expected, asked):
    """A refused alternate's reason names the denominator the question names: its dcid when
    listed, else the question's own words for it."""
    assert check_denominator(shape, named) == expected
    elected = _shape("x", "Debt", "amount", "measuredValue", None, _GDP)
    reason = inadmissible_reason(QueryPins(named_denominator=named), elected, shape)
    if asked is None:
        assert reason is None or "names the denominator" not in reason
    else:
        assert reason is not None and asked in reason


@pytest.mark.parametrize(
    "variable,stated,named",
    [
        ("debt to GDP", "rate", {"rate"}),
        ("debt to GDP", None, set()),
        # A form a phrase names outranks the one the model states.
        ("debt to GDP ratio", "per_person", {"rate"}),
    ],
    ids=["model-form-joins", "no-form-stated", "phrase-form-wins"],
)
def test_with_stated_form_joins_only_when_no_phrase_names_a_form(variable, stated, named):
    requirement = with_stated_form(quantity_requirement([variable]), stated)
    assert requirement.named == named
    assert requirement.named <= requirement.framing


@pytest.mark.parametrize(
    "variable,shape,expected",
    [
        ("debt to GDP ratio", _shape("x", "Debt", "amount", "measuredValue", None, None), False),
        (
            "debt as a share of GDP",
            _shape("x", "Debt", "amount", "measuredValue", None, _GDP),
            True,
        ),
        ("population density", _shape("x", "Person", "count", "measuredValue", None, None), False),
        # "Incidence per capita" names a rate and a per-person form; a rate series that is
        # not per person honours only the rate.
        (
            "incidence per capita",
            _shape("x", "Event", "incidence", "measuredValue", None, None),
            False,
        ),
        (
            "homicide rate per 100,000 people",
            _shape("x", "Event", "rate", "measuredValue", None, None),
            True,
        ),
    ],
    ids=[
        "level-for-rate",
        "series-for-share",
        "count-for-density",
        "rate-not-per-person",
        "other-per-head-base",
    ],
)
def test_serves_every_form_holds_every_named_form(variable, shape, expected):
    assert serves_every_form(quantity_requirement([variable]), shape) is expected


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


def test_per_area_alternate_refused_when_the_phrase_also_names_a_per_person_form():
    """ "area per person" frames an area and names a per-person form. A density honours the
    area alone, so the guard refuses it as the materialise gate refuses an elected shape
    that does not honour every form named."""
    elected, density = _r7_pair()
    variable = "area per person"
    result = inadmissible_alternates(
        query_pins(variable, quantity=quantity_requirement([variable])), elected, [(0, density)]
    )
    assert _code(result[0]) == "pin:normalisation", result


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
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen_shape_index,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


def _sv_refs(spec) -> set[str]:
    return {sv.ref.ref for sv in spec.resolved.members}


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


# A query naming density holds a plain count off the menu, so bind elects the density;
# the judge keeps the density nomination itself (`test_per_area_alternate_kept_when_phrase_names_the_framing`).


def test_pipeline_r7_density_query_never_elects_a_plain_count():
    count_sv, density_sv = "Test_Count_Person", "Test_Count_Person_PerArea"
    result = _run(
        query="population density of Testland",
        nodes={
            count_sv: _node(pop_type="Person", meas_prop="count", name="Total population"),
            density_sv: _node(
                pop_type="Person",
                meas_prop="count",
                name="Population Density",
                meas_denom="area",
            ),
        },
        svs=[count_sv, density_sv],
        scores=[0.90, 0.89],
        chosen_shape_index=0,
        also_plausible=None,
        obs={
            f"{count_sv}|country/TST": [_facet()],
            f"{density_sv}|country/TST": [_facet()],
        },
    )
    assert result.status == "definite"
    (spec,) = result.specs
    assert _sv_refs(spec) == {density_sv}


def _flow_amount(label: str, prop: str, value: str | None) -> ShapeDraft:
    """A FinancialFlow amount with one member and no slot, holding `value` on `prop` when
    given."""
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if value is not None:
        arcs["constraintProperties"] = {"nodes": [{"dcid": prop}]}
        arcs[prop] = {"nodes": [{"dcid": value}]}
    return ShapeDraft(
        shape_id=label,
        label=label,
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={f"Amount_{label}": arcs},
    )


@pytest.mark.parametrize(
    ("prop", "elected", "stray", "held", "measured", "admitted"),
    [
        ("crsPurpose", None, "CRSPurposeCode120", "CRSPurposeCode120", "amount", True),
        ("crsPurpose", None, "CRSPurposeCode120", "CRSPurposeCode121", "amount", True),
        ("crsPurpose", None, "CRSPurposeCode120", "CRSPurposeCode110", "amount", False),
        ("crsPurpose", None, "CRSPurposeCode120", None, "amount", False),
        ("crsPurpose", None, "CRSPurposeCode120", "CRSPurposeCode110", "adjustedAmount", False),
        ("crsPurpose", None, "CRSPurposeCode120", None, "adjustedAmount", False),
        ("crsPurpose", None, "CRSPurposeCode120", "CRSPurposeCode121", "adjustedAmount", True),
        (
            "crsPurpose",
            "CRSPurposeCode120",
            "CRSPurposeCode121",
            "CRSPurposeCode121",
            "amount",
            True,
        ),
        (
            "crsPurpose",
            "CRSPurposeCode120",
            "CRSPurposeCode121",
            "CRSPurposeCode122",
            "amount",
            False,
        ),
        ("gender", "Female", "Male", "Male", "amount", True),
        ("gender", "Female", "Male", "Female", "amount", False),
    ],
    ids=[
        "the-stray-code",
        "a-descendant",
        "another-code",
        "no-value-on-the-property",
        "another-family-another-code",
        "another-family-no-value-on-the-property",
        "another-family-a-descendant",
        "elected-fixes-the-parent-member-holds-the-stray",
        "elected-fixes-the-parent-member-holds-a-sibling",
        "elected-fixes-female-member-holds-the-stray",
        "elected-fixes-female-member-holds-it-too",
    ],
)
def test_the_guard_admits_a_stray_value_exactly_where_the_ledger_judges_it_met(
    prop, elected, stray, held, measured, admitted
):
    """The bind drafted `stray` on `prop`, which the elected amount has no slot for: its
    one member holds no value there, or holds `elected` at a fixed value. An alternate
    member is admitted exactly when its value applies the stray (`applies_value`), as a
    binding of the stray reads, whatever the elected reading holds and whichever family the
    alternate measures: purpose 120's descendant 121 applies it, education 110, the sibling
    122 and a member with no value on `prop` do not, and a member holding Female does not
    apply Male. The ledger judges the stray met on exactly the members the guard admits,
    and every other member is pinned out on `prop`."""
    variable = "aid"
    pins = query_pins(
        variable, stray_values=[(prop, stray)], quantity=quantity_requirement([variable])
    )
    requirement = Requirement(
        term=ResolvedQualifier(
            text=stray,
            named=[GraphRef(ref=stray, label=stray)],
            disposition="unbound",
            evidence="model",
        ),
        origin="stray",
        rule=ValueRule(values=frozenset({(prop, stray)})),
    )
    alternate = dataclasses.replace(
        _flow_amount(held or "missing", prop, held), meas_prop_dcid=measured
    )

    reason = AlternateJudge(pins, _flow_amount("elected", prop, elected)).verdict(alternate).reason
    unmet = judge_value_requirements([requirement], alternate, [])

    assert (reason is None, not unmet) == (admitted, admitted)
    if not admitted:
        assert reason is not None and _code(reason) == f"pin:{prop}"


def test_the_guard_offers_a_mixed_alternate_as_the_members_that_meet_the_stray():
    """The bind drafted Male on gender, which the elected amount has no slot for. An
    alternate whose members hold Male and Female is offered narrowed to the Male member
    (`AlternateJudge.offered`), the one member that applies the stray, and the ledger
    judges the stray met on the alternate as offered, and unmet on it whole."""
    variable = "aid"
    stray = ("gender", "Male")
    pins = query_pins(variable, stray_values=[stray], quantity=quantity_requirement([variable]))
    requirement = Requirement(
        term=ResolvedQualifier(
            text="Male", named=[GraphRef(ref="Male", label="Male")], disposition="unbound"
        ),
        origin="stray",
        rule=ValueRule(values=frozenset({stray})),
    )
    male, female = (
        _flow_amount("male", "gender", "Male"),
        _flow_amount("female", "gender", "Female"),
    )
    mixed = dataclasses.replace(
        male, sv_arc_facts={**(male.sv_arc_facts or {}), **(female.sv_arc_facts or {})}
    )
    judge = AlternateJudge(pins, _flow_amount("elected", "gender", None))

    verdict = judge.verdict(mixed)
    offered = judge.offered(mixed)

    assert verdict.reason is None
    assert [member.dcid for member in verdict.passing] == ["Amount_male"]
    assert list(offered.sv_arc_facts or {}) == ["Amount_male"]
    assert judge_value_requirements([requirement], offered, []) == ()
    assert judge_value_requirements([requirement], mixed, []) == (requirement,)


@pytest.mark.parametrize(
    ("prop", "drafted", "held", "admitted"),
    [
        ("gender", ("Female", "Male"), ("Female",), True),
        ("gender", ("Female", "Male"), ("Male",), True),
        ("gender", ("Female", "Male"), ("Female", "Male"), True),
        ("gender", ("Female", "Male"), (None,), False),
        ("crsPurpose", ("CRSPurposeCode120", "CRSPurposeCode121"), ("CRSPurposeCode121",), True),
        ("crsPurpose", ("CRSPurposeCode120", "CRSPurposeCode121"), ("CRSPurposeCode110",), False),
    ],
    ids=[
        "one-of-the-set",
        "the-other-of-the-set",
        "members-holding-each",
        "no-value-on-the-property",
        "a-code-under-one",
        "a-code-outside-the-set",
    ],
)
def test_the_guard_and_the_ledger_read_a_stray_set_by_membership(prop, drafted, held, admitted):
    """The bind drafted the set `drafted` on `prop`, which the elected amount has no slot
    for, and the ledger holds it as one requirement. A reading is admitted exactly when
    every member holds a value in the set (`applies_one_of`), as a set bound on a slot
    reads, and the ledger judges the requirement met on exactly those readings: a reading
    of Female members, of Male members or of both applies Female and Male, and a member
    with no value on `prop`, or with a code outside the set, does not."""
    variable = "aid"
    pins = query_pins(
        variable,
        stray_values=[(prop, value) for value in drafted],
        quantity=quantity_requirement([variable]),
    )
    requirement = Requirement(
        term=ResolvedQualifier(
            text=" and ".join(drafted),
            named=[GraphRef(ref=value, label=value) for value in drafted],
            disposition="unbound",
        ),
        origin="stray",
        rule=ValueRule(values=frozenset((prop, value) for value in drafted)),
    )
    readings = [_flow_amount(value or "missing", prop, value) for value in held]
    alternate = dataclasses.replace(
        readings[0],
        sv_arc_facts={
            dcid: arcs for r in readings for dcid, arcs in (r.sv_arc_facts or {}).items()
        },
    )

    verdict = AlternateJudge(pins, _flow_amount("elected", prop, None)).verdict(alternate)

    assert (verdict.reason is None, meets_stray_values([requirement], alternate, [])) == (
        admitted,
        admitted,
    )
    if admitted:
        assert len(verdict.passing) == len(held)


def test_a_value_word_naming_two_properties_is_not_read_as_a_set():
    """Membership is the rule for a set the bind drafts on one missing slot. A word of the
    question naming a value on two properties (origin ``value``) is met by an answer every
    member of which holds one same pair, so members split between the two properties leave
    it unmet."""
    requirement = Requirement(
        term=ResolvedQualifier(text="female", named=[], disposition="no_vocabulary"),
        origin="value",
        rule=ValueRule(values=frozenset({("gender", "Female"), ("testSex", "Female")})),
    )
    by_gender, by_sex = (
        _flow_amount("gender", "gender", "Female"),
        _flow_amount("sex", "testSex", "Female"),
    )
    split = dataclasses.replace(
        by_gender, sv_arc_facts={**(by_gender.sv_arc_facts or {}), **(by_sex.sv_arc_facts or {})}
    )

    assert judge_value_requirements([requirement], split, []) == (requirement,)
    assert judge_value_requirements([requirement], by_gender, []) == ()


def test_the_guard_and_the_ledger_hold_a_reading_to_each_stray_property():
    """The bind drafted Female on `gender` and purpose 120 on `crsPurpose`, neither a slot
    of the elected amount: two requirements, one per property. A member holding both is
    admitted and meets both; a member holding one is pinned out on the other and leaves
    its requirement unmet."""
    strays = [("gender", "Female"), ("crsPurpose", "CRSPurposeCode120")]
    pins = query_pins("aid", stray_values=strays, quantity=quantity_requirement(["aid"]))
    requirements = [
        Requirement(
            term=ResolvedQualifier(
                text=value, named=[GraphRef(ref=value, label=value)], disposition="unbound"
            ),
            origin="stray",
            rule=ValueRule(values=frozenset({(prop, value)})),
        )
        for prop, value in strays
    ]
    female = _flow_amount("female", "gender", "Female")
    ((dcid, arcs),) = (female.sv_arc_facts or {}).items()
    both = dataclasses.replace(
        female,
        sv_arc_facts={
            dcid: {
                **arcs,
                "constraintProperties": {"nodes": [{"dcid": "gender"}, {"dcid": "crsPurpose"}]},
                "crsPurpose": {"nodes": [{"dcid": "CRSPurposeCode121"}]},
            }
        },
    )
    judge = AlternateJudge(pins, _flow_amount("elected", "gender", None))

    assert judge.verdict(both).reason is None
    assert meets_stray_values(requirements, both, [])
    assert _code(judge.verdict(female).reason or "") == "pin:crsPurpose"
    assert judge_value_requirements(requirements, female, []) == (requirements[1],)


def test_a_stray_value_is_held_by_values_alone_and_not_by_a_members_name():
    """A member named "Adult female literacy rate" whose arcs hold no value on `gender`
    does not apply a Female the bind drafted on `gender`: the guard pins it out, and the
    ledger leaves the stray unmet, since a stray value is judged by values alone. A word
    requirement for the same "female" is still met by the name."""
    alternate = _flow_amount("named", "gender", None)
    ((dcid, arcs),) = (alternate.sv_arc_facts or {}).items()
    named = dataclasses.replace(
        alternate,
        sv_arc_facts={dcid: {**arcs, "name": {"nodes": [{"value": "Adult female literacy rate"}]}}},
    )
    stray, word = (
        Requirement(
            term=ResolvedQualifier(text="Female", named=[], disposition="unbound"),
            origin=origin,
            rule=ValueRule(values=frozenset({("gender", "Female")})),
        )
        for origin in ("stray", "value")
    )
    pins = query_pins(
        "aid", stray_values=[("gender", "Female")], quantity=quantity_requirement(["aid"])
    )

    reason = AlternateJudge(pins, _flow_amount("elected", "gender", None)).verdict(named).reason

    assert reason is not None and _code(reason) == "pin:gender"
    assert judge_value_requirements([stray, word], named, []) == (stray,)


def _drafted_set(prop: str, values: tuple[str, ...]) -> list[Requirement]:
    """The ledger entries of `values` drafted together on `prop`, a slot the elected reading
    lacks: the set, and each value disclosed on its own (`_stray_requirements`)."""
    named = [GraphRef(ref=value, label=value) for value in values]
    return [
        Requirement(
            term=ResolvedQualifier(text=" and ".join(values), named=named, disposition="unbound"),
            origin="stray",
            rule=ValueRule(values=frozenset((prop, value) for value in values)),
            departure=unapplied_departure(
                " and ".join(values), asked=" and ".join(values), property_dcid=prop
            ),
        ),
        *(
            Requirement(
                term=ResolvedQualifier(text=value, named=[ref], disposition="unbound"),
                origin="set_value",
                rule=ValueRule(values=frozenset({(prop, value)})),
                departure=unapplied_departure(value, asked=value, property_dcid=prop),
            )
            for value, ref in zip(values, named, strict=True)
        ),
    ]


def test_an_answer_reading_each_value_of_a_drafted_set_discloses_none():
    """An answer whose members hold Female and Male, read without a binding on gender, meets
    the set and each of its values: a value of a drafted set is met when some selected
    member holds it, so nothing departs."""
    female, male = (
        _flow_amount("female", "gender", "Female"),
        _flow_amount("male", "gender", "Male"),
    )
    both = dataclasses.replace(
        female, sv_arc_facts={**(female.sv_arc_facts or {}), **(male.sv_arc_facts or {})}
    )

    judgement = judge(_drafted_set("gender", ("Female", "Male")), [(both, [])])

    assert judgement.outcomes == (("met", "met", "met"),)
    assert judgement.departures() == ()


def test_a_value_word_and_a_drafted_value_of_one_phrase_depart_once():
    """The request's word "male" (origin ``value``) and the drafted label "Male" are one
    phrase once case-folded, so an answer holding only Female departs it once."""
    word = Requirement(
        term=ResolvedQualifier(text="male", named=[], disposition="no_vocabulary"),
        origin="value",
        rule=ValueRule(values=frozenset({("gender", "Male")})),
        departure=unapplied_departure("male", asked="male", property_dcid="gender"),
    )
    female = _flow_amount("female", "gender", "Female")

    departed = judge(
        [*_drafted_set("gender", ("Female", "Male")), word], [(female, [])]
    ).departures()

    assert [fact.asked.casefold() for _warning, fact in departed] == ["male"]


def test_the_price_basis_vocabulary_cache_holds_no_measure_index():
    """A measure index a release swap replaces must be freeable, so the memo keys on the
    qualifier rows' text. A key holding the index adds a reference to it."""
    index = dataclasses.replace(get_measure_index(), nouns=())
    before = sys.getrefcount(index)

    vocabulary = _price_basis_vocabulary(index)

    assert vocabulary
    assert sys.getrefcount(index) == before
