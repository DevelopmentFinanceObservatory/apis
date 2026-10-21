"""Admissibility, of a part's readings before bind and of an alternate beside the elected
reading.

The interpretation table judges each reading of a corpus question under its frame
(`region.admissibility`) and asserts the refusal it names, with the pinned part it fails,
or that it is admitted. The alternate table pairs an elected shape with an alternate under a
query and asserts the reason code the alternate guard returns (`None` when the alternate is
admitted). Rows whose shapes come from another test file reproduce that file's fixture. A
row decided before bind by the default-reading collapse asserts the guard's own decision
for the same pair, which is the decision the guard takes when the pair reaches it.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Collection, Mapping, Sequence
from typing import Literal, NamedTuple, cast

import pytest

from code_index.shared import get_index, get_measure_index
from entity_resolution import Admission, EntityResolver
from search.engine.bind import DraftBindingEvidence, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import Binder
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.extract import DateRequest
from search.engine.interpretation import Interpretation, InterpretationSet
from search.engine.qualifier_match import QualifierMatch, QualifierReadings
from search.engine.quantity import quantity_requirement
from search.engine.quantity_read import QuantityOutput
from search.engine.reading_coverage import ObservedYears
from search.engine.region import alternate_guard, resolve_variable
from search.engine.region.admissibility import (
    judge_measure_and_roles,
    judge_pinned_parts,
    pinned_code_records,
)
from search.engine.region.alternate_guard import (
    AlternateJudge,
    QueryPins,
    _registry_twins,
    inadmissible_alternates,
    inadmissible_reason,
    offered_alternates,
    query_pins,
    twin_alternates,
)
from search.engine.region.candidates import alternate_representative
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.equivalent_series import CLUSTER_BY_MEMBER, EquivalentCluster
from search.engine.region.state import ResolveState
from search.engine.region.status_rule import CLOSED_AXES, OpenAxes
from search.engine.retrieval.bound_values import (
    DefaultReading,
    collapse_measure_families,
    index_member_rows,
    price_basis_order,
)
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import DefinitePart, Operation, Resolution, RoleAxis, TimeWindow
from tests.engine._binders import FixedBindReplyLLM, ReplyBinder, elected_reply
from tests.engine._builders import place_sets
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve, ref_dcid
from tests.engine._quantity import NO_QUANTITY, QuantityReadOnlyLLM
from tests.engine.test_count_units import _MORTALITY, _REPORTED_CASES
from tests.fixtures import FakeGraph, make_resolver

_GDP = "GrossDomesticProduction"
_GNI = "GrossNationalIncome"
_GVA = "GrossValueAdded"
_NOMINAL_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_IMF_GDP_SV = (
    "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal"
)


def _arcs(constraints: dict[str, str], names: dict[str, str] | None = None) -> dict:
    """Member arcs declaring `constraints`, each value node carrying its name in `names`."""
    arcs: dict = {"constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]}}
    for prop, value in constraints.items():
        node = {"dcid": value}
        if names and value in names:
            node["name"] = names[value]
        arcs[prop] = {"nodes": [node]}
    return arcs


def _shape(
    pop: str,
    prop: str,
    qual: str | None,
    denom: str | None,
    *,
    label: str = "",
    stat: str = "measuredValue",
    unresolved: bool = False,
    members: dict[str, dict] | tuple[str, ...] = (),
    roles: tuple[str, ...] | None = None,
    units: frozenset[str | None] | None = None,
    place_available: bool | None = None,
) -> ShapeDraft:
    """A shape. `members` maps member dcids to arcs, or lists dcids with no constraint arcs."""
    facts = {dcid: {} for dcid in members} if isinstance(members, tuple) else members
    return ShapeDraft(
        shape_id=label or f"{pop}/{prop}/{qual}/{denom}",
        label=label,
        pop_type_dcid=pop,
        meas_prop_dcid=prop,
        stat_type_dcid=stat,
        meas_qual_dcid=qual,
        meas_denom_dcid=denom,
        slot_keys=(),
        meas_denom_unresolved=unresolved,
        sv_arc_facts=facts or None,
        declared_roles=roles,
        place_units=units,
        place_available=place_available,
    )


def _binding(
    prop: str, values: list[str], evidence: DraftBindingEvidence = "mentioned"
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="what",
        property_dcid=prop,
        kind="value" if len(values) == 1 else "set",
        value_dcids=values,
        evidence=evidence,
    )


def _qualifier(dcid: str) -> QualifierMatch:
    return QualifierMatch(
        code_system="measurementQualifier",
        code_dcid=dcid,
        label=dcid,
        source="synonym",
        matched_text=dcid,
    )


def _code(reason: str | None) -> str | None:
    """The reason code a refusal opens with, such as `pin:normalisation`."""
    return None if reason is None else reason.partition(": ")[0]


# Shapes several rows share.


def _gdp(qual: str | None, denom: str | None = None, **kwargs) -> ShapeDraft:
    return _shape("EconomicActivity", "amount", qual, denom, **kwargs)


def _activity(dcid: str, activity: str, **more: str) -> dict[str, dict]:
    return {dcid: _arcs({"activitySource": activity, **more})}


_GDP_PLAIN = _gdp("Nominal", label="Nominal gross domestic product", members=("sv_gdp",))
_GDP_PER_CAPITA = _gdp(
    "Nominal", "PerCapita", label="Nominal gross domestic product per capita", members=("sv_pc",)
)
_POPULATION = _shape(
    "Person", "count", None, None, label="Total population", members=("Count_Person",)
)
_DENSITY = _shape(
    "Person", "count", None, "area", label="Population Density", members=("Count_Person_PerArea",)
)
_ELECTRICITY = _shape(
    "Electricity",
    "worldBank/EG_ELC_ACCS_ZS",
    None,
    None,
    label="Access to electricity (% of population)",
    members=("worldBank/EG_ELC_ACCS_ZS",),
)
_CRS_BIODIVERSITY = _shape(
    "FinancialFlow",
    "grantEquivalent",
    "Nominal",
    None,
    label="Official development assistance grant equivalents for biodiversity",
    members=(
        "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_CRSPurposeCode41030_Nominal",
    ),
    roles=("financingProvider", "financingRecipient"),
)
_SDG_CREDIT = _shape(
    "SDG_FC_ACC_SSID",
    "value",
    None,
    None,
    label="Proportion of small-scale industries with a loan or line of credit",
    members=("sdg/FC_ACC_SSID.ACTIVITY--ISIC4_C",),
    roles=("observationAbout",),
)
_UNDER5_SERIES = _shape(
    "Person",
    "",
    None,
    "Count_BirthEvent_LiveBirth",
    label="Mortality Rate of Children Younger Than 4 Years",
    members=("sv_u5",),
)
# The per-capita GDP and GNI pair of gdp-per-capita-vietnam-2023, with their member values.
_GDP_PC_UNRESOLVED = _gdp(
    "Nominal",
    unresolved=True,
    label="Nominal gross domestic product per capita",
    members=_activity("Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita", _GDP),
)
_GNI_PPP_PC = _gdp(
    "PurchasingPowerParity",
    unresolved=True,
    label="Gross National Income Based on Purchasing Power Parity Per Capita",
    members=_activity(
        "Amount_EconomicActivity_GrossNationalIncome_PurchasingPowerParity_PerCapita", _GNI
    ),
)
# The country-level GDP shapes of the Kenya and India menus.
_NOMINAL_GDP = _gdp(
    "Nominal", label="Nominal gross domestic product", members=_activity(_NOMINAL_GDP_SV, _GDP)
)
_NOMINAL_GDP_WITH_SECTORS = _gdp(
    "Nominal",
    label="Gross value added: agriculture",
    members={
        **{
            f"Amount_EconomicActivity_GrossValueAdded_{sector}_Nominal": _arcs(
                {"activitySource": _GVA, "economicSector": sector}
            )
            for sector in ("Agriculture", "Industry", "Services")
        },
        _NOMINAL_GDP_SV: _arcs({"activitySource": _GDP}),
    },
)
_INFLATION_ADJUSTED_GDP = _gdp(
    "InflationAdjusted",
    label="Inflation-adjusted gross domestic product",
    members=_activity("Amount_EconomicActivity_GrossDomesticProduction_InflationAdjusted", _GDP),
)
_REAL_GDP = _gdp(
    "RealValue",
    label="Real gross domestic product",
    members=_activity("Amount_EconomicActivity_GrossDomesticProduction_RealValue", _GDP),
)
_GNI_PPP = _gdp(
    "PurchasingPowerParity",
    label="Gross national income at purchasing power parity",
    members=_activity("Amount_EconomicActivity_GrossNationalIncome_PurchasingPowerParity", _GNI),
)
_GDP_PER_PERSON = _gdp(
    "Nominal",
    "Count_Person",
    label="Nominal GDP per person",
    members=_activity(
        "Amount_EconomicActivity_GrossDomesticProduction_Nominal_AsAFractionOf_Count_Person", _GDP
    ),
)
_IMF_GDP = _shape(
    "EconomicProduction",
    "valueAdded",
    "Nominal",
    None,
    label="Gross domestic product (GDP), Current prices",
    members=(_IMF_GDP_SV,),
)
_IMF_GDP_PER_CAPITA = _shape(
    "EconomicProduction",
    "valueAdded",
    "Nominal",
    "Count_Person",
    label="GDP, Current prices, Per capita",
    members=(f"{_IMF_GDP_SV}_AsAFractionOf_Count_Person",),
)
_GDP_GROWTH = _gdp(
    None,
    stat="growthRate",
    label="Gross domestic product growth rate",
    members=_activity("GrowthRate_Amount_EconomicActivity_GrossDomesticProduction", _GDP),
)
_GROWTH_REAL = _gdp(
    "RealValue",
    stat="growthRate",
    label="Growth rate amount of economic activity (real value)",
    members=("RealValueGrowthRateZeroObsGuard",),
)


class Row(NamedTuple):
    """One table row: the query, the elected and alternate shapes, the pins beyond the
    query's words, and the expected reason code. `open_axes` are the measure axes the status
    rule reads as open for the elected shape. `both_orders` also judges the pair with the
    roles swapped."""

    query: str
    elected: ShapeDraft
    alternate: ShapeDraft
    expected: str | None
    bindings: tuple[SlotBindingDraft, ...] = ()
    qualifiers: tuple[str, ...] = ()
    elected_has_data: bool = False
    both_orders: bool = False
    open_axes: OpenAxes = CLOSED_AXES

    def pins(self) -> QueryPins:
        return query_pins(
            self.query,
            quantity=quantity_requirement([self.query]),
            bindings=self.bindings,
            qualifier_matches=[_qualifier(dcid) for dcid in self.qualifiers],
            elected_has_data=self.elected_has_data,
            open_axes=self.open_axes,
        )


_GDP_MENTIONED = (_binding("activitySource", [_GDP]),)
_GDP_SOLE_OPTION = (_binding("activitySource", [_GDP], "sole_option"),)

ROWS = [
    pytest.param(
        Row(
            "maternal mortality ratio",
            _shape("SDG_SH_STA_MORT", "value", None, None, members=("sdg/SH_STA_MORT.SEX--F",)),
            _shape(
                "MaternalMortalityEvent",
                "count",
                None,
                None,
                members=("Count_MaternalMortalityEvent",),
            ),
            "pin:normalisation",
        ),
        id="maternal-mortality-ratio-refuses-the-event-count",
    ),
    pytest.param(
        Row(
            "HIV incidence rate",
            _shape("SDG_SH_HIV_INCD", "value", None, None, members=("sdg/SH_HIV_INCD",)),
            _shape(
                "MedicalConditionIncident",
                "count",
                None,
                None,
                members=("Count_MedicalConditionIncident_ConditionHIVAIDS",),
            ),
            "pin:normalisation",
        ),
        id="hiv-incidence-rate-refuses-the-event-count",
    ),
    pytest.param(
        Row(
            "GDP per capita",
            _gdp("Nominal", unresolved=True, members=("sv_pc",)),
            _GDP_PLAIN,
            "pin:normalisation",
        ),
        id="gdp-per-capita-unresolved-refuses-plain-gdp",
    ),
    pytest.param(
        Row(
            "greenhouse gas emissions",
            _shape(
                "Emissions", "amount", "Annual", None, members=("Annual_Emissions_GreenhouseGas",)
            ),
            _shape(
                "Emissions",
                "amount",
                None,
                None,
                unresolved=True,
                members=("Amount_Emissions_CarbonDioxide_PerCapita",),
            ),
            "specialisation:normalisation",
        ),
        id="greenhouse-gases-refuse-unresolved-per-capita-co2",
    ),
    pytest.param(
        Row(
            "access to electricity",
            _ELECTRICITY,
            _shape(
                "Electricity",
                "worldBank/EG_ELC_ACCS_RU_ZS",
                None,
                None,
                label="Access to electricity, rural (% of rural population)",
                members=("worldBank/EG_ELC_ACCS_RU_ZS",),
            ),
            "specialisation:residence",
        ),
        id="electricity-refuses-the-rural-series",
    ),
    pytest.param(
        Row(
            "access to electricity",
            _ELECTRICITY,
            _shape(
                "Electricity",
                "worldBank/EG_ELC_ACCS_UR_ZS",
                None,
                None,
                label="Access to electricity, urban (% of urban population)",
                members=("worldBank/EG_ELC_ACCS_UR_ZS",),
            ),
            "specialisation:residence",
        ),
        id="electricity-refuses-the-urban-series",
    ),
    pytest.param(
        Row(
            "access to electricity",
            _ELECTRICITY,
            _shape(
                "SDG_EG_ACS_ELEC",
                "value",
                "SDG_ReportingStandard",
                None,
                unresolved=True,
                members=(
                    "sdg/EG_ACS_ELEC",
                    "sdg/EG_ACS_ELEC.URBANISATION--R",
                    "sdg/EG_ACS_ELEC.URBANISATION--U",
                ),
            ),
            None,
        ),
        id="electricity-admits-the-sdg-series-and-its-urban-rural-members",
    ),
    pytest.param(
        Row(
            "biodiversity aid to Zambia since 2018 in grant equivalents",
            _CRS_BIODIVERSITY,
            _shape(
                "SDG_DC_ODA_BDVDL",
                "value",
                None,
                None,
                members=("sdg/DC_ODA_BDVDL",),
                roles=("observationAbout",),
            ),
            "cross_population",
        ),
        id="biodiversity-aid-refuses-the-sdg-oda-series",
    ),
    pytest.param(
        Row("grant financing for Mozambique", _CRS_BIODIVERSITY, _SDG_CREDIT, "cross_population"),
        id="grant-financing-refuses-the-sdg-credit-series",
    ),
    pytest.param(
        Row(
            "concessional loans to Sri Lanka",
            _shape(
                "FinancialObligation",
                "amount",
                "Nominal",
                None,
                members=("Amount_FinancialObligation_Commitment_CRSPurposeCode15112_Nominal",),
                roles=("financingProvider", "financingRecipient"),
            ),
            _SDG_CREDIT,
            "cross_population",
        ),
        id="concessional-loans-refuse-the-sdg-credit-series",
    ),
    pytest.param(
        Row(
            "biodiversity aid to Zambia since 2018 in grant equivalents",
            _CRS_BIODIVERSITY,
            _shape(
                "FinancialFlow",
                "amount",
                None,
                None,
                members=("Amount_FinancialFlow_Net_Disbursement_CRSPurposeCode41030_Nominal",),
                roles=("financingProvider", "financingRecipient"),
            ),
            None,
        ),
        id="biodiversity-aid-admits-the-crs-amount-flow",
    ),
    pytest.param(
        Row(
            "access to electricity",
            _ELECTRICITY,
            _shape(
                "undata/sdg/SERIES-EG_ACS_ELEC",
                "value",
                None,
                None,
                members=("undata/sdg/EG_ACS_ELEC",),
                roles=("unit",),
            ),
            None,
        ),
        id="electricity-admits-the-sdg-series-with-a-unit-role",
    ),
    pytest.param(
        Row("GDP", _GDP_PLAIN, _GDP_PER_CAPITA, "specialisation:normalisation"),
        id="gdp-refuses-per-capita",
    ),
    pytest.param(
        Row("GDP per head", _GDP_PLAIN, _GDP_PER_CAPITA, None), id="gdp-per-head-admits-per-capita"
    ),
    pytest.param(
        Row("GDP per year", _GDP_PLAIN, _GDP_PER_CAPITA, "specialisation:normalisation"),
        id="gdp-per-year-refuses-per-capita",
    ),
    pytest.param(
        Row("GDP", _GDP_PER_CAPITA, _GDP_PLAIN, None), id="per-capita-elected-admits-plain-gdp"
    ),
    pytest.param(
        Row(
            "GDP",
            _GDP_PLAIN,
            _gdp("Nominal", "Count_Person", members=("sv_fraction",)),
            "specialisation:normalisation",
        ),
        id="gdp-refuses-a-fraction-of-persons",
    ),
    pytest.param(
        Row("population of Kenya", _POPULATION, _DENSITY, "specialisation:normalisation"),
        id="population-refuses-density",
    ),
    pytest.param(
        Row("population density of Kenya", _POPULATION, _DENSITY, None),
        id="population-density-admits-density",
    ),
    pytest.param(
        Row("population per year", _POPULATION, _DENSITY, "specialisation:normalisation"),
        id="population-per-year-refuses-density",
    ),
    pytest.param(
        Row("population density", _DENSITY, _POPULATION, "pin:normalisation"),
        id="density-elected-refuses-total-population",
    ),
    pytest.param(
        Row("GDP", _GDP_PLAIN, _gdp("Nominal", "area"), "specialisation:normalisation"),
        id="gdp-refuses-per-area",
    ),
    pytest.param(
        Row("population", _shape("Person", "count", None, None, unresolved=True), _DENSITY, None),
        id="unresolved-population-admits-density",
    ),
    pytest.param(
        Row(
            "births per capita",
            _shape("BirthEvent", "count", None, "PerCapita"),
            _shape("BirthEvent", "count", None, "area"),
            "pin:normalisation",
        ),
        id="births-per-capita-refuses-per-area",
    ),
    pytest.param(
        Row(
            "births",
            _shape("BirthEvent", "count", None, None),
            _shape("BirthEvent", "count", None, "Count_Person"),
            "specialisation:normalisation",
        ),
        id="births-refuse-a-per-person-fraction",
    ),
    pytest.param(
        Row("measles incidence rate", _MORTALITY, _REPORTED_CASES, "pin:normalisation"),
        id="measles-incidence-rate-refuses-reported-cases",
    ),
    pytest.param(
        Row(
            "compare forest cover in Kenya and Tanzania",
            _shape(
                "SDG_AG_LND_FRST",
                "value",
                None,
                None,
                label="Proportion of forest area",
                members=("sv_a",),
            ),
            _shape(
                "SDG_AG_LND_FRSTN",
                "value",
                None,
                None,
                label="Forest area (national)",
                members=("sv_b",),
            ),
            None,
            both_orders=True,
        ),
        id="forest-proportion-and-national-area-admit-each-other",
    ),
    pytest.param(
        Row(
            "share of renewable energy in total energy",
            _shape("SDG_EG_FEC_RNEW", "value", None, None, members=("sv_a",)),
            _shape(
                "Energy", "worldBank/4_1_SHARE_RE_IN_ELECTRICITY", None, None, members=("sv_b",)
            ),
            None,
            both_orders=True,
        ),
        id="renewable-energy-share-series-admit-each-other",
    ),
    pytest.param(
        Row(
            "poverty in Bangladesh",
            _shape(
                "SDG_SI_POV_DAY1",
                "value",
                None,
                None,
                label="Proportion of population below international poverty line",
                members=("sdg/SI_POV_DAY1",),
            ),
            _shape(
                "SDG_SI_POV_NAHC",
                "value",
                None,
                None,
                label="Proportion of population living below the national poverty line",
                members={
                    "sdg/SI_POV_NAHC": _arcs({}),
                    "sdg/SI_POV_NAHC.URBANISATION--U": _arcs(
                        {"sdg_urbanisation": "SDG_UrbanisationEnum_U"},
                        {"SDG_UrbanisationEnum_U": "Urban"},
                    ),
                },
            ),
            None,
            both_orders=True,
        ),
        id="poverty-line-series-admit-each-other",
    ),
    pytest.param(
        Row(
            "government education spending Kenya",
            _gdp(None, "Amount_EconomicActivity_ExpenditureActivity_Government", members=("sv_a",)),
            _gdp(
                None, "Amount_EconomicActivity_GrossDomesticProduction_Nominal", members=("sv_b",)
            ),
            None,
            both_orders=True,
        ),
        id="government-spending-and-gdp-denominators-admit-each-other",
    ),
    pytest.param(
        Row(
            "GDP India",
            _GDP_PLAIN,
            _gdp("Nominal", unresolved=True, members=("sv_pc",)),
            "specialisation:normalisation",
        ),
        id="gdp-against-gdp-per-capita-unresolved-refused",
    ),
    pytest.param(
        Row("GDP India", _gdp("Nominal", unresolved=True, members=("sv_pc",)), _GDP_PLAIN, None),
        id="gdp-per-capita-unresolved-elected-admits-plain-gdp",
    ),
    pytest.param(
        Row(
            "birth rate Ethiopia",
            _shape(
                "BirthEvent",
                "count",
                None,
                "Count_Person",
                label="Crude Birth Rate",
                members=("sv_a",),
            ),
            _shape(
                "Person",
                "fertilityRate",
                None,
                None,
                label="Fertility rate of women",
                members=("sv_b",),
            ),
            None,
            both_orders=True,
        ),
        id="birth-rate-and-fertility-rate-admit-each-other",
    ),
    pytest.param(
        Row(
            "under-5 child mortality rate",
            _UNDER5_SERIES,
            _shape(
                "MortalityEvent",
                "count",
                None,
                "Count_BirthEvent_LiveBirth",
                label="Infant Mortality Rate",
                members=("sv_b",),
            ),
            None,
            both_orders=True,
        ),
        id="under-5-rate-and-infant-mortality-rate-admit-each-other",
    ),
    pytest.param(
        Row(
            "under-5 child mortality rate",
            _UNDER5_SERIES,
            _shape("SDG_SH_DYN_MORT", "value", None, None, members=("sv_b",)),
            None,
            both_orders=True,
        ),
        id="under-5-rate-and-sdg-under-5-mortality-admit-each-other",
    ),
    pytest.param(
        Row(
            "under-5 child mortality rate",
            _UNDER5_SERIES,
            _shape(
                "Person",
                "who/CM_01",
                None,
                None,
                label="Number of Under-Five Deaths",
                members=("sv_b",),
            ),
            None,
            both_orders=True,
        ),
        id="under-5-rate-and-under-five-deaths-admit-each-other",
    ),
    pytest.param(
        Row(
            "income in California",
            _shape("Person", "income", None, None, stat="medianValue", members=("sv_a",)),
            _shape("Household", "income", None, None, stat="medianValue", members=("sv_b",)),
            None,
            both_orders=True,
        ),
        id="person-and-household-income-admit-each-other",
    ),
    pytest.param(
        Row(
            "child mortality rate Ethiopia",
            _shape(
                "MortalityEvent", "count", None, "Count_BirthEvent_LiveBirth", members=("sv_a",)
            ),
            _UNDER5_SERIES,
            None,
            both_orders=True,
        ),
        id="child-mortality-rate-and-under-5-series-admit-each-other",
    ),
    pytest.param(
        Row(
            "most recent data on internet access",
            _shape(
                "Person",
                "count",
                None,
                None,
                unresolved=True,
                label="Percent of Internet Users",
                members=("sv_a",),
            ),
            _shape("SDG_IT_USE_ii99", "value", None, None, members=("sv_b",)),
            None,
            both_orders=True,
        ),
        id="internet-access-unresolved-percent-and-sdg-admit-each-other",
    ),
    pytest.param(
        Row(
            "percentage of individuals using the internet",
            _shape(
                "Person",
                "count",
                None,
                "PerCapita",
                label="Percent of Internet Users",
                members=("Count_Person_IsInternetUser_PerCapita",),
            ),
            _shape("SDG_IT_USE_ii99", "value", None, None, members=("sv_b",)),
            None,
            both_orders=True,
        ),
        id="internet-use-percentage-and-sdg-admit-each-other",
    ),
    pytest.param(
        Row(
            "inward remittances to Kenya",
            _shape(
                "Remittance",
                "amount",
                None,
                "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
                members=("sv_b",),
            ),
            _shape("Remittance", "amount", None, None, members=("sv_a",)),
            None,
        ),
        id="remittances-per-gdp-admits-the-level",
    ),
    pytest.param(
        Row(
            "interest payments on external debt as a share of GNI",
            _shape("FinancialFlow", "amount", "Nominal", None, members=("sv_a",)),
            _shape(
                "FinancialFlow",
                "amount",
                None,
                "Amount_EconomicActivity_GrossNationalIncome_Nominal",
                members=("sv_b",),
            ),
            None,
        ),
        id="interest-payments-admit-the-share-of-gni-the-query-names",
    ),
    pytest.param(
        Row(
            "greenhouse gas emissions in Nigeria",
            _shape("Emissions", "amount", "Annual", None, members=("sv_a",)),
            _shape("SDG_EN_ATM_CO2", "value", None, None, members=("sv_b",)),
            None,
            both_orders=True,
        ),
        id="annual-emissions-and-sdg-co2-admit-each-other",
    ),
    pytest.param(
        Row("GDP per capita", _GDP_PC_UNRESOLVED, _GNI_PPP_PC, "concept:activitySource"),
        id="gdp-per-capita-refuses-gni-ppp-per-capita",
    ),
    pytest.param(
        Row(
            "GDP per capita",
            _GDP_PC_UNRESOLVED,
            _GNI_PPP_PC,
            "concept:activitySource",
            _GDP_MENTIONED,
        ),
        id="gdp-per-capita-with-gdp-mentioned-refuses-gni-ppp-per-capita",
    ),
    pytest.param(
        Row(
            "GNI per capita PPP",
            _GNI_PPP_PC,
            _GDP_PC_UNRESOLVED,
            "concept:activitySource",
            (_binding("activitySource", [_GNI]),),
        ),
        id="gni-per-capita-ppp-refuses-gdp-per-capita",
    ),
    pytest.param(
        Row(
            "GDP per capita",
            _GDP_PC_UNRESOLVED,
            _gdp(
                "PurchasingPowerParity",
                unresolved=True,
                members=_activity("Amount_GDP_Manufacturing_PerCapita", _GDP, naics="NAICS/31-33"),
            ),
            "specialisation:naics",
            _GDP_MENTIONED,
        ),
        id="gdp-mentioned-refuses-a-naics-manufacturing-reading",
    ),
    pytest.param(
        Row(
            "GDP per capita",
            _GDP_PC_UNRESOLVED,
            _gdp(
                "PurchasingPowerParity",
                unresolved=True,
                members={
                    **_activity("Amount_GNI_PPP_PerCapita", _GNI),
                    "Amount_EconomicActivity_PurchasingPowerParity_PerCapita": _arcs({}),
                },
            ),
            None,
            _GDP_MENTIONED,
        ),
        id="gdp-mentioned-admits-ppp-reading-with-a-member-lacking-activity",
    ),
    pytest.param(
        Row(
            "health aid",
            _gdp(
                "Nominal",
                unresolved=True,
                members={"Amount_Aid_Health": _arcs({"crsPurposeCode": "CRSPurposeCode120"})},
            ),
            _gdp(
                "Nominal",
                unresolved=True,
                members={"Amount_Aid_BasicHealth": _arcs({"crsPurposeCode": "CRSPurposeCode122"})},
            ),
            None,
            (_binding("crsPurposeCode", ["CRSPurposeCode120"]),),
        ),
        id="health-aid-admits-basic-health-purpose",
    ),
    pytest.param(
        Row(
            "GDP per capita of Vietnam",
            _GDP_PC_UNRESOLVED,
            _GNI_PPP_PC,
            "concept:activitySource",
            _GDP_MENTIONED,
        ),
        id="gdp-per-capita-of-vietnam-mentioned-refuses-gni-ppp",
    ),
    pytest.param(
        Row(
            "GDP per capita of Vietnam",
            _GDP_PC_UNRESOLVED,
            _GNI_PPP_PC,
            "concept:activitySource",
            _GDP_SOLE_OPTION,
        ),
        id="gdp-per-capita-of-vietnam-sole-option-refuses-gni-ppp",
    ),
    pytest.param(
        Row(
            "under-five mortality",
            _gdp(
                "Nominal",
                members={
                    "MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth": _arcs(
                        {"age": "YearsUpto4"}
                    )
                },
            ),
            _gdp(
                "Nominal",
                members={
                    "undata/sdg/SH_DYN_MORT.AGE--Y0T4": _arcs({"undataAge": "undata/AGE-Y0T4"})
                },
            ),
            None,
            (_binding("age", ["YearsUpto4"]),),
        ),
        id="under-five-mortality-admits-the-sdg-age-twin",
    ),
    pytest.param(
        Row("GDP in Kenya", _NOMINAL_GDP, _GNI_PPP, "concept:activitySource", _GDP_MENTIONED),
        id="gdp-mentioned-refuses-gni-ppp",
    ),
    pytest.param(
        Row(
            "GDP in Kenya",
            _NOMINAL_GDP,
            _REAL_GDP,
            "default:qualifier",
            _GDP_MENTIONED,
            elected_has_data=True,
        ),
        id="nominal-with-data-refuses-real-gdp-by-default",
    ),
    pytest.param(
        Row("GDP in Kenya", _NOMINAL_GDP, _GNI_PPP, "concept:activitySource", _GDP_SOLE_OPTION),
        id="gdp-sole-option-refuses-gni-ppp",
    ),
    pytest.param(
        Row(
            "GDP growth in Ghana",
            _GROWTH_REAL,
            _gdp(
                None,
                stat="growthRate",
                label="Rural gross domestic product growth rate",
                members=("RuralBareGrowthRateZeroObsGuard",),
            ),
            "specialisation:residence",
        ),
        id="real-growth-refuses-rural-growth",
    ),
    pytest.param(
        Row(
            "GDP growth in Ghana",
            _GROWTH_REAL,
            _gdp(
                None,
                stat="growthRate",
                label="Gross domestic product growth rate",
                members=("sv_b",),
            ),
            None,
        ),
        id="real-growth-admits-bare-growth",
    ),
    pytest.param(
        Row(
            "GDP in Kenya",
            _INFLATION_ADJUSTED_GDP,
            _GNI_PPP,
            "concept:activitySource",
            _GDP_MENTIONED,
        ),
        id="inflation-adjusted-gdp-refuses-gni-ppp",
    ),
    pytest.param(
        Row("GDP in Kenya", _INFLATION_ADJUSTED_GDP, _NOMINAL_GDP, None, _GDP_MENTIONED),
        id="inflation-adjusted-gdp-admits-nominal-gdp",
    ),
    pytest.param(
        Row(
            "gross national income of Ghana",
            _gdp(
                "Nominal",
                label="Gross national income (current US$)",
                members=_activity("GNI_IDS", _GNI),
                roles=("financingRecipient",),
            ),
            _gdp(
                "PurchasingPowerParity",
                members=_activity("GNI_PPP", _GNI),
                roles=("observationAbout",),
            ),
            None,
            elected_has_data=True,
            both_orders=True,
        ),
        id="gni-with-data-admits-ppp-in-both-orders",
    ),
    pytest.param(
        Row(
            "gross national income of Ghana",
            _gdp("Nominal", members=_activity("GNI_IDS", _GNI), roles=("financingRecipient",)),
            _gdp("PurchasingPowerParity", members=_activity("GNI_PPP", _GNI)),
            None,
        ),
        id="gni-with-recipient-role-admits-ppp",
    ),
    pytest.param(
        Row(
            "GDP of Kenya",
            _NOMINAL_GDP_WITH_SECTORS,
            _IMF_GDP_PER_CAPITA,
            "specialisation:normalisation",
            _GDP_MENTIONED,
        ),
        id="gdp-with-sectors-refuses-imf-per-capita",
    ),
    pytest.param(
        Row(
            "GDP of Kenya",
            _NOMINAL_GDP_WITH_SECTORS,
            _GDP_PER_PERSON,
            "specialisation:normalisation",
            _GDP_MENTIONED,
        ),
        id="gdp-with-sectors-refuses-gdp-per-person",
    ),
    pytest.param(
        Row(
            "GDP of Kenya",
            _NOMINAL_GDP,
            _gdp(
                "RealValue",
                label="Real gross domestic product by economic activity",
                members={
                    "RealValue_GVA_Agriculture": _arcs(
                        {"activitySource": _GVA, "economicSector": "Agriculture"}
                    ),
                    "RealValue_GDP": _arcs({"activitySource": _GDP}),
                },
            ),
            "default:qualifier",
            _GDP_MENTIONED,
            elected_has_data=True,
        ),
        id="nominal-with-data-refuses-real-gdp-by-activity",
    ),
    pytest.param(
        Row(
            "GDP of Kenya",
            _NOMINAL_GDP,
            _INFLATION_ADJUSTED_GDP,
            "default:qualifier",
            _GDP_MENTIONED,
            elected_has_data=True,
        ),
        id="nominal-with-data-refuses-inflation-adjusted-gdp",
    ),
    pytest.param(
        Row("GDP of Kenya", _NOMINAL_GDP, _GDP_GROWTH, "concept:statType", _GDP_MENTIONED),
        id="gdp-refuses-a-growth-rate",
    ),
    pytest.param(
        Row("GDP growth of Kenya", _GDP_GROWTH, _NOMINAL_GDP, "concept:statType", _GDP_MENTIONED),
        id="a-growth-rate-on-a-closed-stat-type-refuses-the-level",
    ),
    pytest.param(
        Row(
            "GDP of Kenya",
            _GDP_GROWTH,
            _NOMINAL_GDP,
            None,
            _GDP_MENTIONED,
            open_axes=OpenAxes(stat_type=True, denominator=False, price_basis=True),
        ),
        id="a-growth-rate-on-an-open-stat-type-admits-the-level",
    ),
    pytest.param(
        Row(
            "GDP of Kenya",
            _REAL_GDP,
            _INFLATION_ADJUSTED_GDP,
            "default:qualifier",
            _GDP_MENTIONED,
            elected_has_data=True,
        ),
        id="real-with-data-on-a-closed-price-basis-refuses-inflation-adjusted-gdp",
    ),
    pytest.param(
        Row(
            "GDP of Kenya",
            _REAL_GDP,
            _INFLATION_ADJUSTED_GDP,
            None,
            _GDP_MENTIONED,
            elected_has_data=True,
            open_axes=OpenAxes(stat_type=False, denominator=False, price_basis=True),
        ),
        id="real-with-data-on-an-open-price-basis-admits-inflation-adjusted-gdp",
    ),
    pytest.param(
        Row("GDP India", _INFLATION_ADJUSTED_GDP, _IMF_GDP, None, _GDP_SOLE_OPTION),
        id="inflation-adjusted-gdp-admits-imf-gdp",
    ),
    pytest.param(
        Row(
            "GDP India",
            _INFLATION_ADJUSTED_GDP,
            _GDP_PER_PERSON,
            "specialisation:normalisation",
            _GDP_SOLE_OPTION,
        ),
        id="inflation-adjusted-gdp-refuses-gdp-per-person",
    ),
    pytest.param(
        Row(
            "GDP India",
            _NOMINAL_GDP,
            _gdp(
                "RealValue",
                members={
                    "RealValue_GVA_Agriculture": _arcs(
                        {"activitySource": _GVA, "economicSector": "Agriculture"}
                    ),
                    "RealValue_GDP_Agriculture": _arcs(
                        {"activitySource": _GDP, "economicSector": "Agriculture"}
                    ),
                    "RealValue_Unconstrained": _arcs({}),
                },
            ),
            "default:qualifier",
            elected_has_data=True,
        ),
        id="nominal-with-data-refuses-real-gdp-with-unconstrained-member",
    ),
    pytest.param(
        Row(
            "GDP India",
            _NOMINAL_GDP,
            _IMF_GDP,
            None,
            _GDP_SOLE_OPTION,
            elected_has_data=True,
            both_orders=True,
        ),
        id="nominal-with-data-admits-imf-twin-in-both-orders",
    ),
    pytest.param(
        Row("GDP India", _IMF_GDP, _IMF_GDP_PER_CAPITA, "specialisation:normalisation"),
        id="imf-gdp-refuses-imf-per-capita",
    ),
    pytest.param(
        Row(
            "GDP of Brazil",
            _gdp("Nominal", members=("Amount_EconomicActivity_GrossDomesticProduction_Nominal",)),
            _gdp(
                "Nominal",
                unresolved=True,
                members=("Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita",),
            ),
            "specialisation:normalisation",
        ),
        id="gdp-of-brazil-refuses-unresolved-per-capita",
    ),
    pytest.param(
        Row(
            "GNI per capita PPP in Ethiopia",
            _GNI_PPP_PC,
            _GDP_PC_UNRESOLVED,
            "concept:activitySource",
            (_binding("activitySource", [_GNI]),),
        ),
        id="gni-per-capita-ppp-in-ethiopia-refuses-gdp-per-capita",
    ),
    pytest.param(
        Row(
            "GNI per capita PPP in Ethiopia",
            _GNI_PPP_PC,
            _GDP_PC_UNRESOLVED,
            "pin:qualifier",
            (_binding("activitySource", [_GNI]),),
            qualifiers=("PurchasingPowerParity",),
        ),
        id="gni-per-capita-ppp-in-ethiopia-with-ppp-named-pins-qualifier",
    ),
    pytest.param(
        Row(
            "GDP PPP of Kenya",
            _gdp("PurchasingPowerParity", members=_activity("Amount_GDP_PPP", _GDP)),
            _NOMINAL_GDP,
            "pin:qualifier",
            _GDP_MENTIONED,
            qualifiers=("PurchasingPowerParity",),
        ),
        id="gdp-ppp-named-refuses-nominal",
    ),
    pytest.param(
        Row(
            "real GDP of Zambia",
            _REAL_GDP,
            _NOMINAL_GDP,
            "pin:qualifier",
            _GDP_MENTIONED,
            qualifiers=("RealValue",),
        ),
        id="real-gdp-named-refuses-nominal",
    ),
    pytest.param(
        Row(
            "real GDP of Zambia",
            _REAL_GDP,
            _INFLATION_ADJUSTED_GDP,
            None,
            _GDP_MENTIONED,
            qualifiers=("RealValue",),
            elected_has_data=True,
        ),
        id="real-gdp-named-admits-inflation-adjusted",
    ),
    pytest.param(
        Row("GDP of Kenya", _NOMINAL_GDP, _REAL_GDP, "default:qualifier", elected_has_data=True),
        id="nominal-with-data-refuses-real-gdp",
    ),
    pytest.param(
        Row("GDP of Kenya", _NOMINAL_GDP, _REAL_GDP, None),
        id="nominal-without-data-admits-real-gdp",
    ),
    pytest.param(
        Row(
            "greenhouse gas emissions",
            _shape(
                "Emissions",
                "amount",
                None,
                None,
                unresolved=True,
                members=("Amount_Emissions_CarbonDioxide_PerCapita",),
            ),
            _shape(
                "Emissions", "amount", "Annual", None, members=("Annual_Emissions_GreenhouseGas",)
            ),
            None,
        ),
        id="greenhouse-gases-admit-unresolved-co2-without-arcs",
    ),
    pytest.param(
        Row(
            "greenhouse gas emissions",
            _shape(
                "Emissions",
                "amount",
                None,
                None,
                unresolved=True,
                members={
                    "Amount_Emissions_CarbonDioxide_PerCapita": _arcs(
                        {"emittedThing": "CarbonDioxide"}
                    )
                },
            ),
            _shape(
                "Emissions",
                "amount",
                "Annual",
                None,
                members={
                    "Annual_Emissions_GreenhouseGas": _arcs({"emittedThing": "GreenhouseGas"})
                },
            ),
            "concept:emittedThing",
        ),
        id="greenhouse-gases-refuse-co2-by-emitted-thing",
    ),
    pytest.param(
        Row(
            "births in Ethiopia",
            _shape("BirthEvent", "count", None, None, members=("Count_BirthEvent",)),
            _shape(
                "BirthEvent",
                "count",
                None,
                "Count_Person",
                members={
                    "Count_BirthEvent_LiveBirth_AsFractionOf_Count_Person": _arcs(
                        {"medicalStatus": "LiveBirth"}
                    )
                },
            ),
            "specialisation:normalisation",
        ),
        id="births-refuse-live-birth-fraction",
    ),
    pytest.param(
        Row(
            "access to electricity",
            _shape(
                "Electricity",
                "worldBank/EG_ELC_ACCS_RU_ZS",
                None,
                None,
                label="Access to electricity, rural (% of rural population)",
                members=("worldBank/EG_ELC_ACCS_RU_ZS",),
            ),
            _shape(
                "Electricity",
                "worldBank/EG_ELC_ACCS_UR_ZS",
                None,
                None,
                label="Access to electricity, urban (% of urban population)",
                members=("worldBank/EG_ELC_ACCS_UR_ZS",),
            ),
            "concept:residence",
        ),
        id="rural-electricity-refuses-urban-series",
    ),
    pytest.param(
        Row(
            "births",
            _shape("BirthEvent", "count", None, "PerCapita"),
            _shape("BirthEvent", "count", None, "area"),
            None,
        ),
        id="births-per-capita-admits-per-area-unpinned",
    ),
    # An area noun that names a place frames no density, so the pin stays off.
    pytest.param(
        Row(
            "people living in rural areas in Kenya",
            _POPULATION,
            _shape(
                "SDG_SP_RUR_POPN",
                "value",
                None,
                None,
                label="Proportion of population living in rural areas",
                members=("sdg/SP_RUR_POPN",),
            ),
            None,
        ),
        id="rural-areas-share",
    ),
    pytest.param(
        Row(
            "number of people in protected areas",
            _POPULATION,
            _shape(
                "SDG_ER_PTD_POPN",
                "value",
                None,
                None,
                label="Population living in protected areas",
                members=("sdg/ER_PTD_POPN",),
            ),
            None,
        ),
        id="protected-areas",
    ),
    pytest.param(
        Row("population density of Kenya", _DENSITY, _POPULATION, "pin:normalisation"),
        id="density-pins-a-total",
    ),
    # A numbered per-head base is a rate per person, which a series with no stated
    # denominator may hold.
    pytest.param(
        Row(
            "homicides per 100,000 people in Kenya",
            _shape(
                "MortalityEvent",
                "count",
                None,
                "Count_Person",
                label="Intentional homicides per capita",
                members=("Count_Death_IntentionalHomicide_AsFractionOf_Count_Person",),
            ),
            _shape(
                "SDG_VC_IHR_PSRC",
                "value",
                None,
                None,
                label="Number of victims of intentional homicide per 100,000 population",
                members=("sdg/VC_IHR_PSRC",),
            ),
            None,
        ),
        id="homicides-per-100000-people",
    ),
]


@pytest.mark.parametrize("row", ROWS)
def test_row(row: Row) -> None:
    pins = row.pins()
    assert _code(inadmissible_reason(pins, row.elected, row.alternate)) == row.expected
    if row.both_orders:
        assert _code(inadmissible_reason(pins, row.alternate, row.elected)) == row.expected


def test_a_ppp_reading_of_the_elected_measure_gives_way_once_the_election_has_data():
    """GDP at purchasing power parity holds the elected activity, so it stands before the
    elected shape's data is known and ranks below the nominal election once that has data."""
    gdp_ppp = _gdp(
        "PurchasingPowerParity",
        unresolved=True,
        members=_activity("Amount_GDP_PPP_PerCapita", _GDP),
    )
    before = query_pins(
        "GDP per capita", quantity=quantity_requirement(["GDP per capita"]), bindings=_GDP_MENTIONED
    )
    after = query_pins(
        "GDP per capita",
        quantity=quantity_requirement(["GDP per capita"]),
        bindings=_GDP_MENTIONED,
        elected_has_data=True,
    )

    assert inadmissible_reason(before, _GDP_PC_UNRESOLVED, gdp_ppp) is None
    assert _code(inadmissible_reason(after, _GDP_PC_UNRESOLVED, gdp_ppp)) == "default:qualifier"


def test_no_default_when_a_price_basis_phrase_went_unread():
    gdp_ppp = _gdp(
        "PurchasingPowerParity",
        unresolved=True,
        members=_activity("Amount_GDP_PPP_PerCapita", _GDP),
    )
    pins = query_pins(
        "GDP per capita",
        quantity=quantity_requirement(["GDP per capita"]),
        bindings=_GDP_MENTIONED,
        elected_has_data=True,
        unread_price_basis=True,
    )

    assert inadmissible_reason(pins, _GDP_PC_UNRESOLVED, gdp_ppp) is None


def test_a_member_passes_on_the_elected_values_or_a_named_property():
    """The GNI member fails beside an elected GDP reading, passes beside an elected GNI
    reading, and passes beside a reading that holds no activity when the query names it."""
    gni_member = list(_GNI_PPP.sv_arc_facts or {})
    no_activity = _gdp("Nominal", members={"sv_total": _arcs({"economicSector": "Agriculture"})})
    named = query_pins(
        "GDP in Kenya",
        quantity=quantity_requirement(["GDP in Kenya"]),
        bindings=[_binding("activitySource", [_GDP])],
    )

    def offered(pins: QueryPins, elected: ShapeDraft) -> list[str]:
        return list(offered_alternates(pins, elected, {1: _GNI_PPP})[1].sv_arc_facts or {})

    assert (
        offered(
            query_pins("GDP in Kenya", quantity=quantity_requirement(["GDP in Kenya"])),
            _NOMINAL_GDP,
        )
        == []
    )
    assert (
        offered(
            query_pins("GNI in Kenya", quantity=quantity_requirement(["GNI in Kenya"])),
            _gdp("Nominal", members=_activity("GNI", _GNI)),
        )
        == gni_member
    )
    assert offered(named, no_activity) == gni_member


def _availability_shape(label: str, **kwargs) -> ShapeDraft:
    """A same-cohort count or rate of the availability prune's fixtures."""
    return _with_taxonomy(_shape("Person", "count", None, None, label=label, **kwargs))


def _with_taxonomy(shape: ShapeDraft) -> ShapeDraft:
    return dataclasses.replace(shape, slot_taxonomy={"what:age": ["YearsUpto4"]})


def test_the_national_poverty_shape_is_offered_through_its_total():
    day1 = _shape("SDG_SI_POV_DAY1", "value", None, None, members=("sdg/SI_POV_DAY1",))
    nahc = _shape(
        "SDG_SI_POV_NAHC",
        "value",
        None,
        None,
        members={
            "sdg/SI_POV_NAHC": _arcs({}),
            "sdg/SI_POV_NAHC.URBANISATION--U": _arcs(
                {"sdg_urbanisation": "SDG_UrbanisationEnum_U"}, {"SDG_UrbanisationEnum_U": "Urban"}
            ),
        },
    )

    offered = offered_alternates(
        query_pins(
            "poverty in Bangladesh", quantity=quantity_requirement(["poverty in Bangladesh"])
        ),
        day1,
        {1: nahc},
    )[1]

    assert list(offered.sv_arc_facts or {}) == ["sdg/SI_POV_NAHC"]


def test_a_rural_reading_is_no_reading_of_a_general_rate():
    general = _availability_shape("Under-five mortality rate", place_available=False)
    rural = _availability_shape("Rural under-five mortality rate", place_available=True)
    pins = query_pins(
        "under-five mortality rate", quantity=quantity_requirement(["under-five mortality rate"])
    )

    assert _code(inadmissible_reason(pins, general, rural)) == "specialisation:residence"


def test_an_unresolved_per_capita_spelling_stands_in_for_an_empty_one():
    empty = _gdp(None, "Count_Person", place_available=False)
    survivor = _gdp(None, unresolved=True, place_available=True)

    assert (
        inadmissible_reason(
            query_pins("GDP per capita", quantity=quantity_requirement(["GDP per capita"])),
            empty,
            survivor,
        )
        is None
    )


def test_the_nominal_shape_is_represented_by_nominal_gdp():
    """Beside an inflation-adjusted GDP election with no data, the nominal shape's sector
    members fail on activitySource and nominal GDP represents it."""
    pins = query_pins(
        "GDP India", quantity=quantity_requirement(["GDP India"]), bindings=_GDP_SOLE_OPTION
    )

    offered = offered_alternates(pins, _INFLATION_ADJUSTED_GDP, {1: _NOMINAL_GDP_WITH_SECTORS})[1]

    assert inadmissible_reason(pins, _INFLATION_ADJUSTED_GDP, _NOMINAL_GDP_WITH_SECTORS) is None
    assert list(offered.sv_arc_facts or {}) == [_NOMINAL_GDP_SV]
    assert (
        alternate_representative(
            offered, elected_bindings=list(_GDP_SOLE_OPTION), elected_has_data=False
        )
        == _NOMINAL_GDP_SV
    )


def test_the_reason_depends_on_the_binding():
    """With activitySource bound, every member of the real remainder fails on the elected
    values. With it unbound, the elected reading spans value added and GDP by sector, the
    members pass, and the price basis refuses the remainder beside a nominal election with
    data."""
    remainder = _gdp(
        "RealValue",
        members={
            "RealValue_GVA_Agriculture": _arcs(
                {"activitySource": _GVA, "economicSector": "Agriculture"}
            ),
            "RealValue_GDP_Agriculture": _arcs(
                {"activitySource": _GDP, "economicSector": "Agriculture"}
            ),
        },
    )
    bound = query_pins(
        "GDP India",
        quantity=quantity_requirement(["GDP India"]),
        bindings=_GDP_SOLE_OPTION,
        elected_has_data=True,
    )
    unbound = query_pins(
        "GDP India", quantity=quantity_requirement(["GDP India"]), elected_has_data=True
    )

    assert (
        _code(inadmissible_reason(bound, _NOMINAL_GDP_WITH_SECTORS, remainder))
        == "specialisation:economicSector"
    )
    assert (
        _code(inadmissible_reason(unbound, _NOMINAL_GDP_WITH_SECTORS, remainder))
        == "default:qualifier"
    )


# GNI at purchasing power parity is a different concept from GDP and is refused at every
# entry point.


def test_973_nominated_alternate():
    result = inadmissible_alternates(
        query_pins("GDP of Kenya", quantity=quantity_requirement(["GDP of Kenya"])),
        _NOMINAL_GDP,
        [(1, _GNI_PPP)],
    )

    assert _code(result[1]) == "concept:activitySource"


def test_973_gni_is_no_reading_of_gdp():
    """GNI with data is no reading of nominal GDP, by its member values."""
    empty_gdp = _gdp(
        "Nominal", "Count_Person", members=_activity("GDP_PC", _GDP), place_available=False
    )
    gni = _gdp("Nominal", unresolved=True, members=_activity("GNI_PC", _GNI), place_available=True)

    assert (
        _code(
            inadmissible_reason(
                query_pins("GDP of Kenya", quantity=quantity_requirement(["GDP of Kenya"])),
                empty_gdp,
                gni,
            )
        )
        == "concept:activitySource"
    )


_KEN = "country/KEN"
_FACETS = [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]
_KENYA = {"Kenya": _KEN}


def _node(
    dcid_label: str,
    qualifier: str,
    constraints: dict[str, str],
    *,
    population: str = "EconomicActivity",
    measured: str = "amount",
) -> dict:
    return {
        "label": dcid_label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": dcid_label}]},
            "populationType": {"nodes": [{"dcid": population}]},
            "measuredProperty": {"nodes": [{"dcid": measured}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementQualifier": {"nodes": [{"dcid": qualifier}]},
            **_arcs(constraints),
        },
    }


class _SoleOptionLLM:
    """Elects shape 0, binds `bindings`, and nominates `also_plausible`."""

    def __init__(
        self, also_plausible: list[int] | None, bindings: Sequence[SlotBindingDraft]
    ) -> None:
        self._also_plausible = also_plausible
        self._bindings = list(bindings)

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=self._bindings, also_plausible=self._also_plausible), None


def _resolve_gdp_of_kenya(
    svs: dict[str, dict],
    scores: list[float],
    obs: Mapping[str, str | None],
    also_plausible,
    bindings: Sequence[SlotBindingDraft] = _GDP_SOLE_OPTION,
    *,
    binder: Binder | None = None,
    places: Mapping[str, str] = _KENYA,
    absent: Collection[tuple[str, str]] = (),
):
    """Resolve "GDP of <places>" over `svs`, each StatVar in `obs` holding data at every
    place under the provenance it maps to, except the (StatVar, place) pairs in `absent`."""
    query = "GDP of " + " and ".join(places)
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in places.items()},
            **svs,
        },
        detect={query: {"svs": list(svs), "cosine_scores": scores}},
        resolve=dict(places),
        obs={
            f"{sv}|{dcid}": [{**facet, "provenanceId": provenance} for facet in _FACETS]
            for sv, provenance in obs.items()
            for dcid in places.values()
            if (sv, dcid) not in absent
        },
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=list(places),
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            binder=binder,
            llm=_SoleOptionLLM(also_plausible, bindings),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


def _served(result) -> list[list[str]]:
    return [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs]


def test_973_candidate_admission():
    """A sole-option GDP binding refuses the GNI PPP nomination at candidate admission."""
    result = _resolve_gdp_of_kenya(
        {
            "NominalGdp973": _node(
                "Nominal gross domestic product", "Nominal", {"activitySource": _GDP}
            ),
            "PppGni973": _node(
                "Gross national income at purchasing power parity",
                "PurchasingPowerParity",
                {"activitySource": _GNI},
            ),
        },
        [0.9, 0.85],
        {"NominalGdp973": None, "PppGni973": None},
        also_plausible=[1],
    )

    assert result.status == "definite"
    assert _served(result) == [["NominalGdp973"]]


def test_973_reelection_sibling():
    """An empty inflation-adjusted GDP election is not answered by GNI PPP's data."""
    result = _resolve_gdp_of_kenya(
        {
            "InflationAdjustedGdp973": _node(
                "Inflation-adjusted gross domestic product",
                "InflationAdjusted",
                {"activitySource": _GDP},
            ),
            "PppGni973": _node(
                "Gross national income at purchasing power parity",
                "PurchasingPowerParity",
                {"activitySource": _GNI},
            ),
        },
        [0.9, 0.85],
        {"PppGni973": None},
        also_plausible=None,
    )

    assert result.status == "no_data"


def test_the_imf_series_is_a_twin_of_the_elected_gdp_reading():
    """Nominal GDP's shape holds sector value added beside GDP. With activitySource bound
    the elected reading is GDP alone, so the IMF series is its registered twin. Bound to a
    sector, the reading has no twin."""
    gdp = query_pins(
        "GDP India", quantity=quantity_requirement(["GDP India"]), bindings=_GDP_SOLE_OPTION
    )
    agriculture = query_pins(
        "GDP India",
        quantity=quantity_requirement(["GDP India"]),
        bindings=[_binding("activitySource", [_GVA]), _binding("economicSector", ["Agriculture"])],
    )

    assert inadmissible_reason(gdp, _NOMINAL_GDP_WITH_SECTORS, _IMF_GDP) is None
    assert _registry_twins(
        AlternateJudge(gdp, _NOMINAL_GDP_WITH_SECTORS).served_dcids, _IMF_GDP
    ).keys() == {_IMF_GDP_SV}
    assert not _registry_twins(
        AlternateJudge(agriculture, _NOMINAL_GDP_WITH_SECTORS).served_dcids, _IMF_GDP
    )


class _RecordingClusters(Mapping[str, EquivalentCluster]):
    """The registry's cluster map, recording every StatVar looked up in it."""

    def __init__(self, clusters: Mapping[str, EquivalentCluster]) -> None:
        self._clusters = clusters
        self.read: list[str] = []

    def __getitem__(self, dcid: str) -> EquivalentCluster:
        self.read.append(dcid)
        return self._clusters[dcid]

    def __iter__(self):
        return iter(self._clusters)

    def __len__(self) -> int:
        return len(self._clusters)

    def keys(self):
        return self._clusters.keys()


def test_the_twin_check_looks_up_registered_members_alone(monkeypatch):
    """An election of hundreds of members outside the registry, as a CRS flow's is, costs
    the twin check no lookup; a GDP election looks up its registered member and the
    alternate's."""
    clusters = _RecordingClusters(CLUSTER_BY_MEMBER)
    monkeypatch.setattr(alternate_guard, "CLUSTER_BY_MEMBER", clusters)
    flows = _shape(
        "AidFlowReadingTest", "amount", "Nominal", None, members=tuple(f"F{i}" for i in range(500))
    )
    real_flows = _shape(
        "AidFlowReadingTest", "amount", "RealValue", None, members=tuple(f"R{i}" for i in range(30))
    )

    assert (
        twin_alternates(
            query_pins("aid to Kenya", quantity=quantity_requirement(["aid to Kenya"])),
            flows,
            [(1, real_flows)],
        )
        == []
    )
    assert clusters.read == []

    gdp = query_pins(
        "GDP India", quantity=quantity_requirement(["GDP India"]), bindings=_GDP_SOLE_OPTION
    )
    assert twin_alternates(gdp, _NOMINAL_GDP_WITH_SECTORS, [(1, _IMF_GDP)]) == [1]
    assert sorted(clusters.read) == sorted([_NOMINAL_GDP_SV, _IMF_GDP_SV])


_WB_GVA_AGRICULTURE_SV = "Amount_EconomicActivity_GrossValueAdded_Agriculture_Nominal"
_WB_NOMINAL_GDP_NODES = {
    _WB_GVA_AGRICULTURE_SV: _node(
        "Gross value added: agriculture",
        "Nominal",
        {"activitySource": _GVA, "economicSector": "Agriculture"},
    ),
    _NOMINAL_GDP_SV: _node("Nominal gross domestic product", "Nominal", {"activitySource": _GDP}),
}
_IMF_CONSTRAINTS = {
    "fixedCapitalConsumptionBasis": "GrossOfFixedCapitalConsumption",
    "productTaxBasis": "IncludingProductTaxes",
}
_IMF_GDP_NODE = _node(
    "Gross domestic product (GDP), Current prices",
    "Nominal",
    _IMF_CONSTRAINTS,
    population="EconomicProduction",
    measured="valueAdded",
)
_GDP_PROVENANCE = {
    _WB_GVA_AGRICULTURE_SV: "dc/base/WorldDevelopmentIndicators",
    _NOMINAL_GDP_SV: "dc/base/WorldDevelopmentIndicators",
    _IMF_GDP_SV: "provenance/IMFWorldEconomicOutlook",
}


def _equivalent_readings(result) -> list[list[list[str]]]:
    return [
        [[sv.ref.ref for sv in twin.resolved.members] for twin in twins]
        for spec in result.specs
        if (twins := spec.resolved.equivalent_readings)
    ]


_IMF_BINDINGS = [_binding(prop, [value], "sole_option") for prop, value in _IMF_CONSTRAINTS.items()]
_SOMALIA = {"Kenya": _KEN, "Somalia": "country/SOM"}
# Recall scores within the near-neighbour margin, which nominates the World Bank shape when
# bind does not, and scores outside it that keep the shape on the menu un-nominated.
_TIED = [0.9, 0.89, 0.85]
_APART = [0.9, 0.85, 0.84]


@pytest.mark.parametrize(
    (
        "svs",
        "bindings",
        "scores",
        "also_plausible",
        "places",
        "absent",
        "served",
        "listed",
        "binder",
    ),
    [
        pytest.param(
            {**_WB_NOMINAL_GDP_NODES, _IMF_GDP_SV: _IMF_GDP_NODE},
            _GDP_SOLE_OPTION,
            _TIED,
            [1],
            _KENYA,
            (),
            _NOMINAL_GDP_SV,
            _IMF_GDP_SV,
            None,
            id="world-bank-elected",
        ),
        pytest.param(
            {**_WB_NOMINAL_GDP_NODES, _IMF_GDP_SV: _IMF_GDP_NODE},
            _GDP_SOLE_OPTION,
            _APART,
            None,
            _KENYA,
            (),
            _NOMINAL_GDP_SV,
            _IMF_GDP_SV,
            None,
            id="world-bank-elected-not-nominated",
        ),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _IMF_BINDINGS,
            _TIED,
            [1],
            _KENYA,
            (),
            _NOMINAL_GDP_SV,
            _IMF_GDP_SV,
            None,
            id="imf-elected-world-bank-preferred",
        ),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _IMF_BINDINGS,
            _APART,
            None,
            _KENYA,
            (),
            _NOMINAL_GDP_SV,
            _IMF_GDP_SV,
            None,
            id="imf-elected-world-bank-not-nominated",
        ),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _IMF_BINDINGS,
            _TIED,
            [1],
            _SOMALIA,
            ((_NOMINAL_GDP_SV, "country/SOM"),),
            _IMF_GDP_SV,
            _NOMINAL_GDP_SV,
            ReplyBinder(),
            id="imf-elected-world-bank-lacks-a-place",
        ),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _IMF_BINDINGS,
            _APART,
            None,
            _SOMALIA,
            ((_NOMINAL_GDP_SV, "country/SOM"),),
            _IMF_GDP_SV,
            _NOMINAL_GDP_SV,
            None,
            id="imf-elected-world-bank-not-nominated-lacks-a-place",
        ),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _IMF_BINDINGS,
            _TIED,
            [1],
            {"Cuba": "country/CUB"},
            ((_IMF_GDP_SV, "country/CUB"),),
            _NOMINAL_GDP_SV,
            None,
            None,
            id="imf-elected-without-data-world-bank-answers",
        ),
    ],
)
def test_a_bare_gdp_lists_its_registered_twin_as_an_equivalent_reading(
    svs, bindings, scores, also_plausible, places, absent, served, listed, binder
):
    """A bare GDP answers World Bank nominal GDP, the registry's preferred twin, and lists
    the IMF series under equivalent readings, whichever of the two bind elects and whether or
    not bind nominates the other. Beside an IMF election the World Bank shape,
    which holds sector value added beside GDP, is offered through its twin member. Where the
    World Bank series lacks a place the IMF series covers, the elected IMF series stays
    first. Where the elected IMF series has no data at the place, the World Bank series
    answers for it and lists nothing, since the IMF series holds nothing to read."""
    result = _resolve_gdp_of_kenya(
        svs,
        scores,
        _GDP_PROVENANCE,
        also_plausible=also_plausible,
        bindings=bindings,
        places=places,
        absent=absent,
        binder=binder,
    )

    assert result.status == "definite", result
    assert _served(result) == [[served]]
    assert _equivalent_readings(result) == ([[[listed]]] if listed else [])


_PPP_READINGS = QualifierReadings(
    matched_text="PPP",
    code=QualifierMatch(
        code_system="crsPurposeCode",
        code_dcid="CRSPurposeCode25010",
        label="public-private partnership",
        source="synonym",
        matched_text="PPP",
    ),
    measure=_qualifier("PurchasingPowerParity"),
)


_GDP_PPP = _gdp("PurchasingPowerParity", members=_activity("Amount_GDP_PPP", _GDP))


@pytest.mark.parametrize(
    ("elected", "alternate", "elected_has_data", "expected"),
    [
        pytest.param(_GDP_PPP, _NOMINAL_GDP, False, "pin:qualifier", id="ppp-elected"),
        pytest.param(_NOMINAL_GDP, _GDP_PPP, True, None, id="nominal-elected-with-data"),
    ],
)
def test_ppp_names_a_price_basis_either_shape_carries(
    elected, alternate, elected_has_data, expected
):
    """A "PPP" span names the price basis of whichever shape carries its measure reading:
    elected, it pins a nominal alternate out; as the alternate beside a nominal election
    with data, it keeps the default price basis from refusing it."""
    pins = query_pins(
        "GDP PPP of Kenya",
        quantity=quantity_requirement(["GDP PPP of Kenya"]),
        bindings=_GDP_MENTIONED,
        qualifier_readings=[_PPP_READINGS],
        elected_has_data=elected_has_data,
    )

    assert _code(inadmissible_reason(pins, elected, alternate)) == expected


def test_ppp_naming_a_crs_purpose_pins_no_price_basis():
    """On an aid flow that declares the CRS purpose, "PPP" reads as public-private
    partnership, so a real-terms alternate of the flow is not pinned out."""
    slot = SlotKeyDraft(axis="how", property_dcid="crsPurposeCode", label="purpose")
    purpose = {"crsPurposeCode": "CRSPurposeCode25010"}
    flow_nominal = dataclasses.replace(
        _shape(
            "AidFlowReadingTest", "amount", "Nominal", None, members={"FlowNominal": _arcs(purpose)}
        ),
        slot_keys=(slot,),
    )
    flow_real = dataclasses.replace(
        _shape(
            "AidFlowReadingTest", "amount", "RealValue", None, members={"FlowReal": _arcs(purpose)}
        ),
        slot_keys=(slot,),
    )
    pins = query_pins(
        "PPP aid to Kenya",
        quantity=quantity_requirement(["PPP aid to Kenya"]),
        qualifier_readings=[_PPP_READINGS],
    )

    assert inadmissible_reason(pins, flow_nominal, flow_real) is None


def _imputed_pins() -> tuple[ShapeDraft, Mapping[str, str], Mapping[str, str]]:
    """The imputed multilateral shape the index carries, the slots its dataset pins,
    and the constraints of one of its education rows. DAC2a's aid type 106 shares the
    five-tuple, so the shape declares the imputed family's properties as slots: those
    keys select its family, as they do for the shape Search builds from it."""
    index = get_index()
    ((dataset, five_tuple),) = (
        key for key, pins in index.dataset_pins.items() if pins and index.datasets[key[0]].opt_in
    )
    (family,) = (
        family for family in index.families_by_five_tuple[five_tuple] if dataset in family.datasets
    )
    elected = dataclasses.replace(
        _shape(
            five_tuple.population_type or "",
            five_tuple.measured_property or "",
            five_tuple.measurement_qualifier,
            five_tuple.measurement_denominator,
            stat=five_tuple.stat_type or "",
            label="imputed multilateral",
            roles=family.rows[0].observation_properties,
        ),
        slot_keys=tuple(
            SlotKeyDraft(axis="what", property_dcid=prop, label=prop)
            for prop in sorted(family.properties)
        ),
        sv_arc_facts_complete=False,
    )
    education = next(
        row.constraint_properties
        for row in index_member_rows(elected, index)
        if row.constraint_properties["crsPurpose"].startswith("CRSPurposeCode11")
    )
    return elected, index.dataset_pins[dataset, five_tuple], education


@pytest.mark.parametrize("pins_bound", [True, False], ids=["pins-bound", "before-settle"])
@pytest.mark.parametrize(
    ("aid_type", "expected"),
    [(None, None), ("DAC2aAidTypeOther", "concept:dac2aAidType")],
    ids=["pinned-values", "another-aid-type"],
)
def test_a_slot_the_five_tuple_pins_is_held_by_the_imputed_reading(
    pins_bound: bool, aid_type: str | None, expected: str | None
):
    """Every row of the imputed five-tuple holds its pinned slots, so the elected reading
    holds them whether or not the settle step has bound them yet. A sibling that carries
    the pinned values passes, and one with another aid type differs in concept. Neither is
    a specialisation the question did not name."""
    elected, pinned, education = _imputed_pins()
    constraints = dict(education)
    if aid_type is not None:
        constraints["dac2aAidType"] = aid_type
    sibling = _shape(
        elected.pop_type_dcid,
        elected.meas_prop_dcid,
        "RealValue",
        elected.meas_denom_dcid,
        stat=elected.stat_type_dcid,
        roles=elected.declared_roles,
        members={"ImputedSibling": _arcs(constraints)},
    )
    bindings = [_binding("crsPurpose", ["CRSPurposeCode110"])]
    if pins_bound:
        bindings += [_binding(prop, [value], "sole_option") for prop, value in pinned.items()]
    pins = query_pins(
        "imputed multilateral education aid",
        quantity=quantity_requirement(["imputed multilateral education aid"]),
        bindings=bindings,
    )

    assert set(pinned) == {"crsTypeOfFlow", "dac2aAidType"}
    assert _code(inadmissible_reason(pins, elected, sibling)) == expected
    assert inadmissible_reason(pins, elected, dataclasses.replace(elected, shape_id="twin")) is None


def _named(name: str) -> dict:
    return {"name": {"nodes": [{"value": name}]}}


def test_a_member_of_a_larger_shape_reads_residence_from_its_own_name():
    """A total recalled beside its rural sibling, in a shape labelled from the rural one,
    is not read as rural: it passes and represents the shape."""
    population = _shape(
        "Person", "count", None, None, label="Total population", members=("sv_total",)
    )
    mixed = _shape(
        "Person",
        "count",
        None,
        None,
        label="Rural population",
        members={
            "Count_Person_Rural": _named("Rural population"),
            "Count_Person": _named("Total population"),
        },
    )
    pins = query_pins("population of Kenya", quantity=quantity_requirement(["population of Kenya"]))

    assert inadmissible_reason(pins, population, mixed) is None
    assert list(offered_alternates(pins, population, {1: mixed})[1].sv_arc_facts or {}) == [
        "Count_Person"
    ]


# The default reading removes, before bind, the members of a shape the index does
# not carry that a better-ranked price basis covers at the query's places and dates.

_ZMB = "country/ZMB"
_SOM = "country/SOM"


def _held(first: int, last: int, *, dated: bool = True) -> ObservedYears:
    """Every year from `first` to `last`, listed when `dated`, else known by its ends alone."""
    return ObservedYears(
        first=first, last=last, years=frozenset(range(first, last + 1)) if dated else None
    )


_YEARS = _held(2015, 2022)


def _covering(
    shape: ShapeDraft, places: dict[str, ObservedYears] | None, *, score: float | None = None
) -> ShapeDraft:
    """`shape` with every member observed at `places` (`None`: availability unknown)."""
    coverage = None if places is None else dict.fromkeys(shape.sv_arc_facts or {}, places)
    return dataclasses.replace(shape, member_coverage=coverage, representative_score=score)


def _collapse(
    *shapes: ShapeDraft,
    named: tuple[str, ...] = (),
    window: TimeWindow | Literal["latest"] | None = None,
) -> list[tuple[str, list[str]]]:
    reading = DefaultReading(
        order=price_basis_order(get_measure_index(), named),
        window=window,
    )
    collapsed = collapse_measure_families(
        shapes, get_index(), get_measure_index().axes, reading=reading
    )
    return [(shape.label, list(shape.sv_arc_facts or {})) for shape in collapsed]


def _entry(shape: ShapeDraft) -> tuple[str, list[str]]:
    return shape.label, list(shape.sv_arc_facts or {})


@pytest.mark.parametrize(
    "inflation_adjusted_places",
    [{}, {_KEN: _YEARS}],
    ids=["inflation-adjusted-empty", "inflation-adjusted-covered"],
)
@pytest.mark.parametrize(
    "nominal", [_NOMINAL_GDP, _NOMINAL_GDP_WITH_SECTORS], ids=["kenya-menu", "india-menu"]
)
def test_inflation_adjusted_gdp_leaves_beside_nominal_gdp(inflation_adjusted_places, nominal):
    """InflationAdjusted ranks inside RealValue's reading, below Nominal, and its counterpart
    member, Nominal GDP, covers it at the place whether or not it has data there. On the
    India menu the counterpart sits among the sector value-added members."""
    collapsed = _collapse(
        _covering(_INFLATION_ADJUSTED_GDP, inflation_adjusted_places),
        _covering(nominal, {_KEN: _YEARS}),
    )

    assert collapsed == [_entry(nominal)]


@pytest.mark.parametrize(
    ("nominal_places", "kept"),
    [
        pytest.param({_KEN: _YEARS}, [_NOMINAL_GDP], id="nominal-covered"),
        pytest.param({}, [_NOMINAL_GDP, _REAL_GDP], id="nominal-empty"),
        pytest.param(None, [_NOMINAL_GDP, _REAL_GDP], id="nominal-unknown"),
    ],
)
def test_real_gdp_leaves_only_where_nominal_gdp_covers_it(nominal_places, kept):
    collapsed = _collapse(
        _covering(_NOMINAL_GDP, nominal_places), _covering(_REAL_GDP, {_KEN: _YEARS})
    )

    assert collapsed == [_entry(shape) for shape in kept]


def test_a_member_with_data_nowhere_leaves_beside_a_counterpart_of_unknown_coverage():
    collapsed = _collapse(_covering(_NOMINAL_GDP, None), _covering(_REAL_GDP, {}))

    assert collapsed == [_entry(_NOMINAL_GDP)]


def test_only_members_with_a_nominal_counterpart_leave():
    """The real-value shape loses its GDP and agriculture members, whose constraint maps a
    nominal member holds, and keeps its mining member, with the taxonomy read over it."""
    real_by_activity = dataclasses.replace(
        _gdp(
            "RealValue",
            label="Real gross domestic product by economic activity",
            members={
                "RealValue_GVA_Agriculture": _arcs(
                    {"activitySource": _GVA, "economicSector": "Agriculture"}
                ),
                "RealValue_GVA_Mining": _arcs({"activitySource": _GVA, "economicSector": "Mining"}),
                "RealValue_GDP": _arcs({"activitySource": _GDP}),
            },
        ),
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="economicSector", label="sector"),),
        slot_taxonomy={"how:economicSector": ["Agriculture", "Mining"]},
    )

    (nominal, real) = collapse_measure_families(
        [
            _covering(_NOMINAL_GDP_WITH_SECTORS, {_KEN: _YEARS}, score=0.8),
            _covering(real_by_activity, {_KEN: _YEARS}, score=0.9),
        ],
        get_index(),
        get_measure_index().axes,
        reading=DefaultReading(order=price_basis_order(get_measure_index(), ())),
    )

    assert nominal.representative_score == 0.8
    assert list(real.sv_arc_facts or {}) == ["RealValue_GVA_Mining"]
    assert real.slot_taxonomy == {"how:economicSector": ["Mining"]}
    assert real.representative_score == 0.9


def test_a_named_real_reading_ranks_first():
    """ "real GDP" ranks RealValue, then its alias InflationAdjusted, before Nominal. The
    real-value shape survives and takes the stronger score and the earlier position."""
    (survivor,) = collapse_measure_families(
        [
            _covering(_NOMINAL_GDP, {_ZMB: _YEARS}, score=0.9),
            _covering(_INFLATION_ADJUSTED_GDP, {_ZMB: _YEARS}, score=0.85),
            _covering(_REAL_GDP, {_ZMB: _YEARS}, score=0.8),
        ],
        get_index(),
        get_measure_index().axes,
        reading=DefaultReading(order=price_basis_order(get_measure_index(), ("RealValue",))),
    )

    assert _entry(survivor) == _entry(_REAL_GDP)
    assert survivor.representative_score == 0.9


def test_a_named_ppp_reading_ranks_first():
    gdp_ppp = _gdp(
        "PurchasingPowerParity", label="GDP PPP", members=_activity("Amount_GDP_PPP", _GDP)
    )

    collapsed = _collapse(
        _covering(_NOMINAL_GDP, {_KEN: _YEARS}),
        _covering(gdp_ppp, {_KEN: _YEARS}),
        named=("PurchasingPowerParity",),
    )

    assert collapsed == [_entry(gdp_ppp)]


def test_an_unranked_qualifier_takes_no_part():
    """A growth rate stating no qualifier, or a non-price-basis one, neither removes nor is
    removed by the real-value growth rate."""
    annual = _gdp(
        "Annual",
        stat="growthRate",
        label="Annual growth rate",
        members=_activity("GrowthRate_Annual", _GDP),
    )
    shapes = [
        _covering(_GDP_GROWTH, {_KEN: _YEARS}),
        _covering(annual, {_KEN: _YEARS}),
        _covering(
            dataclasses.replace(_GROWTH_REAL, sv_arc_facts=_activity("RealGrowth", _GDP)), {}
        ),
    ]

    assert _collapse(*shapes) == [_entry(shape) for shape in shapes]


def test_a_role_bearing_pair_reaches_bind_when_its_availability_is_unknown():
    """The probe reads subject facets, so a role-bearing shape's coverage is unknown and the
    guard's default operation decides after bind."""
    roles = ("observationAbout", "financingRecipient")
    nominal = _gdp("Nominal", label="Nominal", members=_activity("NominalFlow", _GDP), roles=roles)
    real = _gdp("RealValue", label="Real", members=_activity("RealFlow", _GDP), roles=roles)

    assert _collapse(nominal, real) == [_entry(nominal), _entry(real)]


@pytest.mark.parametrize(
    ("nominal_places", "window", "real_leaves"),
    [
        pytest.param(
            {_KEN: _held(2010, 2019)},
            TimeWindow(start_year=2020, end_year=2020),
            False,
            id="nominal-lacks-the-year",
        ),
        pytest.param(
            {_KEN: _held(2010, 2019)},
            TimeWindow(start_year=2015, end_year=2018),
            True,
            id="nominal-holds-the-window",
        ),
        pytest.param(
            {_KEN: _held(2010, 2019, dated=False)},
            TimeWindow(start_year=2015, end_year=2018),
            False,
            id="nominal-span-proves-only-its-ends",
        ),
        pytest.param(
            {_KEN: ObservedYears(first=2010, last=2019, years=frozenset({2010, 2019}))},
            TimeWindow(start_year=2015, end_year=2015),
            False,
            id="nominal-gap-at-the-year",
        ),
        pytest.param(
            {_KEN: _held(2010, 2019)},
            TimeWindow(start_year=2023),
            True,
            id="real-holds-no-requested-year",
        ),
        pytest.param({_KEN: _held(2010, 2019)}, "latest", False, id="real-is-more-recent"),
        pytest.param({_KEN: _held(2010, 2021)}, "latest", True, id="nominal-is-as-recent"),
        pytest.param(
            {_KEN: ObservedYears(first=None, last=None, years=None)},
            TimeWindow(start_year=2015),
            False,
            id="nominal-years-unknown",
        ),
        pytest.param({_KEN: _held(2010, 2021)}, None, True, id="undated"),
    ],
)
@pytest.mark.parametrize("real_dated", [True, False], ids=["real-dated", "real-span"])
def test_cover_is_date_aware(nominal_places, window, real_leaves, real_dated):
    """Real GDP holds 2010 to 2021 at Kenya, every year listed or known by its ends alone; it
    leaves when Nominal GDP is shown to hold every requested year it may hold. A year
    between a span's ends is shown only by a dated read."""
    collapsed = _collapse(
        _covering(_NOMINAL_GDP, nominal_places),
        _covering(_REAL_GDP, {_KEN: _held(2010, 2021, dated=real_dated)}),
        window=window,
    )

    assert (_entry(_REAL_GDP) not in collapsed) is real_leaves


def test_a_nominal_span_around_the_year_keeps_the_only_reading_holding_it():
    """ "GDP of Kenya in 2015": Nominal GDP holds 2010 and 2020 at Kenya and Real GDP holds
    2015 alone. The nominal span encloses 2015 without holding it, so the dated probe's
    years keep Real GDP, which answers."""
    real_sv = "Amount_EconomicActivity_GrossDomesticProduction_RealValue"
    query = "GDP of Kenya in 2015"
    svs = {
        real_sv: _node("Real gross domestic product", "RealValue", {"activitySource": _GDP}),
        _NOMINAL_GDP_SV: _WB_NOMINAL_GDP_NODES[_NOMINAL_GDP_SV],
    }
    provenance = "dc/base/WorldDevelopmentIndicators"
    graph = FakeGraph(
        nodes={_KEN: {"label": "Kenya", "type": "Country"}, **svs},
        detect={query: {"svs": list(svs), "cosine_scores": [0.9, 0.85]}},
        resolve={"Kenya": _KEN},
        obs={
            f"{real_sv}|{_KEN}": [
                {
                    "earliestDate": "2015",
                    "latestDate": "2015",
                    "obsCount": 1,
                    "dates": ["2015"],
                    "values": [10.0],
                    "unit": "USDollar",
                    "provenanceId": provenance,
                }
            ],
            f"{_NOMINAL_GDP_SV}|{_KEN}": [
                {
                    "earliestDate": "2010",
                    "latestDate": "2020",
                    "obsCount": 2,
                    "dates": ["2010", "2020"],
                    "values": [5.0, 15.0],
                    "unit": "USDollar",
                    "provenanceId": provenance,
                }
            ],
        },
    )

    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=DateRequest(
                window=TimeWindow(start_year=2015, end_year=2015), latest=False
            ),
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_SoleOptionLLM(None, _GDP_SOLE_OPTION),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "definite", result
    assert _served(result) == [[real_sv]]


def test_cover_is_per_place():
    """Nominal GDP lacks Somalia, so Real GDP stays for "GDP of Kenya and Somalia"."""
    collapsed = _collapse(
        _covering(_NOMINAL_GDP, {_KEN: _YEARS}),
        _covering(_REAL_GDP, {_KEN: _YEARS, _SOM: _YEARS}),
    )

    assert collapsed == [_entry(_NOMINAL_GDP), _entry(_REAL_GDP)]


def test_real_gdp_of_zambia_keeps_its_named_price_basis_in_the_un_pass():
    """ "real GDP of Zambia", replayed from its recording. Bind elects the RealValue GDP
    shape, whose members hold no Zambia data on the dev graph. The World Bank's constant
    US dollar series is `worldBank/NY_GDP_MKTP_KD`, which declares no qualifier and is not
    recalled. The qualifier pin refuses every Nominal sibling, so nominal GDP, which has
    Zambia data, is kept out, and the UN pass answers with UNIDO's constant-price
    series. UN series declare no qualifier, so "real" reads as not applied. The reply rates
    the current-price series it nominates below the nomination bar, so it is no rival."""
    result = offline_resolve(make_request("real GDP of Zambia"))

    assert isinstance(result, Resolution)
    part = result.parts[0]
    assert isinstance(part, DefinitePart)
    assert ref_dcid(part.spec.resolved.shape.population_type) == "undata/unido/SERIES-NA_GDPCOD"
    assert QUERY_QUALIFIER_DROPPED in [warning.code for warning in part.warnings]


# Interpretation admissibility: each rule refuses one reading of a corpus question and admits
# another, each refusal under the reason a part with no admissible reading answers with.

_ADM_KENYA = "country/KEN"
_ADM_FRANCE = "country/FRA"
_ADM_FLOW_ROLES = ("financingProvider", "financingRecipient")
_ADM_GDP_NOMINAL = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_ADM_GNI_NOMINAL = "Amount_EconomicActivity_GrossNationalIncome_Nominal"
_ADM_HEALTH_CODE = QualifierMatch(
    code_system="expenditureType",
    code_dcid="HealthcareExpenditure",
    label="Healthcare expenditure",
    source="label",
    matched_text="health expenditure",
)


def _flow(label: str, roles: tuple[str, ...]) -> ShapeDraft:
    return _shape("FinancialFlow", "amount", "Nominal", None, label=label, roles=roles)


def _with_slot(shape: ShapeDraft, prop: str, members: dict[str, dict]) -> ShapeDraft:
    return dataclasses.replace(
        shape,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=prop, label=prop),),
        sv_arc_facts=members,
    )


def _class_named(shape: ShapeDraft, name: str) -> ShapeDraft:
    return dataclasses.replace(shape, pop_type_name=name)


def _part(
    variable: str,
    *,
    places: Mapping[str, tuple[str, Sequence[RoleAxis]]] | None = None,
    operations: Sequence[Operation] = (),
) -> ResolveState:
    """The state a part reaches admissibility in: `variable`, and each place by its name,
    with its dcid and the flow axes the door gave it."""

    places = places or {}
    state = ResolveState(
        variable=variable,
        entities=list(places),
        date_request=None,
        detect_query=variable,
        role_query=variable,
        dropped_qualifier_phrases=[],
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=QuantityReadOnlyLLM(),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
        operations=list(operations),
    )
    state.place_sets = place_sets({name: dcid for name, (dcid, _axes) in places.items()})
    state.entity_roles = {
        name: [DoorRole(axis=axis, span=name) for axis in axes]
        for name, (_dcid, axes) in places.items()
    }
    state.role_bearing_outranks = True
    return state


def _judged(
    state: ResolveState,
    shape: ShapeDraft,
    matches: Mapping[frozenset[str], tuple[QualifierMatch, ...]] | None = None,
) -> Interpretation:
    (judged,) = judge_measure_and_roles(state, [shape])
    (judged,) = judge_pinned_parts(state, [judged], matches or {})
    return judged


_ADM_TO_KENYA = {"Kenya": (_ADM_KENYA, ("destination",))}
_ADM_FRANCE_TO_KENYA = {
    "France": (_ADM_FRANCE, ("origin",)),
    "Kenya": (_ADM_KENYA, ("destination",)),
}
_ADM_EDUCATION_SHARE = _with_slot(
    _gdp(None, _ADM_GDP_NOMINAL, label="percent of nominal GDP spent on education"),
    "expenditureType",
    {"worldBank/SE_XPD_TOTL_GD_ZS": _arcs({"expenditureType": "EducationExpenditure"})},
)
_ADM_HEALTH_SHARE = _with_slot(
    _gdp(None, _ADM_GDP_NOMINAL, label="Current health expenditure (% of GDP)"),
    "expenditureType",
    {"worldBank/SH_XPD_CHEX_GD_ZS": _arcs({"expenditureType": "HealthcareExpenditure"})},
)

INTERPRETATION_CASES = [
    pytest.param(
        _part("HIV prevalence in Nigeria"),
        _class_named(
            _shape("SDG_SH_HIV_INCD", "value", None, None, label="HIV incidence"),
            "Number of new HIV infections per 1,000 uninfected population",
        ),
        ("variable_not_resolved", None),
        id="hiv-prevalence-refuses-a-new-infections-series-on-its-measure-class",
    ),
    pytest.param(
        _part("HIV prevalence in Nigeria"),
        _class_named(
            _shape("SDG_SH_HIV_PREV", "value", None, None, label="HIV prevalence"),
            "Prevalence of HIV",
        ),
        (None, None),
        id="hiv-prevalence-admits-a-prevalence-series",
    ),
    pytest.param(
        _part("aid from France to Kenya", places=_ADM_FRANCE_TO_KENYA),
        _gdp("Nominal", label="Nominal gross domestic product"),
        ("role_incompatible", None),
        id="both-flow-ends-refuse-a-subject-only-series",
    ),
    pytest.param(
        _part("aid from France to Kenya", places=_ADM_FRANCE_TO_KENYA),
        _flow("official development assistance", _ADM_FLOW_ROLES),
        (None, None),
        id="both-flow-ends-admit-a-flow-with-both-roles",
    ),
    pytest.param(
        _part("aid to Kenya", places=_ADM_TO_KENYA),
        _flow("donor totals", ("financingProvider",)),
        ("role_incompatible", None),
        id="a-recipient-refuses-a-provider-only-flow",
    ),
    pytest.param(
        _part("aid to Kenya", places=_ADM_TO_KENYA),
        _gdp("Nominal", label="Nominal gross domestic product"),
        (None, None),
        id="a-directed-place-is-a-subject-only-series-subject",
    ),
    pytest.param(
        _part(
            "external debt by creditor",
            places=_ADM_TO_KENYA,
            operations=[Operation(verb="rank", phrase="top 5 creditors", over="origin")],
        ),
        _flow("external debt stocks", ("financingRecipient",)),
        ("role_incompatible", None),
        id="a-rank-over-creditors-refuses-a-recipient-only-debt-series",
    ),
    pytest.param(
        _part("debt to GDP ratio", places={"Ghana": ("country/GHA", ())}),
        _shape(
            "FinancialPosition",
            "amount",
            "Nominal",
            None,
            label="Outstanding public or publicly guaranteed external debt owed by the general "
            "government (current US$)",
        ),
        ("denominator_not_available", "form"),
        id="a-ratio-refuses-the-dollar-debt-level-on-its-form",
    ),
    pytest.param(
        _part("debt to GDP ratio", places={"Ghana": ("country/GHA", ())}),
        _shape(
            "FinancialPosition",
            "amount",
            None,
            _ADM_GNI_NOMINAL,
            label="Outstanding external debt (% of GNI)",
        ),
        (None, None),
        id="a-ratio-admits-the-share-of-gni-whose-denominator-bind-judges",
    ),
    pytest.param(
        _part("measles incidence rate Kenya", places={"Kenya": (_ADM_KENYA, ())}),
        _REPORTED_CASES,
        ("denominator_not_available", "form"),
        id="an-incidence-rate-refuses-the-reported-case-count",
    ),
    pytest.param(
        _part("measles incidence rate Kenya", places={"Kenya": (_ADM_KENYA, ())}),
        _MORTALITY,
        (None, None),
        id="an-incidence-rate-admits-the-mortality-rate",
    ),
    pytest.param(
        _part("birth rate Ethiopia", places={"Ethiopia": ("country/ETH", ())}),
        _shape("BirthEvent", "count", None, None, label="Count of birth events"),
        ("denominator_not_available", "form"),
        id="a-birth-rate-refuses-the-raw-birth-count",
    ),
    pytest.param(
        _part("Current health expenditure as share of GDP", places={"Kenya": (_ADM_KENYA, ())}),
        _ADM_EDUCATION_SHARE,
        ("qualifier_not_bound", "pinned_code"),
        id="health-expenditure-refuses-the-education-share-on-the-code-it-names",
    ),
    pytest.param(
        _part("Current health expenditure as share of GDP", places={"Kenya": (_ADM_KENYA, ())}),
        _ADM_HEALTH_SHARE,
        (None, None),
        id="health-expenditure-admits-the-health-share",
    ),
]


@pytest.mark.parametrize(("state", "shape", "expected"), INTERPRETATION_CASES)
def test_each_rule_admits_or_refuses_a_reading(state, shape, expected):
    judged = _judged(state, shape, {frozenset({"expenditureType"}): (_ADM_HEALTH_CODE,)})

    assert (judged.reason, judged.pinned_failure) == expected
    assert judged.admissible is (expected[0] is None)


def test_a_reading_refused_on_a_pinned_code_names_the_codes_it_cannot_bind():
    state = _part("Current health expenditure as share of GDP", places={"Kenya": (_ADM_KENYA, ())})

    judged = _judged(
        state, _ADM_EDUCATION_SHARE, {frozenset({"expenditureType"}): (_ADM_HEALTH_CODE,)}
    )

    assert judged.pinned_codes == (_ADM_HEALTH_CODE,)
    (record,) = pinned_code_records(judged)
    assert (record.text, record.slot, record.disposition) == (
        "health expenditure",
        "expenditureType",
        "unbound",
    )
    assert [ref.ref for ref in record.named] == ["HealthcareExpenditure"]


@pytest.mark.parametrize(
    ("reasons", "elected"),
    [
        (("denominator_not_available", "role_incompatible"), "role_incompatible"),
        (("qualifier_not_bound", "denominator_not_available"), "qualifier_not_bound"),
        (("denominator_not_available", "variable_not_resolved"), "variable_not_resolved"),
        ((), None),
    ],
)
def test_a_part_with_no_admissible_reading_refuses_with_the_first_reason_in_order(reasons, elected):
    """The no-data resolution order elects the reason among the refused readings'."""
    interpretations = InterpretationSet(
        interpretations=tuple(
            Interpretation(shape=_gdp(None, label=f"reading {i}"), reason=reason)
            for i, reason in enumerate(reasons)
        )
    )

    assert interpretations.refusal_reason() == elected


def test_a_declined_reading_flagged_in_place_of_its_family_names_the_reading_asked():
    """corpus nn-05: the model declines every admissible reading of "measles incidence rate"
    and names the reported case counts, refused on the rate the question names. The flagged
    match names measles, the condition the phrase states, and not every condition the
    family counts. The bind reply is that decline, held fixed."""
    llm = FixedBindReplyLLM(
        _UniversalBindOutput(
            chosen_shape_index=0, ask="no shape measures the phrase", nearest_refused_index=0
        )
    )
    response = offline_resolve(make_request("measles incidence rate Kenya"), llm=llm)

    (prompt,) = llm.bind_prompts
    assert "  0: Count of Measles incidents" in prompt.partition("NOT ELECTABLE")[2]

    (part,) = response.parts
    assert part.status == "no_data"
    assert part.no_data.reason == "denominator_not_available"
    flagged = part.no_data.flagged_match
    assert flagged is not None
    condition = next(
        slot for slot in flagged.resolved.slots if slot.key.property.ref == "medicalCondition"
    )
    assert condition.binding.kind == "value"
    assert condition.binding.value.ref.ref == "Measles"
    assert flagged.resolved.stat_vars.member_count == 1
