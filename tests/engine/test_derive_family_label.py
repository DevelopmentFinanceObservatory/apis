"""Tests the label `derive_shapes` gives a shape: an index family reads as what every
row of the family shares, a multi-member UN System family as its population type plus
the values every member shares, and every other shape keeps its member's name. A resubmit
of a first answer's identity takes the first answer's label."""

from __future__ import annotations

import asyncio

import pytest

from code_index.shared import get_index
from origin_catalogue import UN_SYSTEM
from search.engine.bind import SlotBindingDraft
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.retrieval.derive import derive_shapes
from search.models import ResubmitIdentityPayload
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_CME = "undata/unicef/SERIES-CME_MR"
_NUT = "undata/who/SERIES-NUT_ANTH_SH"

_UN_LABELS = {
    _CME: "Child mortality rate",
    _NUT: "Prevalence of malnutrition among children",
    "undataAge": "Age",
    "undataSex": "Sex",
    "undataNutStatus": "Nutritional status",
    "undata/AGE-Y0T4": "Under 5 years old",
    "undata/AGE-Y0": "Under 1 year old",
    "undata/AGE-M24T59": "24 to 59 months old",
    "undata/AGE-M0T23": "0 to 23 months old",
    "undata/SEX-M": "Male",
    "undata/SEX-F": "Female",
    "undata/NUT_STATUS-STUNTING": "Stunting (height-for-age <-2 SD)",
}


def _un_member(name: str, population_type: str, constraints: dict[str, str]) -> dict:
    arcs: dict = {
        "name": {"nodes": [{"value": name}]},
        "populationType": {"nodes": [{"dcid": population_type}]},
        "measuredProperty": {"nodes": [{"dcid": "value"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
    }
    for prop, value in constraints.items():
        arcs[prop] = {"nodes": [{"dcid": value}]}
    return {"label": name, "arcs": arcs}


def _graph(members: dict[str, dict]) -> FakeGraph:
    nodes = {dcid: {"label": label} for dcid, label in _UN_LABELS.items()}
    nodes.update(members)
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _only_shape(members: dict[str, dict], recalled: list[str]) -> str:
    shapes = derive_shapes(
        confirmed_svs=list(members),
        graph=_graph(members),
        sv_scores={sv: 0.9 - i * 0.01 for i, sv in enumerate(recalled)},
    )
    assert len(shapes) == 1
    return shapes[0].label


def test_un_family_spanning_sex_and_age_reads_as_its_population_type():
    """The recalled member's Age and Sex values stay out of the family label."""
    members = {
        "undata/unicef/CME_MR.AGE--Y0T4__SEX--M": _un_member(
            "Child mortality rate [Age=Under 5 years old, Sex=Male]",
            _CME,
            {"undataAge": "undata/AGE-Y0T4", "undataSex": "undata/SEX-M"},
        ),
        "undata/unicef/CME_MR.AGE--Y0T4__SEX--F": _un_member(
            "Child mortality rate [Age=Under 5 years old, Sex=Female]",
            _CME,
            {"undataAge": "undata/AGE-Y0T4", "undataSex": "undata/SEX-F"},
        ),
        "undata/unicef/CME_MR.AGE--Y0__SEX--M": _un_member(
            "Child mortality rate [Age=Under 1 year old, Sex=Male]",
            _CME,
            {"undataAge": "undata/AGE-Y0", "undataSex": "undata/SEX-M"},
        ),
    }

    label = _only_shape(members, recalled=["undata/unicef/CME_MR.AGE--Y0T4__SEX--M"])

    assert label == "Child mortality rate"


def test_un_family_keeps_the_value_every_member_shares():
    """Nutritional status is Stunting for every member; Age varies and stays out."""
    members = {
        "undata/who/NUT_ANTH_SH.AGE--M24T59__NUT_STATUS--STUNTING": _un_member(
            "Prevalence of malnutrition among children "
            "[Age=24 to 59 months old, Nutritional status=Stunting (height-for-age <-2 SD)]",
            _NUT,
            {"undataAge": "undata/AGE-M24T59", "undataNutStatus": "undata/NUT_STATUS-STUNTING"},
        ),
        "undata/who/NUT_ANTH_SH.AGE--M0T23__NUT_STATUS--STUNTING": _un_member(
            "Prevalence of malnutrition among children "
            "[Age=0 to 23 months old, Nutritional status=Stunting (height-for-age <-2 SD)]",
            _NUT,
            {"undataAge": "undata/AGE-M0T23", "undataNutStatus": "undata/NUT_STATUS-STUNTING"},
        ),
    }

    label = _only_shape(
        members, recalled=["undata/who/NUT_ANTH_SH.AGE--M24T59__NUT_STATUS--STUNTING"]
    )

    assert label == (
        "Prevalence of malnutrition among children "
        "[Nutritional status=Stunting (height-for-age <-2 SD)]"
    )


def test_single_member_un_shape_keeps_its_member_name():
    member = "undata/unicef/CME_MR.AGE--Y0T4__SEX--M"
    members = {
        member: _un_member(
            "Child mortality rate [Age=Under 5 years old, Sex=Male]",
            _CME,
            {"undataAge": "undata/AGE-Y0T4", "undataSex": "undata/SEX-M"},
        )
    }

    label = _only_shape(members, recalled=[member])

    assert label == "Child mortality rate [Age=Under 5 years old, Sex=Male]"


def test_base_family_keeps_its_recalled_member_name():
    """A base Data Commons family keeps its recalled member's name, as it always has,
    even when that name carries a gender the family spans."""

    def base_member(name: str, gender: str) -> dict:
        return {
            "label": name,
            "arcs": {
                "name": {"nodes": [{"value": name}]},
                "populationType": {"nodes": [{"dcid": "Person"}]},
                "measuredProperty": {"nodes": [{"dcid": "lifeExpectancy"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "constraintProperties": {"nodes": [{"dcid": "gender"}]},
                "gender": {"nodes": [{"dcid": gender}]},
            },
        }

    members = {
        "LifeExpectancy_Person_Female": base_member("Female life expectancy", "Female"),
        "LifeExpectancy_Person_Male": base_member("Male life expectancy", "Male"),
    }
    graph = FakeGraph(
        nodes={"Person": {"label": "Person"}, "gender": {"label": "gender"}, **members},
        obs={},
        detect={},
        resolve={},
    )

    shapes = derive_shapes(
        confirmed_svs=list(members),
        graph=graph,
        sv_scores={"LifeExpectancy_Person_Female": 0.9, "LifeExpectancy_Person_Male": 0.8},
    )

    assert [shape.label for shape in shapes] == ["Female life expectancy"]


def _index_member(dcid: str, name: str) -> dict:
    """A graph node for the fixture index row `dcid`, named `name`."""
    row = next(row for row in get_index().statvars if row.dcid == dcid)
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": name}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }
    if row.measurement_denominator is not None:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": row.measurement_denominator}]}
    return {"label": name, "type": "StatisticalVariable", "arcs": arcs}


@pytest.mark.parametrize(
    ("dcid", "name", "expected"),
    [
        (
            "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1906_Nominal",
            "Relief food aid (net disbursements)",
            "Financial flow disbursements, OECD DAC1",
        ),
        (
            "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1010_Nominal_AsAFractionOf_"
            "PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal",
            "Official development assistance (net disbursements, % of GNI)",
            "Financial flow net disbursements as a share of GNI, OECD DAC1",
        ),
        (
            "GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1010_Nominal_AsAFractionOf_"
            "PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal",
            "Official development assistance (grant equivalents, % of GNI)",
            "Financial flow grant equivalents as a share of GNI, official development assistance, "
            "OECD DAC1",
        ),
        (
            "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
            "CRSPurposeCode11110_Nominal",
            "Official development assistance gross disbursements for education policy",
            "Financial flow disbursements, OECD Creditor Reporting System (CRS)",
        ),
    ],
    ids=["dac1-amount", "dac1-net-share-of-gni", "dac1-ge-share-of-gni", "crs-amount"],
)
def test_index_family_reads_as_what_every_row_shares(dcid, name, expected):
    """A one-member sample of an index family takes the family's label, led by its
    population type's graph name, so the member's aid type or purpose, which an open slot
    spans, stays out of it."""
    member = _index_member(dcid, name)

    shapes = derive_shapes(
        confirmed_svs=[dcid],
        graph=FakeGraph(
            nodes={dcid: member, "FinancialFlow": {"label": "financial flow"}},
            obs={},
            detect={},
            resolve={},
        ),
    )

    assert [shape.label for shape in shapes] == [expected]


_MORT = "undata/who/SERIES-MORT_U5_RATE"
_MORT_TOTAL = "undata/who/MORT_U5_RATE"
_MORT_FEMALE = "undata/who/MORT_U5_RATE.SEX--F"
_MORT_MALE = "undata/who/MORT_U5_RATE.SEX--M"
_MORT_NAMES = {
    _MORT_TOTAL: "Under-five mortality rate",
    _MORT_FEMALE: "Under-five mortality rate [Sex=Female]",
    _MORT_MALE: "Under-five mortality rate [Sex=Male]",
}
_MORT_SEX = {_MORT_FEMALE: ("undata/SEX-F", "Female"), _MORT_MALE: ("undata/SEX-M", "Male")}


class _BindSexLLM:
    """Elects the first shape and binds its sex slot to `sexes`."""

    def __init__(self, sexes: list[str]) -> None:
        self._sexes = sexes

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        sex = SlotBindingDraft(
            axis="how",
            property_dcid="undataSex",
            kind="value" if len(self._sexes) == 1 else "set",
            value_dcids=self._sexes,
        )
        return elected_reply(0, bindings=[sex]), None


@pytest.mark.parametrize(
    ("sexes", "served", "expected"),
    [
        pytest.param(
            ["undata/SEX-F", "undata/SEX-M"],
            [_MORT_FEMALE, _MORT_MALE],
            _MORT_NAMES[_MORT_TOTAL],
            id="two-series-read-as-the-family",
        ),
        pytest.param(
            ["undata/SEX-F"],
            [_MORT_FEMALE],
            _MORT_NAMES[_MORT_FEMALE],
            id="one-series-reads-as-itself",
        ),
    ],
)
def test_a_resubmit_labels_a_held_family_as_its_first_answer_does(
    bind_catalogue, sexes: list[str], served: list[str], expected: str
):
    """A UN series the catalogue holds reads as its total's name on a first answer that
    serves two by-sex members, and the resubmit of that answer's identity, whose members
    are the by-sex members alone, reads the same. An answer that serves one by-sex member
    reads as that member's name, first and resubmitted."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("undata/p/WHO", UN_SYSTEM, segment="who")],
            un_mirror=un_mirror(
                {
                    _MORT_TOTAL: un_variable(_MORT, name=_MORT_NAMES[_MORT_TOTAL]),
                    **{
                        dcid: un_variable(
                            _MORT,
                            name=_MORT_NAMES[dcid],
                            constraints={"undataSex": ("Sex", *value)},
                        )
                        for dcid, value in _MORT_SEX.items()
                    },
                }
            ),
        )
    )
    query = "under-five mortality by sex in Kenya"
    facet = {"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": label} for dcid, label in _UN_LABELS.items()},
            "country/KEN": {"label": "Kenya", "type": "Country"},
            _MORT: {"label": "Under-five mortality rate"},
            _MORT_TOTAL: _un_member(_MORT_NAMES[_MORT_TOTAL], _MORT, {}),
            **{
                dcid: _un_member(_MORT_NAMES[dcid], _MORT, {"undataSex": value})
                for dcid, (value, _label) in _MORT_SEX.items()
            },
        },
        obs={f"{dcid}|country/KEN": [facet] for dcid in _MORT_NAMES},
        detect={query: {"svs": [_MORT_FEMALE], "cosine_scores": [0.9]}},
        resolve={"Kenya": "country/KEN"},
    )
    sdmx = FakeSdmxClient(observations={}, availability={})

    first = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=sdmx,
            llm=_BindSexLLM(sexes),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )
    (answer,) = first.specs
    resubmit = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=answer.definition.identity),
        graph=graph,
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
        options=walk_options(graph),
    )
    (again,) = resubmit.specs

    assert sorted(sv.ref.ref for sv in answer.resolved.members) == served
    assert sorted(sv.ref.ref for sv in again.resolved.members) == served
    assert answer.resolved.shape.label == expected
    assert again.resolved.shape.label == expected
