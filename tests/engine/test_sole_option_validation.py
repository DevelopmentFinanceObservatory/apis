"""Pins validation of `sole_option` claims against the population of the shape bound.

A claim stands when every member of the family the index carries, or the catalogue holds,
asserts the value. On a recalled sample, a member asserting another value or a total of the
sample holding none leaves the slot unbound.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import pytest

from code_index import shared as code_index_shared
from code_index.query import CodeIndex
from code_index.types import DatasetDeclaration, IndexStamp, StatVarRow
from origin_catalogue import BASE
from search.engine.bind import DraftBindingEvidence, SlotBindingDraft
from search.engine.region.phase_election import (
    _keep_the_noun_pinned_property,
    _settle_sole_options,
    _unbind_implied_values,
    _validate_sole_option_bindings,
)
from search.engine.region.unrealisable import _with_forced_values
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue
from tests.origin_catalogue.support import build_arc_tables

_FLOW = "crsTypeOfFlow"


def _shape(pop_type: str = "FinancialFlow") -> ShapeDraft:
    return ShapeDraft(
        shape_id="financialflow_amount_measuredvalue_nominal",
        label="net disbursements",
        pop_type_dcid=pop_type,
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=_FLOW, label="CRS type of flow"),),
        sv_arc_facts_complete=False,
    )


def _flow(value: str, evidence: DraftBindingEvidence) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid=_FLOW, kind="value", value_dcids=[value], evidence=evidence
    )


def test_a_sole_option_claim_on_an_index_slot_offering_several_values_leaves_it_unbound():
    """ "Net disbursements" names no flow. The recalled members all hold one flow and the
    index's family holds four, so the slot offers four."""
    (validated,) = _validate_sole_option_bindings(
        [_flow("CRSTypeOfFlow21", "sole_option")], _shape()
    )

    assert (validated.kind, validated.value_dcids) == ("unbound", [])
    assert validated.property_dcid == _FLOW


@pytest.mark.parametrize(
    ("evidence", "pop_type"),
    [
        # The phrase named the value.
        ("mentioned", "FinancialFlow"),
        # The shape has no members to check the claim against.
        ("sole_option", "UnindexedFlow"),
    ],
)
def test_a_binding_its_slot_or_the_phrase_supports_stays(evidence, pop_type):
    binding = _flow("CRSTypeOfFlow21", evidence)

    validated = _validate_sole_option_bindings([binding], _shape(pop_type))

    assert validated == [binding]


# ---------------------------------------------------------------------------
# An index family with and without a row lacking the value
# ---------------------------------------------------------------------------

_DATASET = "provenance/SoleOptionValidation"
_OTHER = "provenance/SoleOptionValidationOther"


def _row(dcid: str, *, provenance: str = _DATASET, **constraints: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=("financingProvider",),
        constraint_properties=constraints,
        provenances=(provenance,),
    )


def _use_index(monkeypatch: pytest.MonkeyPatch, *rows: StatVarRow) -> None:
    """Serve an index of `rows` wherever a loaded module reads the fixture index."""
    index = CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="sole-option-validation",
        ),
        declarations=(),
        codes=(),
        statvars=rows,
        datasets=tuple(
            DatasetDeclaration(
                provenance=provenance,
                label=provenance,
                description="A test table.",
                opt_in=False,
                absent_observation="zero",
                account="provider",
                rank=rank,
                invariants={},
            )
            for rank, provenance in enumerate((_DATASET, _OTHER))
        ),
    )
    packaged = code_index_shared.get_index
    for module in list(sys.modules.values()):
        if vars(module).get("get_index") is packaged:
            monkeypatch.setattr(module, "get_index", lambda: index)


@pytest.mark.parametrize(
    ("rows", "kept"),
    [
        pytest.param(
            (
                _row("Flow_Oda", crsTypeOfFlow="CRSTypeOfFlow10"),
                _row("Flow_Oda_Grants", crsTypeOfFlow="CRSTypeOfFlow10", grant="Yes"),
            ),
            True,
            id="every-row-asserts-the-value",
        ),
        pytest.param(
            (_row("Flow_Total"), _row("Flow_Oda", crsTypeOfFlow="CRSTypeOfFlow10")),
            False,
            id="a-total-asserts-none",
        ),
    ],
)
def test_a_sole_option_claim_on_an_index_family_stands_only_when_every_row_asserts_it(
    monkeypatch, rows, kept
):
    """The family offers one flow either way. A total holding no flow leaves the slot a
    reading without one."""
    _use_index(monkeypatch, *rows)
    binding = _flow("CRSTypeOfFlow10", "sole_option")

    (validated,) = _validate_sole_option_bindings([binding], _shape())

    assert (validated == binding) is kept
    assert validated.kind == ("value" if kept else "unbound")
    assert validated.sampled is False


@pytest.mark.parametrize(
    ("named", "kept"),
    [
        pytest.param(None, False, id="an-unnamed-part-under-an-exclusion"),
        pytest.param(_DATASET, True, id="a-part-naming-its-dataset"),
    ],
)
def test_a_sole_option_claim_is_judged_on_the_family_the_identity_denotes(monkeypatch, named, kept):
    """The part answers from one of two tables of the family, the other excluded, so its
    menu offers one flow. A part naming no dataset mints the identity of the whole family,
    where the excluded table's row holds another flow, so the slot stays open. A part
    naming its dataset denotes that table's rows alone, which all hold the flow."""
    _use_index(
        monkeypatch,
        _row("Flow_Oda", crsTypeOfFlow="CRSTypeOfFlow10"),
        _row("Flow_Oof", provenance=_OTHER, crsTypeOfFlow="CRSTypeOfFlow20"),
    )
    shape = replace(_shape(), datasets=(_DATASET,), named_dataset=named)
    binding = _flow("CRSTypeOfFlow10", "sole_option")

    (validated,) = _validate_sole_option_bindings([binding], shape)

    assert (validated.kind, validated.value_dcids) == (
        ("value", ["CRSTypeOfFlow10"]) if kept else ("unbound", [])
    )


# ---------------------------------------------------------------------------
# A family the catalogue holds
# ---------------------------------------------------------------------------

_TOTAL = "LifeExpectancy_Person"
_FEMALE = "LifeExpectancy_Person_Female"
_RESIDENT = "LifeExpectancy_Person_Resident"
_FEMALE_RESIDENT = "LifeExpectancy_Person_Female_Resident"
_MALE = "LifeExpectancy_Person_Male"
_MALE_RESIDENT = "LifeExpectancy_Person_Male_Resident"
_WDI = "dc/base/WorldDevelopmentIndicators"


def _life_expectancy_arcs(**constraints: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Life expectancy"}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "lifeExpectancy"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


_LIFE_EXPECTANCY_ARCS = {
    _TOTAL: _life_expectancy_arcs(),
    _FEMALE: _life_expectancy_arcs(gender="Female"),
    _RESIDENT: _life_expectancy_arcs(residentStatus="Resident"),
    _FEMALE_RESIDENT: _life_expectancy_arcs(gender="Female", residentStatus="Resident"),
    _MALE: _life_expectancy_arcs(gender="Male"),
    _MALE_RESIDENT: _life_expectancy_arcs(gender="Male", residentStatus="Resident"),
}


@pytest.mark.parametrize(
    ("family", "kept"),
    [
        pytest.param((_TOTAL, _FEMALE, _RESIDENT, _FEMALE_RESIDENT), False, id="with-a-total"),
        pytest.param((_RESIDENT, _FEMALE_RESIDENT), True, id="every-member-resident"),
        pytest.param((_FEMALE, _MALE, _FEMALE_RESIDENT, _MALE_RESIDENT), False, id="no-total"),
    ],
)
def test_a_sole_option_claim_on_a_held_family_stands_only_when_every_member_asserts_it(
    bind_catalogue, family, kept
):
    """ "Life expectancy in Kenya" names no resident status. The family the catalogue holds
    offers `Resident` as the slot's one value, but a member holding no resident status
    leaves the slot unforced, whether or not it is the family's total. Recall found one
    member, so only the held family shows the others."""
    arcs = {dcid: _LIFE_EXPECTANCY_ARCS[dcid] for dcid in family}
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={dcid: ([_WDI], []) for dcid in arcs},
            arcs=build_arc_tables(dfo={}, base=arcs, schema={}),
        )
    )
    graph = FakeGraph(
        nodes={dcid: {"label": dcid, "arcs": a} for dcid, a in arcs.items()},
        obs={},
        detect={},
        resolve={},
    )
    (shape,) = derive_shapes(confirmed_svs=[_FEMALE_RESIDENT], graph=graph)
    assert shape.held_family is not None
    binding = SlotBindingDraft(
        axis="what",
        property_dcid="residentStatus",
        kind="value",
        value_dcids=["Resident"],
        evidence="sole_option",
    )

    (validated,) = _validate_sole_option_bindings([binding], shape)

    assert (validated == binding) is kept
    assert validated.kind == ("value" if kept else "unbound")
    assert validated.sampled is False


# ---------------------------------------------------------------------------
# A recalled sample
# ---------------------------------------------------------------------------

_SERVICING = "Amount_FinancialFlow_ServicingPayment_ByCreditor_Nominal"
_CONCESSIONAL = "Amount_FinancialFlow_ServicingPayment_IDSConcessional_ByCreditor_Nominal"
_LONG_TERM = "Amount_FinancialFlow_ServicingPayment_LongTermMaturity_ByCreditor_Nominal"
_LONG_TERM_CONCESSIONAL = (
    "Amount_FinancialFlow_ServicingPayment_LongTermMaturity_IDSConcessional_ByCreditor_Nominal"
)
_SHORT_TERM_CONCESSIONAL = (
    "Amount_FinancialFlow_ServicingPayment_ShortTermMaturity_IDSConcessional_ByCreditor_Nominal"
)
_NET_FLOWS = "Amount_FinancialFlow_Net_Disbursement_PrincipalRepayment_Nominal"
_NET_TRANSFERS = "Amount_FinancialFlow_Net_Disbursement_ServicingPayment_Nominal"
_NET_TRANSFERS_CONCESSIONAL = (
    "Amount_FinancialFlow_Net_Disbursement_ServicingPayment_IDSConcessional_Nominal"
)
_CONCESSION = ("idsConcessionality", "IDSConcessional")
_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GVA_AGRICULTURE = "Amount_EconomicActivity_GrossValueAdded_Agriculture_Nominal"


def _debt_service_arcs(**constraints: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "UnindexedFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "observationProperties": {
            "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
        },
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


@pytest.mark.parametrize(
    ("members", "bound", "claim", "kept"),
    [
        pytest.param(
            {
                _SERVICING: _debt_service_arcs(financialFlowType="ServicingPayment"),
                _CONCESSIONAL: _debt_service_arcs(
                    financialFlowType="ServicingPayment", idsConcessionality="IDSConcessional"
                ),
            },
            {},
            _CONCESSION,
            False,
            id="a-member-asserts-none",
        ),
        pytest.param(
            {
                _CONCESSIONAL: _debt_service_arcs(
                    financialFlowType="ServicingPayment", idsConcessionality="IDSConcessional"
                ),
            },
            {},
            _CONCESSION,
            True,
            id="every-member-asserts-the-value",
        ),
        pytest.param(
            {
                _CONCESSIONAL: _debt_service_arcs(
                    financialFlowType="ServicingPayment", idsConcessionality="IDSConcessional"
                ),
                _LONG_TERM: _debt_service_arcs(
                    financialFlowType="ServicingPayment", maturityBand="LongTermMaturity"
                ),
            },
            {},
            _CONCESSION,
            True,
            id="a-member-narrows-on-another-property",
        ),
        pytest.param(
            {
                _LONG_TERM: _debt_service_arcs(
                    financialFlowType="ServicingPayment", maturityBand="LongTermMaturity"
                ),
                _LONG_TERM_CONCESSIONAL: _debt_service_arcs(
                    financialFlowType="ServicingPayment",
                    maturityBand="LongTermMaturity",
                    idsConcessionality="IDSConcessional",
                ),
                _SHORT_TERM_CONCESSIONAL: _debt_service_arcs(
                    financialFlowType="ServicingPayment",
                    maturityBand="ShortTermMaturity",
                    idsConcessionality="IDSConcessional",
                ),
            },
            {"maturityBand": "LongTermMaturity"},
            _CONCESSION,
            False,
            id="a-bound-member-is-the-total",
        ),
        pytest.param(
            {
                _NET_FLOWS: _debt_service_arcs(
                    financialFlowType="Disbursement", netOf="PrincipalRepayment"
                ),
                _NET_TRANSFERS: _debt_service_arcs(
                    financialFlowType="Disbursement", netOf="ServicingPayment"
                ),
                _NET_TRANSFERS_CONCESSIONAL: _debt_service_arcs(
                    financialFlowType="Disbursement",
                    netOf="ServicingPayment",
                    idsConcessionality="IDSConcessional",
                ),
            },
            {},
            _CONCESSION,
            False,
            id="a-sample-split-on-an-open-slot-holds-a-broader-member",
        ),
        pytest.param(
            {
                _GDP: _debt_service_arcs(activitySource="GrossDomesticProduction"),
                _GVA_AGRICULTURE: _debt_service_arcs(
                    activitySource="GrossValueAdded", economicSector="Agriculture"
                ),
            },
            {},
            ("economicSector", "Agriculture"),
            False,
            id="a-member-asserting-none-splits-on-a-property-both-state",
        ),
    ],
)
def test_a_sole_option_claim_on_a_recalled_sample_falls_to_a_total_lacking_the_value(
    members, bound, claim, kept
):
    """ "Debt service" names no concessionality. A recalled sample is part of the family, so
    its by-creditor total, which states only the flow type every member shares, leaves the
    slot unforced for the family too. A member stating a maturity band in place of a
    concessionality narrows the family elsewhere and shows no reading without the value, as
    "infant mortality rate" binds the age of zero beside members stating an age group
    classification. A bound property does not keep its member from being the total: with
    the long-term maturity band named, the long-term member stating no concessionality is
    the total. A sample that splits on what a flow is net of holds no total, and the net
    transfers member stating no concessionality is the reading without one beside the
    member that states it. "GDP" recalls gross domestic product, which states no economic
    sector, beside the gross value added of agriculture: the two differ on the activity
    source both state, so they are siblings across a split, and the side without the
    sector is a reading of the part without it. The shape holds no held family and the
    index carries none of its rows, so a claim the sample does not refute stands marked
    sampled, for the walk of the family to judge."""
    prop, value = claim
    shape = replace(
        _shape("UnindexedFlow"),
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=prop, label=prop),),
        sv_arc_facts=members,
    )
    assert shape.held_family is None
    binding = SlotBindingDraft(
        axis="how", property_dcid=prop, kind="value", value_dcids=[value], evidence="sole_option"
    )

    stated = [
        SlotBindingDraft(
            axis="how", property_dcid=prop, kind="value", value_dcids=[value], evidence="mentioned"
        )
        for prop, value in bound.items()
    ]

    *_, validated = _validate_sole_option_bindings([*stated, binding], shape)

    assert (validated.kind, validated.value_dcids, validated.sampled) == (
        ("value", [value], True) if kept else ("unbound", [], False)
    )


# ---------------------------------------------------------------------------
# A value the measure's headline series already means
# ---------------------------------------------------------------------------


def _member_arcs(population: str, measured: str, **constraints: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": population}]},
        "measuredProperty": {"nodes": [{"dcid": measured}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


def _measure_shape(population: str, measured: str, members: dict[str, dict]) -> ShapeDraft:
    properties = sorted({prop for arcs in members.values() for prop in _constraints(arcs)})
    return ShapeDraft(
        shape_id=f"{population.lower()}_{measured.lower()}_measuredvalue",
        label=measured,
        pop_type_dcid=population,
        meas_prop_dcid=measured,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=tuple(SlotKeyDraft(axis="how", property_dcid=p, label=p) for p in properties),
        sv_arc_facts=members,
    )


def _constraints(arcs: dict) -> list[str]:
    return [node["dcid"] for node in arcs["constraintProperties"]["nodes"]]


def _life_expectancy(**constraints: str) -> dict:
    return _member_arcs("Person", "lifeExpectancy", **constraints)


def _gdp(**constraints: str) -> dict:
    return _member_arcs("EconomicActivity", "amount", **constraints)


_LIFE_EXPECTANCY_MEMBERS = {
    "LifeExpectancy_Person": _life_expectancy(),
    "LifeExpectancy_Person_Female": _life_expectancy(gender="Female"),
    "LifeExpectancy_Person_0Years": _life_expectancy(age="Years0"),
    "LifeExpectancy_Person_0Years_Female": _life_expectancy(age="Years0", gender="Female"),
    "LifeExpectancy_Person_60Years": _life_expectancy(age="Years60"),
}
_GDP_MEMBERS = {
    "Amount_EconomicActivity_GrossDomesticProduction": _gdp(
        activitySource="GrossDomesticProduction"
    ),
    "Amount_EconomicActivity_GrossDomesticProduction_AllEconomicActivities": _gdp(
        activitySource="GrossDomesticProduction", economicSector="AllEconomicActivities"
    ),
    "Amount_EconomicActivity_GrossDomesticProduction_Agriculture": _gdp(
        activitySource="GrossDomesticProduction", economicSector="Agriculture"
    ),
    "Amount_EconomicActivity_GrossValueAdded_Agriculture": _gdp(
        activitySource="GrossValueAdded", economicSector="Agriculture"
    ),
}
# A family every member of which states all economic activities: its total states the value,
# so the value separates nothing and binding it costs no total.
_SECTOR_TOTAL_MEMBERS = {
    "Amount_EconomicActivity_GrossValueAdded_AllEconomicActivities": _gdp(
        activitySource="GrossValueAdded", economicSector="AllEconomicActivities"
    ),
    "Amount_EconomicActivity_GrossValueAdded_AllEconomicActivities_Basic": _gdp(
        activitySource="GrossValueAdded",
        economicSector="AllEconomicActivities",
        priceBasis="BasicPrice",
    ),
}


def _drafted(
    drafted_by: str = "reply", evidence: DraftBindingEvidence = "mentioned", **values: str
) -> list[SlotBindingDraft]:
    return [
        SlotBindingDraft(
            axis="how",
            property_dcid=prop,
            kind="value",
            value_dcids=[value],
            evidence=evidence,
            drafted_by=drafted_by,
        )
        for prop, value in values.items()
    ]


@pytest.mark.parametrize(
    ("members", "drafted", "released"),
    [
        pytest.param(
            _LIFE_EXPECTANCY_MEMBERS, {"age": "Years0"}, {"age"}, id="life-expectancy-at-birth"
        ),
        pytest.param(
            _LIFE_EXPECTANCY_MEMBERS,
            {"age": "Years0", "gender": "Female"},
            {"age"},
            id="female-life-expectancy-at-birth",
        ),
        # Years60 is no value the index declares life expectancy implies.
        pytest.param(_LIFE_EXPECTANCY_MEMBERS, {"age": "Years60"}, set(), id="undeclared-value"),
        pytest.param(
            _GDP_MEMBERS,
            {
                "activitySource": "GrossDomesticProduction",
                "economicSector": "AllEconomicActivities",
            },
            {"economicSector"},
            id="gdp-of-all-economic-activities",
        ),
        # Gross domestic production is no implied value: value added holds other readings.
        pytest.param(
            _GDP_MEMBERS,
            {"activitySource": "GrossDomesticProduction"},
            set(),
            id="gdp-names-its-activity-source",
        ),
        # With the activity source open, gross value added splits the family: no total.
        pytest.param(
            _GDP_MEMBERS,
            {"economicSector": "AllEconomicActivities"},
            set(),
            id="no-total",
        ),
        pytest.param(
            _SECTOR_TOTAL_MEMBERS,
            {"economicSector": "AllEconomicActivities"},
            set(),
            id="the-total-asserts-the-value",
        ),
    ],
)
@pytest.mark.parametrize("evidence", ["mentioned", "sole_option"])
def test_a_value_the_headline_series_implies_is_released_only_beside_a_total_without_it(
    members, drafted, released, evidence
):
    """ "Life expectancy at birth" binds age 0, which life expectancy already means; the
    total states no age, so the binding elects the age-0 series (1980 to 2020) over it. The
    code index declares the value implied, and the binding is released where the family's
    totals under the other bindings state nothing on the property."""
    population = next(iter(members.values()))["populationType"]["nodes"][0]["dcid"]
    measured = next(iter(members.values()))["measuredProperty"]["nodes"][0]["dcid"]
    shape = _measure_shape(population, measured, members)
    bindings = _drafted(evidence=evidence, **drafted)

    settled = _unbind_implied_values(bindings, shape)

    assert {b.property_dcid for b in settled if b.kind == "unbound"} == released
    assert [b for b in settled if b.property_dcid not in released] == [
        b for b in bindings if b.property_dcid not in released
    ]


def test_a_value_the_engine_bound_is_not_released():
    """Only a value the bind reply drafted is released: the engine's own bindings rest on
    rules of their own."""
    bindings = _drafted(drafted_by="engine", age="Years0")
    shape = _measure_shape("Person", "lifeExpectancy", _LIFE_EXPECTANCY_MEMBERS)

    assert _unbind_implied_values(bindings, shape) == bindings


# ---------------------------------------------------------------------------
# One value drafted on two properties no member holds together
# ---------------------------------------------------------------------------


def _ids_flow(**constraints: str) -> dict:
    return {
        **_member_arcs("FinancialFlow", "amount", **constraints),
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
    }


_DEBT_FLOWS = {
    "Amount_FinancialFlow_ServicingPayment_Nominal": _ids_flow(
        financialFlowType="ServicingPayment"
    ),
    "Amount_FinancialFlow_Net_Disbursement_ServicingPayment_Nominal": _ids_flow(
        financialFlowType="Disbursement", netOf="ServicingPayment", nettingBasis="Net"
    ),
    "Amount_FinancialFlow_Net_Disbursement_PrincipalRepayment_Nominal": _ids_flow(
        financialFlowType="Disbursement", netOf="PrincipalRepayment", nettingBasis="Net"
    ),
}


def _debt_shape(members: dict[str, dict]) -> ShapeDraft:
    return replace(
        _measure_shape("FinancialFlow", "amount", members),
        shape_id="financialflow_amount_measuredvalue_nominal",
        meas_qual_dcid="Nominal",
    )


@pytest.mark.parametrize(
    ("members", "drafted", "settled"),
    [
        pytest.param(
            _DEBT_FLOWS,
            {"financialFlowType": "ServicingPayment", "netOf": "ServicingPayment"},
            {"financialFlowType": "ServicingPayment"},
            id="external-debt-service-on-both-properties",
        ),
        # A member holds both values, so the draft excludes nothing.
        pytest.param(
            {
                **_DEBT_FLOWS,
                "Amount_FinancialFlow_ServicingPayment_NetOfServicingPayment": _ids_flow(
                    financialFlowType="ServicingPayment", netOf="ServicingPayment"
                ),
            },
            {"financialFlowType": "ServicingPayment", "netOf": "ServicingPayment"},
            {"financialFlowType": "ServicingPayment", "netOf": "ServicingPayment"},
            id="a-member-holds-both",
        ),
        # "Net transfers on external debt" binds what the transfers are net of, with the
        # flow type Disbursement: one value on one property, which net transfers hold.
        pytest.param(
            _DEBT_FLOWS,
            {
                "financialFlowType": "Disbursement",
                "netOf": "ServicingPayment",
                "nettingBasis": "Net",
            },
            {
                "financialFlowType": "Disbursement",
                "netOf": "ServicingPayment",
                "nettingBasis": "Net",
            },
            id="net-transfers-on-external-debt",
        ),
    ],
)
def test_a_value_drafted_on_two_properties_keeps_the_one_a_measure_noun_pins(
    members, drafted, settled
):
    """ "External debt service" names `ServicingPayment` as the flow type, and net transfers
    are net of it too. A reply binding it on both excludes every member, and the answer
    read without one refused the term. The code index's noun for external debt service pins
    the flow type; the one for net transfers pins `netOf` beside the flow type
    `Disbursement`, which the draft contradicts. So the flow type keeps the value and
    `netOf` is unbound, and the part answers the debt service total."""
    shape = _debt_shape(members)

    result = _keep_the_noun_pinned_property(_drafted(**drafted), shape)

    assert {b.property_dcid: b.value_dcids[0] for b in result if b.kind == "value"} == settled
    assert {b.property_dcid for b in result} == set(drafted)


# ---------------------------------------------------------------------------
# A single-member slot the reply leaves unbound
# ---------------------------------------------------------------------------


def _unbound(prop: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid=prop, kind="unbound", value_dcids=[], drafted_by="reply"
    )


_INFANT_MEMBERS = {
    "Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth": _member_arcs(
        "UnindexedMortalityEvent", "count", age="Years0"
    ),
    "Count_MortalityEvent_Infant_AsAFractionOf_Count_BirthEvent_LiveBirth": _member_arcs(
        "UnindexedMortalityEvent", "count", ageGroupClassification="Infant"
    ),
    "Count_MortalityEvent_Newborn_AsAFractionOf_Count_BirthEvent_LiveBirth": _member_arcs(
        "UnindexedMortalityEvent", "count", ageGroupClassification="Newborn"
    ),
}
_MALE_MEMBERS = {"Count_Person_Male": _member_arcs("UnindexedPerson", "count", gender="Male")}
_POPULATION_MEMBERS = {
    "Count_Person": _member_arcs("UnindexedPerson", "count"),
    "Count_Person_Female": _member_arcs("UnindexedPerson", "count", gender="Female"),
}


@pytest.mark.parametrize(
    ("population", "members", "taxonomy", "bound"),
    [
        pytest.param(
            "UnindexedMortalityEvent",
            _INFANT_MEMBERS,
            {"how:age": ["Years0"], "how:ageGroupClassification": ["Infant", "Newborn"]},
            {"age": "Years0"},
            id="infant-mortality-binds-the-one-age",
        ),
        pytest.param(
            "UnindexedPerson",
            _POPULATION_MEMBERS,
            {"how:gender": ["Female"]},
            {},
            id="population-sample-of-one-gender-stays-unbound",
        ),
    ],
)
def test_a_single_member_slot_the_reply_leaves_unbound_binds_where_the_population_forces_it(
    population, members, taxonomy, bound
):
    """The bind prompt asks the reply to bind a slot whose taxonomy holds one value, and the
    recorded reply to "infant mortality rate" leaves `age` unbound over the one value
    `Years0`. The engine claims it as a `sole_option` value and the population judges the
    claim. Beside members stating an age group classification instead, the age of zero is
    forced, while the two-value age group classification stays unbound. A population sample
    holding one gender beside its total forces none, and the slot stays as the reply
    drafted it."""
    shape = replace(_measure_shape(population, "count", members), slot_taxonomy=taxonomy)
    drafted = [_unbound(key.removeprefix("how:")) for key in taxonomy]

    settled = _settle_sole_options(drafted, shape)

    assert {b.property_dcid: b.value_dcids[0] for b in settled if b.kind == "value"} == bound
    assert all(b.evidence == "sole_option" for b in settled if b.kind == "value")
    assert [b for b in settled if b.kind != "value"] == [
        b for b in drafted if b.property_dcid not in bound
    ]


@pytest.mark.parametrize(
    ("released", "bound"),
    [
        pytest.param(("Male",), [], id="released-value"),
        pytest.param(("Male", "Female"), [], id="released-in-a-set"),
        pytest.param(("Female",), ["Male"], id="another-value-released"),
    ],
)
def test_an_unbinding_binds_no_value_it_released_again(released, bound):
    """A single-member slot the unbinding opens takes the value its population forces,
    unless the unbinding released that value from it, alone or in a set: no member holds
    it beside the rest of the bindings."""
    shape = replace(
        _measure_shape("UnindexedPerson", "count", _MALE_MEMBERS),
        slot_taxonomy={"how:gender": ["Male"]},
    )

    (settled,) = _with_forced_values([_unbound("gender")], shape, {"gender": released})

    assert settled.value_dcids == bound
