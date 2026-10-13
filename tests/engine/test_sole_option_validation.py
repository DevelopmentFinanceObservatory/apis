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
from search.engine.bind import SlotBindingDraft
from search.engine.region.phase_rank_bind import _validate_sole_option_bindings
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


def _flow(value: str, evidence: str) -> SlotBindingDraft:
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


def _row(dcid: str, **constraints: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=("financingProvider",),
        constraint_properties=constraints,
        provenances=(_DATASET,),
    )


def _use_index(monkeypatch: pytest.MonkeyPatch, *rows: StatVarRow) -> None:
    """Serve an index of `rows` wherever a loaded module reads the packaged index."""
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
        datasets=(
            DatasetDeclaration(
                provenance=_DATASET,
                label=_DATASET,
                description="A test table.",
                opt_in=False,
                absent_observation="zero",
                account="provider",
                rank=0,
                invariants={},
            ),
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
    ("members", "bound", "kept"),
    [
        pytest.param(
            {
                _SERVICING: _debt_service_arcs(financialFlowType="ServicingPayment"),
                _CONCESSIONAL: _debt_service_arcs(
                    financialFlowType="ServicingPayment", idsConcessionality="IDSConcessional"
                ),
            },
            {},
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
            False,
            id="a-bound-member-is-the-total",
        ),
    ],
)
def test_a_sole_option_claim_on_a_recalled_sample_falls_to_a_total_lacking_the_value(
    members, bound, kept
):
    """ "Debt service" names no concessionality. A recalled sample is part of the family, so
    its by-creditor total, which states only the flow type every member shares, leaves the
    slot unforced for the family too. A member stating a maturity band in place of a
    concessionality narrows the family elsewhere and shows no reading without the value, as
    "infant mortality rate" binds the age of zero beside members stating an age group
    classification. A bound property does not keep its member from being the total: with
    the long-term maturity band named, the long-term member stating no concessionality is
    the total. The shape holds no held family and the index carries none of its rows."""
    shape = replace(
        _shape("UnindexedFlow"),
        slot_keys=(
            SlotKeyDraft(
                axis="how", property_dcid="idsConcessionality", label="IDS concessionality"
            ),
        ),
        sv_arc_facts=members,
    )
    assert shape.held_family is None
    binding = SlotBindingDraft(
        axis="how",
        property_dcid="idsConcessionality",
        kind="value",
        value_dcids=["IDSConcessional"],
        evidence="sole_option",
    )

    stated = [
        SlotBindingDraft(
            axis="how", property_dcid=prop, kind="value", value_dcids=[value], evidence="mentioned"
        )
        for prop, value in bound.items()
    ]

    *_, validated = _validate_sole_option_bindings([*stated, binding], shape)

    assert (validated == binding) is kept
    assert validated.kind == ("value" if kept else "unbound")
