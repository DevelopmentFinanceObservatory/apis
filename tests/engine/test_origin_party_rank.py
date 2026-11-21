"""A shape that publishes the organisation a part names as the origin of its flow ranks
first among the shapes measuring alike (`phase_rank_bind._party_ranked`).

The index holds one DAC2a disbursement row, which publishes IDA as a provider. Two series
the index does not carry stand beside it: an IDS disbursement of IDA grants, measuring
alike, and an IDS debt service, measuring otherwise. Recall ranks both IDS series above
DAC2a.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from code_index.query import CodeIndex
from code_index.types import (
    CodeSystemDeclaration,
    DatasetDeclaration,
    IndexStamp,
    PartyRow,
    StatVarRow,
)
from search.engine.region.phase_rank_bind import _party_ranked
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from tests.graph_release.support import serving_index

_DAC2A = "provenance/OECDDAC2A"
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_IDA = "InternationalDevelopmentAssociation"
_ROLES = ("financingProvider", "financingRecipient")

_DAC2A_ROW = StatVarRow(
    dcid="TESTFIXTURE/Party/Dac2aDisbursement",
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
    measurement_denominator=None,
    observation_properties=_ROLES,
    constraint_properties={"financialFlowType": "Disbursement", "dac2aAidType": "Total"},
    provenances=(_DAC2A,),
)


def _index() -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={"dac2aAidType": 0},
            statvar_shape_counts={},
            content_hash="origin-party-rank",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system="dac2aAidType",
                hierarchical=False,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="aid type",
            ),
        ),
        codes=(),
        statvars=(_DAC2A_ROW,),
        datasets=(
            DatasetDeclaration(
                provenance=_DAC2A,
                label="OECD DAC2a",
                description="A test table.",
                opt_in=False,
                absent_observation="unknown",
                account="provider",
                rank=0,
                invariants={},
            ),
        ),
        parties=(
            PartyRow(
                dcid=_IDA,
                kind="party",
                published=((_DAC2A, "origin"), (_IDS, "origin")),
            ),
        ),
    )


def _arcs(constraints: dict[str, str]) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
    }


def _shape(shape_id: str, dcid: str, constraints: dict[str, str], score: float) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={dcid: _arcs(constraints)},
        representative_score=score,
    )


_IDS_GRANTS = _shape(
    "ids_grants",
    "TESTFIXTURE/Party/IdsIdaGrants",
    {"financialFlowType": "Disbursement", "sourceIndicatorCode": "DT.DIS.IDAG.CD"},
    0.70,
)
_IDS_DEBT_SERVICE = _shape(
    "ids_debt_service",
    "TESTFIXTURE/Party/IdsDebtService",
    {"financialFlowType": "DebtService", "sourceIndicatorCode": "DT.TDS.MIDA.CD"},
    0.65,
)
_DAC2A_SHAPE = _shape("dac2a", _DAC2A_ROW.dcid, dict(_DAC2A_ROW.constraint_properties), 0.55)
_SHAPES = (_IDS_GRANTS, _IDS_DEBT_SERVICE, _DAC2A_SHAPE)


def _state(*, kind: str = "organisation", axis: str = "origin", stated: bool = True):
    name = "International Development Association"
    return SimpleNamespace(
        place_sets={
            name: PlaceSet(
                name=name,
                kind="place",
                node=PlaceNode(dcid=_IDA, graph_types=("Organization",), kind=kind),
            )
        },
        entity_roles={name: [DoorRole(axis=axis, span="IDA", stated=stated)]},
    )


def _ranked(state) -> list[str]:
    with serving_index(_index()):
        return [shape.shape_id for shape in _party_ranked(state, _SHAPES)]


def test_the_table_publishing_the_origin_organisation_ranks_first_among_its_measure():
    """DAC2a publishes IDA as a provider, so it ranks above the IDS disbursement of IDA
    grants that recall scored higher; IDS debt service measures otherwise and keeps its
    place."""
    assert _ranked(_state()) == ["ids_debt_service", "dac2a", "ids_grants"]


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(_state(axis="destination"), id="organisation-as-recipient"),
        pytest.param(_state(stated=False), id="role-not-stated"),
        pytest.param(_state(kind="country"), id="not-an-organisation"),
    ],
)
def test_recall_order_stands_without_a_stated_origin_organisation(state):
    """A part naming no organisation as the stated origin of its flow ranks by recall."""
    assert _ranked(state) == ["ids_grants", "ids_debt_service", "dac2a"]
