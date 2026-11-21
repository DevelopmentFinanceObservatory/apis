"""A shape that publishes the organisation a part names as the origin of its flow leads
the seated menu among the shapes measuring alike (`phase_rank_bind._promoted`).

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
from search.engine.origin_words import owns
from search.engine.region import phase_rank_bind
from search.engine.region.phase_rank_bind import _origin_party, _OriginParty, _promoted
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from tests.engine.test_expansion_rank_order import _STRONG_ROOT as _ROOT
from tests.engine.test_expansion_rank_order import _run_prune
from tests.engine.test_expansion_rank_order import _shape as _prune_shape
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
_DAC2A_NET_ROW = StatVarRow(
    dcid="TESTFIXTURE/Party/Dac2aNetDisbursement",
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
    measurement_denominator=None,
    observation_properties=_ROLES,
    constraint_properties={"financialFlowType": "Disbursement", "dac2aAidType": "TotalNet"},
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
        statvars=(_DAC2A_ROW, _DAC2A_NET_ROW),
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
_IDS_LOANS = _shape(
    "ids_a_loans",
    "TESTFIXTURE/Party/IdsIdaLoans",
    {"financialFlowType": "Disbursement", "sourceIndicatorCode": "DT.DIS.MIDA.CD"},
    0.60,
)
_IDS_DEBT_SERVICE = _shape(
    "ids_debt_service",
    "TESTFIXTURE/Party/IdsDebtService",
    {"financialFlowType": "DebtService", "sourceIndicatorCode": "DT.TDS.MIDA.CD"},
    0.65,
)
_DAC2A_SHAPE = _shape("dac2a", _DAC2A_ROW.dcid, dict(_DAC2A_ROW.constraint_properties), 0.55)
_DAC2A_NET_SHAPE = _shape(
    "dac2a_net", _DAC2A_NET_ROW.dcid, dict(_DAC2A_NET_ROW.constraint_properties), 0.50
)
# The seated menu, in recall order: grants, debt service, loans, DAC2a.
_MENU = (_IDS_GRANTS, _IDS_DEBT_SERVICE, _IDS_LOANS, _DAC2A_SHAPE)
_NAME = "International Development Association"


def _state(*, kind: str = "organisation", axis: str = "origin", stated: bool = True):
    return SimpleNamespace(
        variable="IDA disbursements",
        place_sets={
            _NAME: PlaceSet(
                name=_NAME,
                kind="place",
                node=PlaceNode(dcid=_IDA, graph_types=("Organization",), kind=kind),
            )
        },
        entity_roles={_NAME: [DoorRole(axis=axis, span="IDA", stated=stated)]},
        entity_words={_NAME: "IDA"},
    )


def _promoted_ids(state, menu=_MENU) -> list[str]:
    with serving_index(_index()):
        return [shape.shape_id for shape in _promoted(_origin_party(state), menu)]


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(_state(), id="stated-provider"),
        pytest.param(_state(stated=False), id="unstated-origin"),
    ],
)
def test_the_table_publishing_the_origin_organisation_takes_its_measures_first_slot(state):
    """DAC2a publishes IDA as a provider, so it takes the first slot, held by the IDS
    disbursement of IDA grants. IDS debt service measures otherwise and keeps the second
    slot, and the two IDS disbursements follow in their order in the slots left. A raw-text
    provider whose cue states no direction, IDA's own name, counts as the origin the
    extraction read."""
    assert _promoted_ids(state) == ["dac2a", "ids_debt_service", "ids_grants", "ids_a_loans"]


def test_every_publisher_of_a_measure_leads_the_shapes_measuring_alike():
    """DAC2a's gross and net disbursements both publish IDA and both measure as the IDS
    disbursement of IDA grants does, so the three are one set: the two publishers take its
    first two slots in their order and the IDS grants follow."""
    menu = (_IDS_GRANTS, _DAC2A_SHAPE, _DAC2A_NET_SHAPE)
    assert _promoted_ids(_state(), menu) == ["dac2a", "dac2a_net", "ids_grants"]


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(_state(axis="destination"), id="organisation-as-recipient"),
        pytest.param(_state(kind="country"), id="not-an-organisation"),
    ],
)
def test_the_menu_stands_without_an_origin_organisation(state):
    """A part naming no organisation as the origin of its flow keeps its seated order."""
    assert _promoted_ids(state) == [shape.shape_id for shape in _MENU]


def test_a_publisher_seated_as_a_topic_leaf_still_leads_the_menu(monkeypatch):
    """The publisher arrives as the leaf of a topic root, which seating stands behind every
    scored shape (`seat_menu`). Promotion is the menu's last ordering step, so the leaf
    leads the bind menu all the same. The leaf publishes the part's origin organisation and
    measures as the scored shape does, as the fakes below say."""
    scored = _prune_shape("scored", score=0.9)
    leaf = _prune_shape("leaf", score=None, rank=0.85, root=_ROOT)

    class _LeafPublishes(_OriginParty):
        def publishes(self, shape):
            return shape.shape_id == "leaf"

    monkeypatch.setattr(
        phase_rank_bind,
        "_origin_party",
        lambda state: _LeafPublishes(organisations={}, published=frozenset(), owned=()),
    )
    monkeypatch.setattr(phase_rank_bind, "_measures_alike", lambda shape, other: True)

    assert [shape.shape_id for shape in _run_prune([scored, leaf])] == ["leaf", "scored"]


@pytest.mark.parametrize(
    ("start", "end", "owns_it"),
    [
        pytest.param(0, 3, True, id="owned-word"),
        pytest.param(4, 5, False, id="no-word"),
        pytest.param(6, 9, False, id="unowned-word"),
    ],
)
def test_owned_ranges_own_a_span_only_by_its_words(start, end, owns_it):
    """A span is owned when it holds a word and the owned ranges cover every one; a span
    holding no word, as the dash in "IDA - aid", is owned by nothing."""
    assert owns("IDA - aid", ((0, 5),), start, end) is owns_it
