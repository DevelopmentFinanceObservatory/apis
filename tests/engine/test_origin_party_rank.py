"""A shape that publishes the organisation a part names as the origin of its flow leads
the seated menu among the shapes measuring alike (`phase_rank_bind._promoted`).

The shapes are built from rows of the recorded dev code index. IDA publishes on the origin
axis in the CRS, DAC2a and DAC2b, which give the provider's account of their flows, and in
World Bank IDS, which the index lists without a declared dataset. Recall ranks IDS's series
of IDA grants, debt service and concessional loans above DAC2a's disbursements, as Studio
saw it for "IDA disbursements".
"""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace

import pytest

from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from search.engine.origin_words import owns
from search.engine.region import phase_rank_bind
from search.engine.region.phase_rank_bind import _origin_party, _OriginParty, _promoted
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from tests.code_index.support import INDEX_FIXTURE, load_index, load_measure_axes
from tests.engine.test_expansion_rank_order import _STRONG_ROOT as _ROOT
from tests.engine.test_expansion_rank_order import _run_prune
from tests.engine.test_expansion_rank_order import _shape as _prune_shape
from tests.graph_release.support import serving_index

_IDA = "InternationalDevelopmentAssociation"
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_DAC2A = "provenance/OECDDAC2A"
_NAME = "International Development Association"


@cache
def _recorded() -> tuple[CodeIndex, MeasureIndex]:
    """The recorded dev code index and its measure axes."""
    return load_index(INDEX_FIXTURE), load_measure_axes(INDEX_FIXTURE)


def _shape(shape_id: str, dcid: str, score: float) -> ShapeDraft:
    """A shape of the recorded index's row `dcid`, recalled at `score`."""
    row = _recorded()[0].statvar_by_dcid(dcid)
    assert row is not None, dcid
    constraints = row.constraint_properties
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=row.population_type,
        meas_prop_dcid=row.measured_property,
        stat_type_dcid=row.stat_type,
        meas_qual_dcid=row.measurement_qualifier,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            dcid: {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "populationType": {"nodes": [{"dcid": row.population_type}]},
                "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
                "statType": {"nodes": [{"dcid": row.stat_type}]},
                "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
                "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
                **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
                "observationProperties": {
                    "nodes": [{"dcid": role} for role in row.observation_properties]
                },
            }
        },
        representative_score=score,
    )


# IDS's IDA grants (DT.DIS.IDAG.CD) and concessional loans (DT.DIS.MIDA.CD), both listed rows.
_IDS_GRANTS = _shape(
    "ids_grants", "Amount_FinancialFlow_Disbursement_UnrequitedTransfer_Nominal", 0.70
)
_IDS_LOANS = _shape(
    "ids_loans", "Amount_FinancialFlow_Disbursement_DebtInstrument_IDSConcessional_Nominal", 0.60
)
_IDS_DEBT_SERVICE = _shape(
    "ids_debt_service", "Amount_FinancialFlow_ServicingPayment_DebtInstrument_Nominal", 0.65
)
_DAC2A_GROSS = _shape(
    "dac2a_gross", "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal", 0.55
)
_DAC2A_NET = _shape(
    "dac2a_net", "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal", 0.50
)
_DAC2A_DEPOSITS = _shape(
    "dac2a_deposits", "Amount_FinancialFlow_Disbursement_DAC2aAidType210_Nominal", 0.50
)
# The seated menu, in recall order: grants, debt service, loans, DAC2a.
_MENU = (_IDS_GRANTS, _IDS_DEBT_SERVICE, _IDS_LOANS, _DAC2A_GROSS)


def _state(*, kind: str = "organisation", axis: str = "origin", door: str = "parsed"):
    return SimpleNamespace(
        door=door,
        variable="IDA disbursements",
        place_sets={
            _NAME: PlaceSet(
                name=_NAME,
                kind="place",
                node=PlaceNode(dcid=_IDA, graph_types=("Organization",), kind=kind),
            )
        },
        entity_roles={_NAME: [DoorRole(axis=axis, span="IDA")]},
        entity_words={_NAME: "IDA"},
    )


def _promoted_ids(state, menu=_MENU) -> list[str]:
    with serving_index(*_recorded()):
        return [shape.shape_id for shape in _promoted(_origin_party(state), menu)]


@pytest.mark.parametrize(
    ("menu", "promoted"),
    [
        pytest.param(
            _MENU,
            ["dac2a_gross", "ids_debt_service", "ids_grants", "ids_loans"],
            id="debt-service-keeps-its-slot",
        ),
        pytest.param(
            (_IDS_GRANTS, _IDS_LOANS, _DAC2A_DEPOSITS),
            ["dac2a_deposits", "ids_grants", "ids_loans"],
            id="capital-subscriptions",
        ),
        pytest.param(
            (_IDS_GRANTS, _DAC2A_GROSS, _DAC2A_NET),
            ["dac2a_gross", "dac2a_net", "ids_grants"],
            id="gross-and-net",
        ),
    ],
)
def test_the_provider_account_table_leads_the_shapes_measuring_alike(menu, promoted):
    """DAC2a gives IDA's own account of its disbursements, so its shapes take the first
    slots of the shapes measuring alike, held by IDS's grants and loans, which IDS lists for
    IDA as a creditor of its debtors' reports. IDS debt service measures otherwise and keeps
    its slot. DAC2a's gross and net disbursements measure apart from each other and alike
    with the IDS grants, so the three are one set and both DAC2a shapes lead it."""
    assert _promoted_ids(_state(), menu) == promoted


def test_a_dataset_declaring_no_account_publishes_no_origin_organisation():
    """IDS publishes IDA on the origin axis, and the index declares no account for it, so
    of IDA's origin publishers the part keeps those giving the provider's account."""
    with serving_index(*_recorded()):
        published = _origin_party(_state()).published
    assert (_DAC2A, "origin") in published
    assert (_IDS, "origin") not in published


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(_state(axis="destination"), id="organisation-as-recipient"),
        pytest.param(_state(kind="country"), id="not-an-organisation"),
        pytest.param(_state(door="raw_text"), id="raw-text-door"),
    ],
)
def test_the_menu_stands_without_an_origin_organisation(state):
    """A part naming no organisation as the origin of its flow keeps its seated order, and
    its origin organisations own no word of its variable. The raw-text door names none: its
    roles are the extraction's reading, not the caller's."""
    assert _promoted_ids(state) == [shape.shape_id for shape in _MENU]
    with serving_index(*_recorded()):
        assert _origin_party(state).owned == ()


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
