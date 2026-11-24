"""A part naming an organisation as the origin of its flow keeps, among the flows, only the
tables that publish it in the provider's account (`phase_rank_bind._from_origin_publishers`).

The shapes are built from rows of the recorded dev code index, in the pool the dev graph
recalls for "disbursements" with IDA the provider. IDA publishes on the origin axis in the
CRS, DAC2a and DAC2b, which give the provider's account of their flows, and in World Bank
IDS, which the index lists without a declared dataset. Recall ranks IDS's disbursements by
role above the DAC2a, DAC1 and DAC2b families and the DAC1 grant equivalents.
"""

from __future__ import annotations

import asyncio
from functools import cache
from typing import cast

import pytest

from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.origin_words import owns
from search.engine.region.phase_rank_bind import _origin_party, _phase_prune_and_rank
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.region.state import ResolveState
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from search.models import EntryPath
from tests.code_index.support import INDEX_FIXTURE, load_index, load_measure_axes
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._quantity import QuantityReadOnlyLLM
from tests.graph_release.support import serving_index

_IDA = "InternationalDevelopmentAssociation"
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_DAC2A = "provenance/OECDDAC2A"


@cache
def _recorded() -> tuple[CodeIndex, MeasureIndex]:
    """The recorded dev code index and its measure axes."""
    return load_index(INDEX_FIXTURE), load_measure_axes(INDEX_FIXTURE)


def _arcs(dcid: str) -> dict:
    """The graph arcs of the recorded index's row `dcid`."""
    row = _recorded()[0].statvar_by_dcid(dcid)
    assert row is not None, dcid
    constraints = row.constraint_properties
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
        "observationProperties": {"nodes": [{"dcid": role} for role in row.observation_properties]},
    }


def _shape(shape_id: str, members: tuple[str, ...], score: float) -> ShapeDraft:
    """A shape of the recorded index's rows `members`, recalled at `score`, declaring the
    roles any member declares, those some member lacks optional."""
    rows = [_recorded()[0].statvar_by_dcid(dcid) for dcid in members]
    assert all(rows), members
    row = rows[0]
    roles = tuple(dict.fromkeys(role for r in rows for role in r.observation_properties))
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=row.population_type,
        meas_prop_dcid=row.measured_property,
        stat_type_dcid=row.stat_type,
        meas_qual_dcid=row.measurement_qualifier,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={dcid: _arcs(dcid) for dcid in members},
        representative_score=score,
        declared_roles=roles,
        optional_roles=frozenset(
            role for role in roles if any(role not in r.observation_properties for r in rows)
        ),
    )


def _pool() -> list[ShapeDraft]:
    """The pool the dev graph recalls for "disbursements" with IDA the provider."""
    return [
        _shape(
            "ids_by_role",
            (
                "Amount_FinancialFlow_Disbursement_DebtInstrument_Nominal",
                "Amount_FinancialFlow_Disbursement_IDSCreditorTypePROP_Nominal",
                "Amount_FinancialFlow_Disbursement_UnrequitedTransfer_Nominal",
            ),
            0.682,
        ),
        _shape(
            "dac2a",
            (
                "Amount_FinancialFlow_Disbursement_DAC2aAidType210_Nominal",
                "Amount_FinancialFlow_Disbursement_DAC2aAidType211_Nominal",
            ),
            0.6655,
        ),
        _shape(
            "dac1", ("Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1100_Nominal",), 0.6647
        ),
        _shape(
            "dac2b",
            (
                "Amount_FinancialFlow_Gross_Disbursement_DAC2bAidType972_Nominal",
                "Amount_FinancialFlow_Net_Disbursement_DAC2bAidType296_Nominal",
            ),
            0.6495,
        ),
        _shape(
            "dac1_grant_equivalent",
            ("GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1621_Nominal",),
            0.644,
        ),
    ]


def _state(
    *, kind: str = "organisation", axis: str = "origin", door: EntryPath = "parsed"
) -> ResolveState:
    """A part posting "disbursements" with IDA in the role `axis` on `door`, IDA resolving
    to a node of `kind`, its pool the dev graph's (`_pool`)."""
    return ResolveState(
        variable="disbursements",
        entities=["IDA"],
        date_request=None,
        detect_query="disbursements",
        role_query="disbursements",
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
        shapes_found=_pool(),
        coverage=empty_coverage_snapshot(),
        door=door,
        place_sets={
            "IDA": PlaceSet(
                name="IDA",
                kind="place",
                node=PlaceNode(dcid=_IDA, graph_types=("Organization",), kind=kind),
            )
        },
        entity_roles={"IDA": [DoorRole(axis=axis, span="IDA")]},
        entity_words={"IDA": "IDA"},
    )


def _menu(state: ResolveState) -> list[str]:
    """The shape ids of the menu the rank phase seats for `state`."""
    with serving_index(*_recorded()):
        result = asyncio.run(_phase_prune_and_rank(state))
    assert result is None, result
    return [shape.shape_id for shape in state.capped_shapes]


def test_the_provider_account_tables_are_the_only_flows_left():
    """DAC2a and DAC2b give IDA's own account of its disbursements. IDS lists IDA as a
    creditor of its debtors' reports and DAC1 publishes no IDA, so their flows leave before
    the family election, which would otherwise keep the IDS shape on its recall score."""
    assert _menu(_state()) == ["dac2a", "dac2b"]


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(_state(axis="destination"), id="organisation-as-recipient"),
        pytest.param(_state(kind="country"), id="not-an-organisation"),
        pytest.param(_state(door="raw_text"), id="raw-text-door"),
    ],
)
def test_the_flows_stand_without_an_origin_organisation(state):
    """A part naming no organisation as the origin of its flow keeps every flow the family
    election keeps, the IDS shape first, and its origin organisations own no word of its
    variable. The raw-text door names none: its roles are the extraction's reading, not the
    caller's."""
    assert _menu(state)[0] == "ids_by_role"
    with serving_index(*_recorded()):
        assert _origin_party(state).owned == ()


def test_a_dataset_declaring_no_account_publishes_no_origin_organisation():
    """IDS publishes IDA on the origin axis, and the index declares no account for it, so
    of IDA's origin publishers the part keeps those giving the provider's account."""
    with serving_index(*_recorded()):
        published = _origin_party(_state()).published
    assert (_DAC2A, "origin") in published
    assert (_IDS, "origin") not in published


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
