"""A table publishing the organisation a part names as the origin of its flow in the
provider's account replaces the flows that cannot publish it and those measuring as it does
(`phase_rank_bind._from_origin_publishers`).

The shapes are built from rows of the recorded dev code index, in the pool the dev graph
recalls for "disbursements" with IDA the provider. IDA publishes on the origin axis in the
CRS, DAC2a and DAC2b, which give the provider's account of their flows, and in World Bank
IDS, which the index lists without a declared dataset. DAC1 publishes no IDA. Recall ranks
IDS's disbursements by role above the DAC2a, DAC1 and DAC2b families and the DAC1 grant
equivalents.
"""

from __future__ import annotations

import asyncio
import dataclasses
from functools import cache
from typing import cast

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import MeasureIndex
from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.interpretation import Interpretation
from search.engine.origin_words import owns
from search.engine.region import phase_rank_bind
from search.engine.region.phase_rank_bind import (
    _from_origin_publishers,
    _origin_party,
    _OriginParty,
    _phase_prune_and_rank,
    _refused_on_pinned,
)
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.region.state import ResolveState
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from search.engine.source_scope import SourceScope
from search.models import EntryPath
from tests.code_index.support import INDEX_FIXTURE, load_index, load_measure_axes
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._quantity import QuantityReadOnlyLLM
from tests.engine._source_scopes import part_scope
from tests.engine.test_expansion_rank_order import _ROLE_BEARING, _run_prune
from tests.engine.test_expansion_rank_order import _STRONG_ROOT as _ROOT
from tests.engine.test_expansion_rank_order import _shape as _prune_shape
from tests.graph_release.support import serving_index

_IDA = "InternationalDevelopmentAssociation"
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_DAC2A = "provenance/OECDDAC2A"
_VARIABLE = "IDA disbursements"


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
    *,
    kind: str = "organisation",
    axis: str = "origin",
    door: EntryPath = "parsed",
    scope: SourceScope | None = None,
) -> ResolveState:
    """A part posting "IDA disbursements" with IDA in the role `axis` on `door` under
    `scope`, IDA resolving to a node of `kind`, its pool the dev graph's (`_pool`)."""
    return ResolveState(
        variable=_VARIABLE,
        entities=["IDA"],
        date_request=None,
        detect_query=_VARIABLE,
        role_query=_VARIABLE,
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
        scope=scope,
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


@pytest.fixture(autouse=True)
def _recorded_index():
    """Every test reads the recorded dev code index."""
    with serving_index(*_recorded()):
        yield


def _menu(state: ResolveState) -> list[str]:
    """The shape ids of the menu the rank phase seats for `state`."""
    result = asyncio.run(_phase_prune_and_rank(state))
    assert result is None, result
    return [shape.shape_id for shape in state.capped_shapes]


def test_the_provider_account_tables_answer_the_named_provider():
    """DAC2a and DAC2b give IDA's own account of its disbursements. IDS's disbursements
    measure as DAC2a's do, and DAC1 publishes no IDA, so their flows leave before the family
    election, which would otherwise keep the IDS shape on its recall score."""
    assert _menu(_state()) == ["dac2a", "dac2b"]


@pytest.mark.parametrize(
    ("state", "menu"),
    [
        pytest.param(_state(axis="destination"), ["ids_by_role"], id="organisation-as-recipient"),
        pytest.param(
            _state(kind="country"),
            ["ids_by_role", "dac1_grant_equivalent"],
            id="not-an-organisation",
        ),
        pytest.param(
            _state(door="raw_text"), ["ids_by_role", "dac1_grant_equivalent"], id="raw-text-door"
        ),
    ],
)
def test_the_flows_stand_without_an_origin_organisation(state, menu):
    """A part naming no organisation as the origin of its flow keeps every flow the family
    election and the axis rules keep, and its origin organisations own no word of its
    variable. With IDA the recipient, the DAC1 tables publish no recipient and leave
    (`without_open_axis_gaps`). The raw-text door names none: its roles are the extraction's reading, not the
    caller's."""
    assert _menu(state) == menu
    assert _origin_party(state).owned == ()


def test_the_provider_owns_its_words_of_the_variable():
    """With IDA the provider, "IDA" in "IDA disbursements" is IDA's, and "disbursements"
    is no organisation's."""
    owned = _origin_party(_state()).owned
    assert owns(_VARIABLE, owned, 0, 3)
    assert not owns(_VARIABLE, owned, 4, len(_VARIABLE))


def _kept_on_the_menu(party: _OriginParty, shapes: list[ShapeDraft]) -> list[str]:
    return [shape.shape_id for shape in _from_origin_publishers(party, shapes, None)]


def _listed_as_not_electable(party: _OriginParty, shapes: list[ShapeDraft]) -> list[str]:
    refused = [
        Interpretation(shape=shape, reason="denominator_not_available", pinned_failure="form")
        for shape in shapes
    ]
    return [i.shape.shape_id for i in _refused_on_pinned(party, refused, None)]


@pytest.mark.parametrize(
    "kept", [_kept_on_the_menu, _listed_as_not_electable], ids=["menu", "not-electable-list"]
)
def test_a_flow_no_provider_account_table_measures_stays(kept):
    """IDS's grants and concessional loans are disbursements, as DAC2a's gross
    disbursements are, so they leave for DAC2a. IDS debt service measures otherwise and
    stays. DAC1 publishes no IDA, so its grant equivalents leave. The not-electable list
    keeps the shapes the menu keeps."""
    shapes = [
        _shape(
            "ids_grants", ("Amount_FinancialFlow_Disbursement_UnrequitedTransfer_Nominal",), 0.70
        ),
        _shape(
            "ids_debt_service",
            ("Amount_FinancialFlow_ServicingPayment_DebtInstrument_Nominal",),
            0.65,
        ),
        _shape(
            "ids_loans",
            ("Amount_FinancialFlow_Disbursement_DebtInstrument_IDSConcessional_Nominal",),
            0.60,
        ),
        _shape(
            "dac2a_gross",
            ("Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal",),
            0.55,
        ),
        _shape(
            "dac1_grant_equivalent",
            ("GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1621_Nominal",),
            0.50,
        ),
    ]
    assert kept(_origin_party(_state()), shapes) == ["ids_debt_service", "dac2a_gross"]


def test_an_excluded_provider_account_table_replaces_no_flow():
    """The part asks "IDA disbursements, excluding OECD data". The exclusion removes every
    table that publishes IDA in the provider's account, so none replaces the IDS flows, and
    IDS answers. The scope allows IDS, a provenance the index declares no dataset for, as a
    part naming no source allows every admitted provenance."""
    oecd = [dataset for dataset in get_index().datasets if dataset.startswith("provenance/OECD")]
    excluding = part_scope(excluded=oecd)
    assert excluding is not None
    scope = dataclasses.replace(excluding, allowed=excluding.allowed | {_IDS})
    assert _menu(_state(scope=scope)) == ["ids_by_role"]


class _LeafPublishes(_OriginParty):
    """An origin party that the leaf publishes in the provider's account and every shape may
    hold."""

    def publishes(self, shape):
        return shape.shape_id == "leaf"

    def holds(self, shape):
        return True


@pytest.mark.parametrize(
    ("scored_roles", "menu"),
    [
        pytest.param(_ROLE_BEARING, ["leaf"], id="replaced-flow"),
        pytest.param(("observationAbout",), ["scored", "leaf"], id="subject-only-shape"),
    ],
)
def test_a_publisher_reached_as_a_topic_leaf_replaces_the_flows_measuring_alike(
    monkeypatch, scored_roles, menu
):
    """The publisher arrives as the leaf of a topic root, which seating stands behind every
    scored shape (`seat_menu`). A scored flow measuring as it does leaves for it, so the
    leaf alone answers; a scored shape with no flow role keeps its rank ahead of the leaf.
    The fakes say the leaf publishes the part's origin organisation and every shape
    measures alike."""
    scored = _prune_shape("scored", score=0.9, declared_roles=scored_roles)
    leaf = _prune_shape("leaf", score=None, rank=0.85, root=_ROOT, declared_roles=_ROLE_BEARING)
    monkeypatch.setattr(
        phase_rank_bind,
        "_origin_party",
        lambda state: _LeafPublishes(
            organisations={}, published=frozenset(), held=frozenset(), owned=()
        ),
    )
    monkeypatch.setattr(phase_rank_bind, "_measures_alike", lambda shape, other: True)

    assert [shape.shape_id for shape in _run_prune([scored, leaf])] == menu


def test_a_dataset_declaring_no_account_publishes_no_origin_organisation():
    """IDS publishes IDA on the origin axis, and the index declares no account for it, so
    of IDA's origin publishers the part keeps those giving the provider's account; IDS still
    holds IDA."""
    party = _origin_party(_state())
    assert (_DAC2A, "origin") in party.published
    assert (_IDS, "origin") not in party.published
    assert (_IDS, "origin") in party.held


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
