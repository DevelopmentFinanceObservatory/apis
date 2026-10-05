"""Test that resubmits preserve five posted slot keys across edits and candidate promotion.

Slot keys are the posted identity's own, not the union of what its members assert.
"""

from __future__ import annotations

import pytest

from search.engine.clients.graph import Facet
from search.engine.entry.dataset_entries import _candidate_promote_input
from search.engine.region import resolve_spec_resubmit
from search.engine.shape import identity_shape_for
from search.models import RemovedParts, ResubmitCandidate, ResubmitIdentityPayload
from series_coverage.schema import RoleNames
from tests.engine._base_identity import (
    GDP_SHAPE,
    destination_entity,
    make_base,
    origin_entity,
    subject_entity,
)
from tests.engine._builders import value_slot
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_FRA = "country/FRA"
_DEU = "country/DEU"
_TGO = "country/TGO"

_ROLES = ("financingProvider", "financingRecipient")

# The exact five how slots the live definition served.
_HOW_PROPS = {
    "crsPurpose": "CRSPurposeCode110",
    "crsTypeOfFinance": "CRSTypeOfFinance100",
    "crsTypeOfFlow": "CRSTypeOfFlow10",
    "financialFlowType": "Disbursement",
    "nettingBasis": "Gross",
}
# Served members assert crsCooperationModality as an open-axis fact.
_MODALITY_PROP = "crsCooperationModality"
_MODALITY_VALUE = "CRSCooperationModalityA02"

_SEED_SV = "TESTFIXTURE/OpenAxisMemberOnly/Seed"
_MODALITY_SV = "TESTFIXTURE/OpenAxisMemberOnly/Modality"

_SHAPE = identity_shape_for(
    pop_type_dcid="TESTFIXTURE/OpenAxisMemberOnlyFlow",
    meas_prop_dcid="amount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="Nominal",
    meas_denom_dcid=None,
)


def _sv_arcs(constraint_properties: dict[str, str]) -> dict:
    """Build StatVar arcs with the fixture's shape, roles, and constraints."""
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "TESTFIXTURE/OpenAxisMemberOnlyFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": p} for p in _ROLES]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraint_properties.items()},
    }


def _graph() -> FakeGraph:
    nodes = {
        _FRA: {"label": "France", "type": "Country"},
        _DEU: {"label": "Germany", "type": "Country"},
        _TGO: {"label": "Togo", "type": "Country"},
        _SEED_SV: {
            "label": "seed",
            "type": "StatisticalVariable",
            "arcs": _sv_arcs(_HOW_PROPS),
        },
        _MODALITY_SV: {
            "label": "modality",
            "type": "StatisticalVariable",
            "arcs": _sv_arcs({**_HOW_PROPS, _MODALITY_PROP: _MODALITY_VALUE}),
        },
    }
    return FakeGraph(nodes=nodes, obs={})


def _facet(facet_id: str) -> Facet:
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=1,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
    )


def _sdmx() -> FakeSdmxClient:
    """Provide France- and Germany-bound rows for both served members."""
    rows = {
        (("financingProvider", _FRA), ("financingRecipient", _TGO)): [_facet("f-fra")],
        (("financingProvider", _DEU), ("financingRecipient", _TGO)): [_facet("f-deu")],
    }
    observations = {sv: dict(rows) for sv in (_SEED_SV, _MODALITY_SV)}
    return FakeSdmxClient(observations=observations, availability={})


_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")


def _coverage(monkeypatch):
    """Mirror `_sdmx()` with coverage rows for both providers and members."""
    rows = [
        (sv, facet_id, donor, _TGO, [2020])
        for sv in (_SEED_SV, _MODALITY_SV)
        for facet_id, donor in (("f-fra", _FRA), ("f-deu", _DEU))
    ]
    index = patch_coverage(
        monkeypatch, rows, role_names=dict.fromkeys((_SEED_SV, _MODALITY_SV), _ROLE_NAMES)
    )
    return FakeCoverageService(index).snapshot()


def _how_slots() -> list:
    return [value_slot("how", prop, prop, value) for prop, value in _HOW_PROPS.items()]


def _served_definition():
    """Build the five-slot France-to-Togo identity used by these tests."""
    return make_base(
        _SHAPE,
        _how_slots(),
        entities=[origin_entity(_FRA), destination_entity(_TGO)],
    )


# ---------------------------------------------------------------------------
# No-op, origin removal, and Germany rebind cases resolve.
# ---------------------------------------------------------------------------


def test_noop_resubmit_of_the_served_definition_remints_the_base_spec_id(monkeypatch):
    base_spec_id, base_identity = _served_definition()
    inp = ResubmitIdentityPayload(
        identity=base_identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_SEED_SV, _MODALITY_SV],
    )
    region = resolve_spec_resubmit(
        inp=inp, graph=_graph(), sdmx=_sdmx(), coverage=_coverage(monkeypatch)
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.spec_id == base_spec_id
    # The five posted keys remain slots. Modality stays an open-axis fact.
    assert {slot.property for slot in spec.definition.identity.slots} == set(_HOW_PROPS)
    open_props = {fact.property for fact in spec.resolved.open_axis_facts or ()}
    assert _MODALITY_PROP in open_props


@pytest.mark.parametrize(
    "edit_entities, removed, check",
    [
        pytest.param(
            lambda entities: {k: v for k, v in entities.items() if k != "origin"},
            RemovedParts(entities=["origin"]),
            lambda entities: "origin" not in entities,
            id="origin_removed_resolves_any_donor",
        ),
        pytest.param(
            lambda entities: {**entities, "origin": [_DEU]},
            None,
            lambda entities: entities.get("origin") == [_DEU],
            id="origin_changed_resolves",
        ),
    ],
)
def test_resubmit_origin_edit_resolves(edit_entities, removed, check, monkeypatch):
    base_spec_id, base_identity = _served_definition()
    identity = base_identity.model_copy(update={"entities": edit_entities(base_identity.entities)})
    inp = ResubmitIdentityPayload(
        identity=identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        removed=removed,
        resolved_stat_var_dcids=[_SEED_SV, _MODALITY_SV],
    )
    region = resolve_spec_resubmit(
        inp=inp, graph=_graph(), sdmx=_sdmx(), coverage=_coverage(monkeypatch)
    )

    assert region.status == "definite", region.no_data_reason
    assert check(region.specs[0].definition.identity.entities)


# ---------------------------------------------------------------------------
# A candidate promote re-mints the candidate's own spec_id.
# ---------------------------------------------------------------------------

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"
_ACTIVITY_SOURCE_SLOT = value_slot(
    "how", "activitySource", "activity source", "GrossDomesticProduction"
)


def test_candidate_promote_remints_the_candidates_own_spec_id():
    """Candidate promotion must mint the candidate identity's own spec id."""
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    candidate_spec_id, candidate_identity = make_base(
        GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities
    )
    candidate = ResubmitCandidate(
        identity=candidate_identity, resolved_stat_var_dcids=[_GDP_SV_DCID]
    )

    inp = _candidate_promote_input(candidate)
    region = resolve_spec_resubmit(
        inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].spec_id == candidate_spec_id
