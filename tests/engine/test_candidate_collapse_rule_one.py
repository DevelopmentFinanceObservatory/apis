"""Status rule 1 in the candidate collapse: what answers when the elected shape has no data.

Each test builds the grounded state `assemble_candidates` settles, beside an elected
shape with no data whose StatVar the equivalent-series registry pairs with the UN
electricity series, and reads which specs reach definite assembly."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import cast

import pytest

from search.engine.region import candidate_collapse
from search.engine.region.candidate_admission import _AlternatesAdmission
from search.engine.region.candidate_grounding import _GroundedCandidates
from search.engine.region.candidates import _Candidate
from search.engine.region.place_context import PlaceContext
from search.engine.region.state import ResolveState
from search.engine.retrieval.retrieve import Materialised, NoDataDraft
from search.engine.shape import ShapeDraft
from search.models import CandidateBasis, Spec
from tests.engine._assemble_fixtures import _make_spec, _standard_shape_draft
from tests.engine.test_candidate_grounding_thread import _state

_ELECTED_SV = "worldBank/EG_ELC_ACCS_ZS"
_TWIN_SV = "undata/sdg/EG_ACS_ELEC"
_STAND_IN_SV = "Test_Stand_In"


def _shape(shape_id: str) -> ShapeDraft:
    return dataclasses.replace(_standard_shape_draft(), shape_id=shape_id, label=shape_id)


def _candidate(shape: ShapeDraft, basis: CandidateBasis | None) -> _Candidate:
    return _Candidate(
        shape=shape,
        bindings=[],
        mat=cast(Materialised, None),
        place=cast(PlaceContext, None),
        walked=None,
        basis=basis,
    )


def _grounded(*rows: tuple[Spec, str, ShapeDraft, CandidateBasis | None]) -> _GroundedCandidates:
    """Grounded candidates from (spec, StatVar, shape, basis) rows, in order."""
    return _GroundedCandidates(
        specs=[spec for spec, *_ in rows],
        elected_spec_id=None,
        cand_scores={},
        cand_retrieved={},
        cand_populations={},
        cand_probe_inputs={},
        cand_role_axis_facts={spec.spec_id: (shape, [], None) for spec, _, shape, _ in rows},
        cand_entity_options_raw={},
        cand_measure_partitions={},
        cand_measure_siblings={},
        cand_open_axis_facts={},
        cand_obs_entities={spec.spec_id: ([sv], {}) for spec, sv, _, _ in rows},
        cand_probe_entities={},
        cand_elected={},
        cand_bulk_scope={},
        cand_plans={},
        cand_confirmed_facets={},
        cand_answered_roles={},
        cand_places={},
        answer_t0=0,
        cand_candidates={spec.spec_id: _candidate(shape, basis) for spec, _, shape, basis in rows},
    )


def _settle(
    monkeypatch: pytest.MonkeyPatch, state: ResolveState, grounded: _GroundedCandidates
) -> tuple[list[Spec], ResolveState]:
    """The specs `assemble_candidates` hands definite assembly for `grounded`."""
    monkeypatch.setattr(candidate_collapse, "ground_candidates", lambda _state, _c: grounded)

    async def _definite(_state, specs, _grounded, _rate_cache):
        return specs

    monkeypatch.setattr(candidate_collapse, "_assemble_definite", _definite)
    specs = asyncio.run(
        candidate_collapse.assemble_candidates(state, _AlternatesAdmission(), rate_cache=None)
    )
    return cast(list[Spec], specs), state


def _no_data_state() -> ResolveState:
    state = _state()
    state.shape_draft = _shape("elected")
    state.mat_result = NoDataDraft(reason="no_observations", sv_dcids=[_ELECTED_SV])
    return state


def test_a_twin_that_misses_the_named_quantity_is_not_served(monkeypatch) -> None:
    """ "External debt to GDP, Kenya": the elected debt-to-GNI shape has no data, and its
    registry twin has data but misses the named base, as the elected shape does. The quantity
    filter drops the twin before the twin diversion, so the stand-in that serves the base
    answers, with no twin among its equivalent readings."""
    twin_shape, stand_in_shape = _shape("twin"), _shape("stand_in")
    twin = _make_spec(_TWIN_SV, "twin", "country/KEN")
    stand_in = _make_spec(_STAND_IN_SV, "stand_in", "country/KEN")
    state = _no_data_state()
    state.quantity_refusal = state.mat_result
    monkeypatch.setattr(
        candidate_collapse, "serves_leg_quantity", lambda _state, shape: shape is stand_in_shape
    )

    specs, state = _settle(
        monkeypatch,
        state,
        _grounded(
            (twin, _TWIN_SV, twin_shape, "twin"),
            (stand_in, _STAND_IN_SV, stand_in_shape, "structural"),
        ),
    )

    assert [spec.spec_id for spec in specs] == [stand_in.spec_id]
    assert specs[0].resolved.equivalent_readings is None
    assert [warning.code for warning in state.warnings] == ["elected_shape_no_data"]


def test_an_unconfirmed_twin_is_no_stand_in(monkeypatch) -> None:
    """A registry twin flagged without data at the place is not diverted
    (`_apply_twin_diversion`) and keeps its `twin` basis. Rule 1 installs only the
    structural sibling, so it answers alone with the sibling warning, and the twin is not a
    second stand-in."""
    twin_shape, stand_in_shape = _shape("twin"), _shape("stand_in")
    twin = _make_spec(_TWIN_SV, "twin", "country/KEN")
    twin = twin.model_copy(
        update={"resolved": twin.resolved.model_copy(update={"available_at_place": False})}
    )
    stand_in = _make_spec(_STAND_IN_SV, "stand_in", "country/KEN")

    specs, state = _settle(
        monkeypatch,
        _no_data_state(),
        _grounded(
            (twin, _TWIN_SV, twin_shape, "twin"),
            (stand_in, _STAND_IN_SV, stand_in_shape, "structural"),
        ),
    )

    assert [spec.spec_id for spec in specs] == [stand_in.spec_id]
    assert [warning.code for warning in state.warnings] == ["elected_shape_no_data"]
