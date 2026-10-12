"""Tests for `no_data_role_axis`. The role axis is open only when materialisation
finds no data; a later empty-window result retains the discovered recipients.
"""

from __future__ import annotations

from typing import cast

from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.region.state import ResolveState
from search.engine.retrieval.retrieve import Materialised, NoDataDraft
from search.engine.shape import ShapeDraft
from search.models import CoverageBare
from tests.engine._fake_coverage import empty_coverage_snapshot

_SV = "TESTFIXTURE/NoDataRoleAxis/Flow"


def _role_bearing_shape() -> ShapeDraft:
    """Build a shape whose sole declared role, `debtor`, maps to `destination`."""
    return ShapeDraft(
        shape_id="test_no_data_role_axis_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=("debtor",),
    )


def _state(*, mat_result: Materialised | NoDataDraft) -> ResolveState:
    """Build the fields that `no_data` reads; other state fields are placeholders."""
    return ResolveState(
        variable="aid",
        entities=[],
        date_request=None,
        detect_query="aid",
        role_query="aid",
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
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
        shape_draft=_role_bearing_shape(),
        mat_result=mat_result,
        coverage=empty_coverage_snapshot(),
    )


def test_no_data_role_axis_is_none_when_mat_result_is_already_materialised():
    state = _state(
        mat_result=Materialised(
            sv_dcids=[_SV],
            has_data=True,
            coverage=CoverageBare(has_data=True),
        )
    )

    result = state.no_data("no_observations")

    assert result.no_data_role_axis is None


def test_no_data_role_axis_names_the_open_axis_for_a_no_data_draft():
    """A `no_observations` draft exposes the unresolved destination axis."""
    state = _state(mat_result=NoDataDraft(reason="no_observations"))

    result = state.no_data("no_observations")

    assert result.no_data_role_axis == "destination"
