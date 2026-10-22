"""A part refused before bind answers with the reason its refused readings elect.

A menu emptied with no refusal recorded is a fault upstream: the part answers
`variable_not_resolved` and the engine logs it at ERROR, once.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import cast

import pytest

from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.interpretation import Interpretation
from search.engine.region import phase_rank_bind
from search.engine.region.phase_rank_bind import _refusal_before_bind
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot

_VARIABLE = "education spend per student"


def _state() -> ResolveState:
    """An empty menu, with no interpretation judged."""
    return ResolveState(
        variable=_VARIABLE,
        entities=[],
        date_request=None,
        detect_query=_VARIABLE,
        role_query=_VARIABLE,
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
        coverage=empty_coverage_snapshot(),
    )


def _refused_on_form() -> Interpretation:
    shape = ShapeDraft(
        shape_id="education_spend",
        label="education spend",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
    )
    return Interpretation(shape=shape, reason="denominator_not_available", pinned_failure="form")


def _error_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_an_empty_menu_with_no_refusal_recorded_logs_an_error_and_refuses(
    caplog: pytest.LogCaptureFixture,
):
    state = _state()
    assert state.capped_shapes == []

    with caplog.at_level(logging.ERROR, logger=phase_rank_bind.__name__):
        result = _refusal_before_bind(state)

    assert (result.status, result.no_data_reason) == ("no_data", "variable_not_resolved")
    (record,) = _error_records(caplog)
    assert _VARIABLE in record.getMessage()


def test_a_recorded_refusal_answers_with_its_reason_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
):
    state = _state()
    state.interpretations = replace(state.interpretations, interpretations=(_refused_on_form(),))

    with caplog.at_level(logging.ERROR, logger=phase_rank_bind.__name__):
        result = _refusal_before_bind(state)

    assert (result.status, result.no_data_reason) == ("no_data", "denominator_not_available")
    assert _error_records(caplog) == []
