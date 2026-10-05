"""Pins cached response serialization, restoration, and storability."""

from __future__ import annotations

import json

import pytest

from dfo_platform.trace import reset_trace_id, set_trace_id
from search.engine.envelope import make_diagnostics, make_query_echo, now_ms
from search.engine.response_cache_keys import from_entry, is_storable, to_entry
from search.models import (
    CatalogueAnswer,
    DeclinedAnswer,
    GraphRef,
    NoData,
    NoDataPart,
    Scope,
    Warning,
)
from tests.engine._harness import make_request, member_walk_reader, offline_resolve


def _base_response():
    return offline_resolve(
        make_request("total population India"), materialise_graph=member_walk_reader()
    )


def _scoped_answer(kind: str, *, warnings: list[Warning] | None = None):
    """A catalogue or declined answer for Kenya, built the way the engine builds an envelope."""
    query = "what data do you have on Kenya?"
    fields = {
        "query_echo": make_query_echo(query, extract_skipped=False),
        "diagnostics": make_diagnostics("test-build", warnings or [], {}, 5),
        "scope": Scope(place=GraphRef(dcid="country/KEN", label="Kenya")),
    }
    if kind == "catalogue":
        return CatalogueAnswer(**fields)
    return DeclinedAnswer(reason="out_of_scope", **fields)


# ---------------------------------------------------------------------------
# to_entry / from_entry round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_parts_sentence_and_warnings():
    base = _base_response()
    entry, _ = to_entry(base)

    restored = from_entry(
        entry,
        query=base.query_echo.raw_query,
        extract_skipped=base.query_echo.extract_skipped,
        start_ms=now_ms(),
        llm_usage=None,
    )

    assert restored.parts == base.parts
    assert restored.rendered_sentence == base.rendered_sentence
    assert restored.diagnostics.warnings == base.diagnostics.warnings


def test_restore_replaces_query_echo_with_the_new_wording():
    base = _base_response()
    entry, _ = to_entry(base)

    restored = from_entry(
        entry,
        query="how many people live in India",
        extract_skipped=False,
        start_ms=now_ms(),
        llm_usage=None,
    )

    assert restored.query_echo.raw_query == "how many people live in India"
    assert restored.query_echo.normalized_query == "how many people live in India"
    assert restored.query_echo.raw_query != base.query_echo.raw_query


def test_restore_replaces_diagnostics_with_the_current_run_facts():
    stored_token = set_trace_id("stored-trace-id")
    try:
        base = _base_response()
    finally:
        reset_trace_id(stored_token)
    assert base.diagnostics.trace_id == "stored-trace-id"

    entry, _ = to_entry(base)

    hit_token = set_trace_id("hit-trace-id")
    try:
        restored = from_entry(
            entry,
            query=base.query_echo.raw_query,
            extract_skipped=False,
            start_ms=now_ms() - 5_000,
            llm_usage={"input_tokens": 11, "output_tokens": 22, "cached_tokens": 33},
        )
    finally:
        reset_trace_id(hit_token)

    assert restored.diagnostics.trace_id == "hit-trace-id"
    assert restored.diagnostics.llm_usage == {
        "input_tokens": 11,
        "output_tokens": 22,
        "cached_tokens": 33,
    }
    assert restored.diagnostics.timing_ms is not None
    assert restored.diagnostics.timing_ms.total >= 5_000


def test_restore_threads_timing_by_step_when_given_and_omits_it_otherwise():
    base = _base_response()
    entry, _ = to_entry(base)

    with_step = from_entry(
        entry,
        query=base.query_echo.raw_query,
        extract_skipped=True,
        start_ms=now_ms(),
        llm_usage=None,
        timing_by_step={"response_cache_layer1": 3},
    )
    without_step = from_entry(
        entry,
        query=base.query_echo.raw_query,
        extract_skipped=True,
        start_ms=now_ms(),
        llm_usage=None,
    )

    assert with_step.diagnostics.timing_ms.by_step == {"response_cache_layer1": 3}
    assert without_step.diagnostics.timing_ms.by_step is None


def test_two_restores_from_one_entry_do_not_share_a_parts_object():
    base = _base_response()
    entry, _ = to_entry(base)

    restored_1 = from_entry(
        entry, query="q", extract_skipped=False, start_ms=now_ms(), llm_usage=None
    )
    restored_2 = from_entry(
        entry, query="q", extract_skipped=False, start_ms=now_ms(), llm_usage=None
    )

    restored_1.parts[0].warnings.append(Warning(code="probe", severity="warn", message="m"))

    assert restored_2.parts[0].warnings == []


@pytest.mark.parametrize("kind", ["catalogue", "declined"])
def test_round_trip_restores_a_part_less_answer_as_its_own_kind(kind):
    stored = _scoped_answer(kind)
    entry, _ = to_entry(stored)

    restored = from_entry(
        entry,
        query="What data do you have on Kenya?",
        extract_skipped=True,
        start_ms=now_ms(),
        llm_usage=None,
    )

    assert type(restored) is type(stored)
    assert restored.model_dump(exclude={"query_echo", "diagnostics"}) == stored.model_dump(
        exclude={"query_echo", "diagnostics"}
    )
    assert restored.query_echo.raw_query == "What data do you have on Kenya?"


def test_round_trip_restores_a_resolution_as_a_resolution():
    base = _base_response()
    entry, _ = to_entry(base)

    restored = from_entry(
        entry,
        query=base.query_echo.raw_query,
        extract_skipped=True,
        start_ms=now_ms(),
        llm_usage=None,
    )

    assert restored.kind == "resolution"
    assert json.loads(entry)["kind"] == "resolution"


def test_to_entry_reported_size_is_the_real_retained_length():
    """`entry` is the exact bytes `ResponseCache` retains under a key (see
    `to_entry`'s own docstring), so `size` must equal `len(entry)` and not
    some other estimate of it."""
    base = _base_response()
    entry, size = to_entry(base)

    assert isinstance(entry, bytes)
    assert size == len(entry)


def test_to_entry_output_round_trips_through_json_loads():
    base = _base_response()
    entry, _ = to_entry(base)

    assert json.loads(entry) == base.model_dump(mode="json")


# ---------------------------------------------------------------------------
# is_storable
# ---------------------------------------------------------------------------

_UNRESOLVED_PLACEHOLDER = NoDataPart(
    status="no_data", no_data=NoData(reason="variable_not_resolved")
)


def test_is_storable_true_for_a_clean_complete_response():
    base = _base_response()
    assert is_storable(base, timed_out=False, degraded_leg=False) is True


def test_is_storable_false_when_timed_out():
    base = _base_response()
    assert is_storable(base, timed_out=True, degraded_leg=False) is False


def test_is_storable_false_when_a_leg_degraded():
    base = _base_response()
    assert is_storable(base, timed_out=False, degraded_leg=True) is False


def test_is_storable_false_on_skills_index_unavailable_warning():
    base = _base_response()
    tainted_diagnostics = base.diagnostics.model_copy(
        update={
            "warnings": [
                Warning(code="skills_index_unavailable", severity="warn", message="not ready")
            ]
        }
    )
    tainted = base.model_copy(update={"diagnostics": tainted_diagnostics})

    assert is_storable(tainted, timed_out=False, degraded_leg=False) is False


def test_is_storable_false_on_skills_index_incomplete_warning_on_a_part():
    base = _base_response()
    tainted_part = base.parts[0].model_copy(
        update={
            "warnings": [
                Warning(code="skills_index_incomplete", severity="warn", message="building")
            ]
        }
    )
    tainted = base.model_copy(update={"parts": [tainted_part]})

    assert is_storable(tainted, timed_out=False, degraded_leg=False) is False


def test_is_storable_false_when_an_unresolved_placeholder_sits_alongside_a_resolved_part():
    base = _base_response()
    mixed = base.model_copy(update={"parts": [*base.parts, _UNRESOLVED_PLACEHOLDER]})

    assert is_storable(mixed, timed_out=False, degraded_leg=False) is False


def test_is_storable_true_for_a_genuine_single_variable_no_data_answer():
    """A query with nothing to return resolves to one no_data/variable_not_resolved
    part and nothing else. That is a real answer, not a placeholder, and the
    narrower predicate (see `is_storable`'s own docstring) must not refuse it."""
    base = _base_response()
    only_placeholder = base.model_copy(update={"parts": [_UNRESOLVED_PLACEHOLDER]})

    assert is_storable(only_placeholder, timed_out=False, degraded_leg=False) is True


@pytest.mark.parametrize("kind", ["catalogue", "declined"])
def test_is_storable_true_for_a_clean_part_less_answer(kind):
    assert is_storable(_scoped_answer(kind), timed_out=False, degraded_leg=False) is True


@pytest.mark.parametrize("kind", ["catalogue", "declined"])
def test_is_storable_false_for_a_part_less_answer_that_timed_out(kind):
    assert is_storable(_scoped_answer(kind), timed_out=True, degraded_leg=False) is False


@pytest.mark.parametrize("kind", ["catalogue", "declined"])
def test_is_storable_false_for_a_part_less_answer_with_an_unavailable_warning(kind):
    warning = Warning(code="un_system_unavailable", severity="warn", message="down")
    answer = _scoped_answer(kind, warnings=[warning])
    assert is_storable(answer, timed_out=False, degraded_leg=False) is False
