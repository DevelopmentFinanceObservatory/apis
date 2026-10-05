"""Tests for the warning-code and input-door allowlist."""

from typing import get_args

from search.engine.codes import WARNING_CODES
from search.eval.assertable_warnings import (
    ASSERTABLE_WARNINGS,
    CASE_DOORS,
    _pairs_not_registered,
    check_cases_assertable_warnings,
    doors_not_registered,
    warning_violations,
)
from search.eval.cases import Case, validate_case
from search.models import EntryPath


def _case(case_id: str, door: str, expect: dict) -> Case:
    raw: dict = {"id": case_id, "query": "aid to Kenya", "why": "test", "door": door}
    if door == "parsed":
        raw["parsed"] = [{"variable_text": "aid", "entities": [{"text": "Kenya"}]}]
    if door == "dataset_resubmit":
        raw["resubmit"] = {
            "variable_text": "aid",
            "shape": {"population_type": "FinancialFlow"},
            "slots": [],
        }
    return validate_case({**raw, "expect": expect}, families={})


def test_every_allowlisted_code_is_a_real_warning_code():
    for code, _door in ASSERTABLE_WARNINGS:
        assert code in WARNING_CODES


def test_case_doors_are_the_request_input_kinds():
    assert set(get_args(EntryPath)) == CASE_DOORS


def test_retrieval_score_weak_on_raw_text_is_rejected():
    case = _case("bad-case", "raw_text", {"definite": {"warnings": ["retrieval_score_weak"]}})
    violations = warning_violations(case)
    assert len(violations) == 1
    assert "bad-case" in violations[0]
    assert "retrieval_score_weak" in violations[0]
    assert "raw_text" in violations[0]


def test_role_scope_defaulted_on_dataset_resubmit_is_accepted():
    case = _case(
        "good-case", "dataset_resubmit", {"warnings": ["role_scope_defaulted"], "definite": {}}
    )
    assert warning_violations(case) == []


def test_warnings_inside_parts_are_checked():
    case = _case(
        "parts-case",
        "raw_text",
        {"parts": [{"definite": {"warnings": ["variables_clamped"]}}, {"definite": {}}]},
    )
    assert len(warning_violations(case)) == 1


def test_check_cases_collects_violations_across_the_list():
    cases = [
        _case("ok", "parsed", {"definite": {"warnings": ["binding_folded"]}}),
        _case("bad-1", "raw_text", {"definite": {"warnings": ["variables_clamped"]}}),
        _case("bad-2", "raw_text", {"no_data": {"warnings": ["degenerate_binding_detected"]}}),
    ]
    violations = check_cases_assertable_warnings(cases)
    assert len(violations) == 2
    assert any("bad-1" in v for v in violations)
    assert any("bad-2" in v for v in violations)


def test_every_allowlisted_pair_is_registered_for_that_door():
    """Every allowlisted code/door pair appears in the pinned registry."""
    assert doors_not_registered() == []


def test_a_code_absent_from_the_registry_is_reported_as_not_registered():
    pairs = [("totally_made_up_code_xyz", "raw_text")]
    assert _pairs_not_registered(pairs, {}) == [("totally_made_up_code_xyz", "raw_text")]


def test_a_registered_code_with_no_doors_restriction_applies_to_every_door():
    pairs = [("role_scope_defaulted", "raw_text"), ("role_scope_defaulted", "parsed")]
    assert _pairs_not_registered(pairs, {"role_scope_defaulted": None}) == []


def test_default_not_declared_is_not_assertable_because_completion_never_emits_it():
    """Unit completion reads declared sources and refuses when they are absent."""
    case = _case("dead-warning", "parsed", {"definite": {"warnings": ["default_not_declared"]}})
    assert len(warning_violations(case)) == 1
