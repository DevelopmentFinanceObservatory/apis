"""Pins case-file validation and `score()` metrics using scanner doubles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pytest

from scripts.eval_role_direction import CASES_YAML, load_cases, score

_VALID_CATEGORIES = {
    "coordinated_donor_list",
    "coordinated_recipient_list",
    "three_plus_list",
    "possessive",
    "verb_no_preposition",
    "passive",
    "leading_unevidenced",
    "bare_comparison",
    "between_x_and_y",
    "substring_collision",
    "non_english",
    "world_knowledge_bait",
    "prior_contradicting",
    "adjectival_origin",
}
_VALID_AXES = {"origin", "destination", "unspecified"}


def test_case_file_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 20, "the eval set should cover every required category with room to spare"

    seen_ids: set[str] = set()
    seen_categories: set[str] = set()
    for case in cases:
        for key in ("id", "category", "query", "entities"):
            assert key in case, f"{case}: missing {key!r}"
        assert case["id"] not in seen_ids, f"duplicate id: {case['id']}"
        seen_ids.add(case["id"])
        assert case["category"] in _VALID_CATEGORIES, (
            f"{case['id']}: unknown category {case['category']!r}"
        )
        seen_categories.add(case["category"])
        assert case["query"].strip(), f"{case['id']}: empty query"
        assert case["entities"], f"{case['id']}: no entities"
        for entity in case["entities"]:
            for key in ("surface", "dcid", "expected_axis"):
                assert key in entity, f"{case['id']}: entity missing {key!r}: {entity}"
            assert entity["dcid"], f"{case['id']}: entity has no dcid: {entity}"
            assert entity["expected_axis"] in _VALID_AXES, (
                f"{case['id']}: invalid expected_axis {entity['expected_axis']!r}"
            )

    missing = _VALID_CATEGORIES - seen_categories
    assert not missing, f"required categories with no case: {missing}"


def test_case_file_has_unspecified_cases():
    """The false-confident-rate metric needs a nonzero unspecified-truth
    population to be meaningful; guard against the set drifting to
    all-confident labels."""
    cases = load_cases()
    unspecified = sum(
        1
        for case in cases
        for entity in case["entities"]
        if entity["expected_axis"] == "unspecified"
    )
    assert unspecified >= 5


def _always_none(query: str, entities: Sequence[tuple[str | None, str]]) -> list[None]:
    return [None] * len(entities)


def _always_origin(
    query: str, entities: Sequence[tuple[str | None, str]]
) -> list[Literal["origin"]]:
    return ["origin"] * len(entities)


def _perfect(cases):
    """Builds a scanner that always returns each case's own expected_axis
    (mapping "unspecified" to None, since that is the scanner's own way of
    expressing no evidence) -- used to prove score() reports 100% accuracy
    and 0% false-confident rate when the scanner is exactly right."""
    by_query = {}
    for case in cases:
        by_query[case["query"]] = [
            None if e["expected_axis"] == "unspecified" else e["expected_axis"]
            for e in case["entities"]
        ]

    def scanner(query: str, entities: Sequence[tuple[str | None, str]]) -> list[str | None]:
        return by_query[query]

    return scanner


def test_score_perfect_scanner_is_100_percent():
    cases = load_cases()
    result = score(cases, _perfect(cases))
    assert result["accuracy"] == 1.0
    assert result["false_confident_rate"] == 0.0
    assert result["confidently_wrong_count"] == 0
    assert result["confidently_wrong_rate"] == 0.0
    assert result["entity_count"] == sum(len(c["entities"]) for c in cases)


def test_score_always_none_scanner_matches_only_unspecified_truth():
    cases = load_cases()
    result = score(cases, _always_none)
    total = sum(len(c["entities"]) for c in cases)
    unspecified = sum(
        1 for c in cases for e in c["entities"] if e["expected_axis"] == "unspecified"
    )
    assert result["accuracy"] == pytest.approx(unspecified / total)
    assert result["false_confident_rate"] == 0.0, (
        "None is never confident, so it can never be false-confident"
    )
    assert result["confidently_wrong_count"] == 0, (
        "None is never a directional axis, so it can never be the opposite of one"
    )


def test_score_always_origin_scanner_reports_confident_errors():
    """Origin predictions are confidently wrong for unspecified and destination truth.

    A prediction of "unspecified" against a directional truth is a miss, not
    confidently wrong; a prediction of the opposite directional axis is.
    """
    cases = load_cases()
    result = score(cases, _always_origin)
    assert result["false_confident_rate"] == 1.0, (
        "every unspecified-truth entity got a confident (wrong) axis"
    )

    destination_truth = sum(
        1 for c in cases for e in c["entities"] if e["expected_axis"] == "destination"
    )
    directional_truth = sum(
        1 for c in cases for e in c["entities"] if e["expected_axis"] in ("origin", "destination")
    )
    assert result["confidently_wrong_count"] == destination_truth
    assert result["confidently_wrong_rate"] == pytest.approx(destination_truth / directional_truth)


def test_score_reports_per_category_breakdown():
    cases = load_cases()
    result = score(cases, _always_none)
    categories = {c["category"] for c in cases}
    assert set(result["by_category"]) == categories
    for bucket in result["by_category"].values():
        assert bucket["correct"] <= bucket["total"]


def test_score_rejects_mismatched_verdict_length():
    cases = load_cases()[:1]

    def short_scanner(query: str, entities: Sequence[tuple[str | None, str]]) -> list[None]:
        return []

    with pytest.raises(SystemExit):
        score(cases, short_scanner)


def test_cases_yaml_path_exists():
    assert CASES_YAML.is_file()
