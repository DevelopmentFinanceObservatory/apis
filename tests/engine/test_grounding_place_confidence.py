"""Pins confidence grades for entity-less definite results and leaves non-matches ungraded (#305)."""

from __future__ import annotations

from search.engine.config import settings
from search.engine.region.grounding import _place_confidence


def test_entity_less_definite_strong_score_grades_high():
    available_at_place, confidence = _place_confidence(
        destination_dcid=None,
        available_at_place=None,
        representative_score=settings().weak_score_threshold,
        has_data=True,
    )
    assert available_at_place is None
    assert confidence == "high"


def test_entity_less_definite_weak_score_grades_medium():
    available_at_place, confidence = _place_confidence(
        destination_dcid=None,
        available_at_place=None,
        representative_score=settings().weak_score_threshold - 0.01,
        has_data=True,
    )
    assert available_at_place is None
    assert confidence == "medium"


def test_entity_less_definite_never_grades_low():
    """A weak no-entity match has no availability signal and remains medium confidence."""
    _, confidence = _place_confidence(
        destination_dcid=None,
        available_at_place=None,
        representative_score=0.0,
        has_data=True,
    )
    assert confidence == "medium"


def test_entity_less_non_match_stays_ungraded():
    """A flagged non-match with has_data=False has no availability or confidence grade."""
    available_at_place, confidence = _place_confidence(
        destination_dcid=None,
        available_at_place=None,
        representative_score=0.99,
        has_data=False,
    )
    assert available_at_place is None
    assert confidence is None


def test_has_data_defaults_to_true_for_existing_callers():
    """Callers that omit has_data retain the default graded behaviour."""
    _, confidence = _place_confidence(
        destination_dcid=None,
        available_at_place=None,
        representative_score=settings().weak_score_threshold,
    )
    assert confidence == "high"
