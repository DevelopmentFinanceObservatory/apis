"""The weak retrieval score: the band a reading's recall score warns in.

A weak score is stated as `retrieval_score_weak` and lowers confidence beside a rival
(`grounding._grade_confidence`). It sets no status (`status_rule`).
"""

from __future__ import annotations

from search.engine.config import settings


def _representative_score_is_weak(score: float | None) -> bool:
    """True when the score is above the relevance floor and below the weak-score threshold.

    A missing score carries no evidence of weakness.
    """
    return (
        score is not None
        and settings().relevance_threshold < score < settings().weak_score_threshold
    )
