"""`GenerationStamp.key`/`max_age_s`."""

from __future__ import annotations

from graph_generation.config import (
    GRAPH_CACHE_MAX_AGE_SECONDS,
    GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
)
from graph_generation.stamp import UNKNOWN_KEY, UNKNOWN_STAMP, GenerationStamp


def test_known_stamp_keys_and_ages_by_its_token() -> None:
    stamp = GenerationStamp(token="rev-1")
    assert stamp.key == "rev-1"
    assert stamp.max_age_s == float(GRAPH_CACHE_MAX_AGE_SECONDS)


def test_unknown_stamp_keys_as_the_literal_unknown_and_ages_short() -> None:
    stamp = GenerationStamp(token=None)
    assert stamp.key == UNKNOWN_KEY
    assert stamp.max_age_s == float(GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS)
    assert stamp == UNKNOWN_STAMP
