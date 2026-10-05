"""`dataset_store.snapshot.working_tier_expired`: age-based, from
`created_at`, against an injected `now` -- no real wall-clock time involved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dataset_store.snapshot import WORKING_TIER_RETENTION, working_tier_expired

_CREATED = datetime(2026, 1, 1, tzinfo=UTC)


def test_not_expired_the_moment_it_is_created():
    assert not working_tier_expired(_CREATED, now=_CREATED)


def test_not_expired_one_second_before_the_window_closes():
    now = _CREATED + WORKING_TIER_RETENTION - timedelta(seconds=1)
    assert not working_tier_expired(_CREATED, now=now)


def test_not_expired_exactly_at_the_window_boundary():
    # The exact boundary remains inside the 30-day window.
    now = _CREATED + WORKING_TIER_RETENTION
    assert not working_tier_expired(_CREATED, now=now)


def test_expired_one_second_past_the_window():
    now = _CREATED + WORKING_TIER_RETENTION + timedelta(seconds=1)
    assert working_tier_expired(_CREATED, now=now)
