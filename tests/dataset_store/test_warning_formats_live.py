"""`python -m dataset_store.warning_formats` against a real Postgres. Skipped unless
DFO_DATASET_STORE_LIVE_DSN is set, as `test_live.py` is."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest

from dataset_store import LiveDatasetStore, warning_formats
from tests.dataset_store._warning_refs import counts_since, format_count_bodies, format_counts
from tests.dataset_store.test_live import (
    _DSN,
    _delete_versions,
    _insert_as_the_previous_image,
    _pool,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _DSN, reason="set DFO_DATASET_STORE_LIVE_DSN to run live dataset_store tests"
    ),
]


@pytest.fixture
def live_store():
    pool = _pool()
    try:
        yield LiveDatasetStore(connections=pool)
    finally:
        pool.close_all()


def test_warning_formats_counts_each_old_format(live_store, monkeypatch, capsys):
    """The command prints a line per format with the warnings and version rows a row of each
    old format adds, checks `member_count` against the trace, and counts the rows 0039
    rewrites."""
    suffix = uuid4().hex
    bodies = format_count_bodies()
    subjects = [f"spec_warning_formats_{name}_{suffix}" for name in bodies]
    before = format_counts(cast(str, _DSN), monkeypatch, capsys)
    assert list(before) == [
        *(name for name, _condition in warning_formats.FORMATS),
        warning_formats.REWRITTEN_ROWS,
    ]
    for subject_id, body in zip(subjects, bodies.values(), strict=True):
        _insert_as_the_previous_image(
            live_store, subject_id=subject_id, content_version="cv1", body=body
        )
    try:
        assert counts_since(before, format_counts(cast(str, _DSN), monkeypatch, capsys)) == {
            "coverage_below_scope, unserved entities named": (3, 3),
            "coverage_below_scope, members named, window not a range": (2, 2),
            "coverage_below_scope, members named, range window": (1, 1),
            "coverage_below_scope, members named, trace holds no resolved_stat_vars": (1, 1),
            "coverage_below_scope, named refs beside the countries fallback": (2, 2),
            "currency_unmapped_omitted, entities named": (2, 2),
            "currency_unmapped_omitted, entities named, truncated": (1, 1),
            "deflator_missing_omitted, entities named": (2, 2),
            "deflator_missing_omitted, entities named, truncated": (1, 1),
            "currency or deflator, neither named nor counted": (1, 1),
            "member_count equals the trace's resolved_stat_vars": (2, 2),
            "member_count differs from the trace's resolved_stat_vars": (1, 1),
            "member_count beside a trace holding no resolved_stat_vars": (1, 1),
            warning_formats.REWRITTEN_ROWS: (None, 9),
        }
    finally:
        _delete_versions(live_store, subjects)
