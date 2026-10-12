"""`records_plain_file` counts requested-format files alone."""

from __future__ import annotations

import pytest

from datasets_api.recorded_rendering import records_plain_file

_SUMMED = {"format": "csv", "bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}}


@pytest.mark.parametrize(
    ("rendering", "plain"),
    [
        ({"format": "csv", "bundle": False}, True),
        ({"format": "frame"}, False),
        ({"format": "frame1"}, False),
        ({"statistics": "series"}, False),
    ],
    ids=["requested-csv", "canonical-frame", "frame1", "statistics"],
)
def test_a_stored_frame_or_statistics_file_beside_a_sum_is_no_plain_file(
    rendering: dict[str, object], plain: bool
):
    assert records_plain_file([_SUMMED, rendering]) is plain
