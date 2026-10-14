"""`datasets_api.recorded_rendering` reads the options that minted a version from the renderings
of the files linked under it."""

from __future__ import annotations

import pytest

from datasets_api.mint_models import GroupbyOptions
from datasets_api.recorded_rendering import (
    MintingOptions,
    derived_version_sums,
    recorded_mintings,
    records_plain_file,
)

_SUMMED = {"format": "csv", "bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}}
_CUT = {"format": "csv", "bundle": False, "without_party_totals": True}
_CUT_AND_SUMMED = {**_SUMMED, "without_party_totals": True}
_GROUPBY = GroupbyOptions(by=["recipient"], op="sum")


@pytest.mark.parametrize(
    ("rendering", "plain"),
    [
        ({"format": "csv", "bundle": False}, True),
        ({"format": "frame"}, False),
        ({"format": "frame1"}, False),
        ({"statistics": "series"}, False),
        (_CUT, False),
    ],
    ids=["requested-csv", "canonical-frame", "frame1", "statistics", "party-totals-cut"],
)
def test_a_stored_frame_or_statistics_file_beside_a_sum_is_no_plain_file(
    rendering: dict[str, object], plain: bool
):
    assert records_plain_file([_SUMMED, rendering]) is plain


def test_each_file_records_its_groupby_and_party_totals_cut():
    renderings = [
        {"format": "frame"},
        {"format": "csv", "bundle": False},
        _SUMMED,
        _CUT,
        _CUT_AND_SUMMED,
        {**_SUMMED, "groupby": {"by": ["recipient"], "op": "mean"}},
    ]

    assert recorded_mintings(renderings) == [
        MintingOptions(groupby=_GROUPBY),
        MintingOptions(without_party_totals=True),
        MintingOptions(groupby=_GROUPBY, without_party_totals=True),
    ]


@pytest.mark.parametrize(
    ("trace", "renderings", "sums"),
    [
        ({}, [_SUMMED], True),
        ({"party_totals_left_out": [{"role": "recipient", "entities": ["X"]}]}, [_CUT], False),
        (
            {"party_totals_left_out": [{"role": "recipient", "entities": ["X"]}]},
            [_CUT_AND_SUMMED],
            True,
        ),
    ],
    ids=["collapse", "cut", "collapse-of-a-cut"],
)
def test_a_derived_version_sums_unless_it_is_a_cut_alone(trace, renderings, sums):
    assert derived_version_sums(trace, renderings) is sums
