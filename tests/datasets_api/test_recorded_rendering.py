"""`datasets_api.recorded_rendering` reads the options that minted a version from the renderings
of the files linked under it, and the stored frames that serve a version."""

from __future__ import annotations

from dataclasses import replace

import pytest

from datasets_api.mint_models import GroupbyOptions
from datasets_api.recorded_rendering import (
    MintingOptions,
    ServingFrame,
    derived_version_sums,
    recorded_mintings,
    records_plain_file,
    serving_frames,
)
from datasets_api.render import FrameMode

from ._snapshot_doubles import _version

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


_LEFT_OUT = {"party_totals_left_out": [{"role": "recipient", "entities": ["X"]}]}
_RAW = ServingFrame(content_version="cv_base", mode=FrameMode.RAW)


def _own(mode: FrameMode) -> ServingFrame:
    return ServingFrame(content_version="cv1", mode=mode)


def test_a_raw_version_is_served_by_its_own_frame_alone():
    assert serving_frames(_version(), renderings=[_SUMMED]) == (_own(FrameMode.RAW),)


@pytest.mark.parametrize(
    ("frame_summed", "trace", "renderings", "frames"),
    [
        (True, _LEFT_OUT, [_CUT_AND_SUMMED], (_own(FrameMode.SUMMED), _RAW)),
        (False, _LEFT_OUT, [_CUT_AND_SUMMED], (_own(FrameMode.CUT), _RAW)),
        (False, _LEFT_OUT, [_CUT], (_own(FrameMode.CUT), _RAW)),
        (True, {}, [_SUMMED], (_own(FrameMode.SUMMED), _RAW)),
        (None, {}, [_SUMMED], (_own(FrameMode.SUMMED), _RAW)),
        (None, _LEFT_OUT, [_CUT], (_own(FrameMode.CUT), _RAW)),
        (None, _LEFT_OUT, [_CUT_AND_SUMMED, _CUT], (_own(FrameMode.CUT), _RAW)),
        (None, _LEFT_OUT, [_CUT_AND_SUMMED], (_RAW,)),
        (None, _LEFT_OUT, [{"format": "frame"}], (_RAW,)),
    ],
    ids=[
        "recorded-collapse-of-a-cut",
        "recorded-cut-a-groupby-sums-on-render",
        "recorded-cut",
        "recorded-collapse",
        "unrecorded-collapse",
        "unrecorded-cut",
        "unrecorded-cut-also-minted-with-a-groupby",
        "unrecorded-cut-with-a-groupby",
        "unrecorded-cut-recording-no-options",
    ],
)
def test_a_derived_version_is_served_by_its_own_frame_when_the_record_says_what_it_holds(
    frame_summed, trace, renderings, frames
):
    version = _version(base_content_version="cv_base")
    version = replace(version, facts=replace(version.facts, trace=trace), frame_summed=frame_summed)

    assert serving_frames(version, renderings=renderings) == frames
