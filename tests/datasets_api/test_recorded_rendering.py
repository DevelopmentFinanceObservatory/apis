"""`datasets_api.recorded_rendering` reads the options that minted a version from the renderings
of the files linked under it, and the stored frames that serve a version."""

from __future__ import annotations

from dataclasses import replace

import pytest

from datasets_api.mint_models import GroupbyOptions
from datasets_api.recorded_rendering import (
    MintingOptions,
    RenderingNotRecorded,
    ServingFrame,
    StoredRendering,
    derived_version_sums,
    recorded_mintings,
    records_plain_file,
    rendering_of,
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


def test_a_code_selection_is_a_minting_of_its_own_and_an_absent_one_records_as_before():
    selected = {
        **_SUMMED,
        "groupby": {"by": ["recipient"], "op": "sum", "codes": {"recipient": ["B", "A"]}},
    }

    mintings = recorded_mintings([_SUMMED, selected, {**selected}])

    assert mintings[0] == MintingOptions(groupby=_GROUPBY)
    assert mintings[1] == mintings[2] != mintings[0]
    assert hash(mintings[1]) == hash(mintings[2])
    assert mintings[1].groupby is not None
    assert mintings[1].groupby.codes == {"recipient": ["A", "B"]}
    assert _GROUPBY.model_dump(mode="json") == {"by": ["recipient"], "op": "sum", "levels": None}


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


_PLAIN = {"format": "csv", "bundle": False}
_CUT_MINTING = MintingOptions(without_party_totals=True)
_SUMMED_MINTING = MintingOptions(groupby=_GROUPBY)
_CUT_AND_SUMMED_MINTING = MintingOptions(groupby=_GROUPBY, without_party_totals=True)


@pytest.mark.parametrize(
    ("base", "renderings", "refreshes_as"),
    [
        (None, [_CUT_AND_SUMMED, _CUT], _CUT_MINTING),
        (None, [_CUT, _CUT_AND_SUMMED], _CUT_MINTING),
        (None, [_CUT_AND_SUMMED, _PLAIN], None),
        ("cv_base", [_CUT_AND_SUMMED, _SUMMED], _SUMMED_MINTING),
        ("cv_base", [_SUMMED, _CUT_AND_SUMMED], _SUMMED_MINTING),
    ],
    ids=[
        "raw-summed-cut-first",
        "raw-cut-first",
        "raw-plain-file-last",
        "derived-summed-cut-first",
        "derived-groupby-first",
    ],
)
def test_a_version_recorded_under_several_renderings_refreshes_as_one_the_file_order_leaves(
    base, renderings, refreshes_as
):
    rendering = rendering_of(
        spec_id="spec_a",
        version=_version(base_content_version=base),
        renderings=[{"format": "frame"}, *renderings],
    )

    assert (rendering.refreshes_as, rendering.ambiguous) == (refreshes_as, True)
    assert StoredRendering.from_record(rendering.to_record()) == rendering


def _minting_record(minting: MintingOptions) -> dict[str, object]:
    return {
        "groupby": minting.groupby.model_dump(mode="json") if minting.groupby else None,
        "without_party_totals": minting.without_party_totals,
    }


def test_a_card_record_keeps_the_link_order_and_adds_only_plain():
    """A raw version linking a plain file, a groupby file and a cut file records its mintings in
    the order the files were linked, `minting` as the first by `ordered_mintings`, and the
    other keys a card revision recorded before `plain` existed, plus `plain`."""
    rendering = rendering_of(
        spec_id="spec_a",
        version=_version(),
        renderings=[{"format": "frame"}, _PLAIN, _SUMMED, _CUT],
    )

    assert rendering.to_record() == {
        "base_content_version": "cv1",
        "minting": _minting_record(_CUT_MINTING),
        "ambiguous": True,
        "collapsed": False,
        "mintings": [_minting_record(_SUMMED_MINTING), _minting_record(_CUT_MINTING)],
        "plain": True,
    }
    assert rendering.refreshes_as is None


@pytest.mark.parametrize(
    "rendering",
    [
        StoredRendering(base_content_version="cv1"),
        StoredRendering(base_content_version="cv1", mintings=(_CUT_MINTING,)),
        StoredRendering(
            base_content_version="cv1", mintings=(_SUMMED_MINTING, _CUT_MINTING), plain=True
        ),
        StoredRendering(base_content_version="cv1", mintings=(_CUT_MINTING, _SUMMED_MINTING)),
        StoredRendering(
            base_content_version="cv_base",
            collapsed=True,
            mintings=(_CUT_AND_SUMMED_MINTING, _SUMMED_MINTING),
        ),
        StoredRendering(
            base_content_version="cv_base",
            collapsed=True,
            mintings=(
                MintingOptions(
                    groupby=GroupbyOptions(by=["recipient"], op="sum", codes={"recipient": ["A"]})
                ),
            ),
        ),
    ],
    ids=[
        "plain",
        "single",
        "ambiguous-raw",
        "several-mintings-raw",
        "ambiguous-derived",
        "code-selection",
    ],
)
def test_a_card_record_reads_back_as_the_rendering_it_records(rendering: StoredRendering):
    assert StoredRendering.from_record(rendering.to_record()) == rendering


@pytest.mark.parametrize(
    ("rendering", "ambiguous", "minting"),
    [
        (
            StoredRendering(
                base_content_version="cv1", mintings=(_SUMMED_MINTING, _CUT_MINTING), plain=True
            ),
            True,
            _CUT_MINTING,
        ),
        (
            StoredRendering(base_content_version="cv1", mintings=(_CUT_MINTING, _SUMMED_MINTING)),
            False,
            _CUT_MINTING,
        ),
        (
            StoredRendering(
                base_content_version="cv_base",
                collapsed=True,
                mintings=(_CUT_AND_SUMMED_MINTING, _SUMMED_MINTING),
            ),
            False,
            _SUMMED_MINTING,
        ),
    ],
    ids=["plain-and-mintings", "several-mintings-raw", "several-mintings-derived"],
)
def test_a_card_record_is_ambiguous_only_when_the_plain_rendering_is_one_of_several(
    rendering: StoredRendering, ambiguous: bool, minting: MintingOptions
):
    """A reader that predates `plain` reads `ambiguous` as the plain rendering being one of
    several and otherwise refreshes the record as its `minting`, the first by
    `ordered_mintings`."""
    record = rendering.to_record()

    assert (record["ambiguous"], record["minting"]) == (ambiguous, _minting_record(minting))


def test_a_card_record_of_mintings_linked_out_of_order_names_the_one_it_refreshes_as():
    """A derived version linking the summed cut before the groupby records `minting` as the
    groupby, the minting this build refreshes it as, so a reader that predates `plain` refreshes
    it as the same minting."""
    rendering = StoredRendering(
        base_content_version="cv_base",
        collapsed=True,
        mintings=(_CUT_AND_SUMMED_MINTING, _SUMMED_MINTING),
    )

    assert rendering.to_record()["minting"] == _minting_record(_SUMMED_MINTING)
    assert rendering.refreshes_as == _SUMMED_MINTING


@pytest.mark.parametrize(
    "cause", [{"unmade": True}, {"agreed_on": "cv_other"}], ids=["unmade", "agreed-on"]
)
def test_a_rebuild_refusal_of_a_plain_frame_states_the_rebuild_cause(cause: dict[str, object]):
    """A refusal after a rebuild, of a rendering the rebuilt version cannot make or of
    renderings that agree on another version, states that cause for a raw version recording the
    plain rendering as for any other version."""
    named = {"spec_id": "spec_a", "content_version": "cv1", "renderings": 2}

    assert str(RenderingNotRecorded(**named, plain=True, **cause)) == str(
        RenderingNotRecorded(**named, **cause)
    )


def test_a_card_record_that_records_no_rendering_is_refused():
    record = {**StoredRendering(base_content_version="cv1").to_record(), "plain": False}

    with pytest.raises(ValueError):
        StoredRendering.from_record(record)


def test_a_card_record_from_before_the_plain_key_reads_its_plain_rendering():
    """A card revision recorded a raw version linking a plain file and a groupby file with no
    `plain` key; it reads back as recorded under both, refreshing as the plain rendering."""
    record = StoredRendering(
        base_content_version="cv1", mintings=(_SUMMED_MINTING,), plain=True
    ).to_record()
    del record["plain"]

    read = StoredRendering.from_record(record)

    assert (read.renderings, read.refreshes_as) == ((None, _SUMMED_MINTING), None)


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
