"""Read the rendering a stored dataset version was minted as from the files linked under it.

Two options mint a version of their own (`datasets_api.render.DerivedVersion`): a `groupby`
collapse and a party-totals cut (`without_party_totals`), alone or together. A version record
states only the raw version a derived version came from (`base_content_version`). The options
themselves are in the `rendering` of the requested-format file linked under the version, which
records `without_party_totals` only when true (`render.artifact_rendering`). The stored frame's
rendering (canonical frame or frame1) is the same for every frame and the statistics file's for
every version, so neither records them. A file linked as incomplete
(`render.incomplete_rendering`) holds the same rows as its complete rendering, so it records
the options and counts as a requested-format file.

The frame refresh (`datasets_api.doors.refresh`), the versions list
(`datasets_api.doors.recipe`) and the description backfill (`datasets_api.maintenance`) resolve
a version's options here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from datasets_api.mint_models import GroupbyOptions
from datasets_api.party_totals import TRACE_KEY
from datasets_api.render import (
    is_canonical_frame_rendering,
    is_frame1_rendering,
    is_statistics_rendering,
)

_GROUPBY_KEY = "groupby"
_WITHOUT_PARTY_TOTALS_KEY = "without_party_totals"


@dataclass(frozen=True)
class MintingOptions:
    """The options of a rendering that mint a version of their own: its `groupby`, and
    whether it leaves out party totals. A plain rendering holds neither."""

    groupby: GroupbyOptions | None = None
    without_party_totals: bool = False


def _recorded(rendering: Mapping[str, object]) -> MintingOptions | None:
    """The minting options `rendering` records, or `None` when it records none, or a groupby
    this build's grammar refuses."""
    groupby = rendering.get(_GROUPBY_KEY)
    without_party_totals = rendering.get(_WITHOUT_PARTY_TOTALS_KEY) is True
    if groupby is None:
        return MintingOptions(without_party_totals=True) if without_party_totals else None
    if not isinstance(groupby, Mapping):
        return None
    try:
        recorded = GroupbyOptions.model_validate(groupby)
    except ValidationError:
        return None
    return MintingOptions(groupby=recorded, without_party_totals=without_party_totals)


def recorded_mintings(renderings: Iterable[Mapping[str, object]]) -> list[MintingOptions]:
    """The minting options each of `renderings` records, in their order. A rendering whose
    groupby this build's grammar refuses counts as recording none."""
    return [recorded for rendering in renderings if (recorded := _recorded(rendering))]


def recorded_minting(renderings: Iterable[Mapping[str, object]]) -> MintingOptions | None:
    """The minting options the first of `renderings` to record some records, or `None`."""
    return next(iter(recorded_mintings(renderings)), None)


def records_plain_file(renderings: Iterable[Mapping[str, object]]) -> bool:
    """Whether one of `renderings` is a requested-format file with no minting options: any file
    other than the stored frame and statistics files that records no groupby and leaves no
    party totals out."""
    return any(
        _GROUPBY_KEY not in rendering
        and rendering.get(_WITHOUT_PARTY_TOTALS_KEY) is not True
        and not is_canonical_frame_rendering(rendering)
        and not is_frame1_rendering(rendering)
        and not is_statistics_rendering(rendering)
        for rendering in renderings
    )


def derived_version_sums(
    trace: Mapping[str, object], renderings: Iterable[Mapping[str, object]]
) -> bool:
    """Whether a derived version with `trace`, linking files of `renderings`, is a collapse
    that sums its definition's variables. A version whose trace names no party totals left out
    is a collapse. One that names some is a cut, and a collapse of it when a file linked under
    it records a groupby; a groupby that left the cut frame unchanged is recorded the same way
    and reads as a collapse here."""
    if TRACE_KEY not in trace:
        return True
    return any(minting.groupby is not None for minting in recorded_mintings(renderings))


__all__ = [
    "MintingOptions",
    "derived_version_sums",
    "recorded_minting",
    "recorded_mintings",
    "records_plain_file",
]
