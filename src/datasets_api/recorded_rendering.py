"""Read the rendering a stored dataset version was minted as from the files linked under it.

A version record states only the raw version a groupby collapse came from
(`base_content_version`). The groupby itself is in the `rendering` of the requested-format file
linked under the version. The frame1 file's rendering is the same for every frame and the
statistics file's for every version, so neither records it.

The frame refresh (`datasets_api.doors.refresh`) and the versions list
(`datasets_api.doors.recipe`) both resolve a version's groupby here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import suppress

from pydantic import ValidationError

from datasets_api.mint_models import GroupbyOptions
from datasets_api.render import is_frame1_rendering, is_statistics_rendering

_GROUPBY_KEY = "groupby"


def recorded_groupbys(renderings: Iterable[Mapping[str, object]]) -> list[GroupbyOptions]:
    """Each groupby `renderings` record, in their order. A recorded groupby this build's
    grammar refuses counts as not recorded."""
    groupbys: list[GroupbyOptions] = []
    for rendering in renderings:
        recorded = rendering.get(_GROUPBY_KEY)
        if not isinstance(recorded, Mapping):
            continue
        with suppress(ValidationError):
            groupbys.append(GroupbyOptions.model_validate(recorded))
    return groupbys


def recorded_groupby(renderings: Iterable[Mapping[str, object]]) -> GroupbyOptions | None:
    """The groupby the first of `renderings` to hold one records, or `None` when none does."""
    return next(iter(recorded_groupbys(renderings)), None)


def records_plain_file(renderings: Iterable[Mapping[str, object]]) -> bool:
    """Whether one of `renderings` is a requested-format file with no groupby: any file other
    than the frame1 and statistics files."""
    return any(
        _GROUPBY_KEY not in rendering
        and not is_frame1_rendering(rendering)
        and not is_statistics_rendering(rendering)
        for rendering in renderings
    )


__all__ = ["recorded_groupby", "recorded_groupbys", "records_plain_file"]
