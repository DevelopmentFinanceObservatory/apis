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

`rendering_of` resolves the rendering a version is (`StoredRendering`) from its record and the
files linked under it, and `StoredRendering.to_record` and `from_record` carry it as the JSON
object a card revision stores. A version recorded under several renderings, which were one
frame when their files were linked under it, is all of them. The frame refresh answers it while
they make one frame from the head, and refreshes it as the plain rendering when that is one of
them, else as the first of its minting options by `ordered_mintings`. The frame refresh
(`datasets_api.doors.refresh`), the versions list (`datasets_api.doors.versions`) and the
description backfill (`datasets_api.maintenance`) resolve a version's options here.

`serving_frames` is the one rule for which stored frames serve a version and how a render reads
each, from the version record and the renderings linked under it, and `recorded_party_cut` the
cut a render of a derived version from its base frame replays.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from pydantic import ValidationError

from dataset_store.snapshot import ManifestFacts, StoredVersion
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.party_totals import TRACE_KEY, PartyTotalsCut
from datasets_api.render import (
    FrameMode,
    is_canonical_frame_rendering,
    is_frame1_rendering,
    is_statistics_rendering,
)
from dfo_platform.problem import Refusal

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


def minting_key(minting: MintingOptions) -> tuple[str, bool]:
    """A hashable key of `minting`'s options, the same for equal options: its groupby's
    canonical JSON, or `""`, and whether it leaves out party totals."""
    groupby = minting.groupby
    return (
        json.dumps(groupby.model_dump(mode="json"), sort_keys=True) if groupby is not None else "",
        minting.without_party_totals,
    )


def ordered_mintings(mintings: Iterable[MintingOptions]) -> tuple[MintingOptions, ...]:
    """Each different set of `mintings`, in an order of the options alone: a party-totals cut
    alone first, then each groupby by its canonical JSON, without the cut before with it. The
    order of the files that recorded them does not move it."""
    return tuple(sorted(_distinct(mintings), key=minting_key))


def _distinct(mintings: Iterable[MintingOptions]) -> tuple[MintingOptions, ...]:
    """Each different set of `mintings`, in the order each first appears."""
    distinct: list[MintingOptions] = []
    for minting in mintings:
        if minting not in distinct:
            distinct.append(minting)
    return tuple(distinct)


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


@dataclass(frozen=True)
class ServingFrame:
    """A stored frame that can serve a version: the frame stored under `content_version`,
    which a render reads as `mode` says."""

    content_version: str
    mode: FrameMode


def serving_frames(
    version: StoredVersion, *, renderings: Iterable[Mapping[str, object]]
) -> tuple[ServingFrame, ...]:
    """The stored frames that can serve `version`, in the order a reader tries them, of the
    version record and the `renderings` of the files linked under it alone.

    A raw version is served by its own frame (`FrameMode.RAW`). A derived version is served by
    its own frame when the record says what that frame holds (`own_frame_mode`), then by its
    base version's frame (`FrameMode.RAW`), from which every derived version renders anew."""
    base = version.base_content_version
    if base is None:
        return (ServingFrame(content_version=version.content_version, mode=FrameMode.RAW),)
    base_frame = ServingFrame(content_version=base, mode=FrameMode.RAW)
    own_mode = own_frame_mode(version, renderings=renderings)
    if own_mode is None:
        return (base_frame,)
    return (ServingFrame(content_version=version.content_version, mode=own_mode), base_frame)


def own_frame_mode(
    version: StoredVersion, *, renderings: Iterable[Mapping[str, object]]
) -> FrameMode | None:
    """What derived `version`'s own frame holds, or `None` when the record cannot say.

    `StoredVersion.frame_summed` says it when recorded. Without it, a version whose trace names
    no party totals left out is a collapse alone, so its frame is the collapse. A party-totals
    cut linking a file minted with no groupby is the cut frame, since that file's render minted
    the cut frame as this version. A cut whose files all record a groupby, or record no minting
    options, holds the collapse when the groupby summed and the cut frame when it did not, and
    the record does not say which."""
    if version.frame_summed is not None:
        return FrameMode.SUMMED if version.frame_summed else FrameMode.CUT
    # Only a dataset version names a base.
    assert isinstance(version.facts, ManifestFacts)
    if TRACE_KEY not in version.facts.trace:
        return FrameMode.SUMMED
    if any(minting.groupby is None for minting in recorded_mintings(renderings)):
        return FrameMode.CUT
    return None


def recorded_party_cut(version: StoredVersion) -> PartyTotalsCut:
    """The party-totals cut derived `version`'s trace records (`TRACE_KEY`), which a render of
    it from its base version's frame replays (`PartyTotalsCut.of_trace_entries`), so the cut
    frame hashes to the version it minted whatever the code index and the cut's rule now
    decide. A trace naming no totals records a cut that left nothing out."""
    # Only a dataset version names a base.
    assert isinstance(version.facts, ManifestFacts)
    return PartyTotalsCut.of_trace_entries(
        cast("list[Mapping[str, object]]", version.facts.trace.get(TRACE_KEY, []))
    )


class RenderingNotRecorded(Refusal):
    """The frame is not established as one rendering of the current data. Maps to
    `rendering_not_recorded` (422). Its detail names the case:

    - with neither keyword, a derived frame with no linked file recording its minting options;
    - with `renderings` and `plain`, a raw version recorded under that many renderings, the
      plain one among them, that are no longer one frame of the current data, whichever of the
      cases below it also is;
    - with `renderings`, a version recorded under that many renderings, which were one frame
      when recorded and make different frames from the current data;
    - with `renderings` and `agreed_on`, renderings that make one frame from the current data,
      `agreed_on`, which is not the version the rebuild served;
    - with `renderings` and `unmade`, a rendering the raw version a rebuild materialised cannot
      make;
    - with `unreadable_record`, a card revision's recorded rendering that names no rendering
      this build reads.
    """

    code = "rendering_not_recorded"

    def __init__(
        self,
        *,
        spec_id: str,
        content_version: str,
        renderings: int = 1,
        agreed_on: str | None = None,
        unmade: bool = False,
        plain: bool = False,
        unreadable_record: bool = False,
    ) -> None:
        self.renderings = renderings
        named = f"{spec_id!r} at content_version {content_version!r}"
        if unreadable_record:
            detail = f"the card's recorded rendering of {named} names no rendering this build reads"
        elif renderings > 1 and plain:
            detail = (
                f"the rendering of {named} is ambiguous: the version links a groupby sum or "
                "party-totals cut and a plain file, and the renderings of the current data are "
                "no longer one frame"
            )
        elif unmade:
            detail = (
                f"{named} was recorded under {renderings} renderings, and one of them cannot be "
                "made from the dataset as rebuilt"
            )
        elif agreed_on is not None:
            detail = (
                f"{named} was recorded under {renderings} renderings; they make "
                f"{agreed_on!r} from the current data, which is not the version the rebuild "
                "served"
            )
        elif renderings > 1:
            detail = (
                f"{named} was recorded under {renderings} renderings, which were one frame "
                "then; the current data renders them as different frames, so the frame is no "
                "longer one rendering"
            )
        else:
            detail = (
                f"{named} is a derived version, and no file linked under it records the "
                "groupby or party-totals cut it was minted with"
            )
        super().__init__(detail)


def frame_options(minting: MintingOptions | None) -> MintOptions:
    """The options a mint of a frame takes: the bare frame-layout parquet `ai` mints a frame
    with, plus the minting options."""
    if minting is None:
        return MintOptions(format="parquet", bundle=False, layout="frame")
    return MintOptions(
        format="parquet",
        bundle=False,
        groupby=minting.groupby,
        without_party_totals=minting.without_party_totals,
        layout="frame",
    )


@dataclass(frozen=True)
class StoredRendering:
    """The rendering of its dataset a stored frame is, read from its version record and the
    files linked under it.

    A version is recorded under each rendering a file linked under it records (`renderings`):
    the plain rendering (a raw version's own frame, recorded by a plain requested-format file or
    by no file of minting options) when `plain`, and each of `mintings`. Those renderings were
    one frame when their files were linked, since a version is the content its rows hash to. A
    rendering with no `mintings` is the plain one, so each rendering has one value.

    Raises:
        ValueError: `plain` with no `mintings`, which is the plain rendering already.
    """

    # The raw version the frame is, or was derived from.
    base_content_version: str
    # The frame's version is a derived version of its own, not a raw version.
    collapsed: bool = False
    # Each different set of minting options a file linked under the frame's version records,
    # in the order the files were linked.
    mintings: tuple[MintingOptions, ...] = ()
    # The plain rendering is recorded beside `mintings`.
    plain: bool = False

    def __post_init__(self) -> None:
        if self.plain and not self.mintings:
            raise ValueError("a rendering with no minting options is the plain one already")

    @cached_property
    def renderings(self) -> tuple[MintingOptions | None, ...]:
        """Every rendering the frame's version is recorded under, `None` for the plain one,
        which comes first when recorded, then `mintings` in `ordered_mintings` order. The first
        is the one the frame refreshes and rebuilds as."""
        plain: tuple[MintingOptions | None, ...] = (
            (None,) if self.plain or not self.mintings else ()
        )
        return plain + ordered_mintings(self.mintings)

    @property
    def minting(self) -> MintingOptions | None:
        """The first of `mintings`, `None` when there are none."""
        return next(iter(self.mintings), None)

    @property
    def ambiguous(self) -> bool:
        """Whether the frame's version is recorded under two or more renderings."""
        return len(self.renderings) > 1

    @property
    def refreshes_as(self) -> MintingOptions | None:
        """The rendering the frame refreshes and rebuilds as, the first of `renderings`."""
        return self.renderings[0]

    @property
    def options(self) -> MintOptions:
        """The options a mint of the rendering the frame refreshes as takes (`frame_options`)."""
        return frame_options(self.refreshes_as)

    def to_record(self) -> dict[str, object]:
        """This rendering as the JSON object a card revision stores
        (`presentation.component_revision.frame_rendering`), read back by `from_record`."""
        return {
            "base_content_version": self.base_content_version,
            "minting": _minting_record(self.minting),
            # A reader of records without `plain` reads `ambiguous` as the plain rendering
            # being one of several, and refreshes any other record as `minting`.
            "ambiguous": self.plain,
            "collapsed": self.collapsed,
            "mintings": [_minting_record(minting) for minting in self.mintings],
            "plain": None in self.renderings,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> StoredRendering:
        """The rendering `to_record` wrote. A record with no `plain` key was written when only
        a raw version linking a plain file was recorded under several renderings, so its
        `ambiguous`, or a `minting` of `None`, says the plain rendering is one of them.

        Raises:
            ValueError: `record` is not an object `to_record` writes, holds a groupby this
                build's grammar refuses, or records no rendering: no minting options and no
                plain rendering.
        """
        try:
            recorded = cast("Sequence[Mapping[str, object]]", record["mintings"])
            minting = _minting_of(cast("Mapping[str, object] | None", record["minting"]))
            ambiguous = cast(bool, record["ambiguous"])
            mintings = _distinct(cast(MintingOptions, _minting_of(entry)) for entry in recorded)
            plain = cast(bool, record.get("plain", ambiguous or minting is None))
            if not mintings and not plain:
                raise ValueError("the record names no rendering")
            return cls(
                base_content_version=cast(str, record["base_content_version"]),
                collapsed=cast(bool, record["collapsed"]),
                mintings=mintings,
                plain=plain and bool(mintings),
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise ValueError(f"not a stored rendering record: {exc}") from exc


def _minting_record(minting: MintingOptions | None) -> dict[str, object] | None:
    if minting is None:
        return None
    return {
        "groupby": (
            minting.groupby.model_dump(mode="json") if minting.groupby is not None else None
        ),
        "without_party_totals": minting.without_party_totals,
    }


def _minting_of(record: Mapping[str, object] | None) -> MintingOptions | None:
    if record is None:
        return None
    groupby = record["groupby"]
    return MintingOptions(
        groupby=GroupbyOptions.model_validate(groupby) if groupby is not None else None,
        without_party_totals=cast(bool, record["without_party_totals"]),
    )


def rendering_of(
    *, spec_id: str, version: StoredVersion, renderings: Iterable[Mapping[str, object]]
) -> StoredRendering:
    """The rendering of `version`, stored under `spec_id` and linking files of `renderings`.

    A derived version is recorded under each set of minting options its files record; a raw
    version under those and the plain rendering, which a plain requested-format file records
    and which a raw version linking no file of minting options is. The frame refreshes as the
    plain rendering when that is recorded, else as the first minting options by
    `ordered_mintings`; `StoredRendering.mintings` keeps the order the files were linked in.

    Raises:
        RenderingNotRecorded: The version is derived and no file linked under it records
            minting options.
    """
    content_version = version.content_version
    renderings = list(renderings)
    mintings = _distinct(recorded_mintings(renderings))
    if version.base_content_version is not None:
        if not mintings:
            raise RenderingNotRecorded(spec_id=spec_id, content_version=content_version)
        return StoredRendering(
            base_content_version=version.base_content_version, collapsed=True, mintings=mintings
        )
    return StoredRendering(
        base_content_version=content_version,
        mintings=mintings,
        plain=bool(mintings) and records_plain_file(renderings),
    )


__all__ = [
    "MintingOptions",
    "RenderingNotRecorded",
    "ServingFrame",
    "StoredRendering",
    "derived_version_sums",
    "frame_options",
    "minting_key",
    "ordered_mintings",
    "own_frame_mode",
    "recorded_minting",
    "recorded_mintings",
    "recorded_party_cut",
    "records_plain_file",
    "rendering_of",
    "serving_frames",
]
