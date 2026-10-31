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
object a card revision stores. The frame refresh (`datasets_api.doors.refresh`), the versions list
(`datasets_api.doors.versions`) and the description backfill (`datasets_api.maintenance`) resolve
a version's options here.

`serving_frames` is the one rule for which stored frames serve a version and how a render reads
each, from the version record and the renderings linked under it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

from dataset_store.snapshot import ManifestFacts, StoredVersion
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.party_totals import TRACE_KEY
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


class RenderingNotRecorded(Refusal):
    """The stored records do not say which rendering the frame is: a derived frame with no
    linked file recording its minting options, or with `ambiguous` a raw version linking a file
    of minting options and a plain file whose renderings of the current data differ, or with
    `sums` above one a derived version, or a raw version with no plain file, linking files of
    that many different minting options. Maps to `rendering_not_recorded` (422)."""

    code = "rendering_not_recorded"

    def __init__(
        self, *, spec_id: str, content_version: str, ambiguous: bool = False, sums: int = 1
    ) -> None:
        self.ambiguous = ambiguous
        if sums > 1:
            detail = (
                f"the rendering of {spec_id!r} at content_version {content_version!r} is "
                f"ambiguous: the version links files of {sums} different renderings that "
                "mint a version of their own, and its records do not say which the frame is"
            )
        elif ambiguous:
            detail = (
                f"the rendering of {spec_id!r} at content_version {content_version!r} is "
                "ambiguous: the version links a groupby sum or party-totals cut and a plain "
                "file, and the renderings of the current data are no longer one frame"
            )
        else:
            detail = (
                f"{spec_id!r} at content_version {content_version!r} is a derived version, "
                "and no file linked under it records the groupby or party-totals cut it was "
                "minted with"
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
    """The rendering of its dataset a stored frame is, read from its version record."""

    # The raw version the frame is, or was derived from.
    base_content_version: str
    # The groupby and party-totals cut the frame was minted with, `None` for the plain one.
    minting: MintingOptions | None
    # The frame's version is a raw one linking a plain file and a file of each of `mintings`,
    # so the frame is one of several renderings that were the same rows when it was minted.
    ambiguous: bool = False
    # The frame's version is a derived version of its own, not a raw version.
    collapsed: bool = False
    # Each different set of minting options a file linked under the frame's version records.
    # `minting` is the first.
    mintings: tuple[MintingOptions, ...] = ()

    @property
    def options(self) -> MintOptions:
        """The options a mint of this rendering takes (`frame_options`)."""
        return frame_options(self.minting)

    def to_record(self) -> dict[str, object]:
        """This rendering as the JSON object a card revision stores
        (`presentation.component_revision.frame_rendering`), read back by `from_record`."""
        return {
            "base_content_version": self.base_content_version,
            "minting": _minting_record(self.minting),
            "ambiguous": self.ambiguous,
            "collapsed": self.collapsed,
            "mintings": [_minting_record(minting) for minting in self.mintings],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> StoredRendering:
        """The rendering `to_record` wrote.

        Raises:
            ValueError: `record` is not an object `to_record` writes, or holds a groupby this
                build's grammar refuses.
        """
        try:
            mintings = cast("Sequence[Mapping[str, object]]", record["mintings"])
            return cls(
                base_content_version=cast(str, record["base_content_version"]),
                minting=_minting_of(cast("Mapping[str, object] | None", record["minting"])),
                ambiguous=cast(bool, record["ambiguous"]),
                collapsed=cast(bool, record["collapsed"]),
                mintings=tuple(cast(MintingOptions, _minting_of(minting)) for minting in mintings),
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
    """The rendering of `version`, stored under `spec_id` and linking files of `renderings`,
    by the rule `stored_rendering` states.

    Raises:
        RenderingNotRecorded: As `stored_rendering` does.
    """
    content_version = version.content_version
    renderings = list(renderings)
    mintings: tuple[MintingOptions, ...] = ()
    for recorded in recorded_mintings(renderings):
        if recorded not in mintings:
            mintings += (recorded,)
    minting = mintings[0] if mintings else None
    if version.base_content_version is not None:
        if minting is None:
            raise RenderingNotRecorded(spec_id=spec_id, content_version=content_version)
        if len(mintings) > 1:
            raise RenderingNotRecorded(
                spec_id=spec_id,
                content_version=content_version,
                ambiguous=True,
                sums=len(mintings),
            )
        return StoredRendering(
            base_content_version=version.base_content_version,
            minting=minting,
            collapsed=True,
            mintings=mintings,
        )
    plain_file = records_plain_file(renderings)
    if len(mintings) > 1 and not plain_file:
        raise RenderingNotRecorded(
            spec_id=spec_id, content_version=content_version, ambiguous=True, sums=len(mintings)
        )
    return StoredRendering(
        base_content_version=content_version,
        minting=minting,
        ambiguous=minting is not None and plain_file,
        mintings=mintings,
    )


__all__ = [
    "MintingOptions",
    "RenderingNotRecorded",
    "ServingFrame",
    "StoredRendering",
    "derived_version_sums",
    "frame_options",
    "own_frame_mode",
    "recorded_minting",
    "recorded_mintings",
    "records_plain_file",
    "rendering_of",
    "serving_frames",
]
