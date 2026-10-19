"""Publishes a release, holds or releases the pointer, and prunes old
releases. This is the only code that writes under `GRAPH_RELEASE_URI`.

A publish uploads the parts the bucket lacks, writes the manifest, then moves
the pointer with a generation precondition. Parts and manifests are
create-only, and the pointer write is the commit point: a run that stops
before it leaves the serving release untouched. A pointer with `hold` set
pins the release it names until `clear_hold`, and a publish leaves it alone.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from artifact_publication.uri import (
    delete_uri_if_unchanged,
    join_uri,
    list_uri_object_generations,
    list_uri_object_times,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes_if_unchanged,
)
from graph_release.format import (
    MANIFESTS_DIR,
    PARTS_DIR,
    Manifest,
    Pointer,
    ReleaseFormatError,
    StagedPart,
    manifest_uri,
    part_uri,
    pointer_uri,
    read_pointer,
)

logger = logging.getLogger(__name__)

KEEP_RELEASES: Final[int] = 10
"""Pruning keeps the manifests of this many most recently written releases."""

KEEP_SECONDS: Final[float] = 24 * 3600
"""Pruning keeps every manifest younger than this."""


class PublishStatus(StrEnum):
    PUBLISHED = "published"
    """The pointer names the release."""
    HELD = "held"
    """The pointer has `hold` set, so nothing was written."""
    POINTER_MOVED = "pointer_moved"
    """The pointer's generation is not the expected one. Parts and the
    manifest may have been written. The pointer was not."""


@dataclass(frozen=True)
class PublishOutcome:
    """What a publish did. `uploaded` lists the part file names this call
    wrote, so a republish of unchanged parts uploads none."""

    status: PublishStatus
    release_id: str
    uploaded: tuple[str, ...]


def publish_release(
    root_uri: str,
    *,
    staged: Sequence[StagedPart],
    manifest: Manifest,
    expected_pointer_generation: str | None,
) -> PublishOutcome:
    """Publishes `manifest` with the part files in `staged`.

    `expected_pointer_generation` is the pointer's token from the caller's
    read (`None` when it read no pointer). The pointer is checked before
    anything is uploaded, so a held or moved pointer costs one read.
    `staged` must hold exactly the parts the manifest names. Raises
    `ValueError` when it does not, and `ReleaseFormatError` when the pointer
    or an existing manifest under this release id does not parse.
    """
    release_id = manifest.release_id
    staged_names = {part.ref.file_name for part in staged}
    named = {ref.file_name for ref in manifest.parts}
    if staged_names != named:
        raise ValueError(
            f"staged parts differ from the manifest's: {len(staged_names - named)} staged "
            f"part(s) unnamed, {len(named - staged_names)} named part(s) unstaged"
        )

    found = read_uri_bytes_and_generation(pointer_uri(root_uri))
    if (found[1] if found is not None else None) != expected_pointer_generation:
        return PublishOutcome(PublishStatus.POINTER_MOVED, release_id, ())
    if found is not None and Pointer.from_json(found[0]).hold is not None:
        return PublishOutcome(PublishStatus.HELD, release_id, ())

    # One listing finds the parts earlier releases share, so a republish
    # uploads none of them. A create-only write that loses to a concurrent
    # writer still leaves the right bytes, since a part's name is its hash.
    present = list_uri_object_times(join_uri(root_uri, PARTS_DIR))
    uploaded = []
    for part in sorted(staged, key=lambda part: part.ref.file_name):
        if part.ref.file_name in present:
            continue
        if write_uri_bytes_if_unchanged(
            part_uri(root_uri, part.ref), part.path.read_bytes(), expected_generation=None
        ):
            uploaded.append(part.ref.file_name)

    # A manifest already under this id names the same parts, since the id
    # hashes them, so it is kept as written and the pointer moves to it.
    if not write_uri_bytes_if_unchanged(
        manifest_uri(root_uri, release_id), manifest.to_json(), expected_generation=None
    ):
        Manifest.from_json(read_uri_bytes(manifest_uri(root_uri, release_id)))

    if not write_uri_bytes_if_unchanged(
        pointer_uri(root_uri),
        Pointer(release_id=release_id).to_json(),
        expected_generation=expected_pointer_generation,
    ):
        return PublishOutcome(PublishStatus.POINTER_MOVED, release_id, tuple(uploaded))
    return PublishOutcome(PublishStatus.PUBLISHED, release_id, tuple(uploaded))


class HoldError(Exception):
    """A hold or clear that cannot land: an unknown release, no pointer, or a
    pointer that moved between the read and the write."""


def hold_release(root_uri: str, release_id: str) -> Pointer:
    """Points at `release_id` and holds it there. Returns the pointer written.
    Raises `HoldError` when no manifest names the release or the pointer
    moved during the call, and `ReleaseFormatError` when the manifest does
    not parse."""
    try:
        Manifest.from_json(read_uri_bytes(manifest_uri(root_uri, release_id)))
    except FileNotFoundError as exc:
        raise HoldError(f"no manifest for release {release_id!r}") from exc
    found = read_uri_bytes_and_generation(pointer_uri(root_uri))
    pointer = Pointer(release_id=release_id, hold=release_id)
    _write_pointer(root_uri, pointer, expected_generation=found[1] if found else None)
    return pointer


def clear_hold(root_uri: str) -> Pointer:
    """Clears `hold`, leaving the held release serving until the builder
    publishes the next one. Returns the pointer as it now stands. Raises
    `HoldError` when no pointer exists or it moved during the call."""
    found = read_uri_bytes_and_generation(pointer_uri(root_uri))
    if found is None:
        raise HoldError(f"no release pointer at {pointer_uri(root_uri)}")
    current = Pointer.from_json(found[0])
    if current.hold is None:
        return current
    pointer = Pointer(release_id=current.release_id)
    _write_pointer(root_uri, pointer, expected_generation=found[1])
    return pointer


def _write_pointer(root_uri: str, pointer: Pointer, *, expected_generation: str | None) -> None:
    if not write_uri_bytes_if_unchanged(
        pointer_uri(root_uri), pointer.to_json(), expected_generation=expected_generation
    ):
        raise HoldError("the release pointer moved during this call; nothing was written")


def prune_releases(root_uri: str, *, now: float) -> tuple[str, ...]:
    """Deletes the manifests pruning does not keep and the parts no kept
    manifest names, and returns their paths under `root_uri`, sorted.

    Kept: the `KEEP_RELEASES` most recently written manifests, any younger
    than `KEEP_SECONDS`, and the pointer's release, which is the held one
    when the pointer is held.
    Each delete is conditional on the token listed, so an object rewritten
    since stays. A missing or unreadable pointer, or a manifest this reader
    cannot parse (such as one of another format), deletes nothing.

    Run it while holding the build lease. A publish running beside it could
    reuse an old part this deletes before its manifest names it.
    """
    try:
        pointer = read_pointer(root_uri)
    except ReleaseFormatError:
        logger.warning("prune: the release pointer does not parse; pruning nothing", exc_info=True)
        return ()
    if pointer is None:
        return ()

    manifests = list_uri_object_generations(join_uri(root_uri, MANIFESTS_DIR))
    parts_by_manifest: dict[str, frozenset[str]] = {}
    for name in manifests:
        try:
            manifest = Manifest.from_json(read_uri_bytes(join_uri(root_uri, MANIFESTS_DIR, name)))
        except (FileNotFoundError, ReleaseFormatError):
            logger.warning(
                "prune: manifest %r does not parse; pruning nothing", name, exc_info=True
            )
            return ()
        parts_by_manifest[name] = frozenset(ref.file_name for ref in manifest.parts)

    newest_first = sorted(manifests, key=lambda name: manifests[name][0], reverse=True)
    kept = {
        *newest_first[:KEEP_RELEASES],
        *(name for name, (written, _) in manifests.items() if now - written < KEEP_SECONDS),
        f"{pointer.release_id}.json",
    }
    named = {part for name in kept & manifests.keys() for part in parts_by_manifest[name]}

    # Manifests go first, so no manifest left behind names a deleted part.
    deleted = _delete_unkept(root_uri, MANIFESTS_DIR, manifests, kept=kept)
    parts = list_uri_object_generations(join_uri(root_uri, PARTS_DIR))
    return (*deleted, *_delete_unkept(root_uri, PARTS_DIR, parts, kept=named))


def _delete_unkept(
    root_uri: str, directory: str, listed: Mapping[str, tuple[float, str]], *, kept: Set[str]
) -> list[str]:
    """Deletes each listed object under `directory` that `kept` lacks, at its
    listed token, and returns the paths it deleted, sorted."""
    return [
        f"{directory}/{name}"
        for name in sorted(listed.keys() - kept)
        if delete_uri_if_unchanged(
            join_uri(root_uri, directory, name), expected_generation=listed[name][1]
        )
    ]


__all__ = [
    "KEEP_RELEASES",
    "KEEP_SECONDS",
    "HoldError",
    "PublishOutcome",
    "PublishStatus",
    "clear_hold",
    "hold_release",
    "prune_releases",
    "publish_release",
]
