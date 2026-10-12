"""The snapshot door serves `GET /api/datasets/{id}/{content_version}`.

It reads the stored version and its servable artifacts in one `read_servable_version`
transaction and lists each artifact under its stable download link, signing nothing. The path
id resolves to the served ids the registry binds to it, and a version recorded under the posted
id itself answers too (`read_under_served_id`). A request naming the served id honours
revocations under it, and a request naming a posted id honours revocations under the posted id
and every served id bound to it. A quarantined path id, or a quarantined served id bound to
it, refuses `identity_quarantined`. The links
name the path id, so a click honours the revocations the request did (`render.linked_under`).
`output_options` uses the current capability document. Working artifacts expire 30 days after
`stored_at`. Pinned artifacts are permanent. An empty servable listable-artifact set returns
`artifact_expired`. Revocation returns `snapshot_revoked` before expiry is checked.

Reads run on the surface's read executor (`datasets_api.deps`). Full capacity or
shutdown returns retryable 503 `SnapshotBackpressure`. A broken executor logs and
raises an error mapped to 500.

It mounts after the meta door. Its `dfo_spec_id` converter accepts only
`spec_` ids, so operation ids cannot match its route.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from artifact_store import Tier
from code_index.shared import get_index
from dataset_store.clock import Clock
from dataset_store.snapshot import (
    ArtifactNotFound,
    ServableVersion,
    SnapshotNotFound,
    SnapshotReader,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
    read_servable_version,
    select_servable_locations,
)

# Register the route converter before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.artifact_delivery import LISTABLE_MEDIA_TYPES
from datasets_api.backpressure import Backpressure
from datasets_api.capabilities import build_capabilities
from datasets_api.codes_snapshot import ArtifactExpired, SpecNotFound
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.headers import no_store_headers
from datasets_api.manifest import stored_manifest, stored_manifest_facts, stored_output_profile
from datasets_api.manifest_models import Manifest
from datasets_api.output_options import derive_output_options
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.render import (
    DATA_LAYOUT_VERSION,
    FRAME_LAYOUT,
    LAYOUT_KEY,
    RENDITION_KEY,
    is_canonical_frame_rendering,
    is_incomplete_rendering,
    is_statistics_rendering,
    linked_under,
    rendition_token,
    stored_artifact_entry,
)
from datasets_api.subject_ids import served_subject_ids
from datasets_api.timings import door_call, run_with_timings
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from spec_registry import IdentityQuarantined
from spec_registry.reader import SpecReader

logger = logging.getLogger(__name__)


class SnapshotBackpressure(Backpressure):
    """A full snapshot semaphore maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the snapshot door is at its concurrent-read capacity right now; retry"


def _stored_snapshot_manifest(
    stored_version: StoredVersion,
    *,
    settings: DatasetsBuildSettings,
) -> Manifest:
    """The stored version's manifest with no artifacts, its `output_options` derived from its
    stored profile and current capabilities."""
    capabilities = build_capabilities(
        capability_version=DATASETS_CAPABILITY_VERSION,
        build=build_commit(),
        code_index_stamp=get_index().stamp,
        declarations=get_index().declarations,
        settings=settings,
    )
    profile = stored_output_profile(stored_manifest_facts(stored_version))
    return stored_manifest(
        stored_version,
        output_options=(
            derive_output_options(profile, capabilities=capabilities)
            if profile is not None
            else None
        ),
    )


def read_under_served_id[T](
    *,
    id: str,
    spec_reader: SpecReader,
    read: Callable[[str, tuple[str, ...]], T],
) -> T:
    """`read(subject_id, subject_ids)` under the first of `subject_ids`, the
    `served_subject_ids(id)`, whose record of the version answers, so a version-addressed door
    answers under the posted id what it answers under the served one. `read` raises
    `SnapshotNotFound` for a subject that does not record the version and `ArtifactNotFound`
    for one whose record does not link the asked digest; either moves on to the next subject,
    and the last subject's refusal propagates, except that a version some subject records
    refuses `ArtifactNotFound` where the last raises `SnapshotNotFound`.

    A request naming the served id honours revocations under it, and a request naming a posted
    id honours revocations under the posted id and every served id bound to it: `read` checks
    revocation under all of `subject_ids` (`revoked_under`) in the transaction that reads it,
    so moving on never passes a revocation. Raises `SpecNotFound` for an unregistered `id` and
    `IdentityQuarantined` when `id` or a served id bound to it is quarantined."""
    *earlier, last = served_subject_ids(id, spec_reader=spec_reader)
    subject_ids = (*earlier, last)
    unlinked: ArtifactNotFound | None = None
    for subject_id in earlier:
        try:
            return read(subject_id, subject_ids)
        except SnapshotNotFound:
            pass
        except ArtifactNotFound as exc:
            unlinked = unlinked or exc
    try:
        return read(last, subject_ids)
    except SnapshotNotFound:
        if unlinked is None:
            raise
        raise unlinked from None


def read_stored_version(
    *,
    id: str,
    content_version: str,
    spec_reader: SpecReader | None,
    snapshot_reader: SnapshotReader,
    now: datetime,
    revoked_under: Sequence[str] = (),
) -> ServableVersion:
    """The stored version at `content_version` of a registered `id`, neither it nor its base
    version revoked under `id` or any id in `revoked_under`, with its locations judged
    servable at `now` (`read_servable_version`).

    `spec_reader` checks that `id` is registered first. `None` skips that read, for a caller
    that has already read the definition.

    Raises `SpecNotFound` for an unregistered id, `SnapshotNotFound` for an unknown version,
    and `SnapshotRevoked` for a revoked one.
    """
    if spec_reader is not None and spec_reader.read_definition(id) is None:
        raise SpecNotFound(id=id)
    return read_servable_version(
        snapshot_reader,
        subject_id=id,
        content_version=content_version,
        now=now,
        revoked_under=revoked_under,
    )


def servable_listable_locations(
    locations: Sequence[StoredArtifactLocation], *, now: datetime, rendition: str
) -> list[StoredArtifactLocation]:
    """The caller-facing renderings among `locations` a snapshot lists at `now`, in listing
    order (`_listed_locations`). The canonical frame counts. A frame1 document never does. It
    has no entry in `LISTABLE_MEDIA_TYPES`. An unpinned bundle of a superseded rendition or
    unpinned incomplete artifact does not count either. A version for which this is empty
    answers `artifact_expired`."""
    listable = [loc for loc in locations if loc.media_type in LISTABLE_MEDIA_TYPES]
    return _listed_locations(select_servable_locations(listable, now=now), rendition=rendition)


def _superseded_bundle(loc: StoredArtifactLocation, *, rendition: str) -> bool:
    """A bundle rendered under another rendition than `rendition`, or before bundles carried
    one."""
    return loc.rendering.get("bundle") is True and loc.rendering.get(RENDITION_KEY) != rendition


def _listed_only_pinned(loc: StoredArtifactLocation, *, rendition: str) -> bool:
    """A superseded bundle, or an artifact a graph metadata read failed for
    (`render.incomplete_rendering`). No request's rendering key matches either, so a snapshot
    lists one only once a caller has pinned its bytes."""
    return _superseded_bundle(loc, rendition=rendition) or is_incomplete_rendering(loc.rendering)


def _old_bare_layout(loc: StoredArtifactLocation) -> bool:
    """A bare data file written in another layout than `DATA_LAYOUT_VERSION` or `FRAME_LAYOUT`,
    or before bare files carried one."""
    return (
        loc.rendering.get("bundle") is not True
        and not is_statistics_rendering(loc.rendering)
        and not is_canonical_frame_rendering(loc.rendering)
        and loc.rendering.get(LAYOUT_KEY) not in (DATA_LAYOUT_VERSION, FRAME_LAYOUT)
    )


def _listing_group(loc: StoredArtifactLocation, *, rendition: str) -> int:
    """Where `loc` sorts in a snapshot listing: 0 for a current data artifact in the long
    layout, 1 for a bare file in `FRAME_LAYOUT`, 2 for a superseded bundle, an incomplete
    artifact or an old bare file, 3 for the canonical frame, 4 for a statistics artifact."""
    if is_statistics_rendering(loc.rendering):
        return 4
    if is_canonical_frame_rendering(loc.rendering):
        return 3
    if _listed_only_pinned(loc, rendition=rendition) or _old_bare_layout(loc):
        return 2
    if loc.rendering.get(LAYOUT_KEY) == FRAME_LAYOUT:
        return 1
    return 0


def _listed_locations(
    servable: list[StoredArtifactLocation], *, rendition: str
) -> list[StoredArtifactLocation]:
    """The servable locations a snapshot lists, in order. Bare artifacts and bundles carrying
    the current `rendition` are listed. A superseded bundle and an incomplete artifact
    (`_listed_only_pinned`) are listed only when pinned. Current long-layout data artifacts
    come first, then frame-layout files, then superseded bundles, incomplete artifacts and
    bare files in an old layout (`_old_bare_layout`), then the canonical frame, then the
    statistics artifacts (`_listing_group`), so the first artifact matching a format is the
    current long layout. Old bare files stay listed, so a version holding only those can be
    opened. The sort is stable, so each group keeps the reader's order.

    An incomplete render and the complete render after it can give the same bytes, and a pin
    is per digest. An incomplete location whose digest a complete rendering also lists is
    left out, so those bytes are listed once, without the marker."""
    complete_digests = {
        loc.artifact_digest for loc in servable if not is_incomplete_rendering(loc.rendering)
    }
    listed = [
        loc
        for loc in servable
        if not _listed_only_pinned(loc, rendition=rendition)
        or (
            loc.tier == Tier.pinned.value
            and not (
                is_incomplete_rendering(loc.rendering) and loc.artifact_digest in complete_digests
            )
        )
    ]
    return sorted(listed, key=lambda loc: _listing_group(loc, rendition=rendition))


def _narrow_to_rendering(
    servable: Sequence[StoredArtifactLocation], *, rendering_key: str
) -> list[StoredArtifactLocation]:
    """The locations among `servable` matching `rendering_key`, followed by every statistics
    artifact among them. A statistics artifact has its own fixed key, distinct from every
    `MintOptions` rendering, so it is kept across renderings."""
    narrowed = [loc for loc in servable if loc.rendering_key == rendering_key]
    narrowed_keys = {loc.rendering_key for loc in narrowed}
    narrowed.extend(
        loc
        for loc in servable
        if is_statistics_rendering(loc.rendering) and loc.rendering_key not in narrowed_keys
    )
    return narrowed


def rendering_servable(
    *,
    subject_id: str,
    content_version: str,
    rendering_key: str,
    snapshot_reader: SnapshotReader,
    now: Clock,
    settings: DatasetsBuildSettings,
) -> bool:
    """Whether a snapshot read of `(subject_id, content_version)` narrowed to `rendering_key`
    lists an artifact other than statistics.

    It applies the selection `read_snapshot_manifest(rendering_key=...)` lists from, under the
    rendition `settings.site` gives (`servable_listable_locations`, then
    `_narrow_to_rendering`), so the check and the read agree. It reads artifact locations only:
    no spec or version lookup, no revocation check and no signing. A bundle linked only as
    incomplete, a bundle of a superseded rendition and an expired working-tier artifact all
    answer `False`.

    Derive `rendering_key` from the request's options with `render.artifact_rendering_key`.
    """
    servable = servable_listable_locations(
        snapshot_reader.read_artifact_locations(subject_id, content_version),
        now=now(),
        rendition=rendition_token(settings.site),
    )
    return any(
        not is_statistics_rendering(loc.rendering)
        for loc in _narrow_to_rendering(servable, rendering_key=rendering_key)
    )


def read_snapshot_manifest(
    *,
    id: str,
    content_version: str,
    spec_reader: SpecReader | None,
    snapshot_reader: SnapshotReader,
    now: Clock,
    settings: DatasetsBuildSettings,
    rendering_key: str | None = None,
    revoked_under: Sequence[str] = (),
) -> Manifest:
    """Load a snapshot and list its servable artifacts under their stable download links.

    `spec_reader` checks the id is registered (`read_stored_version`), and `revoked_under`
    names further ids whose revocation withdraws the version. One
    `read_servable_version` transaction checks the version and its revocation and reads its
    locations. Listing signs nothing. The download link checks revocation again at the click.

    The default read raises `ArtifactExpired` for an empty servable set. A
    rendering-specific read returns matching renderings and statistics
    artifacts (`_narrow_to_rendering`). An empty result tells the definition door to
    render that variant fresh. `rendering_servable` answers whether this read lists
    a non-statistics artifact. A bundle under a superseded rendition and an
    incomplete artifact are listed only when pinned, after the current artifacts
    (`_listed_locations`).
    """
    read_at = now()
    stored = read_stored_version(
        id=id,
        content_version=content_version,
        spec_reader=spec_reader,
        snapshot_reader=snapshot_reader,
        now=read_at,
        revoked_under=revoked_under,
    )
    manifest = _stored_snapshot_manifest(stored.version, settings=settings)
    servable = servable_listable_locations(
        stored.servable, now=read_at, rendition=rendition_token(settings.site)
    )
    if rendering_key is None:
        if not servable:
            raise ArtifactExpired(id=id, content_version=content_version)
    else:
        servable = _narrow_to_rendering(servable, rendering_key=rendering_key)
    artifacts = [
        stored_artifact_entry(location, subject_id=id, content_version=content_version)
        for location in servable
    ]
    return manifest.model_copy(update={"artifacts": artifacts})


def read_snapshot_currency(
    *,
    id: str,
    content_version: str,
    spec_reader: SpecReader | None,
    snapshot_reader: SnapshotReader,
    now: Clock,
    settings: DatasetsBuildSettings,
    revoked_under: Sequence[str] = (),
) -> Manifest:
    """Load stored manifest fields for write-free definition currency checks. `revoked_under`
    is `read_snapshot_manifest`'s.

    Check spec, version, and revocation once, then return an empty artifact list.
    """
    stored = read_stored_version(
        id=id,
        content_version=content_version,
        spec_reader=spec_reader,
        snapshot_reader=snapshot_reader,
        now=now(),
        revoked_under=revoked_under,
    )
    return _stored_snapshot_manifest(stored.version, settings=settings)


def build_snapshot_door(*, deps: DepsCell) -> DoorContribution:
    """Build the snapshot door over the surface's dependency cell. Reads run on the read
    executor."""
    router = APIRouter()

    @router.get("/api/datasets/{id:dfo_spec_id}/{content_version}", response_model=Manifest)
    async def get_snapshot(id: str, content_version: str) -> JSONResponse:
        current = deps.get()

        def _read(subject_id: str, subject_ids: tuple[str, ...]) -> Manifest:
            return read_snapshot_manifest(
                id=subject_id,
                content_version=content_version,
                spec_reader=None,
                snapshot_reader=current.snapshot_reader,
                now=current.now,
                settings=current.settings,
                revoked_under=subject_ids,
            )

        with door_call("snapshot", logger=logger) as call:
            manifest = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    read_under_served_id,
                    call.timings,
                    id=id,
                    spec_reader=current.spec_reader,
                    read=_read,
                ),
                refuse=SnapshotBackpressure,
            )
        return JSONResponse(
            linked_under(manifest, id).model_dump(mode="json", exclude_none=True),
            headers=no_store_headers(server_timing=call.server_timing),
        )

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.snapshot"),
        classify=classify_datasets_refusal,
        exception_types=(
            SpecNotFound,
            # The path id resolves through the spec registry, which raises it for a quarantined
            # path id or served id bound to it.
            IdentityQuarantined,
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            SnapshotBackpressure,
        ),
        exception_exemplars=(
            SpecNotFound(id="spec_exemplar"),
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotBackpressure(),
        ),
    )
