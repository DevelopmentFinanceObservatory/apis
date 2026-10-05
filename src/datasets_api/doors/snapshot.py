"""The snapshot door serves `GET /api/datasets/{id}/{content_version}`.

It reads the stored version, joins servable artifacts, and signs their URLs
fresh. `output_options` uses the current capability document. Working artifacts
expire 30 days after `stored_at`. Pinned artifacts are permanent. An empty
servable listable-artifact set returns `artifact_expired`. Revocation returns
`snapshot_revoked` before expiry is checked.

Reads use a dedicated executor with `DATASETS_SNAPSHOT_MAX_CONCURRENCY` workers
(default 4) and a same-sized non-blocking semaphore. Full capacity or shutdown
returns retryable 503 `SnapshotBackpressure`. A broken executor logs and raises
an error mapped to 500. Completion callbacks release permits when work ends.
Shutdown cancels queued calls, then drains running calls within the shared
budget before closing the store and pool.

It mounts after the meta door. Its `dfo_subject_id` converter accepts only
`spec_` and `col_` ids, so operation ids cannot match its route.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import cast

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from artifact_store import (
    ArtifactSigningError,
    ArtifactStore,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from code_index.shared import get_index
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import (
    LiveSnapshotReader,
    SnapshotReader,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
    select_servable_locations,
)

# Register the route converter before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.artifact_delivery import (
    LISTABLE_MEDIA_TYPES,
    artifact_content_disposition,
)
from datasets_api.capabilities import build_capabilities
from datasets_api.codes_materialise import signing_failure_is_permanent
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    SnapshotNotFound,
    SnapshotSigningFailed,
    SpecNotFound,
)
from datasets_api.config import (
    SIGNED_URL_ADVERTISED_TTL,
    SIGNED_URL_TTL,
    DatasetsBuildSettings,
    DatasetsConfigError,
    DatasetsSettings,
)
from datasets_api.door_executor import build_door_executor
from datasets_api.headers import no_store_headers
from datasets_api.manifest import decode_collection_version_body, decode_version_body
from datasets_api.manifest_models import ArtifactEntryModel, Manifest
from datasets_api.output_options import derive_output_options
from datasets_api.render import (
    MEMBER_RENDERING_KEY,
    RENDITION_KEY,
    is_statistics_rendering,
    rendition_token,
    stored_artifact_file_name,
)
from datasets_api.shutdown import ShutdownBudget
from datasets_api.timings import door_call, phase, run_with_timings
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import rendering_key as compute_rendering_key
from dfo_platform import ProblemSpec
from dfo_platform.build import build_commit
from dfo_platform.doors import DoorContribution
from spec_registry.reader import LiveSpecReader, SpecReader

logger = logging.getLogger(__name__)


class SnapshotBackpressure(Exception):
    """A full snapshot semaphore maps to retryable `service_unavailable` (503)."""


def classify_snapshot_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, SpecNotFound):
        return ProblemSpec(status=404, code="spec_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, SnapshotNotFound):
        return ProblemSpec(status=404, code="snapshot_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, SnapshotRevoked):
        return ProblemSpec(status=410, code="snapshot_revoked", detail=str(exc), retryable=False)
    if isinstance(exc, ArtifactExpired):
        return ProblemSpec(status=410, code="artifact_expired", detail=str(exc), retryable=False)
    if isinstance(exc, SnapshotBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the snapshot door is at its concurrent-read capacity right now; retry",
            retryable=True,
        )
    if isinstance(exc, SnapshotSigningFailed):
        # Stored bytes are unchanged. Permanent signing failures map to 500.
        # Transient or unknown causes map to retryable 503.
        if signing_failure_is_permanent(
            error_class_name=exc.error_class_name, status_code=exc.status_code
        ):
            logger.error(
                "Datasets snapshot door failed to sign a GET url for a stored artifact "
                "for a permanent, operator-clearable reason: %s",
                exc,
                exc_info=exc,
            )
            return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)
        logger.error(
            "Datasets snapshot door failed to sign a GET url for a stored artifact: %s",
            exc,
            exc_info=exc,
        )
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="signing this snapshot's artifact download url failed; retry",
            retryable=True,
        )
    return None


def _id_is_registered(id: str, *, spec_reader: SpecReader) -> bool:
    if id.startswith("col_"):
        return spec_reader.read_collection(id) is not None
    return spec_reader.read_definition(id) is not None


def _decode_manifest_fields(
    stored_version: StoredVersion,
    *,
    settings: DatasetsBuildSettings,
) -> dict[str, object]:
    """Decode stored manifest fields and derive `output_options` from current capabilities."""
    capabilities = build_capabilities(
        capability_version=DATASETS_CAPABILITY_VERSION,
        build=build_commit(),
        code_index_stamp=get_index().stamp,
        declarations=get_index().declarations,
        settings=settings,
    )
    if stored_version.subject_kind == "collection":
        fields, profiles = decode_collection_version_body(stored_version.immutable_body)
        collection = cast(dict[str, object], fields["collection"])
        members = cast(list[dict[str, object]], collection["members"])
        for member, profile in zip(members, profiles, strict=True):
            member["output_options"] = (
                derive_output_options(profile, capabilities=capabilities).model_dump(mode="json")
                if profile is not None
                else None
            )
        return fields

    fields, profile = decode_version_body(stored_version.immutable_body)
    fields["output_options"] = (
        derive_output_options(profile, capabilities=capabilities).model_dump(mode="json")
        if profile is not None
        else None
    )
    return fields


def _refuse_if_revoked(
    *,
    id: str,
    content_version: str,
    base_content_version: str | None,
    snapshot_reader: SnapshotReader,
) -> None:
    """Refuse a revoked version, and a collapsed version whose base version is revoked.

    The error names the requested version and carries the revoked version's reason.
    `base_content_version` is `None` for an uncollapsed version or a collection.
    """
    revocation = snapshot_reader.read_revocation(id, content_version)
    if revocation is None and base_content_version is not None:
        revocation = snapshot_reader.read_revocation(id, base_content_version)
    if revocation is not None:
        raise SnapshotRevoked(id=id, content_version=content_version, reason=revocation.reason)


def _base_content_version(stored_version: StoredVersion) -> str | None:
    """The version a fully collapsed groupby version derives from, or `None` for a collection."""
    if stored_version.subject_kind == "collection":
        return None
    return stored_version.base_content_version


def read_stored_version(
    *,
    id: str,
    content_version: str,
    spec_reader: SpecReader,
    snapshot_reader: SnapshotReader,
) -> StoredVersion:
    """The stored version at `content_version` of a registered `id`, neither it nor its base
    version revoked.

    Raises `SpecNotFound` for an unregistered id, `SnapshotNotFound` for an unknown version,
    and `SnapshotRevoked` for a revoked one, checked before any artifact read.
    """
    if not _id_is_registered(id, spec_reader=spec_reader):
        raise SpecNotFound(id=id)
    stored_version = snapshot_reader.read_version(id, content_version)
    if stored_version is None:
        raise SnapshotNotFound(id=id, content_version=content_version)
    _refuse_if_revoked(
        id=id,
        content_version=content_version,
        base_content_version=_base_content_version(stored_version),
        snapshot_reader=snapshot_reader,
    )
    return stored_version


def servable_listable_locations(
    locations: Sequence[StoredArtifactLocation], *, now: datetime, rendition: str
) -> list[StoredArtifactLocation]:
    """The caller-facing renderings among `locations` a snapshot lists at `now`, in listing
    order (`_listed_locations`). Internal canonical-JSON renderings have no entry in
    `LISTABLE_MEDIA_TYPES` and never count, and an unpinned bundle of a superseded rendition
    does not either. A version for which this is empty answers `artifact_expired`."""
    listable = [loc for loc in locations if loc.media_type in LISTABLE_MEDIA_TYPES]
    return _listed_locations(select_servable_locations(listable, now=now), rendition=rendition)


def _rendering_without_member(rendering: Mapping[str, object]) -> Mapping[str, object]:
    """Remove the member-specific key before comparing a collection artifact's rendering."""
    if MEMBER_RENDERING_KEY not in rendering:
        return rendering
    return {key: value for key, value in rendering.items() if key != MEMBER_RENDERING_KEY}


def _superseded_bundle(loc: StoredArtifactLocation, *, rendition: str) -> bool:
    """A bundle rendered under another rendition than `rendition`, or before bundles carried
    one."""
    return loc.rendering.get("bundle") is True and loc.rendering.get(RENDITION_KEY) != rendition


def _listed_locations(
    servable: list[StoredArtifactLocation], *, rendition: str
) -> list[StoredArtifactLocation]:
    """The servable locations a snapshot lists, in order. Bare artifacts and bundles carrying
    the current `rendition` are listed. A superseded bundle (`_superseded_bundle`) is listed
    only when pinned, after every current data artifact and before the statistics artifacts,
    so the first bundle matching a format is the current layout. The sort is stable, so each
    group keeps the reader's order."""
    listed = [
        loc
        for loc in servable
        if loc.tier == Tier.pinned.value or not _superseded_bundle(loc, rendition=rendition)
    ]
    return sorted(
        listed,
        key=lambda loc: (
            is_statistics_rendering(loc.rendering),
            _superseded_bundle(loc, rendition=rendition),
        ),
    )


def _location_matches_rendering_key(loc: StoredArtifactLocation, rendering_key: str) -> bool:
    """Match the request's rendering key to a stored artifact.

    Stored keys match directly. Bare-collection member artifacts match after
    stripping `MEMBER_RENDERING_KEY` and rehashing.
    """
    if loc.rendering_key == rendering_key:
        return True
    return (
        MEMBER_RENDERING_KEY in loc.rendering
        and compute_rendering_key(_rendering_without_member(loc.rendering)) == rendering_key
    )


def _narrow_to_rendering(
    servable: Sequence[StoredArtifactLocation], *, rendering_key: str
) -> list[StoredArtifactLocation]:
    """The locations among `servable` matching `rendering_key`
    (`_location_matches_rendering_key`), followed by every statistics artifact among them.
    Each definition's or member's statistics artifact has its own fixed key, distinct from
    every `MintOptions` rendering, so it is kept across renderings."""
    narrowed = [loc for loc in servable if _location_matches_rendering_key(loc, rendering_key)]
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
    no spec or version lookup, no revocation check and no signing. An unlinked bundle, a bundle
    of a superseded rendition and an expired working-tier artifact all answer `False`.

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
    spec_reader: SpecReader,
    snapshot_reader: SnapshotReader,
    artifact_store: ArtifactStore,
    now: Clock,
    settings: DatasetsBuildSettings,
    rendering_key: str | None = None,
) -> Manifest:
    """Load a snapshot, check revocation, and sign its servable artifacts.

    Revocation is checked before loading artifacts and again before signing.
    Signed URLs outlive revocation until expiry. `SIGNED_URL_ADVERTISED_TTL`
    bounds that access window.

    The default read raises `ArtifactExpired` for an empty servable set. A
    rendering-specific read returns matching renderings and statistics
    artifacts (`_narrow_to_rendering`). An empty result tells the definition door to
    render that variant fresh. `rendering_servable` answers whether this read lists
    a non-statistics artifact without signing. A bundle under a superseded rendition is
    listed only when pinned, after the current artifacts (`_listed_locations`).
    """
    stored_version = read_stored_version(
        id=id,
        content_version=content_version,
        spec_reader=spec_reader,
        snapshot_reader=snapshot_reader,
    )
    fields = _decode_manifest_fields(stored_version, settings=settings)
    base_content_version = _base_content_version(stored_version)

    read_at = now()
    servable = servable_listable_locations(
        snapshot_reader.read_artifact_locations(id, content_version),
        now=read_at,
        rendition=rendition_token(settings.site),
    )
    if rendering_key is None:
        if not servable:
            raise ArtifactExpired(id=id, content_version=content_version)
    else:
        servable = _narrow_to_rendering(servable, rendering_key=rendering_key)

    # Recheck immediately before signing to catch revocation during artifact reads.
    _refuse_if_revoked(
        id=id,
        content_version=content_version,
        base_content_version=base_content_version,
        snapshot_reader=snapshot_reader,
    )

    manifest = Manifest.model_validate({**fields, "artifacts": []})
    signing_failed = False
    signing_bucket: str | None = None
    signing_object_name: str | None = None
    signing_digest: str | None = None
    signing_error_class_name: str | None = None
    signing_status_code: int | None = None
    try:
        with phase("sign"):
            artifacts = []
            for location in servable:
                file_name = stored_artifact_file_name(location, manifest=manifest)
                artifacts.append(
                    ArtifactEntryModel(
                        artifact_digest=location.artifact_digest,
                        rendering=dict(location.rendering),
                        bytes=location.size_bytes,
                        media_type=location.media_type,
                        # The manifest uses first-sighting time. Retention uses
                        # this tier's `stored_at`. Pinning does not reset creation time.
                        created_at=location.created_at.isoformat(),
                        tier=location.tier,
                        url=artifact_store.signed_get_url(
                            location.artifact_digest,
                            tier=Tier(location.tier),
                            ttl=SIGNED_URL_TTL,
                            response_type=location.media_type,
                            response_disposition=artifact_content_disposition(file_name),
                        ),
                        url_expires_at=(read_at + SIGNED_URL_ADVERTISED_TTL).isoformat(),
                    )
                )
    except ArtifactSigningError as exc:
        # Copy safe fields here, then raise outside the handler so Python
        # leaves `__context__` unset.
        signing_bucket = exc.bucket
        signing_object_name = exc.object_name
        signing_digest = exc.digest
        signing_error_class_name = exc.error_class_name
        signing_status_code = exc.status_code
        signing_failed = True
    if signing_failed:
        raise SnapshotSigningFailed(
            bucket=cast(str, signing_bucket),
            object_name=cast(str, signing_object_name),
            digest=cast(str, signing_digest),
            error_class_name=signing_error_class_name,
            status_code=signing_status_code,
        )
    return manifest.model_copy(update={"artifacts": artifacts})


def read_snapshot_currency(
    *,
    id: str,
    content_version: str,
    spec_reader: SpecReader,
    snapshot_reader: SnapshotReader,
    settings: DatasetsBuildSettings,
) -> Manifest:
    """Load stored manifest fields for write-free definition currency checks.

    Check spec, version, and revocation once, then return an empty artifact list.
    """
    stored_version = read_stored_version(
        id=id,
        content_version=content_version,
        spec_reader=spec_reader,
        snapshot_reader=snapshot_reader,
    )
    fields = _decode_manifest_fields(stored_version, settings=settings)
    return Manifest.model_validate({**fields, "artifacts": []})


def build_snapshot_door(
    *,
    settings: DatasetsBuildSettings | None = None,
    spec_reader: SpecReader | None = None,
    snapshot_reader: SnapshotReader | None = None,
    artifact_store: ArtifactStore | None = None,
    now: Clock = utc_now,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the snapshot door and its executor.

    Inject `spec_reader`, `snapshot_reader`, and `artifact_store` together, or
    leave all three unset for lifespan to build them. A partial set raises
    `DatasetsConfigError` at construction. Lifespan builds all three with one
    pool borrow. The artifact store has its own `LiveArtifactStore` instance.
    `settings` defaults from the environment. `pool_size` defaults to
    `settings.snapshot_max_concurrency`.

    `budget` is shared across mounted doors. An omitted budget creates a
    private shutdown budget.
    """
    missing = [
        name
        for name, value in (
            ("spec_reader", spec_reader),
            ("snapshot_reader", snapshot_reader),
            ("artifact_store", artifact_store),
        )
        if value is None
    ]
    if missing and len(missing) != 3:
        raise DatasetsConfigError(
            "datasets snapshot door built with "
            f"{', '.join(missing)} left uninjected while another dependency is injected. A "
            "caller injecting any one of spec_reader/snapshot_reader/artifact_store must "
            "inject every one of them. This door builds a live dependency for all three "
            "only when every one of them is left None."
        )

    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    _spec_reader: SpecReader | None = spec_reader
    _snapshot_reader: SnapshotReader | None = snapshot_reader
    _artifact_store: ArtifactStore | None = artifact_store

    effective_pool_size = pool_size if pool_size is not None else _settings.snapshot_max_concurrency
    door_executor = build_door_executor(effective_pool_size, "snapshot")
    executor = door_executor.executor

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _spec_reader, _snapshot_reader, _artifact_store
        # One pool borrow serves the readers and artifact inventory.
        # The all-or-nothing check makes `spec_reader is None` the live-store trigger.
        pool_opened_here = False
        store: LiveDatasetStore | None = None

        def _close_pool_if_opened_here() -> None:
            if pool_opened_here:
                close_shared_pool()

        if spec_reader is None:
            store = LiveDatasetStore(connections=shared_pool())
            pool_opened_here = True
            _spec_reader = LiveSpecReader(store=store)
            if snapshot_reader is None:
                _snapshot_reader = LiveSnapshotReader(store=store)
        if artifact_store is None:
            try:
                live_settings = DatasetsSettings.from_env()
                if store is None:
                    store = LiveDatasetStore(connections=shared_pool())
                    pool_opened_here = True
                _artifact_store = LiveArtifactStore(
                    buckets={
                        Tier.working: live_settings.artifact_bucket_working,
                        Tier.pinned: live_settings.artifact_bucket_pinned,
                    },
                    inventory=LiveArtifactInventory(store=store),
                )
            except BaseException:
                # Release the readers' pool borrow if artifact-store setup fails.
                _close_pool_if_opened_here()
                raise
        try:
            yield
        finally:
            # Cancel queued calls. Drain running calls before closing their
            # store and pool dependencies.
            executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(door_executor.drain, _budget)
            _close_pool_if_opened_here()

    router = APIRouter()

    @router.get("/api/datasets/{id:dfo_subject_id}/{content_version}", response_model=Manifest)
    async def get_snapshot(id: str, content_version: str) -> JSONResponse:
        with door_call("snapshot", logger=logger) as call:
            manifest = await door_executor.run(
                functools.partial(
                    run_with_timings,
                    read_snapshot_manifest,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    spec_reader=_spec_reader,
                    snapshot_reader=_snapshot_reader,
                    artifact_store=_artifact_store,
                    now=now,
                    settings=_settings,
                ),
                refuse=SnapshotBackpressure,
            )
        return JSONResponse(
            manifest.model_dump(mode="json", exclude_none=True),
            headers=no_store_headers(server_timing=call.server_timing),
        )

    return DoorContribution(
        router=router,
        classify=classify_snapshot_exception,
        exception_types=(
            SpecNotFound,
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            SnapshotBackpressure,
            SnapshotSigningFailed,
            # Wrapped before classification, but signing can raise it.
            ArtifactSigningError,
        ),
        exception_exemplars=(
            SpecNotFound(id="spec_exemplar"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotBackpressure(),
            SnapshotSigningFailed(
                bucket="exemplar-bucket",
                object_name="sha256:exemplar",
                digest="sha256:exemplar",
                error_class_name=None,
                status_code=None,
            ),
            ArtifactSigningError(
                bucket="exemplar-bucket", object_name="sha256:exemplar", digest="sha256:exemplar"
            ),
        ),
        lifespan=lifespan,
    )
