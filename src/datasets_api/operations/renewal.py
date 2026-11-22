"""Renews a working-tier file readers use before its retention ends.

A working-tier location is held for `WORKING_TIER_RETENTION` (30 days) from its `stored_at`, and
the working bucket's lifecycle rule deletes each object 30 days after it was created. A read
inside retention signs the stored copy and stores nothing, so a file read every day still ages
out 30 days after it was written, and the next read renders or computes it again.

`renew_if_expiring` is called by a read that found a working-tier location whose retention
ends within `DATASETS_RENEW_MARGIN_SECONDS` (seven days). It records one renewal operation for
the location and creates its task, so the read renews nothing itself:

- an analysis root reading a recorded rendering (`analysis_roots.LiveRootSource`), for the
  derived version's stored frame and the file the root reads;
- a stored analysis answer's hit, for the frame file its body names (`analysis.stored_answer`'s
  `on_hit`, `renew_on_hit`), at the analysis door and the presentation's newest-frame check;
- the files door's servable link (`doors.files.resolve_link`).

The operation is an ordinary row, one per location and generation: its `request_identity` is
`{subject_id, content_version, renew_artifact: <digest>, stored_at: <the location's
stored_at>}`, naming a version that links the file, with no members, `HEAD_CHECK_OPTIONS` as its
rendering and a zero estimate, so no request can name it and nobody polls it. A row on record
for that generation queues nothing more, so reads of one location within the margin queue one
renewal, and once it renews, the location's new `stored_at` names the next generation. A
retryable failed row is queued again by a later read, and a pending row whose task create was
lost is retried by a later read after `ENQUEUE_RECOVERY_SECONDS` (`claim_dispatch_retry`). A
new row leaves half of `DATASETS_OPERATION_BACKLOG_MAX` free, and the task create waits at most
`HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS`. A pinned location, one further than the margin from the
end of retention, one already past it (whose read renders or renews it in the request), a
margin of 0, and a store with no operation store queue nothing. A full backlog, a store error
or a failed create is logged and never fails the read.

The worker renews the location (`renew_location`, `operations.worker.run_operation`): under its
one materialise permit it reads the bytes and writes them through a renewing view of the
artifact store (`artifact_store.RenewingArtifactStore.renewing`) shifted just past the end of the
location's retention, so it renews whatever the worker's own margin or clock, which uploads
them again under
the next renewal name (`<digest>#r<n>`), a new object whose lifecycle age starts then, and moves
the location there with a fresh `stored_at`, so its `held_until` moves 30 days on. A location
whose `stored_at` is no longer the one the read saw (renewed since), and bytes already gone, renew nothing, and the row completes.
The inventory row is the digest's, so every version linking the bytes keeps them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from artifact_store import RenewingArtifactStore, Tier
from artifact_store.errors import ArtifactIntegrityError, ArtifactNotFoundError
from dataset_store.clock import Clock
from dataset_store.snapshot import WORKING_TIER_RETENTION, StoredArtifactLocation
from datasets_api.admission import bounded_materialise_slot
from datasets_api.config import DatasetsBuildSettings
from datasets_api.operation_store import (
    NullOperationStore,
    OperationBacklogFull,
    OperationStore,
    compute_operation_id,
)
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS, dispatch_if_due
from datasets_api.operations.head_check import (
    HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS,
    HEAD_CHECK_HEADROOM_SHARE,
    HEAD_CHECK_OPTIONS,
)
from datasets_api.operations.task_client import TaskClient
from datasets_api.origin import catalogue_build_id
from series_coverage.index import Estimate

if TYPE_CHECKING:
    from datasets_api.current import DefinitionDeps

logger = logging.getLogger(__name__)

# The `request_identity` key that marks an operation as a renewal, holding the digest it renews.
RENEW_MARKER = "renew_artifact"

_ZERO_ESTIMATE = Estimate(rows=0, requests=0)
_RECOVERY = timedelta(seconds=ENQUEUE_RECOVERY_SECONDS)
# How far past a location's end of retention the worker's renewing view judges it, so the write
# renews it however the worker's clock stands against the reader's. A renewal name a concurrent
# renewer uploaded within the last retention is still adopted.
_PAST_RETENTION = timedelta(minutes=1)


class RenewalDeps(Protocol):
    """What a read that queues a renewal holds: the surface's `Deps` or a request's
    `DefinitionDeps`."""

    @property
    def settings(self) -> DatasetsBuildSettings: ...

    @property
    def operation_store(self) -> OperationStore: ...

    @property
    def task_client(self) -> TaskClient: ...

    @property
    def now(self) -> Clock: ...


def renewal_identity(
    *, subject_id: str, content_version: str, location: StoredArtifactLocation
) -> dict[str, object]:
    """The `request_identity` of the renewal of `location`'s generation, naming the version
    `{subject_id, content_version}` that links it."""
    return {
        "subject_id": subject_id,
        "content_version": content_version,
        RENEW_MARKER: location.artifact_digest,
        "stored_at": location.stored_at.isoformat(),
    }


def renewal_target(request_identity: object) -> tuple[str, datetime] | None:
    """The digest and queued `stored_at` a renewal operation's `request_identity` names, or
    `None` for any other operation."""
    if not isinstance(request_identity, Mapping):
        return None
    digest = request_identity.get(RENEW_MARKER)
    stored_at = request_identity.get("stored_at")
    if not isinstance(digest, str) or not isinstance(stored_at, str):
        return None
    return digest, datetime.fromisoformat(stored_at)


def expiring(location: StoredArtifactLocation, *, now: datetime, margin: timedelta) -> bool:
    """Whether `location` is a working-tier location still held at `now` whose retention ends
    within `margin` of it."""
    if location.tier != Tier.working.value:
        return False
    held_until = location.stored_at + WORKING_TIER_RETENTION
    return now <= held_until <= now + margin


def renew_if_expiring(
    deps: RenewalDeps,
    location: StoredArtifactLocation,
    *,
    subject_id: str,
    content_version: str,
) -> None:
    """Queue the renewal of `location`, linked under `{subject_id, content_version}`, when its
    retention ends within `DATASETS_RENEW_MARGIN_SECONDS`, unless a row for its generation is on
    record. See the module docstring. Every failure is logged and swallowed."""
    operation_store = deps.operation_store
    margin = timedelta(seconds=deps.settings.renew_margin_seconds)
    if isinstance(operation_store, NullOperationStore):
        return
    if not expiring(location, now=deps.now(), margin=margin):
        return
    identity = renewal_identity(
        subject_id=subject_id, content_version=content_version, location=location
    )
    options = HEAD_CHECK_OPTIONS.model_dump(mode="json")
    try:
        build_id = catalogue_build_id()
        before = operation_store.get(
            compute_operation_id(
                request_identity=identity, origin_catalogue_build_id=build_id, options=options
            )
        )
        if before is None or (before.state == "failed" and before.retryable):
            operation = operation_store.create_or_get(
                request_identity=identity,
                options=options,
                coverage_stamp="",
                origin_catalogue_build_id=build_id,
                estimate=_ZERO_ESTIMATE,
                headroom=deps.settings.operation_backlog_max // HEAD_CHECK_HEADROOM_SHARE,
            )
            # A new row, or a reset under a new submission, is this read's to create a task
            # for; a row a concurrent read wrote is that read's.
            if before is not None and before.submission_count == operation.submission_count:
                return
        elif before.state == "pending" and before.dispatched_at is None:
            claimed = operation_store.claim_dispatch_retry(
                before.id, submission_count=before.submission_count, retry_after=_RECOVERY
            )
            if claimed is None:
                return
            operation = claimed
        else:
            return
        dispatch_if_due(
            operation,
            operation_store=operation_store,
            task_client=deps.task_client,
            timeout=HEAD_CHECK_ENQUEUE_TIMEOUT_SECONDS,
        )
    except OperationBacklogFull as exc:
        logger.warning("Datasets renewal of %r not queued: %s", location.artifact_digest, exc)
    except Exception:
        logger.warning(
            "Datasets renewal of %r failed to queue", location.artifact_digest, exc_info=True
        )


def renew_on_hit(deps: RenewalDeps) -> Callable[[str, str, StoredArtifactLocation], None]:
    """The `on_hit` a stored analysis answer's reader passes `analysis.stored_answer`: it queues
    the renewal of the answer's frame file under `deps` (`renew_if_expiring`)."""

    def on_hit(spec_id: str, content_version: str, location: StoredArtifactLocation) -> None:
        renew_if_expiring(deps, location, subject_id=spec_id, content_version=content_version)

    return on_hit


def renew_location(
    *,
    subject_id: str,
    content_version: str,
    digest: str,
    queued_stored_at: datetime,
    deps: DefinitionDeps,
) -> bool:
    """Renew the working-tier location of `digest` linked under `{subject_id,
    content_version}` while its `stored_at` is still the `queued_stored_at` its read saw, under
    one materialise permit, and return whether the renewal uploaded the bytes.

    The bytes are written through a renewing view of the artifact store shifted just past the
    end of that location's retention, so the write renews it whatever this worker's own margin
    or clock: it uploads under the next renewal name, or adopts one a concurrent renewer just
    uploaded (no upload). A location whose `stored_at` has moved (renewed since), one no
    longer linked, bytes already gone, and a store with no renewing view renew nothing.

    Raises:
        DatasetsStoreUnavailable: The location read failed.
        MintBackpressure: No materialise permit came free in time.
        ArtifactWriteError, ArtifactInventoryError: The write failed.
    """
    store = deps.artifact_store
    if not isinstance(store, RenewingArtifactStore):
        return False
    location = next(
        (
            linked
            for linked in deps.snapshot_reader.read_artifact_locations(subject_id, content_version)
            if linked.artifact_digest == digest and linked.tier == Tier.working.value
        ),
        None,
    )
    if location is None or location.stored_at != queued_stored_at:
        return False
    held_until = location.stored_at + WORKING_TIER_RETENTION
    within = max(held_until - deps.now(), timedelta(0)) + _PAST_RETENTION
    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        try:
            data = deps.artifact_store.get(artifact_digest=digest, tier=Tier.working)
        except (ArtifactNotFoundError, ArtifactIntegrityError):
            return False
        stored = store.renewing(within=within).put(data, tier=Tier.working)
    return not stored.deduplicated


__all__ = [
    "RENEW_MARKER",
    "RenewalDeps",
    "expiring",
    "renew_if_expiring",
    "renew_location",
    "renew_on_hit",
    "renewal_identity",
    "renewal_target",
]
