"""The pin door promotes an artifact to the pinned tier and records a reference
for an authenticated principal.

Only the principal in `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` may pin, using the
`DATASETS_PRINCIPAL_HEADER` header (default `x-dfo-principal`). The website sends
that value itself on its server-side pin call, and nothing in front of this
service sets or removes the header. Pinning is irreversible on this surface.

The door checks the spec, version, and revocation before copying. The reference
write checks revocation again in the same transaction, closing the race during
the copy. Unknown specs and versions, revoked versions, unknown artifacts, and
expired working-tier artifacts retain their distinct refusals.

The reference write also holds the version's link lock (`lock_version_links`)
and refuses `ArtifactNotFound` when the version no longer links the digest. A
re-render may replace a link whose working copy expired while the copy ran,
before the pinned location existed. The pin then refuses, because snapshot
reads cannot list a pin the version does not link. A re-render that runs after a pin completes
sees the pinned location and keeps the link.

Pinning any artifact of a definition version also retains that version's
canonical frame (the frame1 rendering). The door promotes a working-tier frame1
to the pinned tier and records the same principal's reference to it, with the
same revocation recheck, on every pin call, including a repeat pin of an
artifact that is already pinned, so a retry or a re-pin heals a version whose
frame1 is still working-tier. A version with no servable frame1 (a mint whose
frame1 write failed or exceeded `FRAME1_MAX_ROWS`, or a frame1
that aged out unpinned) pins the named artifact alone. The frame1 copy skips the budget check
and counts toward later calls.

A daily budget bounds the pinned copies this door creates. Before a copy, the
door counts pinned-tier locations stored in the trailing 24 hours whose digest
holds a `pin` reference, and refuses retryable 503 `service_unavailable` with
`Retry-After` once the count reaches `DATASETS_PIN_DAILY_MAX` (default 500).
Presentation pins hold another reference kind and spend none of it. A reference
to an artifact that already has a pinned copy skips the budget and adds nothing
to the count, so a repeat call that repairs a missing reference, or a second
subject sharing the digest, always succeeds. The count runs before the copy,
which cannot hold a pooled connection, so pins admitted together can overshoot
the budget by at most the pins in flight, `DATASETS_PIN_MAX_CONCURRENCY` per
instance.

Pin calls use a dedicated executor with `DATASETS_PIN_MAX_CONCURRENCY` workers
(default 4) and a same-sized, non-blocking semaphore. Full capacity returns
retryable 503 `PinBackpressure`. Shutdown rejection maps to the same refusal.
Completion callbacks release permits when executor work ends. Shutdown cancels
queued futures and drains running calls within the shared budget before closing
the store and pool. A broken executor logs and returns a non-retryable 500.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Protocol

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from artifact_store import ArtifactStore, LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import (
    ArtifactReferenceWriter,
    LiveSnapshotReader,
    SnapshotReader,
    SnapshotRevoked,
    servable_location_of_media_type,
    working_tier_expired,
)
from datasets_api import routing  # noqa: F401 -- registers dfo_spec_id before route construction
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    ArtifactNotFound,
    PinNotPermitted,
    SnapshotNotFound,
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings, DatasetsConfigError, DatasetsSettings
from datasets_api.door_executor import build_door_executor
from datasets_api.pin_models import PinnedCoordinates, PinRequest, PinResponse
from datasets_api.reference_store import (
    ARTIFACT_REFERENCE_WRITE_FAILED_MARKER,
    PIN_REFERENCE_KIND,
    LiveArtifactReferenceWriter,
)
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import door_call, phase, run_with_timings
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from materialise.formats import FRAME1_MEDIA_TYPE
from spec_registry.reader import LiveSpecReader, SpecReader

logger = logging.getLogger(__name__)

PIN_BUDGET_WINDOW = timedelta(hours=24)

# The oldest counted pin leaves the window within a day. An hour lets a client
# retry while the budget frees up without polling.
_PIN_BUDGET_RETRY_AFTER_SECONDS = 3_600


class PinReferenceWriter(ArtifactReferenceWriter, Protocol):
    """A reference writer that can also count the pinned copies the budget limits."""

    def count_pinned_copies_since(self, *, reference_kind: str, since: datetime) -> int:
        """Count pinned-tier locations stored at or after `since` whose digest holds a
        `reference_kind` reference."""
        ...


class PinBackpressure(Exception):
    """A full pin semaphore maps to retryable `service_unavailable` (503)."""


class PinBudgetExhausted(Exception):
    """The trailing-window pin count is at `DATASETS_PIN_DAILY_MAX`. Maps to retryable 503."""

    def __init__(self, *, pinned: int, limit: int) -> None:
        self.pinned = pinned
        self.limit = limit
        super().__init__(
            f"{pinned} pinned copies made in the last 24 hours, at or above the daily pin "
            f"budget of {limit}"
        )


def classify_pin_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, SpecNotFound):
        return ProblemSpec(status=404, code="spec_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, SnapshotNotFound):
        return ProblemSpec(status=404, code="snapshot_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, SnapshotRevoked):
        return ProblemSpec(status=410, code="snapshot_revoked", detail=str(exc), retryable=False)
    if isinstance(exc, ArtifactExpired):
        return ProblemSpec(status=410, code="artifact_expired", detail=str(exc), retryable=False)
    if isinstance(exc, ArtifactNotFound):
        return ProblemSpec(status=404, code="artifact_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, PinNotPermitted):
        return ProblemSpec(status=403, code="pin_not_permitted", detail=str(exc), retryable=False)
    if isinstance(exc, PinBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the pin door is at its concurrent-write capacity right now; retry",
            retryable=True,
        )
    if isinstance(exc, PinBudgetExhausted):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the pin door has reached its daily pin budget; retry later",
            retryable=True,
            headers={"Retry-After": str(_PIN_BUDGET_RETRY_AFTER_SECONDS)},
        )
    return None


def pin_artifact(
    *,
    id: str,
    content_version: str,
    artifact_digest: str,
    principal: str,
    spec_reader: SpecReader,
    snapshot_reader: SnapshotReader,
    artifact_store: ArtifactStore,
    reference_writer: PinReferenceWriter,
    daily_max: int,
    now: Clock,
) -> datetime:
    """Pin an artifact and the version's canonical frame, and return the named artifact's
    reference `created_at` as `pinned_at`.

    A new pinned copy first needs room in the budget: fewer than `daily_max`
    pinned copies of `pin`-referenced digests stored within `PIN_BUDGET_WINDOW`,
    else `PinBudgetExhausted` before any copy. An artifact that already has a
    pinned copy skips the budget.

    The refusals concern the named artifact alone. The frame1 is retained after the named
    artifact's reference is recorded, and only when it is servable at the call's read time.
    A missing or aged-out frame1 leaves the response unchanged. The transactional revocation
    and link rechecks propagate unlogged. Other reference-write failures are logged and
    re-raised. Pinned copies persist, so a retry attempts only the reference writes.
    """
    if spec_reader.read_definition(id) is None:
        raise SpecNotFound(id=id)

    if snapshot_reader.read_version(id, content_version) is None:
        raise SnapshotNotFound(id=id, content_version=content_version)

    revocation = snapshot_reader.read_revocation(id, content_version)
    if revocation is not None:
        raise SnapshotRevoked(id=id, content_version=content_version, reason=revocation.reason)

    read_at = now()
    version_locations = snapshot_reader.read_artifact_locations(id, content_version)
    locations = [
        location for location in version_locations if location.artifact_digest == artifact_digest
    ]
    if not locations:
        raise ArtifactNotFound(
            id=id, content_version=content_version, artifact_digest=artifact_digest
        )

    already_pinned = any(location.tier == "pinned" for location in locations)
    if not already_pinned:
        working = next((location for location in locations if location.tier == "working"), None)
        if working is None or working_tier_expired(working.stored_at, now=read_at):
            raise ArtifactExpired(id=id, content_version=content_version)
        pinned = reference_writer.count_pinned_copies_since(
            reference_kind=PIN_REFERENCE_KIND, since=read_at - PIN_BUDGET_WINDOW
        )
        if pinned >= daily_max:
            raise PinBudgetExhausted(pinned=pinned, limit=daily_max)
        with phase("copy"):
            artifact_store.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)

    pinned_at = _record_pin_reference(
        reference_writer,
        principal=principal,
        artifact_digest=artifact_digest,
        id=id,
        content_version=content_version,
        already_pinned=already_pinned,
        now=read_at,
    )

    frame = servable_location_of_media_type(
        version_locations, media_type=FRAME1_MEDIA_TYPE, now=read_at
    )
    if frame is not None and frame.artifact_digest != artifact_digest:
        frame_already_pinned = frame.tier == "pinned"
        if not frame_already_pinned:
            with phase("copy"):
                artifact_store.copy(
                    frame.artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned
                )
        _record_pin_reference(
            reference_writer,
            principal=principal,
            artifact_digest=frame.artifact_digest,
            id=id,
            content_version=content_version,
            already_pinned=frame_already_pinned,
            now=read_at,
        )
    return pinned_at


def _record_pin_reference(
    reference_writer: ArtifactReferenceWriter,
    *,
    principal: str,
    artifact_digest: str,
    id: str,
    content_version: str,
    already_pinned: bool,
    now: datetime,
) -> datetime:
    """Record the principal's pin reference under the transactional revocation recheck."""
    try:
        with phase("record"):
            return reference_writer.record_reference_unless_revoked(
                principal=principal,
                artifact_digest=artifact_digest,
                subject_id=id,
                content_version=content_version,
                reference_kind=PIN_REFERENCE_KIND,
                now=now,
            )
    except (SnapshotRevoked, ArtifactNotFound):
        # The transactional rechecks catch a revocation or a replaced link during the copy.
        # The copy stays unreferenced, and reconciliation can reclaim it.
        raise
    except Exception:
        # Keep the pinned copy for retry. Log the missing reference write
        # for reconciliation.
        logger.error(
            "%s artifact_digest=%s subject_id=%s content_version=%s principal=%s already_pinned=%s",
            ARTIFACT_REFERENCE_WRITE_FAILED_MARKER,
            artifact_digest,
            id,
            content_version,
            principal,
            already_pinned,
        )
        raise


def build_pin_door(
    *,
    settings: DatasetsBuildSettings | None = None,
    spec_reader: SpecReader | None = None,
    snapshot_reader: SnapshotReader | None = None,
    artifact_store: ArtifactStore | None = None,
    reference_writer: PinReferenceWriter | None = None,
    now: Clock = utc_now,
    principal_header: str | None = None,
    allowed_principals: frozenset[str] | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the pin door and its executor.

    `settings` defaults from the environment. An omitted `allowed_principals`
    uses `settings.pin_first_party_principal`. An empty value yields an empty
    allowlist. Pin and publish permissions use separate principals. The executor size
    defaults to `settings.pin_max_concurrency`. `pool_size` overrides it.

    Inject `spec_reader`, `snapshot_reader`, `artifact_store`, and
    `reference_writer` together, or leave all four unset for lifespan to build
    them with one pool borrow. A partial set raises `DatasetsConfigError` at
    construction. `budget` is shared across mounted doors. An omitted budget
    creates a private shutdown budget.
    """
    missing = [
        name
        for name, value in (
            ("spec_reader", spec_reader),
            ("snapshot_reader", snapshot_reader),
            ("artifact_store", artifact_store),
            ("reference_writer", reference_writer),
        )
        if value is None
    ]
    if missing and len(missing) != 4:
        raise DatasetsConfigError(
            "datasets pin door built with "
            f"{', '.join(missing)} left uninjected while another dependency is injected. A "
            "caller injecting any one of spec_reader/snapshot_reader/artifact_store/"
            "reference_writer must inject every one of them. This door builds a live "
            "dependency for all four only when every one of them is left None."
        )

    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    _spec_reader: SpecReader | None = spec_reader
    _snapshot_reader: SnapshotReader | None = snapshot_reader
    _artifact_store: ArtifactStore | None = artifact_store
    _reference_writer: PinReferenceWriter | None = reference_writer
    _principal_header = (
        principal_header if principal_header is not None else _settings.principal_header
    )
    _allowed_principals = (
        allowed_principals
        if allowed_principals is not None
        else frozenset({_settings.pin_first_party_principal} - {""})
    )

    effective_pool_size = pool_size if pool_size is not None else _settings.pin_max_concurrency
    door_executor = build_door_executor(effective_pool_size, "pin")
    executor = door_executor.executor

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _spec_reader, _snapshot_reader, _artifact_store, _reference_writer
        # The all-or-nothing check makes `spec_reader is None` the store trigger.
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
            if reference_writer is None:
                _reference_writer = LiveArtifactReferenceWriter(store=store)
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
                # Release the reader's pool borrow if artifact-store setup fails.
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

    @router.post("/api/datasets/{id:dfo_spec_id}/{content_version}/pin", response_model=PinResponse)
    async def post_pin(
        id: str, content_version: str, body: PinRequest, request: Request
    ) -> JSONResponse:
        # Read all principal values. Duplicates expose headers added after
        # the proxy's value and trigger refusal.
        principal_values = request.headers.getlist(_principal_header)
        if not principal_values:
            raise PinNotPermitted(principal=None)
        if len(principal_values) > 1:
            logger.warning(
                "Datasets pin door saw %d values for the %r principal header on one "
                "request; refusing rather than trusting either one",
                len(principal_values),
                _principal_header,
            )
            raise PinNotPermitted(principal=principal_values[0])
        principal = principal_values[0]
        if principal not in _allowed_principals:
            raise PinNotPermitted(principal=principal)

        with door_call("pin", logger=logger) as call:
            pinned_at = await door_executor.run(
                functools.partial(
                    run_with_timings,
                    pin_artifact,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    artifact_digest=body.artifact_digest,
                    principal=principal,
                    spec_reader=_spec_reader,
                    snapshot_reader=_snapshot_reader,
                    artifact_store=_artifact_store,
                    reference_writer=_reference_writer,
                    daily_max=_settings.pin_daily_max,
                    now=now,
                ),
                refuse=PinBackpressure,
            )
        response = PinResponse(
            pinned=PinnedCoordinates(
                id=id,
                content_version=content_version,
                artifact_digest=body.artifact_digest,
                pinned_at=pinned_at.isoformat(),
            )
        )
        return JSONResponse(
            response.model_dump(mode="json"),
            headers={"Server-Timing": call.server_timing},
        )

    return DoorContribution(
        router=router,
        classify=classify_pin_exception,
        exception_types=(
            SpecNotFound,
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            ArtifactNotFound,
            PinNotPermitted,
            PinBackpressure,
            PinBudgetExhausted,
            # Surface-level classification handles errors reference writes can raise.
            DatasetsStoreUnavailable,
        ),
        exception_exemplars=(
            SpecNotFound(id="spec_exemplar"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
            ArtifactNotFound(
                id="spec_exemplar", content_version="cv_exemplar", artifact_digest="sha256:exemplar"
            ),
            PinNotPermitted(principal=None),
            PinBackpressure(),
            PinBudgetExhausted(pinned=0, limit=0),
            DatasetsStoreUnavailable("exemplar"),
        ),
        lifespan=lifespan,
    )
