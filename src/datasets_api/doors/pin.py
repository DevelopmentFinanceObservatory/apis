"""The pin door promotes an artifact to the pinned tier and records a reference
for an authenticated principal.

Only the `pin` grant (`datasets_api.principals`) may pin, which the principal in
`DATASETS_PIN_FIRST_PARTY_PRINCIPAL` holds, sent in the `DATASETS_PRINCIPAL_HEADER` header
(default `x-dfo-principal`) and checked before the body is read. The website sends
that value itself on its server-side pin call, and nothing in front of this
service sets or removes the header. The reference and the logs name the caller by
`Principal.label`, a fingerprint of the header value, so the secret is never stored or
logged. Pinning is irreversible on this surface.

The door checks the spec, then the version, its revocation and its base version's, and the
digest's link in one `read_servable_version` transaction, before copying. The path id resolves
to the served ids the registry binds to it, and a version recorded under the posted id itself
pins too (`doors.snapshot.read_under_served_id`); the references name the id the version is
recorded under. A quarantined path id, or a quarantined served id bound to it, refuses
`identity_quarantined`. The reference
write checks revocation again in the same transaction, closing the race during
the copy. Unknown specs and versions, revoked versions, unknown artifacts, and
expired working-tier artifacts retain their distinct refusals. A named artifact whose working
bytes are gone at the copy refuses `ArtifactExpired` as well.

The reference write also holds the version's link lock (`lock_version_links`)
and refuses `ArtifactNotFound` when the version no longer links the digest. A
re-render may replace a link whose working copy expired while the copy ran,
before the pinned location existed. The pin then refuses, because snapshot
reads cannot list a pin the version does not link. A re-render that runs after a pin completes
sees the pinned location and keeps the link.

Pinning any artifact of a definition version also retains a frame that serves that version.
The door walks the version's serving frames (`recorded_rendering.serving_frames`) in the order a
render tries them: a raw version's own frame; a derived version's (`base_content_version` set: a
groupby collapse or a party-totals cut) own frame, then its base version's, or its base version's
alone when the record cannot say what its own frame holds. For each it takes the frame retention
reads (`retention.retained_frame_location`: the servable frame, else the aged one, each its
canonical frame or else its frame1 document), and stops at the first that ends up pinned. It
promotes a working-tier frame to the pinned tier and records the same principal's reference to
it, naming the version the frame is stored under, with the same revocation recheck, on every pin
call, including a repeat pin of an artifact that is already pinned, so a retry or a re-pin heals
a version whose frame is still working-tier. A frame whose bytes are gone is skipped and the walk
moves to the next serving frame. One pinned serving frame keeps the version fetchable in every
format and reporting `retention.pinned`. A version none of whose serving frames survives pins the
named artifact alone. A frame copy skips the budget check. It holds a `pin` reference, so it
counts toward later calls. A pin can add two pinned copies to the count (the named artifact and
one serving frame), and only the first waits on the budget.

A daily budget bounds the pinned copies this door creates. Before a copy, the
door counts pinned-tier locations stored in the trailing 24 hours whose digest
holds a `pin` reference, and refuses retryable 503 `service_unavailable` with
`Retry-After` once the count reaches `DATASETS_PIN_DAILY_MAX` (default 500).
Presentation pins hold another reference kind and spend none of it. A reference
to an artifact that already has a pinned copy skips the budget and adds nothing
to the count, so a repeat call that repairs a missing reference, or a second
subject sharing the digest, always succeeds. The count runs before the copy,
which cannot hold a pooled connection, so pins admitted together can overshoot
the budget by at most the pins in flight, at most `DATASETS_READ_MAX_CONCURRENCY`
per instance.

Pin calls run on the surface's read executor (`datasets_api.deps`). Full capacity
returns retryable 503 `PinBackpressure`. Shutdown rejection maps to the same
refusal. A broken executor logs and returns a non-retryable 500.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from types import MappingProxyType

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from artifact_store import ArtifactNotFoundError, ArtifactStore, Tier
from dataset_store.clock import Clock
from dataset_store.snapshot import (
    ArtifactNotFound,
    ArtifactReferenceWriter,
    SnapshotNotFound,
    SnapshotReader,
    SnapshotRevoked,
    read_servable_version,
)
from datasets_api import routing  # noqa: F401 -- registers dfo_spec_id before route construction
from datasets_api.backpressure import Backpressure
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.doors.snapshot import read_under_served_id
from datasets_api.pin_models import PinnedCoordinates, PinRequest, PinResponse
from datasets_api.principals import Grant, PinNotPermitted, Principals, granted_route_class
from datasets_api.recorded_rendering import serving_frames
from datasets_api.reference_store import (
    ARTIFACT_REFERENCE_WRITE_FAILED_MARKER,
    PIN_REFERENCE_KIND,
    PinReferenceWriter,
)
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.retention import retained_frame_location
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import door_call, phase, run_with_timings
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from spec_registry import IdentityQuarantined
from spec_registry.reader import SpecReader

logger = logging.getLogger(__name__)

PIN_BUDGET_WINDOW = timedelta(hours=24)

# The oldest counted pin leaves the window within a day. An hour lets a client
# retry while the budget frees up without polling.
_PIN_BUDGET_RETRY_AFTER_SECONDS = 3_600


class PinBackpressure(Backpressure):
    """A full pin semaphore maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the pin door is at its concurrent-write capacity right now; retry"


class PinBudgetExhausted(Refusal):
    """The trailing-window pin count is at `DATASETS_PIN_DAILY_MAX`. Maps to retryable 503."""

    code = "service_unavailable"
    retryable = True
    detail = "the pin door has reached its daily pin budget; retry later"
    headers = MappingProxyType({"Retry-After": str(_PIN_BUDGET_RETRY_AFTER_SECONDS)})

    def __init__(self, *, pinned: int, limit: int) -> None:
        self.pinned = pinned
        self.limit = limit
        super().__init__(
            f"{pinned} pinned copies made in the last 24 hours, at or above the daily pin "
            f"budget of {limit}"
        )


def classify_pin_exception(exc: BaseException) -> ProblemSpec | None:
    return classify_datasets_refusal(exc)


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
    """Pin an artifact and the first serving frame of the version that ends up pinned, and return
    the named artifact's reference `created_at` as `pinned_at`.

    A new pinned copy first needs room in the budget: fewer than `daily_max`
    pinned copies of `pin`-referenced digests stored within `PIN_BUDGET_WINDOW`,
    else `PinBudgetExhausted` before any copy. An artifact that already has a
    pinned copy skips the budget.

    The refusals concern the named artifact alone. A serving frame is retained after the named
    artifact's reference is recorded. The serving frames are walked in order, each the frame the
    version's retention reads at the call's read time (`retention.retained_frame_location`),
    servable or aged, and the walk stops at the first that is pinned already, is the named
    artifact, or is copied now. A missing frame, or one whose bytes are gone, moves the walk on
    and leaves the response unchanged, and a frame whose bytes are gone logs a warning. The
    transactional revocation and link rechecks propagate unlogged. Other reference-write failures are logged and
    re-raised. Pinned copies persist, so a retry attempts only the reference writes.
    """
    read_at = now()
    subject_ids, stored = read_under_served_id(
        id=id,
        spec_reader=spec_reader,
        read=lambda subject_id, subject_ids: (
            subject_ids,
            read_servable_version(
                snapshot_reader,
                subject_id=subject_id,
                content_version=content_version,
                artifact_digest=artifact_digest,
                now=read_at,
                revoked_under=subject_ids,
            ),
        ),
    )
    subject_id = stored.version.subject_id
    named = stored.servable_digest(artifact_digest)
    if named is None:
        raise ArtifactExpired(id=id, content_version=content_version)

    already_pinned = named.tier == "pinned"
    if not already_pinned:
        pinned = reference_writer.count_pinned_copies_since(
            reference_kind=PIN_REFERENCE_KIND, since=read_at - PIN_BUDGET_WINDOW
        )
        if pinned >= daily_max:
            raise PinBudgetExhausted(pinned=pinned, limit=daily_max)
        try:
            with phase("copy"):
                artifact_store.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
        except ArtifactNotFoundError as exc:
            # The working copy aged out between the read and the copy.
            raise ArtifactExpired(id=id, content_version=content_version) from exc

    pinned_at = _record_pin_reference(
        reference_writer,
        principal=principal,
        artifact_digest=artifact_digest,
        id=subject_id,
        content_version=content_version,
        revoked_under=subject_ids,
        already_pinned=already_pinned,
        now=read_at,
    )

    own_locations = [*stored.servable, *stored.aged_out]
    renderings = [location.rendering for location in own_locations]
    for serving in serving_frames(stored.version, renderings=renderings):
        locations = (
            own_locations
            if serving.content_version == content_version
            else snapshot_reader.read_artifact_locations(subject_id, serving.content_version)
        )
        frame = retained_frame_location(locations, now=read_at)
        if frame is None:
            continue
        if frame.artifact_digest == artifact_digest:
            break
        frame_already_pinned = frame.tier == "pinned"
        if not frame_already_pinned:
            try:
                with phase("copy"):
                    artifact_store.copy(
                        frame.artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned
                    )
            except ArtifactNotFoundError:
                # The aged frame's bytes are gone, so the next serving frame is tried.
                logger.warning(
                    "pin skipped a frame whose bytes are gone artifact_digest=%s subject_id=%s "
                    "content_version=%s",
                    frame.artifact_digest,
                    subject_id,
                    serving.content_version,
                )
                continue
        _record_pin_reference(
            reference_writer,
            principal=principal,
            artifact_digest=frame.artifact_digest,
            id=subject_id,
            content_version=serving.content_version,
            revoked_under=subject_ids,
            already_pinned=frame_already_pinned,
            now=read_at,
        )
        break
    return pinned_at


def _record_pin_reference(
    reference_writer: ArtifactReferenceWriter,
    *,
    principal: str,
    artifact_digest: str,
    id: str,
    content_version: str,
    revoked_under: Sequence[str],
    already_pinned: bool,
    now: datetime,
) -> datetime:
    """Record the principal's pin reference under the transactional revocation recheck, which
    covers `id` and every id in `revoked_under`."""
    try:
        with phase("record"):
            return reference_writer.record_reference_unless_revoked(
                principal=principal,
                artifact_digest=artifact_digest,
                subject_id=id,
                content_version=content_version,
                reference_kind=PIN_REFERENCE_KIND,
                now=now,
                revoked_under=revoked_under,
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
    deps: DepsCell,
    settings: DatasetsBuildSettings,
    principal_header: str | None = None,
    allowed_principals: frozenset[str] | None = None,
) -> DoorContribution:
    """Build the pin door over the surface's dependency cell.

    The route admits the `pin` grant alone (`datasets_api.principals`), held by
    `allowed_principals`, which defaults to `settings.pin_first_party_principal`, and checked
    before the body is read. The header defaults to `settings.principal_header`.
    """
    require_pin = Principals.resolve(
        settings=settings, principal_header=principal_header, pin=allowed_principals
    ).require(Grant.pin)

    router = APIRouter(route_class=granted_route_class(require_pin))

    @router.post(
        "/api/datasets/{id:dfo_spec_id}/{content_version}/pin",
        response_model=PinResponse,
        description=(
            "Pin an artifact of a version, and the first frame its files can render from that "
            "survives, servable or aged: the version's own frame, else, for a derived version, "
            "its base version's frame. Every format then stays fetchable and the version "
            "reports `retention.pinned`. When no such frame's bytes survive, the named artifact "
            "still pins, and `retention` keeps the frames' `held_until`. Only the named "
            "artifact's copy waits on the daily pin budget; the frame copy skips it and counts "
            "toward later calls, so one pin can add up to two copies to the count."
        ),
    )
    async def post_pin(
        id: str, content_version: str, body: PinRequest, request: Request
    ) -> JSONResponse:
        principal = request.state.principal.label

        current = deps.get()
        with door_call("pin", logger=logger) as call:
            pinned_at = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    pin_artifact,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    artifact_digest=body.artifact_digest,
                    principal=principal,
                    spec_reader=current.spec_reader,
                    snapshot_reader=current.snapshot_reader,
                    artifact_store=current.artifact_store,
                    reference_writer=current.reference_writer,
                    daily_max=current.settings.pin_daily_max,
                    now=current.now,
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
        registry_endpoints=route_registry_endpoints(router, "datasets.pin"),
        classify=classify_pin_exception,
        exception_types=(
            SpecNotFound,
            # The path id resolves through the spec registry, which raises it for a quarantined
            # path id or served id bound to it.
            IdentityQuarantined,
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
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
            ArtifactNotFound(
                id="spec_exemplar", content_version="cv_exemplar", artifact_digest="sha256:exemplar"
            ),
            PinNotPermitted(),
            PinBackpressure(),
            PinBudgetExhausted(pinned=0, limit=0),
            DatasetsStoreUnavailable("exemplar"),
        ),
    )
