"""The pin door promotes an artifact to the pinned tier and records a reference
for its caller.

Any caller may pin. A caller holding the `pin` grant (`datasets_api.principals`), the principal
in `DATASETS_PIN_FIRST_PARTY_PRINCIPAL` sent in the `DATASETS_PRINCIPAL_HEADER` header (default
`x-dfo-principal`), spends only the daily budget below. The website sends that value itself on
its server-side pin call, and nothing in front of this service sets or removes the header. Its
reference and the logs name it by `Principal.label`, a fingerprint of the header value, so the
secret is never stored or logged. Pinning is irreversible on this surface.

Every other caller is an open caller, named by `dfo_platform.caller.resolve_caller` before the
pin runs, and its reference names it by its caller key (`ip:<digest>`). A new pinned copy for an
open caller also spends one unit of the open callers' budget (`DATASETS_PIN_OPEN_DAILY_MAX`,
default 200 a UTC day), refusing the daily budget's retryable 503 past it, so open callers
cannot spend the budget the grant's holders draw on, and one unit of the caller's `pin`
allowance (`DATASETS_PIN_DAILY_ALLOWANCE`, default 50 a UTC day), refusing 429
`allowance_exhausted` once it is spent. A pin spends its units of the daily budget, the open
budget and the allowance together (`allowance.SharedAllowanceLedger.spend_together`), so a
refusal by one spends none of the others; a unit stays spent when the copy then fails. A repeat
pin of an artifact that already has a pinned copy spends nothing.

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
named artifact alone. A frame copy spends no budget.

A daily budget bounds the pinned copies this door creates. Before it copies a named artifact,
the door spends one unit of `DATASETS_PIN_DAILY_MAX` (default 500 a UTC day,
`datasets_api.allowance`), and refuses retryable 503 `service_unavailable` with `Retry-After`
counting the seconds to the next UTC midnight once the day's budget is spent. Each spend is one
conditional upsert, so pins admitted together cannot pass the budget. Presentation pins spend
none of it, and neither does a serving-frame copy. A reference to an artifact that already has a
pinned copy skips the budget, so a repeat call that repairs a missing reference, or a second
subject sharing the digest, always succeeds.

Pin calls run on the surface's read executor (`datasets_api.deps`). Full capacity
returns retryable 503 `PinBackpressure`. Shutdown rejection maps to the same
refusal. A broken executor logs and returns a non-retryable 500.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
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
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    SharedAllowanceLedger,
    Spend,
    seconds_to_reset,
)
from datasets_api.backpressure import Backpressure
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.doors.snapshot import read_under_served_id
from datasets_api.pin_models import PinnedCoordinates, PinRequest, PinResponse
from datasets_api.principals import (
    Grant,
    Principal,
    Principals,
    granted_route_class,
)
from datasets_api.recorded_rendering import serving_frames
from datasets_api.reference_store import (
    ARTIFACT_REFERENCE_WRITE_FAILED_MARKER,
    PIN_REFERENCE_KIND,
)
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.retention import retained_frame_location
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import door_call, phase, run_with_timings
from dfo_platform import ProblemSpec
from dfo_platform.caller import (
    Caller,
    CallerHeaderInvalid,
    CallerUnresolved,
    resolve_caller,
)
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from spec_registry import IdentityQuarantined
from spec_registry.reader import SpecReader

logger = logging.getLogger(__name__)


class PinBackpressure(Backpressure):
    """A full pin semaphore maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the pin door is at its concurrent-write capacity right now; retry"


class PinBudgetExhausted(Refusal):
    """The day's `DATASETS_PIN_DAILY_MAX` new pinned copies are made. Maps to retryable 503 with
    `Retry-After` counting the seconds to the next UTC midnight."""

    code = "service_unavailable"
    retryable = True
    detail = "the pin door has reached its daily pin budget; retry after the next UTC midnight"

    def __init__(self, *, limit: int, now: datetime) -> None:
        self.limit = limit
        self.headers = MappingProxyType({"Retry-After": str(seconds_to_reset(now))})
        super().__init__(f"the pin door has made today's budget of {limit} new pinned copies")


class OpenPinBudgetExhausted(Refusal):
    """The day's `DATASETS_PIN_OPEN_DAILY_MAX` new pinned copies for callers without the pin
    grant are made. Maps to retryable 503 with `Retry-After` counting the seconds to the next
    UTC midnight."""

    code = "service_unavailable"
    retryable = True
    detail = (
        "the pin door has reached its daily pin budget for open callers; retry after the next "
        "UTC midnight"
    )

    def __init__(self, *, limit: int, now: datetime) -> None:
        self.limit = limit
        self.headers = MappingProxyType({"Retry-After": str(seconds_to_reset(now))})
        super().__init__(
            f"the pin door has made today's budget of {limit} new pinned copies for callers "
            "without the pin grant"
        )


_BUDGET_REFUSALS = {
    AllowanceKind.pin_budget: PinBudgetExhausted,
    AllowanceKind.open_pin_budget: OpenPinBudgetExhausted,
}


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
    reference_writer: ArtifactReferenceWriter,
    allowances: SharedAllowanceLedger,
    now: Clock,
    caller: Caller | None = None,
) -> datetime:
    """Pin an artifact and the first serving frame of the version that ends up pinned, and return
    the named artifact's reference `created_at` as `pinned_at`.

    A new pinned copy first spends one unit of the day's pin budget in `allowances`, and for an
    open caller (`caller` set) one unit of the open callers' budget and one of `caller`'s `pin`
    allowance, all together before any copy. A spent budget refuses `PinBudgetExhausted` or
    `OpenPinBudgetExhausted`, a spent allowance `AllowanceExhausted`, and a refusal spends none
    of the others. An artifact that already has a pinned copy spends nothing.

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
        _spend_new_copy(allowances, caller=caller, now=read_at)
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


def _spend_new_copy(
    allowances: SharedAllowanceLedger, *, caller: Caller | None, now: datetime
) -> None:
    """Spend the units one new pinned copy takes. See `pin_artifact`."""
    spends = [Spend(BUDGET_KEY, AllowanceKind.pin_budget)]
    if caller is not None:
        spends += [
            Spend(BUDGET_KEY, AllowanceKind.open_pin_budget),
            Spend(caller.key, AllowanceKind.pin),
        ]
    try:
        allowances.spend_together(spends, now)
    except AllowanceExhausted as exhausted:
        budget = _BUDGET_REFUSALS.get(exhausted.kind)
        if budget is None:
            raise
        raise budget(limit=exhausted.limit, now=now) from None


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

    The route admits every caller. The principal header, read before the body, says whether
    the caller holds the `pin` grant (`datasets_api.principals`), held by `allowed_principals`,
    which defaults to `settings.pin_first_party_principal`. The header defaults to
    `settings.principal_header`. Any other caller is resolved (`resolve_caller`) and pins under
    its allowance and the open callers' ceiling.
    """
    principals = Principals.resolve(
        settings=settings, principal_header=principal_header, pin=allowed_principals
    )

    router = APIRouter(route_class=granted_route_class(principals.parse))

    @router.post(
        "/api/datasets/{id:dfo_spec_id}/{content_version}/pin",
        response_model=PinResponse,
        description=(
            "Pin an artifact of a version, and the first frame its files can render from that "
            "survives, servable or aged: the version's own frame, else, for a derived version, "
            "its base version's frame. Every format then stays fetchable and the version "
            "reports `retention.pinned`. When no such frame's bytes survive, the named artifact "
            "still pins, and `retention` keeps the frames' `held_until`. Only the named "
            "artifact's copy spends the daily pin budget; the frame copy spends none. A caller "
            "without the pin grant also spends one unit of its daily pin allowance on a new "
            "copy, within a daily budget shared by every such caller; a repeat pin of an "
            "artifact already pinned spends nothing."
        ),
    )
    async def post_pin(
        id: str, content_version: str, body: PinRequest, request: Request
    ) -> JSONResponse:
        current = deps.get()
        granted: Principal = request.state.principal
        caller: Caller | None = None
        if Grant.pin in granted.grants:
            # A grant is held only by a presented value, so the label is set.
            assert granted.label is not None
            principal = granted.label
        else:
            caller = resolve_caller(request, current.settings.caller)
            principal = caller.key
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
                    allowances=current.allowances,
                    now=current.now,
                    caller=caller,
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
            CallerHeaderInvalid,
            CallerUnresolved,
            PinBackpressure,
            PinBudgetExhausted,
            OpenPinBudgetExhausted,
            AllowanceExhausted,
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
            CallerHeaderInvalid(),
            CallerUnresolved("exemplar"),
            PinBackpressure(),
            PinBudgetExhausted(limit=0, now=datetime(2026, 1, 1, tzinfo=UTC)),
            OpenPinBudgetExhausted(limit=0, now=datetime(2026, 1, 1, tzinfo=UTC)),
            AllowanceExhausted(
                kind=AllowanceKind.pin, limit=0, now=datetime(2026, 1, 1, tzinfo=UTC)
            ),
            DatasetsStoreUnavailable("exemplar"),
        ),
    )
