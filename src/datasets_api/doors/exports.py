"""The exports door prepares a frame as a file and serves the link to it (`datasets_api.exports`).

- `POST /api/datasets/exports`, body `ExportRequestBody`, answers 200 `ExportResponse`. A request
  whose export is recorded answers the record on the read executor, rendering and spending
  nothing. Any other prepares on the work executor under the definition deadline, as the file
  selector renders. While a prepare of the request runs on any instance, the door waits for it
  on the event loop, holding no work permit (`_prepared`), for at most
  `materialise_wait_seconds`.
  It admits every caller, and the site's proxy forwards it. The caller is
  named as the pin door names it: a holder of the `pin` grant spends the day's pin budget
  alone, and any other caller is resolved (`resolve_caller`) and spends its `pin` allowance, so
  data-mcp forwards `x-dfo-caller` and `x-dfo-caller-principal`. A frame
  of a dataset id alone (`{spec_id}`) is first resolved on the work executor to the version the
  analysis runner reads for a Spec root of it (`exports.resolve_export_frame`), and the request
  then runs as one naming that pair.
- `GET /api/datasets/exports/{export_id}` answers 302 to a URL signed for five minutes to the
  export's pinned file, with `Cache-Control: no-store`. It admits every caller and never
  renders. A version revoked since the export was prepared refuses `snapshot_revoked`, and a
  quarantined identity `identity_quarantined`.

The prepare (`build_exports_door`) binds the `datasets.exports` registry endpoint and the link
(`build_export_link_door`) `datasets.export_link`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from artifact_store import ArtifactSigningError
from dataset_store.snapshot import SnapshotNotFound, SnapshotRevoked
from datasets_api.allowance import AllowanceExhausted, AllowanceKind
from datasets_api.backpressure import Backpressure
from datasets_api.codes_exports import (
    ExportFileIncomplete,
    ExportNotFound,
    ExportOverCeiling,
    ExportPrepareRunning,
    LicenceNotRedistributable,
    NewestVersionUnservable,
    ProvenanceUnavailable,
)
from datasets_api.codes_snapshot import ArtifactExpired, SnapshotSigningFailed, SpecNotFound
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import DefinitionDeps, StoredTraceMissingResolvedStatVars
from datasets_api.deps import Deps, DepsCell
from datasets_api.doors.definition import _run_retrieval, request_definition_deps
from datasets_api.doors.files import (
    REDIRECT_RESPONSE,
    SELECTOR_RENDERING_EXCEPTION_TYPES,
    classify_files_exception,
)
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.mint_inventory import MINT_EXCEPTION_EXEMPLARS
from datasets_api.exports import (
    BareExportFrame,
    ExportRequest,
    ExportRequestBody,
    ExportResponse,
    export_id,
    passes_through,
    prepare_export,
    recorded_export,
    resolve_export_frame,
    sign_export,
)
from datasets_api.headers import no_store_headers
from datasets_api.principals import Grant, Principal, Principals, granted_route_class
from datasets_api.retention_admission import (
    ExportClaimUnsettled,
    OpenPinBudgetExhausted,
    PinBudgetExhausted,
    PinUnitHeld,
)
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import door_call, run_with_timings
from dfo_platform.caller import Caller, CallerHeaderInvalid, CallerUnresolved, resolve_caller
from dfo_platform.doors import DoorContribution
from dfo_platform.timings import PhaseTimings
from spec_registry import IdentityQuarantined
from spec_registry.errors import IdentityCollision

logger = logging.getLogger(__name__)

# How often a request waiting for another prepare of its export reads the claim again: first
# this often, doubling up to the most it sleeps.
_CLAIM_POLL_SECONDS = 0.05
_CLAIM_POLL_MAX_SECONDS = 1.0

PREPARE_PATH = "/api/datasets/exports"
LINK_PATH = "/api/datasets/exports/{export_id}"


class ExportBackpressure(Backpressure):
    """A full read or work executor maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the exports door is at its concurrent-call capacity right now; retry"


async def _prepared(
    export_request: ExportRequest,
    *,
    request: Request,
    granted: Principal,
    current: Deps,
    retrieval: DefinitionDeps,
    timings: PhaseTimings,
) -> Mapping[str, Any]:
    """The facts of the export `export_request` names: its record, else a new prepare's. While
    any instance's prepare holds its claim, the call sleeps on the event loop, holding no work
    permit, and reads only the claim, backing off from `_CLAIM_POLL_SECONDS` to
    `_CLAIM_POLL_MAX_SECONDS`. Once a call has waited, each hop waits for its permit until the wait
    ends, so waiters that wake together all get one. Any other call sheds at a full pool.

    Raises:
        ExportBackpressure: A pool is full, and this call has not waited.
        ExportPrepareRunning: The other prepare still runs at `materialise_wait_seconds`, or at
            the deadline if that comes first.
    """
    wait_seconds = current.settings.materialise_wait_seconds
    ends = min(time.monotonic() + wait_seconds, cast(float, retrieval.deadline))
    has_waited = False

    def permit_wait() -> float:
        return max(0.0, ends - time.monotonic()) if has_waited else 0.0

    async def read[T](fn: Callable[..., T], /, **kwargs: object) -> T:
        return await current.read_executor.run(
            functools.partial(run_with_timings, fn, timings, **kwargs),
            refuse=ExportBackpressure,
            wait_seconds=permit_wait(),
        )

    async def waited_for_claim() -> bool:
        """Sleep while a live claim holds the export, and say whether any did."""
        nonlocal has_waited
        delay = _CLAIM_POLL_SECONDS
        started = time.perf_counter()
        waited = False
        while True:
            claim = await read(current.allowances.read_claim, export_id=export)
            if claim is None or claim.expires_at <= current.now():
                if waited:
                    timings.record("rebuild_wait", (time.perf_counter() - started) * 1000)
                return waited
            if time.monotonic() >= ends:
                raise ExportPrepareRunning(wait_seconds=wait_seconds)
            await asyncio.sleep(delay)
            waited = has_waited = True
            delay = min(delay * 2, _CLAIM_POLL_MAX_SECONDS)

    export = export_id(export_request)
    while True:
        facts = await read(recorded_export, request=export_request, current=current)
        if facts is not None:
            return facts
        if await waited_for_claim():
            continue
        # Only a new export can spend, so a repeat needs no caller.
        caller: Caller | None = (
            None
            if Grant.pin in granted.grants
            else resolve_caller(request, current.settings.caller)
        )
        facts = await _run_retrieval(
            current.work_executor,
            prepare_export,
            deps=retrieval,
            timings=timings,
            refuse=ExportBackpressure,
            wait_seconds=permit_wait(),
            request=export_request,
            caller=caller,
            current=current,
        )
        if facts is not None:
            return facts


def build_exports_door(
    *,
    deps: DepsCell,
    settings: DatasetsBuildSettings,
    allowed_principals: frozenset[str] | None = None,
) -> DoorContribution:
    """The prepare route, on the `datasets.exports` endpoint. `allowed_principals` holds the
    `pin` grant, as on the pin door, defaulting to `settings.pin_first_party_principal`."""
    principals = Principals.resolve(settings=settings, pin=allowed_principals)
    router = APIRouter(route_class=granted_route_class(principals.parse))

    @router.post(PREPARE_PATH, response_model=ExportResponse)
    async def post_export(body: ExportRequestBody, request: Request) -> JSONResponse:
        """Prepare one exact frame version as a file in the pinned tier and answer its facts
        and its link's path. A frame of a dataset id alone exports that dataset's newest
        version, which the answer's `frame` names. A repeat request answers the same facts and
        spends nothing. A new pinned copy spends one unit of the caller's daily pin
        allowance."""
        current = deps.get()
        granted: Principal = request.state.principal
        retrieval = request_definition_deps(current)
        with door_call("export", logger=logger) as call:
            frame = body.frame
            if isinstance(frame, BareExportFrame):
                frame = await _run_retrieval(
                    current.work_executor,
                    resolve_export_frame,
                    deps=retrieval,
                    timings=call.timings,
                    refuse=ExportBackpressure,
                    frame=frame,
                    current=current,
                )
            facts = await _prepared(
                body.resolved(frame),
                request=request,
                granted=granted,
                current=current,
                retrieval=retrieval,
                timings=call.timings,
            )
        return JSONResponse(facts, headers=no_store_headers(server_timing=call.server_timing))

    exemplar_day = datetime(2026, 1, 1, tzinfo=UTC)
    return DoorContribution(
        router=router,
        registry_endpoints={PREPARE_PATH: "datasets.exports"},
        classify=classify_files_exception,
        exception_types=(
            SpecNotFound,
            IdentityQuarantined,
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            ProvenanceUnavailable,
            LicenceNotRedistributable,
            ExportOverCeiling,
            ExportFileIncomplete,
            NewestVersionUnservable,
            CallerHeaderInvalid,
            CallerUnresolved,
            PinBudgetExhausted,
            OpenPinBudgetExhausted,
            AllowanceExhausted,
            ExportPrepareRunning,
            ExportClaimUnsettled,
            PinUnitHeld,
            ExportBackpressure,
            # A derived file's slot and a render from the stored frame each raise this at the
            # shared materialise permit.
            MintBackpressure,
            DatasetsStoreUnavailable,
            *SELECTOR_RENDERING_EXCEPTION_TYPES,
        ),
        exception_exemplars=(
            SpecNotFound(id="spec_exemplar"),
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
            ProvenanceUnavailable(spec_id="ops_exemplar", content_version="ocv_exemplar"),
            LicenceNotRedistributable(sources=[("dcid:exemplar", "exemplar")]),
            ExportOverCeiling(format="csv", count=1, cap=0),
            ExportFileIncomplete(id="spec_exemplar", content_version="cv_exemplar"),
            NewestVersionUnservable(code="service_unavailable", detail="exemplar"),
            NewestVersionUnservable(code="internal_error", detail="exemplar"),
            NewestVersionUnservable(code="snapshot_not_found", detail="exemplar"),
            CallerHeaderInvalid(),
            CallerUnresolved("exemplar"),
            PinBudgetExhausted(limit=0, now=exemplar_day),
            OpenPinBudgetExhausted(limit=0, now=exemplar_day),
            AllowanceExhausted(kind=AllowanceKind.pin, limit=0, now=exemplar_day),
            ExportPrepareRunning(wait_seconds=30),
            ExportClaimUnsettled(retry_after=1),
            PinUnitHeld(retry_after=1),
            ExportBackpressure(),
            DatasetsStoreUnavailable("exemplar"),
            StoredTraceMissingResolvedStatVars(
                subject_id="spec_exemplar", content_version="cv_exemplar"
            ),
            *(
                exemplar
                for exemplar in MINT_EXCEPTION_EXEMPLARS
                if isinstance(exemplar, SELECTOR_RENDERING_EXCEPTION_TYPES)
            ),
            # A bare `spec_id`'s rebuild raises what a mint and the registry raise, and the
            # ones whose codes this endpoint registers pass through.
            *(
                exemplar
                for exemplar in (
                    *MINT_EXCEPTION_EXEMPLARS,
                    IdentityCollision(subject_id="spec_exemplar"),
                )
                if passes_through(exemplar)
            ),
        ),
    )


def build_export_link_door(*, deps: DepsCell) -> DoorContribution:
    """The export link route, on the `datasets.export_link` endpoint. It admits every caller."""
    router = APIRouter()

    @router.get(
        LINK_PATH,
        status_code=302,
        response_class=RedirectResponse,
        responses={302: REDIRECT_RESPONSE},
    )
    async def get_export(export_id: str) -> RedirectResponse:
        """Redirect to the export's file, signed for five minutes. Renders nothing."""
        current = deps.get()
        with door_call("export_link", logger=logger) as call:
            url = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    sign_export,
                    call.timings,
                    export_id=export_id,
                    current=current,
                ),
                refuse=ExportBackpressure,
            )
        return RedirectResponse(
            url, status_code=302, headers=no_store_headers(server_timing=call.server_timing)
        )

    return DoorContribution(
        router=router,
        registry_endpoints={LINK_PATH: "datasets.export_link"},
        classify=classify_files_exception,
        exception_types=(
            ExportNotFound,
            # The recorded id resolves through the spec registry, which raises it for a
            # quarantined id or served id bound to it.
            IdentityQuarantined,
            SnapshotRevoked,
            LicenceNotRedistributable,
            ExportBackpressure,
            SnapshotSigningFailed,
            # Wrapped before classification, but signing can raise it.
            ArtifactSigningError,
            DatasetsStoreUnavailable,
        ),
        exception_exemplars=(
            ExportNotFound(export_id="exp_exemplar"),
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            LicenceNotRedistributable(sources=[("dcid:exemplar", "exemplar")]),
            ExportBackpressure(),
            SnapshotSigningFailed(
                bucket="exemplar-bucket",
                object_name="sha256:exemplar",
                digest="sha256:exemplar",
                error_class_name=None,
                status_code=None,
            ),
            # A permission failure an operator must clear answers 500.
            SnapshotSigningFailed(
                bucket="exemplar-bucket",
                object_name="sha256:exemplar",
                digest="sha256:exemplar",
                error_class_name=None,
                status_code=403,
            ),
            ArtifactSigningError(
                bucket="exemplar-bucket", object_name="sha256:exemplar", digest="sha256:exemplar"
            ),
            DatasetsStoreUnavailable("exemplar"),
        ),
    )


__all__ = ["ExportBackpressure", "build_export_link_door", "build_exports_door"]
