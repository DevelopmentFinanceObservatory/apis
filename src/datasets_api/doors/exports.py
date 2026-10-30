"""The exports door prepares a frame as a file and serves the link to it (`datasets_api.exports`).

- `POST /api/datasets/exports`, body `ExportRequest`, answers 200 `ExportResponse`. A request
  whose export is recorded answers the record on the read executor, rendering and spending
  nothing. Any other prepares on the work executor under the definition deadline, as the file
  selector renders. It admits every caller, and the site's proxy forwards it. The caller is
  named as the pin door names it: a holder of the `pin` grant spends the
  day's pin budget alone, and any other caller is resolved (`resolve_caller`) and spends its
  `pin` allowance, so data-mcp forwards `x-dfo-caller` and `x-dfo-caller-principal`. A frame
  of a dataset id alone (`{spec_id}`) is first resolved on the work executor to the version the
  analysis runner reads for a Spec root of it (`exports.resolve_export_frame`), and the request
  then runs as one naming that pair.
- `GET /api/datasets/exports/{export_id}` answers 302 to a URL signed for five minutes to the
  export's pinned file, with `Cache-Control: no-store`. It admits every caller and never
  renders. A version revoked since the export was prepared refuses `snapshot_revoked`.

The prepare (`build_exports_door`) binds the `datasets.exports` registry endpoint and the link
(`build_export_link_door`) `datasets.export_link`.
"""

from __future__ import annotations

import functools
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from artifact_store import ArtifactSigningError
from dataset_store.snapshot import SnapshotNotFound, SnapshotRevoked
from datasets_api.allowance import AllowanceExhausted, AllowanceKind
from datasets_api.backpressure import Backpressure
from datasets_api.codes_exports import (
    ExportNotFound,
    ExportOverCeiling,
    LicenceNotRedistributable,
    NewestVersionUnservable,
    ProvenanceUnavailable,
)
from datasets_api.codes_snapshot import ArtifactExpired, SnapshotSigningFailed, SpecNotFound
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import StoredTraceMissingResolvedStatVars
from datasets_api.deps import DepsCell
from datasets_api.doors.definition import _run_retrieval, request_definition_deps
from datasets_api.doors.files import (
    SELECTOR_RENDERING_EXCEPTION_TYPES,
    classify_files_exception,
)
from datasets_api.doors.mint_errors import MintBackpressure, classify_mint_exception
from datasets_api.doors.mint_inventory import MINT_EXCEPTION_EXEMPLARS
from datasets_api.exports import (
    BareExportFrame,
    ExportRequestBody,
    ExportResponse,
    prepare_export,
    recorded_export,
    resolve_export_frame,
    sign_export,
)
from datasets_api.headers import no_store_headers
from datasets_api.principals import Grant, Principal, Principals, granted_route_class
from datasets_api.retention_admission import OpenPinBudgetExhausted, PinBudgetExhausted
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import door_call, run_with_timings
from dfo_platform.caller import Caller, CallerHeaderInvalid, CallerUnresolved, resolve_caller
from dfo_platform.codes import bound_registry_endpoint
from dfo_platform.doors import DoorContribution
from spec_registry import IdentityQuarantined
from spec_registry.errors import IdentityCollision

logger = logging.getLogger(__name__)

PREPARE_PATH = "/api/datasets/exports"
LINK_PATH = "/api/datasets/exports/{export_id}"


class ExportBackpressure(Backpressure):
    """A full read or work executor maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the exports door is at its concurrent-call capacity right now; retry"


_REDIRECT_RESPONSE = {
    "description": "a redirect to the export's file",
    "headers": {
        "Location": {
            "description": "a GET URL signed for at most five minutes",
            "schema": {"type": "string"},
        }
    },
}


def _passes_through(exemplar: BaseException) -> bool:
    """Whether a refusal a bare `spec_id`'s rebuild raises reaches the caller unchanged: its
    code is one `datasets.exports` registers, as `exports.resolve_export_frame` decides."""
    try:
        with bound_registry_endpoint("datasets.exports"):
            return classify_mint_exception(exemplar) is not None
    except LookupError:
        return False


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
        spends nothing; a new pinned copy spends one unit of the caller's daily pin
        allowance."""
        current = deps.get()
        granted: Principal = request.state.principal
        caller: Caller | None = (
            None
            if Grant.pin in granted.grants
            else resolve_caller(request, current.settings.caller)
        )
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
            export_request = body.resolved(frame)
            facts = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    recorded_export,
                    call.timings,
                    request=export_request,
                    store=current.export_store,
                ),
                refuse=ExportBackpressure,
            )
            if facts is None:
                facts = await _run_retrieval(
                    current.work_executor,
                    prepare_export,
                    deps=retrieval,
                    timings=call.timings,
                    refuse=ExportBackpressure,
                    request=export_request,
                    caller=caller,
                    current=current,
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
            NewestVersionUnservable,
            CallerHeaderInvalid,
            CallerUnresolved,
            PinBudgetExhausted,
            OpenPinBudgetExhausted,
            AllowanceExhausted,
            ExportBackpressure,
            # A render from the stored frame takes the shared materialise permit.
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
            *(
                NewestVersionUnservable(code=code, detail="exemplar")
                for code in ("service_unavailable", "internal_error", "snapshot_not_found")
            ),
            CallerHeaderInvalid(),
            CallerUnresolved("exemplar"),
            PinBudgetExhausted(limit=0, now=exemplar_day),
            OpenPinBudgetExhausted(limit=0, now=exemplar_day),
            AllowanceExhausted(kind=AllowanceKind.pin, limit=0, now=exemplar_day),
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
                if _passes_through(exemplar)
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
        responses={302: _REDIRECT_RESPONSE},
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
