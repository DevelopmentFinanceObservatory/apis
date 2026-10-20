"""The files door serves `GET /api/datasets/{id}/{content_version}/files/{digest}`, the stable
download link every manifest lists for each artifact (`artifact_delivery.artifact_download_path`).

The link resolves at the click. The path id resolves to the served ids the registry binds to
it, and a version, or a digest, recorded under the posted id itself resolves too
(`doors.snapshot.read_under_served_id`). A link naming the served id honours revocations under
it, and a link naming a posted id honours revocations under the posted id and every served id
bound to it. A quarantined path id, or a quarantined served id bound to it, refuses
`identity_quarantined` and signs nothing. A
manifest names its links under the id its request named (`render.linked_under`): the snapshot
door, a definition GET and a mint that posts a `spec_id` name the requested id, while a mint
that posts a definition or a scope, and an operation poll, name the served id. Revocations are
written by operators in SQL, and a version revoked under its served id is taken down
everywhere.

One `read_servable_version` transaction checks that the version exists, that neither it nor
its base version is revoked under those ids, and that the digest is linked to it, so a revoked
version refuses `snapshot_revoked` and signs nothing. A digest the version no longer links
refuses `artifact_expired` when this deployment stored its bytes, since a re-render replaced
it, and `artifact_not_found` otherwise. A servable location is signed in hand for
`DOWNLOAD_SIGNED_URL_TTL` and answered 302.

A digest whose working location is past retention renews on the work executor under a
materialise permit and the request deadline. When the digest's object is still present, `get`
verifies it and `put` renews it under a new object name (`artifact_store`), whatever its
rendering, statistics included. When the object is gone and the rendering is one this build
renders, the stored frame renews the same way and the rendering re-renders from its bytes,
which renews the rendering when the bytes match. An incomplete rendering
(`render.incomplete_rendering`) re-renders as the complete rendering its request named. The
read then runs again, and a digest still unservable, because the frame's object is gone too,
the rendering is not one this build renders, or the render produced other bytes, refuses
`artifact_expired`, pointing the caller at definition retrieval.

The read and the signing run on the read executor. Presentation reads frames in process and
never calls this route.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from artifact_store import ArtifactSigningError, ArtifactStore, Tier
from artifact_store.errors import ArtifactIntegrityError, ArtifactNotFoundError
from dataset_store.snapshot import (
    ArtifactNotFound,
    ServableVersion,
    SnapshotNotFound,
    SnapshotRevoked,
    StoredArtifactLocation,
    read_servable_version,
)

# Register the route converter before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.admission import bounded_materialise_slot
from datasets_api.artifact_delivery import artifact_content_disposition
from datasets_api.backpressure import Backpressure
from datasets_api.codes_materialise import signing_failure_is_permanent
from datasets_api.codes_snapshot import ArtifactExpired, SnapshotSigningFailed, SpecNotFound
from datasets_api.config import DOWNLOAD_SIGNED_URL_TTL
from datasets_api.current import DefinitionDeps, StoredFrame, head_rendering, plan_stored_render
from datasets_api.deps import Deps, DepsCell
from datasets_api.doors.definition import _run_retrieval, request_definition_deps
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.snapshot import read_under_served_id
from datasets_api.headers import no_store_headers
from datasets_api.manifest import stored_manifest, stored_manifest_facts
from datasets_api.manifest_models import CitationModel
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.render import (
    complete_rendering,
    rendering_options,
    stored_artifact_file_name,
    stored_frame_location,
)
from datasets_api.timings import door_call, phase, run_with_timings
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from spec_registry import IdentityQuarantined

logger = logging.getLogger(__name__)


class DownloadBackpressure(Backpressure):
    """A full read or work executor maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the download door is at its concurrent-call capacity right now; retry"


def classify_files_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, SnapshotSigningFailed):
        # Stored bytes are unchanged. Permanent signing failures map to 500.
        # Transient or unknown causes map to retryable 503.
        if signing_failure_is_permanent(
            error_class_name=exc.error_class_name, status_code=exc.status_code
        ):
            logger.error(
                "Datasets download door failed to sign a GET url for a stored artifact "
                "for a permanent, operator-clearable reason: %s",
                exc,
                exc_info=exc,
            )
            return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)
        logger.error(
            "Datasets download door failed to sign a GET url for a stored artifact: %s",
            exc,
            exc_info=exc,
        )
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="signing this snapshot's artifact download url failed; retry",
            retryable=True,
        )
    return classify_datasets_refusal(exc)


def _signed_url(
    location: StoredArtifactLocation, *, stored: ServableVersion, artifact_store: ArtifactStore
) -> str:
    """A URL signed for `DOWNLOAD_SIGNED_URL_TTL` over the location in hand, naming the download
    as the mint names it. Raises `SnapshotSigningFailed` when signing fails."""
    facts = stored_manifest_facts(stored.version)
    file_name = stored_artifact_file_name(
        location,
        citation=CitationModel.model_validate(facts.citation),
        prepared_at=facts.prepared_at,
        content_version=stored.version.content_version,
    )
    try:
        with phase("sign"):
            url = artifact_store.signed_location_url(
                location.artifact_digest,
                tier=Tier(location.tier),
                bucket=location.bucket,
                object_name=location.object_name,
                ttl=DOWNLOAD_SIGNED_URL_TTL,
                response_type=location.media_type,
                response_disposition=artifact_content_disposition(file_name),
            )
    except ArtifactSigningError as exc:
        # Copy safe fields here, then raise outside the handler so Python leaves
        # `__context__` unset.
        failure = (exc.error_class_name, exc.status_code)
    else:
        return url
    raise SnapshotSigningFailed(
        bucket=location.bucket,
        object_name=location.object_name,
        digest=location.artifact_digest,
        error_class_name=failure[0],
        status_code=failure[1],
    )


def _read(
    *, id: str, content_version: str, digest: str, deps: Deps | DefinitionDeps
) -> tuple[ServableVersion, tuple[str, ...], StoredArtifactLocation | None]:
    """The version the link names, the ids whose revocation withdraws it, and the digest's
    servable location, `None` when every location of it is past retention.

    A digest the version no longer links, though this deployment stored its bytes, is one a
    re-render replaced, and refuses `ArtifactExpired`. A digest it never stored refuses
    `ArtifactNotFound`."""

    def _read_under(
        subject_id: str, subject_ids: tuple[str, ...]
    ) -> tuple[ServableVersion, tuple[str, ...]]:
        stored = read_servable_version(
            deps.snapshot_reader,
            subject_id=subject_id,
            content_version=content_version,
            artifact_digest=digest,
            now=deps.now(),
            revoked_under=subject_ids,
        )
        return stored, subject_ids

    try:
        stored, subject_ids = read_under_served_id(
            id=id, spec_reader=deps.spec_reader, read=_read_under
        )
    except ArtifactNotFound:
        if deps.snapshot_reader.artifact_inventoried(digest):
            raise ArtifactExpired(id=id, content_version=content_version) from None
        raise
    return stored, subject_ids, stored.servable_digest(digest)


def resolve_link(*, id: str, content_version: str, digest: str, deps: Deps) -> str | None:
    """The signed URL a servable link redirects to, or `None` when the digest's working
    location is past retention and must renew first (`renew_and_resolve`).

    Raises `SpecNotFound`, `SnapshotNotFound`, `SnapshotRevoked`, `ArtifactExpired` or
    `ArtifactNotFound` before any signing.
    """
    stored, _subject_ids, location = _read(
        id=id, content_version=content_version, digest=digest, deps=deps
    )
    if location is None:
        return None
    return _signed_url(location, stored=stored, artifact_store=deps.artifact_store)


def _reupload(location: StoredArtifactLocation, *, deps: DefinitionDeps) -> bytes | None:
    """Renew `location`, past working retention, under a new object name, and return its bytes,
    or `None` when its object is missing or corrupt. The caller holds a materialise permit
    from before this call until it has dropped the bytes, since they are held whole.

    The lifecycle rule deletes aged objects at no fixed time, so the object may still be
    present. `get` verifies the bytes against the digest, and `put` of the same bytes renews
    them.
    """
    tier = Tier(location.tier)
    try:
        data = deps.artifact_store.get(artifact_digest=location.artifact_digest, tier=tier)
    except (ArtifactNotFoundError, ArtifactIntegrityError):
        return None
    deps.artifact_store.put(data, tier=tier)
    return data


def _renew(location: StoredArtifactLocation, *, deps: DefinitionDeps) -> bool:
    """Renew `location`'s object (`_reupload`) inside one materialise slot and return whether
    it renewed. The bytes stay inside the slot."""
    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        return _reupload(location, deps=deps) is not None


def _render_renewed(
    aged_frame: StoredArtifactLocation,
    *,
    render: Callable[..., object],
    deps: DefinitionDeps,
) -> None:
    """Renew `aged_frame`'s object (`_reupload`) and render from the renewed bytes with
    `render`, on a coverage snapshot taken here. Bytes that are gone render nothing. The
    caller holds a materialise permit and binds no frame itself, so the bytes are freed when
    this returns, before the permit is released."""
    data = _reupload(aged_frame, deps=deps)
    if data is None:
        return
    render(
        coverage_snapshot=deps.coverage.snapshot(),
        frame=StoredFrame(location=aged_frame, data=data),
    )


def _re_render(
    clicked: StoredArtifactLocation,
    *,
    stored: ServableVersion,
    subject_ids: tuple[str, ...],
    deps: DefinitionDeps,
) -> None:
    """Render `clicked`'s rendering again from the stored frame it was rendered from, through
    the definition head path (`current.head_rendering`), which links the result.

    A derived version (`base_content_version` set: a groupby collapse or a party-totals cut)
    renders from its base version's frame, as its mint did, and the head path links the result
    under the derived version again. A frame past
    retention renews first (`_reupload`) and the render takes the renewed bytes, reading the
    frame once: one materialise slot holds from before the read to the end of the render, and
    the coverage snapshot is taken inside it. A frame still in retention is read by the head
    path under its own slot, and the head path decides the source version's refusals
    (`current.plan_stored_render`) once it has found a servable frame location, so a version
    with no frame renders nothing and the link answers `artifact_expired`. Before an aged
    frame renews, `plan_stored_render` decides from the source version's record the refusals
    it alone decides, so each raises with no frame re-uploaded or rendered and no slot taken.
    Whether the aged frame's bytes are still in the store is known only by fetching them, so
    that refusal also answers a link whose frame bytes are gone.

    The options are those of the complete form of `clicked`'s rendering
    (`render.complete_rendering`), so an incomplete rendering renders as the complete one its
    request named and the head path links the result under the complete key. The caller's
    read after the render decides the outcome by digest: a render that gives `clicked`'s
    bytes makes its link servable, and one that gives other bytes leaves it past retention.
    The stored frame (`render.stored_frame_location`), statistics renderings, a superseded
    bundle rendition and an old bare layout have no options that render them
    (`render.rendering_options`), and a gone frame leaves nothing to render from."""
    options = rendering_options(complete_rendering(clicked.rendering), site=deps.settings.site)
    if options is None:
        return
    source = stored
    if stored.version.base_content_version is not None:
        source = read_servable_version(
            deps.snapshot_reader,
            subject_id=stored.version.subject_id,
            content_version=stored.version.base_content_version,
            now=deps.now(),
            revoked_under=subject_ids,
        )
    render = functools.partial(
        head_rendering,
        subject_id=source.version.subject_id,
        content_version=source.version.content_version,
        deps=deps,
        options=options,
        trace_id=None,
    )
    aged_frame = stored_frame_location(source.aged_out)
    if aged_frame is None:
        render(coverage_snapshot=deps.coverage.snapshot())
        return
    plan = plan_stored_render(
        stored_manifest(source.version),
        options=options,
        subject_id=source.version.subject_id,
        content_version=source.version.content_version,
        read_version=lambda: source.version,
        deps=deps,
    )
    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        _render_renewed(aged_frame, render=functools.partial(render, plan=plan), deps=deps)


def renew_and_resolve(*, id: str, content_version: str, digest: str, deps: DefinitionDeps) -> str:
    """Renew the digest's object, then resolve the link as `resolve_link` does.

    The clicked digest's own object renews when still present (`_renew`); otherwise its
    rendering renders again (`_re_render`). The version is then read again in full, so a
    re-render that linked other bytes in the digest's place refuses `ArtifactExpired`, as does
    a digest still not servable.
    """
    stored, subject_ids, location = _read(
        id=id, content_version=content_version, digest=digest, deps=deps
    )
    if location is None:
        clicked = next(loc for loc in stored.aged_out if loc.artifact_digest == digest)
        if not _renew(clicked, deps=deps):
            _re_render(clicked, stored=stored, subject_ids=subject_ids, deps=deps)
        stored = read_servable_version(
            deps.snapshot_reader,
            subject_id=stored.version.subject_id,
            content_version=content_version,
            now=deps.now(),
            revoked_under=subject_ids,
        )
        location = stored.servable_digest(digest)
    if location is None:
        raise ArtifactExpired(id=id, content_version=content_version)
    return _signed_url(location, stored=stored, artifact_store=deps.artifact_store)


def build_files_door(*, deps: DepsCell) -> DoorContribution:
    """Build the download door over the surface's dependency cell. The read and the signing run
    on the read executor, and a renewal on the work executor under the request deadline."""
    router = APIRouter()

    @router.get(
        "/api/datasets/{id:dfo_spec_id}/{content_version}/files/{digest}",
        status_code=302,
        response_class=RedirectResponse,
        responses={
            302: {
                "description": "a redirect to the artifact's bytes",
                "headers": {
                    "Location": {
                        "description": "a GET URL signed for at most five minutes",
                        "schema": {"type": "string"},
                    }
                },
            }
        },
    )
    async def get_file(id: str, content_version: str, digest: str) -> RedirectResponse:
        current = deps.get()
        with door_call("files", logger=logger) as call:
            url = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    resolve_link,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    digest=digest,
                    deps=current,
                ),
                refuse=DownloadBackpressure,
            )
            if url is None:
                url = await _run_retrieval(
                    current.work_executor,
                    renew_and_resolve,
                    deps=request_definition_deps(current),
                    timings=call.timings,
                    refuse=DownloadBackpressure,
                    id=id,
                    content_version=content_version,
                    digest=digest,
                )
        return RedirectResponse(
            url, status_code=302, headers=no_store_headers(server_timing=call.server_timing)
        )

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.files"),
        classify=classify_files_exception,
        exception_types=(
            SpecNotFound,
            # The path id resolves through the spec registry, which raises it for a quarantined
            # path id or served id bound to it.
            IdentityQuarantined,
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactNotFound,
            ArtifactExpired,
            DownloadBackpressure,
            # A renewal renders under the shared materialise permit.
            MintBackpressure,
            SnapshotSigningFailed,
            # Wrapped before classification, but signing can raise it.
            ArtifactSigningError,
        ),
        exception_exemplars=(
            SpecNotFound(id="spec_exemplar"),
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            ArtifactNotFound(
                id="spec_exemplar", content_version="cv_exemplar", artifact_digest="sha256:exemplar"
            ),
            ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
            DownloadBackpressure(),
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
    )
