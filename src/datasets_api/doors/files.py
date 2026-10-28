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

`GET /api/datasets/{id}/{content_version}/file?format=&bundle=&layout=` selects one file of
the version and answers 302 to it, signed as a link is. The query takes the closed
namespace's `format`, `bundle` and `layout` (`layout` takes only `frame`), with the defaults
of the definition GET, and any other key refuses `request_invalid` before the version is read. The
version reads under the same served-id, quarantine and revocation rules as a link. Its options
are the requested ones laid over the version's own minting options
(`recorded_rendering.recorded_minting`), so a derived version renders as itself, and a derived
version whose files record none refuses `artifact_expired`. A file the version's snapshot lists
under that rendering, linked under a key current for the serving code index's
`enrichment_digest` (`version_store.current_rendering_keys`), is signed in hand. Otherwise,
on the work executor under a materialise permit and the request deadline, a file of that
rendering past working retention whose bytes survive renews as its link's renewal does, and failing that the version renders from its stored
frame (`render_version_from_frame`, which a link's renewal also calls) and the redirect names
the file the render returned. A frame past working retention renews first, and the selector
refuses `artifact_expired` only when neither the file's bytes nor the frame's survive. The
render's refusals answer as those of the definition GET do: an `xlsx` over a sheet's limit refuses
`excel_limit` (422). A bundle render reads graph metadata for its citations.

The read and the signing run on the read executor. Presentation reads frames in process and
never calls these routes.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping, Sequence

from fastapi import APIRouter, Request
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
from datasets_api.codes_materialise import CapabilityUnavailable, signing_failure_is_permanent
from datasets_api.codes_snapshot import ArtifactExpired, SnapshotSigningFailed, SpecNotFound
from datasets_api.config import DOWNLOAD_SIGNED_URL_TTL
from datasets_api.current import (
    DefinitionDeps,
    StoredFrame,
    StoredTraceMissingResolvedStatVars,
    head_rendering,
    plan_stored_render,
)
from datasets_api.deps import Deps, DepsCell
from datasets_api.doors.definition import _run_retrieval, request_definition_deps
from datasets_api.doors.mint_errors import MintBackpressure, classify_mint_exception
from datasets_api.doors.mint_inventory import MINT_EXCEPTION_EXEMPLARS
from datasets_api.doors.snapshot import read_under_served_id, servable_listable_locations
from datasets_api.headers import no_store_headers
from datasets_api.manifest import stored_manifest, stored_manifest_facts
from datasets_api.manifest_models import CitationModel, Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.query_options import QueryOptionsInvalid, parse_query_options, pointer_for_key
from datasets_api.recorded_rendering import recorded_minting, serving_frames
from datasets_api.render import (
    artifact_rendering,
    artifact_rendering_keys,
    complete_rendering,
    rendering_options,
    rendition_token,
    stored_artifact_file_name,
    stored_frame_location,
)
from datasets_api.timings import door_call, phase, run_with_timings
from datasets_api.version_store import current_rendering_keys, rendering_key
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from materialise.formats import ExcelIllegalCharacterError, ExcelLimitError, ParquetSizeBoundError
from materialise.reshape import RollupNotSummableError
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
    # A selector render raises the mint's rendering refusals, which its classifier maps.
    return classify_mint_exception(exc)


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
    try:
        stored, subject_ids = _read_version(
            id=id, content_version=content_version, deps=deps, artifact_digest=digest
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
    render: Callable[..., Manifest | None],
    deps: DefinitionDeps,
) -> Manifest | None:
    """Renew `aged_frame`'s object (`_reupload`) and render from the renewed bytes with
    `render`, on a coverage snapshot taken here, returning what `render` returns. Bytes that
    are gone render nothing and return `None`. The caller holds a materialise permit and binds
    no frame itself, so the bytes are freed when this returns, before the permit is
    released."""
    data = _reupload(aged_frame, deps=deps)
    if data is None:
        return None
    return render(
        coverage_snapshot=deps.coverage.snapshot(),
        frame=StoredFrame(location=aged_frame, data=data),
    )


def _render_from_stored_frame(
    source: ServableVersion,
    *,
    subject_ids: tuple[str, ...],
    options: MintOptions,
    deps: DefinitionDeps,
) -> Manifest | None:
    """Render `options` of the version `source` from its stored frame through the definition
    head path (`current.head_rendering`), and return the manifest that path gives, or `None`
    when no frame bytes of `source` are left to render from or its frame renders nothing
    (`current.StoredRenderPlan.frame_mode` `None`, or a render that gives another version).

    A servable frame renders first, read by the head path under its own slot. Only when the
    version lists none, or it renders nothing (its object gone, or a frame1 document a
    party-totals cut cannot read), does a frame past retention renew (`_reupload`) and the
    render take the renewed bytes, reading the frame once. One materialise slot holds from
    before the read to the end of the render, and the coverage snapshot is taken inside it.
    Before an aged frame renews, `plan_stored_render` decides from the version's record the
    refusals it alone decides, so each raises with no frame re-uploaded or rendered and no
    slot taken. Whether the aged frame's bytes are still in the store is known only by
    fetching them, so that refusal also answers a version whose frame bytes are gone."""
    render = functools.partial(
        head_rendering,
        subject_id=source.version.subject_id,
        content_version=source.version.content_version,
        deps=deps,
        options=options,
        trace_id=None,
        revoked_under=subject_ids,
    )
    if stored_frame_location(source.servable) is not None:
        rendered = render(coverage_snapshot=deps.coverage.snapshot())
        if rendered is not None:
            return rendered
    aged_frame = stored_frame_location(source.aged_out)
    if aged_frame is None:
        return None
    plan = plan_stored_render(
        stored_manifest(source.version),
        options=options,
        subject_id=source.version.subject_id,
        content_version=source.version.content_version,
        read_version=lambda: source.version,
        read_locations=lambda: (*source.servable, *source.aged_out),
        deps=deps,
    )
    if plan.frame_mode is None:
        return None
    with bounded_materialise_slot(
        deps.materialise_semaphore,
        wait_seconds=deps.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        return _render_renewed(aged_frame, render=functools.partial(render, plan=plan), deps=deps)


def render_version_from_frame(
    *,
    stored: ServableVersion,
    subject_ids: tuple[str, ...],
    options: MintOptions,
    deps: DefinitionDeps,
) -> Manifest | None:
    """Render `options` of the version `stored` from the first of its serving frames
    (`recorded_rendering.serving_frames`) that renders it (`_render_from_stored_frame`),
    through the definition head path, which links the result, and return the manifest that
    path gives, or `None` when no frame bytes are left to render from. The files door's
    renewal (`renew_and_resolve`) and the version file selector (`render_and_select`) both
    render here.

    A derived version (`base_content_version` set: a groupby collapse or a party-totals cut)
    stored its own frame when it was minted, and renders from it first when its record says
    what that frame holds, so the render reads only what the version pinned. `options` carry
    the derived version's own minting options, which name the rendering. With no bytes of
    that frame left, or a frame that renders nothing, the derived version renders from its
    base version's frame, as its mint did, read under `subject_ids`, so a revoked base
    refuses `snapshot_revoked`. That render derives again under the current declarations and
    serves the version only when it gives the same content version, which the callers
    check."""
    frames = serving_frames(
        stored.version,
        renderings=[location.rendering for location in (*stored.servable, *stored.aged_out)],
    )
    for frame in frames:
        source = (
            stored
            if frame.content_version == stored.version.content_version
            else read_servable_version(
                deps.snapshot_reader,
                subject_id=stored.version.subject_id,
                content_version=frame.content_version,
                now=deps.now(),
                revoked_under=subject_ids,
            )
        )
        rendered = _render_from_stored_frame(
            source, subject_ids=subject_ids, options=options, deps=deps
        )
        if rendered is not None:
            return rendered
    return None


def renew_and_resolve(*, id: str, content_version: str, digest: str, deps: DefinitionDeps) -> str:
    """Renew the digest's object, then resolve the link as `resolve_link` does.

    The clicked digest's own object renews when still present (`_renew`). Otherwise its
    rendering renders again from the stored frame (`render_version_from_frame`) under the
    options of its complete form (`render.complete_rendering`), so an incomplete rendering
    renders as the complete one its request named and the head path links the result under
    the complete key. The stored frame (`render.stored_frame_location`), statistics renderings,
    a superseded bundle rendition and an old bare layout have no options that render them
    (`render.rendering_options`) and render nothing. The version is then read again in full and
    decides the outcome by digest: a render that gives the clicked bytes makes the link
    servable, and a render that linked other bytes in the digest's place refuses
    `ArtifactExpired`, as does a digest still not servable.
    """
    stored, subject_ids, location = _read(
        id=id, content_version=content_version, digest=digest, deps=deps
    )
    if location is None:
        clicked = next(loc for loc in stored.aged_out if loc.artifact_digest == digest)
        if not _renew(clicked, deps=deps):
            # The complete form of an incomplete rendering renders as the request named it.
            options = rendering_options(
                complete_rendering(clicked.rendering), site=deps.settings.site
            )
            if options is not None:
                render_version_from_frame(
                    stored=stored, subject_ids=subject_ids, options=options, deps=deps
                )
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


SELECTOR_QUERY_KEYS = frozenset({"format", "bundle", "layout"})
"""The query keys the version file selector reads. A reshape (`groupby.*`, `pivot.*`,
`columns`, `without_party_totals`) makes another dataset, which `GET /api/datasets/{id}`
serves."""

# The rendering refusals a render from the stored frame can raise for a format, a bundle, a
# layout and a derived version's own minting options.
_SELECTOR_RENDERING_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    StoredTraceMissingResolvedStatVars,
    CapabilityUnavailable,
    ExcelLimitError,
    ExcelIllegalCharacterError,
    ParquetSizeBoundError,
    RollupNotSummableError,
)


def selector_options(items: Sequence[tuple[str, str]]) -> MintOptions:
    """The options the selector's query `items` name, parsed as the definition GET parses its
    closed namespace (`query_options.parse_query_options`). A key outside
    `SELECTOR_QUERY_KEYS` raises `QueryOptionsInvalid` at its body pointer."""
    for key, _value in items:
        if key not in SELECTOR_QUERY_KEYS:
            raise QueryOptionsInvalid(
                pointer=pointer_for_key(key),
                detail=(
                    f"query key {key!r} does not select a stored version's file; the file "
                    "selector reads format, bundle and layout, and GET /api/datasets/{id} "
                    "reshapes a dataset"
                ),
            )
    return parse_query_options(items)


def _read_version(
    *,
    id: str,
    content_version: str,
    deps: Deps | DefinitionDeps,
    artifact_digest: str | None = None,
) -> tuple[ServableVersion, tuple[str, ...]]:
    """The version the path names, read under the served ids bound to `id`
    (`read_under_served_id`), and the ids whose revocation withdraws it. With `artifact_digest`
    the same transaction checks that the version links it."""

    def _read_under(
        subject_id: str, subject_ids: tuple[str, ...]
    ) -> tuple[ServableVersion, tuple[str, ...]]:
        stored = read_servable_version(
            deps.snapshot_reader,
            subject_id=subject_id,
            content_version=content_version,
            artifact_digest=artifact_digest,
            now=deps.now(),
            revoked_under=subject_ids,
        )
        return stored, subject_ids

    return read_under_served_id(id=id, spec_reader=deps.spec_reader, read=_read_under)


def _version_options(stored: ServableVersion, requested: MintOptions, *, id: str) -> MintOptions:
    """`requested` laid over the minting options of the version `stored`, so a derived version
    renders as itself. A raw version has none. A derived version takes the ones a file linked
    under it records (`recorded_rendering.recorded_minting`), and one whose files record none
    refuses `ArtifactExpired`, as it has no options that render it."""
    if stored.version.base_content_version is None:
        return requested
    minting = recorded_minting(loc.rendering for loc in (*stored.servable, *stored.aged_out))
    if minting is None:
        raise ArtifactExpired(id=id, content_version=stored.version.content_version)
    return requested.model_copy(
        update={
            "groupby": minting.groupby,
            "without_party_totals": minting.without_party_totals,
        }
    )


def select_file(
    *, id: str, content_version: str, requested: MintOptions, enrichment_digest: str, deps: Deps
) -> str | None:
    """The signed URL of the file the version links under the rendering `requested` names, or
    `None` when the version lists no servable file of that rendering and it must render first
    (`render_and_select`). The version's listable files are the ones its snapshot lists
    (`doors.snapshot.servable_listable_locations`), and a file holding party cells is the one
    linked under a key current for `enrichment_digest`, the serving code index's.

    Raises `SpecNotFound`, `IdentityQuarantined`, `SnapshotNotFound`, `SnapshotRevoked` or
    `ArtifactExpired` before any signing."""
    stored, _subject_ids = _read_version(id=id, content_version=content_version, deps=deps)
    keys = artifact_rendering_keys(
        _version_options(stored, requested, id=id),
        site=deps.settings.site,
        enrichment_digest=enrichment_digest,
    )
    listable = servable_listable_locations(
        stored.servable,
        now=deps.now(),
        rendition=rendition_token(deps.settings.site),
        # `keys` are the current keys, so matching them keeps only current links.
        enrichment_digest=None,
    )
    location = next((loc for loc in listable if loc.rendering_key in keys), None)
    if location is None:
        return None
    return _signed_url(location, stored=stored, artifact_store=deps.artifact_store)


def _render_digest(
    *,
    stored: ServableVersion,
    subject_ids: tuple[str, ...],
    options: MintOptions,
    rendering: Mapping[str, object],
    deps: DefinitionDeps,
) -> str | None:
    """The digest of the artifact whose complete rendering is `rendering` in the manifest a
    render of `options` from the version's stored frame gives (`render_version_from_frame`), or
    `None` when no frame bytes are left or the render gives another version."""
    manifest = render_version_from_frame(
        stored=stored, subject_ids=subject_ids, options=options, deps=deps
    )
    if manifest is None or manifest.content_version != stored.version.content_version:
        return None
    key = rendering_key(rendering)
    return next(
        (
            artifact.artifact_digest
            for artifact in manifest.artifacts
            if rendering_key(complete_rendering(artifact.rendering)) == key
        ),
        None,
    )


def render_and_select(
    *, id: str, content_version: str, requested: MintOptions, deps: DefinitionDeps
) -> str:
    """Serve the rendering `requested` names of the version, which lists no servable file of
    it, and return the signed URL of the file served.

    A file of that rendering past working retention whose bytes survive renews (`_renew`), as
    its download link's renewal does. Otherwise the version renders from its stored frame
    (`render_version_from_frame`), and the file served is the manifest's artifact whose
    complete rendering (`render.complete_rendering`) is the requested one, named by digest, so
    a render linked as incomplete (`render.incomplete_rendering`) is served too. A version with
    no frame bytes left, or a render that gives another version, refuses `ArtifactExpired`."""
    stored, subject_ids = _read_version(id=id, content_version=content_version, deps=deps)
    options = _version_options(stored, requested, id=id)
    rendering = artifact_rendering(options, site=deps.settings.site)
    # A file linked under an earlier code index's digest holds stale party cells, so only a
    # file under a current key renews.
    keys = current_rendering_keys(rendering, deps.code_index.enrichment_digest)
    aged = next((loc for loc in stored.aged_out if loc.rendering_key in keys), None)
    if aged is not None and _renew(aged, deps=deps):
        digest = aged.artifact_digest
    else:
        digest = _render_digest(
            stored=stored,
            subject_ids=subject_ids,
            options=options,
            rendering=rendering,
            deps=deps,
        )
    if digest is None:
        raise ArtifactExpired(id=id, content_version=content_version)
    rendered = read_servable_version(
        deps.snapshot_reader,
        subject_id=stored.version.subject_id,
        content_version=content_version,
        now=deps.now(),
        artifact_digest=digest,
        revoked_under=subject_ids,
    )
    location = rendered.servable_digest(digest)
    if location is None:
        raise ArtifactExpired(id=id, content_version=content_version)
    return _signed_url(location, stored=rendered, artifact_store=deps.artifact_store)


def _option_query_schema(name: str) -> dict[str, object]:
    """The query parameter schema of the `MintOptions` field `name`: its one non-null type, its
    allowed values and its non-null default."""
    prop = MintOptions.model_json_schema()["properties"][name]
    (branch,) = [b for b in prop.get("anyOf", [prop]) if b.get("type") != "null"]
    schema: dict[str, object] = {"type": branch["type"]}
    values = branch.get("enum", [branch["const"]] if "const" in branch else None)
    if values is not None:
        schema["enum"] = values
    if prop.get("default") is not None:
        schema["default"] = prop["default"]
    return schema


SELECTOR_QUERY_PARAMETERS: list[dict[str, object]] = [
    {
        "name": "format",
        "in": "query",
        "required": False,
        "description": "the file format; csv when absent",
        "schema": _option_query_schema("format"),
    },
    {
        "name": "bundle",
        "in": "query",
        "required": False,
        "description": "true for a zip bundle with its documentation; true when absent",
        "schema": _option_query_schema("bundle"),
    },
    {
        "name": "layout",
        "in": "query",
        "required": False,
        "description": "frame for a bare parquet file in the frame layout; needs format=parquet and bundle=false",
        "schema": _option_query_schema("layout"),
    },
]

_REDIRECT_RESPONSE = {
    "description": "a redirect to the artifact's bytes",
    "headers": {
        "Location": {
            "description": "a GET URL signed for at most five minutes",
            "schema": {"type": "string"},
        }
    },
}


def build_files_door(*, deps: DepsCell) -> DoorContribution:
    """Build the download door over the surface's dependency cell. The read and the signing run
    on the read executor, and a renewal on the work executor under the request deadline."""
    router = APIRouter()

    @router.get(
        "/api/datasets/{id:dfo_spec_id}/{content_version}/files/{digest}",
        status_code=302,
        response_class=RedirectResponse,
        responses={302: _REDIRECT_RESPONSE},
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

    @router.get(
        "/api/datasets/{id:dfo_spec_id}/{content_version}/file",
        status_code=302,
        response_class=RedirectResponse,
        responses={302: _REDIRECT_RESPONSE},
        openapi_extra={"parameters": SELECTOR_QUERY_PARAMETERS},
    )
    async def get_version_file(id: str, content_version: str, request: Request) -> RedirectResponse:
        """Redirect to the version's file in the format, bundle and layout the query names,
        renewing an aged copy whose bytes survive, else rendering it from the version's stored
        frame."""
        requested = selector_options(request.query_params.multi_items())
        current = deps.get()
        with door_call("file_selector", logger=logger) as call:
            url = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    select_file,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    requested=requested,
                    # Read here, where the request binds its graph release.
                    enrichment_digest=current.code_index.enrichment_digest,
                    deps=current,
                ),
                refuse=DownloadBackpressure,
            )
            if url is None:
                url = await _run_retrieval(
                    current.work_executor,
                    render_and_select,
                    deps=request_definition_deps(current),
                    timings=call.timings,
                    refuse=DownloadBackpressure,
                    id=id,
                    content_version=content_version,
                    requested=requested,
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
            # The file selector's query and its render from the stored frame.
            QueryOptionsInvalid,
            *_SELECTOR_RENDERING_EXCEPTION_TYPES,
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
            QueryOptionsInvalid(pointer="/options/exemplar", detail="exemplar"),
            StoredTraceMissingResolvedStatVars(
                subject_id="spec_exemplar", content_version="cv_exemplar"
            ),
            *(
                exemplar
                for exemplar in MINT_EXCEPTION_EXEMPLARS
                if isinstance(exemplar, _SELECTOR_RENDERING_EXCEPTION_TYPES)
            ),
        ),
    )
