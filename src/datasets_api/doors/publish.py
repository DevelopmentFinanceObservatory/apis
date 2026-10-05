"""The publish door stores derived `ops_`/`ocv_` frames from analysis-mcp so
`LiveFrameStore` can serve them through the existing version and artifact path.

The request sends grammar-independent JSON in `role_keys` and `rows`. dfo-apis
and dfo-ai pin different `dfo-spec-kit` revisions, so this door builds frame1
with `Frame.build` and computes its digest with `render_and_store_frame1`.

The `datasets.publish` registry maps `publish_conflict` (409),
`snapshot_revoked` (410), `publish_not_permitted` (403), `request_invalid`
(400), `over_ceiling` (422), `service_unavailable` (503), and
`internal_error` (500). Byte and lineage conflicts share `publish_conflict`
but have distinct `constraint_id` values.

The immutable body stores `spec_id` as `op_spec_hash` and carries
`input_content_versions` as the derivation trace. Derived frames use their own
reference as the sole provenance input.

A repeat publish succeeds with `created=False` only when frame1 bytes and
`input_content_versions` both match. Different bytes or lineage raises a
non-retryable `publish_conflict` (409). The latter matters because `ocv_`
identifies a computation recipe. Reporting success would discard the second
call's lineage. `publish_frame` checks existing records before writing and
rereads after writing to detect a concurrent winner and compare its bytes and
lineage.

A divergent digest under `ocv_` means the same recipe may have produced
different bytes. The door logs this reproducibility failure at error level,
while returning the client-facing 409 `publish_conflict`.

Revocation uses `snapshot_reader.read_revocation` before digest comparison and
after recording. The first check makes `snapshot_revoked` (410) outrank byte or
lineage conflicts. The second runs even when this call inserted the version,
catching a concurrent publish followed by revocation. The foreign key requires
an existing version row. The first check runs for recorded pairs only.
`SnapshotRevoked` is non-retryable. Revocation rows are permanent.

Publishing requires the principal in `DATASETS_FIRST_PARTY_PRINCIPAL`, sent in
`DATASETS_PRINCIPAL_HEADER` (default `x-dfo-principal`). This allowlist is
independent of the pin door's `DATASETS_PIN_FIRST_PARTY_PRINCIPAL`, because
pinning is irreversible on this surface. An unset value yields an empty
allowlist.

`publish_frame` rejects more than 250,000 rows with `over_ceiling` (422), before
`Frame.build`. Coverage cannot estimate derived-frame size because analysis-mcp
computes the rows. Frame grammar rejects empty strings and lone surrogates.
Other layers set row, column, and string-length limits. Accepted frames are
permanent and eligible for pinned-tier retention.

`_admit_publish_content_encoding` and `_admit_publish_body_size` run as
middleware before FastAPI reads or decodes the body. The first refuses any
`Content-Encoding` other than `identity` (or none) and `gzip`. The second
refuses lengths above `MAX_PUBLISH_BODY_BYTES`, as well as missing or
unparseable lengths, before parsing chunked bodies. The legitimate client holds
its body in memory and can send `Content-Length`.

For non-chunked requests, ASGI reads at most the declared length, so understating
it truncates the body. Row and `PublishFrameRequest` field limits also apply to
parsed bodies.

The route class from `_publish_route_class` then admits each request in this
order: principal, permit, body read or gzip inflation, JSON decode.
`_admit_publish_request` checks the principal from the headers, so an
unauthorized caller gets `publish_not_permitted` (403) without holding a permit
or having its body read, whatever the body holds. The permit comes from the
door's semaphore, sized by `publish_max_concurrency`, and stays held until the
build ends, so it bounds body decoding and JSON parsing as well as the build.
When no permit is free the door refuses with `service_unavailable` (503).

The route decodes a gzip body before FastAPI reads it, in steps of at most
`_INFLATE_STEP_BYTES`, and refuses it with `PublishBodyTooLarge` as soon as the
decoded size passes `MAX_PUBLISH_BODY_BYTES`. Corrupt or truncated gzip data,
or bytes after the gzip member, refuse with `PublishBodyEncodingInvalid`. Both
map to `request_invalid` (400) before JSON decoding. A client disconnect during
the read returns the same 400 as one during an identity read.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import zlib
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from concurrent.futures import BrokenExecutor, Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dfo_frame_grammar import FrameError
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive

from artifact_store import (
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactStore,
    ArtifactWriteError,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.snapshot import LiveSnapshotReader, SnapshotReader, SnapshotRevoked
from datasets_api.config import DatasetsBuildSettings, DatasetsConfigError, DatasetsSettings
from datasets_api.door_executor import DoorExecutor, build_door_executor
from datasets_api.publish_models import (
    PublishedFrameCoordinates,
    PublishFrameRequest,
    PublishFrameResponse,
)
from datasets_api.render import FRAME1_MAX_ROWS, render_and_store_frame1
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import PhaseTimings, door_call, phase, run_with_timings
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import VersionRecord, VersionRecordWriter
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.trace import get_trace_id
from materialise.errors import OverCeilingError
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.frame_build import build_frame

logger = logging.getLogger(__name__)

# Fixed row ceiling. Coverage cannot estimate the rows analysis-mcp computes. Every published
# frame is stored as frame1, so the ceiling is frame1's.
PUBLISH_MAX_ROWS = FRAME1_MAX_ROWS

# Bound on the bytes one publish holds and JSON-decodes: the declared
# Content-Length of any body, and the decoded size of a gzip body. The door's
# memory grows with these bytes, so the bound comes from measured rows rather
# than from schema limits, which allow rows over 40 KiB. The 250,000-row
# benchmark body, with two role keys and synthetic keys, is 54,470,651 bytes of
# JSON, 218 bytes a row. Real DCIDs and StatVar names are longer, and 512 bytes
# a row is about 2.3 times the measured row. The 1,000,000 bytes cover metadata
# and JSON punctuation. At 250,000 rows the bound is 129,000,000 bytes, about
# 123 MiB. The door adds about 12 times the body size in RSS (645 MiB for the
# 54.5 MB body), so a body at the bound adds about 1.5 GiB. Cloud Run caps an
# HTTP/1 request at 32 MiB on the wire, so there only a gzip body can reach the
# bound.
_ADMISSION_ROW_BYTES = 512
MAX_PUBLISH_BODY_BYTES = PUBLISH_MAX_ROWS * _ADMISSION_ROW_BYTES + 1_000_000

# Most decoded bytes one `decompress` call returns, so inflating a gzip body
# allocates at most this much past `MAX_PUBLISH_BODY_BYTES` before refusing.
_INFLATE_STEP_BYTES = 1 << 20

_PUBLISH_CONTENT_ENCODINGS = frozenset({"identity", "gzip"})


class PublishFrameInvalid(Exception):
    """Frame grammar rejected `role_keys` or `rows`. Maps to non-retryable
    `request_invalid` (400), like schema validation failures.
    """

    def __init__(self, *, category: str, message: str) -> None:
        self.category = category
        super().__init__(message)


class PublishBodyTooLarge(Exception):
    """The body exceeds `MAX_PUBLISH_BODY_BYTES`, either by its declared
    `Content-Length` or, with `inflated`, by its gzip-decoded size. Maps to
    non-retryable `request_invalid` (400) before JSON decoding.

    Middleware constructs and classifies the declared-length case directly. A
    raise in middleware would be re-raised by `ServerErrorMiddleware`. The
    decoded-size case raises from the publish route class.
    """

    def __init__(self, *, content_length: int, limit: int, inflated: bool = False) -> None:
        self.content_length = content_length
        self.limit = limit
        self.inflated = inflated
        if inflated:
            super().__init__(
                f"the gzip request body (Content-Length {content_length}) inflates past this "
                f"door's own admission limit of {limit} bytes; refusing to decode more of a "
                "body than this door holds in memory for one publish at its own row ceiling"
            )
        else:
            super().__init__(
                f"request Content-Length {content_length} exceeds this door's own admission "
                f"limit of {limit} bytes; refusing to read a body larger than this door "
                "holds in memory for one publish at its own row ceiling"
            )


class PublishBodyEncodingInvalid(Exception):
    """The body's `Content-Encoding` is neither `identity` nor `gzip`, or its
    gzip data is corrupt, truncated, or followed by trailing bytes. Maps to
    non-retryable `request_invalid` (400) before JSON decoding.
    """


class PublishContentLengthRequired(Exception):
    """`Content-Length` is absent or unparseable, so middleware refuses to
    read an unbounded body. It maps to non-retryable `request_invalid` (400).

    Middleware classifies the exception directly. A live raise triggers
    `ServerErrorMiddleware`'s re-raise.
    """

    def __init__(self, *, content_length_header: str | None) -> None:
        self.content_length_header = content_length_header
        super().__init__(
            "this door requires a Content-Length header naming the request body's size in "
            "bytes before it will read that body; "
            + (
                "none was sent"
                if content_length_header is None
                else f"the value sent ({content_length_header!r}) does not parse as an integer"
            )
            + ". This door's only legitimate client posts a JSON body it already holds in "
            "memory, so send Content-Length rather than a chunked/unsized transfer"
        )


class PublishBytesDiverge(Exception):
    """This pair already holds different frame1 bytes. Maps to non-retryable
    `publish_conflict` (409). The caller must use the recorded bytes or a new id.
    """

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} already holds "
            "byte-different frame1 content than this publish call's own bytes; refusing "
            "to report success for a digest that was not actually recorded under this "
            "pair. This most likely means the analysis engine that minted this "
            "content_version did not reproduce the same bytes on a second run -- publish "
            "the bytes already held under this id, or mint a new content_version for the "
            "corrected recipe; retrying with these same bytes will refuse identically"
        )


class PublishLineageDiverge(Exception):
    """This pair holds identical frame1 bytes but a different
    `input_content_versions` trace. Because `ocv_` identifies a recipe,
    reporting success would discard the new lineage. Maps to non-retryable
    `publish_conflict` (409), distinguished from byte conflicts by `constraint_id`.
    """

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} already holds "
            "byte-identical frame1 content but a different input_content_versions "
            "derivation trace than this publish call's own; refusing to report success "
            "for a derivation trace that was not actually recorded under this pair. "
            "This most likely means this content_version was minted twice for genuinely "
            "different inputs -- mint a new content_version for the corrected derivation, "
            "or publish the input_content_versions already held under this id; retrying "
            "with these same claimed inputs will refuse identically"
        )


class PublishRecordMissing(Exception):
    """The post-write read returned `None` for the version row. The frame
    artifact is stored. This signals a writer/reader mismatch or failed insert.
    Maps to non-retryable `internal_error` (500). Retries with the same wiring
    fail identically.
    """

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} rendered and wrote "
            "frame1 bytes to the artifact store, but no version row is readable back "
            "through this door's own snapshot_reader immediately after recording one; this "
            "publish door was composed with a version_writer that does not share "
            "snapshot_reader's own store, or that writer's insert silently failed to land -- "
            "retrying will refuse identically under the same wiring"
        )


class PublishNotPermitted(Exception):
    """No allowed publish principal. Maps to `publish_not_permitted` (403)."""

    def __init__(self, *, principal: str | None) -> None:
        self.principal = principal
        detail = "the publish permission is not granted to this caller"
        if principal is None:
            detail = f"no principal header was presented; {detail}"
        super().__init__(detail)


class PublishBackpressure(Exception):
    """Publish capacity is full. Maps to retryable `service_unavailable` (503)."""


@dataclass(frozen=True)
class PublishResult:
    spec_id: str
    content_version: str
    artifact_digest: str
    created: bool


def _immutable_body(*, spec_id: str, input_content_versions: Sequence[str]) -> Mapping[str, object]:
    """Build the derived-frame manifest body from its op-spec id and lineage."""
    return {
        "op_spec_hash": spec_id,
        "input_content_versions": list(input_content_versions),
    }


def publish_frame(
    *,
    spec_id: str,
    content_version: str,
    role_keys: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    input_content_versions: Sequence[str],
    artifact_store: ArtifactStore,
    snapshot_reader: SnapshotReader,
    version_writer: VersionRecordWriter,
    trace_id: str | None = None,
    ceiling: int = PUBLISH_MAX_ROWS,
) -> PublishResult:
    """Publish one frame synchronously and verify the stored version.

    Reject rows over `ceiling` before frame construction. Store frame1 first,
    then check for an existing version. Unreferenced bytes remain in the
    working tier if recording fails. Revocation takes precedence. Matching
    bytes and lineage return `created=False`, while divergence raises the
    matching conflict. Otherwise record the version, then reread it to verify
    the winning bytes, lineage, and revocation state. `created` comes from the
    writer's insert result. Missing read-back raises `PublishRecordMissing`.
    """
    if len(rows) > ceiling:
        raise OverCeilingError(
            f"published frame has {len(rows)} rows, exceeding the sync materialisation "
            f"ceiling of {ceiling} rows",
            detail=(
                f"published frame has {len(rows)} rows, exceeding the sync materialisation "
                f"ceiling of {ceiling} rows"
            ),
            ceiling_kind="rows",
            count=len(rows),
            cap=ceiling,
        )

    try:
        with phase("build"):
            # Use the materialise frame builder for shared frame1 construction.
            frame = build_frame(role_keys=role_keys, rows=rows)
    except FrameError as exc:
        raise PublishFrameInvalid(category=exc.category, message=str(exc)) from exc

    artifact = render_and_store_frame1(
        frame_chunks=(frame,),
        subject_id=spec_id,
        content_version=content_version,
        artifact_store=artifact_store,
    )

    requested_body = _immutable_body(spec_id=spec_id, input_content_versions=input_content_versions)

    def _recorded_frame1_digest() -> str | None:
        locations = snapshot_reader.read_artifact_locations(spec_id, content_version)
        frame1_locations = [loc for loc in locations if loc.media_type == FRAME1_MEDIA_TYPE]
        return frame1_locations[0].artifact_digest if frame1_locations else None

    def _raise_if_diverges(stored: Mapping[str, object]) -> None:
        """Compare stored lineage after the caller confirms frame1 bytes match."""
        if dict(stored) != dict(requested_body):
            raise PublishLineageDiverge(spec_id=spec_id, content_version=content_version)

    with phase("check"):
        existing_version = snapshot_reader.read_version(spec_id, content_version)
        if existing_version is not None:
            revocation = snapshot_reader.read_revocation(spec_id, content_version)
            if revocation is not None:
                raise SnapshotRevoked(
                    id=spec_id, content_version=content_version, reason=revocation.reason
                )
            recorded_digest = _recorded_frame1_digest()
            if recorded_digest != artifact.artifact_digest:
                raise PublishBytesDiverge(spec_id=spec_id, content_version=content_version)
            _raise_if_diverges(existing_version.immutable_body)
            return PublishResult(
                spec_id=spec_id,
                content_version=content_version,
                artifact_digest=artifact.artifact_digest,
                created=False,
            )

    with phase("record"):
        created = version_writer.record_version(
            VersionRecord(
                subject_id=spec_id,
                subject_kind="spec",
                content_version=content_version,
                schema_version=1,
                immutable_body=requested_body,
                artifacts=(artifact,),
            ),
            trace_id=trace_id,
        )

    with phase("verify"):
        recorded_version = snapshot_reader.read_version(spec_id, content_version)
        recorded_revocation = snapshot_reader.read_revocation(spec_id, content_version)
        recorded_digest = _recorded_frame1_digest()
    if recorded_version is None:
        raise PublishRecordMissing(spec_id=spec_id, content_version=content_version)
    if recorded_revocation is not None:
        raise SnapshotRevoked(
            id=spec_id, content_version=content_version, reason=recorded_revocation.reason
        )
    if recorded_digest != artifact.artifact_digest:
        raise PublishBytesDiverge(spec_id=spec_id, content_version=content_version)
    _raise_if_diverges(recorded_version.immutable_body)

    return PublishResult(
        spec_id=spec_id,
        content_version=content_version,
        artifact_digest=artifact.artifact_digest,
        created=created,
    )


def classify_publish_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, PublishFrameInvalid):
        return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)

    if isinstance(exc, PublishBodyTooLarge):
        return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)

    if isinstance(exc, PublishBodyEncodingInvalid):
        return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)

    if isinstance(exc, PublishContentLengthRequired):
        return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)

    if isinstance(exc, PublishBytesDiverge):
        logger.error(
            "Datasets publish door refused a publish for spec_id %s content_version %s: "
            "byte-different content is already held under this exact pair -- most likely "
            "the analysis engine that minted this content_version did not reproduce the "
            "same bytes on a second run",
            exc.spec_id,
            exc.content_version,
        )
        return ProblemSpec(
            status=409,
            code="publish_conflict",
            detail=str(exc),
            retryable=False,
            constraint_id="publish_requires_matching_frame1_bytes",
        )

    if isinstance(exc, PublishLineageDiverge):
        # Both conflicts use `publish_conflict`. `constraint_id` distinguishes them.
        logger.error(
            "Datasets publish door refused a publish for spec_id %s content_version %s: "
            "byte-identical content is already held under this exact pair, but this call's "
            "own input_content_versions derivation trace differs from what is recorded -- "
            "most likely this content_version was minted twice for genuinely different "
            "inputs",
            exc.spec_id,
            exc.content_version,
        )
        return ProblemSpec(
            status=409,
            code="publish_conflict",
            detail=str(exc),
            retryable=False,
            constraint_id="publish_requires_matching_input_content_versions",
        )

    if isinstance(exc, SnapshotRevoked):
        # Reuse the pin door's refusal for the same revoked-version condition.
        return ProblemSpec(status=410, code="snapshot_revoked", detail=str(exc), retryable=False)

    if isinstance(exc, PublishRecordMissing):
        logger.error(
            "Datasets publish door wrote frame1 bytes for spec_id %s content_version %s but "
            "found no version row for that pair through its own snapshot_reader right after "
            "recording one: this door's version_writer does not share snapshot_reader's own "
            "store, or that writer's insert silently failed to land",
            exc.spec_id,
            exc.content_version,
        )
        return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)

    if isinstance(exc, PublishNotPermitted):
        return ProblemSpec(
            status=403, code="publish_not_permitted", detail=str(exc), retryable=False
        )

    if isinstance(exc, OverCeilingError):
        return ProblemSpec(
            status=422,
            code="over_ceiling",
            detail=exc.detail or str(exc),
            retryable=False,
            ceiling_kind=exc.ceiling_kind,
            count=exc.count,
            cap=exc.cap,
        )

    if isinstance(exc, PublishBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the publish door is at its concurrent-write capacity right now; retry",
            retryable=True,
        )

    if isinstance(exc, ArtifactRecordConflictError):
        # The digest points to different location metadata, often after a
        # bucket change. Reconcile the unrecorded upload as an infrastructure failure.
        logger.warning(
            "Datasets publish door's artifact write hit a record conflict for digest %s, "
            "tier %s: the freshly uploaded object has no inventory row pointing at it and "
            "needs manual reconciliation",
            exc.digest,
            exc.tier.value,
        )
        return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)

    if isinstance(exc, ArtifactInventoryError):
        logger.error(
            "Datasets publish door's artifact inventory write failed: %s", exc, exc_info=exc
        )
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="writing the published frame's artifact failed; retry",
            retryable=True,
        )

    if isinstance(exc, ArtifactWriteError):
        logger.error("Datasets publish door's artifact store write failed: %s", exc, exc_info=exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="writing the published frame's artifact failed; retry",
            retryable=True,
        )

    return None


_PUBLISH_ROUTE = "/api/datasets/frames/publish"
"""Shared by the route decorator below and the admission checks, so their
scoping can never drift from the route they guard."""


def _publish_content_encoding(request: Request) -> str:
    """Return the request's `Content-Encoding`, lowercased, or `identity` when absent.

    Repeated headers join with commas, so they never match an accepted value.
    """
    return ", ".join(request.headers.getlist("content-encoding")).strip().lower() or "identity"


def _admit_publish_content_encoding(request: Request) -> ProblemSpec | None:
    """Refuse a publish body in any encoding other than `identity` or `gzip` before reading it."""
    if request.method != "POST" or request.url.path != _PUBLISH_ROUTE:
        return None
    content_encoding = _publish_content_encoding(request)
    if content_encoding in _PUBLISH_CONTENT_ENCODINGS:
        return None
    return classify_publish_exception(
        PublishBodyEncodingInvalid(
            f"request Content-Encoding {content_encoding!r} is not accepted; this door "
            "decodes only 'gzip' and 'identity' (or no Content-Encoding header)"
        )
    )


def _admit_publish_body_size(request: Request) -> ProblemSpec | None:
    """Reject oversized or unusable `Content-Length` values before body parsing.

    The middleware applies to every request under the surface. This check
    returns `None` outside the publish route. Missing and unparseable lengths
    fail closed. It classifies constructed exceptions directly. This avoids
    raising them through `ServerErrorMiddleware`.
    """
    if request.method != "POST" or request.url.path != _PUBLISH_ROUTE:
        return None
    content_length_header = request.headers.get("content-length")
    if content_length_header is None:
        return classify_publish_exception(PublishContentLengthRequired(content_length_header=None))
    try:
        content_length = int(content_length_header)
    except ValueError:
        return classify_publish_exception(
            PublishContentLengthRequired(content_length_header=content_length_header)
        )
    if content_length <= MAX_PUBLISH_BODY_BYTES:
        return None
    return classify_publish_exception(
        PublishBodyTooLarge(content_length=content_length, limit=MAX_PUBLISH_BODY_BYTES)
    )


async def _inflate_gzip_body(request: Request, *, limit: int) -> bytes:
    """Decode a single-member gzip body, refusing it once its decoded size passes `limit`.

    Each `decompress` call returns at most `_INFLATE_STEP_BYTES`, so a gzip
    bomb allocates no more than `limit` plus one step before the refusal.
    """
    decompressor = zlib.decompressobj(wbits=31)
    pieces: list[bytes] = []
    inflated = 0

    def keep(piece: bytes) -> None:
        nonlocal inflated
        inflated += len(piece)
        if inflated > limit:
            raise PublishBodyTooLarge(
                content_length=int(request.headers["content-length"]), limit=limit, inflated=True
            )
        pieces.append(piece)

    trailing = PublishBodyEncodingInvalid(
        "the request body carries bytes after the end of its gzip member; this door "
        "decodes one gzip member per body"
    )
    try:
        async for chunk in request.stream():
            if chunk and decompressor.eof:
                raise trailing
            pending = chunk
            # Each pass hands the decompressor its pending bytes. A pass that leaves
            # `unconsumed_tail` has filled the step with output, so the loop advances;
            # one that reaches the member's end leaves the loop and refuses what remains.
            while pending and not decompressor.eof:
                keep(decompressor.decompress(pending, _INFLATE_STEP_BYTES))
                pending = decompressor.unconsumed_tail
            if decompressor.eof and (pending or decompressor.unused_data):
                raise trailing
        keep(decompressor.flush())
    except zlib.error as exc:
        raise PublishBodyEncodingInvalid(f"the request body is not valid gzip data: {exc}") from exc
    except ClientDisconnect as exc:
        # FastAPI answers a disconnect during an identity body read with this 400.
        raise HTTPException(status_code=400, detail="There was an error parsing the body") from exc
    if not decompressor.eof:
        raise PublishBodyEncodingInvalid(
            "the request body's gzip data ends before its gzip member does; the body is truncated"
        )
    return b"".join(pieces)


def _replay_body(body: bytes, receive: Receive) -> Receive:
    """Return a `receive` that delivers `body` as the whole request, then defers to `receive`."""
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if delivered:
            return await receive()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return replay


class _PublishPermit:
    """One publish's hold on the door's concurrency semaphore, taken before its body is read.

    `_run_publish` hands the permit to the executor future, which releases it
    when the build ends, even if the request is cancelled first. A permit the
    request never handed off releases when the request ends.
    """

    def __init__(self, semaphore: threading.BoundedSemaphore) -> None:
        self._semaphore = semaphore
        self._handed_off = False

    @classmethod
    def take(cls, semaphore: threading.BoundedSemaphore) -> _PublishPermit:
        """Acquire a permit without waiting, refusing with `PublishBackpressure` when none is free."""
        if not semaphore.acquire(blocking=False):
            raise PublishBackpressure()
        return cls(semaphore)

    def hand_off(self, future: Future[Any]) -> None:
        self._handed_off = True
        future.add_done_callback(lambda _future: self._semaphore.release())

    def release_unless_handed_off(self) -> None:
        if not self._handed_off:
            self._semaphore.release()


@dataclass
class _PublishResolvedDeps:
    """Store dependencies resolved in place by lifespan."""

    artifact_store: ArtifactStore | None
    snapshot_reader: SnapshotReader | None
    version_writer: VersionRecordWriter | None


@dataclass
class _PublishStoreBorrow:
    """Tracks the pool borrow and store so lifespan can release owned resources."""

    store: LiveDatasetStore | None = None
    opened_here: bool = False


@dataclass(frozen=True)
class _PublishDoorState:
    """State shared by lifespan, execution, and route handlers."""

    resolved: _PublishResolvedDeps
    settings: DatasetsBuildSettings
    budget: ShutdownBudget
    door_executor: DoorExecutor
    # The factory accepts all three dependencies injected or all three unset.
    store_deps_injected: bool
    principal_header: str
    allowed_principals: frozenset[str]
    ceiling: int


def _publish_open_store_deps(state: _PublishDoorState, borrow: _PublishStoreBorrow) -> None:
    """Build the live reader and writer when the store dependencies are unset.

    This phase has no local cleanup path.
    """
    if state.store_deps_injected:
        return
    store = LiveDatasetStore(connections=shared_pool())
    borrow.store = store
    borrow.opened_here = True
    state.resolved.snapshot_reader = LiveSnapshotReader(store=store)
    state.resolved.version_writer = LiveVersionRecordWriter(store=store)


def _publish_open_artifact_store(state: _PublishDoorState, borrow: _PublishStoreBorrow) -> None:
    """Build the live artifact store, reusing the store borrow when available."""
    if state.store_deps_injected:
        return
    live_settings = DatasetsSettings.from_env()
    store = borrow.store
    if store is None:
        store = LiveDatasetStore(connections=shared_pool())
        borrow.store = store
        borrow.opened_here = True
    state.resolved.artifact_store = LiveArtifactStore(
        buckets={
            Tier.working: live_settings.artifact_bucket_working,
            Tier.pinned: live_settings.artifact_bucket_pinned,
        },
        inventory=LiveArtifactInventory(store=store),
    )


@asynccontextmanager
async def _publish_lifespan(_app: FastAPI, *, state: _PublishDoorState) -> AsyncIterator[None]:
    """Build missing store dependencies and drain admitted calls before closing the pool."""
    borrow = _PublishStoreBorrow()
    _publish_open_store_deps(state, borrow)
    try:
        _publish_open_artifact_store(state, borrow)
    except BaseException:
        # Release the pool borrow if artifact-store setup fails.
        if borrow.opened_here:
            close_shared_pool()
        raise
    try:
        yield
    finally:
        state.door_executor.executor.shutdown(wait=False, cancel_futures=True)
        await asyncio.to_thread(state.door_executor.drain, state.budget)
        if borrow.opened_here:
            close_shared_pool()


async def _run_publish(
    state: _PublishDoorState, permit: _PublishPermit, fn, /, *, timings: PhaseTimings, **kwargs
) -> PublishResult:
    """Run a publish call on the dedicated executor under the request's permit.

    A shutdown submission failure raises `PublishBackpressure`. `BrokenExecutor`
    signals a failed worker pool. The submitted future releases the permit when
    the call ends.
    """
    try:
        future = state.door_executor.executor.submit(
            functools.partial(run_with_timings, fn, timings, **kwargs)
        )
    except BrokenExecutor:
        logger.error(
            "Datasets publish door's own executor pool is permanently broken; every "
            "future submitted to it will fail the same way",
            exc_info=True,
        )
        raise
    except RuntimeError:
        raise PublishBackpressure() from None
    permit.hand_off(future)
    return await asyncio.wrap_future(future)


def _admit_publish_request(request: Request, *, state: _PublishDoorState) -> None:
    """Check authorization from the headers alone, before the door reads the body.

    Reject missing, repeated, or unlisted principal headers. `getlist` exposes
    duplicate values after the first header.
    """
    principal_values = request.headers.getlist(state.principal_header)
    if not principal_values:
        raise PublishNotPermitted(principal=None)
    if len(principal_values) > 1:
        logger.warning(
            "Datasets publish door saw %d values for the %r principal header on one "
            "request; refusing rather than trusting either one",
            len(principal_values),
            state.principal_header,
        )
        raise PublishNotPermitted(principal=principal_values[0])
    principal = principal_values[0]
    if principal not in state.allowed_principals:
        raise PublishNotPermitted(principal=principal)


def _publish_route_class(state: _PublishDoorState) -> type[APIRoute]:
    """Return the publish route class, which admits a request before FastAPI reads its body.

    FastAPI turns any exception raised while it reads a body into a generic
    400, so the route inflates a gzip body itself, raising exceptions this door
    classifies, and hands FastAPI the decoded bytes through `_replay_body`.
    """

    class _AdmittingPublishRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()

            async def admit_then_handle(request: Request) -> Response:
                _admit_publish_request(request, state=state)
                permit = _PublishPermit.take(state.door_executor.semaphore)
                try:
                    if _publish_content_encoding(request) == "gzip":
                        body = await _inflate_gzip_body(request, limit=MAX_PUBLISH_BODY_BYTES)
                        request = Request(request.scope, _replay_body(body, request.receive))
                    request.state.publish_permit = permit
                    return await handler(request)
                finally:
                    permit.release_unless_handed_off()

            return admit_then_handle

    return _AdmittingPublishRoute


async def _post_publish_impl(
    body: PublishFrameRequest, *, permit: _PublishPermit, state: _PublishDoorState
) -> JSONResponse:
    trace_id = get_trace_id()
    with door_call("publish", logger=logger) as call:
        result = await _run_publish(
            state,
            permit,
            publish_frame,
            timings=call.timings,
            spec_id=body.spec_id,
            content_version=body.content_version,
            role_keys=body.role_keys,
            rows=body.rows,
            input_content_versions=body.input_content_versions,
            artifact_store=state.resolved.artifact_store,
            snapshot_reader=state.resolved.snapshot_reader,
            version_writer=state.resolved.version_writer,
            trace_id=trace_id,
            ceiling=state.ceiling,
        )
    response = PublishFrameResponse(
        published=PublishedFrameCoordinates(
            spec_id=result.spec_id,
            content_version=result.content_version,
            artifact_digest=result.artifact_digest,
            created=result.created,
        )
    )
    return JSONResponse(
        response.model_dump(mode="json"),
        headers={"Server-Timing": call.server_timing},
    )


def build_publish_door(
    *,
    settings: DatasetsBuildSettings | None = None,
    artifact_store: ArtifactStore | None = None,
    snapshot_reader: SnapshotReader | None = None,
    version_writer: VersionRecordWriter | None = None,
    principal_header: str | None = None,
    allowed_principals: frozenset[str] | None = None,
    ceiling: int = PUBLISH_MAX_ROWS,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the publish door with a dedicated executor and admission semaphore.

    `settings` defaults from the environment. `pool_size` defaults to
    `settings.publish_max_concurrency`. `ceiling` defaults to 250,000 rows.
    Principal settings default to `settings.principal_header` and
    `settings.first_party_principal`. Only allowlisted principals may publish, and an empty
    allowlist permits none.
    The allowlist is independent of the pin door's principal.

    Inject `artifact_store`, `snapshot_reader`, and `version_writer` together,
    or leave all three unset for lifespan to build them with one pool borrow.
    The writer and reader share a store because each publish verifies its
    version row. Pass `NullVersionRecordWriter` explicitly to disable writes.
    `budget` is shared across mounted doors. An omitted budget creates a private one.
    """
    missing = [
        name
        for name, value in (
            ("artifact_store", artifact_store),
            ("snapshot_reader", snapshot_reader),
            ("version_writer", version_writer),
        )
        if value is None
    ]
    if missing and len(missing) != 3:
        raise DatasetsConfigError(
            "datasets publish door built with "
            f"{', '.join(missing)} left uninjected while another dependency is injected. A "
            "caller injecting any one of artifact_store/snapshot_reader/version_writer must "
            "inject every one of them. This door builds a live dependency for all three only "
            "when every one of them is left None."
        )

    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    _principal_header = (
        principal_header if principal_header is not None else _settings.principal_header
    )
    _allowed_principals = (
        allowed_principals
        if allowed_principals is not None
        else frozenset({_settings.first_party_principal} - {""})
    )

    effective_pool_size = pool_size if pool_size is not None else _settings.publish_max_concurrency
    door_executor = build_door_executor(effective_pool_size, "publish")

    state = _PublishDoorState(
        resolved=_PublishResolvedDeps(
            artifact_store=artifact_store,
            snapshot_reader=snapshot_reader,
            version_writer=version_writer,
        ),
        settings=_settings,
        budget=_budget,
        door_executor=door_executor,
        store_deps_injected=snapshot_reader is not None,
        principal_header=_principal_header,
        allowed_principals=_allowed_principals,
        ceiling=ceiling,
    )

    router = APIRouter(route_class=_publish_route_class(state))

    @router.post(_PUBLISH_ROUTE, response_model=PublishFrameResponse)
    async def post_publish(body: PublishFrameRequest, request: Request) -> JSONResponse:
        """Publish a validated body under the permit the route class took for this request."""
        return await _post_publish_impl(body, permit=request.state.publish_permit, state=state)

    return DoorContribution(
        router=router,
        classify=classify_publish_exception,
        exception_types=(
            PublishFrameInvalid,
            PublishBodyTooLarge,
            PublishBodyEncodingInvalid,
            PublishContentLengthRequired,
            PublishBytesDiverge,
            PublishLineageDiverge,
            SnapshotRevoked,
            PublishRecordMissing,
            PublishNotPermitted,
            OverCeilingError,
            PublishBackpressure,
            ArtifactRecordConflictError,
            ArtifactInventoryError,
            ArtifactWriteError,
            # Surface-level classification handles exceptions version writes can raise.
            DatasetsStoreUnavailable,
        ),
        exception_exemplars=(
            PublishFrameInvalid(category="schema_violation", message="exemplar"),
            PublishBodyTooLarge(content_length=1, limit=1),
            PublishBodyEncodingInvalid("exemplar"),
            PublishContentLengthRequired(content_length_header=None),
            PublishBytesDiverge(spec_id="ops_exemplar", content_version="ocv_exemplar"),
            PublishLineageDiverge(spec_id="ops_exemplar", content_version="ocv_exemplar"),
            SnapshotRevoked(id="ops_exemplar", content_version="ocv_exemplar", reason=None),
            PublishRecordMissing(spec_id="ops_exemplar", content_version="ocv_exemplar"),
            PublishNotPermitted(principal=None),
            OverCeilingError("exemplar", ceiling_kind="rows", count=1, cap=1),
            PublishBackpressure(),
            ArtifactRecordConflictError(
                digest="sha256:exemplar",
                tier=Tier.working,
                stored_bucket="exemplar-bucket",
                stored_object_name="sha256:exemplar",
                stored_size_bytes=1,
                incoming_bucket="exemplar-bucket-2",
                incoming_object_name="sha256:exemplar",
                incoming_size_bytes=2,
            ),
            ArtifactInventoryError(digest="sha256:exemplar"),
            ArtifactWriteError(object_name="sha256:exemplar", digest="sha256:exemplar"),
            DatasetsStoreUnavailable("exemplar"),
        ),
        lifespan=functools.partial(_publish_lifespan, state=state),
        middleware=(_admit_publish_content_encoding, _admit_publish_body_size),
    )


__all__ = [
    "PublishBackpressure",
    "PublishBodyEncodingInvalid",
    "PublishBodyTooLarge",
    "PublishBytesDiverge",
    "PublishContentLengthRequired",
    "PublishFrameInvalid",
    "PublishLineageDiverge",
    "PublishNotPermitted",
    "PublishRecordMissing",
    "PublishResult",
    "build_publish_door",
    "classify_publish_exception",
    "publish_frame",
]
