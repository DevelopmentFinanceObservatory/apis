"""The publish door stores derived `ops_`/`ocv_` frames from analysis-mcp so
`LiveFrameStore` can serve them through the existing version and artifact path.

The request sends grammar-independent JSON in `role_keys` and `rows`. `apis`
and `ai` pin different `spec-kit` revisions, so this door builds the frame
with `Frame.build` and stores it twice: as the version's canonical frame
(`render_and_store_canonical_frame`), which cards, replay and every later
rendering read, and as its bare frame-layout Parquet file
(`render_and_store_frame_file`), the flat file with a text `unit` a dataset
version serves for `?format=parquet&bundle=false&layout=frame`. A format 2
ref's `artifact.digest` names that file.

The `datasets.publish` registry maps `publish_conflict` (409),
`snapshot_revoked` (410), `publish_not_permitted` (403), `request_invalid`
(400), `over_ceiling` (422), `service_unavailable` (503), and
`internal_error` (500). Byte and lineage conflicts share `publish_conflict`
but have distinct `constraint_id` values.

The version stores `input_content_versions` as the derivation trace
(`dataset_store.snapshot.FrameLineage`). Derived frames use their own
reference as the sole provenance input. Beside it the version stores the
continuations of its series, derived before the write from the stored facts
of the versions its `retrieve` steps name, or without steps the versions its
`input_content_versions` name (`lineage_continuations.derived_continuations`),
so a series its roots' estimates continue is one series from the first read.
A failed read of those versions fails the publish as a store outage. A
repeat publish that fills the steps of a version published without them
replaces its stored continuations with those its retrieved versions give, in
the write that fills the steps, and its record's series facts with those
counted under them; any other repeat publish leaves them as they are.

A repeat publish succeeds with `created=False` only when the stored frame and
`input_content_versions` both match. Frames match by `frame_sha256`, the hash of
the frame's canonical frame1 bytes, which the version records with its first
publish (`StoredVersion.frame_sha256`), so the same rows match whatever PyArrow
version encoded either Parquet file, and the comparison reads no stored object.
A pair recorded without the hash is compared by the digest of its frame1
document, the same hash. A pair holding neither refuses as a different frame.
A matching repeat links this call's canonical frame and frame file to the pair where the one
linked there has no servable location (`LiveVersionRecordWriter.record_artifact`), so republishing
a frame past working-tier retention makes it servable again, and a pair published before it stored
a frame file gains one. The response gives the digest of the frame file the pair links after that
call, which is another encoding's while the earlier file stays servable.
A different frame or lineage, or a pair recording no lineage, raises a
non-retryable `publish_conflict` (409). A lineage conflict matters because `ocv_`
identifies a computation recipe. Reporting success would discard the second
call's lineage. The op-spec identity is the row's `subject_id`, which the lookup
has already matched, so lineage is `input_content_versions` alone.
`publish_frame` checks existing records before writing and rereads after writing
to detect a concurrent winner and compare its frame and lineage.

A divergent digest under `ocv_` means the same recipe may have produced
different bytes. The door logs this reproducibility failure at error level,
while returning the client-facing 409 `publish_conflict`.

Revocation is read through `snapshot_reader` before digest comparison and
after recording. The first check makes `snapshot_revoked` (410) outrank byte or
lineage conflicts. The second runs even when this call inserted the version,
catching a concurrent publish followed by revocation. The foreign key requires
an existing version row. The first check runs for recorded pairs only.
`SnapshotRevoked` is non-retryable. Revocation rows are permanent.

Publishing requires the `publish` grant (`datasets_api.principals`), which
`DATASETS_FIRST_PARTY_PRINCIPAL` holds. No other secret holds it, and none of the
pin or live-refresh secrets opens this route.

`publish_frame` rejects more than 250,000 rows with `over_ceiling` (422), before
`Frame.build`. Coverage cannot estimate derived-frame size because analysis-mcp
computes the rows. Frame grammar rejects empty strings and lone surrogates.
A frame in which a unit ref spells the key text of a converted or composed unit
(`materialise.unit_cells.shared_unit_key`) is refused `request_invalid` at `/rows`
and not stored, since Presentation refuses to open one.

`steps` are the frame's step records (`publish_models`), which the door checks for shape,
structure and the kit's bounds (`MAX_STEPS`, `MAX_STEPS_BYTES`) and refuses as `request_invalid`
(400) with a pointer into `steps` before anything is written. They go to the version's
description record (`datasets.description_record`), which the version row's transaction writes
once. A republish of a pair whose record holds `steps: null` fills them once. A republish whose
steps state other facts than the recorded ones (inputs, operations, policies, parameters, units
or counts, leaving labels aside) refuses as a lineage divergence, `publish_conflict` (409) with
pointer `/steps`, and a republish without steps changes nothing.
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
The `publish` grant check reads the principal from the headers, so an
unauthorized caller gets `publish_not_permitted` (403) without holding a permit
or having its body read, whatever the body holds. The permit comes from the
work executor's semaphore (`DATASETS_WORK_MAX_CONCURRENCY`), and stays held until the
build ends, so it bounds body decoding and JSON parsing as well as the build.
When no permit is free the door refuses with `service_unavailable` (503).

The route decodes a gzip body before FastAPI reads it, in steps of at most
`_INFLATE_STEP_BYTES`, and refuses it with `PublishBodyTooLarge` as soon as the
decoded size passes `MAX_PUBLISH_BODY_BYTES`. Corrupt or truncated gzip data,
or bytes after the gzip member, refuse with `PublishBodyEncodingInvalid`. Both
map to `request_invalid` (400) before JSON decoding. A client disconnect during
the read returns the same 400 as one during an identity read.

A publish may carry the frame's `recipe`. `publish_frame` hands it to the
recipe store once the version row is verified, on a first publish and on a
republish alike, and never stores it on the version. A republish may
therefore add a recipe to a frame published without one, and a republish
without a recipe leaves the stored one in place. The store keeps a pair's
first recipe: a later one with a different `normal_form_hash` or `replay_key`,
or a `replay_key` another frame holds, refuses with `publish_conflict` (409)
under its own `constraint_id` (`datasets_api.recipe_store`). The frame is
recorded by then, so the caller may republish without the recipe. A
`publish_frame` call given no recipe store refuses a publish carrying a recipe
with `internal_error` (500) before it writes anything.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import zlib
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import BrokenExecutor, Future
from dataclasses import dataclass, replace
from typing import Any

from dfo_frame_grammar import (
    UNIT_COLUMN,
    Frame,
    FrameError,
    frame_sha256,
    unit_cell_to_json,
    unit_cells,
    unit_codes,
)
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive

from artifact_store import (
    ArtifactInventoryError,
    ArtifactRecordConflictError,
    ArtifactStore,
    ArtifactWriteError,
    Tier,
)
from dataset_store.snapshot import (
    FrameLineage,
    SnapshotReader,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
)
from datasets_api.backpressure import Backpressure
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.description_record import computed_inputs
from datasets_api.door_executor import DoorExecutor
from datasets_api.lineage_continuations import derived_continuations, lineage_inputs
from datasets_api.principals import Grant, Principal, Principals, PublishNotPermitted
from datasets_api.publish_models import (
    MAX_ECHO_CHARS,
    MAX_RECIPE_BYTES,
    MAX_STEPS_BYTES,
    PublishedFrameCoordinates,
    PublishFrameRequest,
    PublishFrameResponse,
    Recipe,
)
from datasets_api.recipe_store import (
    RecipeDiverges,
    RecipeRecord,
    RecipeStore,
    ReplayKeyHeld,
)
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.render import render_and_store_canonical_frame, render_and_store_frame_file
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import PhaseTimings, door_call, phase, run_with_timings
from datasets_api.version_descriptions import publish_describer
from datasets_api.version_store import recorded_frame_sha256
from datasets_api.versions import DescriptionDiverges, VersionRecord, VersionRecordWriter
from description.steps import Step
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from dfo_platform.trace import get_trace_id
from materialise import EnrichmentLookup
from materialise.errors import OverCeilingError
from materialise.frame_build import build_frame
from materialise.unit_cells import shared_unit_key

logger = logging.getLogger(__name__)

# Fixed row ceiling for publish admission. Coverage cannot estimate the rows analysis-mcp
# computes, and the door builds the whole frame in memory before storing it.
PUBLISH_MAX_ROWS = 250_000

# Bound on the bytes one publish holds and JSON-decodes: the declared
# Content-Length of any body, and the decoded size of a gzip body. The door's
# memory grows with these bytes, so the bound comes from measured rows rather
# than from schema limits, which allow rows over 40 KiB. The 250,000-row
# benchmark body, with two role keys and synthetic keys, is 54,470,651 bytes of
# JSON, 218 bytes a row. Real DCIDs and StatVar names are longer, and 512 bytes
# a row is about 2.3 times the measured row. The 1,000,000 bytes cover metadata
# and JSON punctuation, and are the recipe's own allowance (`MAX_RECIPE_BYTES`). Step
# records add `MAX_STEPS_BYTES` (65,536) of their own. At 250,000 rows the bound is
# 129,065,536 bytes, about 123 MiB. The door adds about 12 times the body size in RSS (645 MiB for the
# 54.5 MB body), so a body at the bound adds about 1.5 GiB. Cloud Run caps an
# HTTP/1 request at 32 MiB on the wire, so there only a gzip body can reach the
# bound.
_ADMISSION_ROW_BYTES = 512
MAX_PUBLISH_BODY_BYTES = (
    PUBLISH_MAX_ROWS * _ADMISSION_ROW_BYTES + MAX_RECIPE_BYTES + MAX_STEPS_BYTES
)

# Most decoded bytes one `decompress` call returns, so inflating a gzip body
# allocates at most this much past `MAX_PUBLISH_BODY_BYTES` before refusing.
_INFLATE_STEP_BYTES = 1 << 20

_PUBLISH_CONTENT_ENCODINGS = frozenset({"identity", "gzip"})


class PublishFrameInvalid(Refusal):
    """Frame grammar rejected `role_keys` or `rows`, or the built frame is one no reader can
    open. Maps to non-retryable `request_invalid` (400), like schema validation failures.
    """

    code = "request_invalid"

    def __init__(self, *, category: str, message: str, pointer: str | None = None) -> None:
        self.category = category
        self.pointer = pointer
        super().__init__(message)


class PublishBodyTooLarge(Refusal):
    """The body exceeds `MAX_PUBLISH_BODY_BYTES`, either by its declared
    `Content-Length` or, with `inflated`, by its gzip-decoded size. Maps to
    non-retryable `request_invalid` (400) before JSON decoding.

    Middleware constructs and classifies the declared-length case directly. A
    raise in middleware would be re-raised by `ServerErrorMiddleware`. The
    decoded-size case raises from the publish route class.
    """

    code = "request_invalid"

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


class PublishBodyEncodingInvalid(Refusal):
    """The body's `Content-Encoding` is neither `identity` nor `gzip`, or its
    gzip data is corrupt, truncated, or followed by trailing bytes. Maps to
    non-retryable `request_invalid` (400) before JSON decoding.
    """

    code = "request_invalid"


class PublishContentLengthRequired(Refusal):
    """`Content-Length` is absent or unparseable, so middleware refuses to
    read an unbounded body. It maps to non-retryable `request_invalid` (400).

    Middleware classifies the exception directly. A live raise triggers
    `ServerErrorMiddleware`'s re-raise.
    """

    code = "request_invalid"

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


class PublishBytesDiverge(Refusal):
    """This pair already holds different frame bytes. Maps to non-retryable
    `publish_conflict` (409). The caller must use the recorded bytes or a new id.
    """

    code = "publish_conflict"
    constraint_id = "publish_requires_matching_frame1_bytes"

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} already holds a "
            "stored frame whose content differs from this publish call's frame; refusing "
            "to report success for a digest that was not actually recorded under this "
            "pair. This most likely means the analysis engine that minted this "
            "content_version did not reproduce the same frame on a second run -- publish "
            "the frame already held under this id, or mint a new content_version for the "
            "corrected recipe; retrying with this same frame will refuse identically"
        )


class PublishLineageDiverge(Refusal):
    """This pair holds identical frame bytes but a different
    `input_content_versions` trace, or step records whose facts differ from the posted
    `steps` (`steps_message`, with `pointer` `/steps`). Because `ocv_` identifies a recipe,
    reporting success would discard the new lineage. Maps to non-retryable
    `publish_conflict` (409), distinguished from byte conflicts by `constraint_id`.
    """

    code = "publish_conflict"
    constraint_id = "publish_requires_matching_input_content_versions"

    def __init__(
        self,
        *,
        spec_id: str,
        content_version: str,
        steps_message: str | None = None,
        pointer: str | None = None,
    ) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        self.pointer = pointer
        if steps_message is not None:
            super().__init__(
                f"{steps_message}; refusing to report success for step records that were not "
                "recorded under this pair. Publish the steps already held under this id, or "
                "mint a new content_version for the other derivation"
            )
            return
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} already holds a "
            "stored frame with this publish call's content but a different input_content_versions "
            "derivation trace than this publish call's own; refusing to report success "
            "for a derivation trace that was not actually recorded under this pair. "
            "This most likely means this content_version was minted twice for genuinely "
            "different inputs -- mint a new content_version for the corrected derivation, "
            "or publish the input_content_versions already held under this id; retrying "
            "with these same claimed inputs will refuse identically"
        )


class PublishRecordMissing(Refusal):
    """The post-write read returned `None` for the version row. The frame
    artifact is stored. This signals a writer/reader mismatch or failed insert.
    Maps to non-retryable `internal_error` (500). Retries with the same wiring
    fail identically.
    """

    code = "internal_error"

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} rendered and wrote "
            "its canonical frame to the artifact store, but no version row is readable back "
            "through this door's own snapshot_reader immediately after recording one; this "
            "publish door was composed with a version_writer that does not share "
            "snapshot_reader's own store, or that writer's insert silently failed to land -- "
            "retrying will refuse identically under the same wiring"
        )


class PublishRecipeStoreMissing(Refusal):
    """A publish carries a recipe and this door has no recipe store to hold it. Maps to
    non-retryable `internal_error` (500). The door refuses before any write, so no frame is
    published with its recipe dropped.
    """

    code = "internal_error"

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        self.spec_id = spec_id
        self.content_version = content_version
        super().__init__(
            f"spec_id {spec_id!r} content_version {content_version!r} was published with a "
            "recipe, but this publish door was composed without a recipe_store; retrying "
            "will refuse identically under the same wiring"
        )


class PublishBackpressure(Backpressure):
    """Publish capacity is full. Maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the publish door is at its concurrent-write capacity right now; retry"


@dataclass(frozen=True)
class PublishResult:
    spec_id: str
    content_version: str
    artifact_digest: str
    created: bool


def frame_lineage(
    *, input_content_versions: Sequence[str], continuations: Sequence[Mapping[str, object]] = ()
) -> FrameLineage:
    """The facts a derived frame stores: its lineage and the continuations of its series."""
    return FrameLineage(
        input_content_versions=list(input_content_versions), continuations=list(continuations)
    )


def _stored_input_content_versions(stored: StoredVersion) -> Sequence[str] | None:
    """The lineage a stored version records, `None` for a dataset version."""
    if isinstance(stored.facts, FrameLineage):
        return stored.facts.input_content_versions
    return None


def write_frame_recipe(
    *, recipe_store: RecipeStore, spec_id: str, content_version: str, recipe: Recipe
) -> bool:
    """Store `recipe` for the frame and index its replay key. Return `True` when this call
    creates the row, `False` when the frame already holds a matching recipe.
    """
    return recipe_store.write_recipe(
        RecipeRecord(
            spec_id=spec_id,
            content_version=content_version,
            normal_form_hash=recipe.normal_form_hash,
            replay_key=recipe.replay_key,
            body=recipe.stored_body(),
        )
    )


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
    recipe: Recipe | None = None,
    recipe_store: RecipeStore | None = None,
    steps: Sequence[Step] | None = None,
    enrichment: EnrichmentLookup | None = None,
) -> PublishResult:
    """Publish one frame synchronously and verify the stored version.

    Reject rows over `ceiling` before frame construction. Store the canonical frame first,
    then check for an existing version. Unreferenced bytes remain in the
    working tier if recording fails. Revocation takes precedence. A matching
    frame and lineage link this call's canonical frame and frame file where the pair's have no
    servable location and return `created=False`, while divergence raises the
    matching conflict. Otherwise record the version, then reread it to verify
    the winning frame, lineage, and revocation state. `created` comes from the
    writer's insert result. Missing read-back raises `PublishRecordMissing`.
    `artifact_digest` is the digest the stored pair links as its frame file
    (`render.FRAME_FILE_OPTIONS`), the file a format 2 ref names, which is another call's
    when that call's encoding of the file is linked.

    A `recipe` goes to `recipe_store` after the version is verified, whether
    this call or an earlier one recorded it. It takes no part in the body
    comparison. A `recipe` with no `recipe_store` raises
    `PublishRecipeStoreMissing` before any write.

    `steps` go to the version's description record, written in the version row's transaction
    (`version_descriptions.publish_describer`), `None` when the publish carried none. A pair
    already recorded gets its record when it holds none, and a record holding `steps: null`
    takes the posted steps once. Posted steps whose facts differ from the recorded ones raise
    `PublishLineageDiverge` at `/steps` before anything is linked, and a publish without steps
    leaves recorded steps as they are. `enrichment` reads the labels the record names; `None`
    reads none.
    """
    if recipe is not None and recipe_store is None:
        raise PublishRecipeStoreMissing(spec_id=spec_id, content_version=content_version)

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
            # Use the materialise frame builder for shared frame construction. A publish
            # builds its frame on a work thread outside the materialise permits
            # (`admission.MaterialisePermits`). The work executor's permit alone bounds how
            # many publishes hold a frame, and a release swap does not wait for them.
            frame = build_frame(role_keys=role_keys, rows=rows)
    except FrameError as exc:
        raise PublishFrameInvalid(category=exc.category, message=str(exc)) from exc
    return publish_built_frame(
        frame=frame,
        spec_id=spec_id,
        content_version=content_version,
        input_content_versions=input_content_versions,
        artifact_store=artifact_store,
        snapshot_reader=snapshot_reader,
        version_writer=version_writer,
        trace_id=trace_id,
        ceiling=ceiling,
        recipe=recipe,
        recipe_store=recipe_store,
        steps=steps,
        enrichment=enrichment,
    )


def publish_built_frame(
    *,
    frame: Frame,
    spec_id: str,
    content_version: str,
    input_content_versions: Sequence[str],
    artifact_store: ArtifactStore,
    snapshot_reader: SnapshotReader,
    version_writer: VersionRecordWriter,
    trace_id: str | None = None,
    ceiling: int = PUBLISH_MAX_ROWS,
    recipe: Recipe | None = None,
    recipe_store: RecipeStore | None = None,
    steps: Sequence[Step] | None = None,
    enrichment: EnrichmentLookup | None = None,
) -> PublishResult:
    """`publish_frame` for a frame already built, as the analysis runner holds one
    (`datasets_api.analysis`). Every check after the build is `publish_frame`'s: the row
    ceiling, the shared unit key, the recipe store, then the store, record and verify steps.
    """
    if recipe is not None and recipe_store is None:
        raise PublishRecipeStoreMissing(spec_id=spec_id, content_version=content_version)

    if frame.num_rows > ceiling:
        raise OverCeilingError(
            f"published frame has {frame.num_rows} rows, exceeding the sync materialisation "
            f"ceiling of {ceiling} rows",
            detail=(
                f"published frame has {frame.num_rows} rows, exceeding the sync materialisation "
                f"ceiling of {ceiling} rows"
            ),
            ceiling_kind="rows",
            count=frame.num_rows,
            cap=ceiling,
        )

    unit_column = frame.table.column(UNIT_COLUMN)
    shared = shared_unit_key(
        cells=[unit_cell_to_json(cell) for cell in unit_cells(unit_column, unit_codes(unit_column))]
    )
    if shared is not None:
        # Presentation groups rows by each unit's key text and refuses a frame that shares one.
        raise PublishFrameInvalid(
            category="invalid_unit",
            message=(
                f"the unit ref {shared[:MAX_ECHO_CHARS]!r} reads as the same text as a converted or composed "
                "unit in this frame; units must read apart"
            ),
            pointer="/rows",
        )

    canonical = render_and_store_canonical_frame(
        frame_chunks=(frame,),
        subject_id=spec_id,
        content_version=content_version,
        artifact_store=artifact_store,
    )
    frame_file = render_and_store_frame_file(
        frame_chunks=lambda: iter((frame,)),
        subject_id=spec_id,
        content_version=content_version,
        artifact_store=artifact_store,
    )

    published_frame_sha256 = frame_sha256(frame)
    with phase("lineage"):
        continuations = derived_continuations(
            lineage_inputs(
                steps_inputs=None if steps is None else computed_inputs(steps),
                lineage=FrameLineage(input_content_versions=input_content_versions),
            ),
            reader=snapshot_reader,
        )
    requested = frame_lineage(
        input_content_versions=input_content_versions, continuations=continuations
    )
    version = VersionRecord(
        subject_id=spec_id,
        subject_kind="spec",
        content_version=content_version,
        schema_version=1,
        facts=requested,
        artifacts=(canonical, frame_file),
        frame_sha256=published_frame_sha256,
    )
    described = replace(
        version,
        describer=publish_describer(version, steps=steps, frame=frame, enrichment=enrichment),
    )

    def _record_described(record: VersionRecord) -> bool:
        """Record `record` with its description, refusing steps the stored record
        contradicts."""
        try:
            return version_writer.record_version(record, trace_id=trace_id)
        except DescriptionDiverges as exc:
            raise PublishLineageDiverge(
                spec_id=spec_id,
                content_version=content_version,
                steps_message=str(exc),
                pointer=exc.pointer,
            ) from exc

    def _recorded_frame_matches(
        stored: StoredVersion, locations: Sequence[StoredArtifactLocation] | None = None
    ) -> bool:
        """Whether `stored` holds this call's frame: by the frame hash it records, else by the
        digest of the pair's frame1 document, the same hash. `locations` are the pair's when
        the caller holds them, else they are read. No stored object is read."""
        if stored.frame_sha256 is None and locations is None:
            locations = snapshot_reader.read_artifact_locations(spec_id, content_version)
        recorded = recorded_frame_sha256(stored.frame_sha256, locations or ())
        return recorded == published_frame_sha256

    def _raise_if_diverges(stored: StoredVersion) -> None:
        """Compare stored lineage after the caller confirms the frame bytes match."""
        stored_lineage = _stored_input_content_versions(stored)
        if stored_lineage is None or list(stored_lineage) != list(requested.input_content_versions):
            raise PublishLineageDiverge(spec_id=spec_id, content_version=content_version)

    def _store_recipe() -> None:
        if recipe is None or recipe_store is None:
            return
        with phase("recipe"):
            write_frame_recipe(
                recipe_store=recipe_store,
                spec_id=spec_id,
                content_version=content_version,
                recipe=recipe,
            )

    with phase("check"):
        existing_version = snapshot_reader.read_version(spec_id, content_version)
        if existing_version is not None:
            revocation = snapshot_reader.read_revocation(spec_id, content_version)
            if revocation is not None:
                raise SnapshotRevoked(
                    id=spec_id, content_version=content_version, reason=revocation.reason
                )
            if not _recorded_frame_matches(existing_version):
                raise PublishBytesDiverge(spec_id=spec_id, content_version=content_version)
            _raise_if_diverges(existing_version)
            # The version row stands; this writes or settles its description record only.
            with phase("record"):
                _record_described(replace(described, artifacts=()))
            # Each link is kept while its file has a servable location.
            with phase("record"):
                linked = [
                    version_writer.record_artifact(
                        subject_id=spec_id,
                        content_version=content_version,
                        artifact=artifact,
                        trace_id=trace_id,
                    )
                    for artifact in (canonical, frame_file)
                ]
            linked_digest = linked[-1]
            _store_recipe()
            return PublishResult(
                spec_id=spec_id,
                content_version=content_version,
                artifact_digest=linked_digest,
                created=False,
            )

    with phase("record"):
        created = _record_described(described)

    with phase("verify"):
        recorded = snapshot_reader.read_version_state(spec_id, content_version)
    if recorded is None:
        raise PublishRecordMissing(spec_id=spec_id, content_version=content_version)
    if recorded.revocation is not None:
        raise SnapshotRevoked(
            id=spec_id, content_version=content_version, reason=recorded.revocation.reason
        )
    if not _recorded_frame_matches(recorded.version, recorded.locations):
        raise PublishBytesDiverge(spec_id=spec_id, content_version=content_version)
    _raise_if_diverges(recorded.version)
    # A concurrent first publish can link another encoding of the file, which the link keeps.
    linked_digest = next(
        (
            location.artifact_digest
            for location in recorded.locations
            if location.rendering == frame_file.rendering
        ),
        None,
    )
    if linked_digest is None:
        raise PublishRecordMissing(spec_id=spec_id, content_version=content_version)
    _store_recipe()

    return PublishResult(
        spec_id=spec_id,
        content_version=content_version,
        artifact_digest=linked_digest,
        created=created,
    )


def classify_publish_exception(exc: BaseException) -> ProblemSpec | None:
    return classify_datasets_refusal(exc)


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
    """One publish's hold on the work executor's semaphore, taken before its body is read.

    `_run_publish` hands the permit to the executor future, which releases it
    when the build ends, even if the request is cancelled first. A permit the
    request never handed off releases when the request ends.
    """

    def __init__(self, executor: DoorExecutor) -> None:
        self.executor = executor
        self._handed_off = False

    @classmethod
    def take(cls, executor: DoorExecutor) -> _PublishPermit:
        """Acquire a permit without waiting, refusing with `PublishBackpressure` when none is free."""
        if not executor.semaphore.acquire(blocking=False):
            raise PublishBackpressure()
        return cls(executor)

    def hand_off(self, future: Future[Any]) -> None:
        self._handed_off = True
        future.add_done_callback(lambda _future: self.executor.semaphore.release())

    def release_unless_handed_off(self) -> None:
        if not self._handed_off:
            self.executor.semaphore.release()


@dataclass(frozen=True)
class _PublishDoorState:
    """State shared by the route class and handlers."""

    deps: DepsCell
    require_publish: Callable[[Request], Principal]
    ceiling: int


async def _run_publish(
    permit: _PublishPermit, fn, /, *, timings: PhaseTimings, **kwargs
) -> PublishResult:
    """Run a publish call on the work executor under the request's permit and context.

    A shutdown submission failure raises `PublishBackpressure`. `BrokenExecutor`
    signals a failed worker pool. The submitted future releases the permit when
    the call ends.
    """
    try:
        future = permit.executor.executor.submit(
            functools.partial(run_with_timings, fn, timings, **kwargs)
        )
    except BrokenExecutor:
        logger.error(
            "Datasets work executor pool is permanently broken; every future submitted to it "
            "will fail the same way",
            exc_info=True,
        )
        raise
    except RuntimeError:
        raise PublishBackpressure() from None
    permit.hand_off(future)
    return await asyncio.wrap_future(future)


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
                state.require_publish(request)
                permit = _PublishPermit.take(state.deps.get().work_executor)
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
    deps = state.deps.get()
    with door_call("publish", logger=logger) as call:
        result = await _run_publish(
            permit,
            publish_frame,
            timings=call.timings,
            spec_id=body.spec_id,
            content_version=body.content_version,
            role_keys=body.role_keys,
            rows=body.rows,
            input_content_versions=body.input_content_versions,
            artifact_store=deps.artifact_store,
            snapshot_reader=deps.snapshot_reader,
            version_writer=deps.version_writer,
            trace_id=trace_id,
            ceiling=state.ceiling,
            recipe=body.recipe,
            recipe_store=deps.recipe_store,
            steps=body.steps,
            enrichment=deps.enrichment,
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
    deps: DepsCell,
    settings: DatasetsBuildSettings,
    principal_header: str | None = None,
    allowed_principals: frozenset[str] | None = None,
    ceiling: int = PUBLISH_MAX_ROWS,
) -> DoorContribution:
    """Build the publish door over the surface's dependency cell.

    `ceiling` defaults to 250,000 rows. The route admits the `publish` grant alone, held by
    `allowed_principals`, which defaults to `settings.first_party_principal`. An empty set
    admits nobody. Publishes run on the work executor, and the writer and reader share the
    surface's store, so each publish verifies its version row.
    """
    state = _PublishDoorState(
        deps=deps,
        require_publish=Principals.resolve(
            settings=settings, principal_header=principal_header, publish=allowed_principals
        ).require(Grant.publish),
        ceiling=ceiling,
    )

    router = APIRouter(route_class=_publish_route_class(state))

    @router.post(_PUBLISH_ROUTE, response_model=PublishFrameResponse)
    async def post_publish(body: PublishFrameRequest, request: Request) -> JSONResponse:
        """Publish a validated body under the permit the route class took for this request."""
        return await _post_publish_impl(body, permit=request.state.publish_permit, state=state)

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.publish"),
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
            PublishRecipeStoreMissing,
            PublishNotPermitted,
            OverCeilingError,
            PublishBackpressure,
            ArtifactRecordConflictError,
            ArtifactInventoryError,
            ArtifactWriteError,
            RecipeDiverges,
            ReplayKeyHeld,
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
            PublishRecipeStoreMissing(spec_id="ops_exemplar", content_version="ocv_exemplar"),
            PublishNotPermitted(),
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
            RecipeDiverges(
                spec_id="ops_exemplar", content_version="ocv_exemplar", field="normal_form_hash"
            ),
            ReplayKeyHeld(
                replay_key="rpk_exemplar",
                spec_id="ops_exemplar",
                content_version="ocv_exemplar",
                holder_spec_id="ops_holder",
                holder_content_version="ocv_holder",
            ),
            DatasetsStoreUnavailable("exemplar"),
        ),
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
    "PublishRecipeStoreMissing",
    "PublishRecordMissing",
    "PublishResult",
    "build_publish_door",
    "classify_publish_exception",
    "publish_built_frame",
    "publish_frame",
    "write_frame_recipe",
]
