"""Exports: a frame prepared once as a file a user keeps, behind a link that only signs.

`prepare_export` turns one exact frame version, a dataset's (`spec_`) or a derived frame's
(`ops_`), into a CSV, xlsx or Parquet file, alone or in a bundle zip, and records it under
`export_id`, a hash of the request alone. It resolves the version the request names and no
other. A request naming a dataset id alone (`BareExportFrame`) is first resolved to the version
the analysis runner reads for a Spec root of that id (`resolve_export_frame`), the newest in its
default enrichment, as the Presentation frames door reads a bare id. The request then runs, and
is keyed and answered, as one naming that pair. A dataset version's file is the one its file
selector serves (`doors.files`) in the long layout with the version's default enrichment columns
(`default_enrichment`, from the roles the version holds), the columns a Spec root of
it reads: each typed role's `_type` cells beside the layout's own `_label` and `_kind` cells.
It is the file the version lists under that rendering, else one rendered from its stored frame.
A derived frame's file is the one its file route renders (`derived.render_derived_file`). Its
render, its count (`file_shape`) and its store run inside one materialise slot, so its bytes stay
inside a slot, as a dataset file's do. The file is then copied to the pinned tier through
`retention_admission.copy_to_pinned`, which spends the caller's `pin` allowance and the day's
pin budgets as a pin does. A file with a pinned copy needs no copy and spends nothing, whichever
version or export pinned it. A caller whose `pin` allowance is spent is refused before any render
as the copy would refuse it (`retention_admission.refuse_spent_caller`), by reads that spend
nothing, so a listed dataset file with a pinned copy still serves it. A file whose digest only its
render gives, a derived frame's or a dataset file its version does not list, is refused to that
caller even when the same bytes happen to be pinned. An `artifact_reference` row of kind `export`
(`reference_store.EXPORT_REFERENCE_KIND`) then keeps the copy from any deletion keyed on references, under the transactional revocation
recheck a pin's reference takes. The record keeps the pinned copy's location and the facts the
prepare answered.

A request whose id holds a record answers that record, rendering and spending nothing.
Concurrent prepares of one id in one process share one prepare (`admission.RebuildFlights`). The
first runs, and each other waits for it, then answers the record as a repeat prepare does. After
a prepare that failed, each waiter runs its own. Two instances can prepare one id together. Every
render is deterministic (bundle members carry fixed timestamps), so they give one file, which the
content-addressed store holds once: the second copy finds the pinned bytes stored and gives its
units back, and the second record write returns the first's row.

The first prepare, a repeat prepare and the export link (`sign_export`) answer a record through
one guard (`_servable_facts`), so they cannot disagree. A version's revocation, or its base
version's, under the id it is recorded under, the id the request named or any served id bound
to that id refuses `snapshot_revoked`, a quarantine of the named id or a served id bound to it
refuses `identity_quarantined`, and the recorded version and file stay the ones served. A
derived frame holds its roots' data, so the same guard covers every root of its recipe. The link
then signs a URL to the pinned copy. It never renders. The pinned tier never expires its
objects, so a link keeps working until its version is taken down.

The record keys the version by the id it is recorded under, the served id for a dataset the
request named by a posted id (`doors.snapshot.read_under_served_id`), as a pin's reference does.
The facts name the frame as the request did.

Refusals:

- A CSV or xlsx export of more than `EXPORT_MAX_ROWS` rows refuses `over_ceiling`
  (`ExportOverCeiling`). Parquet has no row ceiling.
- A derived frame whose sources cannot all be named, because it has no stored recipe, its
  recipe has no roots or a root cannot be cited, refuses `provenance_unavailable`.
- A render linked incomplete (`render.incomplete_rendering`) refuses retryable
  `service_unavailable` (`ExportFileIncomplete`) and pins and records nothing.
- A source on the licence deny-list, `licence_restrictions.json` beside this module, keyed by
  source ref, refuses `licence_not_redistributable` at prepare and at the link. Every source is
  checked against the list as read when the process started, and the link checks the sources
  the record names, so a source added to the list withdraws its exports' links.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, cast

import openpyxl
import pyarrow.parquet as pq
import rfc8785
from pydantic import BaseModel, ConfigDict, Field

from artifact_store import ArtifactNotFoundError, StoredArtifact, Tier
from dataset_store.snapshot import ServableVersion, SnapshotRevoked, read_servable_version
from datasets_api.admission import bounded_materialise_slot
from datasets_api.allowance import AllowanceExhausted
from datasets_api.analysis import AnalysisPending
from datasets_api.analysis_roots import newest_spec_version
from datasets_api.codes_exports import (
    ExportFileIncomplete,
    ExportNotFound,
    ExportOverCeiling,
    LicenceNotRedistributable,
    NewestVersionUnservable,
    ProvenanceUnavailable,
)
from datasets_api.codes_snapshot import ArtifactExpired, SpecNotFound
from datasets_api.current import DefinitionDeps
from datasets_api.default_enrichment import resolve_enrichment
from datasets_api.deps import Deps
from datasets_api.derived import full_recipe_citation, render_derived_file, stored_recipe_roots
from datasets_api.doors.derived import derived_reads
from datasets_api.doors.files import (
    artifact_file_name,
    listed_file_location,
    read_version,
    render_file_location,
    signed_download_url,
)
from datasets_api.doors.pin import record_pin_reference
from datasets_api.doors.publish import PUBLISH_MAX_ROWS
from datasets_api.export_store import ExportRecord
from datasets_api.manifest import stored_manifest_facts
from datasets_api.manifest_models import CitationModel, CoverageModel
from datasets_api.mint_models import MintOptions
from datasets_api.recipe_store import RecipeRecord
from datasets_api.reference_store import EXPORT_REFERENCE_KIND
from datasets_api.render import is_incomplete_rendering
from datasets_api.retention_admission import (
    OpenPinBudgetExhausted,
    PinBudgetExhausted,
    copy_to_pinned,
    refuse_spent_caller,
)
from datasets_api.subject_ids import served_subject_ids
from datasets_api.timings import phase
from dfo_platform.caller import Caller
from dfo_platform.codes import bound_registry_endpoint, registered_status
from dfo_platform.problem import Refusal
from dfo_platform.trace import get_trace_id
from materialise.formats import DATA_SHEET_TITLE

EXPORT_MAX_ROWS = PUBLISH_MAX_ROWS
"""The most rows a CSV or xlsx export holds: the frame ceiling a publish admits, 250,000."""

EXPORT_LINK_PREFIX = "/api/datasets/exports"
"""The path of the export link route, which the site's proxy serves on its own origin."""

_LICENCE_RESTRICTIONS = Path(__file__).with_name("licence_restrictions.json")

type ExportFormat = Literal["csv", "xlsx", "parquet"]


class ExportFrame(BaseModel):
    """One exact frame version: a dataset's `spec_` id or a derived frame's `ops_` id, and its
    content version."""

    model_config = ConfigDict(extra="forbid")
    spec_id: str = Field(pattern=r"^(spec|ops)_[^/]+$")
    content_version: str = Field(min_length=1)


class BareExportFrame(BaseModel):
    """A dataset named by its id alone, which stands for its newest version
    (`resolve_export_frame`)."""

    model_config = ConfigDict(extra="forbid")
    spec_id: str = Field(pattern=r"^spec_[0-9a-f]{32}$")


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: ExportFrame
    format: ExportFormat = Field(default="csv", description="the file format")
    bundle: bool = Field(
        default=False,
        description="true for a zip bundle of the file beside its README, datapackage.json and "
        "CITATION.cff",
    )


class ExportRequestBody(BaseModel):
    """`POST /api/datasets/exports`'s body: an `ExportRequest` whose frame may name a dataset id
    alone."""

    model_config = ConfigDict(extra="forbid")
    frame: ExportFrame | BareExportFrame = Field(
        description="the frame version to export, or `{spec_id}` alone for a dataset's newest "
        "version, which the answer's `frame` names"
    )
    format: ExportFormat = Field(default="csv", description="the file format")
    bundle: bool = Field(
        default=False,
        description="true for a zip bundle of the file beside its README, datapackage.json and "
        "CITATION.cff",
    )

    def resolved(self, frame: ExportFrame) -> ExportRequest:
        """The request of this body's format and bundle for `frame`."""
        return ExportRequest(frame=frame, format=self.format, bundle=self.bundle)


class ExportLicence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(description="the source's ref, the key of the licence deny-list")
    label: str
    licence: str | None = Field(description="the licence's label; null where none is declared")
    url: str | None = Field(description="the licence's terms page; null where none is declared")


class _StoredFrame(BaseModel):
    spec_id: str
    content_version: str


class _StoredLicence(BaseModel):
    source: str


class _StoredFacts(BaseModel):
    """What the guard and the link read of a stored `ExportResponse` body. Any other field is
    ignored and may be absent, so a later change to the reply leaves lasting links working."""

    frame: _StoredFrame
    licences: list[_StoredLicence]
    media_type: str
    file_name: str


class ExportResponse(BaseModel):
    """The facts of a prepared export. A repeat prepare answers the same facts."""

    model_config = ConfigDict(extra="forbid")
    export_id: str = Field(pattern=r"^exp_[0-9a-f]{32}$")
    url_path: str = Field(
        description="the export link's path, relative to the public site's origin. It needs no "
        "key and does not expire: each request answers 302 to a URL signed for five minutes"
    )
    file_name: str
    media_type: str
    format: ExportFormat
    bundle: bool
    bytes: int
    rows: int = Field(description="rows in the file's data table, header excluded")
    columns: int = Field(description="columns in the file's data table")
    sha256: str = Field(description="the hex SHA-256 of the file's bytes")
    frame: ExportFrame
    citation: str
    licences: list[ExportLicence] = Field(description="each source of the frame once")
    pinned: bool = Field(description="true: the file is held in the pinned tier, kept for good")


def export_id(request: ExportRequest) -> str:
    """`exp_` and the first 32 hex digits of the SHA-256 of the RFC 8785 form of the request's
    `{spec_id, content_version, format, bundle}`. A bare request is keyed by the id it names and
    the version it resolved to, so it shares its export with a request naming that pair."""
    preimage = {
        "spec_id": request.frame.spec_id,
        "content_version": request.frame.content_version,
        "format": request.format,
        "bundle": request.bundle,
    }
    return "exp_" + hashlib.sha256(rfc8785.dumps(preimage)).hexdigest()[:32]


@functools.cache
def restricted_sources() -> Mapping[str, str]:
    """The licence deny-list: each source ref whose licence does not allow redistribution, with
    the reason."""
    return cast(Mapping[str, str], json.loads(_LICENCE_RESTRICTIONS.read_text(encoding="utf-8")))


def check_licences(refs: Iterable[str]) -> None:
    """Refuse `LicenceNotRedistributable` when the deny-list names any of `refs`."""
    restricted = restricted_sources()
    listed = [(ref, restricted[ref]) for ref in dict.fromkeys(refs) if ref in restricted]
    if listed:
        raise LicenceNotRedistributable(sources=listed)


def _licences(citation: CitationModel) -> list[ExportLicence]:
    return [
        ExportLicence(
            source=source.ref, label=source.label, licence=source.license, url=source.license_url
        )
        for source in citation.sources_by_ref().values()
    ]


def file_shape(data: bytes, *, format: ExportFormat, bundle: bool) -> tuple[int, int]:
    """The rows, header excluded, and columns of the data table of the file `data`: a bundle's
    first data resource, and an xlsx workbook's `data` sheet. The table is read as a stream, so
    the count holds neither a decoded copy of a CSV nor an unzipped copy of a bundle's member."""
    if not bundle:
        return _table_shape(io.BytesIO(data), format=format)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        package = json.loads(archive.read("datapackage.json"))
        with archive.open(package["resources"][0]["path"]) as member:
            return _table_shape(member, format=format)


def _table_shape(table: IO[bytes], *, format: ExportFormat) -> tuple[int, int]:
    """`file_shape` of the data table read from the seekable stream `table`."""
    if format == "parquet":
        metadata = pq.read_metadata(table)
        return metadata.num_rows, metadata.num_columns
    if format == "csv":
        with io.TextIOWrapper(table, encoding="utf-8", newline="") as text:
            reader = csv.reader(text)
            header = next(reader)
            return sum(1 for _ in reader), len(header)
    workbook = openpyxl.load_workbook(table, read_only=True)
    try:
        # `materialise.formats.write_excel` writes each sheet's dimension, so the sheet's size
        # is read from it in place of scanning every row.
        sheet = workbook[DATA_SHEET_TITLE]
        return cast(int, sheet.max_row) - 1, cast(int, sheet.max_column)
    finally:
        workbook.close()


def _check_rows(request: ExportRequest, rows: int) -> None:
    if request.format != "parquet" and rows > EXPORT_MAX_ROWS:
        raise ExportOverCeiling(format=request.format, count=rows, cap=EXPORT_MAX_ROWS)


def _refuse_spent_caller(caller: Caller | None, *, current: Deps) -> None:
    """Refuse the open caller `caller` when it has no `pin` unit left, before a render it could
    not pin (`refuse_spent_caller`). It spends nothing. `copy_to_pinned` still spends, and judges
    a race for the last unit. A holder of the pin grant, `None`, has no allowance."""
    if caller is not None:
        refuse_spent_caller(current.allowances, caller, now=current.now())


def _pinned_location(digest: str, *, current: Deps) -> tuple[str, str] | None:
    """The bucket and object name of `digest`'s pinned copy, `None` when it has none."""
    location = current.artifact_store.location(digest, tier=Tier.pinned)
    return None if location is None else (location.bucket, location.object_name)


def _copied_to_pinned(
    digest: str, *, frame: ExportFrame, caller: Caller | None, current: Deps
) -> tuple[str, str]:
    """Copy `digest` to the pinned tier (`copy_to_pinned`) and return the copy's bucket and
    object name. Working bytes gone before the copy refuse `ArtifactExpired`. A spent unit
    refuses only while no pinned copy exists, since a concurrent prepare may have spent the last
    one on these bytes."""
    try:
        with phase("copy"):
            copied = copy_to_pinned(
                current.artifact_store,
                digest,
                allowances=current.allowances,
                caller=caller,
                now=current.now(),
            )
    except ArtifactNotFoundError as exc:
        raise ArtifactExpired(id=frame.spec_id, content_version=frame.content_version) from exc
    except (AllowanceExhausted, PinBudgetExhausted, OpenPinBudgetExhausted):
        pinned = _pinned_location(digest, current=current)
        if pinned is None:
            raise
        return pinned
    return copied.bucket, copied.object_name


def _pinned_copy(
    *,
    digest: str,
    tier: str,
    bucket: str,
    object_name: str,
    frame: ExportFrame,
    subject_id: str,
    subject_ids: tuple[str, ...],
    export: str,
    caller: Caller | None,
    current: Deps,
) -> tuple[str, str]:
    """The bucket and object name of the pinned copy of `digest`, stored at `tier`, copying it
    from the working tier unless a pinned copy exists (`_copied_to_pinned`), and record the
    export's reference to it under the version `subject_id` records, unless a revocation under
    it or `subject_ids` refuses."""
    pinned = (
        (bucket, object_name) if tier == Tier.pinned else _pinned_location(digest, current=current)
    )
    already_pinned = pinned is not None
    if pinned is None:
        pinned = _copied_to_pinned(digest, frame=frame, caller=caller, current=current)
    record_pin_reference(
        current.reference_writer,
        principal=f"export:{export}",
        artifact_digest=digest,
        id=subject_id,
        content_version=frame.content_version,
        revoked_under=subject_ids,
        already_pinned=already_pinned,
        now=current.now(),
        reference_kind=EXPORT_REFERENCE_KIND,
        require_link=False,
    )
    return pinned


def _linked_complete(stored: ServableVersion, digest: str) -> bool:
    """Whether the version links `digest` under a complete rendering. An incomplete render and
    the complete render after it can give the same bytes."""
    return any(
        location.artifact_digest == digest and not is_incomplete_rendering(location.rendering)
        for location in stored.servable
    )


def _stored_file_shape(
    digest: str, *, request: ExportRequest, current: Deps, deps: DefinitionDeps
) -> tuple[int, int]:
    """`file_shape` of the pinned file `digest`, whose bytes stay inside one materialise
    slot. Only the file states its table's shape, and the artifact store reads a whole object,
    so the whole file is read."""
    with bounded_materialise_slot(
        current.materialise_semaphore,
        wait_seconds=current.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        return file_shape(
            current.artifact_store.get(artifact_digest=digest, tier=Tier.pinned),
            format=request.format,
            bundle=request.bundle,
        )


def _prepare_dataset(
    request: ExportRequest,
    *,
    export: str,
    caller: Caller | None,
    current: Deps,
    deps: DefinitionDeps,
) -> ExportRecord:
    frame = request.frame
    stored, subject_ids = read_version(
        id=frame.spec_id, content_version=frame.content_version, deps=deps
    )
    facts = stored_manifest_facts(stored.version)
    citation = CitationModel.model_validate(facts.citation)
    licences = _licences(citation)
    check_licences(licence.source for licence in licences)
    _check_rows(request, cast(int, facts.coverage["rows"]))

    # The file holds the enrichment a client's frame of the version holds: the columns a Spec
    # root reads it with, added to the long layout's own.
    requested = resolve_enrichment(
        MintOptions(format=request.format, bundle=request.bundle, enrichment="default"),
        coverage=CoverageModel.model_validate(facts.coverage),
    )
    location = listed_file_location(
        stored,
        id=frame.spec_id,
        requested=requested,
        enrichment_digest=deps.code_index.enrichment_digest,
        deps=deps,
    )
    if location is None:
        _refuse_spent_caller(caller, current=current)
        stored, location = render_file_location(
            stored, id=frame.spec_id, subject_ids=subject_ids, requested=requested, deps=deps
        )
    if not _linked_complete(stored, location.artifact_digest):
        raise ExportFileIncomplete(id=frame.spec_id, content_version=frame.content_version)
    bucket, object_name = _pinned_copy(
        digest=location.artifact_digest,
        tier=location.tier,
        bucket=location.bucket,
        object_name=location.object_name,
        frame=frame,
        subject_id=stored.version.subject_id,
        subject_ids=subject_ids,
        export=export,
        caller=caller,
        current=current,
    )
    rows, columns = _stored_file_shape(
        location.artifact_digest, request=request, current=current, deps=deps
    )
    return _record(
        request,
        export=export,
        subject_id=stored.version.subject_id,
        digest=location.artifact_digest,
        bucket=bucket,
        object_name=object_name,
        file_name=artifact_file_name(location, stored=stored),
        media_type=location.media_type,
        size_bytes=location.size_bytes,
        rows=rows,
        columns=columns,
        citation=citation.text,
        licences=licences,
    )


@dataclass(frozen=True, kw_only=True)
class _StoredDerivedFile:
    """A derived frame's file in the working tier, with the facts its bytes gave."""

    stored: StoredArtifact
    file_name: str
    media_type: str
    rows: int
    columns: int


def _render_and_store(request: ExportRequest, *, current: Deps) -> _StoredDerivedFile:
    """Render the derived frame's file, count it, and store it in the working tier.

    Raises:
        ExportOverCeiling: The file holds more rows than a CSV or xlsx export may.
    """
    rendered = render_derived_file(
        spec_id=request.frame.spec_id,
        content_version=request.frame.content_version,
        options=MintOptions(format=request.format, bundle=request.bundle),
        reads=derived_reads(current),
        site=current.settings.site,
    )
    rows, columns = file_shape(rendered.data, format=request.format, bundle=request.bundle)
    _check_rows(request, rows)
    with phase("store"):
        stored = current.artifact_store.put(rendered.data, tier=Tier.working)
    return _StoredDerivedFile(
        stored=stored,
        file_name=rendered.file_name,
        media_type=rendered.media_type,
        rows=rows,
        columns=columns,
    )


def _stored_derived_file(
    request: ExportRequest, *, current: Deps, deps: DefinitionDeps
) -> _StoredDerivedFile:
    """`_render_and_store` inside one materialise slot, which releases with the bytes freed."""
    with bounded_materialise_slot(
        current.materialise_semaphore,
        wait_seconds=current.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    ):
        return _render_and_store(request, current=current)


def _prepare_derived(
    request: ExportRequest,
    *,
    export: str,
    caller: Caller | None,
    current: Deps,
    deps: DefinitionDeps,
) -> ExportRecord:
    frame = request.frame
    read_servable_version(
        current.snapshot_reader,
        subject_id=frame.spec_id,
        content_version=frame.content_version,
        now=current.now(),
    )
    recipe = current.recipe_store.read_recipe(
        spec_id=frame.spec_id, content_version=frame.content_version
    )
    citation = full_recipe_citation(current.snapshot_reader, recipe)
    if citation is None:
        raise ProvenanceUnavailable(spec_id=frame.spec_id, content_version=frame.content_version)
    _check_roots(recipe, frame=frame, current=current)
    licences = _licences(citation)
    check_licences(licence.source for licence in licences)

    _refuse_spent_caller(caller, current=current)
    file = _stored_derived_file(request, current=current, deps=deps)
    stored = file.stored
    bucket, object_name = _pinned_copy(
        digest=stored.digest,
        tier=stored.tier,
        bucket=stored.bucket,
        object_name=stored.object_name,
        frame=frame,
        subject_id=frame.spec_id,
        subject_ids=(frame.spec_id,),
        export=export,
        caller=caller,
        current=current,
    )
    return _record(
        request,
        export=export,
        subject_id=frame.spec_id,
        digest=stored.digest,
        bucket=bucket,
        object_name=object_name,
        file_name=file.file_name,
        media_type=file.media_type,
        size_bytes=stored.size_bytes,
        rows=file.rows,
        columns=file.columns,
        citation=citation.text,
        licences=licences,
    )


def _record(
    request: ExportRequest,
    *,
    export: str,
    subject_id: str,
    digest: str,
    bucket: str,
    object_name: str,
    file_name: str,
    media_type: str,
    size_bytes: int,
    rows: int,
    columns: int,
    citation: str,
    licences: Sequence[ExportLicence],
) -> ExportRecord:
    body = ExportResponse(
        export_id=export,
        url_path=f"{EXPORT_LINK_PREFIX}/{export}",
        file_name=file_name,
        media_type=media_type,
        format=request.format,
        bundle=request.bundle,
        bytes=size_bytes,
        rows=rows,
        columns=columns,
        sha256=digest.removeprefix("sha256:"),
        frame=request.frame,
        citation=citation,
        licences=list(licences),
        pinned=True,
    )
    return ExportRecord(
        export_id=export,
        subject_id=subject_id,
        content_version=request.frame.content_version,
        artifact_digest=digest,
        bucket=bucket,
        object_name=object_name,
        body=body.model_dump(mode="json"),
    )


def resolve_export_frame(
    frame: ExportFrame | BareExportFrame, *, current: Deps, deps: DefinitionDeps
) -> ExportFrame:
    """`frame` itself when it names its version, else the dataset version its bare `spec_id`
    reads, as the analysis runner reads a Spec root of it (`analysis_roots.newest_spec_version`)
    and the Presentation frames door reads a bare id. The pair keeps the id sent in place of the
    served id, so `_servable_facts` keeps honouring a revocation or quarantine under it.

    Raises:
        NewestVersionUnservable: The version is building, or the serve refused with a code
            `datasets.exports` does not register.
        Exception: A refusal of the serve whose code `datasets.exports` registers, unchanged.
    """
    if isinstance(frame, ExportFrame):
        return frame
    try:
        dataset = newest_spec_version(
            frame.spec_id, deps=deps, recipes=current.recipe_store, trace_id=get_trace_id()
        )
    except AnalysisPending as pending:
        raise NewestVersionUnservable(
            code="service_unavailable",
            retryable=True,
            detail=f"the newest version of {frame.spec_id} is building; retry",
            headers={"Retry-After": str(math.ceil(pending.retry_after_s))},
        ) from None
    except Exception as exc:
        if passes_through(exc):
            raise
        raise _unservable(frame.spec_id, exc) from exc
    return ExportFrame(spec_id=frame.spec_id, content_version=dataset.content_version)


def passes_through(exc: BaseException) -> bool:
    """Whether the exports door answers `exc`, a failure resolving a bare `spec_id`, unchanged:
    it is no refusal, so the surface classifies it, or a refusal whose code `datasets.exports`
    registers."""
    return not isinstance(exc, Refusal) or _status(exc.code, "datasets.exports") is not None


def _status(code: str, endpoint: str) -> int | None:
    """The status the code registry gives `code` on `endpoint`, `None` where it gives none."""
    try:
        with bound_registry_endpoint(endpoint):
            return registered_status(code)
    except LookupError:
        return None


def resolve_endpoint(exc: BaseException) -> str | None:
    """The registry endpoint whose statuses classify `exc`, a failure resolving a bare
    `spec_id`, `None` for a refusal neither reads: `datasets.analysis`, whose resolve this is,
    else `datasets.exports`, which registers the rendering refusals a rebuild raises and the
    analysis door serves none of. A non-refusal classifies under the first."""
    if not isinstance(exc, Refusal):
        return "datasets.analysis"
    for endpoint in ("datasets.analysis", "datasets.exports"):
        if _status(exc.code, endpoint) is not None:
            return endpoint
    return None


def _unservable(spec_id: str, exc: Exception) -> NewestVersionUnservable:
    """The `NewestVersionUnservable` of the registered code that stands for `exc`, a failure
    resolving `spec_id` that does not pass through, read from the status its `resolve_endpoint`
    gives its code. A code no endpoint registers is the server's own rendering failing, an
    `internal_error`. Nothing is classified here, so the door classifies the failure once."""
    assert isinstance(exc, Refusal)
    detail = f"the newest version of {spec_id} cannot be served: {exc.code}: {exc.detail or exc}"
    endpoint = resolve_endpoint(exc)
    status = None if endpoint is None else _status(exc.code, endpoint)
    if status is None:
        return NewestVersionUnservable(
            code="internal_error",
            detail=f"the newest version of {spec_id} cannot be served: {exc}",
        )
    if exc.retryable:
        return NewestVersionUnservable(
            code="service_unavailable", retryable=True, detail=detail, headers=exc.headers
        )
    if status >= 500:
        return NewestVersionUnservable(code="internal_error", detail=detail)
    return NewestVersionUnservable(code="snapshot_not_found", detail=detail)


def _served_ids(id: str, *, current: Deps) -> tuple[str, ...]:
    """`id` and each served id bound to it, whose revocations withdraw a version named `id`
    (`served_subject_ids`). An `ops_` id, or an id the registry does not hold, binds none.

    Raises:
        IdentityQuarantined: `id` or a served id bound to it is quarantined.
    """
    try:
        return served_subject_ids(id, spec_reader=current.spec_reader)
    except SpecNotFound:
        return (id,)


def _check_version(
    subject_id: str,
    content_version: str,
    *,
    named: str,
    frame: ExportFrame | _StoredFrame,
    current: Deps,
) -> None:
    """Refuse the export of `frame` when the version `subject_id` records, or its base version,
    is revoked under `subject_id`, under `named` or under a served id bound to `named`.

    Raises:
        IdentityQuarantined: `named` or a served id bound to it is quarantined.
        SnapshotRevoked: The version is revoked.
        RuntimeError: No version record stands behind the export, which no write allows.
    """
    state = current.snapshot_reader.read_version_state(
        subject_id, content_version, revoked_under=_served_ids(named, current=current)
    )
    if state is None:
        raise RuntimeError(
            f"the export of {frame.spec_id!r} at {frame.content_version!r} rests on "
            f"{subject_id!r} at {content_version!r}, which no version record holds"
        )
    if state.revocation is not None:
        raise SnapshotRevoked(
            id=frame.spec_id,
            content_version=frame.content_version,
            reason=state.revocation.reason,
        )


def _check_roots(
    recipe: RecipeRecord | None, *, frame: ExportFrame | _StoredFrame, current: Deps
) -> None:
    """`_check_version` of every root of the derived frame `frame`'s stored recipe.

    Raises:
        RuntimeError: The frame has no recipe or a recipe with no roots, so no root can be
            checked. A prepare refuses such a frame first, so no export rests on one.
    """
    roots = stored_recipe_roots(recipe)
    if not roots:
        raise RuntimeError(
            f"the export of {frame.spec_id!r} at {frame.content_version!r} has no recipe roots "
            "to check"
        )
    for root in roots:
        _check_version(
            root["spec_id"],
            root["content_version"],
            named=root["spec_id"],
            frame=frame,
            current=current,
        )


def _servable_facts(record: ExportRecord, *, current: Deps) -> Mapping[str, Any]:
    """The facts of `record`, as stored, once every guard a served export answers to passes. See the
    module docstring.

    Raises:
        IdentityQuarantined: The named id, or a root of a derived frame, is quarantined.
        SnapshotRevoked: The version, its base version, or a root of a derived frame is revoked.
        LicenceNotRedistributable: The deny-list names a source of the export.
        RuntimeError: The version record or the recipe the export rests on is gone
            (`_check_version`, `_check_roots`), which answers `internal_error`.
    """
    facts = _StoredFacts.model_validate(record.body)
    frame = facts.frame
    with phase("revocation"):
        _check_version(
            record.subject_id,
            record.content_version,
            named=frame.spec_id,
            frame=frame,
            current=current,
        )
        if frame.spec_id.startswith("ops_"):
            recipe = current.recipe_store.read_recipe(
                spec_id=frame.spec_id, content_version=frame.content_version
            )
            _check_roots(recipe, frame=frame, current=current)
    check_licences(licence.source for licence in facts.licences)
    return record.body


def recorded_export(*, request: ExportRequest, current: Deps) -> Mapping[str, Any] | None:
    """The facts of the export recorded under the request's id, or `None` when none is.

    Raises:
        IdentityQuarantined, SnapshotRevoked, LicenceNotRedistributable: As `_servable_facts`.
    """
    record = current.export_store.read_export(export_id(request))
    if record is None:
        return None
    return _servable_facts(record, current=current)


def prepare_export(
    *, request: ExportRequest, caller: Caller | None, current: Deps, deps: DefinitionDeps
) -> Mapping[str, Any]:
    """Prepare the file `request` names, pin it, record it, and return its facts, or answer the
    record a concurrent prepare left. See the module docstring. `caller` names an open caller,
    whose `pin` allowance a new pinned copy spends, and `None` a caller holding the pin grant.
    `deps` is the request's retrieval view of `current`, under its deadline.

    Raises:
        SpecNotFound, IdentityQuarantined, SnapshotNotFound, SnapshotRevoked, ArtifactExpired:
            The version cannot be read, or its file can no longer be rendered.
        ProvenanceUnavailable: A source of a derived frame cannot be named.
        ExportFileIncomplete: The file rendered without its sources' documentation.
        LicenceNotRedistributable: The deny-list names a source of the frame.
        ExportOverCeiling: A CSV or xlsx export of more than `EXPORT_MAX_ROWS` rows.
        PinBudgetExhausted, OpenPinBudgetExhausted, AllowanceExhausted: The copy's units are
            spent.
        MintBackpressure: A concurrent prepare of the request is still running at the
            deadline, or no materialise permit came free.
    """
    export = export_id(request)
    with deps.rebuilds.turn(
        export, deadline=deps.deadline, wait_seconds=deps.settings.materialise_wait_seconds
    ):
        # A prepare can end between the door's read and this turn, so the turn reads again
        # whether or not it waited.
        recorded = recorded_export(request=request, current=current)
        if recorded is not None:
            return recorded
        prepare = _prepare_derived if request.frame.spec_id.startswith("ops_") else _prepare_dataset
        record = prepare(request, export=export, caller=caller, current=current, deps=deps)
        with phase("record"):
            stored = current.export_store.write_export(record)
        return _servable_facts(stored, current=current)


def sign_export(*, export_id: str, current: Deps) -> str:
    """A URL to the export's pinned file, signed for `DOWNLOAD_SIGNED_URL_TTL` and naming the
    download by its file name.

    The version is read again first (`_servable_facts`), so a takedown reaches the link as it
    reaches every other file link.

    Raises:
        ExportNotFound: No export is recorded under `export_id`.
        IdentityQuarantined, SnapshotRevoked, LicenceNotRedistributable: As `_servable_facts`.
        SnapshotSigningFailed: Signing failed.
    """
    record = current.export_store.read_export(export_id)
    if record is None:
        raise ExportNotFound(export_id=export_id)
    body = _servable_facts(record, current=current)
    return signed_download_url(
        record.artifact_digest,
        tier=Tier.pinned,
        bucket=record.bucket,
        object_name=record.object_name,
        media_type=body["media_type"],
        file_name=body["file_name"],
        artifact_store=current.artifact_store,
    )
