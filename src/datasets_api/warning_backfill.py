"""Rewrite the stored member sentences of `coverage_below_scope` into counts and fields.

A group mint once named the members its facet election did not count in one sentence,
"No observations were served for dcid:a, dcid:b and 3 more, which hold none at the observation
period the other members serve" (`dataset_store.warning_formats.MEMBER_SENTENCE`). A mint now
classifies them (`group_facets.classify_uncounted`) into those off cadence, those out of the
requested years and those holding no facet of the elected group, counts each in a sentence of
its own (`finish.member_sentences`) and names them in `off_cadence_members`,
`out_of_window_members` and `unserved_members` beside `member_count` (`finish.member_fields`).
This task rewrites each stored warning holding the old sentence that way, in the `warnings`
column and in the `warnings` of `immutable_body`, which encodes the same facts for an image
that reads bodies (`manifest.version_body`); this image reads the column alone
(`manifest.stored_manifest`). It rewrites no other format. The codes that
`warning_rewrite.sql` leaves (`dataset_store.warning_formats.CODES_LEFT_BY_SQL`) keep their
stored messages: those in `CODES_STILL_WRITTEN`, which the serving image still writes in
their old shape, and the rest, whose writers count them.

For one version row, the old sentence names the first five members it left out, in the
trace's dcid order, and counts K in all. The members holding no observation are those the
trace records with an `observation_count` of 0. The task probes every member's facets as the
mint did (`facets.probe_collection_candidates`, under the definition's read scope), each
request carrying the dataset the trace's provenance selection elected when the code index
resolves the definition (`facets.dataset_candidates`), and rebuilds the mint's selections
from `trace.facets` and `trace.continuations` as every trace reader reads them
(`continued_series.continued_series_of_trace`) and its window, an open end closed at the year
the row was recorded. Each zero-count member is classified as the mint classifies it
(`group_facets.classify_uncounted`). The K members left out are the named ones and, for the
rest, the zero-count members after the last named one in dcid order, those with a jointly
matching facet first, since the old sentence named only such members. Each of the K is off
cadence, out of the window, or, when it holds no facet of the elected dataset at all, out of
the window, which a member the graph no longer holds is (`gone`). Every zero-count member the
classifier calls unserved is named under `unserved_members`, as a mint names each, whether or
not it is one of the K. The out-of-window members are dropped when the group reads an absent
member as zero (`generation.reads_absent_as_zero`), as a mint drops them. A warning whose
member sentences all drop keeps its other sentences (`emptied`), and is dropped when a mint
would not write it (`finish.writes_coverage_below_scope`, `dropped`).

A row is refused for good (`permanent`) when it cannot be rewritten from what it stores: a
warning that is no object or a `coverage_below_scope` warning that is no `WarningModel`
(`malformed_warning`), no definition or trace, a named member the trace says served observations, fewer zero-count
members after the last named one than K counts, a body and column holding different warnings
(`mismatched`), a composite, an identity the adapter refuses, a probe over its pair ceiling
or refusing its requests, a graph or SDMX read a later read would repeat
(`_clears_on_resend`: a 4xx outside `dc_sdmx.client.TRANSIENT_CLIENT_ERRORS`, an SDMX request
refused before it was sent, as `SdmxUnsupportedFilterError`, a repeated `nextToken`, a read
still reporting one at its page cap, or dcids the graph omits again when asked for them
alone), a dataset the code index disagrees with, or a continuation the trace cannot state.
The last is a method selection the mint's group election continued
(`group_facets.group_election`) with a method the trace does not name, since the trace
names one only when it served a row, so the selection cannot be rebuilt. A permanent
refusal follows from the stored row, the graph's answer and this image, so every run
refuses it again, and its failed line carries a `reason` code an operator filters by. Any
other graph or SDMX read failure, in transport, by timeout or with a 5xx, a 3xx, an
unreadable 2xx or a status among `TRANSIENT_CLIENT_ERRORS`, a UN read recorded as failed
beside ours (`origin_catalogue.UnBlockedPlaces.failures`), or a member whose default place
scope is not settled leaves the row for the next run (`transient`). The places the UN
front door refuses do not: it refused them at the mint too. An update the store fails as an
outage does (`clears_on_retry`) stops the run, and the rows not yet tried count as
`remaining`. A failed read of the next page of rows stops the run too: it emits the summary,
counting the rows not tried as `remaining`, and raises. Any other exception is a defect in this task (`failed`).

The run exits 1 when a later run could still rewrite a row, because its read did not
complete (`transient`), the row changed under the update (`conflicts`), or the deadline or a
store outage stopped the run before it (`remaining`), or when a row failed with a defect
(`failed`), each with an `ERROR` line naming the row but the rows not tried. A refusal
exits 0 with a `WARNING` line naming the row and its count in the summary, so a scheduled
run does not fail on every run over rows that will never change.

Rows come by subject a page at a time (`SelectedRows`), and a probe, or its failure, is kept until the subject changes, keyed
by its members, the mint's window and the provenance the selections dispatch, so the
versions of one definition probe once. Each row is written in a transaction of its own,
and the update holds the row to the warnings it read in both the column and the body, so a
row changed meanwhile is left (`conflicts`).

With `dry_run` it does all but the update, writes one `datasets_warning_backfill_row` line per
row with K and its split, and classifies the rows already in the new format, with member
fields and no old sentence, the same way, comparing its split with the stored one
(`new_format`), each filled only with the kinds a mint counts in those fields. That measures
how far the graph has moved since those rows were written. A row in the new format sets no
exit code unless it fails with a defect, and one that disagrees or fails to read is logged at
`WARNING`. A row this task rewrote is in the new format, so a
dry run after a real one counts it under `new_format_rows`; the real run's `rewritten` counts
the rows it rewrote.
"""

from __future__ import annotations

import itertools
import json
import re
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from functools import cached_property, partial
from typing import cast

from dfo_spec_kit.identity import IdentityError
from pydantic import ValidationError

from artifact_publication.jobs import emit
from code_index.members import absent_member_reads_zero
from code_index.query import CodeIndex
from dataset_store import DatasetStore
from dataset_store.warning_formats import MEMBER_SENTENCE, MEMBER_SENTENCE_CODE
from datasets_api.adapter import (
    composite_components,
    definition_read_scope,
    identity_requested_window,
    to_materialisation_request,
)
from datasets_api.codes_materialise import CompositeUnservable, CurrencyRoleUndeclared
from datasets_api.manifest import warning_model
from datasets_api.operation_store import Clock
from datasets_api.shape import StatVarShape
from dc_graph import GraphNotAdvancingError, GraphOmittedDcidsError
from dc_sdmx import SdmxClientError, SdmxHttpError, SdmxResponseError, SdmxTransportError
from dc_sdmx.client import TRANSIENT_CLIENT_ERRORS
from materialise.codes import MATERIALISE_CODE_CATALOGUE, MemberFields
from materialise.continued_series import continued_series_of_trace
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DatasetDisagreementError,
    MaterialisationUsageError,
)
from materialise.facets import (
    ElectedDataset,
    FacetElection,
    FacetSelection,
    MemberProbe,
    continues_selection,
    continuing_candidates,
    dataset_candidates,
    probe_collection_candidates,
)
from materialise.finish import (
    COUNTRIES_FALLBACK_SENTENCE,
    MAX_COUNTED_REFS,
    member_fields,
    member_sentences,
    writes_coverage_below_scope,
)
from materialise.generation import dispatched_provenances, reads_absent_as_zero
from materialise.graph import Facet, GraphReader, GraphReadError
from materialise.group_facets import (
    UncountedKind,
    classify_uncounted,
    group_breadth,
    group_election,
)
from materialise.pipeline import MaterialisationRequest
from materialise.place_scope import AvailabilityReader, PlaceScopeReadError
from materialise.window import RequestedWindow
from origin_catalogue import recording_un_blocked_places
from series_coverage.index import CoverageIndex

# The most seconds one row's probe may read for.
PROBE_DEADLINE_SECONDS = 300.0

_MEMBER = re.compile(MEMBER_SENTENCE)
_SPLIT_KEYS = ("off_cadence_members", "out_of_window_members")
_MEMBER_KEYS = (*_SPLIT_KEYS, "unserved_members")
_METHOD_AXIS = "measurementMethod"
_DETAIL = MATERIALISE_CODE_CATALOGUE[MEMBER_SENTENCE_CODE][1]


@dataclass(frozen=True)
class WarningRow:
    """One version row holding a warning to rewrite or, for a dry run, one in the new format,
    as stored: its `definition` and `trace` are read when the row is handled."""

    subject_id: str
    content_version: str
    recorded_at: datetime
    definition: object
    trace: object
    warnings: list[object]
    # The `warnings` of `immutable_body`, `None` when the body holds none.
    body_warnings: object


# The version rows `select_rows` reads at a time.
PAGE_ROWS = 100

# The rows holding a `coverage_below_scope` warning with the old member sentence, and with the
# third parameter those holding one in the new format, with member fields.
_SELECTED = (
    "FROM datasets.version_record "
    "WHERE jsonb_typeof(warnings) = 'array' AND EXISTS ("
    "SELECT 1 FROM jsonb_array_elements(warnings) AS w "
    "WHERE w ->> 'code' = %s AND (w ->> 'message' ~ %s OR (%s AND ("
    "w -> 'off_cadence_members' IS NOT NULL "
    "OR w -> 'out_of_window_members' IS NOT NULL))))"
)


@dataclass(frozen=True)
class SelectedRows:
    """The version rows `select_rows` selects, `count` of them when it counted, read by
    subject and then oldest first one page of at most `page_rows` at a time, each page in a
    transaction of its own after the last row of the one before it (a keyset), so a run
    holds one page whatever the row count. A row a run rewrites drops out of the selection,
    and the keyset never reads before the last row read."""

    store: DatasetStore
    new_format: bool
    page_rows: int
    count: int

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[WarningRow]:
        after: tuple[str, datetime, str] | None = None
        while True:
            page = self._page(after)
            yield from page
            if len(page) < self.page_rows:
                return
            last = page[-1]
            after = (last.subject_id, last.recorded_at, last.content_version)

    def _page(self, after: tuple[str, datetime, str] | None) -> list[WarningRow]:
        keyset = (
            ""
            if after is None
            else " AND (subject_id, recorded_at, content_version) > (%s, %s, %s)"
        )
        with self.store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT subject_id, content_version, recorded_at, definition, trace, warnings, "
                f"immutable_body -> 'warnings' AS body_warnings {_SELECTED}{keyset} "
                "ORDER BY subject_id, recorded_at, content_version LIMIT %s",
                (
                    MEMBER_SENTENCE_CODE,
                    MEMBER_SENTENCE,
                    self.new_format,
                    *(after or ()),
                    self.page_rows,
                ),
            )
        return [
            WarningRow(
                subject_id=str(row["subject_id"]),
                content_version=str(row["content_version"]),
                recorded_at=cast(datetime, row["recorded_at"]),
                definition=row["definition"],
                trace=row["trace"],
                warnings=cast("list[object]", row["warnings"]),
                body_warnings=row["body_warnings"],
            )
            for row in rows
        ]


def select_rows(
    store: DatasetStore, *, new_format: bool, page_rows: int = PAGE_ROWS
) -> SelectedRows:
    """The version rows holding a `coverage_below_scope` warning with the old member sentence,
    and with `new_format` those holding one in the new format, with member fields, by subject
    and then oldest first, so a run keeps the probes of one subject at a time (`Readers.probes`).
    Only their count is read here; the rows are read a page at a time (`SelectedRows`)."""
    with store.transaction() as tx:
        counted = tx.fetch_one(
            f"SELECT count(*) AS count {_SELECTED}",
            (MEMBER_SENTENCE_CODE, MEMBER_SENTENCE, new_format),
        )
    return SelectedRows(
        store=store,
        new_format=new_format,
        page_rows=page_rows,
        count=int(cast(int, counted["count"])) if counted is not None else 0,
    )


class _Refused(Exception):
    """The row cannot be rewritten from what it stores. A later run refuses it too. `reason`
    is the code an operator filters the refusal by."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class _Mismatched(_Refused):
    """The row's body holds other warnings than its column, so no rewrite of one fits both."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="mismatch")


class _Unread(Exception):
    """The graph read the rewrite needs did not complete. A later run can get past it."""


class _Outage(Exception):
    """The store failed an update in a way a later run can get past. The run stops."""


@dataclass(frozen=True)
class _Member:
    """One member's probe and the candidates its elected dataset publishes."""

    probe: MemberProbe
    electable: tuple[Facet, ...]

    @property
    def candidates(self) -> tuple[Facet, ...]:
        return self.probe.candidates


@dataclass
class _Probed:
    """One probe's members, keyed by ref in `stat_vars` order, under `window`, and the
    group's elections over them, by selections."""

    members: Mapping[str, _Member]
    stat_vars: tuple[str, ...]
    window: RequestedWindow
    elections: dict[tuple[tuple[str, FacetSelection], ...], FacetElection] = field(
        default_factory=dict
    )

    @cached_property
    def breadth(self) -> Mapping[str, int] | None:
        """The series counts ranking the group's election (`group_facets.group_breadth`)."""
        return group_breadth(
            [self.members[ref].probe for ref in self.stat_vars], window=self.window
        )

    def election(self, selections: Mapping[str, FacetSelection]) -> FacetElection:
        """The group's election under `selections` (`group_facets.group_election`), ranked by
        `breadth`, built once per selections."""
        key = tuple(sorted(selections.items(), key=lambda item: item[0]))
        if key not in self.elections:
            self.elections[key] = group_election(
                [self.members[ref].candidates for ref in self.stat_vars],
                selections,
                stat_vars=self.stat_vars,
                window=self.window,
                breadth=self.breadth,
            )
        return self.elections[key]


@dataclass(frozen=True)
class Split:
    """The members a warning leaves out, as a mint classifies them, and how many of the K the
    graph holds no facet of."""

    off_cadence: tuple[str, ...]
    out_of_window: tuple[str, ...]
    unserved_members: tuple[str, ...]
    gone: int

    @property
    def empty(self) -> bool:
        """Whether no member sentence is left, as when every member left out drops."""
        return not (self.off_cadence or self.out_of_window or self.unserved_members)

    def sentences(self, *, member_count: int) -> tuple[str, ...]:
        return member_sentences(
            off_cadence=self.off_cadence,
            out_of_window=self.out_of_window,
            unserved_members=self.unserved_members,
            member_count=member_count,
        )

    def fields(self, *, member_count: int) -> MemberFields:
        return member_fields(
            off_cadence=self.off_cadence,
            out_of_window=self.out_of_window,
            unserved_members=self.unserved_members,
            member_count=member_count,
        )


@dataclass
class Readers:
    """What a run reads the graph through, the probes it has kept, failed ones included, and
    how many it made."""

    graph: GraphReader
    availability: AvailabilityReader
    code_index: CodeIndex
    coverage: CoverageIndex | None
    # The current subject's probes, by `_probed`'s key.
    probes: dict[tuple[object, ...], _Probed | Exception] = field(default_factory=dict)
    probe_calls: int = 0


@dataclass
class _Counts:
    rewritten: int = 0
    transient: int = 0
    permanent: int = 0
    conflicts: int = 0
    mismatched: int = 0
    remaining: int = 0
    gone: int = 0
    emptied: int = 0
    dropped: int = 0
    new_format_rows: int = 0
    new_format_agreed: int = 0
    new_format_failed: int = 0
    failed: int = 0


def run_backfill_warnings(
    rows: SelectedRows | Sequence[WarningRow],
    *,
    store: DatasetStore,
    readers: Readers,
    dry_run: bool,
    deadline: datetime,
    now: Clock,
    clears_on_retry: Callable[[BaseException], bool],
) -> int:
    """Rewrite the member sentences of `rows` (`select_rows`), starting none once `deadline`
    has passed, emit one `datasets_warnings_backfilled` line, and return the number of rows
    that set the exit code: those that failed transiently (`transient`), lost the compare
    (`conflicts`), were not tried before the deadline or an outage (`remaining`), or failed
    with a defect (`failed`).

    Each row that fails gets one `datasets_warning_backfill_failed` line with its refusal
    `reason` (null for any other failure) and whether it was `refused`, `transient` or a
    `conflict`, which is transient too: at `ERROR` for a defect, a store outage, a conflict
    and a row to rewrite whose read did not complete, and at `WARNING` for a refusal and a
    new-format row whose read did not complete. An update the store fails and
    `clears_on_retry` calls an outage stops the run; any other failure is the row's own."""
    counts = _Counts()
    subject = None
    selected = iter(rows)
    for position in itertools.count():
        try:
            row = next(selected, None)
        except Exception:
            # The next page's read failed, so the rows from here are left for a later run.
            counts.remaining = _left(rows, position)
            _emit_backfilled(counts, dry_run=dry_run, probes=readers.probe_calls)
            raise
        if row is None:
            break
        if now() >= deadline:
            counts.remaining = _left(rows, position)
            break
        if row.subject_id != subject:
            # Rows come by subject, so no later row reads an earlier subject's probes.
            readers.probes.clear()
            subject = row.subject_id
        # A row whose warnings cannot be read is refused as one to rewrite.
        task = True
        outage = False
        try:
            task = any(_member_sentence(warning) for warning in _read_warnings(row))
            if task:
                _backfill_row(
                    row,
                    store=store,
                    readers=readers,
                    dry_run=dry_run,
                    counts=counts,
                    clears_on_retry=clears_on_retry,
                )
            else:
                counts.new_format_rows += 1
                counts.new_format_agreed += _new_format_row(row, readers=readers)
        except Exception as exc:
            # One row that fails is left as it is; the run carries on with the others unless
            # the store is out.
            outage = isinstance(exc, _Outage)
            transient = outage or isinstance(exc, _Unread)
            defect = not (transient or isinstance(exc, _Refused))
            if outage:
                counts.remaining = _left(rows, position)
            elif defect:
                counts.failed += 1
            elif not task:
                counts.new_format_failed += 1
            elif transient:
                counts.transient += 1
            else:
                counts.permanent += 1
                counts.mismatched += isinstance(exc, _Mismatched)
            emit(
                "datasets_warning_backfill_failed",
                # A refusal and a new-format row's failed read leave the exit code alone.
                severity="ERROR" if defect or (task and transient) else "WARNING",
                subject_id=row.subject_id,
                content_version=row.content_version,
                new_format=not task,
                reason=exc.reason if isinstance(exc, _Refused) else None,
                refused=not (transient or defect),
                transient=transient,
                conflict=False,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        if outage:
            break
    return _emit_backfilled(counts, dry_run=dry_run, probes=readers.probe_calls)


def _left(rows: SelectedRows | Sequence[WarningRow], position: int) -> int:
    """The rows of `rows` from `position` on, at least the one there or the one a failed page
    read may hold: a page read after the count can hold rows the count missed."""
    return max(len(rows) - position, 1)


def emit_nothing_backfilled(*, dry_run: bool) -> None:
    """The `datasets_warnings_backfilled` line of a run that found no row."""
    _emit_backfilled(_Counts(), dry_run=dry_run, probes=0)


def _emit_backfilled(counts: _Counts, *, dry_run: bool, probes: int) -> int:
    """Emit `counts`, the probes made and whether the run was dry, at `ERROR` when a row is
    left for a later run or failed unexpectedly, at `WARNING` when a row was refused or a
    new-format row failed or disagreed, and return how many rows set the exit code."""
    unfinished = counts.transient + counts.conflicts + counts.remaining + counts.failed
    measured = counts.permanent or counts.new_format_failed
    disagreed = counts.new_format_agreed < counts.new_format_rows - counts.new_format_failed
    emit(
        "datasets_warnings_backfilled",
        severity="ERROR" if unfinished else "WARNING" if measured or disagreed else "INFO",
        dry_run=dry_run,
        probes=probes,
        **asdict(counts),
    )
    return unfinished


def _read_warnings(row: WarningRow) -> list[Mapping[str, object]]:
    """`row`'s warnings, each a mapping and each of `MEMBER_SENTENCE_CODE` a `WarningModel`.
    Raises `_Refused` for a row holding any other, which every run would read alike."""
    for position, warning in enumerate(row.warnings):
        if not isinstance(warning, Mapping):
            raise _Refused(
                f"warning {position} is a {type(warning).__name__}, not an object",
                reason="malformed_warning",
            )
        if warning.get("code") == MEMBER_SENTENCE_CODE:
            try:
                warning_model(cast("Mapping[str, object]", warning))
            except ValidationError as exc:
                raise _Refused(
                    f"warning {position} is no warning: {exc.error_count()} errors, first "
                    f"{exc.errors()[0]['msg']!r} at {exc.errors()[0]['loc']!r}",
                    reason="malformed_warning",
                ) from exc
    return cast("list[Mapping[str, object]]", row.warnings)


def _member_sentence(warning: Mapping[str, object]) -> re.Match[str] | None:
    if warning.get("code") != MEMBER_SENTENCE_CODE:
        return None
    return _MEMBER.search(cast(str, warning["message"]))


def _backfill_row(
    row: WarningRow,
    *,
    store: DatasetStore,
    readers: Readers,
    dry_run: bool,
    counts: _Counts,
    clears_on_retry: Callable[[BaseException], bool],
) -> None:
    """Rewrite each of `row`'s warnings holding the old member sentence, and update the row
    unless `dry_run` or the stored warnings changed since they were read. Raises
    `_Mismatched` when the body's warnings differ from the column's."""
    if row.body_warnings != row.warnings:
        raise _Mismatched("immutable_body holds other warnings than the warnings column")
    trace = _trace(row)
    warnings: list[Mapping[str, object]] = []
    splits: list[tuple[int, Split]] = []
    dropped = 0
    for warning in _read_warnings(row):
        match = _member_sentence(warning)
        if match is None:
            warnings.append(warning)
            continue
        named = match[1].split(", ")
        count = len(named) + int(match[2] or 0)
        split = _split(row, named=named, count=count, readers=readers)
        splits.append((count, split))
        rewritten = _rewritten(warning, match, split, member_count=_member_count(trace))
        if _unwritten(warning, match, split, trace):
            dropped += 1
        else:
            warnings.append(rewritten)
    if dry_run:
        for count, split in splits:
            emit(
                "datasets_warning_backfill_row",
                subject_id=row.subject_id,
                content_version=row.content_version,
                new_format=False,
                left_out=count,
                off_cadence=len(split.off_cadence),
                out_of_window=len(split.out_of_window),
                unserved_members=len(split.unserved_members),
                gone=split.gone,
                emptied=split.empty,
                dropped=bool(dropped),
            )
    else:
        updated = _update(store, row, warnings, clears_on_retry=clears_on_retry)
        if updated is None:
            counts.conflicts += 1
            emit(
                "datasets_warning_backfill_failed",
                severity="ERROR",
                subject_id=row.subject_id,
                content_version=row.content_version,
                new_format=False,
                reason=None,
                refused=False,
                # A rerun reads the changed row again; `conflicts` counts it apart.
                transient=True,
                conflict=True,
                error="the row's warnings changed after they were read",
            )
            return
    counts.rewritten += 1
    counts.gone += sum(split.gone for _count, split in splits)
    counts.emptied += sum(split.empty for _count, split in splits)
    counts.dropped += dropped


def _update(
    store: DatasetStore,
    row: WarningRow,
    warnings: Sequence[Mapping[str, object]],
    *,
    clears_on_retry: Callable[[BaseException], bool],
) -> Mapping[str, object] | None:
    """Write `warnings` to `row`'s column and body while both still hold what was read, and
    return the row's key, or `None` when either changed. Raises `_Outage` when the store fails
    as `clears_on_retry` says an outage does."""
    try:
        with store.transaction() as tx:
            return tx.fetch_one(
                "UPDATE datasets.version_record SET warnings = %s::jsonb, "
                "immutable_body = jsonb_set(immutable_body, '{warnings}', %s::jsonb) "
                "WHERE subject_id = %s AND content_version = %s AND warnings = %s::jsonb "
                "AND immutable_body -> 'warnings' = %s::jsonb "
                "RETURNING subject_id",
                (
                    json.dumps(warnings),
                    json.dumps(warnings),
                    row.subject_id,
                    row.content_version,
                    json.dumps(row.warnings),
                    json.dumps(row.body_warnings),
                ),
            )
    except Exception as exc:
        if clears_on_retry(exc):
            raise _Outage(f"the store failed: {type(exc).__name__}: {exc}") from exc
        raise


def _rewritten(
    warning: Mapping[str, object], match: re.Match[str], split: Split, *, member_count: int
) -> dict[str, object]:
    """`warning` with its old member sentence replaced by the sentences counting `split`,
    and the fields naming it, as a mint writes them. Raises when the result is no
    `WarningModel`."""
    message = cast(str, warning["message"])
    sentences = "".join(f" {sentence}" for sentence in split.sentences(member_count=member_count))
    rewritten: dict[str, object] = {
        **warning,
        "message": f"{message[: match.start()]}{sentences}{message[match.end() :]}",
        **split.fields(member_count=member_count),
    }
    warning_model(rewritten)
    return rewritten


def _unwritten(
    warning: Mapping[str, object],
    match: re.Match[str],
    split: Split,
    trace: Mapping[str, object],
) -> bool:
    """Whether a mint would write no `coverage_below_scope` for the stored `warning` once
    `split` replaces its old sentence `match` (`finish.writes_coverage_below_scope`). Its
    window was clamped when the trace says so, and an entity went unserved when the warning
    names one in its fields or its message states anything past the catalogue detail and the
    old sentence but the countries-fallback sentence, which says the request was served the
    countries with data."""
    applied = trace.get("applied_window")
    clamped = isinstance(applied, Mapping) and applied.get("origin") == "coverage_clamp"
    message = cast(str, warning["message"])
    rest = f"{message[: match.start()]}{message[match.end() :]}"
    stated = rest.removeprefix(_DETAIL)
    countries_fallback = f" {COUNTRIES_FALLBACK_SENTENCE}" in stated
    stated = stated.replace(f" {COUNTRIES_FALLBACK_SENTENCE}", "")
    return not writes_coverage_below_scope(
        below_scope=clamped or "unserved_entities" in warning or bool(stated.strip()),
        countries_fallback=countries_fallback,
        off_cadence=split.off_cadence,
        out_of_window=split.out_of_window,
        unserved_members=split.unserved_members,
    )


def _new_format_row(row: WarningRow, *, readers: Readers) -> bool:
    """Whether the split of each of `row`'s warnings in the new format equals the stored one, as
    one `datasets_warning_backfill_row` line per warning states."""
    trace = _trace(row)
    agreed = True
    for warning in _read_warnings(row):
        if warning.get("code") != MEMBER_SENTENCE_CODE or not any(
            key in warning for key in _MEMBER_KEYS
        ):
            continue
        stored = [
            cast("Mapping[str, object]", warning[key]) for key in _SPLIT_KEYS if key in warning
        ]
        refs = [
            cast(str, ref) for counted in stored for ref in cast("list[object]", counted["refs"])
        ]
        # Each field names its first five in the mint's order; the sentence named the first
        # five in the trace's dcid order across both kinds, which those fields hold.
        order = [cast(str, entry["ref"]) for entry in _resolved(trace)]
        named = sorted(refs, key=lambda ref: order.index(ref) if ref in order else len(order))[
            :MAX_COUNTED_REFS
        ]
        count = sum(cast(int, counted["count"]) for counted in stored)
        split = _split(row, named=named, count=count, readers=readers, new_format=True)
        found = split.fields(member_count=_member_count(trace))
        held = {key: warning[key] for key in ("member_count", *_MEMBER_KEYS) if key in warning}
        agrees = found == held
        agreed = agreed and agrees
        emit(
            "datasets_warning_backfill_row",
            severity="INFO" if agrees else "WARNING",
            subject_id=row.subject_id,
            content_version=row.content_version,
            new_format=True,
            agrees=agrees,
            stored=held,
            found=found,
            gone=split.gone,
        )
    return agreed


def _trace(row: WarningRow) -> Mapping[str, object]:
    if not isinstance(row.trace, Mapping):
        raise _Refused("the row stores no trace", reason="no_trace")
    return cast("Mapping[str, object]", row.trace)


def _identity(row: WarningRow) -> Mapping[str, object]:
    identity = row.definition.get("identity") if isinstance(row.definition, Mapping) else None
    if not isinstance(identity, Mapping):
        raise _Refused("the row stores no definition identity", reason="no_definition")
    return cast("Mapping[str, object]", identity)


def _member_count(trace: Mapping[str, object]) -> int:
    return len(_resolved(trace))


def _resolved(trace: Mapping[str, object]) -> list[Mapping[str, object]]:
    return cast("list[Mapping[str, object]]", trace.get("resolved_stat_vars") or [])


def _split(
    row: WarningRow,
    *,
    named: Sequence[str],
    count: int,
    readers: Readers,
    new_format: bool = False,
) -> Split:
    """The members a warning naming `named` of `count` leaves out, split as the module
    docstring says, where `new_format` says `count` is that of a mint's member fields. Raises
    `_Refused` when the row cannot state them and `_Unread` when the probe does not
    complete."""
    trace, identity = _trace(row), _identity(row)
    resolved = _resolved(trace)
    zero = [entry for entry in resolved if entry.get("observation_count") == 0]
    zero_refs = [cast(str, entry["ref"]) for entry in zero]
    missing = [ref for ref in named if ref not in zero_refs]
    if missing:
        raise _Refused(
            f"named members hold observations in the trace or are not in it: {missing}",
            reason="named_member_served",
        )
    after_named = zero_refs[max((zero_refs.index(ref) for ref in named), default=-1) + 1 :]
    if count - len(named) > len(after_named):
        raise _Refused(
            f"{count - len(named)} more members left out, {len(after_named)} zero-count members "
            "follow the last named one",
            reason="too_few_members",
        )
    window = identity_requested_window(identity, clock_year=row.recorded_at.year)
    selections = _selections(trace)
    probed = _probed(row, identity, resolved, window=window, selections=selections, readers=readers)
    members = probed.members
    _continuation_stated(selections, probed)
    kinds: dict[str, UncountedKind | None] = {
        ref: classify_uncounted(
            members[ref].candidates,
            electable=members[ref].electable,
            selections=selections,
            window=window,
        )
        for ref in zero_refs
    }
    absent_reads_zero = reads_absent_as_zero(
        selections, partial(absent_member_reads_zero, readers.code_index)
    )
    # The old sentence named members with a jointly matching facet, so those fill the count
    # first, each part in dcid order. A mint's fields count only the kinds it names, and none
    # out of the window when the group reads an absent member as zero; fewer of those than
    # the fields count is the graph's movement, which the comparison reports.
    kinds_counted = (
        ("off_cadence",) if new_format and absent_reads_zero else ("off_cadence", "out_of_window")
    )
    matched = [ref for ref in after_named if kinds[ref] in kinds_counted]
    rest = [] if new_format else [ref for ref in after_named if ref not in matched]
    left_out = [*named, *[*matched, *rest][: count - len(named)]]
    return Split(
        off_cadence=tuple(ref for ref in left_out if kinds[ref] == "off_cadence"),
        # A member holding no facet of the elected dataset was named as the old sentence named
        # it, and holds no observation in the requested years.
        out_of_window=()
        if absent_reads_zero
        else tuple(ref for ref in left_out if kinds[ref] in ("out_of_window", None)),
        unserved_members=tuple(ref for ref in zero_refs if kinds[ref] == "unserved"),
        gone=sum(1 for ref in left_out if not members[ref].candidates),
    )


def _selections(trace: Mapping[str, object]) -> dict[str, FacetSelection]:
    """The mint's selections, from the first `trace.facets` entry on each axis, with the
    `measurementMethod` selection continued by the method `trace.continuations` names, read as
    every reader of a trace reads them (`continued_series.continued_series_of_trace`)."""
    continued = continued_series_of_trace(trace)
    selections: dict[str, FacetSelection] = {}
    for entry in cast("list[Mapping[str, object]]", trace.get("facets") or []):
        axis = cast(str, entry["axis"])
        if axis in selections:
            continue
        continued_by = continued[0].method if axis == _METHOD_AXIS and continued else None
        selections[axis] = FacetSelection.of_trace_entry(entry, continued_by=continued_by)
    return selections


def _continuation_stated(selections: Mapping[str, FacetSelection], probed: _Probed) -> None:
    """Raise `_Refused` when the mint continued the selection on `measurementMethod` with a
    method the trace does not name.

    `trace.continuations` names a continuing method only when it served a row, so a trace
    naming none leaves a continuation that served none unknown. The group's election is
    rebuilt from the selections and the probed members' candidates in trace order
    (`_Probed.election`), ranked by their series under the window as the mint ranked them
    (`_Probed.breadth`).
    The job binds no whole-variable breadth reader (`facets.bound_variable_breadth`), so a
    single-place tie the mint broke by a variable's whole breadth falls here to the unit's
    name, and its cadence can differ from the mint's. When no member holds a facet matching
    every selection any more, the cadence and the served series are gone, and the row is
    refused if the mint could have continued them (`_may_continue`)."""
    method = selections.get(_METHOD_AXIS)
    if method is None or method.continued_by is not None:
        return
    election = probed.election(selections)
    if election.cadence is None:
        candidates = [probed.members[ref].candidates for ref in probed.stat_vars]
        if _may_continue(candidates, selections, window=probed.window):
            raise _Refused(
                "no member holds the selected facets any more, and an estimated facet of the "
                "selected source and unit could have continued them",
                reason="continuation_unknown",
            )
        return
    continued_by = election.selections[_METHOD_AXIS].continued_by
    if continued_by is not None:
        raise _Refused(
            f"the mint continued the selected method with {continued_by}, which the trace "
            "does not name",
            reason="continuation_unnamed",
        )


def _may_continue(
    candidates: Sequence[Sequence[Facet]],
    selections: Mapping[str, FacetSelection],
    *,
    window: RequestedWindow,
) -> bool:
    """Whether the mint could have continued a served series of `selections` that the graph no
    longer holds with one of `candidates`: its method selection is continuable
    (`facets.continues_selection`), and a candidate continues a series of the selected method,
    source and unit (`facets.continuing_candidates`) whose end is unknown, so no year bounds
    it. Where no source or unit is selected, each one the candidates carry is tried."""
    method = selections.get(_METHOD_AXIS)
    if method is None or not continues_selection(method):
        return False
    facets = [facet for member in candidates for facet in member]
    provenance, unit = selections.get("provenance"), selections.get("unit")
    sources = {provenance.ref} if provenance is not None else {f.provenance_id for f in facets}
    units = {unit.ref} if unit is not None else {f.unit for f in facets}
    return any(
        continuing_candidates(
            facets,
            provenance=source,
            method=method.ref,
            units=units,
            after=None,
            window=window,
            serves=lambda _facet_id: True,
        )
        for source in sources
    )


def _probed(
    row: WarningRow,
    identity: Mapping[str, object],
    resolved: Sequence[Mapping[str, object]],
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    readers: Readers,
) -> _Probed:
    """The `resolved` members' candidates and series counts, probed as the mint probed them,
    or as a probe, or its failure, earlier in the run left them for the same requests: the subject,
    its members, the mint's window and the provenance its selections dispatch, which elects
    each member's dataset (`_request`). A failure kept from an earlier row is raised afresh,
    of its class when it is a refusal or an incomplete read."""
    key = (
        row.subject_id,
        tuple(_ref(entry) for entry in resolved),
        (window.kind, window.start_year, window.end_year),
        tuple(dispatched_provenances(selections)),
    )
    held = readers.probes.get(key)
    if isinstance(held, Exception):
        message = f"an earlier row's probe of the same requests failed: {held}"
        if isinstance(held, _Refused):
            raise _Refused(message, reason=held.reason) from held
        if isinstance(held, _Unread):
            raise _Unread(message) from held
        raise RuntimeError(message) from held
    if held is not None:
        return held
    try:
        members = _probe(identity, resolved, window=window, selections=selections, readers=readers)
    except Exception as exc:
        readers.probes[key] = exc
        raise
    found = _Probed(
        members=members, stat_vars=tuple(_ref(entry) for entry in resolved), window=window
    )
    readers.probes[key] = found
    return found


def _probe(
    identity: Mapping[str, object],
    resolved: Sequence[Mapping[str, object]],
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    readers: Readers,
) -> dict[str, _Member]:
    """Probe the `resolved` members as one group under the definition's read scope, and keep
    the candidates each one's elected dataset publishes."""
    readers.probe_calls += 1
    try:
        if composite_components(identity, code_index=readers.code_index) is not None:
            raise _Refused(
                "a composite's members are probed under its component identities",
                reason="composite",
            )
        requests = [
            _request(identity, entry, window=window, selections=selections, readers=readers)
            for entry in resolved
        ]
    except (CompositeUnservable, CurrencyRoleUndeclared, IdentityError) as exc:
        raise _Refused(
            f"the identity is refused: {type(exc).__name__}: {exc}", reason="identity_refused"
        ) from exc
    try:
        with (
            definition_read_scope(identity, code_index=readers.code_index),
            recording_un_blocked_places() as un_record,
        ):
            probes = probe_collection_candidates(
                requests,
                graph=readers.graph,
                availability=readers.availability,
                settle=True,
                coverage=readers.coverage,
                deadline=time.monotonic() + PROBE_DEADLINE_SECONDS,
            )
    except (CollectionProbeCeilingExceededError, MaterialisationUsageError) as exc:
        raise _Refused(
            f"the probe refused: {type(exc).__name__}: {exc}", reason="probe_refused"
        ) from exc
    except (GraphReadError, SdmxClientError) as exc:
        if _clears_on_resend(exc):
            raise _Unread(f"the probe failed: {type(exc).__name__}: {exc}") from exc
        raise _Refused(
            f"the probe was refused: {type(exc).__name__}: {exc}", reason=_refusal_reason(exc)
        ) from exc
    except TimeoutError as exc:
        # `place_scope.PlaceScopeDeadlineError` among them: the read ran past its deadline.
        raise _Unread(f"the probe timed out: {exc}") from exc
    if failures := un_record.failures():
        raise _Unread(f"a UN read failed beside ours: {failures[0].detail}")
    if unsettled := [r.stat_var for r, p in zip(requests, probes, strict=True) if not p.settled]:
        raise _Unread(f"the default place scope is not settled for {unsettled}")
    try:
        return {
            request.stat_var: _Member(
                probe=probe,
                electable=tuple(dataset_candidates(request, probe.candidates)),
            )
            for request, probe in zip(requests, probes, strict=True)
        }
    except DatasetDisagreementError as exc:
        raise _Refused(f"the code index disagrees: {exc}", reason="dataset_disagreement") from exc


def _clears_on_resend(exc: GraphReadError | SdmxClientError) -> bool:
    """Whether a later read can get past `exc`, as the SDMX client judges a resend: a graph
    read no response answered, an SDMX transport failure or unparsable 2xx, or any status
    but a 4xx outside `dc_sdmx.client.TRANSIENT_CLIENT_ERRORS`, so a 5xx, a 3xx and a 2xx
    body the client could not read are retried. A 4xx outside that set, an SDMX request the
    client refuses before sending it (no status, as `SdmxUnsupportedFilterError`, or
    `SdmxQueryError` and `SdmxCapExceededError`), a paged read whose `nextToken` repeats or
    that still reports one at its page cap (`GraphNotAdvancingError`) and dcids the graph
    omits from a read and its re-request (`GraphOmittedDcidsError`) would repeat, at any
    depth of the cause chain. An availability read failing for a place scope
    (`PlaceScopeReadError`) is judged by the client error it wraps."""
    if any(
        isinstance(cause, GraphNotAdvancingError | GraphOmittedDcidsError) for cause in _causes(exc)
    ):
        return False
    if isinstance(exc, PlaceScopeReadError) and isinstance(
        exc.__cause__, GraphReadError | SdmxClientError
    ):
        return _clears_on_resend(exc.__cause__)
    if isinstance(exc, SdmxTransportError | SdmxResponseError):
        return True
    if not isinstance(exc, GraphReadError | SdmxHttpError):
        return False
    status = exc.upstream_status
    if status is None:
        return isinstance(exc, GraphReadError)
    return not 400 <= status < 500 or status in TRANSIENT_CLIENT_ERRORS


def _refusal_reason(exc: BaseException) -> str:
    """The reason code of a read a later read would repeat (`_clears_on_resend`): a paged read
    that does not advance, dcids the graph omits again, or a request the server or client
    refuses."""
    for cause in _causes(exc):
        if isinstance(cause, GraphNotAdvancingError):
            return "not_advancing"
        if isinstance(cause, GraphOmittedDcidsError):
            return "omitted"
    return "client_refusal"


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """`exc` and each exception it was raised from, in turn."""
    cause: BaseException | None = exc
    while cause is not None:
        yield cause
        cause = cause.__cause__


def _ref(entry: Mapping[str, object]) -> str:
    return cast(str, entry["ref"])


def _request(
    identity: Mapping[str, object],
    entry: Mapping[str, object],
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    readers: Readers,
) -> MaterialisationRequest:
    """The mint's request for the member `entry`: its roles, the mint's `window`,
    and, for a definition the code index resolves, the dataset the trace's provenance
    selection elected with every dataset the index records the member under."""
    stat_var = cast(str, entry["ref"])
    request = to_materialisation_request(
        identity,
        shape=StatVarShape(
            stat_var=stat_var,
            declared_roles=tuple(cast("list[str]", entry["roles"])),
            slots=(),
            measured_property=None,
            population_type=None,
        ),
        code_index=readers.code_index,
    )
    request = replace(request, window=window)
    provenances = dispatched_provenances(selections)
    if not request.index_sourced or len(provenances) != 1:
        return request
    row = readers.code_index.statvar_by_dcid(stat_var.removeprefix("dcid:"))
    return replace(
        request,
        dataset=ElectedDataset(
            provenance=provenances[0],
            member_provenances=row.provenances if row is not None else (),
        ),
    )
