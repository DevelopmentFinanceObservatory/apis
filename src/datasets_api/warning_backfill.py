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
(`manifest.stored_manifest`). It rewrites no other format. The codes in
`dataset_store.warning_formats.CODES_LEFT_BY_SQL`, which `warning_rewrite.sql` leaves for
their writers, keep their stored messages.

For one version row, the old sentence names the first five members it left out, in the
trace's dcid order, and counts K in all. The members holding no observation are those the
trace records with an `observation_count` of 0. The task probes those members' facets as the
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

A row is refused for good (`permanent`) when it cannot be rewritten from what it stores: no
definition or trace, a named member the trace says served observations, fewer zero-count
members after the last named one than K counts, a body and column holding different warnings
(`mismatched`), a composite, an identity the adapter refuses, a probe over its pair ceiling
or refusing its requests, a graph or SDMX read answering 4xx or refusing its query, a dataset
the code index disagrees with, any failure the probe does not document, or a continuation the
trace cannot state. The last is a range window with an end year whose trace names no
continuation of a selected method that is not estimated, while a zero-count member holds an
estimated facet matching every other selection (`facets.continuation`): the trace leaves out
a continuing method that served no row, so the selection cannot be rebuilt. A permanent
refusal follows from the stored row, the graph's answer and this image, so every run refuses
it again; it is reported on every run and sets no exit code. A graph or SDMX read failing in
transport, timing out or answering 5xx, a UN read recorded as failed beside ours
(`origin_catalogue.UnBlockedPlaces.failures`), or a member whose default place scope is not
settled leaves the row for the next run (`transient`). The places the UN front door refuses do
not: it refused them at the mint too. An update the store fails as an outage does
(`clears_on_retry`) stops the run, and the rows not yet tried count as `remaining`.

Rows come by subject, and a probe, or its failure, is kept until the subject changes, keyed
by its zero-count members, the mint's window and the provenance the selections dispatch, so
the versions of one definition probe once. Each row is written in a transaction of its own,
and the update holds the row to the warnings it read in both the column and the body, so a
row changed meanwhile is left (`conflicts`).

With `dry_run` it does all but the update, writes one `datasets_warning_backfill_row` line per
row with K and its split, and classifies the rows already in the new format, with member
fields and no old sentence, the same way, comparing its split with the stored one
(`new_format`). That measures how far the graph has moved since those rows were written. A
row in the new format sets no exit code. A row this task rewrote is in the new format, so a
dry run after a real one counts it under `new_format_rows`; the real run's `rewritten` counts
the rows it rewrote.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from functools import partial
from typing import cast

from dfo_spec_kit.identity import IdentityError

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
from dc_sdmx import SdmxCapExceededError, SdmxClientError, SdmxHttpError, SdmxQueryError
from materialise.codes import MATERIALISE_CODE_CATALOGUE, MemberFields
from materialise.continued_series import continued_series_of_trace
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DatasetDisagreementError,
    MaterialisationUsageError,
)
from materialise.facets import (
    ElectedDataset,
    FacetSelection,
    dataset_candidates,
    facet_axis_value,
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
from materialise.group_facets import UncountedKind, classify_uncounted
from materialise.method_status import status_of
from materialise.pipeline import MaterialisationRequest
from materialise.place_scope import AvailabilityReader
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
    warnings: list[dict[str, object]]
    # The `warnings` of `immutable_body`, `None` when the body holds none.
    body_warnings: object


def select_rows(store: DatasetStore, *, new_format: bool) -> list[WarningRow]:
    """The version rows holding a `coverage_below_scope` warning with the old member sentence,
    and with `new_format` those holding one in the new format, with member fields, by subject
    and then oldest first, so a run keeps the probes of one subject at a time (`Readers.probes`)."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, content_version, recorded_at, definition, trace, warnings, "
            "immutable_body -> 'warnings' AS body_warnings "
            "FROM datasets.version_record "
            "WHERE jsonb_typeof(warnings) = 'array' AND EXISTS ("
            "SELECT 1 FROM jsonb_array_elements(warnings) AS w "
            "WHERE w ->> 'code' = %s AND (w ->> 'message' ~ %s OR (%s AND ("
            "w -> 'off_cadence_members' IS NOT NULL "
            "OR w -> 'out_of_window_members' IS NOT NULL)))) "
            "ORDER BY subject_id, recorded_at, content_version",
            (MEMBER_SENTENCE_CODE, MEMBER_SENTENCE, new_format),
        )
    return [
        WarningRow(
            subject_id=str(row["subject_id"]),
            content_version=str(row["content_version"]),
            recorded_at=cast(datetime, row["recorded_at"]),
            definition=row["definition"],
            trace=row["trace"],
            warnings=cast("list[dict[str, object]]", row["warnings"]),
            body_warnings=row["body_warnings"],
        )
        for row in rows
    ]


class _Refused(Exception):
    """The row cannot be rewritten from what it stores. A later run refuses it too."""


class _Mismatched(_Refused):
    """The row's body holds other warnings than its column, so no rewrite of one fits both."""


class _Unread(Exception):
    """The graph read the rewrite needs did not complete. A later run can get past it."""


class _Outage(Exception):
    """The store failed an update in a way a later run can get past. The run stops."""


@dataclass(frozen=True)
class _Member:
    """One zero-count member's probed candidates and those its elected dataset publishes."""

    candidates: tuple[Facet, ...]
    electable: tuple[Facet, ...]


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
    # The current subject's probes, by `_members`'s key.
    probes: dict[tuple[object, ...], Mapping[str, _Member] | Exception] = field(
        default_factory=dict
    )
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


def run_backfill_warnings(
    rows: Sequence[WarningRow],
    *,
    store: DatasetStore,
    readers: Readers,
    dry_run: bool,
    deadline: datetime,
    now: Clock,
    clears_on_retry: Callable[[BaseException], bool],
) -> int:
    """Rewrite the member sentences of `rows` (`select_rows`), starting none once `deadline`
    has passed, emit one `datasets_warnings_backfilled` line, and return the number of rows a
    later run can still rewrite: those that failed transiently, lost the compare or were left.

    Each row that fails gets one `datasets_warning_backfill_failed` line, at `ERROR` saying
    whether it was `transient`, and a new-format row's at `WARNING`. An update the store fails
    and `clears_on_retry` calls an outage stops the run; any other failure is the row's own."""
    counts = _Counts()
    subject = None
    for position, row in enumerate(rows):
        if now() >= deadline:
            counts.remaining = len(rows) - position
            break
        if row.subject_id != subject:
            # Rows come by subject, so no later row reads an earlier subject's probes.
            readers.probes.clear()
            subject = row.subject_id
        task = any(_member_sentence(warning) for warning in row.warnings)
        outage = False
        try:
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
            if outage:
                counts.remaining = len(rows) - position
            elif not task:
                counts.new_format_failed += 1
            elif transient:
                counts.transient += 1
            else:
                counts.permanent += 1
                counts.mismatched += isinstance(exc, _Mismatched)
            emit(
                "datasets_warning_backfill_failed",
                # A new-format row measures the graph and leaves the exit code alone.
                severity="ERROR" if task or outage else "WARNING",
                subject_id=row.subject_id,
                content_version=row.content_version,
                new_format=not task,
                transient=transient,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        if outage:
            break
    return _emit_backfilled(counts, dry_run=dry_run, probes=readers.probe_calls)


def emit_nothing_backfilled(*, dry_run: bool) -> None:
    """The `datasets_warnings_backfilled` line of a run that found no row."""
    _emit_backfilled(_Counts(), dry_run=dry_run, probes=0)


def _emit_backfilled(counts: _Counts, *, dry_run: bool, probes: int) -> int:
    """Emit `counts`, the probes made and whether the run was dry, at `ERROR` when a row failed
    or was left, and return how many rows a later run can still rewrite."""
    unfinished = counts.transient + counts.conflicts + counts.remaining
    emit(
        "datasets_warnings_backfilled",
        severity="ERROR" if unfinished or counts.permanent else "INFO",
        dry_run=dry_run,
        probes=probes,
        **asdict(counts),
    )
    return unfinished


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
    warnings: list[dict[str, object]] = []
    splits: list[tuple[int, Split]] = []
    dropped = 0
    for warning in row.warnings:
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
            return
    counts.rewritten += 1
    counts.gone += sum(split.gone for _count, split in splits)
    counts.emptied += sum(split.empty for _count, split in splits)
    counts.dropped += dropped


def _update(
    store: DatasetStore,
    row: WarningRow,
    warnings: list[dict[str, object]],
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
    stated = rest.removeprefix(_DETAIL) if rest.startswith(_DETAIL) else rest
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
    for warning in row.warnings:
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
        split = _split(row, named=named, count=count, readers=readers)
        found = split.fields(member_count=_member_count(trace))
        held = {key: warning[key] for key in ("member_count", *_MEMBER_KEYS) if key in warning}
        agrees = found == held
        agreed = agreed and agrees
        emit(
            "datasets_warning_backfill_row",
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
        raise _Refused("the row stores no trace")
    return cast("Mapping[str, object]", row.trace)


def _identity(row: WarningRow) -> Mapping[str, object]:
    identity = row.definition.get("identity") if isinstance(row.definition, Mapping) else None
    if not isinstance(identity, Mapping):
        raise _Refused("the row stores no definition identity")
    return cast("Mapping[str, object]", identity)


def _member_count(trace: Mapping[str, object]) -> int:
    return len(_resolved(trace))


def _resolved(trace: Mapping[str, object]) -> list[Mapping[str, object]]:
    return cast("list[Mapping[str, object]]", trace.get("resolved_stat_vars") or [])


def _split(row: WarningRow, *, named: Sequence[str], count: int, readers: Readers) -> Split:
    """The members a warning naming `named` of `count` leaves out, split as the module
    docstring says. Raises `_Refused` when the row cannot state them and `_Unread` when the
    probe does not complete."""
    trace, identity = _trace(row), _identity(row)
    zero = [entry for entry in _resolved(trace) if entry.get("observation_count") == 0]
    zero_refs = [cast(str, entry["ref"]) for entry in zero]
    missing = [ref for ref in named if ref not in zero_refs]
    if missing:
        raise _Refused(f"named members hold observations in the trace or are not in it: {missing}")
    after_named = zero_refs[max((zero_refs.index(ref) for ref in named), default=-1) + 1 :]
    if count - len(named) > len(after_named):
        raise _Refused(
            f"{count - len(named)} more members left out, {len(after_named)} zero-count members "
            "follow the last named one"
        )
    window = identity_requested_window(identity, clock_year=row.recorded_at.year)
    selections = _selections(trace)
    members = _members(row, identity, zero, window=window, selections=selections, readers=readers)
    _continuation_stated(selections, members.values(), window=window)
    kinds: dict[str, UncountedKind | None] = {
        ref: classify_uncounted(
            member.candidates, electable=member.electable, selections=selections, window=window
        )
        for ref, member in members.items()
    }
    # The old sentence named members with a jointly matching facet, so those fill the count
    # first, each part in dcid order.
    matched = [ref for ref in after_named if kinds[ref] in ("off_cadence", "out_of_window")]
    rest = [ref for ref in after_named if ref not in matched]
    left_out = [*named, *[*matched, *rest][: count - len(named)]]
    absent_reads_zero = reads_absent_as_zero(
        selections, partial(absent_member_reads_zero, readers.code_index)
    )
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


def _continuation_stated(
    selections: Mapping[str, FacetSelection],
    members: Iterable[_Member],
    *,
    window: RequestedWindow,
) -> None:
    """Raise `_Refused` when the selection on `measurementMethod` may have been continued by a
    method the trace does not name and a member holds a facet only that method could admit.

    A series continues only under a range window with an end year, with an estimated method
    (`method_status.status_of`) of its source and unit, and only when it is not estimated
    itself (`facets.continuation`). `trace.continuations` names a continuing method only when
    it served a row, so such a window whose trace names none leaves the continuation unknown.
    It changes nothing unless a member holds an estimated facet matching every other
    selection."""
    method = selections.get(_METHOD_AXIS)
    provenance = selections.get("provenance")
    source = provenance.ref if provenance is not None else None
    if (
        window.kind != "range"
        or window.end_year is None
        or method is None
        or method.continued_by is not None
        or status_of(method.ref, source) == "estimated"
    ):
        return
    others = {axis: selection for axis, selection in selections.items() if axis != _METHOD_AXIS}
    for member in members:
        for facet in member.candidates:
            if (
                facet.facet_id is not None
                and not method.admits(facet_axis_value(facet, _METHOD_AXIS))
                and status_of(facet.measurement_method, facet.provenance_id) == "estimated"
                and all(
                    selection.admits(facet_axis_value(facet, axis))
                    for axis, selection in others.items()
                )
            ):
                raise _Refused(
                    "the trace names no continuation of the selected method, and "
                    f"{facet.facet_id} under {facet.measurement_method} would match one"
                )


def _members(
    row: WarningRow,
    identity: Mapping[str, object],
    zero: Sequence[Mapping[str, object]],
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    readers: Readers,
) -> Mapping[str, _Member]:
    """Each of the `zero` members' candidates, probed as the mint probed them, or as a probe,
    or its failure, earlier in the run left them for the same requests: the subject, its
    zero-count members, the mint's window and the provenance its selections dispatch, which
    elects each member's dataset (`_request`)."""
    key = (
        row.subject_id,
        tuple(cast(str, entry["ref"]) for entry in zero),
        (window.kind, window.start_year, window.end_year),
        tuple(dispatched_provenances(selections)),
    )
    held = readers.probes.get(key)
    if isinstance(held, Exception):
        raise held
    if held is not None:
        return held
    try:
        found = _probe(identity, zero, window=window, selections=selections, readers=readers)
    except Exception as exc:
        readers.probes[key] = exc
        raise
    readers.probes[key] = found
    return found


def _probe(
    identity: Mapping[str, object],
    zero: Sequence[Mapping[str, object]],
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    readers: Readers,
) -> dict[str, _Member]:
    """Probe the `zero` members as one group under the definition's read scope, and keep the
    candidates each one's elected dataset publishes."""
    readers.probe_calls += 1
    try:
        if composite_components(identity, code_index=readers.code_index) is not None:
            raise _Refused("a composite's members are probed under its component identities")
        requests = [
            _request(identity, entry, window=window, selections=selections, readers=readers)
            for entry in zero
        ]
    except (CompositeUnservable, CurrencyRoleUndeclared, IdentityError) as exc:
        raise _Refused(f"the identity is refused: {type(exc).__name__}: {exc}") from exc
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
    except (
        CollectionProbeCeilingExceededError,
        MaterialisationUsageError,
        SdmxQueryError,
        SdmxCapExceededError,
    ) as exc:
        raise _Refused(f"the probe refused: {type(exc).__name__}: {exc}") from exc
    except (GraphReadError, SdmxClientError) as exc:
        status = exc.upstream_status if isinstance(exc, GraphReadError | SdmxHttpError) else None
        if status is not None and 400 <= status < 500:
            raise _Refused(f"the probe was refused: {type(exc).__name__}: {exc}") from exc
        raise _Unread(f"the probe failed: {type(exc).__name__}: {exc}") from exc
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
                candidates=probe.candidates,
                electable=tuple(dataset_candidates(request, probe.candidates)),
            )
            for request, probe in zip(requests, probes, strict=True)
        }
    except DatasetDisagreementError as exc:
        raise _Refused(f"the code index disagrees: {exc}") from exc


def _request(
    identity: Mapping[str, object],
    entry: Mapping[str, object],
    *,
    window: RequestedWindow,
    selections: Mapping[str, FacetSelection],
    readers: Readers,
) -> MaterialisationRequest:
    """The mint's request for the zero-count member `entry`: its roles, the mint's `window`,
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
