"""Rewrite the stored member sentences of `coverage_below_scope` into counts and fields.

A group mint once named the members its facet election did not count in one sentence,
"No observations were served for dcid:a, dcid:b and 3 more, which hold none at the observation
period the other members serve" (`dataset_store.warning_formats.MEMBER_SENTENCE`). A mint now
splits them into those off cadence, those out of the requested years and those holding no
facet of the elected group, counts each in a sentence of its own (`finish.member_sentences`)
and names them in `off_cadence_members`, `out_of_window_members` and `unserved_members`
beside `member_count` (`finish.member_fields`). This task rewrites each stored warning holding
the old sentence that way, in the `warnings` column and in the `warnings` of `immutable_body`,
which encodes the same facts for an image that reads bodies (`manifest.version_body`); this
image reads the column alone (`manifest.stored_manifest`). A row whose body and column hold
different warnings is refused for good (`mismatched`). It rewrites no other format; `dataset_store.warning_formats.TASK_CODES` names the
codes whose new fields no reader holds yet, and their stored messages stay as written.

For one version row, the old sentence gives the first five members it named and the count K
of all it named. The members holding no observation are those the trace records with an
`observation_count` of 0. A row is refused for good unless every named member is one of them
and K is at most their number. The task probes those members' facets as the mint did
(`facets.probe_collection_candidates`, under the definition's read scope), rebuilds the
mint's selections from `trace.facets` and `trace.continuations`
(`FacetSelection.of_trace_entry`), and takes K members: the named ones, then the others with a
facet jointly matching the selections, then the rest. Each is off cadence when such a facet
holds a point in the requested years (`facets.facets_in_window`), out of the window when none
does, out of the window when it holds no facet at all, which a member the graph no longer holds
does (`gone`), and unserved when it holds facets of which none jointly matches. K stays what
the mint counted. The out-of-window members are dropped when the one provenance the selections
dispatch reads an absent member as zero (`code_index.members.absent_member_reads_zero`), as a
mint drops them; a warning whose member sentences all drop keeps its other sentences
(`emptied`).

A probe that raises, a UN read recorded as failed beside ours
(`origin_catalogue.UnBlockedPlaces.failures`), or a member whose default place scope is not
settled leaves the row as it is for the next run, a transient failure. The places the UN front
door refuses do not: it refused them at the mint too. A probe is kept for the run, keyed by
the subject and its zero-count members, so the versions of one definition share it. Each row
is written in a transaction of its own, and the update holds the row to the warnings it read
in both the column and the body, so a row changed meanwhile is left for the next run.

With `dry_run` it does all but the update, writes one `datasets_warning_backfill_row` line per
row with K and its split, and classifies the rows a mint already split the same way, comparing
its split with the stored one (`control`). That measures how far the graph has moved since
those mints.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import cast

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
from datasets_api.manifest import warning_model
from datasets_api.operation_store import Clock
from datasets_api.shape import StatVarShape
from materialise.codes import MemberFields
from materialise.facets import FacetSelection, facets_in_window, probe_collection_candidates
from materialise.finish import member_fields, member_sentences
from materialise.generation import dispatched_provenances
from materialise.graph import Facet, GraphReader
from materialise.place_scope import AvailabilityReader
from origin_catalogue import recording_un_blocked_places
from series_coverage.index import CoverageIndex

# The most seconds one row's probe may read for.
PROBE_DEADLINE_SECONDS = 300.0

_MEMBER = re.compile(MEMBER_SENTENCE)
_MEMBER_KEYS = ("off_cadence_members", "out_of_window_members", "unserved_members")


@dataclass(frozen=True)
class WarningRow:
    """One version row holding a warning to rewrite or, for the control, one a mint split."""

    subject_id: str
    content_version: str
    identity: Mapping[str, object]
    trace: Mapping[str, object]
    warnings: list[dict[str, object]]
    # The `warnings` of `immutable_body`, `None` when the body holds none.
    body_warnings: object


def select_rows(store: DatasetStore, *, control: bool) -> list[WarningRow]:
    """The version rows holding a `coverage_below_scope` warning with the old member sentence,
    oldest first, and with `control` those holding one a mint split into member fields."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, content_version, definition, trace, warnings, "
            "immutable_body -> 'warnings' AS body_warnings "
            "FROM datasets.version_record "
            "WHERE jsonb_typeof(warnings) = 'array' AND EXISTS ("
            "SELECT 1 FROM jsonb_array_elements(warnings) AS w "
            "WHERE w ->> 'code' = %s AND (w ->> 'message' ~ %s OR (%s AND ("
            "w -> 'off_cadence_members' IS NOT NULL "
            "OR w -> 'out_of_window_members' IS NOT NULL)))) "
            "ORDER BY recorded_at, subject_id, content_version",
            (MEMBER_SENTENCE_CODE, MEMBER_SENTENCE, control),
        )
    return [
        WarningRow(
            subject_id=str(row["subject_id"]),
            content_version=str(row["content_version"]),
            identity=cast(
                "Mapping[str, object]", cast("Mapping[str, object]", row["definition"])["identity"]
            ),
            trace=cast("Mapping[str, object]", row["trace"]),
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


@dataclass(frozen=True)
class Split:
    """The K members a warning leaves out, as a mint splits them, and how many of them the
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
    """What a run reads the graph through, the probes it has kept, and how many it made."""

    graph: GraphReader
    availability: AvailabilityReader
    code_index: CodeIndex
    coverage: CoverageIndex | None
    probes: dict[tuple[str, tuple[str, ...]], Mapping[str, tuple[Facet, ...]]] = field(
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
    control_rows: int = 0
    control_agreed: int = 0
    control_failed: int = 0


def run_backfill_warnings(
    rows: Sequence[WarningRow],
    *,
    store: DatasetStore,
    readers: Readers,
    dry_run: bool,
    deadline: datetime,
    now: Clock,
) -> int:
    """Rewrite the member sentences of `rows` (`select_rows`), starting none once `deadline`
    has passed, emit one `datasets_warnings_backfilled` line, and return the number of rows
    that failed, lost the compare or were left. Each row that fails gets one
    `datasets_warning_backfill_failed` `ERROR` line saying whether it was `transient`, and each
    control row that fails one at `WARNING`."""
    counts = _Counts()
    for position, row in enumerate(rows):
        if now() >= deadline:
            counts.remaining = len(rows) - position
            break
        task = any(_member_sentence(warning) for warning in row.warnings)
        try:
            if task:
                _backfill_row(row, store=store, readers=readers, dry_run=dry_run, counts=counts)
            else:
                counts.control_rows += 1
                counts.control_agreed += _control_row(row, readers=readers)
        except Exception as exc:
            # One row that fails is left as it is; the run carries on with the others.
            transient = isinstance(exc, _Unread)
            if not task:
                counts.control_failed += 1
            elif transient:
                counts.transient += 1
            else:
                counts.permanent += 1
                counts.mismatched += isinstance(exc, _Mismatched)
            emit(
                "datasets_warning_backfill_failed",
                # A control row measures the graph and leaves the exit code alone.
                severity="ERROR" if task else "WARNING",
                subject_id=row.subject_id,
                content_version=row.content_version,
                control=not task,
                transient=transient,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
    return _emit_backfilled(counts, dry_run=dry_run, probes=readers.probe_calls)


def emit_nothing_backfilled(*, dry_run: bool) -> None:
    """The `datasets_warnings_backfilled` line of a run that found no row."""
    _emit_backfilled(_Counts(), dry_run=dry_run, probes=0)


def _emit_backfilled(counts: _Counts, *, dry_run: bool, probes: int) -> int:
    """Emit `counts`, the probes made and whether the run was dry, at `ERROR` when a row
    failed, lost the compare or was left, and return how many did. A control row that fails
    measures the graph and counts toward none of them."""
    unfinished = counts.transient + counts.permanent + counts.conflicts + counts.remaining
    emit(
        "datasets_warnings_backfilled",
        severity="ERROR" if unfinished else "INFO",
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
    row: WarningRow, *, store: DatasetStore, readers: Readers, dry_run: bool, counts: _Counts
) -> None:
    """Rewrite each of `row`'s warnings holding the old member sentence, and update the row
    unless `dry_run` or the stored warnings changed since they were read. Raises
    `_Mismatched` when the body's warnings differ from the column's."""
    if row.body_warnings != row.warnings:
        raise _Mismatched("immutable_body holds other warnings than the warnings column")
    warnings: list[dict[str, object]] = []
    splits: list[tuple[int, Split]] = []
    for warning in row.warnings:
        match = _member_sentence(warning)
        if match is None:
            warnings.append(warning)
            continue
        named = match[1].split(", ")
        count = len(named) + int(match[2] or 0)
        split = _split(row, named=named, count=count, readers=readers)
        splits.append((count, split))
        warnings.append(_rewritten(warning, match, split, member_count=_member_count(row)))
    if dry_run:
        for count, split in splits:
            emit(
                "datasets_warning_backfill_row",
                subject_id=row.subject_id,
                content_version=row.content_version,
                control=False,
                left_out=count,
                off_cadence=len(split.off_cadence),
                out_of_window=len(split.out_of_window),
                unserved_members=len(split.unserved_members),
                gone=split.gone,
                emptied=split.empty,
            )
    else:
        with store.transaction() as tx:
            updated = tx.fetch_one(
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
        if updated is None:
            counts.conflicts += 1
            return
    counts.rewritten += 1
    counts.gone += sum(split.gone for _count, split in splits)
    counts.emptied += sum(split.empty for _count, split in splits)


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


def _control_row(row: WarningRow, *, readers: Readers) -> bool:
    """Whether the split of each of `row`'s warnings a mint split equals the stored one, as
    one `datasets_warning_backfill_row` line per warning states."""
    agreed = True
    for warning in row.warnings:
        if warning.get("code") != MEMBER_SENTENCE_CODE or not any(
            key in warning for key in _MEMBER_KEYS
        ):
            continue
        stored = [
            cast("Mapping[str, object]", warning[key]) for key in _MEMBER_KEYS if key in warning
        ]
        named = [
            cast(str, ref) for counted in stored for ref in cast("list[object]", counted["refs"])
        ]
        count = sum(cast(int, counted["count"]) for counted in stored)
        split = _split(row, named=named, count=count, readers=readers)
        found = split.fields(member_count=_member_count(row))
        held = {key: warning[key] for key in ("member_count", *_MEMBER_KEYS) if key in warning}
        agrees = found == held
        agreed = agreed and agrees
        emit(
            "datasets_warning_backfill_row",
            subject_id=row.subject_id,
            content_version=row.content_version,
            control=True,
            agrees=agrees,
            stored=held,
            found=found,
            gone=split.gone,
        )
    return agreed


def _member_count(row: WarningRow) -> int:
    return len(_resolved(row))


def _resolved(row: WarningRow) -> list[Mapping[str, object]]:
    return cast("list[Mapping[str, object]]", row.trace.get("resolved_stat_vars") or [])


def _split(row: WarningRow, *, named: Sequence[str], count: int, readers: Readers) -> Split:
    """The `count` members a warning naming `named` leaves out, split as the module docstring
    says. Raises `_Refused` when the trace cannot hold them and `_Unread` when the probe does
    not complete."""
    zero = [entry for entry in _resolved(row) if entry.get("observation_count") == 0]
    zero_refs = [cast(str, entry["ref"]) for entry in zero]
    missing = [ref for ref in named if ref not in zero_refs]
    if missing:
        raise _Refused(f"named members hold observations in the trace or are not in it: {missing}")
    if count > len(zero_refs):
        raise _Refused(f"{count} members left out, {len(zero_refs)} hold no observation")
    candidates = _candidates(row, zero, readers=readers)
    selections = _selections(row.trace)
    window = identity_requested_window(row.identity)
    in_window = {
        ref: facets_in_window(candidates[ref], selections=selections, window=window)
        for ref in zero_refs
    }
    matched = [ref for ref in zero_refs if in_window[ref] is not None]
    left_out = list(dict.fromkeys([*named, *matched, *zero_refs]))[:count]
    served = dispatched_provenances(selections)
    absent_reads_zero = len(served) == 1 and absent_member_reads_zero(readers.code_index, served[0])
    return Split(
        off_cadence=tuple(ref for ref in left_out if in_window[ref] is True),
        out_of_window=()
        if absent_reads_zero
        else tuple(
            ref
            for ref in left_out
            if in_window[ref] is False or (in_window[ref] is None and not candidates[ref])
        ),
        unserved_members=tuple(
            ref for ref in left_out if in_window[ref] is None and candidates[ref]
        ),
        gone=sum(1 for ref in left_out if not candidates[ref]),
    )


def _selections(trace: Mapping[str, object]) -> dict[str, FacetSelection]:
    """The mint's selections, from `trace.facets` with the methods `trace.continuations`
    names on their axes."""
    continued_by = {
        cast(str, entry["axis"]): cast(str, entry["method"])
        for entry in cast("list[Mapping[str, object]]", trace.get("continuations") or [])
    }
    selections: dict[str, FacetSelection] = {}
    for entry in cast("list[Mapping[str, object]]", trace.get("facets") or []):
        axis = cast(str, entry["axis"])
        selections[axis] = FacetSelection.of_trace_entry(entry, continued_by=continued_by.get(axis))
    return selections


def _candidates(
    row: WarningRow, zero: Sequence[Mapping[str, object]], *, readers: Readers
) -> Mapping[str, tuple[Facet, ...]]:
    """Each of the `zero` members' facet candidates, probed as the mint probed them, or as a
    probe earlier in the run of the same subject and members found them."""
    key = (row.subject_id, tuple(cast(str, entry["ref"]) for entry in zero))
    held = readers.probes.get(key)
    if held is not None:
        return held
    readers.probe_calls += 1
    if composite_components(row.identity, code_index=readers.code_index) is not None:
        raise _Refused("a composite's members are probed under its component identities")
    requests = [
        to_materialisation_request(
            row.identity,
            shape=StatVarShape(
                stat_var=cast(str, entry["ref"]),
                declared_roles=tuple(cast("list[str]", entry["roles"])),
                slots=(),
                measured_property=None,
                population_type=None,
            ),
            code_index=readers.code_index,
        )
        for entry in zero
    ]
    try:
        with (
            definition_read_scope(row.identity, code_index=readers.code_index),
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
    except Exception as exc:
        raise _Unread(f"the probe failed: {type(exc).__name__}: {exc}") from exc
    if failures := un_record.failures():
        raise _Unread(f"a UN read failed beside ours: {failures[0].detail}")
    if unsettled := [r.stat_var for r, p in zip(requests, probes, strict=True) if not p.settled]:
        raise _Unread(f"the default place scope is not settled for {unsettled}")
    found = {
        request.stat_var: probe.candidates for request, probe in zip(requests, probes, strict=True)
    }
    readers.probes[key] = found
    return found
