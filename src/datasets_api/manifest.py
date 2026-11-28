"""Assembles manifests, records the facts each version stores, and assembles a stored
version's manifest from its facts at read time.

`assemble_manifest` combines registry records, materialisation results, artifacts, citations,
warnings, and output options. Matching `record_*_version` functions write version records, whose
facts `manifest_facts` builds (`dataset_store.snapshot.ManifestFacts`). Every read of a stored
version assembles its wire manifest from those facts in `stored_manifest`, so a wire change is a
read-side change.

A version stores its `OutputProfile` and leaves out artifacts and deployment-specific
`output_options`. Reads derive options from the stored profile and current capabilities
(`stored_output_profile`). A version without a profile returns `output_options=None`. A converted
definition also stores one conversion record per series (`materialise.series_conversion`), off the
wire and outside the content-version preimage, and `stored_series_conversions` reads them. A
definition also stores whether the mint's default place scope was plural
(`ManifestFacts.default_scope_plural`), which decides the stored version's preview mode. Definition
records include a canonical frame artifact that the response omits. Snapshot manifests list all
servable renderings. A version's `coverage` also sets its frame's one year beside the years
around it, counted from the coverage index its run elected over (`year_coverage`) and stored
with the version.

Each serve call reads the name, url, licence, source, Data Commons and documentation of every
dispatched provenance ref once (`materialise.provenance_doc.read_source_citations`), and every
manifest's citation is built from it. Every manifest carries the `prepared_at` its caller passes,
the instant of the version's first mint, which a re-mint reads back from the stored record.

A manifest's `diagnostics` are facts about how the response was produced, which a version does
not store. The graph has no processing-level convention, so `processing_level` is `None` and
each non-empty citation adds `processing_level_undeclared`. Every new manifest's trace records
the request's origin catalogue build as `origin_catalogue_build_id`, and a stale catalogue adds
`origin_catalogue_stale` (`datasets_api.origin`). What a mint chose and summed, and the
conversion terms it projected, are steps and caveats of the version's description record,
built from the trace (`datasets_api.description_record`), so a new manifest raises no
`facet_default_selected`, `stat_vars_summed` or `deflator_projected` warning. A stored manifest
serves the warnings it was minted with.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, cast

from dfo_frame_grammar import FRAME_GRAMMAR_VERSION
from dfo_frame_grammar.rolling import ANCHOR_SHARE

from code_index.query import CodeIndex
from dataset_store.snapshot import FrameLineage, ManifestFacts, StoredVersion
from datasets_api.admission import bound_roles
from datasets_api.capabilities import build_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.decode import statvar_code_systems
from datasets_api.description_record import mint_steps, minted_flow_family
from datasets_api.manifest_models import (
    ArtifactEntryModel,
    ChangeModel,
    CitationModel,
    CitationSource,
    CitedDataCommons,
    CitedDocumentation,
    CitedSourceNode,
    CoverageDimension,
    CoverageModel,
    DefinitionModel,
    DiagnosticModel,
    DimensionNoun,
    HeldBack,
    IncompleteAfter,
    Manifest,
    OutputOptionsModel,
    ReleaseModel,
    ReportedShare,
    TransformationStepModel,
    WarningModel,
    YearCoverage,
)
from datasets_api.origin import catalogue_build_id, stale_warnings
from datasets_api.output_options import derive_output_options
from datasets_api.party_totals import TRACE_KEY
from datasets_api.timings import phase
from datasets_api.versions import (
    NULL_VERSION_RECORD_WRITER,
    ArtifactRecord,
    VersionDescriber,
    VersionRecord,
    VersionRecordWriter,
    link_digest,
    unlinked_state,
)
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dc_graph import GraphReadError
from description.citation import citation_line, figures_processed
from dfo_platform.roles import FlowFamily, axis_for_role, role_wording
from materialise import MaterialisationRequest, MaterialisationResult
from materialise.bundle import BUNDLE_CODE_CATALOGUE
from materialise.cadence import cadence_label
from materialise.generation import bare_stat_var_ref
from materialise.graph import GraphReader
from materialise.held_back import HeldKind, Judgement, judge, last_year_before, source_of
from materialise.held_runs import HeldRun
from materialise.member_folds import dataset_name
from materialise.partial_period import reaches, start_of
from materialise.profile import OutputProfile
from materialise.provenance_doc import (
    CoverageFacts,
    ProvenanceDoc,
    SourceCitation,
    coverage_facts_from_index,
    read_source_citations,
)
from materialise.reported_share import ReportedShareFigure, YearTotals, reported_share
from materialise.series_conversion import (
    SeriesConversion,
    SeriesKey,
    decode_series_conversion_entries,
    encode_series_conversions,
)
from materialise.statistics import SeriesFigures
from materialise.trace import ManifestTrace
from materialise.window import Coverage, RequestedWindow
from origin_catalogue import Catalogue, CatalogueNotBound, current_catalogue
from series_coverage.coverage_types import YearCount
from series_coverage.index import CoverageIndex, Role
from spec_registry.reader import StoredDefinition

if TYPE_CHECKING:
    from datasets_api.render import DerivedVersion

logger = logging.getLogger(__name__)

_PROCESSING_LEVEL_UNDECLARED_TITLE, _PROCESSING_LEVEL_UNDECLARED_DETAIL = BUNDLE_CODE_CATALOGUE[
    "processing_level_undeclared"
]


def warning_model(warning: Mapping[str, object]) -> WarningModel:
    """Converts a complete `{code, severity, message}` warning mapping, with any structured
    fields it carries (`materialise.codes.MaterialisationWarning`), without reformatting it."""
    return WarningModel.model_validate(warning)


def reshape_warning_models(warnings: Sequence[Mapping[str, object]]) -> tuple[WarningModel, ...]:
    """Formats reshape warnings from this surface's catalogue.

    `enrichment_unavailable` adds its column and empty-cell count to the message.
    """
    models: list[WarningModel] = []
    for warning in warnings:
        code = cast(str, warning["code"])
        _title, detail, severity = DATASETS_WARNING_CATALOGUE[code]
        message = detail
        if code == "enrichment_unavailable":
            column = cast(str, warning["column"])
            count = cast(int, warning["count"])
            plural = "" if count == 1 else "s"
            message = f"{detail} Column {column!r} left {count} cell{plural} empty."
        models.append(WarningModel(code=code, severity=severity, message=message))
    return tuple(models)


def citation_diagnostics(citation: CitationModel) -> tuple[DiagnosticModel, ...]:
    """`processing_level_undeclared` when a cited source declares no processing level, which
    every source does, since the graph has no processing-level convention."""
    if not any(source.processing_level is None for source in citation.sources):
        return ()
    return (
        DiagnosticModel(
            code="processing_level_undeclared",
            severity="info",
            message=_PROCESSING_LEVEL_UNDECLARED_DETAIL,
        ),
    )


def _stale_diagnostics() -> tuple[DiagnosticModel, ...]:
    """`origin_catalogue_stale` when the catalogue bound to this request is past its staleness
    limit (`datasets_api.origin.stale_warnings`)."""
    return tuple(
        DiagnosticModel(
            code="origin_catalogue_stale", severity=warning.severity, message=warning.message
        )
        for warning in stale_warnings()
    )


def coverage_model(coverage: Coverage, *, family: FlowFamily | None) -> CoverageModel:
    """Converts materialisation coverage to the wire model. A dimension on a flow axis is
    named by the wording of `family` (`role_wording`), the generic one for `None`."""
    wording = role_wording(family)

    def _noun(label: str) -> DimensionNoun | None:
        axis = axis_for_role(label)
        if axis is None:
            return None
        noun = wording.noun(axis)
        return DimensionNoun(singular=noun.singular, plural=noun.plural)

    return CoverageModel(
        rows=coverage.rows,
        dimensions=[
            CoverageDimension(
                label=cast(str, dim["label"]),
                count=cast(int, dim["count"]),
                noun=_noun(cast(str, dim["label"])),
            )
            for dim in coverage.dimensions
        ],
        window=coverage.window,
    )


@dataclass(frozen=True)
class JudgedIncompleteAfter:
    """An `incomplete_after` record, `entries`, the `held_back` runs judged with it, and the
    `built_at` of the origin catalogue build that judged them, which orders the judgements of
    one version (`version_store._refresh_judged_record`). A dataset version's runs are its
    `coverage.held_back` (`judged_record`), and `None` states none, as a computed version's
    record does. A computed version's record carries the newest build among the judgements it
    derives from, `None` when one names none (`lineage_incomplete.derived_incomplete_after`)."""

    entries: list[dict[str, object]]
    built_at: datetime | None
    held_back: list[dict[str, object]] | None = None


def incomplete_after(result: MaterialisationResult) -> JudgedIncompleteAfter | None:
    """How complete the rows of the frame `result` holds are, judged per source against the
    origin catalogue bound to the mint, which joins the graph release the rows were read from
    (`IncompleteAfter`), with that catalogue's `built_at`. Every judgement is
    `materialise.held_back.judge`'s, the one the mint's cut makes, so a frame records only
    what its window opted in to. The source of a `(provenance, stat_var)` pair is the
    composite component serving the variable (`trace.components`), else the provenance itself
    (`held_back.source_of`).

    A pair holding projected rows is `projected` from the earliest period they start
    (`SeriesFigures.projected_by_source`). Its other rows are judged at the latest period they
    reach (`SeriesFigures.actual_by_source`): `partial` when it reaches the period its source's
    import holds only part of, `unknown` when the catalogue does not document its source
    (`Undocumented`), its rows name no provenance, or the frame grammar cannot read the mark or
    the period. Its actual rows are complete when its source's documentation states no
    `partialObservationDate`, whatever the source's origin, a UN System agency included. A
    complete pair is left out, so an empty list is judged and complete.

    A `partial` entry states how much of the year its mark starts in its source has reported,
    against the year before (`reported_share.reported_share`), from the year sums the read
    folded over the entry's variables before its window cut them
    (`MaterialisationResult.year_totals`). It states none when the read held no series in the
    year before.

    `None`, unknown as a whole, with no catalogue bound or when the catalogue judged no pair, so
    a stale catalogue's mint leaves a stored record standing (`version_store`)."""
    try:
        catalogue = current_catalogue()
    except CatalogueNotBound:
        return None
    component_of = {
        bare_stat_var_ref(cast(str, ref)): cast(str, component["provenance"])
        for component in result.trace.components
        if component.get("provenance") is not None
        for ref in cast("list[object]", component.get("stat_vars") or [])
    }
    figures = result.series_statistics.figures

    def judged(provenance: str | None, stat_var: str, _period: str) -> list[_Group | None]:
        source = (
            None
            if provenance is None
            else source_of(provenance, stat_var, component_of=component_of)
        )
        groups: list[_Group | None] = []
        if (start := figures.projected_by_source.get((provenance, stat_var))) is not None:
            groups.append((provenance, source, "projected", start))
        if (actual := figures.actual_by_source.get((provenance, stat_var))) is not None:
            groups.append(_judged(provenance, source, actual, catalogue=catalogue))
        return groups

    grouped = _grouped(figures, judged)
    known = any(key is None or key[2] in ("partial", "projected") for key in grouped)
    entries = _entries(grouped, shares=_reported_shares(grouped, result.year_totals))
    if entries and not known:
        return None
    return JudgedIncompleteAfter(entries=entries, built_at=catalogue.built_at)


def collapsed_incomplete_after(
    raw: Sequence[Mapping[str, object]] | None, figures: SeriesFigures
) -> list[dict[str, object]] | None:
    """The `incomplete_after` of a collapse, whose series reach the periods `figures` gives,
    of a version whose record is `raw` (a groupby or party-totals cut, `serve` and
    `current`). A collapse keeps each row's `provenance` cell and period; a party-totals
    cut keeps its `stat_var` cell too, and a sum names its rows by the definition's spec id. A
    pair whose variable `raw` names under its provenance takes those entries; any other pair is
    the sum of its provenance's raw cells, so it is partial from the earliest partial period
    among its provenance's raw entries that its rows reach (by start, ties broken by source),
    else unknown when its provenance holds an unknown cell, else complete, and projected as
    well from the earliest projected period among its provenance's raw entries that its rows
    reach. `None` when `raw` is.

    An entry keeps the `reported_share` of the raw entry it takes when its basis is `value`:
    the summed values of the collapse's rows are the raw rows' sums. A `series` share is left
    out, since a collapse holds other series than its base and its stored rows need not hold
    the year the share is counted against.
    """
    if raw is None:
        return None
    by_provenance: dict[object, list[Mapping[str, object]]] = {}
    for entry in raw:
        by_provenance.setdefault(entry.get("provenance"), []).append(entry)
    shares: dict[_Group, Mapping[str, object] | None] = {}

    def judged(provenance: str | None, stat_var: str, period: str) -> list[_Group | None]:
        held = by_provenance.get(provenance, [])
        named = [
            entry
            for entry in held
            if bare_stat_var_ref(stat_var)
            in {bare_stat_var_ref(one) for one in cast("list[str]", entry["stat_vars"])}
        ]
        taken = named or _summed_entries(held, period=period)
        groups: list[_Group | None] = []
        for entry in taken:
            group = (
                provenance,
                cast("str | None", entry.get("source")),
                cast(str, entry["status"]),
                cast("str | None", entry.get("partial_period")),
            )
            shares.setdefault(group, _value_share(entry))
            groups.append(group)
        return groups or [None]

    grouped = _grouped(figures, judged)
    return _entries(grouped, shares=shares)


def _summed_entries(
    held: Sequence[Mapping[str, object]], *, period: str
) -> list[Mapping[str, object]]:
    """The entries of `held` (one provenance's raw `incomplete_after` entries) that a pair
    summing every cell takes when its rows reach `period` (`collapsed_incomplete_after`).
    It takes the earliest partial entry `period` reaches, or else an unknown one, plus the
    earliest projected entry `period` reaches."""
    marked = [entry for entry in held if entry.get("status") == "partial"]
    reached = [reaches(period, cast(str, entry["partial_period"])) for entry in marked]
    partial = [entry for entry, hit in zip(marked, reached, strict=True) if hit]
    unknown = [entry for entry in held if entry.get("status") == "unknown"] + [
        {**entry, "status": "unknown", "partial_period": None}
        for entry, hit in zip(marked, reached, strict=True)
        if hit is None
    ]
    projected = [
        entry
        for entry in held
        if entry.get("status") == "projected"
        and reaches(period, cast(str, entry["partial_period"]))
    ]
    taken: list[Mapping[str, object]] = []
    if partial:
        taken.append(min(partial, key=_by_period_and_source))
    elif unknown:
        taken.append(unknown[0])
    if projected:
        taken.append(min(projected, key=_by_period_and_source))
    return taken


def _by_period_and_source(entry: Mapping[str, object]) -> tuple[str, str]:
    return start_of(cast(str, entry["partial_period"])), cast("str | None", entry["source"]) or ""


def _value_share(entry: Mapping[str, object]) -> Mapping[str, object] | None:
    """`entry`'s `reported_share` when its basis is `value`, which a collapse keeps."""
    share = entry.get("reported_share")
    if not isinstance(share, Mapping):
        return None
    held = cast("Mapping[str, object]", share)
    return held if held.get("basis") == "value" else None


type _Group = tuple[str | None, str | None, str, str | None]
"""An `incomplete_after` entry's provenance, source, status and partial period."""


def _grouped(
    figures: SeriesFigures,
    judge: Callable[[str | None, str, str], Sequence[_Group | None]],
) -> dict[_Group | None, list[str]]:
    """The variables of the `(provenance, stat_var)` pairs of `figures` under each group `judge`
    gives them from their latest period (`SeriesFigures.latest_by_source`), `None` for a
    complete part of a pair."""
    grouped: dict[_Group | None, list[str]] = {}
    for (provenance, stat_var), period in sorted(
        figures.latest_by_source.items(),
        key=lambda item: (item[0][0] or "", item[0][1]),
    ):
        for group in judge(provenance, stat_var, period):
            stat_vars = grouped.setdefault(group, [])
            if stat_var not in stat_vars:
                stat_vars.append(stat_var)
    return grouped


def _entries(
    grouped: Mapping[_Group | None, list[str]],
    *,
    shares: Mapping[_Group, Mapping[str, object] | None] | None = None,
) -> list[dict[str, object]]:
    """The `IncompleteAfter` entries of `grouped`'s partial, projected and unknown groups,
    each with its `reported_share` from `shares` where it holds one."""
    fields = ("provenance", "source", "status", "partial_period")
    shares = shares or {}
    return [
        IncompleteAfter.model_validate(
            {
                **dict(zip(fields, group, strict=True)),
                "stat_vars": stat_vars,
                "reported_share": shares.get(group),
            }
        ).model_dump()
        for group, stat_vars in grouped.items()
        if group is not None
    ]


def _reported_shares(
    grouped: Mapping[_Group | None, list[str]],
    year_totals: Mapping[tuple[str | None, str], YearTotals],
) -> dict[_Group, Mapping[str, object] | None]:
    """Each partial group's `reported_share`, read over its variables' year sums at the year
    its mark starts in. A mark the frame grammar cannot read states none."""
    shares: dict[_Group, Mapping[str, object] | None] = {}
    for group, stat_vars in grouped.items():
        if group is None or group[2] != "partial":
            continue
        against = last_year_before(Judgement(status="partial", partial_period=group[3]))
        if against is None:
            continue
        totals = [held for stat_var in stat_vars if (held := year_totals.get((group[0], stat_var)))]
        shares[group] = _share_entry(reported_share(totals, year=against + 1))
    return shares


def _share_entry(figure: ReportedShareFigure | None) -> Mapping[str, object] | None:
    if figure is None:
        return None
    return {"basis": figure.basis, "share": figure.share, "against": figure.against}


def _judged(
    provenance: str | None, source: str | None, period: str, *, catalogue: Catalogue
) -> _Group | None:
    """The `(provenance, source, status, partial period)` group of the actual rows of one pair
    of `incomplete_after`, whose latest is `period` (`held_back.judge`), `None` when complete."""
    if provenance is None:
        return None, None, "unknown", None
    judgement = judge(source=source, method=None, period=period, catalogue=catalogue)
    if judgement is None:
        return None
    return provenance, source, judgement.status, judgement.partial_period


# The calendar years `coverage.fullest_year` is the fullest of, the frame's newest included.
_FULLEST_YEAR_SPAN = 10


def held_back_coverage(
    result: MaterialisationResult, *, requests: Sequence[MaterialisationRequest]
) -> list[dict[str, object]]:
    """The `held_back` runs of the frame `result` holds (`manifest_models.HeldBack`): every run
    of partial and projected years the mint's read found its sources hold past their served
    actual years, whatever the requests' window (`MaterialisationResult.held_runs`), by the
    rules Search states them by (`materialise.held_runs`). A run is `served` when the window's
    `include` names its kind and the window holds a year of it.

    A partial run states the share of its first year its source has reported against the year
    before, from the year sums the mint folded over the source's rows of the two years before
    its window and held-back cuts (`MaterialisationResult.year_totals`,
    `reported_share.reported_share`), whatever the window opts in to. It states none where the
    read held no series of the year before.

    Empty with no catalogue bound, which a mint refuses before it gets here."""
    if not requests or not result.held_runs:
        return []
    try:
        catalogue = current_catalogue()
    except CatalogueNotBound:
        return []
    window = requests[0].window
    return [
        _stated_run(
            result,
            run=run,
            served=run.kind in window.include and _window_holds(window, run),
            catalogue=catalogue,
        )
        for run in result.held_runs
    ]


def _stated_run(
    result: MaterialisationResult,
    *,
    run: HeldRun,
    served: bool,
    catalogue: Catalogue | None = None,
) -> dict[str, object]:
    """`run` as the frame `result` holds states it (`manifest_models.HeldBack`): `catalogue`,
    by default the one bound, names its source (`materialise.member_folds.dataset_name`, the
    name a declared dataset's members share, else its declared label), and a partial run
    carries its reported share (`_reported_share`)."""
    catalogue = catalogue or current_catalogue()
    share = (
        _reported_share(result, run=run, component_of=_component_of(result))
        if run.kind == "partial"
        else None
    )
    return HeldBack(
        kind=run.kind,
        served=served,
        source=run.source,
        source_name=dataset_name(run.source, catalogue=catalogue),
        start_year=run.start_year,
        end_year=run.end_year,
        reported_share=None if share is None else ReportedShare.model_validate(share),
    ).model_dump(mode="json")


def _window_holds(window: RequestedWindow, run: HeldRun) -> bool:
    """Whether a read of `window` holds a year of `run`: every date, `latest` (which serves
    each series' newest year, a held run's when the window opts in to it), or a range
    sharing a year with it."""
    if window.kind != "range":
        return True
    return (window.end_year is None or window.end_year >= run.start_year) and (
        window.start_year is None or window.start_year <= run.end_year
    )


def _component_of(result: MaterialisationResult) -> dict[str, str]:
    """Each member variable of a composite's frame, bare, mapped to the provenance of the
    component serving it (`trace.components`). Empty for any other frame."""
    return {
        bare_stat_var_ref(cast(str, ref)): cast(str, component["provenance"])
        for component in result.trace.components
        if component.get("provenance") is not None
        for ref in cast("list[object]", component.get("stat_vars") or [])
    }


def _reported_share(
    result: MaterialisationResult,
    *,
    run: HeldRun,
    component_of: Mapping[str, str],
) -> Mapping[str, object] | None:
    """The share of a partial `run`'s first year reported against the year before, from the
    year sums the mint folded over the rows of `run`'s source (`held_back_coverage`)."""
    totals = [
        held
        for (provenance, stat_var), held in result.year_totals.items()
        if provenance is not None
        and source_of(provenance, stat_var, component_of=component_of) == run.source
    ]
    return _share_entry(reported_share(totals, year=run.start_year)) if totals else None


def judged_record(
    result: MaterialisationResult, *, requests: Sequence[MaterialisationRequest]
) -> JudgedIncompleteAfter | None:
    """The judged record of the frame `result` holds: its `incomplete_after`
    (`incomplete_after`) with its `held_back` runs (`held_back_coverage`), judged under one
    catalogue build and stored together (`version_store._refresh_judged_record`). Every
    partial or projected entry lies in a served run of its kind and source
    (`_reconciled_runs`), so a reader of the runs alone finds every year the frame serves held
    back. `None` when `incomplete_after` judged nothing, so a stale catalogue's mint leaves
    both stored members standing."""
    judged = incomplete_after(result)
    if judged is None:
        return None
    runs = _reconciled_runs(
        held_back_coverage(result, requests=requests), judged.entries, result=result
    )
    return replace(judged, held_back=runs)


def _reconciled_runs(
    runs: Sequence[Mapping[str, object]],
    entries: Sequence[Mapping[str, object]],
    *,
    result: MaterialisationResult,
) -> list[dict[str, object]]:
    """`runs`, the frame's `held_back`, with every partial or projected entry of `entries`,
    its `incomplete_after`, inside a served run of the entry's kind and source. An entry
    names rows the frame serves, so its run is served: a run of its kind and source is marked
    served and widened to the year its `partial_period` starts in, and a missing one is added
    from that year to the newest year the entry's rows reach
    (`SeriesFigures.latest_by_source`), a partial one with its reported share
    (`_reported_share`). Partial runs come first, then by source, as `merged_runs` orders
    them."""
    by_key = {(run["kind"], run["source"]): dict(run) for run in runs}
    latest = result.series_statistics.figures.latest_by_source
    for entry in entries:
        kind, source = entry.get("status"), entry.get("source")
        year = _year_of(entry.get("partial_period"))
        if kind not in ("partial", "projected") or not isinstance(source, str) or year is None:
            continue
        known = by_key.get((kind, source))
        if known is None:
            provenance = cast("str | None", entry.get("provenance"))
            reached = [
                _year_of(latest.get((provenance, cast(str, stat_var))))
                for stat_var in cast("list[object]", entry.get("stat_vars") or [])
            ]
            start, end = year, max([year, *(one for one in reached if one is not None)])
        else:
            start = min(cast(int, known["start_year"]), year)
            end = max(cast(int, known["end_year"]), year)
            if known["served"] and (start, end) == (known["start_year"], known["end_year"]):
                continue
        run = HeldRun(kind=cast("HeldKind", kind), source=source, start_year=start, end_year=end)
        by_key[(kind, source)] = _stated_run(result, run=run, served=True)
    return [by_key[key] for key in sorted(by_key, key=lambda k: (k[0] != "partial", str(k[1])))]


def _year_of(period: object) -> int | None:
    """The calendar year a `time_period` cell or mark starts in, `None` when the frame
    grammar cannot read it."""
    if not isinstance(period, str):
        return None
    try:
        return int(start_of(period)[:4])
    except ValueError:
        return None


def year_coverage(
    result: MaterialisationResult,
    *,
    requests: Sequence[MaterialisationRequest],
    index: CoverageIndex | None,
    judged: JudgedIncompleteAfter | None,
) -> dict[str, object]:
    """The members of the `coverage` of the frame `result` holds that its mint counts once
    and stores with the version: the year members (`year_members`) and the `held_back` runs
    of `judged`, its judged record (`judged_record`), each absent when not stated. A mint
    whose catalogue judged nothing (`judged` None) states the runs its read found
    (`held_back_coverage`), projected runs included, which need no catalogue. A version
    already on record keeps its stored runs over these (`serve._base_record`)."""
    members = year_members(result, requests=requests, index=index)
    runs = held_back_coverage(result, requests=requests) if judged is None else judged.held_back
    if runs:
        members["held_back"] = [dict(run) for run in runs]
    return members


def with_held_back(
    coverage: Mapping[str, object], held: Sequence[Mapping[str, object]] | None
) -> Mapping[str, object]:
    """`coverage`, a version's stored `coverage`, stating the `held_back` runs `held` in place
    of its own: without the member when `held` is empty, and as it stands when `held` is None,
    a write that stated no runs."""
    if held is None:
        return coverage
    refreshed = {key: value for key, value in coverage.items() if key != "held_back"}
    if held:
        refreshed["held_back"] = [dict(run) for run in held]
    return refreshed


def year_members(
    result: MaterialisationResult,
    *,
    requests: Sequence[MaterialisationRequest],
    index: CoverageIndex | None,
) -> dict[str, object]:
    """The `recent_years`, `elected_year` and `fullest_year` of the `coverage` of the frame
    `result` holds, as `index`, the one its run elected over, counts them over the series
    `requests` read. They are the members that set the frame's one year beside the years
    around it, counted once at the mint and stored with the version. Empty when they are
    not stated.

    They are stated for a frame of one year whose requests ask for `latest` or for one
    named year, and whose StatVars the index holds under one pair of role names. The counts
    cover the series each request's run read (`MaterialisationResult.read_series`): those of
    its StatVar's selected facets, less the entities a conversion omitted.

    Under `latest` each StatVar must have served the year it elected
    (`MaterialisationResult.latest_years`), since the index's counts otherwise describe
    other rows than the frame's. A named year states the members a `latest` frame of the
    requests would: the StatVars must elect one year between them over the series read
    (`CoverageIndex.elected_years`), at or before the named year, which the index's build
    year must not precede. An earlier named year states nothing, and neither does a StatVar
    the index elects no year for, as one of a facet finer than a year.

    `recent_years` is `CoverageIndex.headline`'s three calendar years, newest first, up to
    the index's build year, and `elected_year` the newest year a request's StatVar elected.
    `fullest_year` counts the year holding the most series of the ten ending at the newest
    of `recent_years`, the newest of them on a tie, and is `None` unless that year is
    earlier than the three and the frame's year holds under `ANCHOR_SHARE` of it. When one
    of the three is the fullest, `elected_year` already names a year holding at least that
    share.

    Every count leaves out the years the requests' window holds back unless it opts in to
    them (`RequestedWindow.include`), as the read's own election does, so a `latest` frame's
    `elected_year` is the year it serves.

    Each `role_counts` is keyed by the role's graph name, the `label` of the role's entry in
    `coverage.dimensions`.
    """
    window = result.coverage.window or {}
    year = window.get("start_year")
    asked = requests[0].window
    include = asked.include
    latest = asked.kind == "latest"
    named = asked.kind == "range" and asked.start_year == asked.end_year and not asked.end_unstated
    if (
        index is None
        or not isinstance(year, int)
        or window.get("end_year") != year
        or not (latest or named)
    ):
        return {}
    variables = [bare_stat_var_ref(request.stat_var) for request in requests]
    names = index.role_names.get(variables[0])
    bound = bound_roles(requests[0], index=index)
    read = result.read_series
    if names is None or any(
        index.role_names.get(variable) != names
        or set(names.declared) != set(request.declared_roles)
        or bound_roles(request, index=index) != bound
        or request.stat_var not in read
        for variable, request in zip(variables, requests, strict=True)
    ):
        return {}
    labels: dict[Role, str | None] = {"donor": names.donor, "recipient": names.recipient}
    coverage_role = {label: role for role, label in labels.items()}
    served = [
        (variable, facet_id, None, None)
        for variable, request in zip(variables, requests, strict=True)
        for facet_id in read[request.stat_var].facet_ids
    ]
    excluded = [
        (variable, coverage_role[role], entity)
        for variable, request in zip(variables, requests, strict=True)
        for role, entity in read[request.stat_var].omitted
        if role in coverage_role
    ]
    elected: dict[str, int] = {}
    if latest:
        for variable, request in zip(variables, requests, strict=True):
            recorded = result.latest_years.get(request.stat_var)
            if recorded is None or recorded.elected is None or recorded.served != recorded.elected:
                return {}
            elected[variable] = recorded.elected
    else:
        elected = index.elected_years(
            variables=variables, bound=bound, served=served, excluded=excluded, include=include
        )
        if (
            any(variable not in elected for variable in variables)
            or len(set(elected.values())) > 1
            or year < elected[variables[0]]
            or (index.build_year is not None and year > index.build_year)
        ):
            return {}
    counted = functools.partial(
        index.headline,
        variables=variables,
        bound=bound,
        served=served,
        excluded=excluded,
        include=include,
    )
    headline = counted(elected=elected)
    recent = headline.recent_years if headline is not None else ()
    held = next((entry for entry in recent if entry.year == year), None)
    if held is None:
        return {}

    def _entry(count: YearCount) -> YearCoverage:
        return YearCoverage(
            year=count.year,
            series=count.series,
            role_counts={
                label: count.role_counts[role]
                for role, label in labels.items()
                if label is not None and role in count.role_counts
            },
        )

    fullest: YearCount | None = None
    newest = recent[0].year
    holding = index.series_by_year(
        variables=variables,
        bound=bound,
        served=served,
        excluded=excluded,
        first_year=newest - _FULLEST_YEAR_SPAN + 1,
        last_year=newest,
        include=include,
    )
    fullest_year = max(holding, key=lambda year: (holding[year], year))
    if fullest_year < recent[-1].year and held.series < ANCHOR_SHARE * holding[fullest_year]:
        at_fullest = counted(first_year=fullest_year, last_year=fullest_year)
        fullest = at_fullest.recent_years[0] if at_fullest is not None else None
    return {
        "recent_years": [_entry(count) for count in recent],
        "elected_year": max(elected.values()),
        "fullest_year": _entry(fullest) if fullest is not None else None,
    }


def cited_documentation(doc: ProvenanceDoc) -> CitedDocumentation:
    """The wire snapshot of a source's documentation. `attribution` is the document's
    `citation` (`licenseAttribution`) and `update_cadence` the display label of its
    `sourceReleaseFrequency` token (`materialise.cadence.cadence_label`)."""
    return CitedDocumentation(
        publisher=doc.publisher,
        description=doc.description,
        source_data_url=doc.source_data_url,
        attribution=doc.citation,
        update_cadence=cadence_label(doc.update_cadence),
        last_update=doc.last_update,
        years_covered=doc.years_covered,
        granularity=doc.granularity,
        currencies=list(doc.currencies),
        transformation_log=[
            TransformationStepModel(transformation=step.title, description=step.description)
            for step in doc.transformation_log
        ],
        version_history=[
            ReleaseModel(
                date=release.date.isoformat(),
                title=release.title,
                changes=[
                    ChangeModel(kind=change.kind.value, note=change.note)
                    for change in release.changes
                ],
            )
            for release in doc.version_history
        ],
    )


def cited_source(
    ref: str, *, processing_level: str | None, citation: SourceCitation | None
) -> CitationSource:
    """The `CitationSource` a mint stores for `ref` at `processing_level` from its live citation
    (`read_source_citations` output), or from `ref` alone when there is none."""
    if citation is None:
        return CitationSource(ref=ref, label=ref, processing_level=processing_level)
    return CitationSource(
        ref=ref,
        label=citation.name or ref,
        processing_level=processing_level,
        license=citation.licence,
        license_type=citation.licence_type,
        license_url=citation.licence_url,
        url=citation.url,
        source=(
            CitedSourceNode(ref=citation.source.ref, label=citation.source.label)
            if citation.source is not None
            else None
        ),
        data_commons=(
            CitedDataCommons(id=citation.data_commons.id, label=citation.data_commons.label)
            if citation.data_commons is not None
            else None
        ),
        documentation=(
            cited_documentation(citation.documentation)
            if citation.documentation is not None
            else None
        ),
    )


def build_citation(
    *,
    provenance_refs: Sequence[str],
    source_citations: Mapping[str, SourceCitation],
    retrieved: date,
    processed: bool,
    processing_levels: Sequence[str | None] | None = None,
) -> tuple[CitationModel, tuple[DiagnosticModel, ...]]:
    """Builds one source per distinct `(ref, processing_level)` in first-seen order, with its
    diagnostics (`citation_diagnostics`).

    Levels parallel `provenance_refs` and default to `None`. `text` is
    `description.citation.citation_line` over the distinct source labels, the `retrieved` date
    and whether the DFO changed the version's figures (`processed`, from
    `description.citation.figures_processed`). `source_citations` is
    `read_source_citations` output for at least `provenance_refs`. `label` falls back to the
    ref, and the licence fields, `url`, `source`, `data_commons` and `documentation` stay
    `None` where the citation has none.
    """
    levels = processing_levels if processing_levels is not None else [None] * len(provenance_refs)
    sources: list[CitationSource] = []
    seen: set[tuple[str, str | None]] = set()
    for ref, level in zip(provenance_refs, levels, strict=True):
        key = (ref, level)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            cited_source(ref, processing_level=level, citation=source_citations.get(ref))
        )
    citation = CitationModel(
        text=citation_line(retrieved=retrieved, sources=_labels(sources), processed=processed),
        sources=sources,
    )
    return citation, citation_diagnostics(citation)


def _labels(sources: Sequence[CitationSource]) -> list[str]:
    """The label of each distinct ref among `sources`, in first-seen order."""
    return list({source.ref: source.label for source in sources}.values())


def minted_figures_processed(
    *, spec_id: str, trace: Mapping[str, object], definition: Mapping[str, object]
) -> bool:
    """Whether the DFO changed the figures of a minted version of `spec_id` with `trace` and
    `definition` (`description.citation.figures_processed`), read from the steps its
    description record holds (`datasets_api.description_record.mint_steps`). Steps that
    cannot be built count as unknown, as they do for the description, so the mint goes on
    and cites the route of a version whose steps are unknown."""
    try:
        steps = mint_steps(trace=trace, definition=definition)
    except Exception:
        logger.warning(
            "manifest citation route: steps unbuildable, citing unknown steps spec_id=%s",
            spec_id,
            exc_info=True,
        )
        steps = []
    return figures_processed(version_id=spec_id, steps=steps)


def collapsed_citation(citation: CitationModel, *, base: Manifest) -> CitationModel:
    """The citation of a multi-contributor collapse of the version `base` describes, whose
    citation, as recorded or as this mint assembles it, is `citation`: its sources and
    `base`'s retrieval date, with the line rewritten for the collapse's steps, which sum the
    version's variables."""
    text = citation_line(
        retrieved=base.prepared_at.astimezone(UTC).date(),
        sources=_labels(citation.sources),
        # A collapse always ends in a `sum_variables` step (`collapse_steps`), which
        # changes figures.
        processed=True,
    )
    return citation.model_copy(update={"text": text})


def coverage_facts_of(
    coverage_index: CoverageIndex | None,
) -> Callable[[tuple[str, ...]], CoverageFacts] | None:
    """The currencies-and-granularity lookup a source's documentation reads from this mint's
    coverage snapshot, `None` without one."""
    if coverage_index is None:
        return None
    return functools.partial(coverage_facts_from_index, coverage_index)


HEALED_FIELDS = tuple(
    name for name in CitationSource.model_fields if name not in {"ref", "label", "processing_level"}
)
"""Every `CitationSource` field a live read can supply. `ref`, `label` and `processing_level`
record what the mint cited and keep their stored values."""


def _needs_healing(source: CitationSource) -> bool:
    return any(getattr(source, name) is None for name in HEALED_FIELDS)


def healed_citation(
    citation: CitationModel,
    *,
    graph: GraphReader,
    deadline: float | None,
    coverage_index: CoverageIndex | None,
) -> tuple[CitationModel, bool]:
    """A stored citation as a bundle or card export cites it, and whether every read answered.

    A source with a `HEALED_FIELDS` member that is `None` (a body stored before that field
    existed, a first mint whose documentation read failed, or a source the catalogue and graph
    declare nothing for) is cited again with its documentation (`read_source_citations`: from
    the origin catalogue with no graph read for a source it lists, else from the graph within
    `deadline`), turned into a `CitationSource` by `cited_source` as a first mint does, and only
    its `None` fields are filled from that. Every set field stands, so a source with every field
    set reads nothing. The stored body is left as it is. The second value is `False` when a
    graph read raised `GraphReadError` or a documentation read failed. A source nothing
    documents counts as answered. `GraphCapExceededError` propagates.
    """
    refs = list(dict.fromkeys(source.ref for source in citation.sources if _needs_healing(source)))
    if not refs:
        return citation, True
    try:
        with phase("citation"):
            live = read_source_citations(
                graph,
                refs,
                catalogue=current_catalogue(),
                deadline=deadline,
                documented=True,
                coverage_facts_of=coverage_facts_of(coverage_index),
            )
    except GraphReadError:
        logger.warning(
            "Citation read failed for refs=%s; rendering the stored citation", refs, exc_info=True
        )
        return citation, False
    complete = not any(live[ref].documentation_read_failed for ref in refs if ref in live)
    return citation_healed_from(citation, live), complete


def citation_healed_from(
    citation: CitationModel, live: Mapping[str, SourceCitation]
) -> CitationModel:
    """`citation` with the `None` `HEALED_FIELDS` of each source filled from its `live` citation
    (`read_source_citations` output), turned into a `CitationSource` by `cited_source` as a
    first mint does. Every set field stands, and a source `live` does not hold is kept as
    stored."""
    sources: list[CitationSource] = []
    for stored in citation.sources:
        read = live.get(stored.ref) if _needs_healing(stored) else None
        if read is None:
            sources.append(stored)
            continue
        cited = cited_source(stored.ref, processing_level=stored.processing_level, citation=read)
        sources.append(
            stored.model_copy(
                update={
                    name: getattr(cited, name)
                    for name in HEALED_FIELDS
                    if getattr(stored, name) is None
                }
            )
        )
    return citation.model_copy(update={"sources": sources})


def _code_index_stamp_entry(code_index: CodeIndex) -> dict[str, object]:
    """Returns the three code-index stamp fields carried in the trace."""
    return {
        "content_hash": code_index.stamp.content_hash,
        "exported_at": code_index.stamp.exported_at,
        "graph_endpoint": code_index.stamp.graph_endpoint,
    }


def assemble_manifest(
    *,
    stored: StoredDefinition,
    materialisation: MaterialisationResult,
    artifacts: Sequence[ArtifactEntryModel],
    provenance_refs: Sequence[str],
    source_citations: Mapping[str, SourceCitation],
    prepared_at: datetime,
    capability_version: str,
    materialiser_build: str,
    code_index: CodeIndex,
    settings: DatasetsBuildSettings,
    extra_warnings: Sequence[WarningModel] = (),
    coverage_stamp: str | None = None,
    year_coverage: Mapping[str, object] | None = None,
) -> Manifest:
    """Builds a definition manifest. `definition_version_record` encodes it for storage.

    `year_coverage` holds the year members of its `coverage` (`year_coverage`), counted by
    the caller that ran the materialisation; `None` states none.

    Uses the caller's `CodeIndex` for rendering and the trace stamp. `coverage_stamp` is
    admission-time lineage, excluded from `content_version`. `None` leaves it unset.
    `prepared_at` is the version's first-mint instant. `provenance_refs` are the sources the
    citation names (`source_refs_of`): a composite's component tables.

    `warnings` holds the materialisation's warnings, then `extra_warnings`. `diagnostics` holds
    the citation's (`citation_diagnostics`), then a stale origin catalogue's. A choice of source,
    a sum and a projected conversion term are no warning: the version's description record
    states them from the trace (`datasets_api.description_record`).
    """
    warnings = [warning_model(w) for w in materialisation.warnings] + list(extra_warnings)

    trace = replace(
        materialisation.trace,
        code_index_stamp=_code_index_stamp_entry(code_index),
        origin_catalogue_build_id=catalogue_build_id(),
    ).to_dict()
    definition = DefinitionModel(
        schema_version=stored.schema_version, spec_id=stored.spec_id, identity=stored.identity
    )
    citation, citation_diagnostics_ = build_citation(
        provenance_refs=provenance_refs,
        source_citations=source_citations,
        retrieved=prepared_at.astimezone(UTC).date(),
        processed=minted_figures_processed(
            spec_id=stored.spec_id,
            trace=trace,
            definition=definition.model_dump(mode="json"),
        ),
    )

    return Manifest(
        definition=definition,
        content_version=materialisation.content_version,
        prepared_at=prepared_at,
        coverage=CoverageModel.model_validate(
            coverage_model(
                materialisation.coverage,
                family=minted_flow_family(stored.identity, trace, code_index=code_index),
            ).model_dump(mode="json")
            | dict(year_coverage or {})
        ),
        citation=citation,
        trace=trace,
        warnings=warnings,
        diagnostics=[*citation_diagnostics_, *_stale_diagnostics()],
        artifacts=list(artifacts),
        # The statistics artifact is in `artifacts`. This field carries only its count.
        series_count=materialisation.series_statistics.series_count,
        output_options=manifest_output_options(
            materialisation,
            stored=stored,
            capability_version=capability_version,
            materialiser_build=materialiser_build,
            code_index=code_index,
            settings=settings,
        ),
        series_coverage_stamp=coverage_stamp,
    )


def manifest_output_options(
    materialisation: MaterialisationResult,
    *,
    stored: StoredDefinition,
    capability_version: str,
    materialiser_build: str,
    code_index: CodeIndex,
    settings: DatasetsBuildSettings,
) -> OutputOptionsModel:
    """The output options a manifest of `materialisation` offers (`derive_output_options`)."""
    capabilities = build_capabilities(
        capability_version=capability_version,
        build=materialiser_build,
        code_index_stamp=code_index.stamp,
        declarations=code_index.declarations,
        settings=settings,
    )
    return derive_output_options(
        materialisation.output_profile,
        identity=stored.identity,
        capabilities=capabilities,
        parties=code_index.parties,
        datasets=source_refs_of(materialisation.trace),
        code_systems=frame_code_systems(materialisation.trace, index=code_index),
    )


def manifest_facts(
    manifest: Manifest,
    *,
    output_profile: OutputProfile | None = None,
    series_conversions: Mapping[SeriesKey, SeriesConversion] | None = None,
    default_scope: Sequence[str] | None = None,
    incomplete_after: Sequence[Mapping[str, object]] | None = None,
    incomplete_after_built_at: datetime | None = None,
) -> ManifestFacts:
    """The facts a definition version stores for `manifest`: every manifest member but
    `artifacts`, `output_options` and `content_version` (the record's own column), with
    `output_profile` in place of options. The citation is stored as
    `CitationModel.stored_json` gives it.

    `series_conversions`, when non-empty, stores one entry per converted series
    (`materialise.series_conversion.encode_series_conversions`), and an unconverted version
    stores `None`. `default_scope`, the places of the mint's default place scope
    (`MaterialisationResult.default_scope`), stores whether it holds more than one place, so
    the worldwide total alone and a bound place store `False`; `None` stores `None`.
    `incomplete_after` and `incomplete_after_built_at` are stored as given
    (`ManifestFacts.incomplete_after`, `ManifestFacts.incomplete_after_built_at`).
    """
    dumped = manifest.model_dump(
        mode="json", include={"definition", "coverage", "trace", "warnings"}
    )
    return ManifestFacts(
        prepared_at=manifest.prepared_at,
        definition=dumped["definition"],
        coverage=dumped["coverage"],
        citation=manifest.citation.stored_json(),
        trace=dumped["trace"],
        warnings=dumped["warnings"],
        series_count=manifest.series_count,
        series_coverage_stamp=manifest.series_coverage_stamp,
        output_profile=output_profile.to_json() if output_profile is not None else None,
        series_conversions=(
            encode_series_conversions(series_conversions) if series_conversions else None
        ),
        default_scope_plural=len(default_scope) > 1 if default_scope is not None else None,
        incomplete_after=None if incomplete_after is None else list(incomplete_after),
        incomplete_after_built_at=incomplete_after_built_at,
    )


def provenance_refs_of(facets: Iterable[Mapping[str, object]]) -> list[str]:
    """The provenance refs live or stored `trace.facets` entries select: the one a
    materialisation dispatched, and none for one whose facet probe was empty
    (`MaterialisationResult.probe_empty`), which selects no facet."""
    return [cast(str, facet["ref"]) for facet in facets if facet["axis"] == "provenance"]


def source_refs_of(trace: ManifestTrace | Mapping[str, object]) -> list[str]:
    """The provenance refs a version's figures come from, in first-seen order: for a composite
    each component's (`trace.components`), whose tables its citation names, and otherwise the
    dataset its rows carry (`provenance_refs_of`), a plain table's provenance or a declared
    dataset's id, which every row of its member provenances carries. A live `ManifestTrace` or
    a stored or wire trace mapping. The citation cites these, and the parties of their datasets
    decide which rows are totals; `provenance_refs_of` gives what the rows carry, which the
    render's source guards read."""
    if isinstance(trace, ManifestTrace):
        components: Sequence[Mapping[str, object]] = trace.components
        facets: Iterable[Mapping[str, object]] = trace.facets
    else:
        components = cast("Sequence[Mapping[str, object]]", trace.get("components") or ())
        facets = cast("Iterable[Mapping[str, object]]", trace.get("facets", ()))
    if components:
        return list(
            dict.fromkeys(
                cast(str, component["provenance"])
                for component in components
                if component.get("provenance") is not None
            )
        )
    return provenance_refs_of(facets)


def frame_code_systems(
    trace: ManifestTrace | Mapping[str, object], *, index: CodeIndex
) -> frozenset[str] | None:
    """The declared code systems the StatVars of a version's `trace.resolved_stat_vars` carry
    (`decode.statvar_code_systems`), the axes its frame can sum over, or `None` for a stored
    trace that records no `resolved_stat_vars`. `trace` is a live `ManifestTrace` or a
    stored or wire trace mapping."""
    if isinstance(trace, ManifestTrace):
        entries: Iterable[Mapping[str, object]] | None = trace.resolved_stat_vars
    else:
        entries = cast("Iterable[Mapping[str, object]] | None", trace.get("resolved_stat_vars"))
    if entries is None:
        return None
    return statvar_code_systems((cast(str, entry["ref"]) for entry in entries), index=index)


def stored_manifest_facts(stored: StoredVersion) -> ManifestFacts:
    """The facts of a stored dataset version. A published frame's lineage assembles no
    manifest and raises `TypeError`."""
    if not isinstance(stored.facts, ManifestFacts):
        raise TypeError(
            f"subject_id {stored.subject_id!r} content_version {stored.content_version!r} is a "
            "published frame, which stores no manifest facts"
        )
    return stored.facts


def _stored_manifest_fields(stored: StoredVersion) -> dict[str, object]:
    facts = stored_manifest_facts(stored)
    return {
        "definition": facts.definition,
        "content_version": stored.content_version,
        "prepared_at": facts.prepared_at,
        "coverage": facts.coverage,
        "citation": facts.citation,
        "trace": facts.trace,
        "warnings": facts.warnings,
        "series_count": facts.series_count,
        "series_coverage_stamp": facts.series_coverage_stamp,
    }


def stored_manifest(
    stored: StoredVersion, *, output_options: OutputOptionsModel | None = None
) -> Manifest:
    """The manifest a stored dataset version's facts assemble, with no artifacts and
    `output_options`, and its citation's diagnostics (`citation_diagnostics`). Every read of a
    stored version assembles its manifest here."""
    manifest = Manifest.model_validate(
        {
            **_stored_manifest_fields(stored),
            "artifacts": [],
            "output_options": (
                output_options.model_dump(mode="json") if output_options is not None else None
            ),
        }
    )
    return manifest.model_copy(
        update={"diagnostics": list(citation_diagnostics(manifest.citation))}
    )


def stored_output_profile(facts: ManifestFacts) -> OutputProfile | None:
    """The version's `OutputProfile`, or `None` for a version minted before versions stored
    it. A read derives `output_options` from it and current capabilities."""
    if facts.output_profile is None:
        return None
    return OutputProfile.from_json(facts.output_profile)


def stored_series_conversions(facts: ManifestFacts) -> dict[SeriesKey, SeriesConversion]:
    """The per-series conversion records a definition version stores, keyed by bare StatVar
    and role tuple. Empty for an unconverted version and for a converted version minted
    before versions stored them, which a reader tells apart by `trace.conversion`."""
    if facts.series_conversions is None:
        return {}
    return decode_series_conversion_entries(facts.series_conversions)


def version_document(stored: StoredVersion) -> dict[str, object]:
    """The JSON document a Presentation frame carries as its manifest: a dataset version's
    manifest without `artifacts`, `output_options`, `diagnostics` and `retention`, or a
    published frame's `op_spec_hash` (its subject id), `input_content_versions` and
    `continuations` (`FrameLineage.continuations`): its entries when it stores any, `null` while
    they are unknown, on a version recorded before versions stored them until
    `backfill-descriptions` fills them, and left out when nothing continues its series. Either
    adds `incomplete_after`, how complete the version's rows are, judged per source
    (`ManifestFacts.incomplete_after`, `FrameLineage.incomplete_after`), left out while unknown.
    Its `citation` is the served form, `licence*` members included. `version_body` stores the
    citation through `CitationModel.stored_json`, which drops them."""
    if isinstance(stored.facts, FrameLineage):
        document: dict[str, object] = {
            "op_spec_hash": stored.subject_id,
            "input_content_versions": list(stored.facts.input_content_versions),
        }
        continuations = stored.facts.continuations
        if continuations is None:
            document["continuations"] = None
        elif continuations:
            document["continuations"] = [dict(entry) for entry in continuations]
    else:
        document = stored_manifest(stored).model_dump(
            mode="json", exclude={"artifacts", "output_options", "diagnostics", "retention"}
        )
    if stored.facts.incomplete_after is not None:
        document["incomplete_after"] = [dict(entry) for entry in stored.facts.incomplete_after]
    return document


def version_body(record: VersionRecord) -> dict[str, object]:
    """The `immutable_body` `record`'s facts encode, the JSON an image reading bodies reads.

    A dataset version's body is its `version_document` with its citation as stored
    (`CitationModel.stored_json`), and `output_profile` and a non-null `default_scope_plural`
    added. The `series_conversions` column alone holds the per-series
    conversion records, which run to tens of megabytes on a wide converted mint. A published
    frame's body is its `version_document`, which holds its `continuations` when they are known and some continue
    its series, with a non-empty `composed_stat_vars` added. An absent fact omits its key. Neither
    holds `incomplete_after`, which its own column alone stores, so a body stays one the
    previous image decodes. Nothing in this image reads the body.
    """
    body = version_document(unlinked_state(record).version)
    body.pop("incomplete_after", None)
    facts = record.facts
    if isinstance(facts, FrameLineage):
        if facts.continuations is None:
            # A body states known continuations only, as the previous image's bodies hold none.
            del body["continuations"]
        if facts.composed_stat_vars:
            body["composed_stat_vars"] = [dict(entry) for entry in facts.composed_stat_vars]
        return body
    body["citation"] = CitationModel.model_validate(body["citation"]).stored_json()
    if facts.output_profile is not None:
        body["output_profile"] = dict(facts.output_profile)
    if facts.default_scope_plural is not None:
        body["default_scope_plural"] = facts.default_scope_plural
    return body


def persisted_manifest(
    *,
    stored: StoredVersion,
    served: Manifest,
    reshape_warnings: Sequence[WarningModel] = (),
) -> Manifest:
    """The manifest a write returns when its version record already existed.

    Every field comes from the stored version's facts (`stored_manifest`), so a re-mint after
    a graph rename keeps the stored citation, and its warnings, the origin findings recorded at
    mint included. From `served` it takes only `artifacts`, `output_options`,
    `series_coverage_stamp` and `diagnostics`, this call's. A `reshape_warnings` entry (this
    call's, `reshape_warning_models`) is appended when no stored warning has its `code` and
    `message`. A stored `origin_catalogue_stale` warning is dropped, since this call's
    diagnostics state the staleness of the catalogue bound to it.
    """
    manifest = stored_manifest(stored)
    warnings = [
        warning for warning in manifest.warnings if warning.code != "origin_catalogue_stale"
    ]
    keys = {(warning.code, warning.message) for warning in warnings}
    for warning in reshape_warnings:
        if (warning.code, warning.message) not in keys:
            keys.add((warning.code, warning.message))
            warnings.append(warning)
    return manifest.model_copy(
        update={
            "warnings": warnings,
            "diagnostics": list(served.diagnostics),
            "artifacts": list(served.artifacts),
            "output_options": served.output_options,
            "series_coverage_stamp": served.series_coverage_stamp,
        }
    )


def definition_version_record(
    *,
    manifest: Manifest,
    stored: StoredDefinition,
    materialisation: MaterialisationResult,
    artifacts: Sequence[ArtifactEntryModel],
    record_only_artifacts: Sequence[ArtifactRecord] = (),
    base_content_version: str | None = None,
    frame_summed: bool | None = None,
    code_index_stamp: str | None = None,
    incomplete_after: Sequence[Mapping[str, object]] | None = None,
    incomplete_after_built_at: datetime | None = None,
) -> VersionRecord:
    """The version record of a definition materialisation, linking `artifacts` and
    `record_only_artifacts`. A derived version stores the raw version it was rendered from as
    `base_content_version`, whether its frame holds its summed rows as `frame_summed`
    (`render.DerivedVersion.summed`), and the stamp of the code index and render format it was
    rendered under (`versions.rendering_stamp`) as `code_index_stamp`. A raw version passes none of
    them. `incomplete_after` is how complete the version's rows are (`incomplete_after`), `None`
    when unjudged, and `incomplete_after_built_at` the `built_at` of the catalogue build that
    judged it. With the `held_back` runs of `manifest.coverage` they are the version's judged
    record, which a later write refreshes together (`version_store._refresh_judged_record`).
    A derived version's runs are its base call's, since runs are stated whatever the window.
    Only a raw version stores the per-series conversion records. A derived version's rows are
    not those series, and the observations door, their reader, serves raw versions alone.

    Each of `artifacts`, rendered by this call, is linked under the key its `enrichment_stamp`
    salts (`versions.link_digest`). `record_only_artifacts` carry their own salt."""
    return VersionRecord(
        subject_id=stored.spec_id,
        subject_kind="spec",
        content_version=materialisation.content_version,
        schema_version=stored.schema_version,
        facts=manifest_facts(
            manifest,
            output_profile=materialisation.output_profile,
            series_conversions=(
                materialisation.series_conversions if base_content_version is None else None
            ),
            default_scope=materialisation.default_scope,
            incomplete_after=incomplete_after,
            incomplete_after_built_at=incomplete_after_built_at,
        ),
        artifacts=tuple(
            ArtifactRecord(
                artifact_digest=artifact.artifact_digest,
                rendering=artifact.rendering,
                media_type=artifact.media_type,
                enrichment_digest=link_digest(artifact.rendering, stamp=artifact.enrichment_stamp),
            )
            for artifact in artifacts
        )
        + tuple(record_only_artifacts),
        base_content_version=base_content_version,
        frame_grammar_version=materialisation.trace.frame_grammar_version,
        frame_sha256=materialisation.frame_sha256,
        frame_summed=frame_summed,
        code_index_stamp=code_index_stamp,
    )


def collapsed_head_manifest(
    *,
    manifest: Manifest,
    derived: DerivedVersion,
    artifact: ArtifactEntryModel,
    statistics_artifact: ArtifactEntryModel,
    series_count: int,
    output_options: OutputOptionsModel | None,
    extra_warnings: Sequence[WarningModel] = (),
) -> Manifest:
    """Patches a raw head manifest for a derived identity (`render.DerivedVersion`) without a
    fresh materialisation.

    Updates content version, row count and the dimension counts the derived frame counts anew,
    series count, artifacts, output options, and warnings. A party-totals cut adds the totals it
    left out to the trace (`party_totals.TRACE_KEY`), and a collapse rewrites the citation line
    for its sum (`collapsed_citation`). The cited sources, the rest of the trace, definition and
    `prepared_at` remain unchanged because a derived version preserves provenance and scope and
    renders from the raw version's record.
    """
    coverage = manifest.coverage.model_copy(
        update={
            "rows": derived.rows,
            "dimensions": [
                dimension.model_copy(
                    update={"count": derived.distinct.get(dimension.label, dimension.count)}
                )
                for dimension in manifest.coverage.dimensions
            ],
        }
    )
    trace = (
        {**manifest.trace, TRACE_KEY: list(derived.party_totals_left_out)}
        if derived.party_totals_left_out
        else manifest.trace
    )
    return manifest.model_copy(
        update={
            "content_version": derived.content_version,
            "citation": collapsed_citation(manifest.citation, base=manifest)
            if derived.summed
            else manifest.citation,
            "coverage": coverage,
            "trace": trace,
            "series_count": series_count,
            "warnings": list(manifest.warnings) + list(extra_warnings),
            "artifacts": [artifact, statistics_artifact],
            "output_options": output_options,
        }
    )


def record_collapsed_head_version(
    *,
    manifest: Manifest,
    subject_id: str,
    schema_version: int,
    base_content_version: str,
    output_profile: OutputProfile,
    frame_sha256: str,
    frame_summed: bool,
    code_index_stamp: str,
    incomplete_after: Sequence[Mapping[str, object]] | None = None,
    incomplete_after_built_at: datetime | None = None,
    record_only_artifacts: Sequence[ArtifactRecord] = (),
    version_writer: VersionRecordWriter = NULL_VERSION_RECORD_WRITER,
    trace_id: str | None = None,
    describer_for: Callable[[VersionRecord], VersionDescriber | None] | None = None,
) -> bool:
    """Records a head-hit collapse without a fresh materialisation result and returns `False` if
    the record already existed.

    `manifest` is already patched. `base_content_version` names the raw materialisation, and
    `output_profile`, `frame_sha256` and `frame_summed` (`render.DerivedVersion.summed`) are the
    collapsed frame's, `code_index_stamp` the stamp of the code index and render format it was
    rendered under (`versions.rendering_stamp`), `incomplete_after` its record
    (`collapsed_incomplete_after`), and `incomplete_after_built_at` the `built_at` of the build
    that judged the base's. The `held_back` runs of `manifest.coverage` are the base's, which
    a write of a version already on record stores with `incomplete_after`
    (`version_store._refresh_judged_record`). The artifacts in `manifest`
    are linked as `definition_version_record` links its `artifacts`. `ON CONFLICT DO NOTHING`
    makes repeated writes idempotent. `describer_for` gives the describer that writes the
    version's description record in the same transaction, from the record this call stores;
    `None`, or a describer of `None`, writes none.
    """
    record = VersionRecord(
        subject_id=subject_id,
        subject_kind="spec",
        content_version=manifest.content_version,
        schema_version=schema_version,
        facts=manifest_facts(
            manifest,
            output_profile=output_profile,
            incomplete_after=incomplete_after,
            incomplete_after_built_at=incomplete_after_built_at,
        ),
        artifacts=tuple(
            ArtifactRecord(
                artifact_digest=artifact.artifact_digest,
                rendering=artifact.rendering,
                media_type=artifact.media_type,
                enrichment_digest=link_digest(artifact.rendering, stamp=artifact.enrichment_stamp),
            )
            for artifact in manifest.artifacts
        )
        + tuple(record_only_artifacts),
        base_content_version=base_content_version,
        frame_grammar_version=FRAME_GRAMMAR_VERSION,
        frame_sha256=frame_sha256,
        frame_summed=frame_summed,
        code_index_stamp=code_index_stamp,
    )
    if describer_for is not None:
        record = replace(record, describer=describer_for(record))
    with phase("record"):
        return version_writer.record_version(record, trace_id=trace_id)
