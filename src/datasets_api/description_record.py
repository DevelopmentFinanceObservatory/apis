"""Build a version's `DescriptionRecord` from what the site that writes its version row holds.

A record holds a version's facts and evidence, fixed when the version is made. Three builders
cover the four version-writing sites, and each is pure:

- `mint_record` builds a `spec_` version's record from its stored facts (`ManifestFacts`), the
  `SeriesFigures` of its series statistics, and the code index.
- `collapsed_record` derives a multi-contributor collapse's record, at mint or from a collapsed
  head, from its base version's record and the collapse's own figures, adding one step that
  names what was summed.
- `publish_record` builds an `ops_` version's record from its lineage (`FrameLineage`), the
  posted step records and the published frame, since publish stores no statistics artifact.
- `backfill_record` builds the record of a version made before records existed, from its
  stored facts, what its artifacts still hold, and today's code index.

A builder states every field. A fact nothing recorded is `None`, and a count of nothing is 0.
Labels are evidence read by the caller: `named_refs` lists the dcids a record's facts name, and
`with_labels` adds the labels read for them. Every text the builders copy in, such as a label,
a source's name and licence, or a pipeline step, passes through `bounded_text`, which drops its
U+0000 characters and clips it to the bound its field holds, so a record never fails the mint
or publish whose transaction it is inserted in.

`WARNING_DESTINATIONS` says where each Datasets warning code goes once descriptions carry the
version's facts. The builders read the facts behind a step or caveat from the trace, so a
version whose manifest still lists the warning gets the same record as one whose does not.
`backfill_record` is the one builder that reads stored warning codes: a version made before
records existed holds some of those facts only in its warnings.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.resources
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache
from typing import Literal, NamedTuple, cast

from dfo_frame_grammar import Frame
from pydantic.experimental.missing_sentinel import MISSING

from code_index.flow_family import flow_family
from code_index.query import CodeIndex
from code_index.types import StatVarRow
from dataset_store.snapshot import FrameLineage, ManifestFacts, StoredVersion
from datasets_api.versions import VersionRecord
from description.facts import (
    Basis,
    Builds,
    CaveatFact,
    CodeIndexStamp,
    ComposedVariable,
    ComputedFacts,
    DescriptionRecord,
    DimensionFacts,
    DimensionValue,
    Evidence,
    FixedValue,
    InputVersion,
    LabelEvidence,
    PipelineStep,
    RoleCount,
    SeriesFacts,
    SourceEvidence,
    SourceUnitCount,
    Span,
    StatusCounts,
    VariableFacts,
)
from description.steps import (
    CONTINUE_SERIES_OP,
    Choice,
    ComposedUnit,
    LeftOut,
    SimpleUnit,
    Step,
    StepRef,
    Summed,
    Unit,
    VariableSide,
    VersionRef,
)
from description.steps_text import bound_entities
from description.stored import LONG_TEXT_MAX_LENGTH, TEXT_MAX_LENGTH, clipped
from description.vocabulary import Vocabulary
from dfo_platform.roles import axis_for_role, declared_role_for
from materialise.continued_series import continued_series_of_entries
from materialise.statistics import (
    RoleFigure,
    SeriesFigures,
    SeriesStatisticsSink,
    compute_series_statistics_table,
)

WarningDestination = Literal["gap", "diagnostics", "step", "caveat", "warning", "retired"]
"""Where a warning code's fact goes: a description gap, the manifest's diagnostics, a step, a
caveat, the manifest's warnings, or nowhere for a retired code."""


class WarningRoute(NamedTuple):
    """A warning code's destination and the wire member that carries it there."""

    destination: WarningDestination
    member: str | None


WARNING_DESTINATIONS: Mapping[str, WarningRoute] = {
    "revision_unavailable": WarningRoute("gap", "freshness.source_release"),
    "processing_level_undeclared": WarningRoute("diagnostics", "diagnostics"),
    "origin_catalogue_stale": WarningRoute("diagnostics", "diagnostics"),
    "stat_vars_summed": WarningRoute("step", "steps.summed"),
    "facet_default_selected": WarningRoute("step", "steps.choice"),
    "deflator_projected": WarningRoute("caveat", "caveats.estimated"),
    "currency_unmapped_omitted": WarningRoute("step", "steps.left_out"),
    "deflator_missing_omitted": WarningRoute("step", "steps.left_out"),
    "withheld_by_registry": WarningRoute("caveat", "caveats.coverage_gap"),
    "un_system_unavailable": WarningRoute("caveat", "caveats.coverage_gap"),
    "coverage_below_scope": WarningRoute("warning", "warnings"),
    "empty_result": WarningRoute("warning", "warnings"),
    "enrichment_unavailable": WarningRoute("warning", "warnings"),
    "observation_attributes_in_notes": WarningRoute("warning", "warnings"),
    "members_deduplicated": WarningRoute("retired", None),
}
"""Each Datasets warning code's destination. A code moves into the description when it states a
fact of the version, one every request yielding the same `content_version` shares; a code that
depends on the request, the rendering or the serving instance stays a warning or moves to
diagnostics. `origin_catalogue_stale` also leaves a `sources` gap when a name or licence is
unknown, and the two conversion gaps add a `coverage_gap` caveat pointing at the conversion."""

CHOICE_RANKING = ("places_covered", "years_covered", "observation_count")
"""The ranking a deployment's default facet policy applies, in order."""

_CHOOSE_SOURCE_OP = "choose_source"
_CHOOSE_SOURCE_POLICY = "facet_default_by_coverage"
_SUM_VARIABLES_OP = "sum_variables"
_SUM_VARIABLES_POLICY = "sum_family_members"
_CONVERT_OP = "convert"
_CONVERT_POLICY = "convert_by_rate_and_deflator"
_CONTINUE_SERIES_POLICY = "continue_with_source_estimates"
_RETRIEVE_OP = "retrieve"

# `materialise.facets` names this reason when a deployment's default policy picks one of
# several candidates that disagree; an unopposed fallback is no choice.
_CHOSEN_OVER_DISAGREEMENT = "most_observations_over_disagreement"
# Conversion gaps a trace minted before conversions counted what they left out names only in
# its warnings: a caveat points at the conversion, with no count.
_CONVERSION_GAP_CODES = ("currency_unmapped_omitted", "deflator_missing_omitted")
_MAX_SUMMED_NAMED = 64
_MAX_CHOICE_OTHERS = 32
_DCID_PREFIX = "dcid:"
_YEARS_DIMENSION = "years"
# A collapse's frame names each code axis its groupby keeps `stat_var_<property>`.
_AXIS_COLUMN_PREFIX = "stat_var_"
# Each identity `shape` member a record keeps, and the graph property it names.
_SHAPE_PROPERTIES = {
    "population_type": "populationType",
    "measured_property": "measuredProperty",
    "stat_type": "statType",
    "measurement_qualifier": "measurementQualifier",
}
_METHOD_STATUS_PACKAGE = "materialise"
_METHOD_STATUS_FILE = "method_status.json"

ROLE_VALUES_MAX = 500
"""The most values a role dimension stores, each about 50 bytes in the record. A role holding
more keeps those held by the most variables and is `partial`."""


def mint_record(
    version: VersionRecord, *, figures: SeriesFigures, code_index: CodeIndex, title: str | None
) -> DescriptionRecord:
    """The record of a minted `spec_` version.

    Args:
        version: the version as recorded. On a mint that found the version already recorded,
            its facts are the stored ones, so the record cites and dates as the stored row.
        figures: the version's series figures (`SeriesStatisticsFile.figures`), which also
            give its open roles' values.
        code_index: the code index the dimension facts are derived through.
        title: the title composed at mint, or `None` when none was composed.

    Raises:
        ValueError: `version` holds a published frame's lineage.
    """
    facts = _manifest_facts(version)
    trace = facts.trace
    variables = variable_facts(
        definition=facts.definition,
        trace=trace,
        code_index=code_index,
        roles=figures.entities_by_role,
    )
    steps = mint_steps(trace=trace, definition=facts.definition)
    return DescriptionRecord(
        id=version.subject_id,
        content_version=version.content_version,
        schema_version=1,
        basis="mint",
        index_stamp=_index_stamp(code_index) if variables.dimensions is not None else None,
        variables=variables,
        computed=None,
        series=series_facts(
            figures,
            rows=_coverage_rows(facts.coverage),
            periods=_coverage_years(facts.coverage) if figures.grain == "year" else None,
        ),
        steps=steps,
        caveats=_mint_caveats(facts, steps),
        evidence=Evidence(
            prepared=facts.prepared_at.date(),
            title=bounded_text(title),
            sources=source_evidence(facts.citation),
            source_release=None,
            labels=[],
            membership=None,
            builds=Builds(
                origin_catalogue=bounded_text(trace.get("origin_catalogue_build_id")),
                code_index=_trace_index_stamp(trace),
                coverage=bounded_text(facts.series_coverage_stamp),
                method_status=method_status_build(),
            ),
        ),
        late_fields=[],
    )


def collapsed_record(
    base: DescriptionRecord, collapsed: VersionRecord, *, figures: SeriesFigures
) -> DescriptionRecord:
    """The record of a multi-contributor collapse, derived from its base version's record.

    It keeps the base's population, sources and builds, takes its series facts and its
    dimensions from the collapse's own figures and coverage (`_collapsed_variables`), and adds
    a `sum_variables` step naming the variables with data that were summed. Its caveats are
    the base's as they hold for the sum (`_collapsed_caveats`), and the base's labels stay for
    the refs the collapse still names. A base whose steps are unknown leaves the steps before
    the sum unknown, and the record holds the sum as the step known to follow them
    (`DescriptionRecord.steps_after_unknown`).

    Raises:
        ValueError: `collapsed` holds a published frame's lineage, or names another subject.
    """
    facts = _manifest_facts(collapsed)
    if collapsed.subject_id != base.id:
        raise ValueError(f"{collapsed.subject_id} collapses no version of {base.id}")
    return _collapsed(
        base,
        facts,
        content_version=collapsed.content_version,
        series=series_facts(
            figures,
            rows=_coverage_rows(facts.coverage),
            periods=_coverage_years(facts.coverage) if figures.grain == "year" else None,
        ),
        columns=_collapse_columns(facts, figures),
    )


def _collapsed(
    base: DescriptionRecord,
    facts: ManifestFacts,
    *,
    content_version: str,
    series: SeriesFacts,
    columns: Mapping[str, RoleFigure | None] | None,
) -> DescriptionRecord:
    """`base` as the record of the collapse `content_version`, holding `series`, the
    dimensions its role `columns` keep and, after the base's steps, a `sum_variables` step
    naming the variables with data in the trace (`collapse_steps`). Where the base's steps are
    unknown, the sum is the step known to follow them."""
    summed = _summed_refs(facts.trace)
    steps = collapse_steps(base.known_steps, trace=facts.trace)
    record = base.model_copy(
        update={
            "content_version": content_version,
            "variables": None
            if base.variables is None
            else _collapsed_variables(base.variables, columns=columns),
            "series": series,
            "steps": None if base.steps is None else steps,
            "steps_after_unknown": steps if base.steps is None else MISSING,
            "caveats": None
            if base.caveats is None
            else _collapsed_caveats(base.caveats, variable=base.id, summed=summed),
            "late_fields": [],
        }
    )
    named = set(named_refs(record))
    labels = base.evidence.labels
    evidence = base.evidence.model_copy(
        update={
            "labels": None if labels is None else [label for label in labels if label.ref in named]
        }
    )
    return record.model_copy(update={"evidence": evidence})


def _collapse_columns(
    facts: ManifestFacts, figures: SeriesFigures | None
) -> Mapping[str, RoleFigure | None] | None:
    """The role columns of a collapse's frame, each with its figure: from `figures`, else
    from the stored output profile with each figure unknown, else `None`."""
    if figures is not None:
        return {figure.role: figure for figure in figures.entities_by_role}
    if facts.output_profile is None:
        return None
    columns = cast("Sequence[Mapping[str, object]]", facts.output_profile.get("columns", ()))
    return {cast(str, column["name"]): None for column in columns if column.get("kind") == "role"}


def _collapsed_variables(
    variables: VariableFacts, *, columns: Mapping[str, RoleFigure | None] | None
) -> VariableFacts:
    """A collapse's population: the base's, with the dimensions its frame keeps.

    A collapse keeps each role as a column, and each code axis its groupby keeps as the column
    `stat_var_<property>` (`datasets_api.render.GroupbyCollapseRendering`); every other axis
    is summed away. A kept dimension takes its values from the collapse's figure for its
    column, unknown where the figure is not held. A summed-away dimension narrowed `within` a
    code is that code's total, so the code becomes a fixed value. With `columns` `None` the
    collapse's columns are unknown, and so are its dimensions.
    """
    if variables.dimensions is None:
        return variables
    if columns is None:
        return variables.model_copy(update={"dimensions": None})
    fixed = list(variables.fixed)
    dimensions = []
    for dimension in variables.dimensions:
        column = next(
            (
                name
                for name in (dimension.property, f"{_AXIS_COLUMN_PREFIX}{dimension.property}")
                if name in columns
            ),
            None,
        )
        if column is not None:
            figure = columns[column]
            kept = (
                dimension.model_copy(
                    update={"values": None, "undeclared_variables": 0, "partial": False}
                )
                if figure is None
                else _role_dimension(figure)
            )
            dimensions.append(
                kept.model_copy(update={"property": dimension.property, "within": dimension.within})
            )
        elif dimension.within is not None:
            fixed.append(FixedValue(property=dimension.property, value=dimension.within))
    return variables.model_copy(update={"fixed": fixed, "dimensions": dimensions})


def _collapsed_caveats(
    caveats: Sequence[CaveatFact], *, variable: str, summed: Sequence[str]
) -> list[CaveatFact]:
    """The base's `caveats` on a collapse, whose one `variable` sums the `summed` members.

    A caveat naming no series stays as it is. One naming series and no entities, such as an
    estimate, holds for the sum and names `variable`. A caveat naming entities and one member
    says that member has no data there, which holds for the sum only where every summed
    member has none: those entities, in first-appearance order, become one caveat of the kind
    on `variable`, at the place of the first, leaving out any a caveat naming no series
    already names. An entity only some members lack still holds a summed value, so it is left
    out, and so is each member caveat's `count`, since a place past a member caveat's names
    is not known to be missing from it.
    """
    collapsed: list[CaveatFact] = []
    missing: dict[str, dict[str, set[str]]] = {}
    ordered: dict[str, dict[str, None]] = {}
    for caveat in caveats:
        if caveat.entities and caveat.series:
            if caveat.kind not in missing:
                collapsed.append(caveat)
            missing.setdefault(caveat.kind, {}).setdefault(caveat.series[0], set()).update(
                caveat.entities
            )
            ordered.setdefault(caveat.kind, {}).update(dict.fromkeys(caveat.entities))
        elif caveat.series:
            collapsed.append(caveat.model_copy(update={"series": [variable]}))
        else:
            collapsed.append(caveat)
    covered = {entity for caveat in collapsed if not caveat.series for entity in caveat.entities}
    result = []
    for caveat in collapsed:
        if not (caveat.entities and caveat.series):
            result.append(caveat)
            continue
        by_member = missing[caveat.kind]
        everywhere = [
            entity
            for entity in ordered[caveat.kind]
            if entity not in covered
            and summed
            and all(entity in by_member.get(member, ()) for member in summed)
        ]
        if everywhere:
            result.append(
                caveat.model_copy(
                    update={"series": [variable], "entities": everywhere, "count": None}
                )
            )
    return result


def collapse_steps(steps: Sequence[Step], *, trace: Mapping[str, object]) -> list[Step]:
    """The steps of a multi-contributor collapse of a version whose steps are `steps`: those,
    then a `sum_variables` step naming the variables with data in the version's `trace`."""
    summed = _summed_refs(trace)
    sum_step = Step(
        id=_next_step_id(steps),
        op=_SUM_VARIABLES_OP,
        policy=_SUM_VARIABLES_POLICY,
        summed=Summed(
            count=len(summed),
            variables=[StepRef(ref=ref) for ref in summed[:_MAX_SUMMED_NAMED]],
        ),
    )
    return [*steps, sum_step]


def _summed_refs(trace: Mapping[str, object]) -> list[str]:
    """The variables a collapse of a version with `trace` sums: those with data in it."""
    return [ref for ref, count in _resolved(trace) if count]


def publish_record(
    version: VersionRecord,
    *,
    steps: Sequence[Step] | None,
    frame: Frame,
    computed_at: datetime.datetime,
) -> DescriptionRecord:
    """The record of a published `ops_` version.

    Its inputs are the root versions its `retrieve` steps name, each with its id. Series facts
    come from the frame's rows. Each step that left data out gets a `coverage_gap` caveat
    pointing at it. Without steps, each input is a posted content version, the frame's
    immediate lineage, with an unknown id, and the caveats and label evidence are unknown until
    a later publish fills the steps (`version_descriptions.PublishDescriber.settle`).

    Args:
        version: the version as recorded, with its frame lineage.
        steps: the posted step records, or `None` when the publish carried none.
        frame: the published frame.
        computed_at: when the version was computed, which its citation gives as its date.

    Raises:
        ValueError: `version` holds a dataset version's facts.
    """
    table = frame.table
    lineage = version.facts
    if not isinstance(lineage, FrameLineage):
        raise ValueError(f"{version.subject_id} holds no published frame's lineage")
    return _computed_record(
        version,
        basis="publish",
        steps=steps,
        series=series_facts(
            frame_figures(frame, lineage=lineage),
            rows=table.num_rows,
            periods=len(set(table.column("time_period").to_pylist())),
        ),
        computed_at=computed_at,
    )


def _computed_record(
    version: VersionRecord | StoredVersion,
    *,
    basis: Basis,
    steps: Sequence[Step] | None,
    series: SeriesFacts,
    computed_at: datetime.datetime,
) -> DescriptionRecord:
    lineage = version.facts
    if not isinstance(lineage, FrameLineage):
        raise ValueError(f"{version.subject_id} holds no published frame's lineage")
    return DescriptionRecord(
        id=version.subject_id,
        content_version=version.content_version,
        schema_version=1,
        basis=basis,
        index_stamp=None,
        variables=None,
        computed=ComputedFacts(
            inputs=(
                [InputVersion(id=None, content_version=cv) for cv in lineage.input_content_versions]
                if steps is None
                else computed_inputs(steps)
            ),
            composed=composed_variables(lineage.composed_stat_vars),
        ),
        series=series,
        steps=None if steps is None else list(steps),
        caveats=None if steps is None else _step_caveats(steps),
        evidence=Evidence(
            prepared=computed_at.date(),
            title=None,
            sources=[],
            source_release=None,
            labels=None if steps is None else [],
            membership=None,
            builds=Builds(
                origin_catalogue=None,
                code_index=None,
                coverage=None,
                method_status=method_status_build(),
            ),
        ),
        late_fields=[],
    )


class BackfilledRecord(NamedTuple):
    """A backfilled record, and the stored warning codes whose fact the backfill could not
    state, each left out of the record."""

    record: DescriptionRecord
    unstated_warnings: list[str]


def backfill_record(
    version: StoredVersion,
    *,
    statistics: SeriesFigures | None,
    frame: Frame | None,
    code_index: CodeIndex,
    recorded_at: datetime.datetime,
    backfilled_at: datetime.datetime,
    base: DescriptionRecord | None = None,
) -> BackfilledRecord:
    """The record of a version made before records existed, with basis `backfill`.

    A `spec_` version's variable facts derive through today's code index, whose stamp the
    record names, and its open roles' values come from `statistics`; without them each role
    the trace declares and the definition leaves open is a dimension with unknown values. Its series, row, provider and year counts come from its stored facts, and
    the counts only series statistics hold come from `statistics`, unknown without them. Its
    steps and caveats are the mint's, read from the trace and from the stored warning codes
    `WARNING_DESTINATIONS` routes to a step or caveat; a code whose fact cannot be stated is
    left out and listed in `unstated_warnings` (see `stored_warning_facts`).

    A multi-contributor collapse, a `spec_` version with `base_content_version` set, is derived
    from `base`, its base version's record, as `collapsed_record` derives it at mint, with its
    own series facts and the dimensions its `statistics`, else its stored output profile, say
    its frame keeps.

    An `ops_` version's inputs are its stored content versions with unknown ids, its composed
    variables come from its stored lineage, its steps and what derives from them (caveats,
    label evidence) are unknown, and its series facts come from `frame`, counted under its
    stored continuations, all unknown when `frame` is `None`.

    Each label today's code index holds for a dcid the record names is read on the backfill's
    date.

    Args:
        version: the stored version.
        statistics: a `spec_` version's series figures, from its series statistics artifact or
            its body's inline statistics; `None` when neither is held.
        frame: an `ops_` version's stored frame; `None` when it is not readable.
        code_index: today's code index.
        recorded_at: when the version row was recorded, an `ops_` version's computed date.
        backfilled_at: when the backfill runs.
        base: a collapse's base version's record; `None` when `version` is no collapse.

    Raises:
        ValueError: a `spec_` version's stored facts state no series count and `statistics`
            is `None`, or `version` is a collapse and `base` is `None`.
    """
    facts = version.facts
    unstated: list[str] = []
    if isinstance(facts, FrameLineage):
        series = _UNKNOWN_SERIES
        if frame is not None:
            table = frame.table
            series = series_facts(
                frame_figures(frame, lineage=facts),
                rows=table.num_rows,
                periods=len(set(table.column("time_period").to_pylist())),
            )
        record = _computed_record(
            version, basis="backfill", steps=None, series=series, computed_at=recorded_at
        )
    else:
        record, unstated = _backfilled_dataset_record(
            version, facts, statistics=statistics, code_index=code_index
        )
        if version.base_content_version is not None:
            if base is None:
                raise ValueError("a collapse is backfilled from its base version's record")
            record, unstated = (
                _backfilled_collapse(record, facts, base=base, statistics=statistics),
                [],
            )
    labels = {ref: row.label for ref in named_refs(record) if (row := code_index.code_by_dcid(ref))}
    return BackfilledRecord(with_labels(record, labels, read=backfilled_at.date()), unstated)


def _backfilled_dataset_record(
    version: StoredVersion,
    facts: ManifestFacts,
    *,
    statistics: SeriesFigures | None,
    code_index: CodeIndex,
) -> tuple[DescriptionRecord, list[str]]:
    trace = facts.trace
    variables = variable_facts(
        definition=facts.definition,
        trace=trace,
        code_index=code_index,
        roles=None if statistics is None else statistics.entities_by_role,
    )
    steps, caveats, unstated = stored_warning_facts(
        facts, steps=mint_steps(trace=trace, definition=facts.definition)
    )
    record = DescriptionRecord(
        id=version.subject_id,
        content_version=version.content_version,
        schema_version=1,
        basis="backfill",
        index_stamp=_index_stamp(code_index) if variables.dimensions is not None else None,
        variables=variables,
        computed=None,
        series=_stored_series_facts(facts, statistics),
        steps=steps,
        caveats=caveats,
        evidence=Evidence(
            prepared=facts.prepared_at.date(),
            title=None,
            sources=source_evidence(facts.citation),
            source_release=None,
            labels=[],
            membership=None,
            builds=Builds(
                origin_catalogue=bounded_text(trace.get("origin_catalogue_build_id")),
                code_index=_trace_index_stamp(trace),
                coverage=bounded_text(facts.series_coverage_stamp),
                method_status=None if statistics is None else method_status_build(),
            ),
        ),
        late_fields=[],
    )
    return record, unstated


def _backfilled_collapse(
    own: DescriptionRecord,
    facts: ManifestFacts,
    *,
    base: DescriptionRecord,
    statistics: SeriesFigures | None,
) -> DescriptionRecord:
    """A collapse's backfilled record: derived from `base` with `own`'s series facts and the
    dimensions its `statistics`, else its stored output profile, say its frame keeps."""
    record = _collapsed(
        base,
        facts,
        content_version=own.content_version,
        series=own.series,
        columns=_collapse_columns(facts, statistics),
    )
    return record.model_copy(update={"basis": "backfill"})


_UNKNOWN_SERIES = SeriesFacts(
    series=None,
    rows=None,
    periods=None,
    span=None,
    grain=None,
    latest_period=None,
    series_reaching_latest=None,
    entities_by_role=None,
    by_source_unit=None,
    status=StatusCounts(estimated=None, provisional=None),
)
"""Series facts of a version whose stored facts and artifacts hold none of them."""


def _stored_series_facts(facts: ManifestFacts, statistics: SeriesFigures | None) -> SeriesFacts:
    """Series facts from the stored counts, and from `statistics` for the counts only series
    statistics hold."""
    series = facts.series_count
    if series is None and statistics is not None:
        series = statistics.series
    if series is None:
        raise ValueError("the stored facts state no series count and no statistics are held")
    grain = None if statistics is None else statistics.grain
    roles = [
        RoleCount(role=cast(str, dimension["label"]), entities=cast(int, dimension["count"]))
        for dimension in cast(
            "Sequence[Mapping[str, object]]", facts.coverage.get("dimensions", ())
        )
        if dimension.get("label") != _YEARS_DIMENSION
    ]
    if statistics is None:
        return _UNKNOWN_SERIES.model_copy(
            update={
                "series": series,
                "rows": _coverage_rows(facts.coverage),
                "entities_by_role": roles,
            }
        )
    figured = series_facts(
        statistics,
        rows=_coverage_rows(facts.coverage),
        periods=_coverage_years(facts.coverage) if grain == "year" else None,
    )
    return figured.model_copy(update={"series": series, "entities_by_role": roles})


# Stored warnings


def stored_warning_facts(
    facts: ManifestFacts, *, steps: Sequence[Step]
) -> tuple[list[Step] | None, list[CaveatFact], list[str]]:
    """The steps and caveats of a version's stored warning codes, added to the steps its
    trace gives.

    Each code `WARNING_DESTINATIONS` routes to a step or caveat states its fact as the mint
    builders do: a sum adds a `sum_variables` step, a choice of source and the conversion gaps
    are already steps the trace gives, an estimate or conversion gap is a caveat on each
    conversion step, and a registry withholding or an unanswered UN read is a coverage gap
    naming the variable and places its message lists, unless the trace records them
    (`trace.origin_gaps`), as `_mint_caveats` reads them. A code routed to a gap, diagnostics or
    the manifest's warnings adds nothing to the record.

    A code whose fact cannot be stated, because its message names nothing in a format its
    producer wrote or the trace holds no step for it to point at, adds nothing and is listed
    in the third member. A step it should have stated leaves the steps incomplete, so they
    are `None`, served as a gap, and each caveat pointing at a step goes with them.
    """
    steps = list(steps)
    caveats = _mint_caveats(facts, steps)
    unstated: list[str] = []
    steps_incomplete = False
    for warning in facts.warnings:
        code = warning.get("code")
        route = WARNING_DESTINATIONS.get(code) if isinstance(code, str) else None
        if route is None or route.destination not in ("step", "caveat"):
            continue
        try:
            _STORED_WARNING_TRANSLATIONS[cast(str, code)](warning, facts, steps, caveats)
        except _Unstated:
            unstated.append(cast(str, code))
            steps_incomplete |= route.destination == "step"
    if steps_incomplete:
        return None, [caveat for caveat in caveats if caveat.step is None], unstated
    return steps, caveats, unstated


class _Unstated(Exception):
    """A stored warning's fact cannot be stated: its message names nothing in a format its
    producer wrote, or the trace holds no step for it to point at."""


type _Translation = Callable[
    [Mapping[str, object], ManifestFacts, list[Step], list[CaveatFact]], None
]


def _summed_step(
    warning: Mapping[str, object], facts: ManifestFacts, steps: list[Step], _: list[CaveatFact]
) -> None:
    if any(step.op == _SUM_VARIABLES_OP for step in steps):
        return
    summed = _summed_refs(facts.trace)
    count = warning.get("contributing_member_count")
    steps.append(
        Step(
            id=_next_step_id(steps),
            op=_SUM_VARIABLES_OP,
            policy=_SUM_VARIABLES_POLICY,
            summed=Summed(
                count=count if isinstance(count, int) else len(summed),
                variables=[StepRef(ref=ref) for ref in summed[:_MAX_SUMMED_NAMED]],
            ),
        )
    )


def _chosen_step(
    _warning: Mapping[str, object], _facts: ManifestFacts, steps: list[Step], _: list[CaveatFact]
) -> None:
    if not any(step.op == _CHOOSE_SOURCE_OP for step in steps):
        raise _Unstated


def _conversion_gap(
    _warning: Mapping[str, object], _facts: ManifestFacts, steps: list[Step], _: list[CaveatFact]
) -> None:
    # `_mint_caveats` points a coverage gap at each conversion step for these codes.
    if not any(step.op == _CONVERT_OP for step in steps):
        raise _Unstated


def _projected_caveat(
    _warning: Mapping[str, object],
    _facts: ManifestFacts,
    steps: list[Step],
    caveats: list[CaveatFact],
) -> None:
    conversions = [step.id for step in steps if step.op == _CONVERT_OP]
    if not conversions:
        raise _Unstated
    estimated = {caveat.step for caveat in caveats if caveat.kind == "estimated"}
    caveats.extend(
        CaveatFact(kind="estimated", step=step_id)
        for step_id in conversions
        if step_id not in estimated
    )


_WITHHELD_MESSAGE = re.compile(r"(?P<variable>\S+) served no observations for (?P<places>.+?); ")
_UN_UNREAD_MESSAGE = re.compile(r"No UN data was read for (?P<places>.+)\.$")
_MORE_NAMED = re.compile(r"^(?P<named>.+) and (?P<more>\d+) more$")


def _withheld_caveat(
    warning: Mapping[str, object],
    facts: ManifestFacts,
    _steps: list[Step],
    caveats: list[CaveatFact],
) -> None:
    if any(gap.get("code") == "withheld_by_registry" for gap in _origin_gaps(facts)):
        return
    match = _WITHHELD_MESSAGE.search(cast(str, warning.get("message", "")))
    if match is None:
        raise _Unstated
    entities, count = _named_places(match["places"])
    caveats.append(
        CaveatFact(
            kind="coverage_gap",
            series=[_bare(match["variable"])],
            entities=entities,
            count=count,
        )
    )


def _un_unread_caveat(
    warning: Mapping[str, object],
    facts: ManifestFacts,
    _steps: list[Step],
    caveats: list[CaveatFact],
) -> None:
    if any(gap.get("code") == "un_system_unavailable" for gap in _origin_gaps(facts)):
        return
    match = _UN_UNREAD_MESSAGE.search(cast(str, warning.get("message", "")))
    if match is None:
        raise _Unstated
    entities, count = _named_places(match["places"])
    caveats.append(CaveatFact(kind="coverage_gap", entities=entities, count=count))


def _named_places(named: str) -> tuple[list[str], int | None]:
    """The places a message lists (`dfo_platform.named.named_capped`), and how many it
    counts in all when it names only some."""
    more = 0
    if (capped := _MORE_NAMED.match(named)) is not None:
        named, more = capped["named"], int(capped["more"])
    places = [_bare(place) for place in named.split(", ")]
    return places, len(places) + more if more else None


_STORED_WARNING_TRANSLATIONS: Mapping[str, _Translation] = {
    "stat_vars_summed": _summed_step,
    "facet_default_selected": _chosen_step,
    "currency_unmapped_omitted": _conversion_gap,
    "deflator_missing_omitted": _conversion_gap,
    "deflator_projected": _projected_caveat,
    "withheld_by_registry": _withheld_caveat,
    "un_system_unavailable": _un_unread_caveat,
}
"""How a backfill states each code `WARNING_DESTINATIONS` routes to a step or caveat."""


# Variable facts


def variable_facts(
    *,
    definition: Mapping[str, object],
    trace: Mapping[str, object],
    code_index: CodeIndex,
    roles: Sequence[RoleFigure] | None,
) -> VariableFacts:
    """A `spec_` version's population, from its definition, its trace, the code index and the
    entities filling each role of its series, `None` where nothing recorded them.

    `shape` keeps each member the identity's shape names, by its graph property, and
    `variable` the one variable the trace resolved, when it resolved one. The definition fixes each role it binds to one entity and each constraint its slots bind
    to one code. A role, fixed or open, is named by its graph property, which an identity's
    `entities` key resolves to through `declared_role_for`, as at mint. Every other role in `roles` is an open dimension, and so is every other
    constraint the variables declare. Roles come first, those the definition's entities name
    in its order and then the rest in `roles` order; then the slots that leave their
    constraint open, in slot order; then the other constraints by name. A role's values
    count the variables with data that hold each entity (`role_dimensions`). A slot bound to a
    code with descendants narrows its dimension `within` that code. With `roles` `None`, each
    role the trace says the mint declared and the definition leaves open is a dimension whose
    values are unknown. A constraint's values count
    the variables with data that hold each, or every resolved variable where the trace records
    no observation counts. `dimensions`, role dimensions included, is `None` when the index
    holds none of the variables. A constraint dimension is `partial` when a variable with data
    is missing from the index.
    """
    identity = cast("Mapping[str, object]", definition["identity"])
    resolved = _resolved(trace)
    counts = [count for _, count in resolved]
    with_data = None if None in counts else sum(1 for count in counts if count)
    member_count = trace.get("family_member_count")
    admitted, admitted_from = (
        (member_count, "family_member_count")
        if isinstance(member_count, int)
        else (len(resolved), "resolved_variables")
    )
    counted = [ref for ref, count in resolved if count or with_data is None]
    rows = [row for ref in counted if (row := code_index.statvar_by_dcid(ref)) is not None]
    declared = _declared_roles(trace, roles or ())
    fixed, open_slots = _bindings(identity, code_index, declared=declared)
    dimensions = None
    if any(code_index.statvar_by_dcid(ref) is not None for ref, _ in resolved):
        dimensions = [
            *(
                role_dimensions(identity, roles)
                if roles is not None
                else _unread_role_dimensions(identity, declared)
            ),
            *_dimensions(
                rows,
                open_slots=open_slots,
                fixed={item.property for item in fixed},
                partial=len(rows) < len(counted),
            ),
        ]
    shape = cast("Mapping[str, object]", identity.get("shape", {}))
    return VariableFacts(
        shape=[
            FixedValue(property=prop, value=value)
            for member, prop in _SHAPE_PROPERTIES.items()
            if (value := bounded_text(shape.get(member))) is not None
        ],
        variable=bounded_text(resolved[0][0]) if len(resolved) == 1 else None,
        fixed=fixed,
        dimensions=dimensions,
        admitted_at_mint=admitted,
        admitted_from=admitted_from,
        with_data=with_data,
        family=flow_family(
            index=code_index,
            population_type=bounded_text(shape.get("population_type")),
            stat_vars=counted,
            members=[row.constraint_properties for row in rows],
        ),
    )


def _declared_roles(trace: Mapping[str, object], roles: Sequence[RoleFigure]) -> tuple[str, ...]:
    """The roles the version's series fill, else the roles its trace says the mint declared."""
    if roles:
        return tuple(figure.role for figure in roles)
    entries = cast("Sequence[Mapping[str, object]]", trace.get("resolved_stat_vars", ()))
    declared = (cast("Sequence[str]", entry.get("roles", ())) for entry in entries)
    return tuple(dict.fromkeys(role for named in declared for role in named))


def _bindings(
    identity: Mapping[str, object], code_index: CodeIndex, *, declared: tuple[str, ...]
) -> tuple[list[FixedValue], dict[str, str | None]]:
    """The values the identity fixes, in its order, and each slot it leaves open with the
    code it narrows that slot to. A role is named by the declared role its key resolves to,
    else by the key. Each property and ref passes through `bounded_text`, and one it leaves
    empty is left out."""
    fixed = [
        FixedValue(property=prop, value=value)
        for key, refs in cast("Mapping[str, Sequence[str]]", identity.get("entities", {})).items()
        if isinstance(refs, list)
        and len(refs) == 1
        and (prop := bounded_text(declared_role_for(key, declared) or key)) is not None
        and (value := bounded_text(refs[0])) is not None
    ]
    open_slots: dict[str, str | None] = {}
    for slot in cast("Sequence[Mapping[str, object]]", identity.get("slots", ())):
        if (prop := bounded_text(slot["property"])) is None:
            continue
        binding = cast("Mapping[str, object]", slot.get("binding", {}))
        state = binding.get("state")
        ref = bounded_text(binding.get("ref"))
        if state == "exact" and ref is not None:
            fixed.append(FixedValue(property=prop, value=ref))
        elif state == "value" and ref is not None:
            if len(code_index.expansion(ref)) > 1:
                open_slots[prop] = ref
            else:
                fixed.append(FixedValue(property=prop, value=ref))
        else:
            open_slots[prop] = None
    return fixed, open_slots


def role_dimensions(
    identity: Mapping[str, object], roles: Sequence[RoleFigure]
) -> list[DimensionFacts]:
    """Each role in `roles` the identity does not bind to one entity, as an open dimension.

    An identity's `entities` key resolves to a role through `declared_role_for`, as at mint.
    The roles it names come first, in its order, then the rest in `roles` order. A role
    holding more than `ROLE_VALUES_MAX` entities keeps those the most variables hold, ties by
    ref, and is `partial`; each kept value's count is whole. A role whose name `bounded_text`
    would change is left out, as the record cannot name it.
    """
    held = {figure.role: figure for figure in roles}
    return [_role_dimension(held[role]) for role in _open_roles(identity, tuple(held))]


def _unread_role_dimensions(
    identity: Mapping[str, object], declared: Sequence[str]
) -> list[DimensionFacts]:
    """Each role in `declared` the identity does not bind to one entity, ordered as
    `role_dimensions` orders them, as an open dimension whose values are unknown."""
    return [
        DimensionFacts(
            property=role, within=None, values=None, undeclared_variables=0, partial=False
        )
        for role in _open_roles(identity, tuple(declared))
    ]


def _open_roles(identity: Mapping[str, object], roles: tuple[str, ...]) -> list[str]:
    """The roles of `roles` the identity does not bind to one entity: those its `entities`
    name, in its order, then the rest in `roles` order. A role whose name `bounded_text` would
    change is left out."""
    held = tuple(role for role in roles if bounded_text(role) == role)
    bound: dict[str, int | None] = {}
    entities = cast("Mapping[str, object]", identity.get("entities", {}))
    for key, refs in entities.items():
        if (role := declared_role_for(key, held)) is not None:
            bound[role] = len(refs) if isinstance(refs, list) else None
    ordered = [*bound, *(role for role in held if role not in bound)]
    return [role for role in ordered if bound.get(role) != 1]


def _role_dimension(figure: RoleFigure) -> DimensionFacts:
    ranked = sorted(figure.values, key=lambda value: (-value.variables, value.ref))
    kept = {
        ref: value.variables
        for value in ranked[:ROLE_VALUES_MAX]
        if (ref := bounded_text(_bare(value.ref))) is not None
    }
    return DimensionFacts(
        property=figure.role,
        within=None,
        values=[DimensionValue(value=ref, variables=kept[ref]) for ref in sorted(kept)],
        undeclared_variables=0,
        partial=len(figure.values) > ROLE_VALUES_MAX,
    )


def _dimensions(
    rows: Sequence[StatVarRow],
    *,
    open_slots: Mapping[str, str | None],
    fixed: set[str],
    partial: bool,
) -> list[DimensionFacts]:
    declared = {prop for row in rows for prop in row.constraint_properties}
    unnamed = sorted(declared - fixed - set(open_slots))
    dimensions = []
    for prop in [*open_slots, *unnamed]:
        held: dict[str, int] = {}
        for row in rows:
            if (value := row.constraint_properties.get(prop)) is not None:
                held[value] = held.get(value, 0) + 1
        dimensions.append(
            DimensionFacts(
                property=prop,
                within=open_slots.get(prop),
                values=[
                    DimensionValue(value=value, variables=count)
                    for value, count in sorted(held.items())
                ],
                undeclared_variables=sum(
                    1 for row in rows if prop not in row.constraint_properties
                ),
                partial=partial,
            )
        )
    return dimensions


def _resolved(trace: Mapping[str, object]) -> list[tuple[str, int | None]]:
    """Each resolved variable's bare dcid and observation count, `None` where the trace
    records none."""
    entries = cast("Sequence[Mapping[str, object]]", trace.get("resolved_stat_vars", ()))
    resolved = []
    for entry in entries:
        count = entry.get("observation_count")
        resolved.append((_bare(cast(str, entry["ref"])), count if isinstance(count, int) else None))
    return resolved


# Series facts


def series_facts(figures: SeriesFigures, *, rows: int | None, periods: int | None) -> SeriesFacts:
    """A version's series facts, from its series figures and its row and period counts. Role
    names and source refs pass through `bounded_text`, since a published frame's cells hold
    longer text than a record does."""
    return SeriesFacts(
        series=figures.series,
        rows=rows,
        periods=periods,
        span=None if figures.span is None else Span(start=figures.span[0], end=figures.span[1]),
        grain=figures.grain,
        latest_period=figures.latest_period,
        series_reaching_latest=figures.series_reaching_latest,
        entities_by_role=[
            RoleCount(role=role, entities=count.entities)
            for count in figures.entities_by_role
            if (role := bounded_text(count.role)) is not None
        ],
        by_source_unit=[
            SourceUnitCount(
                source=None if count.source is None else bounded_text(_bare(count.source)),
                unit=unit_from_cell(count.unit),
                series=count.series,
            )
            for count in figures.by_source_unit
        ],
        status=StatusCounts(
            estimated=figures.status.estimated, provisional=figures.status.provisional
        ),
    )


def frame_figures(frame: Frame, *, lineage: FrameLineage) -> SeriesFigures:
    """The series figures of an `ops_` version's frame, folded as a mint folds its statistics
    tables, a series and the estimates continuing it counting once under the continuations its
    `lineage` stores (`FrameLineage.continuations`), as Presentation groups them."""
    sink = SeriesStatisticsSink(continuing=continued_series_of_entries(lineage.continuations))
    sink.write(compute_series_statistics_table(frame))
    return sink.finish().figures


def unit_from_cell(cell: str | Mapping[str, object] | None) -> Unit:
    """A frame `unit` cell's JSON form as a stored unit: a ref, a converted unit
    `{currency, prices, base_year}`, or a composed unit whose sides may count a variable. A ref
    passes through `bounded_text`."""
    if cell is None:
        return None
    if isinstance(cell, str):
        ref = bounded_text(_bare(cell))
        return None if ref is None else SimpleUnit(ref=ref)
    if "numerator" in cell:
        return ComposedUnit(
            numerator=_unit_side(cell["numerator"]), denominator=_unit_side(cell["denominator"])
        )
    return SimpleUnit.model_validate(dict(cell))


def _unit_side(side: object) -> SimpleUnit | VariableSide:
    if isinstance(side, str):
        return SimpleUnit(ref=bounded_text(_bare(side)) or _bare(side))
    side = cast("Mapping[str, object]", side)
    if "stat_var" in side:
        return VariableSide(stat_var=_bare(cast(str, side["stat_var"])))
    return SimpleUnit.model_validate(dict(side))


# Steps and caveats at mint


def mint_steps(*, trace: Mapping[str, object], definition: Mapping[str, object]) -> list[Step]:
    """The DFO's choices at the mint of a version with `trace` and `definition`, in order:
    each source chosen by the default policy over disagreeing candidates, naming the others in
    rank order where the trace records them (`trace.choices`), then each series the same
    source's estimates continue, with the continuing method and the periods it supplies
    (`trace.continuations`), then each conversion with what it left out (`left_out`), into
    the unit `definition` requests."""
    others_by_axis = {
        cast(str, choice["axis"]): [
            StepRef(ref=_bare(ref))
            for candidate in cast("Sequence[Mapping[str, object]]", choice["candidates"])[1:]
            if isinstance(ref := candidate.get("ref"), str)
        ]
        for choice in cast("Sequence[Mapping[str, object]]", trace.get("choices", ()))
    }
    steps: list[Step] = []
    for facet in cast("Sequence[Mapping[str, object]]", trace.get("facets", ())):
        if (
            facet.get("basis") == "deployment_fallback"
            and facet.get("selection_reason") == _CHOSEN_OVER_DISAGREEMENT
            and isinstance(ref := facet.get("ref"), str)
        ):
            others = others_by_axis.get(cast(str, facet.get("axis")), [])
            steps.append(
                Step(
                    id=_next_step_id(steps),
                    op=_CHOOSE_SOURCE_OP,
                    policy=_CHOOSE_SOURCE_POLICY,
                    choice=Choice(
                        chosen=StepRef(ref=_bare(ref)),
                        others=others[:_MAX_CHOICE_OTHERS],
                        ranking=list(CHOICE_RANKING),
                    ),
                )
            )
    for continuation in cast("Sequence[Mapping[str, object]]", trace.get("continuations", ())):
        steps.append(
            Step(
                id=_next_step_id(steps),
                op=CONTINUE_SERIES_OP,
                policy=_CONTINUE_SERIES_POLICY,
                params={"method": continuation["method"], "periods": continuation["periods"]},
            )
        )
    target = _conversion_target(definition)
    for entry in cast("Sequence[Mapping[str, object]]", trace.get("conversion") or ()):
        members = cast("Sequence[Mapping[str, object]]", entry.get("members", ()))
        native = {cast(str, member["native_unit"]) for member in members}
        step = Step(id=_next_step_id(steps), op=_CONVERT_OP, policy=_CONVERT_POLICY)
        update: dict[str, object] = {}
        if len(native) == 1:
            update["converted_from"] = SimpleUnit(ref=_bare(native.pop()))
        if target is not None:
            update["unit"] = target
        if left_out := entry.get("left_out"):
            update["left_out"] = [
                LeftOut.model_validate(item)
                for item in cast("Sequence[Mapping[str, object]]", left_out)
            ]
        steps.append(step.model_copy(update=update) if update else step)
    return steps


def _mint_caveats(facts: ManifestFacts, steps: Sequence[Step]) -> list[CaveatFact]:
    """Values a conversion priced from a projected rate or deflator are estimated. A
    conversion that left data out leaves a coverage gap pointing at it: one whose step counts
    what it left out, and every conversion of a version whose trace predates those counts and
    whose warnings name a conversion gap. Each place a national-currency binding served no row
    at is a coverage gap, one per role and reason (`trace.coverage_gaps`), and so are the places
    a registry withholding or an unanswered UN read left without data (`trace.origin_gaps`),
    naming the variable a withholding names."""
    trace = facts.trace
    conversions = [step for step in steps if step.op == _CONVERT_OP]
    entries = cast("Sequence[Mapping[str, object]]", trace.get("conversion") or ())
    caveats = [
        CaveatFact(kind="estimated", step=step.id, count=projected)
        for step, entry in zip(conversions, entries, strict=True)
        if (projected := cast("Mapping[str, int]", entry.get("projected", {})).get("count"))
    ]
    codes = {cast(str, warning["code"]) for warning in facts.warnings}
    counted = any("left_out" in entry for entry in entries)
    caveats.extend(
        CaveatFact(kind="coverage_gap", step=step.id)
        for step in conversions
        if (isinstance(step.left_out, list) and step.left_out)
        or (not counted and codes.intersection(_CONVERSION_GAP_CODES))
    )
    caveats.extend(
        _entities_caveat(cast("Sequence[str]", gap["entities"]))
        for gap in cast("Sequence[Mapping[str, object]]", trace.get("coverage_gaps", ()))
    )
    caveats.extend(
        _entities_caveat(
            cast("Sequence[str]", gap["entities"]),
            series=[_bare(variable)] if isinstance(variable := gap.get("variable"), str) else [],
        )
        for gap in _origin_gaps(facts)
    )
    return caveats


def _origin_gaps(facts: ManifestFacts) -> Sequence[Mapping[str, object]]:
    """The places the origin reads found the answer holds no data at (`trace.origin_gaps`)."""
    return cast("Sequence[Mapping[str, object]]", facts.trace.get("origin_gaps", ()))


def _entities_caveat(entities: Sequence[str], *, series: Sequence[str] = ()) -> CaveatFact:
    """A coverage gap naming every one of `entities` by bare dcid. They are places the request
    named, so the trace holding them bounds the caveat; the composer serves up to 50 and
    counts the rest."""
    return CaveatFact(
        kind="coverage_gap",
        series=list(series),
        entities=[_bare(entity) for entity in entities],
    )


def _conversion_target(definition: Mapping[str, object]) -> SimpleUnit | None:
    """The unit the definition asks values converted into, where it names an ISO 4217
    currency. A currency keyed by role names no one unit."""
    identity = cast("Mapping[str, object]", definition.get("identity", {}))
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets", {}))
    request = facets.get("unit", {}).get("request")
    if not isinstance(request, Mapping):
        return None
    request = cast("Mapping[str, object]", request)
    currency = request.get("currency")
    if not (isinstance(currency, str) and len(currency) == 3 and currency.isupper()):
        return None
    return SimpleUnit.model_validate(
        {key: request[key] for key in ("currency", "prices", "base_year") if key in request}
    )


def _next_step_id(steps: Sequence[Step]) -> str:
    return f"m{len(steps) + 1}"


# Computed facts


def computed_inputs(steps: Sequence[Step]) -> list[InputVersion]:
    """The versions a computed version was computed from: each version its `retrieve` steps
    name, once, in step order, with the id and label of the first step naming it."""
    retrieved: dict[str, InputVersion] = {}
    for step in steps:
        if step.op == _RETRIEVE_OP and isinstance(step.input, VersionRef):
            ref = step.input
            retrieved.setdefault(
                ref.content_version,
                InputVersion.model_validate(ref.model_dump(exclude={"at_mint"})),
            )
    return list(retrieved.values())


def _step_caveats(steps: Sequence[Step]) -> list[CaveatFact]:
    """A `coverage_gap` caveat pointing at each step that left data out."""
    return [
        CaveatFact(kind="coverage_gap", step=step.id)
        for step in steps
        if isinstance(step.left_out, list) and step.left_out
    ]


def composed_variables(
    entries: Sequence[Mapping[str, object]] | None,
) -> list[ComposedVariable] | None:
    """The composed variables a publish described, by bare dcid; `None` when it described
    none."""
    if not entries:
        return None
    return [
        ComposedVariable(
            variable=_bare(cast(str, entry["stat_var"])),
            relation=cast(str, entry["relation"]),
            first=[_bare(ref) for ref in cast("Sequence[str]", entry["first"])],
            second=[_bare(ref) for ref in cast("Sequence[str]", entry["second"])],
        )
        for entry in entries
    ]


# Evidence


def source_evidence(citation: Mapping[str, object]) -> list[SourceEvidence]:
    """Each cited source once, in citation order, with its name, licence, and the download
    date and pipeline steps of the documentation the mint read. A source the mint read no
    documentation for has neither."""
    seen: set[str] = set()
    sources = []
    for source in cast("Sequence[Mapping[str, object]]", citation.get("sources", ())):
        ref = cast(str, source["ref"])
        if ref in seen:
            continue
        seen.add(ref)
        documentation = cast("Mapping[str, object] | None", source.get("documentation"))
        sources.append(
            SourceEvidence(
                ref=ref,
                name=bounded_text(source.get("label")),
                licence=bounded_text(source.get("license")),
                downloaded=None
                if documentation is None
                else _leading_date(documentation.get("last_update")),
                pipeline_steps=None
                if documentation is None
                else _pipeline_steps(
                    cast(
                        "Sequence[Mapping[str, object]]",
                        documentation.get("transformation_log", ()),
                    )
                ),
            )
        )
    return sources


def named_refs(record: DescriptionRecord) -> list[str]:
    """The dcids a record's facts name, sorted: its one variable, fixed values, dimension
    values and the codes they narrow to, the refs inside its steps and the entities their
    sides hold (`bound_entities`), the variables a unit counts, such as the population of a
    per-capita denominator, and the series and entities its caveats name."""
    refs: set[str] = set()
    if (variables := record.variables) is not None:
        if variables.variable is not None:
            refs.add(variables.variable)
        refs.update(item.value for item in variables.fixed)
        for dimension in variables.dimensions or []:
            refs.update(value.value for value in dimension.values or ())
            if dimension.within is not None:
                refs.add(dimension.within)
    for step in record.known_steps:
        refs.update(ref.ref for ref in _step_refs(step))
        refs.update(bound_entities(step))
    units: list[Unit] = [count.unit for count in record.series.by_source_unit or []]
    for step in record.known_steps:
        units.extend(unit for unit in (step.unit, step.converted_from) if unit is not MISSING)
    for unit in units:
        if isinstance(unit, ComposedUnit):
            refs.update(
                side.stat_var
                for side in (unit.numerator, unit.denominator)
                if isinstance(side, VariableSide)
            )
    for caveat in record.caveats or []:
        refs.update(caveat.series)
        refs.update(caveat.entities)
    return sorted(refs)


def title_refs(record: DescriptionRecord) -> list[str]:
    """The dcids a record's title names (`description.compose.version_title`), sorted: its
    one variable, fixed values and the codes its dimensions narrow to. A computed version's
    title names its inputs by their versions' titles, so it names no dcid."""
    if (variables := record.variables) is None or record.computed is not None:
        return []
    refs = {item.value for item in variables.fixed}
    refs.update(
        dimension.within for dimension in variables.dimensions or [] if dimension.within is not None
    )
    if variables.variable is not None:
        refs.add(variables.variable)
    return sorted(refs)


def title_named(record: DescriptionRecord, *, index: CodeIndex, indexed: Vocabulary) -> set[str]:
    """The dcids a graph read labels for the record's title (`title_refs`): its variable,
    fixed values and the codes its dimensions narrow to, with the properties of those values,
    less what the code index words (`unlabelled`). A computed version's title names its inputs
    by their titles alone, so it names none."""
    variables = record.variables
    properties = (
        []
        if variables is None or record.computed is not None
        else [
            *(fixed.property for fixed in variables.fixed),
            *(d.property for d in variables.dimensions or [] if d.within is not None),
        ]
    )
    return unlabelled(title_refs(record), properties, index=index, indexed=indexed)


def unlabelled(
    refs: Iterable[str], properties: Iterable[str], *, index: CodeIndex, indexed: Vocabulary
) -> set[str]:
    """The `refs` the code index gives no label, and the `properties` no role, code system or
    measure axis names. `indexed` is a vocabulary of the code index's words alone."""
    return {dcid for dcid in refs if index.code_by_dcid(dcid) is None} | {
        prop
        for prop in properties
        if axis_for_role(prop) is None and indexed.axis_name(prop) is None
    }


def with_labels(
    record: DescriptionRecord, labels: Mapping[str, str], *, read: datetime.date
) -> DescriptionRecord:
    """`record` with the label, read on `read`, of each dcid its facts name that `labels`
    holds and its evidence does not. A label that is empty once its U+0000 characters are
    gone is left out."""
    held = {label.ref for label in record.evidence.labels or []}
    added = [
        LabelEvidence(ref=ref, label=label, read=read)
        for ref in named_refs(record)
        if ref not in held and (label := bounded_text(labels.get(ref))) is not None
    ]
    if not added:
        return record
    evidence = record.evidence.model_copy(
        update={
            "labels": sorted([*(record.evidence.labels or []), *added], key=lambda label: label.ref)
        }
    )
    return record.model_copy(update={"evidence": evidence})


def _step_refs(step: Step) -> Iterable[StepRef]:
    if isinstance(step.choice, Choice):
        yield step.choice.chosen
        yield from step.choice.others
    if isinstance(step.summed, Summed):
        yield from step.summed.variables


@cache
def method_status_build() -> str:
    """The content hash of the packaged measurement-method classification that status counts
    are read through."""
    data = (
        importlib.resources.files(_METHOD_STATUS_PACKAGE) / "data" / _METHOD_STATUS_FILE
    ).read_bytes()
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _index_stamp(code_index: CodeIndex) -> CodeIndexStamp:
    stamp = code_index.stamp
    return CodeIndexStamp(
        content_hash=stamp.content_hash,
        exported_at=stamp.exported_at,
        graph_endpoint=stamp.graph_endpoint,
    )


def _trace_index_stamp(trace: Mapping[str, object]) -> CodeIndexStamp | None:
    stamp = trace.get("code_index_stamp")
    return None if stamp is None else CodeIndexStamp.model_validate(stamp)


def _manifest_facts(version: VersionRecord) -> ManifestFacts:
    if not isinstance(version.facts, ManifestFacts):
        raise ValueError(f"{version.subject_id} holds no dataset version's facts")
    return version.facts


def _coverage_rows(coverage: Mapping[str, object]) -> int | None:
    rows = coverage.get("rows")
    return rows if isinstance(rows, int) else None


def _coverage_years(coverage: Mapping[str, object]) -> int | None:
    for dimension in cast("Sequence[Mapping[str, object]]", coverage.get("dimensions", ())):
        if dimension.get("label") == _YEARS_DIMENSION and isinstance(
            count := dimension.get("count"), int
        ):
            return count
    return None


def _leading_date(value: object) -> datetime.date | None:
    """The calendar date a documentation date starts with, or `None` when it starts with
    none."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value[:10])
    except ValueError:
        return None


def bounded_text(value: object, *, limit: int = TEXT_MAX_LENGTH) -> str | None:
    """`value` as a record stores text read at mint: without its U+0000 characters, clipped
    to `limit` characters ending in "…" when longer, and `None` when it is not text or holds
    nothing else. Every text the builders copy in passes through here, so text read from the
    graph or a source's document can never fail the record, whose insert would roll back
    the version row."""
    if not isinstance(value, str) or not (text := value.replace("\x00", "")):
        return None
    return clipped(text, limit=limit)


def _pipeline_steps(log: Sequence[Mapping[str, object]]) -> list[PipelineStep]:
    """The source pipeline's steps as the mint read them. A step without a title is left
    out, and one without a description reads as its title."""
    steps = []
    for entry in log:
        if (title := bounded_text(entry.get("transformation"))) is None:
            continue
        description = bounded_text(entry.get("description"), limit=LONG_TEXT_MAX_LENGTH)
        steps.append(PipelineStep(title=title, description=description or title))
    return steps


def _bare(ref: str) -> str:
    return ref.removeprefix(_DCID_PREFIX)
