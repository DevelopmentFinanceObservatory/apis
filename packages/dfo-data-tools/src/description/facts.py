"""The facts and evidence a description is composed from, as stored once per version.

A `DescriptionRecord` holds a Datasets version's facts and evidence, fixed when the version is
made. Words are joined when a description is served, so facts name things by dcid and the
evidence keeps the labels read at mint. A card's `SelectionFacts` hold what the card draws,
computed per card and never stored in the record.

A fact nothing recorded is `None`, never 0: status counts the table does not record,
`with_data` on a body stored without observation counts, a span the series statistics no
longer hold, the values of a role a backfill found no statistics for. Each such field is required, so a writer states an unknown rather than
defaulting into one. A dimension whose values a variable with data took with it when it
left the code index is marked `partial`.
"""

from __future__ import annotations

import datetime
from typing import Literal, Self

from pydantic import AwareDatetime, Field, StrictBool, field_validator, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from description.steps import Step, StepId, Unit
from description.stored import (
    ContentVersion,
    Count,
    FieldPath,
    LongText,
    Name,
    PartialDate,
    Period,
    Stored,
    Text,
    VersionId,
    same_version_family,
)
from dfo_platform.roles import FlowFamily

SchemaVersion = Literal[1, 2]
"""A record's schema version. Version 2 adds `DimensionFacts.within_set`."""

SCHEMA_VERSION: SchemaVersion = 2
"""The latest `schema_version` of the records this module defines. A record is stamped the
lowest version that describes it (`lowest_schema_version`), so an image of an earlier version
reads every record that holds nothing that version lacks. A version 1 record reads here as
version 2 with `within_set` unset (`datasets_api.description_store.upgraded_record`)."""

Basis = Literal["mint", "publish", "backfill"]
"""What wrote a record or filled a late field: a mint, a publish, or the backfill of a version
made before records existed."""


class Span(Stored):
    """A first and last period, both inclusive."""

    start: Period
    end: Period


class CodeIndexStamp(Stored):
    """A code index build, as the trace stamps it."""

    content_hash: Text
    exported_at: Text
    graph_endpoint: Text


# Variable facts


class FixedValue(Stored):
    """One property the definition fixes, and its value."""

    property: Text
    value: Text


class DimensionValue(Stored):
    """One value of an open dimension, and how many variables with data hold it."""

    value: Text
    variables: Count


class DimensionFacts(Stored):
    """One dimension the definition leaves open.

    `values` lists every value that variables with data hold, so the dimension's count is
    its length. It is `None` when nothing recorded them, as for a role on a backfilled
    version whose series statistics are not held. `partial` is true when a variable with data
    has left the code index since the version was made, so the survivors no longer give the
    whole count. `within` is the code the definition narrows the dimension to, and
    `within_set` the codes, two or more, a definition binding a set narrows it to, sorted; a
    record written before sets were recorded holds neither for a set. A dimension narrows to
    one or the other.
    """

    property: Text
    within: Text | None
    within_set: list[Text] | MISSING = MISSING
    values: list[DimensionValue] | None
    undeclared_variables: Count
    partial: StrictBool

    @model_validator(mode="after")
    def _one_narrowing(self) -> Self:
        if self.within_set is not MISSING and (
            self.within is not None
            or len(self.within_set) < 2
            or self.within_set != sorted(set(self.within_set))
        ):
            raise ValueError("a set narrows a dimension without `within`, to two sorted codes")
        return self

    @property
    def narrowed_to(self) -> list[str]:
        """The codes the definition narrows the dimension to: `within`, else `within_set`,
        else none."""
        if self.within is not None:
            return [self.within]
        return [] if self.within_set is MISSING else self.within_set


class VariableFacts(Stored):
    """The population of a `spec_` version.

    `shape` holds each member the definition's shape names, by graph property
    (`populationType`, `measuredProperty`, `statType`, `measurementQualifier`), in that order;
    a definition that pins one variable names none. The measure is named from `shape` and
    `fixed` together, as Search keys a measure partition. `variable` is the bare dcid of the
    one variable the definition resolved to, and `None` when it resolved to several or none;
    a title with no measure label names it. `fixed` keeps the definition's order. `dimensions` is `None` when the code index lists no
    family for the version's variables, as for a base family such as `Count_Person`, and on a
    backfilled collapse whose frame's columns nothing recorded.
    `admitted_at_mint` is the population the definition admitted; `admitted_from` says
    whether it is the family's member count or, on a body stored without one, the resolved
    variables. `with_data` is `None` on a body stored without observation counts. `family` is
    the flow family the variables count, which names their two parties ("debtor" and
    "creditor" for `debt_stock`), and `None` when the version's variables belong to none,
    whose parties read as recipient and provider.
    """

    shape: list[FixedValue]
    variable: Text | None
    fixed: list[FixedValue]
    dimensions: list[DimensionFacts] | None
    admitted_at_mint: Count
    admitted_from: Literal["family_member_count", "resolved_variables"]
    with_data: Count | None
    family: FlowFamily | None = None


def lowest_schema_version(variables: VariableFacts | None) -> SchemaVersion:
    """The lowest schema version that describes a record holding `variables`: 2 when a
    dimension is narrowed to a set (`DimensionFacts.within_set`), else 1."""
    if variables is not None and any(
        dimension.within_set is not MISSING for dimension in variables.dimensions or ()
    ):
        return 2
    return 1


class ComposedVariable(Stored):
    """A variable an analysis operation wrote rows under, and the variables it combines, as
    an `ops_` body stored them."""

    variable: Text
    relation: Name
    first: list[Text]
    second: list[Text]


class InputVersion(Stored):
    """One version an `ops_` version was computed from, as its `retrieve` step names it.

    `id` is `None` where the version holds no step that retrieves the input, as on a publish
    without steps or a backfill, which keep the content version alone. `label` is the label
    the step carried.
    """

    id: VersionId | None
    content_version: ContentVersion
    label: Text | MISSING = MISSING

    @model_validator(mode="after")
    def _one_family(self) -> Self:
        if self.id is not None and not same_version_family(
            id=self.id, content_version=self.content_version
        ):
            raise ValueError(f"{self.id} cannot carry content version {self.content_version}")
        return self


class ComputedFacts(Stored):
    """What an `ops_` version was computed from: its inputs, each at the version it used, and
    its composed variables. `inputs` are the versions its `retrieve` steps name, or, while its
    steps are unknown, the input versions its lineage lists. `composed` is `None` when the body
    stored none."""

    inputs: list[InputVersion]
    composed: list[ComposedVariable] | None


# Series facts


class StatusCounts(Stored):
    """How many values are estimated and how many provisional. A count is `None` when the
    data records no status to count it from."""

    estimated: Count | None
    provisional: Count | None


class RoleCount(Stored):
    """How many distinct entities fill one role, such as provider or recipient."""

    role: Text
    entities: Count


class SourceUnitCount(Stored):
    """How many series one source holds in one unit."""

    source: Text | None
    unit: Unit
    series: Count


class SeriesFacts(Stored):
    """A version's series and rows.

    `series` is known for a `spec_` version, and is `None` for a backfilled `ops_` version
    whose frame is no longer readable. The rest is `None` where the version's statistics or
    table no longer hold it: `latest_period` and `series_reaching_latest`, `span` and `grain`,
    and the counts by role, by source and unit, and by status.
    """

    series: Count | None
    rows: Count | None
    periods: Count | None
    span: Span | None
    grain: Name | None
    latest_period: Period | None
    series_reaching_latest: Count | None
    entities_by_role: list[RoleCount] | None
    by_source_unit: list[SourceUnitCount] | None
    status: StatusCounts


class CaveatFact(Stored):
    """A fact about the data and what it applies to: series, entities, periods, or the step
    that left data out. `count` says how many values or entities it applies to when more
    than it names. `entities` holds every place a coverage gap left without data, each one
    the request named, and the composer serves up to 50 of them. A caveat naming entities
    names at most one series, so its entities and `count` are that series' own and a card that
    narrows the caveat to the series it draws keeps them whole."""

    kind: Literal["estimated", "provisional", "break_in_series", "coverage_gap"]
    series: list[Text] = Field(default_factory=list, max_length=50)
    entities: list[Text] = Field(default_factory=list)
    periods: list[Period] = Field(default_factory=list, max_length=50)
    step: StepId | None = None
    count: Count | None = None

    @model_validator(mode="after")
    def _names_a_target(self) -> Self:
        if not (self.series or self.entities or self.periods or self.step is not None):
            raise ValueError("a caveat names series, entities, periods or a step")
        if self.entities and len(self.series) > 1:
            raise ValueError("a caveat naming entities names at most one series")
        return self


# Evidence


class LabelEvidence(Stored):
    """A dcid's label as read on `read`: the mint's date, or the backfill's for a label a
    backfill took from the code index."""

    ref: Text
    label: Text
    read: datetime.date


class PipelineStep(Stored):
    """One step the DFO pipeline applied to a source's data, as the mint read it."""

    title: Text
    description: LongText


class SourceEvidence(Stored):
    """One source as the mint read it.

    `name` and `licence` are what the version stored, which may be the bare ref. `downloaded`
    is the date the DFO pipeline last downloaded the source, and `pipeline_steps` the steps it
    applied, both `None` where the mint read no documentation.
    """

    ref: Text
    name: Text | None
    licence: Text | None
    downloaded: datetime.date | None
    pipeline_steps: list[PipelineStep] | None


class MembershipEvidence(Stored):
    """The membership policy a group's members were read under."""

    group: Text
    projection: Name
    as_of: PartialDate | None
    authority: Text


class Builds(Stored):
    """The artifact builds the version was made from. `None` where the version recorded
    none. `method_status` is the content hash of the measurement-method classification its
    status counts were read through."""

    origin_catalogue: Text | None
    code_index: CodeIndexStamp | None
    coverage: Text | None
    method_status: Text | None


class Evidence(Stored):
    """What a version was made from, as read when it was made.

    `prepared` is the date the version was prepared or computed, which its citation gives as
    the retrieval date. `title` is the title composed at mint. `source_release` is the
    publisher's release behind the version, where recorded. `membership` is `None` until a
    definition can bind a group id.
    """

    prepared: datetime.date
    title: Text | None
    sources: list[SourceEvidence]
    source_release: Text | None
    labels: list[LabelEvidence] | None
    membership: MembershipEvidence | None
    builds: Builds


# The record


class LateField(Stored):
    """A field written after the record, and when and by what it arrived."""

    field: FieldPath
    arrived: AwareDatetime
    basis: Basis


class DescriptionRecord(Stored):
    """A version's facts and evidence, keyed by `(id, content_version)`.

    `variables` is set on a `spec_` version and `computed` on an `ops_` version. `index_stamp`
    names the code index the dimension facts were derived through, and is set whenever they
    are. `steps` is `None` while no step records are held, and `[]` when the version made no
    step. An `ops_` record's caveats and label evidence derive from its steps, so a record
    without steps holds each as `None`. The write that fills `steps` fills them, and replaces
    the computed inputs its lineage listed with those its steps retrieve. `steps_after_unknown`
    holds the steps known to follow steps that are unknown, such as the sum of a collapse whose
    base version's steps are unknown, and is set only while `steps` is `None`. A caveat's
    `step` names one of the record's steps. `late_fields` notes each field filled after the
    record was written. `schema_version` is at least the lowest version that describes the
    record (`lowest_schema_version`): a writer stamps that version, and a read lifts an
    earlier version's row to this one.
    """

    id: VersionId
    content_version: ContentVersion
    schema_version: SchemaVersion
    basis: Basis
    index_stamp: CodeIndexStamp | None
    variables: VariableFacts | None
    computed: ComputedFacts | None
    series: SeriesFacts
    steps: list[Step] | None
    steps_after_unknown: list[Step] | MISSING = MISSING
    caveats: list[CaveatFact] | None
    evidence: Evidence
    late_fields: list[LateField]

    @property
    def known_steps(self) -> list[Step]:
        """The steps the record knows were taken, in order: `steps`, else the steps known to
        follow unknown ones (`steps_after_unknown`), else none."""
        if self.steps is not None:
            return self.steps
        return [] if self.steps_after_unknown is MISSING else self.steps_after_unknown

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("schema_version is an integer")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not same_version_family(id=self.id, content_version=self.content_version):
            raise ValueError(f"{self.id} cannot carry content version {self.content_version}")
        is_spec = self.id.startswith("spec_")
        if is_spec != (self.variables is not None) or is_spec == (self.computed is not None):
            raise ValueError("a spec_ record holds variables, an ops_ record holds computed")
        if (
            self.variables is not None
            and self.variables.dimensions is not None
            and self.index_stamp is None
        ):
            raise ValueError("dimension facts name the code index they were derived through")
        if (is_spec or self.steps is not None) and (
            self.caveats is None or self.evidence.labels is None
        ):
            raise ValueError(
                "only an ops_ record whose steps are unknown holds unknown caveats or labels"
            )
        if self.steps_after_unknown is not MISSING and (
            self.steps is not None or not self.steps_after_unknown
        ):
            raise ValueError("steps after unknown ones follow unknown steps and are not empty")
        held = {step.id for step in self.known_steps}
        if any(
            caveat.step is not None and caveat.step not in held for caveat in self.caveats or []
        ):
            raise ValueError("a caveat names a step the record holds")
        late = [late.field for late in self.late_fields]
        if len(late) != len(set(late)):
            raise ValueError("a late field is filled once")
        if self.schema_version < lowest_schema_version(self.variables):
            raise ValueError("a record narrowing a dimension to a set is of schema version 2")
        return self


# Card selection facts


class Coordinate(Stored):
    """One frame column of a drawn series, and its value."""

    column: Text
    value: Text | None


class DrawnSeries(Stored):
    """One series a card draws.

    `first_period` and `last_period` bound the values inside the card's window, and are
    `None` when the window holds no value. `values` counts the points with a value and
    `missing` the points without one.
    """

    id: Text
    coordinates: list[Coordinate]
    source: Text | None
    unit: Unit
    first_period: Period | None
    last_period: Period | None
    values: Count
    missing: Count
    status: StatusCounts

    @model_validator(mode="after")
    def _periods_follow_values(self) -> Self:
        if (self.first_period is None) != (self.last_period is None) or (
            self.first_period is None
        ) != (self.values == 0):
            raise ValueError("a series with values has a first and last period")
        return self


class SelectionFacts(Stored):
    """What a card draws: its window, and each drawn series."""

    window: Span
    series: list[DrawnSeries]
