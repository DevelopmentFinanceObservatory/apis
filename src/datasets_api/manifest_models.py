"""Runtime response models assembled by `datasets_api.manifest` and serialised by mint doors.

`datasets_api.schema.Manifest` remains the published open-dict schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from dfo_platform.roles import RoleAxis
from materialise.roles import axis_for_role


def _is_none(value: object) -> bool:
    return value is None


class DimensionNoun(BaseModel):
    """What one member of a flow dimension is called, and several of them."""

    model_config = ConfigDict(extra="forbid")
    singular: str
    plural: str


class CoverageDimension(BaseModel):
    """One coverage dimension. `axis` is `axis_for_role(label)`: the flow axis of a dimension
    labelled by a declared role, and `None` for `years` and unmapped roles. Validation derives
    it, so a stored body written without it carries it once loaded. `noun` is set at the mint
    from the version's flow family (`dfo_platform.roles.role_wording`), which the label and
    axis do not fix, so a body stored without it carries none (`year_coverage_counted`): a
    `None` noun is left out of every dump."""

    model_config = ConfigDict(extra="forbid")
    label: str
    count: int
    axis: RoleAxis | None = None
    noun: DimensionNoun | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "what the members this dimension counts are called, by the frame's flow family: "
            "provider and recipient for aid, creditor and debtor for debt. Absent for a "
            "dimension on no flow axis, such as `years`, and on a version stored without it"
        ),
    )

    @model_validator(mode="after")
    def _derive_axis(self) -> CoverageDimension:
        self.axis = axis_for_role(self.label)
        return self


def year_coverage_counted(coverage: Mapping[str, object]) -> bool:
    """Whether the mint that stored `coverage` counted its year members
    (`datasets_api.manifest.year_coverage`), stating them or finding none to state.

    That mint names the members of every dimension on a flow axis (`CoverageDimension.noun`)
    in the same assembly, so a flow dimension with no noun marks a coverage stored before
    versions held either. A coverage with no flow dimension has no year members to count and
    reads as counted.
    """
    dimensions = coverage.get("dimensions")
    return all(
        not isinstance(dimension, Mapping)
        or dimension.get("noun") is not None
        or not isinstance(label := dimension.get("label"), str)
        or axis_for_role(label) is None
        for dimension in (dimensions if isinstance(dimensions, list) else ())
    )


class YearCoverage(BaseModel):
    """What the coverage index counts in one calendar year over the series a frame's
    requests read: the series holding an observation in it and, under `role_counts`, the
    distinct members of each role those series carry. A role is keyed by the `label` of its
    entry in the frame's `coverage.dimensions`."""

    model_config = ConfigDict(extra="forbid")
    year: int
    series: int
    role_counts: dict[str, int]


class CoverageModel(BaseModel):
    """A frame's row and dimension counts and its applied window.

    `recent_years`, `elected_year` and `fullest_year` set the frame's one year beside the
    years around it (`datasets_api.manifest.year_coverage`). They are counted once, when
    the version is minted, and stored with it, all three or none, so every read of the
    version states the same ones. A `None` member is left out of every dump."""

    model_config = ConfigDict(extra="forbid")
    rows: int
    dimensions: list[CoverageDimension]
    window: dict[str, object] | None = None
    recent_years: list[YearCoverage] | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "three calendar years, newest first, each with the counts the coverage index "
            "held of the series the frame's run read when the version was minted. They end "
            "at the newest year a counted series held up to the year the index was built "
            "in, so a year after the frame's can hold data the frame leaves out. Stated "
            "for a latest window and for a named year at or after the year a latest window "
            "serves, and absent for an earlier named year"
        ),
    )
    elected_year: int | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "the year a latest window served when the version was minted. It is the "
            "frame's year under a latest window. A frame of a named year after it holds a "
            "year a latest window passes over"
        ),
    )
    fullest_year: YearCoverage | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "the year holding the most series of the ten calendar years ending at the "
            "newest of `recent_years`, stated when it is earlier than those three and the "
            "frame's year holds under the share of its series a latest window's year must hold"
        ),
    )


class CitedSourceNode(BaseModel):
    """The Data Commons `Source` node behind a cited provenance. On the UN
    System Data Commons it is the agency provenance itself."""

    model_config = ConfigDict(extra="forbid")
    ref: str
    label: str


class CitedDataCommons(BaseModel):
    """The Data Commons that serves a cited provenance."""

    model_config = ConfigDict(extra="forbid")
    id: Literal["dfo", "base", "un_system"]
    label: str


class TransformationStepModel(BaseModel):
    """One step the DFO pipeline applied to a source's data, in pipeline order."""

    model_config = ConfigDict(extra="forbid")
    transformation: str = Field(description="the step title, plain text")
    description: str = Field(description="what the step does, as CommonMark")


class ChangeModel(BaseModel):
    """One change in a release of a source's DFO pipeline."""

    model_config = ConfigDict(extra="forbid")
    kind: str = Field(
        description="the Keep a Changelog group, lowercase: added, changed, deprecated, "
        "removed, fixed or security"
    )
    note: str = Field(description="the change, as CommonMark")


class ReleaseModel(BaseModel):
    """One release of a source's DFO pipeline."""

    model_config = ConfigDict(extra="forbid")
    date: str = Field(description="the release date, YYYY-MM-DD")
    title: str = Field(description="the release title, plain text")
    changes: list[ChangeModel] = Field(default_factory=list)


class CitedDocumentation(BaseModel):
    """A cited source's documentation at first mint, read by
    `materialise.provenance_doc.read_provenance_doc`, the reader behind
    `GET /api/graph/documentation/{ref}`. Each member is `None` or empty where the graph
    documents nothing."""

    model_config = ConfigDict(extra="forbid")
    publisher: str | None = Field(default=None, description="the publishing organisation")
    description: str | None = Field(default=None, description="what the source holds")
    source_data_url: str | None = Field(
        default=None, description="where the raw data downloads from"
    )
    attribution: str | None = Field(
        default=None, description="the citation text the publisher asks for, verbatim"
    )
    update_cadence: str | None = Field(
        default=None,
        description="how often the source updates, as display text such as 'Quarterly'; "
        "an unrecognised ISO 8601 duration passes through as stored",
    )
    last_update: str | None = Field(
        default=None, description="the date of the DFO's last refresh of the source"
    )
    years_covered: str | None = Field(default=None, description="YYYY or YYYY-YYYY")
    granularity: str | None = Field(
        default=None, description="such as 'Annual, by provider and recipient'"
    )
    currencies: list[str] = Field(
        default_factory=list, description="sorted ISO 4217 codes of the source's currencies"
    )
    transformation_log: list[TransformationStepModel] = Field(default_factory=list)
    version_history: list[ReleaseModel] = Field(
        default_factory=list, description="the DFO pipeline's releases, newest first"
    )


class CitationSource(BaseModel):
    """One cited source, named through
    `materialise.provenance_doc.read_source_citations`. `label`, `url`,
    `source`, `data_commons`, the licence fields and `documentation` come
    from the origin catalogue where it lists `ref`, and from the graph
    otherwise: `license` is the licence's label, `license_type` its
    `licenseType` dcid and `license_url` its terms page. `license`, `license_type`,
    `license_url` and `url` are `None` where neither the catalogue nor the graph declares one. `source`
    and `data_commons` are `None` for a ref the catalogue does not list.
    `documentation` is the source's documentation at first mint, `None` when
    nothing documents the source or its graph read failed. A mint
    leaves a stored citation as it is. A bundle render fills every field but
    `ref`, `label` and `processing_level` that is `None` from a live read for
    that render only (`datasets_api.manifest.healed_citation`), and every set
    field wins.
    """

    model_config = ConfigDict(extra="forbid")
    ref: str
    label: str
    processing_level: str | None = None
    license: str | None = None
    license_type: str | None = None
    license_url: str | None = None
    url: str | None = None
    source: CitedSourceNode | None = None
    data_commons: CitedDataCommons | None = None
    documentation: CitedDocumentation | None = None


class CitationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    sources: list[CitationSource]

    def sources_by_ref(self) -> dict[str, CitationSource]:
        """`sources` keyed by `ref`. There is one entry per `(ref,
        processing_level)` pair, and no member but `processing_level`
        varies by processing level, so the first entry for a ref stands for
        all of them.
        """
        by_ref: dict[str, CitationSource] = {}
        for source in self.sources:
            by_ref.setdefault(source.ref, source)
        return by_ref


class CountedRefsModel(BaseModel):
    """The things of one kind a warning counts: how many, and the `dcid:` refs of the first
    five, or ten for a conversion's left-out entities (`materialise.codes.CountedRefs`). A
    message states the count and may name none, so a reader labels the refs. A group's members
    are in request order, entities sorted, and the provenances a registry withheld in the order
    they first occur across the withheld series, each list cut to its first five."""

    model_config = ConfigDict(extra="forbid")
    count: int = Field(description="how many there are")
    refs: list[str] = Field(
        description=(
            "the `dcid:` refs of the first five of them, ten for omitted_entities: a "
            "group's members in request order, entities sorted, withheld provenances in the "
            "order they first occur across the withheld series"
        )
    )


class EntityYearsModel(BaseModel):
    """One entity a conversion left rows out of, and the years of those rows
    (`materialise.codes.EntityYears`)."""

    model_config = ConfigDict(extra="forbid")
    ref: str = Field(description="the entity's `dcid:` ref")
    years: list[str] = Field(
        description="the years left out, as runs of consecutive years, `1990-2001` or `2004`"
    )


class CountedEntityYearsModel(BaseModel):
    """The entities a conversion left rows out of, each with its years: how many, and the
    first ten, sorted (`materialise.codes.CountedEntityYears`). The message states the count
    and names none, so a reader labels the refs."""

    model_config = ConfigDict(extra="forbid")
    count: int = Field(description="how many entities there are")
    entities: list[EntityYearsModel] = Field(description="the first ten of them, sorted")


class YearRangeModel(BaseModel):
    """A run of years, both bounds included (`materialise.codes.YearRange`)."""

    model_config = ConfigDict(extra="forbid")
    start: int | None = Field(
        default=None,
        exclude_if=_is_none,
        description="the first year; absent when the request left it open",
    )
    end: int | None = Field(
        default=None,
        exclude_if=_is_none,
        description="the last year; absent when the request left it open",
    )


class WarningModel(BaseModel):
    """One manifest warning. Only `stat_vars_summed` sets its three structured fields, a
    group's `coverage_below_scope` its four member fields, `coverage_below_scope`
    `unserved_entities`, `requested_years` and `served_years`, `currency_unmapped_omitted`
    `omitted_entities`, `deflator_missing_omitted` `omitted_entity_years`, and
    `withheld_by_registry` `unserved_entities`, `withheld_members` and `withheld_provenances`.
    A stored warning can lack a field its code carries, as a `withheld_by_registry` warning
    stored before these fields existed does while still withholding. A `None` field is left out of every dump, so other codes' warnings, on the wire and in stored version
    bodies, stay exactly `{code, severity, message}`.
    """

    model_config = ConfigDict(extra="forbid")
    code: str
    severity: str
    message: str
    contributing_member_count: int | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "stat_vars_summed only: how many StatVars with rows were summed into the "
            "collapsed frame's one series"
        ),
    )
    stat_var_ref: str | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "stat_vars_summed only: the family ref, `dcid:<spec_id>`, that the collapsed "
            "frame's `stat_var` column carries"
        ),
    )
    stat_var_label: str | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "stat_vars_summed only: the family's label, the string the presentation door "
            "gives `stat_var_ref` on a chart (`code_index.labels.family_label` over the spec "
            "identity: its measure partition label, then its bound slots' labels); absent "
            "when neither has a label"
        ),
    )
    member_count: int | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "coverage_below_scope only, when it names a group member: how many series the "
            "group requested"
        ),
    )
    off_cadence_members: CountedRefsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "coverage_below_scope only: the group members holding observations in the "
            "requested years at another frequency than the group serves, which the figures "
            "leave out; absent when there are none"
        ),
    )
    out_of_window_members: CountedRefsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "coverage_below_scope only: the group members holding no observation in the "
            "requested years; absent when there are none, and when the served provenance reads "
            "an absent member as zero"
        ),
    )
    unserved_members: CountedRefsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "coverage_below_scope only: the group members holding no observation under the "
            "source, unit and method the group serves, which the figures leave out; absent "
            "when there are none"
        ),
    )
    unserved_entities: CountedRefsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "coverage_below_scope and withheld_by_registry only: the entities the request binds "
            "that served no observation, sorted; absent when there are none"
        ),
    )
    requested_years: YearRangeModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "coverage_below_scope only: the year range the request asked for, when the data "
            "covers fewer of its years; absent otherwise"
        ),
    )
    served_years: YearRangeModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "coverage_below_scope only: the years the data covers, both bounds set, beside "
            "requested_years; absent otherwise"
        ),
    )
    withheld_members: CountedRefsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "withheld_by_registry only: the series the registry withheld; "
            "absent when there are none, and on a warning stored before this field existed"
        ),
    )
    withheld_provenances: CountedRefsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "withheld_by_registry only: the provenances holding data for the withheld series; "
            "absent when there are none, and on a warning stored before this field existed"
        ),
    )
    omitted_entities: CountedRefsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "currency_unmapped_omitted only: the entities whose rows the conversion left "
            "out, sorted"
        ),
    )
    omitted_entity_years: CountedEntityYearsModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "deflator_missing_omitted only: the entities whose rows the conversion left out "
            "for a missing deflator, sorted, each with the years of those rows"
        ),
    )


class DiagnosticModel(BaseModel):
    """One manifest diagnostic: a fact about how this response was produced that a developer
    reads and a dataset's description leaves out, since it states nothing about the data.
    `processing_level_undeclared` and `origin_catalogue_stale` are the codes it carries."""

    model_config = ConfigDict(extra="forbid")
    code: str
    severity: str
    message: str


class ArtifactEntryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_digest: str
    rendering: dict[str, object] = Field(
        description="the rendering that produced the bytes: the request options of a "
        'requested file, {"statistics": "series"} for the series statistics, or '
        '{"format": "frame"} for the version\'s canonical frame, a Parquet file of the frame '
        "as minted"
    )
    bytes: int
    media_type: str
    created_at: str
    tier: str
    url: str = Field(
        description="the artifact's stable download link, "
        "`/api/datasets/{id}/{content_version}/files/{artifact_digest}`, relative to the API "
        "host. It never expires: each request checks the version and answers 302 to a signed "
        "URL valid for five minutes, or refuses snapshot_revoked once the version is taken down"
    )
    enrichment_stamp: str | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "the digest of the code index's party kinds, classes and whole-total markers this "
            "file's party cells ({role}_kind, or {role}_party_kind, {role}_party_class and "
            "{role}_party_whole) were read under. Present on a file whose frame holds a party "
            "role and whose layout writes party cells (a long-layout file, or one whose columns "
            "name a party field), when it was rendered under the serving code "
            "index. Absent on every other file, including a pinned or older-layout file rendered "
            "under an earlier index, whose kinds may be out of date. A caller naming a frame by "
            "this file's rendering adds the stamp to the name, so a change to a party's kind names "
            "a new frame"
        ),
    )


class ColumnOptionModel(BaseModel):
    """One frame column, verbatim from `materialise.profile.ColumnProfile`."""

    model_config = ConfigDict(extra="forbid")
    name: str
    kind: str
    droppable: bool
    distinct: int | None = Field(default=None, exclude_if=_is_none)


class FormatOptionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    state: str
    reason: str | None = None


class BundleOptionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str
    reason: str | None = None


class EnrichmentCoverageCountsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: int
    covered: int


class MissingEntityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str
    label: str


class EnrichmentOptionModel(BaseModel):
    """One `(entity column, field)` enrichment this deployment enables,
    with `token` naming the exact `{column}_{field}` the `columns` option
    accepts to request it.
    """

    model_config = ConfigDict(extra="forbid")
    column: str
    field: str
    token: str
    state: str
    reason: str | None = None
    coverage: EnrichmentCoverageCountsModel
    missing: list[MissingEntityModel] = Field(default_factory=list)
    truncated: bool = False


class VaryingColumnModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    distinct: int


class UnavailableFormatModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: str
    reason: str


class PivotOptionModel(BaseModel):
    """One ordered `(index, columns)` pivot pair. `varying` is populated
    for a `pivot_not_unique` refusal caused by a duplicate key group. A
    stringification collision, where distinct `columns` values share an
    output column name, is also `unavailable` as `pivot_not_unique`, with
    no `varying` since no duplicate key group exists. `width` and `cells`
    are populated only when the pair is feasible.
    """

    model_config = ConfigDict(extra="forbid")
    index: str
    columns: str
    state: str
    reason: str | None = None
    varying: list[VaryingColumnModel] | None = None
    width: int | None = None
    cells: int | None = None
    unavailable_formats: list[UnavailableFormatModel] | None = None


class GroupbyOptionsModel(BaseModel):
    """`axes` names the axes a `groupby.by` request may keep: every other
    declared-summable axis a frame decoded is collapsed into it. Empty
    while `state` is `unavailable`.
    """

    model_config = ConfigDict(extra="forbid")
    state: str
    reason: str | None = None
    axes: list[dict[str, object]] = Field(default_factory=list)


class OutputOptionsModel(BaseModel):
    """The reshape options this frame supports, joined from its stored
    `materialise.profile.OutputProfile` and the current capability
    document.
    """

    model_config = ConfigDict(extra="forbid")
    capability_version: str
    columns: list[ColumnOptionModel]
    formats: list[FormatOptionModel]
    bundle: BundleOptionModel
    enrichments: list[EnrichmentOptionModel]
    long_layout_defaults: list[str] = Field(
        description=(
            "the enrichment tokens a long-layout download writes with no columns option, in "
            "column order: every role's {role}_label, and {role}_kind for every role on a flow "
            "axis, the party's kind in words: official or private on the provider or creditor "
            "side, country or organisation on the recipient or debtor side, total, whole total "
            "(a total that holds every party of its group in that row's dataset and year) or "
            "unallocated. Naming one in columns gives the same column once"
        )
    )
    long_layout_totals: list[str] = Field(
        description=(
            "the roles whose total rows ({role}_kind is total or whole total) a long-layout CSV "
            "or XLSX download keeps out of its main table. A private party stays in it. A role "
            "is listed when the definition "
            "binds it to several entities one of which the code index declares a total, or "
            "leaves it open and the code index lists a total on its axis in a dataset the "
            "definition read (any dataset, when the read selected none). A role bound to "
            "several parties holds no total row and is not listed. A bundle's CSV writes those "
            "rows to a second data file, data/<name>-totals.csv; an XLSX writes them on a "
            "second sheet, totals. A bare CSV has no README and drops them, so this list names "
            "the roles whose totals it drops. Parquet, a pivot and the frame layout keep every "
            "row. Empty when no role can hold a total row"
        )
    )
    pivots: list[PivotOptionModel]
    groupby: GroupbyOptionsModel


class DefinitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    spec_id: str
    identity: Mapping[str, object] = Field(
        description="the identity-core document, shaped as search.models.identity.BaseIdentity "
        "and published at GET /api/search/schema"
    )


class RetentionModel(BaseModel):
    """How long a version can still be fetched, which `datasets_api.retention` computes when a
    door serves the version. No stored body holds it."""

    model_config = ConfigDict(extra="forbid")
    pinned: bool = Field(
        description="whether the frame the version's files render from is in the pinned tier, "
        "which has no expiry"
    )
    held_until: AwareDatetime | None = Field(
        default=None,
        exclude_if=_is_none,
        description="when the working-tier frame stops being guaranteed, 30 days after it was "
        "stored. A download of the frame after `held_until`, while its bytes survive, renews it "
        "for 30 days. After it, a file selector or download link request "
        "that must render its file again renews the frame for 30 days if the frame's bytes "
        "survive, and otherwise refuses `artifact_expired`. A rebuild of unchanged data through "
        "`GET /api/datasets/{id}` or `POST /api/datasets` stores the frame again. Absent when "
        "pinned. It can be in the past",
    )


class Manifest(BaseModel):
    """`trace` stays a plain dict, since `materialise.trace.ManifestTrace.to_dict()`
    is already this field's exact wire shape.

    `series_count` is the number of distinct series this definition
    resolves, as the version's description counts them (`statistics.SeriesFigures.series`): a
    series and the estimates continuing it count once, though the statistics artifact holds a
    row for each method. `None` means a version record minted before this field existed, whose old
    inline `statistics` no fact carries. The per-series detail itself lives in the
    `series-statistics/1` Parquet artifact every door that returns a
    `Manifest` lists in `artifacts` whenever `series_count` is not `None`
    (`datasets_api.render.render_and_store_series_statistics`), never
    inline on this model.

    `output_options` is `None` on a version record minted before the field
    existed, the same convention `series_count` uses.

    `prepared_at` is when this version was first prepared, written once at its
    first mint and stored with it, so every re-mint and re-render reads the same
    instant.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    definition: DefinitionModel
    content_version: str
    prepared_at: AwareDatetime
    coverage: CoverageModel
    citation: CitationModel
    trace: dict[str, object]
    warnings: list[WarningModel] = Field(default_factory=list)
    diagnostics: list[DiagnosticModel] = Field(
        default_factory=list,
        description="facts about how this response was produced, for developers: a citation "
        "whose sources declare no processing level, and an origin catalogue past its "
        "staleness limit. A version stores none; each response states its own",
    )
    artifacts: list[ArtifactEntryModel] = Field(
        description="every servable artifact for this rendering, in a fixed order: every "
        "data artifact first, the series-statistics/1 artifact (rendering.statistics == "
        "'series') last, if present. A caller selecting the data artifact should check "
        "rendering rather than take the first entry, but the order itself is a stable "
        "contract, not an accident of storage"
    )
    series_count: int | None = None
    output_options: OutputOptionsModel | None = None
    series_coverage_stamp: str | None = Field(
        default=None,
        description=(
            "the series_coverage artifact stamp this request's sync-admission estimate "
            "was computed against, kept as lineage only. content_version is a "
            "pure function of the frame and does not fold this in. "
            "Not the same coverage as the 'coverage' "
            "field above (materialise.window.Coverage's row/dimension/window report)"
        ),
    )
    retention: RetentionModel | None = Field(
        default=None,
        exclude_if=_is_none,
        description="how long this version can still be fetched: the retention of the frame "
        "every file renders from (a derived version's base frame). Each door sets it on its "
        "response",
    )


class OptionsEnvelope(BaseModel):
    """`GET /api/datasets/{id}/options`'s response. It is the projection
    of a materialising door's manifest onto the fields this route's callers
    need (content_version, coverage, output_options, warnings, trace). The
    route serves with no rendering (`doors.definition._serve_spec_options`),
    so no reshape runs and `warnings` here excludes reshape-level warnings
    such as `enrichment_unavailable`.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    definition: DefinitionModel
    content_version: str
    coverage: CoverageModel
    output_options: OutputOptionsModel | None = None
    warnings: list[WarningModel] = Field(default_factory=list)
    trace: dict[str, object]
