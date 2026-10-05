"""Runtime response models assembled by `datasets_api.manifest` and serialised by mint doors.

`datasets_api.schema.Manifest` remains the published open-dict schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from dfo_platform.roles import RoleAxis
from materialise.roles import axis_for_role


class CoverageDimension(BaseModel):
    """One coverage dimension. `axis` is `axis_for_role(label)`: the flow axis of a dimension
    labelled by a declared role, and `None` for `years` and unmapped roles. Validation derives
    it, so a stored body written without it carries it once loaded."""

    model_config = ConfigDict(extra="forbid")
    label: str
    count: int
    axis: RoleAxis | None = None

    @model_validator(mode="after")
    def _derive_axis(self) -> CoverageDimension:
        self.axis = axis_for_role(self.label)
        return self


class CoverageModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: int
    dimensions: list[CoverageDimension]
    window: dict[str, object] | None = None


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
    `source` and `data_commons` come from the origin catalogue's provenance
    table where it lists `ref`, and the licence fields from the graph:
    `license` is the licence's label, `license_type` its `licenseType` dcid
    and `license_url` its terms page. `license`, `license_type`,
    `license_url` and `url` are `None` where neither the catalogue nor the graph declares one. `source`
    and `data_commons` are `None` for a ref the catalogue does not list.
    `documentation` is the source's documentation at first mint, `None` when
    the graph documents nothing for the source or its read failed. A mint
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


def _is_none(value: object) -> bool:
    return value is None


class WarningModel(BaseModel):
    """One manifest warning. Only `stat_vars_summed` sets the three structured fields. A `None`
    field is left out of every dump, so other codes' warnings, on the wire and in stored version
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


class ArtifactEntryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_digest: str
    rendering: dict[str, object]
    bytes: int
    media_type: str
    created_at: str
    tier: str
    url: str
    url_expires_at: str


class ColumnOptionModel(BaseModel):
    """One frame column, verbatim from `materialise.profile.ColumnProfile`."""

    model_config = ConfigDict(extra="forbid")
    name: str
    kind: str
    droppable: bool
    distinct: int | None


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
    bare stringification collision carries no inspectable group and is
    pruned before this model is built. `width` and `cells` are populated
    only when the pair is feasible.
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


class CollectionMemberModel(BaseModel):
    """One collection member. `prepared_at` is when the member's own version was first
    prepared, which can precede the collection's when the member was minted alone first.
    `provenance_refs` are the refs of the member's own sources, which key the collection's
    `citation` and name the member's bundle file and bare artifacts."""

    model_config = ConfigDict(extra="forbid")
    definition: DefinitionModel
    content_version: str
    prepared_at: AwareDatetime
    provenance_refs: list[str]
    coverage: CoverageModel
    series_count: int | None = None
    output_options: OutputOptionsModel | None = None


class CollectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: str
    members: list[CollectionMemberModel]


class Manifest(BaseModel):
    """Exactly one of `definition`/`collection` present. `trace` stays a
    plain dict -- `materialise.trace.ManifestTrace.to_dict()` is already
    this field's exact wire shape.

    `series_count` is the number of distinct series this definition
    resolves, and lives at this top level only for a definition manifest; a
    collection manifest carries its own per-member `series_count` on each
    `CollectionMemberModel` instead, and this top-level field stays `None`
    (see `_series_count_only_at_the_right_level`). `None` also means a
    version record minted before this field existed (`datasets_api.manifest
    .decode_version_body` drops such a record's old inline `statistics` key
    rather than translating it). The per-series detail itself lives in the
    `series-statistics/1` Parquet artifact every door that returns a
    `Manifest` lists in `artifacts` whenever `series_count` is not `None`
    (`datasets_api.render.render_and_store_series_statistics`), never
    inline on this model.

    `output_options` is `None` on a version record minted before the field
    existed, the same convention `series_count` uses. A collection manifest
    carries its own `output_options` per member on `CollectionMemberModel`,
    since reshapes apply per member. This top-level field is a definition
    manifest's document.

    `prepared_at` is when this version was first prepared, written once at its
    first mint and stored with it, so every re-mint and re-render reads the same
    instant.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    definition: DefinitionModel | None = None
    collection: CollectionModel | None = None
    content_version: str
    prepared_at: AwareDatetime
    coverage: CoverageModel
    citation: CitationModel
    trace: dict[str, object]
    warnings: list[WarningModel] = Field(default_factory=list)
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
            "pure function of the frame and does not fold this in; "
            "revision_provenance_live stays false. Not the same coverage as the 'coverage' "
            "field above (materialise.window.Coverage's row/dimension/window report)"
        ),
    )

    @model_validator(mode="after")
    def _definition_xor_collection(self) -> Manifest:
        if (self.definition is None) == (self.collection is None):
            raise ValueError("exactly one of 'definition' or 'collection' must be present")
        return self

    @model_validator(mode="after")
    def _output_options_only_at_the_right_level(self) -> Manifest:
        if self.collection is not None and self.output_options is not None:
            raise ValueError(
                "top-level 'output_options' is only for a definition manifest; "
                "a collection manifest carries output_options per member"
            )
        return self

    @model_validator(mode="after")
    def _series_count_only_at_the_right_level(self) -> Manifest:
        if self.collection is not None and self.series_count is not None:
            raise ValueError(
                "top-level 'series_count' is only for a definition manifest; "
                "a collection manifest carries series_count per member"
            )
        return self


class OptionsEnvelope(BaseModel):
    """`GET /api/datasets/{id}/options`'s response. It is the write-free
    projection of a materialising door's manifest onto the fields this
    route's callers need (content_version, coverage, output_options,
    warnings, trace). Built from the pre-write half of `serve_definition`/
    `serve_collection` (`datasets_api.serve.prewrite_definition`/
    `prewrite_collection`), which runs no reshape, so `warnings` here
    excludes reshape-level warnings such as `enrichment_unavailable`.

    A collection request carries `output_options` per member on
    `CollectionModel.members`, and the top-level field stays forbidden
    once `collection` is set.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    definition: DefinitionModel | None = None
    collection: CollectionModel | None = None
    content_version: str
    coverage: CoverageModel
    output_options: OutputOptionsModel | None = None
    warnings: list[WarningModel] = Field(default_factory=list)
    trace: dict[str, object]

    @model_validator(mode="after")
    def _definition_xor_collection(self) -> OptionsEnvelope:
        if (self.definition is None) == (self.collection is None):
            raise ValueError("exactly one of 'definition' or 'collection' must be present")
        return self

    @model_validator(mode="after")
    def _output_options_only_at_the_right_level(self) -> OptionsEnvelope:
        if self.collection is not None and self.output_options is not None:
            raise ValueError(
                "top-level 'output_options' is only for a definition envelope; "
                "a collection envelope carries output_options per member"
            )
        return self
