"""Wire models for `GET /api/graph/documentation/{dcid}`.

Every top-level member is present. Missing scalars are `null`, missing arrays
are `[]`, and `gaps` names each empty member in spec order.
`document_errors` lists each Markdown document the source carries but this
door could not serve.

The graph fills `details`, `license` and `scope_and_units.years_covered` from
Dataset and Provenance properties (`materialise.provenance_doc`), except
`details.source` and `details.data_commons`, which come from the origin
catalogue's provenance table.
`details.citation` is the `licenseAttribution` text.
`details.source_data_url` is the `sourceDataUrl` file the pipeline downloads,
and `license.url` is the `license` page on the publisher's site. Both serve
the stored string when every Provenance child carries the same one (the
`license.type` rule) and null otherwise. `transformation_log`,
`version_history`, `methodology.description` and `data_notes` come from
Markdown documents on a sole child Provenance, parsed against their templates
by `dfo_provenance_docs`. Their text members carry CommonMark with paragraph
soft-wraps joined by a space.

`scope_and_units.currencies` lists the sorted ISO 4217 codes of every currency
facet the source's Provenances carry in the coverage index, and is empty when
there are none. `scope_and_units.granularity` is a phrase such as
`Annual, by provider and recipient` when every Provenance has coverage series
and together they share one observation period and one pair of flow axes, and
null otherwise.
`scope_and_units.countries` is always empty because the graph has no link
from a source to its StatVars.
Both usage entries are null because no source-dcid request can derive a
runnable sample.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from graph_api.origin_models import DataCommonsRef, SourceNodeRef
from materialise.provenance_doc import DocumentMember

# `dfo_provenance_docs.DocumentErrorReason` values, plus `conflicting_values`
# for a node carrying two distinct values of one document property.
# `tests/graph_api/test_source_models.py` holds this set equal to the grammar's.
DocumentErrorReasonWire = Literal[
    "undecodable",
    "empty_document",
    "content_outside_section",
    "unexpected_heading_level",
    "heading_not_plain_text",
    "empty_heading",
    "empty_section",
    "disallowed_block",
    "raw_html",
    "hard_break",
    "invalid_release_heading",
    "missing_release_title",
    "release_dates_not_descending",
    "duplicate_release_date",
    "unknown_change_kind",
    "duplicate_change_kind",
    "not_single_bullet_list",
    "list_item_not_single_paragraph",
    "document_too_large",
    "nesting_too_deep",
    "conflicting_values",
]

# `dfo_provenance_docs.ChangeKind` values, the Keep a Changelog groups in lower case.
ChangeKindWire = Literal["added", "changed", "deprecated", "removed", "fixed", "security"]


class SourceDetails(BaseModel):
    name_of_source: str | None
    publisher: str | None
    source: SourceNodeRef | None = Field(
        description=(
            "the origin registry's source for the documented Provenances, null "
            "unless the origin catalogue admits every one of them and they share one"
        )
    )
    data_commons: DataCommonsRef | None = Field(
        description=(
            "the Data Commons serving the documented Provenances, null unless the "
            "origin catalogue admits every one of them and they share one"
        )
    )
    description: str | None
    source_url: str | None
    source_data_url: str | None = Field(
        description="the file the pipeline downloads, the source's `sourceDataUrl`"
    )
    citation: str | None = Field(description="the source's `licenseAttribution` text")


class SourceLicense(BaseModel):
    type: str | None
    url: str | None = Field(
        description="the licence text on the publisher's site, the source's `license`"
    )
    update_cadence: str | None
    last_update: str | None
    next_expected_update: str | None


class ScopeAndUnits(BaseModel):
    """`countries` is always `[]`. `years_covered` is filled where the graph
    has it. `currencies` lists every currency unit the coverage index gives the
    source's Provenances, sorted, and `granularity` is filled where it gives
    one period and axis pair for the source.
    """

    countries: list[str] = Field(default_factory=list)
    years_covered: str | None = None
    granularity: str | None = None
    currencies: list[str] = Field(default_factory=list)


class Methodology(BaseModel):
    description: str | None = Field(
        default=None,
        description="how the publisher collects and compiles the data, as CommonMark",
    )


class UsageEntry(BaseModel):
    """The shape a non-null `Usage.api_access`/`.python` entry takes. Both
    of its own members are always `None`, and no dcid produces an entry at
    all -- `graph_api.usage`'s own docstring is the record of why.
    """

    documentation_url: str | None = None
    code: str | None = None


class Usage(BaseModel):
    """`api_access`/`python` are `None` for every source -- an absent
    member, not a stand-in object -- named in `gaps` alongside
    every other absent member. Built by `graph_api.usage.usage_document`,
    never constructed inline at a call site.
    """

    api_access: UsageEntry | None = None
    python: UsageEntry | None = None


class TransformationLogEntry(BaseModel):
    """One step of the pipeline's transformations, in pipeline order."""

    transformation: str = Field(description="the step title, plain text")
    description: str = Field(description="what the step does, as CommonMark")


class ChangeEntry(BaseModel):
    """One change in a release."""

    kind: ChangeKindWire = Field(description="the Keep a Changelog group the change sits under")
    note: str = Field(description="the change, as CommonMark")


class VersionHistoryEntry(BaseModel):
    """One release of the pipeline's changelog, newest first."""

    date: str = Field(description="the release date, YYYY-MM-DD")
    title: str = Field(description="the release title, plain text")
    changes: list[ChangeEntry] = Field(description="every change, group by group in order")


class DocumentErrorEntry(BaseModel):
    """A Markdown document the source carries that this door could not serve.
    The member it fills is empty and named in `gaps`.
    """

    member: DocumentMember = Field(description="the wire path the document fills")
    line: int | None = Field(
        description="1-based line in the decoded document where the problem starts, or null"
    )
    reason: DocumentErrorReasonWire = Field(description="the template rule the document breaks")


class SourceDocumentation(BaseModel):
    """The full `GET /api/graph/documentation/{dcid}` response."""

    details: SourceDetails
    license: SourceLicense
    scope_and_units: ScopeAndUnits = Field(default_factory=ScopeAndUnits)
    methodology: Methodology
    data_notes: list[str] = Field(
        default_factory=list, description="one caveat per entry, as CommonMark"
    )
    usage: Usage = Field(default_factory=Usage)
    transformation_log: list[TransformationLogEntry] = Field(default_factory=list)
    version_history: list[VersionHistoryEntry] = Field(default_factory=list)
    gaps: list[str] = Field(
        description=(
            "every member path that is null or empty for this source, in spec "
            "order, e.g. 'details.citation', 'scope_and_units.countries'"
        )
    )
    document_errors: list[DocumentErrorEntry] = Field(
        default_factory=list,
        description="each Markdown document the source carries but this door could not serve",
    )


class ProvenanceEntry(BaseModel):
    """One admitted provenance: its dcid and the provenance table's name for it."""

    ref: str
    label: str


class Exclusion(BaseModel):
    """A provenance a base source holds that this deployment does not admit, because
    another Data Commons supersedes it."""

    ref: str
    reason: Literal["superseded_by_dfo", "superseded_by_un_system"] = Field(
        description=(
            "superseded_by_dfo: an upstream copy of a dataset DFO imports. "
            "superseded_by_un_system: a UN System Data Commons agency's series replace it"
        )
    )


class DatasetEntry(BaseModel):
    """A Dataset and the admitted provenances that belong to it. A provenance with
    no Dataset forms its own entry, keyed by the provenance, so `ref` and `label`
    are the provenance's and `provenances` holds that one provenance.

    The card facts (`description`, `license_type`, `update_cadence`,
    `years_covered`, `source_url`) equal the same members of
    `GET /api/graph/documentation/{ref}` for the graph the loaded catalogue
    build read. They are null on UN System Data Commons entries."""

    ref: str
    label: str
    provenances: list[ProvenanceEntry]
    description: str | None = Field(
        description="its sole Provenance's description, as `details.description`; null when "
        "it has several"
    )
    license_type: str | None = Field(
        description="licence label every Provenance agrees on, as `license.type`; null when "
        "they disagree or one has none"
    )
    update_cadence: str | None = Field(
        description="release cadence label every Provenance agrees on, as "
        "`license.update_cadence`; null when they disagree or one has none"
    )
    years_covered: str | None = Field(
        description="`YYYY` or `YYYY-YYYY` span of its Provenances' observation dates, as "
        "`scope_and_units.years_covered`; null when either end is unknown"
    )
    source_url: str | None = Field(
        description="the publication's page, as `details.source_url`: the Dataset's url, "
        "else its sole Provenance's"
    )


class SourceCatalogueEntry(BaseModel):
    """One source on one Data Commons, with the datasets this deployment
    admits under it and the provenances it excludes. DFO provenances and UN
    agencies exclude nothing here."""

    source: SourceNodeRef
    data_commons: DataCommonsRef
    datasets: list[DatasetEntry]
    exclusions: list[Exclusion]


class SourceCatalogue(BaseModel):
    """The full `GET /api/graph/sources` response: DFO's sources, then the
    origin registry's base Data Commons entries and UN System Data Commons
    agencies in registry order, each with the provenances the request's
    origin catalogue admits under it, grouped by Dataset. UN agencies are left
    out when the UN origin is off. The response has no count and no pagination."""

    sources: list[SourceCatalogueEntry]
