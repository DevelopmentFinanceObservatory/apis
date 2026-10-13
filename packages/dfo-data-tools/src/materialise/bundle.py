"""Builds Data Package 2.0 bundles as an in-memory `path -> bytes` mapping.

`assemble_bundle` builds a Datasets bundle, and `assemble_component_bundle` builds the bundle for
one presentation card revision's exported rows. Each returns `datapackage.json`, `README.md`
and `SOURCES.md` (from `materialise.bundle_docs`), `CITATION.cff` (from
`materialise.bundle_citation`), and `data/codes.csv` when the caller passes codes. The caller
writes each member's data file at its `ArtifactEntry.path`; its `digest` becomes the resource
`hash` verbatim.

A bundle is a pure function of its inputs. Nothing here reads a clock, so the same inputs give
the same bytes. The caller reads two inputs from the graph, source documentation and code
labels, so a render matches an earlier one while those are unchanged. Runtime imports fit the
`datasets` extra.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from dfo_platform.codes import CodeCatalogue
from dfo_platform.publisher import DFO_NAME
from materialise.bundle_citation import build_citation_cff
from materialise.bundle_docs import blank_to_none, build_readme, build_sources, site_path
from materialise.cell_safety import spreadsheet_safe
from materialise.errors import MaterialisationUsageError
from materialise.license_type import licence_ids, licence_title

DATAPACKAGE_PROFILE_URL = "https://datapackage.org/profiles/2.0/datapackage.json"
"""Data Package 2.0 `$schema` URL. Python frictionless 5.19 recognises only the
legacy `profile` keyword. CRAN frictionless 1.3.0 requires resource-level
`profile: "tabular-data-resource"`. Every resource carries that field alongside v2 `type`.
"""

DFO_EMAIL = "data@one.org"
"""The DFO's contact address."""

CODES_PATH = "data/codes.csv"
"""Bundle path of the codes table."""

CODES_RESOURCE_NAME = "codes"
"""Resource name of the codes table, which data resources' foreign keys reference."""

BUNDLE_CODE_CATALOGUE: CodeCatalogue = {
    "processing_level_undeclared": (
        "Processing Level Undeclared",
        "A source StatVar lacks a processing-level declaration; the citation "
        "credits the source and omits the processing qualifier.",
    ),
    "observation_attributes_in_notes": (
        "Observation Attributes In Notes",
        "Per-observation qualifications (provisional, estimated) ride the "
        "bundle's README and descriptor metadata rather than inline frame columns.",
    ),
}
"""Registry fragment for two Datasets `info` warnings. `datasets_api.manifest` emits
`processing_level_undeclared`. `observation_attributes_in_notes` stays registered in
dfo-spec-kit and has no emitter."""


@dataclass(frozen=True)
class ArtifactEntry:
    """One rendered data file of a member."""

    path: str
    """Bundle-internal path the caller writes the file at, such as `data/<slug>.csv`. Its
    extension is the resource `format`, and the path minus `data/` and the extension, with `/`
    as `-`, is the resource `name`."""
    bytes: int
    """Size of the file in bytes."""
    digest: str
    """`sha256:<hex>` over the file's exact bytes, written as the resource `hash`."""
    media_type: str
    """IANA media type, written as the resource `mediatype`."""
    table_schema: Mapping[str, object]
    """Table Schema with `fields` of `{name, type}`; `ColumnDoc`s add titles and descriptions."""


@dataclass(frozen=True)
class Licence:
    """A source's licence as the graph declares it. Every field is `None` where undeclared."""

    type_id: str | None = None
    """The `licenseType` dcid, such as `CreativeCommons0`, which `materialise.license_type`
    maps to Open Definition and SPDX ids."""
    label: str | None = None
    """The licence node's graph `name`, else its dcid."""
    url: str | None = None
    """The licence's terms page (the graph `license` property)."""


@dataclass(frozen=True)
class TransformationDoc:
    """One step the DFO applied to a source's data."""

    title: str
    """The step heading, such as "Aggregate totals"."""
    description: str
    """The step body as CommonMark."""


@dataclass(frozen=True)
class ChangeDoc:
    """One change in a release of a source's DFO pipeline."""

    kind: str
    """The Keep a Changelog group, lowercase: `added`, `changed`, `deprecated`, `removed`,
    `fixed` or `security`."""
    note: str
    """The change as CommonMark."""


@dataclass(frozen=True)
class ReleaseDoc:
    """One release of a source's DFO pipeline."""

    date: str
    """The release date, `YYYY-MM-DD`."""
    title: str
    """The release title."""
    changes: tuple[ChangeDoc, ...] = ()
    """The release's changes in document order."""


@dataclass(frozen=True)
class SourceDocumentation:
    """A source's documentation as it stood at first mint. Every field is empty where the
    graph documents nothing."""

    publisher: str | None = None
    """The publishing organisation's name, such as "Organisation for Economic Co-operation and
    Development (OECD)", written as the CFF reference author and a `creator` contributor."""
    description: str | None = None
    """What the source holds, as plain text."""
    source_data_url: str | None = None
    """Where the raw data downloads from."""
    attribution: str | None = None
    """The citation text the publisher asks for, quoted verbatim."""
    update_cadence: str | None = None
    """How often the source updates, as display text such as "Quarterly"."""
    last_update: str | None = None
    """The date of the DFO's last refresh, `YYYY-MM-DD`."""
    years_covered: str | None = None
    """The years the source covers, `YYYY` or `YYYY-YYYY`."""
    granularity: str | None = None
    """A display phrase such as "Annual, by provider and recipient"."""
    currencies: tuple[str, ...] = ()
    """ISO 4217 codes of the currencies the source carries, sorted."""
    transformation_log: tuple[TransformationDoc, ...] = ()
    """Each step the DFO applied, in order."""
    version_history: tuple[ReleaseDoc, ...] = ()
    """The DFO pipeline's releases, newest first."""


@dataclass(frozen=True)
class SourceProvenance:
    """One cited source of a member's rows."""

    ref: str | None
    """The provenance dcid as it appears in the data's `provenance` column, `None` for rows
    that carry none."""
    name: str
    """The source's display name, such as "OECD Creditor Reporting System (CRS)"."""
    url: str | None = None
    """The source's landing page."""
    licence: Licence = field(default_factory=Licence)
    """The source's declared licence."""
    documentation: SourceDocumentation | None = None
    """The documentation snapshot taken at mint, `None` when the read failed or predates it."""
    publisher: str | None = None
    """The cited source node's name, such as "Organisation for Economic Co-operation and
    Development (OECD)", known whether or not a documentation snapshot exists."""

    @property
    def credited_publisher(self) -> str | None:
        """The publisher a bundle credits: the documentation's publisher, else `publisher`.
        `None` when neither names one, and the DFO then authors the source's citation. The CFF
        reference author, a `creator` contributor and the README's licence sentence name it."""
        documented = self.documentation.publisher if self.documentation is not None else None
        return blank_to_none(documented) or blank_to_none(self.publisher)


@dataclass(frozen=True)
class ColumnDoc:
    """The meaning of one column of a member's data file."""

    name: str
    """The column name, which must name a field of the member's `table_schema`."""
    title: str
    """A short human title, written as the schema field `title`."""
    description: str
    """What the column holds, written as the schema field `description`."""
    coded: bool = False
    """Cells are codes named in the codes table. With codes present, the column gets a foreign
    key to `codes.code`, so every non-empty cell must be a listed code."""
    empty: bool = False
    """Every cell of the column is empty in this file."""


@dataclass(frozen=True)
class CodeEntry:
    """One row of the codes table."""

    code: str
    """The code exactly as it appears in data cells, `dcid:` prefix included."""
    label: str
    """The code's plain name, such as "Germany" for `country/DEU`."""
    kind: str
    """What the code names, such as `variable`, `place`, `unit` or `source`."""


@dataclass(frozen=True)
class MemberProvenance:
    """One bundle member: its data file, sources and column meanings."""

    spec_id: str
    """The member's spec id, which must match its spec's `spec_id`."""
    title: str
    """The member data's heading, written as the resource `title`."""
    artifact: ArtifactEntry
    """The member's rendered data file."""
    sources: tuple[SourceProvenance, ...]
    """The sources of the member's rows, in first-seen order."""
    columns: tuple[ColumnDoc, ...] = ()
    """One entry per documented column, in file order."""
    description: str | None = None
    """What one row of the member's data holds, written as the resource `description`."""


@dataclass(frozen=True)
class ReproducingCall:
    """The request that serves this data again, written as `dfo.reproducing_call`."""

    path: str
    """The request path. The bundle prefixes `BundleContext.site_url` to form its `url`."""
    options: Mapping[str, object] = field(default_factory=dict)
    """The request's query options."""
    method: str = "GET"
    """The HTTP verb."""


@dataclass(frozen=True)
class BundleLinks:
    """Absolute URLs a bundle points readers to."""

    version: str
    """This exact version: `GET <site>/api/datasets/<id>/<content_version>`, or a card's export
    route. Written as the package `id`."""
    latest: str | None = None
    """The latest data for the same definition, `None` for a card export."""
    site: str | None = None
    """The website page for this version, `None` while the site route is not live and for a
    definition the website cannot open."""


@dataclass(frozen=True)
class BundleContext:
    """Bundle-level facts the caller derives once per version and rendering. Every field is
    fixed for a version, so only the graph metadata the caller reads can change a re-render."""

    heading: str
    """The bundle title, from `bundle_naming.heading` or a card's title."""
    summary: str
    """One line of coverage facts, such as "1,382 rows · 309 variables · 1 donor".
    Written as the package `description` and CFF `abstract`. Empty omits both."""
    slug: str
    """The package `name`, from `bundle_naming.sources_slug` or `card_slug`."""
    version: str
    """The content version of a Datasets bundle, or a card export's `r<revision>`."""
    prepared_at: datetime
    """When the version was first prepared, timezone-aware. Written as `created`."""
    site_url: str
    """The public website origin, which also serves `/api/*`, without a trailing slash."""
    links: BundleLinks
    """Absolute URLs for this version."""
    rebuild_request: Mapping[str, object] | None = None
    """The exact `POST /api/datasets` body that re-mints the dataset, `None` for a card
    export."""


@dataclass(frozen=True)
class ComponentExport:
    """The presentation component revision whose shown rows a bundle holds, and the frame
    those rows come from. `frame_ref` is `{spec_id, content_version}`, and `select` and
    `window` are the revision's stored intent fields, `None` where the intent has none."""

    component_id: str
    revision: int
    frame_ref: Mapping[str, str]
    select: Mapping[str, object] | None
    window: Mapping[str, object] | None


def assemble_bundle(
    *,
    spec: Mapping[str, object],
    members: Sequence[MemberProvenance],
    reproducing_call: ReproducingCall | None,
    context: BundleContext,
    codes: Sequence[CodeEntry],
) -> dict[str, bytes]:
    """Assembles one data resource per member, plus the codes resource when `codes` is
    non-empty.

    `reproducing_call=None` omits `dfo.reproducing_call`.

    Raises `MaterialisationUsageError` for no members, an identity mismatch, a column
    description naming no schema field, colliding resource names or paths, an invalid slug, a
    naive `prepared_at`, a code listed twice with different entries, or a code a spreadsheet
    would read as a formula. Every member must carry the spec's `spec_id`."""
    if not members:
        raise MaterialisationUsageError("assemble_bundle requires at least one member")
    _validate_identity(spec, members)

    dfo_property: dict[str, object] = {
        "spec": dict(spec),
        "content_version": context.version,
        "prepared_at": _rfc3339(context.prepared_at),
        "links": _links_entry(context, members),
    }
    if reproducing_call is not None:
        dfo_property["reproducing_call"] = _reproducing_call_entry(reproducing_call, context)
    definition_id = spec.get("spec_id")
    return _assemble(
        members=members,
        context=context,
        codes=codes,
        dfo_property=dfo_property,
        reproducing_call=reproducing_call,
        definition_id=definition_id if isinstance(definition_id, str) else None,
    )


def assemble_component_bundle(
    *,
    export: ComponentExport,
    member: MemberProvenance,
    reproducing_call: ReproducingCall | None,
    context: BundleContext,
    codes: Sequence[CodeEntry],
) -> dict[str, bytes]:
    """Assembles the bundle for one component revision's exported rows.

    `member` carries the one rendered file and the frame's sources. The descriptor's `dfo`
    property holds `component` (`component_id`, `revision`, `frame_ref`, `select`, `window`),
    `prepared_at`, `links`, and `reproducing_call`, the export route itself. The README names
    the frame's `spec_id` and `content_version`, the Datasets identity of the rows.
    `reproducing_call=None` omits `dfo.reproducing_call`, for rows no public route serves again.
    The README then offers `links.version` as the card's saved export. Raises
    `MaterialisationUsageError` as `assemble_bundle` does."""
    dfo_property: dict[str, object] = {
        "component": {
            "component_id": export.component_id,
            "revision": export.revision,
            "frame_ref": dict(export.frame_ref),
            "select": dict(export.select) if export.select is not None else None,
            "window": dict(export.window) if export.window is not None else None,
        },
        "prepared_at": _rfc3339(context.prepared_at),
        "links": _links_entry(context, (member,)),
    }
    if reproducing_call is not None:
        dfo_property["reproducing_call"] = _reproducing_call_entry(reproducing_call, context)
    return _assemble(
        members=(member,),
        context=context,
        codes=codes,
        dfo_property=dfo_property,
        reproducing_call=reproducing_call,
        export=export,
    )


def _assemble(
    *,
    members: Sequence[MemberProvenance],
    context: BundleContext,
    codes: Sequence[CodeEntry],
    dfo_property: Mapping[str, object],
    reproducing_call: ReproducingCall | None,
    definition_id: str | None = None,
    export: ComponentExport | None = None,
) -> dict[str, bytes]:
    codes_csv = _codes_csv(codes) if codes else None
    descriptor = _build_datapackage(
        members=members, context=context, codes_csv=codes_csv, dfo_property=dfo_property
    )
    sources = _distinct_sources(members)
    readme = build_readme(
        members=members,
        sources=sources,
        context=context,
        codes=codes,
        codes_path=CODES_PATH if codes_csv is not None else None,
        definition_id=definition_id,
        reproducing_call=reproducing_call,
        export=export,
    )
    files = {
        "datapackage.json": (json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
        "CITATION.cff": build_citation_cff(
            sources=sources, context=context, dfo_entity=_dfo_entity(context)
        ),
        "README.md": readme.encode("utf-8"),
        "SOURCES.md": build_sources(sources=sources, context=context).encode("utf-8"),
    }
    if codes_csv is not None:
        files[CODES_PATH] = codes_csv
    return files


def _validate_identity(spec: Mapping[str, object], members: Sequence[MemberProvenance]) -> None:
    """Raises `MaterialisationUsageError` unless every member's `spec_id` matches the spec's.
    A spec's data can fill several files, as a long CSV writes its total rows to a second one."""
    if "spec_id" not in spec:
        raise MaterialisationUsageError(
            "a standalone spec requires a spec_id key to validate "
            f"against its members' spec_id ({members[0].spec_id!r})"
        )
    for member in members:
        if member.spec_id != spec["spec_id"]:
            raise MaterialisationUsageError(
                f"member spec_id {member.spec_id!r} does not match "
                f"spec's spec_id {spec['spec_id']!r}"
            )


def _rfc3339(moment: datetime) -> str:
    """`moment` in UTC to the second, `Z`-suffixed. Raises for a naive datetime, whose instant
    is unknown."""
    if moment.tzinfo is None:
        raise MaterialisationUsageError("prepared_at must be timezone-aware")
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _links_entry(context: BundleContext, members: Sequence[MemberProvenance]) -> dict[str, object]:
    links: dict[str, object] = {"version": context.links.version}
    if context.links.latest is not None:
        links["latest"] = context.links.latest
    if context.links.site is not None:
        links["site"] = context.links.site
    documentation = {
        source.ref: site_path(context, f"/api/graph/documentation/{source.ref}")
        for source in _distinct_sources(members)
        if source.ref is not None
    }
    if documentation:
        links["documentation"] = documentation
    return links


def _reproducing_call_entry(call: ReproducingCall, context: BundleContext) -> dict[str, object]:
    return {
        "method": call.method,
        "url": site_path(context, call.path),
        "options": dict(call.options),
    }


def _distinct_sources(members: Sequence[MemberProvenance]) -> list[SourceProvenance]:
    """Every member's sources, first-seen by `ref`."""
    seen: dict[str | None, SourceProvenance] = {}
    for member in members:
        for source in member.sources:
            seen.setdefault(source.ref, source)
    return list(seen.values())


# ---------------------------------------------------------------------------
# datapackage.json

_NAME_PATTERN = re.compile(r"^[a-z0-9._-]+$")
"""Package and resource names: lowercase letters, digits, `.`, `_` and `-`."""

_DATA_PREFIX = "data/"


def _build_datapackage(
    *,
    members: Sequence[MemberProvenance],
    context: BundleContext,
    codes_csv: bytes | None,
    dfo_property: Mapping[str, object],
) -> dict[str, object]:
    if not _NAME_PATTERN.match(context.slug):
        raise MaterialisationUsageError(
            f"bundle slug {context.slug!r} is not a valid Data Package name"
        )
    resources = [_resource_entry(member, has_codes=codes_csv is not None) for member in members]
    if codes_csv is not None:
        resources.append(_codes_resource_entry(codes_csv))
    _check_resource_collisions(resources)

    sources = _distinct_sources(members)
    descriptor: dict[str, object] = {
        "$schema": DATAPACKAGE_PROFILE_URL,
        "name": context.slug,
        "id": context.links.version,
        "title": context.heading,
    }
    if context.summary:
        descriptor["description"] = context.summary
    descriptor |= {
        "version": context.version,
        "created": _rfc3339(context.prepared_at),
        "homepage": context.site_url,
        "keywords": list(dict.fromkeys(source.name for source in sources)),
    }
    licences = _distinct(
        entry for source in sources if (entry := _licence_entry(source.licence)) is not None
    )
    if licences:
        descriptor["licenses"] = licences
    if sources:
        descriptor["sources"] = [_source_entry(source) for source in sources]
    descriptor["contributors"] = _contributors(sources, context=context)
    descriptor["resources"] = resources
    descriptor["dfo"] = dict(dfo_property)
    return descriptor


def _contributors(
    sources: Sequence[SourceProvenance], *, context: BundleContext
) -> list[dict[str, object]]:
    """The DFO as curator and contact, then each distinct publisher as creator."""
    contributors: list[dict[str, object]] = [
        {
            "title": DFO_NAME,
            "path": context.site_url,
            "email": DFO_EMAIL,
            "roles": ["dataCurator", "contact"],
        }
    ]
    publishers = dict.fromkeys(
        publisher for source in sources if (publisher := source.credited_publisher) is not None
    )
    contributors += [{"title": publisher, "roles": ["creator"]} for publisher in publishers]
    return contributors


def _dfo_entity(context: BundleContext) -> dict[str, str]:
    """The DFO as a CFF entity, the citation's author and contact."""
    return {"name": DFO_NAME, "website": context.site_url, "email": DFO_EMAIL}


def _distinct(entries: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Distinct entries by value, first-seen order."""
    seen: dict[str, dict[str, object]] = {}
    for entry in entries:
        seen.setdefault(json.dumps(entry, sort_keys=True), entry)
    return list(seen.values())


def _check_resource_collisions(resources: Sequence[Mapping[str, object]]) -> None:
    # Path-derived names can collide, and frictionless rejects duplicate resource names.
    seen_names: set[object] = set()
    seen_paths: set[object] = set()
    for resource in resources:
        name = resource["name"]
        path = resource["path"]
        if name in seen_names:
            raise MaterialisationUsageError(
                f"two resources share the name {name!r}; resource names must "
                "be unique across the package"
            )
        if path in seen_paths:
            raise MaterialisationUsageError(
                f"two resources share the path {path!r}; resource paths must "
                "be unique across the package"
            )
        seen_names.add(name)
        seen_paths.add(path)


def _split_extension(path: str) -> tuple[str, str | None]:
    stem, sep, extension = path.rpartition(".")
    if not sep or "/" in extension:
        return path, None
    return stem, extension


def _resource_name(path: str) -> str:
    """`data/oecd-crs.csv` gives `oecd-crs`. `data/<spec_id>/data.csv` gives `<spec_id>-data`."""
    stem, _ = _split_extension(path.removeprefix(_DATA_PREFIX))
    name = stem.replace("/", "-").lower()
    if not _NAME_PATTERN.match(name):
        raise MaterialisationUsageError(
            f"artifact path {path!r} gives the resource name {name!r}, which is not a valid "
            "Data Package name"
        )
    return name


def _resource_entry(member: MemberProvenance, *, has_codes: bool) -> dict[str, object]:
    artifact = member.artifact
    entry: dict[str, object] = {
        "name": _resource_name(artifact.path),
        "title": member.title,
    }
    if member.description is not None:
        entry["description"] = member.description
    entry |= {
        "type": "table",
        # The v1 `profile` rides alongside v2 `type`: the pinned CRAN `frictionless` R
        # reader is v1-only and fails without it. v2 consumers ignore it.
        "profile": "tabular-data-resource",
        "path": artifact.path,
    }
    _, extension = _split_extension(artifact.path)
    if extension:
        entry["format"] = extension
    entry["mediatype"] = artifact.media_type
    if extension == "csv":
        entry["encoding"] = "utf-8"
    entry |= {"bytes": artifact.bytes, "hash": artifact.digest}
    if member.sources:
        # Resource entries keep source order. Package-level entries are distinct unions.
        entry["sources"] = [_source_entry(source) for source in member.sources]
        licences = [
            licence
            for source in member.sources
            if (licence := _licence_entry(source.licence)) is not None
        ]
        if licences:
            entry["licenses"] = licences
    entry["schema"] = _table_schema(member, has_codes=has_codes)
    return entry


def _table_schema(member: MemberProvenance, *, has_codes: bool) -> dict[str, object]:
    """The artifact's schema with each documented field's `title` and `description`, and a
    foreign key from each coded column to `codes.code` when the bundle has codes."""
    schema = dict(member.artifact.table_schema)
    fields = [
        dict(entry) for entry in cast("Sequence[Mapping[str, object]]", schema.get("fields", ()))
    ]
    docs = {column.name: column for column in member.columns}
    unknown = sorted(set(docs) - {entry.get("name") for entry in fields})
    if unknown:
        raise MaterialisationUsageError(
            f"member {member.spec_id} documents columns its data file lacks: {unknown}"
        )
    for entry in fields:
        doc = docs.get(cast(str, entry.get("name")))
        if doc is not None:
            entry["title"] = doc.title
            entry["description"] = doc.description
    schema["fields"] = fields
    if has_codes:
        foreign_keys = [
            {
                "fields": [column.name],
                "reference": {"resource": CODES_RESOURCE_NAME, "fields": ["code"]},
            }
            for column in member.columns
            if column.coded
        ]
        if foreign_keys:
            schema["foreignKeys"] = foreign_keys
    return schema


def _source_entry(source: SourceProvenance) -> dict[str, object]:
    entry: dict[str, object] = {"title": source.name}
    if url := blank_to_none(source.url):
        entry["path"] = url
    return entry


def _licence_entry(licence: Licence) -> dict[str, object] | None:
    """A `licenses[]` entry: the Open Definition `name` where `materialise.license_type` maps
    one, the terms URL as `path`, and the display `title`. `None` when there is neither a name
    nor a path, since the profile requires one."""
    name = licence_ids(licence.type_id).open_definition
    path = blank_to_none(licence.url)
    if name is None and path is None:
        return None
    entry: dict[str, object] = {}
    if name is not None:
        entry["name"] = name
    if path is not None:
        entry["path"] = path
    if title := licence_title(licence.type_id, blank_to_none(licence.label)):
        entry["title"] = title
    return entry


# ---------------------------------------------------------------------------
# data/codes.csv

_CODES_COLUMNS = ("code", "label", "kind")

# The codes table escapes labels and kinds with `spreadsheet_safe`, as every data cell is
# escaped. A code that needs escaping is refused, because the escaped data cells would differ
# from it.


def _codes_csv(codes: Sequence[CodeEntry]) -> bytes:
    """The codes table sorted by code, in the CSV writer's default dialect (CRLF), as the
    data files are written. A repeated identical entry collapses to one row."""
    by_code: dict[str, CodeEntry] = {}
    for entry in codes:
        if by_code.setdefault(entry.code, entry) != entry:
            raise MaterialisationUsageError(
                f"code {entry.code!r} is listed twice with different labels or kinds"
            )
        if spreadsheet_safe(entry.code) != entry.code:
            raise MaterialisationUsageError(
                f"code {entry.code!r} starts with a spreadsheet formula trigger"
            )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CODES_COLUMNS)
    for code in sorted(by_code):
        entry = by_code[code]
        writer.writerow((entry.code, spreadsheet_safe(entry.label), spreadsheet_safe(entry.kind)))
    return buffer.getvalue().encode("utf-8")


def _codes_resource_entry(codes_csv: bytes) -> dict[str, object]:
    return {
        "name": CODES_RESOURCE_NAME,
        "title": "Codes",
        "description": "The plain name of every code in the data.",
        "type": "table",
        "profile": "tabular-data-resource",
        "path": CODES_PATH,
        "format": "csv",
        "mediatype": "text/csv",
        "encoding": "utf-8",
        "bytes": len(codes_csv),
        "hash": "sha256:" + hashlib.sha256(codes_csv).hexdigest(),
        "schema": {
            "fields": [
                {"name": "code", "type": "string", "title": "Code"},
                {"name": "label", "type": "string", "title": "Label"},
                {"name": "kind", "type": "string", "title": "Kind"},
            ],
            "primaryKey": ["code"],
        },
    }
