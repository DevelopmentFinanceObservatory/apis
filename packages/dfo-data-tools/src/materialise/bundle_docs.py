"""Renders the README.md and SOURCES.md every download bundle carries.

Both documents read as plain text in any editor and as Markdown on GitHub. Curated metadata
(headings, source names, labels, attribution text) is neutralised before it lands in either:
newlines and tabs become spaces and backticks are escaped, so a value cannot open a heading or
a code span, and pipes are escaped inside table cells. `ColumnDoc.description` and the
CommonMark fields of `SourceDocumentation` (transformation steps and release notes) are
Markdown the caller means to render, so they keep their backticks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from materialise.bundle_naming import version_tag
from materialise.license_type import PUBLIC_DOMAIN, licence_title

if TYPE_CHECKING:
    from materialise.bundle import (
        BundleContext,
        CodeEntry,
        ColumnDoc,
        ComponentExport,
        Licence,
        MemberProvenance,
        ReproducingCall,
        SourceDocumentation,
        SourceProvenance,
    )

VARIABLE_KIND = "variable"
"""The `CodeEntry.kind` of a variable code, which the README's Variables section lists."""

VARIABLES_SHOWN = 10
"""The most variable labels the README lists before it points to the codes table."""

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_CONTROL_CHARS = str.maketrans({"\n": " ", "\r": " ", "\t": " "})
_SITE_RETENTION = "The link keeps working, and this version's rows stay available as Parquet."
_API_RETENTION = "Files fetched only through the API are kept for 30 days from when they were made."
_SPREADSHEET_EXTENSIONS = {"csv": "CSV", "xlsx": "XLSX file"}
_PANDAS_READERS = {"csv": "read_csv", "xlsx": "read_excel", "parquet": "read_parquet"}
_R_READERS = {"xlsx": "readxl::read_excel", "parquet": "arrow::read_parquet"}


# ---------------------------------------------------------------------------
# Text helpers


def _inline(text: str) -> str:
    """Curated text kept on one line, its backticks escaped."""
    return text.translate(_CONTROL_CHARS).replace("`", "\\`")


def _cell(text: str) -> str:
    """Curated text for a table cell."""
    return _inline(text).replace("|", "\\|")


def _markdown_cell(text: str) -> str:
    """Caller-authored Markdown for a table cell: one line, pipes escaped."""
    return text.translate(_CONTROL_CHARS).replace("|", "\\|")


def _code(text: str) -> str:
    """`text` as a code span, or as escaped text when it holds a backtick itself."""
    if "`" in text:
        return _inline(text)
    return f"`{text.translate(_CONTROL_CHARS)}`"


def _code_cell(text: str) -> str:
    return _code(text).replace("|", "\\|")


def _indented(text: str, indent: int) -> str:
    """CommonMark `text` with every line after the first indented to sit inside a list item."""
    first, *rest = text.strip().splitlines() or [""]
    pad = " " * indent
    return "\n".join([first, *(pad + line if line.strip() else "" for line in rest)])


def blank_to_none(value: str | None) -> str | None:
    """`value`, or `None` when it is `None`, empty or whitespace."""
    if value is None or not value.strip():
        return None
    return value


def _prepared(context: BundleContext) -> date:
    return context.prepared_at.astimezone(UTC).date()


def _long_date(day: date) -> str:
    """`4 October 2026`, the same in every locale."""
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def _plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def _extension(path: str) -> str:
    _, sep, extension = path.rpartition(".")
    return extension if sep and "/" not in extension else ""


def _query_url(base: str, options: Mapping[str, object]) -> str:
    if not options:
        return base
    encoded = {
        key: ("true" if value else "false") if isinstance(value, bool) else str(value)
        for key, value in options.items()
    }
    return f"{base}?{urlencode(encoded, safe=',')}"


def site_path(context: BundleContext, path: str) -> str:
    """`path` on the bundle's public site, an absolute URL."""
    return context.site_url.rstrip("/") + path


# ---------------------------------------------------------------------------
# README.md


def build_readme(
    *,
    members: Sequence[MemberProvenance],
    sources: Sequence[SourceProvenance],
    context: BundleContext,
    codes: Sequence[CodeEntry],
    codes_path: str | None,
    definition_id: str | None,
    is_collection: bool,
    reproducing_call: ReproducingCall | None,
    export: ComponentExport | None,
) -> str:
    """The bundle's README.md: what the data is, its files, variables and columns, the licence,
    how to cite it, how to get it again, how to open it, who the DFO is and which version this
    is.

    `sources` are the distinct sources across `members`, and `codes_path` is the codes table's
    bundle path, `None` when the bundle has none. `definition_id` is the spec or collection id
    the Version section names. A collection gets a Members section and no website or rebuild
    line, because the website opens only standalone definitions. A card export (`export` set)
    names the card revision and its frame's identity in place of a Version section.
    """
    sections = [
        _readme_heading(context, export),
        _files_section(members, codes=codes, codes_path=codes_path, is_collection=is_collection),
    ]
    if is_collection:
        sections.append(_members_section(members))
    sections += [
        _variables_section(codes, codes_path=codes_path, files=len(members)),
        _columns_section(members, codes_path=codes_path),
        _licence_section(sources),
        _citation_section(sources, context=context, export=export),
        _get_again_section(
            context,
            reproducing_call=reproducing_call,
            is_collection=is_collection,
            export=export,
        ),
        _open_section(members[0].artifact.path, codes_path=codes_path),
        _about_section(context),
    ]
    if export is None:
        sections.append(
            _version_section(context, definition_id=definition_id, is_collection=is_collection)
        )
    return "\n\n".join(section for section in sections if section) + "\n"


def _readme_heading(context: BundleContext, export: ComponentExport | None) -> str:
    heading = _inline(context.heading)
    lines = [
        f"# {heading}",
        "",
        "Data downloaded from the Development Finance Observatory (DFO), prepared on "
        f"{_long_date(_prepared(context))}.",
    ]
    if export is not None:
        lines += [
            "",
            f"The rows behind revision {export.revision} of the card {heading}, from dataset "
            f"definition {_inline(export.frame_ref['spec_id'])} at content version "
            f"{_inline(export.frame_ref['content_version'])}.",
        ]
    if context.summary:
        lines += ["", _inline(context.summary)]
    return "\n".join(lines)


def _files_section(
    members: Sequence[MemberProvenance],
    *,
    codes: Sequence[CodeEntry],
    codes_path: str | None,
    is_collection: bool,
) -> str:
    rows: list[tuple[str, str]] = []
    for member in members:
        if is_collection:
            contents = _cell(member.title)
        elif member.description:
            contents = f"The data. {_cell(member.description)}"
        else:
            contents = "The data"
        rows.append((member.artifact.path, contents))
    if codes_path is not None:
        rows.append((codes_path, "The plain name of every code in the data" + _code_example(codes)))
    rows += [
        ("SOURCES.md", "Who publishes the data, its licence, and each step the DFO applied"),
        ("CITATION.cff", "The citation, in a format reference managers read"),
        ("datapackage.json", "A machine-readable description (Frictionless Data Package 2.0)"),
    ]
    lines = ["## Files", "", "| File | Contents |", "| --- | --- |"]
    lines += [f"| {_code_cell(path)} | {contents} |" for path, contents in rows]
    return "\n".join(lines)


def _code_example(codes: Sequence[CodeEntry]) -> str:
    """`, such as <code> for <label>`, naming the first place code, else the first code."""
    if not codes:
        return ""
    places = [entry for entry in codes if entry.kind == "place"]
    example = min(places or codes, key=lambda entry: entry.code)
    return f", such as {_code_cell(example.code)} for {_cell(example.label)}"


def _members_section(members: Sequence[MemberProvenance]) -> str:
    lines = ["## Members", ""]
    for member in members:
        line = f"- {_inline(member.title)}: {_code(member.artifact.path)}"
        if member.row_count is not None:
            line += f", {_plural(member.row_count, 'row')}"
        line += "."
        if member.sources:
            names = ", ".join(_inline(source.name) for source in member.sources)
            line += f" Sources: {names}." if len(member.sources) > 1 else f" Source: {names}."
        lines.append(line)
    return "\n".join(lines)


def _variables_section(codes: Sequence[CodeEntry], *, codes_path: str | None, files: int) -> str:
    """Up to `VARIABLES_SHOWN` variable labels in codes-table order, then a count of the rest.
    Empty when the codes table names no variable."""
    variables: dict[str, str] = {}
    for entry in codes:
        if entry.kind == VARIABLE_KIND:
            variables.setdefault(entry.code, entry.label)
    if not variables or codes_path is None:
        return ""
    labels = [variables[code] for code in sorted(variables)]
    holds = "The file holds" if files == 1 else "The files hold"
    if len(labels) > VARIABLES_SHOWN:
        intro = f"{holds} {_plural(len(labels), 'variable')}. The first {VARIABLES_SHOWN}:"
    else:
        intro = f"{holds} {_plural(len(labels), 'variable')}:"
    lines = ["## Variables", "", intro, ""]
    lines += [f"- {_inline(label)}" for label in labels[:VARIABLES_SHOWN]]
    if len(labels) > VARIABLES_SHOWN:
        lines.append(
            f"- and {len(labels) - VARIABLES_SHOWN:,} more, all named in {_code(codes_path)}."
        )
    return "\n".join(lines)


def _column_set(member: MemberProvenance) -> tuple[tuple[str, ColumnDoc | None], ...]:
    """Each field of the member's data file, in file order, with its documentation."""
    docs = {column.name: column for column in member.columns}
    fields = member.artifact.table_schema.get("fields", ())
    assert isinstance(fields, Sequence)
    names = [str(entry.get("name", "")) for entry in fields if isinstance(entry, Mapping)]
    return tuple((name, docs.get(name)) for name in names)


def _column_meaning(doc: ColumnDoc | None, *, codes_path: str | None, files: int) -> str:
    if doc is None:
        return ""
    parts = [_markdown_cell(doc.description.strip())]
    if doc.coded and codes_path is not None:
        parts.append(f"Names are in {_code(codes_path)}.")
    if doc.empty:
        parts.append("Empty in this file." if files == 1 else "Empty in these files.")
    return " ".join(part for part in parts if part)


def _columns_section(members: Sequence[MemberProvenance], *, codes_path: str | None) -> str:
    """One table per distinct column set. Several sets each get a subheading naming the files
    that share it."""
    groups: dict[tuple[tuple[str, ColumnDoc | None], ...], list[str]] = {}
    for member in members:
        groups.setdefault(_column_set(member), []).append(member.artifact.path)
    lines = ["## Columns"]
    for columns, paths in groups.items():
        if not columns:
            continue
        if len(groups) > 1:
            lines += ["", "### " + ", ".join(_code(path) for path in paths)]
        lines += ["", "| Column | Meaning |", "| --- | --- |"]
        lines += [
            f"| {_code_cell(name)} | "
            f"{_column_meaning(doc, codes_path=codes_path, files=len(paths))} |"
            for name, doc in columns
        ]
    return "\n".join(lines) if len(lines) > 1 else ""


def _licence_sentence(source: SourceProvenance) -> str | None:
    """How the source's publisher licenses it, `None` when the source declares no licence."""
    licence: Licence = source.licence
    url = blank_to_none(licence.url)
    title = licence_title(licence.type_id, blank_to_none(licence.label))
    if title is None and url is None:
        return None
    name = _inline(source.name)
    publisher = source.credited_publisher
    subject = f"{_inline(publisher)} publishes {name}" if publisher else f"{name} is published"
    if licence.type_id == PUBLIC_DOMAIN:
        return f"{subject} in the public domain" + (f" ({url})." if url else ".")
    if title is None:
        return f"{subject} under the terms at {url}."
    return f"{subject} under {_inline(title)}" + (f" ({url})." if url else ".")


def _no_licence_sentence(source: SourceProvenance) -> str:
    page = blank_to_none(source.url)
    check = f"Check its terms at {page} before reuse." if page else "Check its terms before reuse."
    return f"The source declares no licence in the DFO catalogue. {check}"


def _licence_section(sources: Sequence[SourceProvenance]) -> str:
    if not sources:
        return ""
    sentences = [_licence_sentence(source) for source in sources]
    closing = "The DFO adds no terms of its own."
    if len(sources) == 1:
        (sentence,) = sentences
        if sentence is None:
            body = f"{_no_licence_sentence(sources[0])} {closing}"
        else:
            body = f"{sentence} Those terms govern reuse of this data. {closing}"
        return f"## Licence\n\n{body}"
    lines = ["## Licence", ""]
    for source, sentence in zip(sources, sentences, strict=True):
        lines.append(
            f"- {sentence}"
            if sentence is not None
            else f"- {_inline(source.name)}: {_no_licence_sentence(source)}"
        )
    lines += ["", f"Each source's terms govern reuse of its data. {closing}"]
    return "\n".join(lines)


def _source_citation(source: SourceProvenance) -> str:
    """The publisher's own attribution text, else publisher, name and page."""
    documentation = source.documentation
    if documentation is not None and (attribution := blank_to_none(documentation.attribution)):
        return _inline(attribution.strip())
    publisher = source.credited_publisher
    parts = [publisher] if publisher and publisher != source.name else []
    parts.append(source.name)
    if page := blank_to_none(source.url):
        parts.append(page)
    return _inline(", ".join(parts))


def _extract_citation(context: BundleContext, *, export: ComponentExport | None) -> str:
    """The citation of this extract: author, year, title, version and link."""
    edition = (
        f"revision {export.revision}"
        if export is not None
        else f"version {version_tag(context.version)}"
    )
    return (
        f"Development Finance Observatory ({_prepared(context).year}). {context.heading} "
        f"[dataset], {edition}. {context.links.site or context.links.version}"
    )


def _citation_section(
    sources: Sequence[SourceProvenance], *, context: BundleContext, export: ComponentExport | None
) -> str:
    extract = f"> {_inline(_extract_citation(context, export=export))}"
    if not sources:
        return f"## How to cite\n\nCite this extract:\n\n{extract}"
    attributed = all(
        source.documentation is not None and blank_to_none(source.documentation.attribution)
        for source in sources
    )
    subject = "the source" if len(sources) == 1 else "each source"
    intro = f"Cite {subject} as its publisher asks:" if attributed else f"Cite {subject}:"
    lines = ["## How to cite", "", intro]
    for source in sources:
        lines += ["", f"> {_source_citation(source)}"]
    lines += ["", "and this extract:", "", extract]
    return "\n".join(lines)


def _get_again_section(
    context: BundleContext,
    *,
    reproducing_call: ReproducingCall | None,
    is_collection: bool,
    export: ComponentExport | None,
) -> str:
    lines = ["## Get this data again", "", "No API key is needed.", ""]
    if export is not None:
        route = (
            _query_url(site_path(context, reproducing_call.path), reproducing_call.options)
            if reproducing_call is not None
            else context.links.version
        )
        lines.append(
            f'- The rows behind this card: `curl -OJ "{route}"` downloads this bundle again.'
        )
        return "\n".join(lines)

    if context.links.site is not None and not is_collection:
        lines.append(f"- On the website: {context.links.site}. {_SITE_RETENTION}")
    lines.append(
        f'- This exact version: `curl "{context.links.version}"` returns fresh download links. '
        + _API_RETENTION
    )
    if context.links.latest is not None:
        options = reproducing_call.options if reproducing_call is not None else {}
        latest = _query_url(context.links.latest, options)
        lines.append(f'- The latest data for the same definition: `curl "{latest}"`.')
    if context.rebuild_request is not None and not is_collection:
        body = json.dumps(context.rebuild_request, ensure_ascii=False)
        endpoint = site_path(context, "/api/datasets")
        fence = "`" * max(3, _longest_backtick_run(body) + 1)
        lines += [
            "- To rebuild it yourself, save the request body below as `request.json` and run "
            f'`curl -X POST -H "Content-Type: application/json" -d @request.json {endpoint}`.',
            "",
            f"{fence}json",
            body,
            fence,
        ]
    return "\n".join(lines)


def _longest_backtick_run(text: str) -> int:
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return longest


def _open_section(path: str, *, codes_path: str | None) -> str:
    """How to open the first data file in a spreadsheet, Python and R. Empty for a format none
    of them reads."""
    extension = _extension(path)
    lines: list[str] = []
    if spreadsheet := _SPREADSHEET_EXTENSIONS.get(extension):
        line = f"- Spreadsheet: open the {spreadsheet} in Excel, Numbers or Google Sheets."
        if codes_path is not None:
            line += f" Look up codes in {_code(codes_path)}."
        lines.append(line)
    if reader := _PANDAS_READERS.get(extension):
        line = f'- Python: `pandas.{reader}("{path}")`'
        if extension == "csv":
            line += ', or `frictionless.Package("datapackage.json")` for typed columns'
        lines.append(line + ".")
    if extension == "csv":
        lines.append('- R: `frictionless::read_package("datapackage.json")`.')
    elif reader := _R_READERS.get(extension):
        lines.append(f'- R: `{reader}("{path}")`.')
    if not lines:
        return ""
    return "\n".join(["## Open the data", "", *lines])


def _about_section(context: BundleContext) -> str:
    return (
        "## About the Development Finance Observatory\n\n"
        "The Development Finance Observatory brings development finance data together in one "
        "place. It is an initiative of ONE Data, with support from the Rockefeller Foundation. "
        f"Find more data at {context.site_url}. Questions or problems: data@one.org."
    )


def _version_section(
    context: BundleContext, *, definition_id: str | None, is_collection: bool
) -> str:
    facts = [f"Version {version_tag(context.version)} (content version {context.version})"]
    if definition_id is not None:
        kind = "collection definition" if is_collection else "dataset definition"
        facts.append(f"{kind} {_inline(definition_id)}")
    facts.append(f"prepared {_prepared(context).isoformat()}")
    return f"## Version\n\n{', '.join(facts)}. One version always holds the same numbers."


# ---------------------------------------------------------------------------
# SOURCES.md


def build_sources(*, sources: Sequence[SourceProvenance], context: BundleContext) -> str:
    """The bundle's SOURCES.md: one section per source with its publisher, description, links,
    licence, attribution, coverage, the steps the DFO applied, the DFO pipeline's releases, and
    links to the full documentation. Absent fields are left out. A source with no documentation
    snapshot gets its name, publisher, licence and source page only."""
    intro = (
        "# Sources\n\n"
        "Who publishes the data in this download, the terms for its reuse, and each step the "
        "DFO applied, as the DFO catalogue recorded them when this version was first prepared."
    )
    if not sources:
        return f"{intro}\n\nThe data names no source.\n"
    blocks = [intro, *(_source_block(source, context=context) for source in sources)]
    return "\n\n".join(blocks) + "\n"


def _source_block(source: SourceProvenance, *, context: BundleContext) -> str:
    documentation = source.documentation
    blocks = [f"## {_inline(source.name)}"]
    if documentation is not None and (description := blank_to_none(documentation.description)):
        blocks.append(_inline(description.strip()))
    blocks.append("\n".join(_source_facts(source)))
    if documentation is None:
        return "\n\n".join(blocks)

    if attribution := blank_to_none(documentation.attribution):
        blocks.append(f"### How the publisher asks to be cited\n\n> {_inline(attribution.strip())}")
    if documentation.transformation_log:
        blocks.append(_transformations(documentation))
    if documentation.version_history:
        blocks.append(_releases(documentation))
    if source.ref is not None:
        blocks.append(
            "### Full documentation\n\n"
            f"- {site_path(context, f'/data/sources/{source.ref}')}\n"
            f"- {site_path(context, f'/api/graph/documentation/{source.ref}')}"
        )
    return "\n\n".join(blocks)


def _source_facts(source: SourceProvenance) -> list[str]:
    documentation = source.documentation
    facts: list[tuple[str, str | None]] = [
        ("Publisher", source.credited_publisher),
        ("Source page", source.url),
    ]
    if documentation is not None:
        facts.append(("Raw data", documentation.source_data_url))
    sentence = _licence_line(source.licence)
    facts.append(("Licence", sentence or _no_licence_sentence(source)))
    if documentation is not None:
        currencies = ", ".join(documentation.currencies)
        facts += [
            ("Updates", documentation.update_cadence),
            ("Last update", documentation.last_update),
            ("Years covered", documentation.years_covered),
            ("Granularity", documentation.granularity),
            ("Currency" if len(documentation.currencies) == 1 else "Currencies", currencies),
        ]
    return [
        f"- {label}: {_inline(text.strip())}"
        for label, value in facts
        if (text := blank_to_none(value)) is not None
    ]


def _licence_line(licence: Licence) -> str | None:
    """The licence's title and terms page, `None` when neither is declared."""
    title = licence_title(licence.type_id, blank_to_none(licence.label))
    url = blank_to_none(licence.url)
    if title and url:
        return f"{title} ({url})"
    return title or url


def _transformations(documentation: SourceDocumentation) -> str:
    lines = ["### What the DFO did"]
    for number, step in enumerate(documentation.transformation_log, start=1):
        marker = f"{number}. "
        lines += ["", f"{marker}{_inline(step.title.strip())}"]
        if description := blank_to_none(step.description):
            pad = " " * len(marker)
            lines += ["", pad + _indented(description, len(marker))]
    return "\n".join(lines)


def _releases(documentation: SourceDocumentation) -> str:
    lines = ["### Version history", ""]
    for release in documentation.version_history:
        lines.append(f"- {_inline(release.date)}: {_inline(release.title.strip())}")
        lines += [
            f"  - {_inline(change.kind.capitalize())}: {_indented(change.note, 4)}"
            for change in release.changes
        ]
    return "\n".join(lines)
