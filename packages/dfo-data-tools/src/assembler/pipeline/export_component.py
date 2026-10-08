"""Exports the frame rows behind a component revision's shown values as CSV, XLSX, or Parquet.

`export_component_data` reads one stored revision and collects the `(series, period)` pairs the
card shows a value for: each valued point of a line or bar chart series, each valued table row,
and a stat card's headline period plus both of its change basis periods. Series ids map to frame
coordinates through the revision's `series_coordinates`. The frame loads through
`FrameStore.load`, which rechecks revocation and expiry, and a frame row is kept when its
coordinates, normalised with `hashable_cell` as `resolve_selectors` groups them, and its period
name a shown pair. Every shown pair must match exactly one row, observed and holding the value
the card shows, or the export raises `ComponentExportMismatch` and serves nothing. A table's
rows are traced by their `series_id`. A table revision stored before rows carried it raises
`ComponentPredatesRowIdentities`, and one whose rows carry it on only some rows, or carry a
non-string or empty one, raises `ComponentExportMismatch`.

With a `target_frame`, the export is of the revision's intent rendered on that frame through
`assembler.pipeline.render.render_on_frame`. The shown pairs come from that rendering and the
rows from the same frame, so they match by construction. The bundle names the target frame, is
dated by that frame's retrieval date, and carries no `dfo.reproducing_call`, because the route
that serves it needs a credential a reader does not hold. Its `links.version` is the stored
revision's own export.

The file has a `series` column holding the card's own label for the row, then the frame's
columns in frame order, with a `{column}_label` column after `stat_var`, each entity role, and
`provenance`, read from `ResolvedFrame.labels`. A generated name that a frame column or an
earlier generated column already holds gains the first free suffix `_2`, `_3`, and so on, so
every frame column keeps its own name. The Datasets writers render it, so CSV and Excel
cells starting with a formula trigger are neutralised.

A bundle is a zip holding the file at `data/<card-slug>.<format>` beside
`materialise.bundle.assemble_component_bundle`'s documents, `data/codes.csv` among them. Each
provenance ref the kept rows carry is cited from the frame manifest's citation, its licence and
documentation included, through `materialise.bundle_sources`. Columns are documented by
`materialise.bundle_columns`, and the codes table is labelled from `ResolvedFrame.labels` and
the cited source names with no graph read, so a role's codes keep the kind `entity`. Links and
the reproducing call sit under the public site origin the caller passes.

The revision's stored `created_at` is the bundle's prepared date and dates the file name, so
one revision always serves one name. An export on a target frame takes both from that frame's
`retrieved_on`, at midnight UTC. `FrameStore.load` reads the graph's labels for the frame's
codes, and the sources' documentation for a derived frame or a root frame whose stored citation
left fields empty. Bytes match from one request to the next while those reads are unchanged. Names follow
`materialise.bundle_naming`: `dfo_<card-slug>_<YYYY-MM-DD>_<format>_r<revision>.zip` for a
bundle and `dfo_<card-slug>_<YYYY-MM-DD>_r<revision>.<format>` for a bare file, the date
in UTC.

This module imports the `formats` extra, so `assembler.pipeline` does not re-export it.
"""

from __future__ import annotations

import dataclasses
import io
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Literal, cast

from dfo_frame_grammar import STAT_VAR_COLUMN, STATUS_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN
from dfo_frame_grammar.grammar import unit_cell_from_json

from assembler.bind import role_columns
from assembler.frame_view import dimension_columns, hashable_cell
from assembler.mint.provenance import _UNATTRIBUTED_SOURCE_NAME, _citation_sources_by_ref
from assembler.pipeline.render import render_on_frame
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import FrameRef, FrameStore, ReleaseRegistry, ResolvedFrame, RevisionStore
from materialise.bundle import (
    ArtifactEntry,
    BundleContext,
    BundleLinks,
    ColumnDoc,
    ComponentExport,
    MemberProvenance,
    ReproducingCall,
    SourceProvenance,
    assemble_component_bundle,
)
from materialise.bundle_columns import TableScan, bundle_codes, column_docs, summary_line
from materialise.bundle_naming import (
    RESERVED_MEMBER_SLUGS,
    bare_file_name,
    card_slug,
    revision_tag,
    zip_file_name,
)
from materialise.bundle_sources import source_provenance
from materialise.formats import (
    Frame1Table,
    Rendering,
    Sink,
    Table,
    WrittenArtifact,
    table_schema_for,
    write_csv,
    write_excel,
    write_parquet,
)

__all__ = [
    "ComponentExportFile",
    "ComponentExportMismatch",
    "ComponentHoldsNoData",
    "ComponentPredatesRowIdentities",
    "ExportFormat",
    "export_component_data",
    "export_path",
]

type ExportFormat = Literal["csv", "xlsx", "parquet"]

_WRITERS: Mapping[str, Callable[..., WrittenArtifact]] = {
    "csv": write_csv,
    "xlsx": write_excel,
    "parquet": write_parquet,
}
_SERIES_COLUMN = "series"
_PROVENANCE_COLUMN = "provenance"
_LABEL_SUFFIX = "_label"
_ZIP_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_DATA_MEMBER_TYPES = frozenset({"line_chart", "bar_chart", "table", "stat_card"})

type _CoordinateKey = tuple[tuple[str, object], ...]


class ComponentHoldsNoData(Exception):
    """The revision has no frame rows to export: a note or source card, or a revision stored
    before `series_coordinates` existed. Maps to `not_found`."""

    def __init__(self, *, component_id: str, revision: int, detail: str) -> None:
        self.component_id = component_id
        self.revision = revision
        self.detail = detail
        super().__init__(detail)


class ComponentPredatesRowIdentities(ComponentHoldsNoData):
    """A table revision stored before its rows carried `series_id`, so its rows cannot be
    traced to frame rows. Maps to `not_found` as its base class does."""


class ComponentExportMismatch(Exception):
    """The frame rows do not match the values the card shows one for one. Maps to
    `internal_error`, since the stored revision and its pinned frame disagree."""

    def __init__(self, *, component_id: str, revision: int, detail: str) -> None:
        self.component_id = component_id
        self.revision = revision
        self.detail = detail
        super().__init__(f"{component_id} revision {revision}: {detail}")


@dataclass(frozen=True)
class ComponentExportFile:
    """The bytes to serve, their media type, and the attachment file name."""

    data: bytes
    media_type: str
    file_name: str


@dataclass(frozen=True)
class _ShownPoint:
    """One value the card shows. `value` is `None` where the payload does not state it, as
    for a stat card's change basis."""

    series_label: str
    coordinates: _CoordinateKey
    period: str
    value: object | None


def export_path(*, component_id: str, revision: int) -> str:
    """The export route's path, which is also the bundle's reproducing call."""
    return f"/api/presentation/components/{component_id}/revisions/{revision}/data"


def export_component_data(
    *,
    component_id: str,
    revision: int,
    format: ExportFormat,
    bundle: bool,
    revision_store: RevisionStore,
    frame_store: FrameStore,
    site_url: str,
    target_frame: FrameRef | None = None,
    release_registry: ReleaseRegistry | None = None,
) -> ComponentExportFile:
    """Render the frame rows behind revision `revision` of `component_id`. A bundle links
    under `site_url`, the public website origin without a trailing slash.

    With `target_frame`, the rows are those of the revision's intent rendered on that frame,
    under `release_registry`'s release, which is then required. The caller has already
    accepted `target_frame` as a stand-in for the stored frame, as for
    `assembler.pipeline.render.render_component`, and the refusals that function raises for a
    stored intent that does not bind the frame, a note, or a newer release apply here too.
    Nothing is pinned or appended. The file is dated by the frame's `retrieved_on`, and a
    bundle carries no reproducing call.

    Raises `DraftNotFound` for an unknown component, `RevisionNotFound` for a missing revision
    (`EditConflict` when the component has no head),
    `ComponentHoldsNoData` for a revision with no rows to export, including its subclass
    `ComponentPredatesRowIdentities` for a table stored without row ids, `FrameUnavailable` from the
    frame load, and `ComponentExportMismatch` when the rows and the card disagree. A record
    the store returns without `created_at` is a store defect and raises `RuntimeError`."""
    record = read_stored_revision(
        component_id=component_id, revision=revision, revision_store=revision_store
    )
    created_at = record.created_at
    if created_at is None:
        raise RuntimeError(
            f"the revision store returned revision {revision} of {component_id} without "
            "its created_at"
        )
    component = cast("Mapping[str, object]", record.resolved_payload["component"])
    member = cast(str, component["type"])
    if member not in _DATA_MEMBER_TYPES:
        raise ComponentHoldsNoData(
            component_id=component_id,
            revision=revision,
            detail=f"a {member} shows no data values, so it has no rows to download",
        )
    resolved_frame: ResolvedFrame | None = None
    if target_frame is not None:
        if release_registry is None:
            raise TypeError("export_component_data needs release_registry with target_frame")
        rendering = render_on_frame(
            record=record,
            target_frame=target_frame,
            frame_store=frame_store,
            release_registry=release_registry,
        )
        component = rendering.minted.component
        series_coordinates = rendering.series_coordinates
        frame_ref = target_frame
        resolved_frame = rendering.resolved_frame
    elif not record.series_coordinates:
        raise ComponentHoldsNoData(
            component_id=component_id,
            revision=revision,
            detail=(
                f"revision {revision} of {component_id} was stored without its series "
                "coordinates, so its rows cannot be traced; revise the component to export it"
            ),
        )
    else:
        series_coordinates = record.series_coordinates
        frame_ref = FrameRef(**cast("Mapping[str, str]", record.stored_intent["frame_ref"]))

    if member == "table":
        rows = cast("Sequence[Mapping[str, object]]", component["rows"])
        if rows and all(row.get("series_id") is None for row in rows):
            raise ComponentPredatesRowIdentities(
                component_id=component_id,
                revision=revision,
                detail=(
                    f"revision {revision} of {component_id} predates table row identities, so "
                    "its rows cannot be traced; recreate the card to export it"
                ),
            )

    def mismatch(detail: str) -> ComponentExportMismatch:
        return ComponentExportMismatch(component_id=component_id, revision=revision, detail=detail)

    shown = _shown_points(
        component=component,
        member=member,
        series_coordinates=series_coordinates,
        mismatch=mismatch,
    )
    if resolved_frame is None:
        resolved_frame = frame_store.load(frame_ref=frame_ref)
    table, provenance_refs, generated = _export_table(
        resolved_frame=resolved_frame, shown=shown, mismatch=mismatch
    )

    title = _card_title(component, component_id=component_id)
    slug = card_slug(title)
    tag = revision_tag(revision)
    # A rendering on a target frame holds that frame's data, so the frame dates the file.
    prepared_at = (
        created_at
        if target_frame is None
        else datetime.combine(resolved_frame.retrieved_on, time.min, tzinfo=UTC)
    )
    prepared = prepared_at.astimezone(UTC).date()
    if not bundle:
        sink = io.BytesIO()
        written = _WRITERS[format]((table,), rendering=Rendering(), sink=sink)
        return ComponentExportFile(
            data=sink.getvalue(),
            media_type=written.media_type,
            file_name=bare_file_name(slug=slug, prepared=prepared, tag=tag, extension=format),
        )

    export = ComponentExport(
        component_id=component_id,
        revision=revision,
        frame_ref={"spec_id": frame_ref.spec_id, "content_version": frame_ref.content_version},
        select=cast("Mapping[str, object] | None", record.stored_intent.get("select")),
        window=cast("Mapping[str, object] | None", record.stored_intent.get("window")),
    )
    # A card titled "codes" would put its data file on the codes table's path.
    data_slug = f"{slug}-2" if slug in RESERVED_MEMBER_SLUGS else slug
    data_path = f"data/{data_slug}.{format}"
    scan = TableScan(
        role_columns=role_columns(columns=cast("Sequence[str]", resolved_frame.frame["columns"]))
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        with zf.open(zipfile.ZipInfo(data_path, date_time=_ZIP_FIXED_DATE_TIME), mode="w") as entry:
            written = _WRITERS[format](
                scan.observe((table,)), rendering=Rendering(), sink=cast(Sink, entry)
            )
        sources = _sources(resolved_frame=resolved_frame, provenance_refs=provenance_refs)
        codes = bundle_codes(
            (scan,), known_labels=_known_labels(resolved_frame, sources=sources), lookup=None
        ).entries
        member_provenance = MemberProvenance(
            spec_id=frame_ref.spec_id,
            title=title,
            artifact=ArtifactEntry(
                path=data_path,
                bytes=written.size_bytes,
                digest=written.digest,
                media_type=written.media_type,
                table_schema=table_schema_for(table, rendering=Rendering()),
            ),
            sources=sources,
            columns=_column_docs(scan, generated=generated),
            description="One row per value the card shows.",
        )
        route = export_path(component_id=component_id, revision=revision)
        # The render export route needs the live-refresh principal, so a bundle on a target
        # frame names no call a reader could repeat.
        reproducing_call = (
            ReproducingCall(path=route, options={"format": format, "bundle": True})
            if target_frame is None
            else None
        )
        files = assemble_component_bundle(
            export=export,
            member=member_provenance,
            reproducing_call=reproducing_call,
            context=BundleContext(
                heading=title,
                summary=summary_line((scan,), codes=codes),
                slug=slug,
                version=tag,
                prepared_at=prepared_at,
                site_url=site_url,
                links=BundleLinks(version=f"{site_url}{route}"),
            ),
            codes=codes,
        )
        for name in sorted(files):
            zf.writestr(zipfile.ZipInfo(name, date_time=_ZIP_FIXED_DATE_TIME), files[name])
    return ComponentExportFile(
        data=archive.getvalue(),
        media_type="application/zip",
        file_name=zip_file_name(slug=slug, prepared=prepared, format=format, tag=tag),
    )


def _shown_points(
    *,
    component: Mapping[str, object],
    member: str,
    series_coordinates: Sequence[Mapping[str, object]],
    mismatch: Callable[[str], ComponentExportMismatch],
) -> tuple[_ShownPoint, ...]:
    """The values the card shows, in the card's own order."""
    coordinates_by_id = {
        cast(str, entry["id"]): _coordinate_key(cast("Mapping[str, object]", entry["coordinates"]))
        for entry in series_coordinates
    }

    def coordinates_of(series_id: str) -> _CoordinateKey:
        coordinates = coordinates_by_id.get(series_id)
        if coordinates is None:
            raise mismatch(f"series {series_id!r} has no stored coordinates")
        return coordinates

    if member == "stat_card":
        change = cast("Mapping[str, object] | None", component.get("change"))
        headline_period = cast(str, component["period"])
        periods = [headline_period, *cast("Sequence[str]", (change or {}).get("basis", ()))]
        coordinates = _coordinate_key(
            cast("Mapping[str, object]", series_coordinates[0]["coordinates"])
        )
        label = cast(str, component["label"])
        return tuple(
            _ShownPoint(
                series_label=label,
                coordinates=coordinates,
                period=period,
                value=component["value"] if period == headline_period else None,
            )
            for period in dict.fromkeys(periods)
        )

    if member == "table":
        rows = cast("Sequence[Mapping[str, object]]", component["rows"])
        if any(
            not isinstance(series_id := row.get("series_id"), str) or not series_id for row in rows
        ):
            raise mismatch("the table rows carry mixed or malformed series ids")
        dimension_order = [
            cast(str, column["key"])
            for column in cast("Sequence[Mapping[str, object]]", component["columns"])
        ]
        return tuple(
            _ShownPoint(
                series_label=_table_row_label(
                    dimensions=cast("Mapping[str, str]", row["dimensions"]),
                    column_order=dimension_order,
                ),
                coordinates=coordinates_of(cast(str, row["series_id"])),
                period=cast(str, row["period"]),
                value=row["value"],
            )
            for row in rows
            if "value" in row
        )

    return tuple(
        _ShownPoint(
            series_label=cast(str, series["label"]),
            coordinates=coordinates_of(cast(str, series["id"])),
            period=cast(str, point["period"]),
            value=point["value"],
        )
        for series in cast("Sequence[Mapping[str, object]]", component["series"])
        for point in cast("Sequence[Mapping[str, object]]", series["points"])
        if "value" in point
    )


def _card_title(component: Mapping[str, object], *, component_id: str) -> str:
    """The card's title text, else its component id."""
    title = cast("Mapping[str, object] | None", component.get("title"))
    return cast("str | None", (title or {}).get("text")) or component_id


def _table_row_label(*, dimensions: Mapping[str, str], column_order: Sequence[str]) -> str:
    """A table row's dimension labels joined in the table's column order. Stored JSONB does
    not keep key order, so dimensions without a column follow sorted by name."""
    ordered = [name for name in column_order if name in dimensions]
    ordered += sorted(name for name in dimensions if name not in column_order)
    return " / ".join(dimensions[name] for name in ordered)


def _coordinate_key(coordinates: Mapping[str, object]) -> _CoordinateKey:
    return tuple(sorted(coordinates.items()))


def _export_table(
    *,
    resolved_frame: ResolvedFrame,
    shown: Sequence[_ShownPoint],
    mismatch: Callable[[str], ComponentExportMismatch],
) -> tuple[Table, tuple[str | None, ...], dict[str, str | None]]:
    """Match each shown point to its one frame row and build the export table. Also returns
    the distinct provenance refs of the kept rows in first-seen order, and each generated
    column's name mapped to the frame column it labels, `None` for the series column."""
    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    rows = cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"])
    index = {column: position for position, column in enumerate(columns)}
    coordinate_columns = dimension_columns(columns=columns)

    wanted = {(point.coordinates, point.period): point for point in shown}
    if len(wanted) != len(shown):
        raise mismatch("the card shows the same series and period twice")
    matched: dict[tuple[_CoordinateKey, str], Sequence[object]] = {}
    for row in rows:
        key = (
            tuple(sorted((name, hashable_cell(row[index[name]])) for name in coordinate_columns)),
            cast(str, row[index[TIME_PERIOD_COLUMN]]),
        )
        if key not in wanted:
            continue
        if key in matched:
            raise mismatch(f"the frame holds two rows for period {key[1]} of one series")
        matched[key] = row

    for key, point in wanted.items():
        row = matched.get(key)
        if row is None:
            raise mismatch(f"no frame row holds period {point.period} of {point.series_label!r}")
        if STATUS_COLUMN in index and row[index[STATUS_COLUMN]] is not None:
            raise mismatch(
                f"the frame row for {point.period} of {point.series_label!r} has no value"
            )
        if point.value is not None and row[index[VALUE_COLUMN]] != point.value:
            raise mismatch(
                f"the frame value for {point.period} of {point.series_label!r} differs from the card"
            )

    labelled = {STAT_VAR_COLUMN, *role_columns(columns=columns), _PROVENANCE_COLUMN}
    taken = set(columns)
    series_column = _free_name(_SERIES_COLUMN, taken=taken)
    generated: dict[str, str | None] = {series_column: None}
    out_columns: list[str] = [series_column]
    for column in columns:
        out_columns.append(column)
        if column in labelled:
            label_column = _free_name(column + _LABEL_SUFFIX, taken=taken)
            generated[label_column] = column
            out_columns.append(label_column)

    labels = resolved_frame.labels
    out_rows: list[tuple[object, ...]] = []
    for key, point in wanted.items():
        row = matched[key]
        cells: list[object] = [point.series_label]
        for column, cell in zip(columns, row, strict=True):
            cells.append(_export_cell(column=column, cell=cell))
            if column in labelled:
                cells.append(labels.get(cast(str, cell), cell) if cell is not None else None)
        out_rows.append(tuple(cells))

    provenance_refs: tuple[str | None, ...] = ()
    if _PROVENANCE_COLUMN in index:
        provenance_refs = tuple(
            dict.fromkeys(
                cast("str | None", matched[key][index[_PROVENANCE_COLUMN]]) for key in wanted
            )
        )
    return Frame1Table(columns=tuple(out_columns), rows=tuple(out_rows)), provenance_refs, generated


def _free_name(base: str, *, taken: set[str]) -> str:
    """`base`, or `base` with the first free suffix `_2`, `_3`, ..., added to `taken`."""
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}_{suffix}"
        suffix += 1
    taken.add(name)
    return name


def _export_cell(*, column: str, cell: object) -> object:
    """Give a cell the type the Datasets downloads give it: a float value, and a converted or
    composed unit as a `UnitRequest` or `ComposedUnit`, which the writers render as its
    canonical JSON."""
    if column == VALUE_COLUMN and isinstance(cell, int) and not isinstance(cell, bool):
        return float(cell)
    if isinstance(cell, Mapping):
        return unit_cell_from_json(cell)
    return cell


def _sources(
    *, resolved_frame: ResolvedFrame, provenance_refs: Sequence[str | None]
) -> tuple[SourceProvenance, ...]:
    """One source per provenance ref, cited from the frame manifest's citation. Rows without
    a provenance cite one unattributed source."""
    cited_by_ref = _citation_sources_by_ref(resolved_frame.manifest)
    return tuple(
        source_provenance(ref, cited_by_ref.get(ref))
        if ref is not None
        else SourceProvenance(ref=None, name=_UNATTRIBUTED_SOURCE_NAME)
        for ref in provenance_refs
    )


def _known_labels(
    resolved_frame: ResolvedFrame, *, sources: Sequence[SourceProvenance]
) -> dict[str, str]:
    """The codes table's labels: the frame's resolved labels, with each cited source's name
    in place of its provenance ref's graph name. A source the citation lacks is named by its
    ref, so it keeps the frame's label."""
    cited_by_ref = _citation_sources_by_ref(resolved_frame.manifest)
    labels = dict(resolved_frame.labels)
    labels.update(
        (source.ref, source.name)
        for source in sources
        if source.ref is not None and source.ref in cited_by_ref
    )
    return labels


def _column_docs(scan: TableScan, *, generated: Mapping[str, str | None]) -> tuple[ColumnDoc, ...]:
    """Each column documented by `bundle_columns.column_docs`, with the series column and each
    `{column}_label` column described as what they are, the card's label and the name of the
    column they follow."""
    docs = column_docs(scan, code_system_names={})
    # A doc names its column as written, so match `generated`'s frame names by file position.
    titles = {column: doc.title for column, doc in zip(scan.columns, docs, strict=True)}
    out: list[ColumnDoc] = []
    for column, doc in zip(scan.columns, docs, strict=True):
        if column not in generated:
            out.append(doc)
            continue
        labelled = generated[column]
        if labelled is None:
            title, description = "Series", "The card's own label for the row's series."
        else:
            noun = titles[labelled]
            title, description = f"{noun} name", f"The name of the {noun.lower()}."
        out.append(dataclasses.replace(doc, title=title, description=description, coded=False))
    return tuple(out)
