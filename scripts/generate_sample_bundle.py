"""Generate a deterministic Data Package v2 bundle for the R round-trip gate.

The fixed sample table uses `csv.writer`'s default dialect. `assemble_bundle` produces the
descriptor, citation, readme and codes table checked by R's `frictionless` package. The two
entity columns carry foreign keys to the codes table, so the gate reads that shape too.

Writes the CSV at the descriptor's resource path, plus `datapackage.json`, `CITATION.cff`,
`README.md`, `SOURCES.md` and `data/codes.csv`, under the output directory. Exits non-zero if any file is
empty.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from materialise.bundle import (
    ArtifactEntry,
    BundleContext,
    BundleLinks,
    CodeEntry,
    ColumnDoc,
    Licence,
    MemberProvenance,
    ReproducingCall,
    SourceDocumentation,
    SourceProvenance,
    assemble_bundle,
)
from materialise.bundle_naming import heading, sources_slug

_SPEC_ID = "spec_00000000000000000000000000000001"
_CONTENT_VERSION = "cv_1a2b3c4d5e6f708192a3b4c5d6e7f809"
_SITE_URL = "https://example.org"
_SOURCE_NAME = "Sample Source (SS)"

_COLUMNS = ("provider", "recipient", "year", "value")
_ROWS = (
    ("country/FRA", "country/TGO", 2023, 1000.0),
    ("country/FRA", "country/TGO", 2024, 1200.0),
)

_TABLE_SCHEMA = {
    "fields": [
        {"name": "provider", "type": "string"},
        {"name": "recipient", "type": "string"},
        {"name": "year", "type": "year"},
        {"name": "value", "type": "number"},
    ]
}

_COLUMN_DOCS = (
    ColumnDoc(name="provider", title="Provider", description="The donor.", coded=True),
    ColumnDoc(name="recipient", title="Recipient", description="The recipient.", coded=True),
    ColumnDoc(name="year", title="Year", description="The year the value refers to."),
    ColumnDoc(name="value", title="Value", description="The observed value."),
)

_CODES = (
    CodeEntry(code="country/FRA", label="France", kind="place"),
    CodeEntry(code="country/TGO", label="Togo", kind="place"),
)


def _render_csv() -> bytes:
    """Render `_ROWS` with a header using `csv.writer` defaults and return UTF-8 bytes.

    Its `\r\n` line endings keep output stable across operating systems.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_COLUMNS)
    for row in _ROWS:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def build_sample_bundle() -> tuple[str, bytes, dict[str, bytes]]:
    """Assemble the bundle and return the CSV's bundle path, its bytes, and the other files."""
    csv_bytes = _render_csv()
    slug = sources_slug([_SOURCE_NAME])
    title = heading(source_names=[_SOURCE_NAME], start_year=2023, end_year=2024)
    data_path = f"data/{slug}.csv"
    member = MemberProvenance(
        spec_id=_SPEC_ID,
        title=title,
        artifact=ArtifactEntry(
            path=data_path,
            bytes=len(csv_bytes),
            digest=_digest(csv_bytes),
            media_type="text/csv",
            table_schema=_TABLE_SCHEMA,
        ),
        sources=(
            SourceProvenance(
                ref="provenance/SampleSource",
                name=_SOURCE_NAME,
                url="https://example.org/sample-source",
                licence=Licence(
                    type_id="CreativeCommons0",
                    label="Creative Commons 0",
                    url="https://creativecommons.org/publicdomain/zero/1.0/",
                ),
                documentation=SourceDocumentation(publisher="Sample Publisher"),
            ),
        ),
        columns=_COLUMN_DOCS,
        description="One row per observation.",
    )
    context = BundleContext(
        heading=title,
        summary="2 rows · 1 provider, 1 recipient · 2023\u20132024",
        slug=slug,
        version=_CONTENT_VERSION,
        prepared_at=datetime(2026, 1, 1, tzinfo=UTC),
        site_url=_SITE_URL,
        links=BundleLinks(
            version=f"{_SITE_URL}/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}",
            latest=f"{_SITE_URL}/api/datasets/{_SPEC_ID}",
        ),
    )
    files = assemble_bundle(
        spec={"schema_version": 1, "spec_id": _SPEC_ID},
        members=[member],
        reproducing_call=ReproducingCall(
            path=f"/api/datasets/{_SPEC_ID}", options={"format": "csv", "bundle": True}
        ),
        context=context,
        codes=_CODES,
    )
    return data_path, csv_bytes, files


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <output-directory>")
    out_dir = Path(sys.argv[1])

    data_path, csv_bytes, files = build_sample_bundle()
    descriptor = json.loads(files["datapackage.json"])
    if descriptor["resources"][0]["path"] != data_path:
        sys.exit(f"descriptor resource path does not match the CSV path {data_path!r}")

    written = {data_path: csv_bytes, **files}
    for name, contents in written.items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        if target.stat().st_size == 0:
            sys.exit(f"{target} is empty")

    print(f"wrote a sample bundle of {len(written)} files to {out_dir}")


if __name__ == "__main__":
    main()
