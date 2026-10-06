"""README.md, SOURCES.md and CITATION.cff tests, built from real graph documentation of the
OECD CRS, IMF WEO, OECD GDP and UN SDG sources."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from materialise.bundle import (
    ArtifactEntry,
    BundleContext,
    BundleLinks,
    ChangeDoc,
    CodeEntry,
    ColumnDoc,
    ComponentExport,
    Licence,
    MemberProvenance,
    ReleaseDoc,
    ReproducingCall,
    SourceDocumentation,
    SourceProvenance,
    TransformationDoc,
    assemble_bundle,
    assemble_component_bundle,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_SITE = "https://site.example"
_CRS_SPEC_ID = "spec_24e7ab259dcf221c56d7e76edeb977fd"
_CRS_VERSION = "cv_5c9dfd37e657d3a3c7000cc6acf49c43"
_PREPARED_AT = datetime(2026, 10, 4, 9, 30, tzinfo=UTC)
_CRS_REF = "provenance/OECDCreditorReportingSystem"


def _read_json(name: str) -> Any:
    return json.loads((FIXTURES_DIR / name).read_text())


def _source(document: str, *, ref: str | None, licence_type: str | None = None) -> SourceProvenance:
    """A source as `read_source_citations` carries it, from a recorded graph documentation
    read."""
    doc = _read_json(f"bundle-docs-doc-{document}.json")
    details, licence, scope = doc["details"], doc["license"], doc["scope_and_units"]
    return SourceProvenance(
        ref=ref,
        name=details["name_of_source"],
        url=details["source_url"],
        licence=Licence(type_id=licence_type, label=licence["type"], url=licence["url"]),
        documentation=SourceDocumentation(
            publisher=details["publisher"],
            description=details["description"],
            source_data_url=details["source_data_url"],
            attribution=details["citation"],
            update_cadence=licence["update_cadence"],
            last_update=licence["last_update"],
            years_covered=scope["years_covered"],
            granularity=scope["granularity"],
            currencies=tuple(scope["currencies"]),
            transformation_log=tuple(
                TransformationDoc(title=step["transformation"], description=step["description"])
                for step in doc["transformation_log"]
            ),
            version_history=tuple(
                ReleaseDoc(
                    date=release["date"],
                    title=release["title"],
                    changes=tuple(
                        ChangeDoc(kind=change["kind"], note=change["note"])
                        for change in release["changes"]
                    ),
                )
                for release in doc["version_history"]
            ),
        ),
    )


_CRS = _source("provenance_OECDCreditorReportingSystem", ref=_CRS_REF)

_CRS_COLUMNS = (
    ("stat_var", "string", "Variable", "The variable measured.", True, False),
    ("financingProvider", "string", "Donor", "The donor providing the finance.", True, False),
    ("financingRecipient", "string", "Recipient", "The recipient of the finance.", True, False),
    ("time_period", "string", "Period", "The period the value refers to.", False, False),
    ("value", "number", "Value", "The observed value, in the unit named in `unit`.", False, False),
    ("unit", "string", "Unit", "The unit of `value`.", True, False),
    ("measurementMethod", "string", "Method", "How the source measured the value.", False, True),
    ("provenance", "string", "Source", "The source dataset of the row.", True, False),
    (
        "status",
        "string",
        "Status",
        "The source's flag on the value, such as provisional.",
        False,
        True,
    ),
    (
        "stat_var_crsCooperationModality",
        "string",
        "Co-operation modality",
        "The CRS co-operation modality of the variable, in words.",
        False,
        False,
    ),
    (
        "stat_var_crsPurpose",
        "string",
        "Purpose",
        "The CRS sector of the variable, in words.",
        False,
        False,
    ),
    (
        "stat_var_crsTypeOfFinance",
        "string",
        "Type of finance",
        "The CRS type of finance of the variable, in words.",
        False,
        False,
    ),
    (
        "stat_var_crsTypeOfFlow",
        "string",
        "Type of flow",
        "The CRS type of flow of the variable, in words.",
        False,
        False,
    ),
)


def _crs_codes() -> tuple[CodeEntry, ...]:
    """The twelve first variables of the CRS sample and its other codes, padded with
    synthetic variables to the sample's 309, which sort after the real ones."""
    codes = [CodeEntry(**entry) for entry in _read_json("bundle-docs-crs-codes.json")]
    real_variables = sum(entry.kind == "variable" for entry in codes)
    codes += [
        CodeEntry(code=f"dcid:CRSVariable{n:03d}", label=f"CRS variable {n}", kind="variable")
        for n in range(309 - real_variables)
    ]
    return tuple(codes)


def _crs_member(
    *, path: str = "data/oecd-creditor-reporting-system.csv", **changes: object
) -> MemberProvenance:
    member = MemberProvenance(
        spec_id=_CRS_SPEC_ID,
        title="OECD Creditor Reporting System (CRS), 1973\u20132024",
        artifact=ArtifactEntry(
            path=path,
            bytes=412_806,
            digest="sha256:" + "ab" * 32,
            media_type="text/csv",
            table_schema={
                "fields": [{"name": name, "type": kind} for name, kind, *_ in _CRS_COLUMNS]
            },
        ),
        sources=(_CRS,),
        columns=tuple(
            ColumnDoc(name=name, title=title, description=text, coded=coded, empty=empty)
            for name, _, title, text, coded, empty in _CRS_COLUMNS
        ),
        description="One row per observation.",
    )
    return dataclasses.replace(member, **changes)


def _crs_context(**changes: object) -> BundleContext:
    context = BundleContext(
        heading="OECD Creditor Reporting System (CRS), 1973\u20132024",
        summary=(
            "1,382 rows \u00b7 309 variables \u00b7 1 donor, 1 recipient \u00b7 1973\u20132024 \u00b7 Unit: US dollar"
        ),
        slug="oecd-creditor-reporting-system",
        version=_CRS_VERSION,
        prepared_at=_PREPARED_AT,
        site_url=_SITE,
        links=BundleLinks(
            version=f"{_SITE}/api/datasets/{_CRS_SPEC_ID}/{_CRS_VERSION}",
            latest=f"{_SITE}/api/datasets/{_CRS_SPEC_ID}",
            site=f"{_SITE}/data?spec={_CRS_SPEC_ID}&v={_CRS_VERSION}",
        ),
        rebuild_request=_read_json("bundle-docs-crs-mint.json"),
    )
    return dataclasses.replace(context, **changes)


def _crs_bundle(
    *,
    member: MemberProvenance | None = None,
    codes: tuple[CodeEntry, ...] | None = None,
    **changes: object,
) -> dict[str, bytes]:
    return assemble_bundle(
        spec={"schema_version": 1, "spec_id": _CRS_SPEC_ID},
        members=[member or _crs_member()],
        reproducing_call=ReproducingCall(
            path=f"/api/datasets/{_CRS_SPEC_ID}", options={"format": "csv", "bundle": "true"}
        ),
        context=_crs_context(**changes),
        codes=_crs_codes() if codes is None else codes,
    )


def _card_bundle() -> dict[str, bytes]:
    route = "/api/presentation/components/cmp_0000000000000001/revisions/3/data"
    member = _crs_member(path="data/data.csv", title="German aid to Nigeria")
    return assemble_component_bundle(
        export=ComponentExport(
            component_id="cmp_0000000000000001",
            revision=3,
            frame_ref={"spec_id": _CRS_SPEC_ID, "content_version": _CRS_VERSION},
            select=None,
            window=None,
        ),
        member=member,
        reproducing_call=ReproducingCall(path=route, options={"format": "csv", "bundle": "true"}),
        context=BundleContext(
            heading="German aid to Nigeria",
            summary="",
            slug="german-aid-to-nigeria",
            version="r3",
            prepared_at=_PREPARED_AT,
            site_url=_SITE,
            links=BundleLinks(version=f"{_SITE}{route}"),
        ),
        codes=_crs_codes(),
    )


def _text(files: dict[str, bytes], name: str) -> str:
    return files[name].decode("utf-8")


def _section(readme: str, heading: str) -> str:
    """The body of the `## heading` section, up to the next second-level heading."""
    _, _, rest = readme.partition(f"\n## {heading}\n")
    assert rest, f"no {heading} section"
    return rest.split("\n## ", 1)[0]


# Golden documents


@pytest.mark.parametrize(
    ("bundle", "name", "golden"),
    [
        (_crs_bundle, "README.md", "bundle-golden-crs-readme.md"),
        (_crs_bundle, "SOURCES.md", "bundle-golden-crs-sources.md"),
        (_crs_bundle, "CITATION.cff", "bundle-golden-crs-citation.cff"),
    ],
)
def test_documents_match_golden(bundle, name, golden):
    assert bundle()[name] == (FIXTURES_DIR / golden).read_bytes()


# README


def test_variables_list_ten_labels_then_count_the_rest():
    variables = _section(_text(_crs_bundle(), "README.md"), "Variables")
    lines = [line for line in variables.splitlines() if line.startswith("- ")]
    assert "The file holds 309 variables. The first 10:" in variables
    assert len(lines) == 11
    assert lines[-1] == "- and 299 more, all named in `data/codes.csv`."


def test_variables_list_every_label_when_ten_or_fewer_and_drop_without_variables():
    codes = _crs_codes()
    places = tuple(entry for entry in codes if entry.kind != "variable")
    few = (*codes[:3], *places)
    variables = _section(_text(_crs_bundle(codes=few), "README.md"), "Variables")
    assert "The file holds 3 variables:" in variables
    assert "more, all named" not in variables

    assert "## Variables" not in _text(_crs_bundle(codes=places), "README.md")
    assert "## Variables" not in _text(_crs_bundle(codes=()), "README.md")


def test_columns_name_the_codes_table_only_when_the_bundle_has_one():
    with_codes = _section(_text(_crs_bundle(), "README.md"), "Columns")
    without = _section(_text(_crs_bundle(codes=()), "README.md"), "Columns")
    assert "| `unit` | The unit of `value`. Names are in `data/codes.csv`. |" in with_codes
    assert "| `unit` | The unit of `value`. |" in without
    assert (
        "| `status` | The source's flag on the value, such as provisional. Empty in this file. |"
        in without
    )


@pytest.mark.parametrize(
    ("changes", "site_line", "rebuild_line"),
    [
        ({}, True, True),
        ({"links": BundleLinks(version=f"{_SITE}/v", latest=f"{_SITE}/l")}, False, True),
        ({"rebuild_request": None}, True, False),
    ],
)
def test_site_and_rebuild_lines_appear_only_when_set(changes, site_line, rebuild_line):
    get_again = _section(_text(_crs_bundle(**changes), "README.md"), "Get this data again")
    assert ("- On the website: " in get_again) is site_line
    assert ("this version's rows stay available as Parquet" in get_again) is site_line
    assert ("- To rebuild it yourself" in get_again) is rebuild_line
    assert "fetched only through the API are kept for 30 days from when they were made" in get_again


def test_card_readme_names_the_card_revision_and_its_frame():
    readme = _text(_card_bundle(), "README.md")
    route = f"{_SITE}/api/presentation/components/cmp_0000000000000001/revisions/3/data"

    assert readme.startswith("# German aid to Nigeria\n")
    assert (
        f"The rows behind revision 3 of the card German aid to Nigeria, from dataset definition "
        f"{_CRS_SPEC_ID} at content version {_CRS_VERSION}." in readme
    )
    assert f'`curl -OJ "{route}?format=csv&bundle=true"`' in _section(readme, "Get this data again")
    assert "revision 3. " + route in _section(readme, "How to cite")
    headings = [line for line in readme.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Files",
        "## Variables",
        "## Columns",
        "## Licence",
        "## How to cite",
        "## Get this data again",
        "## Open the data",
        "## About the Development Finance Observatory",
    ]


def test_curated_text_cannot_break_the_readme_structure():
    hostile = "Evil | Corp\n# fake heading `data`"
    source = dataclasses.replace(_CRS, name=hostile)
    member = _crs_member(description=hostile, sources=(source,))
    readme = _text(_crs_bundle(member=member), "README.md")
    assert "\n# fake heading" not in readme
    assert "| The data. Evil \\| Corp # fake heading \\`data\\` |" in _section(readme, "Files")


@pytest.mark.parametrize(
    ("licence", "sentence"),
    [
        (
            Licence(
                type_id="CreativeCommons0", label="Creative Commons 0", url="https://cc.example/0"
            ),
            "Organisation for Economic Co-operation and Development (OECD) publishes OECD Creditor "
            "Reporting System (CRS) under Creative Commons 0 (https://cc.example/0). Those terms "
            "govern reuse of this data.",
        ),
        (
            Licence(type_id="PublicDomain", label="PublicDomain"),
            "publishes OECD Creditor Reporting System (CRS) in the public domain. Those terms",
        ),
        (
            Licence(url="https://terms.example"),
            "under the terms at https://terms.example. Those terms",
        ),
        (
            Licence(),
            "The source declares no licence in the DFO catalogue. Check its terms at "
            f"{_CRS.url} before reuse. The DFO adds no terms of its own.",
        ),
    ],
)
def test_licence_section_states_each_licence_shape(licence, sentence):
    member = _crs_member(sources=(dataclasses.replace(_CRS, licence=licence),))
    assert sentence in _section(_text(_crs_bundle(member=member), "README.md"), "Licence")


@pytest.mark.parametrize(
    ("path", "lines"),
    [
        (
            "data/x.xlsx",
            [
                '- Python: `pandas.read_excel("data/x.xlsx")`.',
                '- R: `readxl::read_excel("data/x.xlsx")`.',
            ],
        ),
        (
            "data/x.parquet",
            [
                '- Python: `pandas.read_parquet("data/x.parquet")`.',
                '- R: `arrow::read_parquet("data/x.parquet")`.',
            ],
        ),
    ],
)
def test_open_the_data_follows_the_file_format(path, lines):
    member = _crs_member(path=path)
    opening = _section(_text(_crs_bundle(member=member), "README.md"), "Open the data")
    assert all(line in opening for line in lines)
    assert ("Spreadsheet" in opening) is path.endswith(".xlsx")


# SOURCES.md


def test_a_source_without_documentation_gets_its_name_licence_and_page_only():
    unread = SourceProvenance(
        ref="dc/base/UnreadSource",
        name="Unread Source",
        url="https://unread.example/terms",
        licence=Licence(
            type_id="CreativeCommons0",
            label="Creative Commons 0",
            url="https://creativecommons.org/publicdomain/zero/1.0/",
        ),
    )
    sources = _text(_crs_bundle(member=_crs_member(sources=(_CRS, unread))), "SOURCES.md")
    block = sources.split("## Unread Source\n", 1)[1]
    assert block.strip() == (
        "- Source page: https://unread.example/terms\n"
        "- Licence: Creative Commons 0 (https://creativecommons.org/publicdomain/zero/1.0/)"
    )


def test_bundle_carries_sources_md():
    assert "SOURCES.md" in _crs_bundle()
    assert "SOURCES.md" in _card_bundle()


# CITATION.cff


def test_citation_round_trips_through_yaml_with_the_cff_required_keys():
    citation = yaml.safe_load(_text(_crs_bundle(), "CITATION.cff"))
    assert {"cff-version", "message", "title", "authors"} <= set(citation)
    assert citation["cff-version"] == "1.2.0"
    assert citation["date-released"] == "2026-10-04"
    assert citation["url"] == f"{_SITE}/data?spec={_CRS_SPEC_ID}&v={_CRS_VERSION}"
    (reference,) = citation["references"]
    assert (
        reference["notes"]
        == _read_json("bundle-docs-doc-provenance_OECDCreditorReportingSystem.json")["details"][
            "citation"
        ]
    )
    assert reference["date-accessed"] == "2026-10-02"

    schema = _read_json("bundle-citation-cff-1.2.0-schema.json")
    for files in (_crs_bundle(), _card_bundle()):
        jsonschema.validate(instance=yaml.safe_load(_text(files, "CITATION.cff")), schema=schema)


def test_citation_scalars_survive_quotes_newlines_and_control_characters():
    hostile = 'A "quoted"\nname: with\x85 # and \x9b ﻿ and \\ backslash'
    source = dataclasses.replace(_CRS, name=hostile)
    citation = yaml.safe_load(
        _text(_crs_bundle(member=_crs_member(sources=(source,)), heading=hostile), "CITATION.cff")
    )
    assert citation["title"] == hostile
    assert citation["references"][0]["title"] == hostile


@pytest.mark.parametrize(
    ("licences", "top_level", "reference_keys"),
    [
        (("CreativeCommons0", "CreativeCommons0"), "CC0-1.0", [{"license": "CC0-1.0"}] * 2),
        (
            ("CreativeCommons0", None),
            None,
            [{"license": "CC0-1.0"}, {"license-url": "https://terms.example"}],
        ),
        (
            ("CreativeCommonsAttributionIntergovernmentalOrganizationLicense",) * 2,
            None,
            [{"license-url": "https://terms.example"}] * 2,
        ),
    ],
)
def test_citation_licence_is_top_level_only_when_every_source_shares_one_spdx_id(
    licences, top_level, reference_keys
):
    sources = tuple(
        dataclasses.replace(
            _CRS,
            ref=f"r/{n}",
            name=f"S{n}",
            licence=Licence(type_id=type_id, url="https://terms.example"),
        )
        for n, type_id in enumerate(licences)
    )
    citation = yaml.safe_load(
        _text(_crs_bundle(member=_crs_member(sources=sources)), "CITATION.cff")
    )
    assert citation.get("license") == top_level
    assert [
        {key: reference[key] for key in ("license", "license-url") if key in reference}
        for reference in citation["references"]
    ] == reference_keys
