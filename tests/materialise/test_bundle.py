"""Bundle assembly tests cover the descriptor, codes table, citation and README using artifact
fixtures."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import frictionless
import jsonschema
import pytest
import yaml

from materialise.bundle import (
    DATAPACKAGE_PROFILE_URL,
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
from materialise.bundle_sources import source_provenance
from materialise.errors import MaterialisationUsageError
from materialise.license_type import LicenceIds, licence_ids

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_SITE = "https://site.example"
_SPEC_ID = "spec_00000000000000000000000000000011"
_CONTENT_VERSION = "cv_5c9dfd37e657d3a3c7000cc6acf49c43"
_CALL = ReproducingCall(path=f"/api/datasets/{_SPEC_ID}", options={"format": "csv", "bundle": True})

_OECD = SourceProvenance(
    ref="provenance/OECDCreditorReportingSystem",
    name="OECD Creditor Reporting System (CRS)",
    url="https://data-explorer.oecd.org/crs",
    licence=Licence(
        type_id="OECDTermsAndConditions",
        label="OECD Terms and Conditions",
        url="https://www.oecd.org/en/about/terms-conditions.html",
    ),
    documentation=SourceDocumentation(
        publisher="Organisation for Economic Co-operation and Development (OECD)"
    ),
)
_WORLD_BANK = SourceProvenance(
    ref="dc/base/WorldDevelopmentIndicators",
    name="World Development Indicators",
    url="https://datacatalog.worldbank.org/search/dataset/0037712",
    licence=Licence(
        type_id="CreativeCommons0",
        label="Creative Commons 0",
        url="https://creativecommons.org/publicdomain/zero/1.0/",
    ),
    documentation=SourceDocumentation(publisher="World Bank"),
)
_SDG = SourceProvenance(ref="undata/p/SDG", name="SDG Global Database")

_COLUMNS = (
    ColumnDoc(name="provider", title="Provider", description="The donor.", coded=True),
    ColumnDoc(name="recipient", title="Recipient", description="The recipient.", coded=True),
    ColumnDoc(name="year", title="Year", description="The year."),
)
_CODES = (
    CodeEntry(code="country/TGO", label="Togo", kind="place"),
    CodeEntry(code="country/FRA", label="France", kind="place"),
)


def _read_json(name: str) -> dict[str, object]:
    return json.loads((FIXTURES_DIR / name).read_text())


def _artifact(fixture_name: str = "bundle-artifact-csv.json", **changes: object) -> ArtifactEntry:
    return dataclasses.replace(ArtifactEntry(**_read_json(fixture_name)), **changes)  # type: ignore[arg-type]


def _member(
    spec_id: str = _SPEC_ID,
    *,
    sources: tuple[SourceProvenance, ...] = (_OECD,),
    path: str = "data/data.csv",
    columns: tuple[ColumnDoc, ...] = _COLUMNS,
) -> MemberProvenance:
    return MemberProvenance(
        spec_id=spec_id,
        title="OECD Creditor Reporting System (CRS), 2015\u20132024",
        artifact=_artifact(path=path),
        sources=sources,
        columns=columns,
        description="One row per observation.",
    )


def _context(**changes: object) -> BundleContext:
    context = BundleContext(
        heading="OECD Creditor Reporting System (CRS), 2015\u20132024",
        summary="2 rows · 1 donor, 1 recipient · 2015\u20132024",
        slug="oecd-creditor-reporting-system",
        version=_CONTENT_VERSION,
        prepared_at=datetime(2026, 10, 4, 9, 30, tzinfo=UTC),
        site_url=_SITE,
        links=BundleLinks(
            version=f"{_SITE}/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}",
            latest=f"{_SITE}/api/datasets/{_SPEC_ID}",
        ),
    )
    return dataclasses.replace(context, **changes)  # type: ignore[arg-type]


def _standalone(
    *,
    member: MemberProvenance | None = None,
    codes: tuple[CodeEntry, ...] = (),
    reproducing_call: ReproducingCall | None = _CALL,
    **context_changes: object,
) -> dict[str, bytes]:
    return assemble_bundle(
        spec=_read_json("bundle-spec.json"),
        members=[member or _member()],
        reproducing_call=reproducing_call,
        context=_context(**context_changes),
        codes=codes,
    )


def _multi_source(*, codes: tuple[CodeEntry, ...] = _CODES) -> dict[str, bytes]:
    """One member sourced from OECD, the World Bank, and an SDG source with no licence, url or
    documentation."""
    return _standalone(member=_member(sources=(_OECD, _WORLD_BANK, _SDG)), codes=codes)


def _descriptor(files: dict[str, bytes]) -> dict:
    return json.loads(files["datapackage.json"])


def _write_bundle(directory: Path, files: dict[str, bytes]) -> Path:
    """Writes `files` and the fixture CSV at `data/data.csv`, returning the descriptor path."""
    (directory / "data").mkdir()
    (directory / "data" / "data.csv").write_bytes((FIXTURES_DIR / "bundle-data.csv").read_bytes())
    for name, contents in files.items():
        (directory / name).write_bytes(contents)
    return directory / "datapackage.json"


# Package and resource descriptors


def test_bundle_holds_the_documents_and_the_codes_table_only_with_codes():
    documents = {"datapackage.json", "CITATION.cff", "README.md", "SOURCES.md"}
    assert set(_standalone()) == documents
    assert set(_standalone(codes=_CODES)) == documents | {"data/codes.csv"}


def test_package_carries_identity_provenance_and_contributors():
    descriptor = _descriptor(_standalone())

    assert descriptor["$schema"] == DATAPACKAGE_PROFILE_URL
    assert descriptor["name"] == "oecd-creditor-reporting-system"
    assert descriptor["id"] == f"{_SITE}/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}"
    assert descriptor["title"] == "OECD Creditor Reporting System (CRS), 2015\u20132024"
    assert descriptor["description"] == "2 rows · 1 donor, 1 recipient · 2015\u20132024"
    assert descriptor["version"] == _CONTENT_VERSION
    assert descriptor["created"] == "2026-10-04T09:30:00Z"
    assert descriptor["homepage"] == _SITE
    assert descriptor["keywords"] == ["OECD Creditor Reporting System (CRS)"]
    assert descriptor["sources"] == [
        {
            "title": "OECD Creditor Reporting System (CRS)",
            "path": "https://data-explorer.oecd.org/crs",
        }
    ]
    assert descriptor["contributors"] == [
        {
            "title": "Development Finance Observatory",
            "path": _SITE,
            "email": "data@one.org",
            "roles": ["dataCurator", "contact"],
        },
        {
            "title": "Organisation for Economic Co-operation and Development (OECD)",
            "roles": ["creator"],
        },
    ]


def test_an_empty_summary_omits_the_package_description_and_cff_abstract():
    files = _standalone(summary="")
    assert "description" not in _descriptor(files)
    assert "abstract" not in yaml.safe_load(files["CITATION.cff"])


def test_created_is_written_in_utc_and_a_naive_prepared_at_refuses():
    local = datetime(2026, 10, 4, 1, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert _descriptor(_standalone(prepared_at=local))["created"] == "2026-10-04T06:00:00Z"

    with pytest.raises(MaterialisationUsageError, match="timezone"):
        _standalone(prepared_at=datetime(2026, 10, 4))


@pytest.mark.parametrize(
    ("fixture", "path", "encoding"),
    [
        ("bundle-artifact-csv.json", "data/data.csv", "utf-8"),
        ("bundle-artifact-xlsx.json", "data/data.xlsx", None),
        ("bundle-artifact-parquet.json", "data/data.parquet", None),
    ],
)
def test_resource_maps_artifact_metadata(fixture, path, encoding):
    artifact = _artifact(fixture)
    member = dataclasses.replace(_member(), artifact=artifact, columns=())
    (resource,) = _descriptor(_standalone(member=member))["resources"]

    assert resource["name"] == "data"
    assert resource["title"] == member.title
    assert resource["description"] == "One row per observation."
    assert resource["path"] == path
    assert resource["format"] == path.rsplit(".", 1)[1]
    assert resource["mediatype"] == artifact.media_type
    assert resource.get("encoding") == encoding
    assert resource["hash"] == artifact.digest
    assert resource["bytes"] == artifact.bytes
    assert resource["type"] == "table"
    assert resource["profile"] == "tabular-data-resource"
    assert resource["schema"] == dict(artifact.table_schema)


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("data/oecd-creditor-reporting-system.csv", "oecd-creditor-reporting-system"),
        (f"data/{_SPEC_ID}/data.csv", f"{_SPEC_ID}-data"),
        ("data/SDG-Global.csv", "sdg-global"),
    ],
)
def test_resource_name_follows_the_data_path(path, name):
    (resource,) = _descriptor(_standalone(member=_member(path=path)))["resources"]
    assert resource["name"] == name


def test_column_docs_title_fields_and_unknown_columns_refuse():
    (resource,) = _descriptor(_standalone())["resources"]
    fields = {field["name"]: field for field in resource["schema"]["fields"]}
    assert fields["provider"] == {
        "name": "provider",
        "type": "string",
        "title": "Provider",
        "description": "The donor.",
    }
    assert "title" not in fields["value"]

    stray = (*_COLUMNS, ColumnDoc(name="unit", title="Unit", description="The unit."))
    with pytest.raises(MaterialisationUsageError, match="unit"):
        _standalone(member=_member(columns=stray))


def test_dfo_property_carries_the_version_links_and_an_absolute_reproducing_call():
    dfo = _descriptor(_standalone())["dfo"]

    assert dfo["spec"] == _read_json("bundle-spec.json")
    assert dfo["content_version"] == _CONTENT_VERSION
    assert dfo["prepared_at"] == "2026-10-04T09:30:00Z"
    assert dfo["links"] == {
        "version": f"{_SITE}/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}",
        "latest": f"{_SITE}/api/datasets/{_SPEC_ID}",
        "documentation": {
            "provenance/OECDCreditorReportingSystem": (
                f"{_SITE}/api/graph/documentation/provenance/OECDCreditorReportingSystem"
            )
        },
    }
    assert dfo["reproducing_call"] == {
        "method": "GET",
        "url": f"{_SITE}/api/datasets/{_SPEC_ID}",
        "options": {"format": "csv", "bundle": True},
    }


def test_site_link_appears_only_when_given_and_becomes_the_cff_url():
    site_link = f"{_SITE}/data?spec={_SPEC_ID}&v={_CONTENT_VERSION}"
    files = _standalone(
        links=BundleLinks(version=f"{_SITE}/v", latest=f"{_SITE}/l", site=site_link)
    )
    assert _descriptor(files)["dfo"]["links"]["site"] == site_link
    assert yaml.safe_load(files["CITATION.cff"])["url"] == site_link
    assert "site" not in _descriptor(_standalone())["dfo"]["links"]


def test_absent_reproducing_call_is_omitted():
    assert "reproducing_call" not in _descriptor(_standalone(reproducing_call=None))["dfo"]


# Collections


def test_package_sources_and_licences_are_distinct_unions_and_the_resource_carries_them():
    descriptor = _descriptor(_multi_source(codes=()))
    oecd_licence = {
        "path": "https://www.oecd.org/en/about/terms-conditions.html",
        "title": "OECD Terms and Conditions",
    }
    cc0_licence = {
        "name": "CC0-1.0",
        "path": "https://creativecommons.org/publicdomain/zero/1.0/",
        "title": "Creative Commons 0",
    }

    assert descriptor["licenses"] == [oecd_licence, cc0_licence]
    assert [s["title"] for s in descriptor["sources"]] == [
        "OECD Creditor Reporting System (CRS)",
        "World Development Indicators",
        "SDG Global Database",
    ]
    (resource,) = descriptor["resources"]
    assert resource["licenses"] == [oecd_licence, cc0_licence]
    assert resource["sources"][2] == {"title": "SDG Global Database"}
    assert [c["title"] for c in descriptor["contributors"][1:]] == [
        "Organisation for Economic Co-operation and Development (OECD)",
        "World Bank",
    ]


def test_datapackage_matches_golden():
    golden = (FIXTURES_DIR / "bundle-golden-datapackage.json").read_bytes()
    assert _multi_source()["datapackage.json"] == golden


# Licences


@pytest.mark.parametrize(
    ("licence", "entry"),
    [
        (
            Licence(type_id="CreativeCommons0", label="Creative Commons 0"),
            {"name": "CC0-1.0", "title": "Creative Commons 0"},
        ),
        (
            Licence(type_id="OpenDataCommonsAttributionLicense", url="https://odc.example/by"),
            {"name": "ODC-BY-1.0", "path": "https://odc.example/by"},
        ),
        (Licence(type_id="CreativeCommonsAttributionLicenseGeneric", label="CC BY 2.5"), None),
        (
            Licence(type_id="PublicDomain", label="PublicDomain", url="https://pd.example"),
            {"path": "https://pd.example", "title": "Public domain"},
        ),
        (Licence(type_id="PublicDomain", label="PublicDomain"), None),
        (Licence(label="OECD Terms and Conditions", url="   "), None),
        (Licence(), None),
    ],
)
def test_licence_entry_needs_an_open_definition_name_or_a_terms_url(licence, entry):
    source = dataclasses.replace(_OECD, licence=licence)
    descriptor = _descriptor(_standalone(member=_member(sources=(source,))))
    (resource,) = descriptor["resources"]
    if entry is None:
        assert "licenses" not in descriptor
        assert "licenses" not in resource
    else:
        assert descriptor["licenses"] == resource["licenses"] == [entry]


@pytest.mark.parametrize(
    ("licence_type", "ids"),
    [
        ("CreativeCommons0", LicenceIds("CC0-1.0", "CC0-1.0")),
        ("OpenDataCommonsAttributionLicense", LicenceIds("ODC-BY-1.0", "ODC-By-1.0")),
        (
            "CreativeCommonsAttributionIntergovernmentalOrganizationLicense",
            LicenceIds(None, "CC-BY-3.0-IGO"),
        ),
        ("PublicDomain", LicenceIds(None, None)),
        ("OECDTermsAndConditions", LicenceIds(None, None)),
        (None, LicenceIds(None, None)),
    ],
)
def test_licence_ids_map_open_definition_and_spdx_separately(licence_type, ids):
    assert licence_ids(licence_type) == ids


# Codes table


def test_codes_table_is_sorted_deduplicated_and_described_with_its_hash():
    files = _standalone(codes=(*_CODES, _CODES[0]))
    codes_csv = files["data/codes.csv"]
    assert codes_csv == b"code,label,kind\r\ncountry/FRA,France,place\r\ncountry/TGO,Togo,place\r\n"

    data, codes = _descriptor(files)["resources"]
    assert codes["name"] == "codes"
    assert codes["path"] == "data/codes.csv"
    assert codes["bytes"] == len(codes_csv)
    assert codes["schema"]["primaryKey"] == ["code"]
    assert [f["name"] for f in codes["schema"]["fields"]] == ["code", "label", "kind"]
    assert {"type", "profile", "hash", "bytes", "schema"} <= set(codes)
    assert data["schema"]["foreignKeys"] == [
        {"fields": ["provider"], "reference": {"resource": "codes", "fields": ["code"]}},
        {"fields": ["recipient"], "reference": {"resource": "codes", "fields": ["code"]}},
    ]


def test_without_codes_no_column_gets_a_foreign_key():
    (resource,) = _descriptor(_standalone())["resources"]
    assert "foreignKeys" not in resource["schema"]


def test_codes_table_escapes_a_formula_label_as_data_cells_are_escaped():
    files = _standalone(codes=(CodeEntry(code="AgeBandMinus5", label="-5 to 0", kind="unit"),))
    assert files["data/codes.csv"] == b"code,label,kind\r\nAgeBandMinus5,'-5 to 0,unit\r\n"


def test_codes_table_refuses_conflicting_entries_and_formula_codes():
    with pytest.raises(MaterialisationUsageError, match="country/FRA"):
        _standalone(codes=(*_CODES, CodeEntry(code="country/FRA", label="Francia", kind="place")))
    with pytest.raises(MaterialisationUsageError, match="formula"):
        _standalone(codes=(CodeEntry(code="=cmd", label="x", kind="place"),))


# Usage errors


def test_assemble_bundle_requires_at_least_one_member():
    with pytest.raises(MaterialisationUsageError):
        assemble_bundle(
            spec=_read_json("bundle-spec.json"),
            members=[],
            reproducing_call=_CALL,
            context=_context(),
            codes=(),
        )


@pytest.mark.parametrize(
    ("spec", "spec_ids"),
    [
        ({"schema_version": 1, "spec_id": _SPEC_ID}, ["spec_00000000000000000000000000000099"]),
        (
            {"schema_version": 1, "spec_id": _SPEC_ID},
            [_SPEC_ID, "spec_00000000000000000000000000000099"],
        ),
        ({"schema_version": 1}, [_SPEC_ID]),
    ],
    ids=["standalone-mismatch", "standalone-second-mismatch", "standalone-no-id"],
)
def test_assemble_bundle_refuses_identity_mismatches(spec, spec_ids):
    members = [_member(spec_id, path=f"data/{n}.csv") for n, spec_id in enumerate(spec_ids)]
    with pytest.raises(MaterialisationUsageError):
        assemble_bundle(
            spec=spec, members=members, reproducing_call=_CALL, context=_context(), codes=()
        )


def test_a_member_named_codes_collides_with_the_codes_table():
    with pytest.raises(MaterialisationUsageError, match="codes"):
        _standalone(member=_member(path="data/codes.parquet"), codes=_CODES)


@pytest.mark.parametrize("slug", ["", "OECD", "oecd crs"])
def test_assemble_bundle_refuses_an_invalid_slug(slug):
    with pytest.raises(MaterialisationUsageError, match="slug"):
        _standalone(slug=slug)


# Citation and README


def test_citation_cff_is_valid_yaml_and_conforms_to_the_cff_1_2_0_schema():
    citation = yaml.safe_load(_multi_source()["CITATION.cff"].decode())
    schema = json.loads((FIXTURES_DIR / "bundle-citation-cff-1.2.0-schema.json").read_text())
    jsonschema.validate(instance=citation, schema=schema)


def test_citation_credits_the_dfo_and_each_publisher_with_a_dfo_fallback():
    citation = yaml.safe_load(_multi_source()["CITATION.cff"])
    dfo = {"name": "Development Finance Observatory", "website": _SITE, "email": "data@one.org"}

    assert citation["authors"] == [dfo]
    assert citation["version"] == "5c9dfd"
    assert citation["date-released"] == "2026-10-04"
    assert [r["authors"] for r in citation["references"]] == [
        [{"name": "Organisation for Economic Co-operation and Development (OECD)"}],
        [{"name": "World Bank"}],
        [dfo],
    ]


_OECD_PUBLISHER = "Organisation for Economic Co-operation and Development (OECD)"
_CITED_CRS: dict[str, object] = {
    "ref": "provenance/OECDCreditorReportingSystem",
    "label": "OECD Creditor Reporting System (CRS)",
    "license": "OECD Terms and Conditions",
    "license_url": "https://www.oecd.org/en/about/terms-conditions.html",
    "url": "https://data-explorer.oecd.org/crs",
    "source": {"ref": "dc/s/Oecd", "label": _OECD_PUBLISHER},
}


@pytest.mark.parametrize(
    ("cited", "publisher"),
    [
        (_CITED_CRS, _OECD_PUBLISHER),
        ({**_CITED_CRS, "documentation": {"publisher": None}}, _OECD_PUBLISHER),
        ({**_CITED_CRS, "documentation": {"publisher": "OECD DCD"}}, "OECD DCD"),
        ({**_CITED_CRS, "source": None}, None),
    ],
    ids=["undocumented", "documented-without-publisher", "documented-publisher-wins", "none"],
)
def test_a_cited_source_credits_its_publisher_with_or_without_documentation(cited, publisher):
    """The cited source node names the publisher whether or not documentation was read; the DFO
    authors the reference only when no publisher is known."""
    source = source_provenance(str(cited["ref"]), cited)
    files = _standalone(member=_member(sources=(source,)))
    reference = yaml.safe_load(files["CITATION.cff"])["references"][0]
    creators = [
        entry["title"]
        for entry in _descriptor(files)["contributors"]
        if entry["roles"] == ["creator"]
    ]
    readme = files["README.md"].decode()

    if publisher is None:
        assert reference["authors"] == [
            {"name": "Development Finance Observatory", "website": _SITE, "email": "data@one.org"}
        ]
        assert creators == []
        assert "OECD Creditor Reporting System (CRS) is published under" in readme
    else:
        assert reference["authors"] == [{"name": publisher}]
        assert creators == [publisher]
        assert f"{publisher} publishes OECD Creditor Reporting System (CRS) under" in readme


def test_readme_neutralises_control_characters_and_backticks_in_source_names():
    adversarial = dataclasses.replace(_OECD, name="Evil Corp\n# fake heading `data`")
    readme = _standalone(member=_member(sources=(adversarial,)))["README.md"].decode()
    assert "\n# fake heading" not in readme
    assert "Evil Corp # fake heading \\`data\\`" in readme


# The round-trip gate


def test_bundle_round_trips_through_pinned_frictionless_and_profile(tmp_path):
    """Validate the descriptor with frictionless 5.19.0 and the Data Package 2.0 JSON Schema,
    then the package on disk: hashes, sizes, the codes primary key, and every foreign key from
    the data to the codes table."""
    files = _standalone(codes=_CODES)
    descriptor = _descriptor(files)

    profile = json.loads((FIXTURES_DIR / "bundle-datapackage-2.0-profile.json").read_text())
    jsonschema.validate(instance=descriptor, schema=profile)
    assert not list(frictionless.Package.metadata_validate(descriptor))

    descriptor_path = _write_bundle(tmp_path, files)
    report = frictionless.validate(str(descriptor_path))
    assert report.valid, report.flatten(["type", "note"])

    package = frictionless.Package(str(descriptor_path))
    rows = package.get_resource("data").read_rows()
    assert [row["provider"] for row in rows] == ["country/FRA", "country/FRA"]
    assert [row["code"] for row in package.get_resource("codes").read_rows()] == [
        "country/FRA",
        "country/TGO",
    ]


def test_a_data_cell_missing_from_the_codes_table_fails_validation(tmp_path):
    descriptor_path = _write_bundle(tmp_path, _standalone(codes=(_CODES[0],)))
    report = frictionless.validate(str(descriptor_path))
    assert {error_type for (error_type,) in report.flatten(["type"])} == {"foreign-key"}
