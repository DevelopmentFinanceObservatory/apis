"""Renders a CRS-shaped dataset through `render`, unzips it and validates it with frictionless:
hashes, sizes, the codes table and every foreign key to it."""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import frictionless
import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.shared import get_index
from code_index.types import StatVarRow
from datasets_api.manifest_models import (
    CitationSource,
    CitedDocumentation,
    CitedSourceNode,
    TransformationStepModel,
)
from datasets_api.mint_models import MintOptions, PivotOptions
from datasets_api.render import (
    render_and_store,
)
from dc_graph.errors import GraphReadError
from dfo_platform.config import PublicSiteSettings
from materialise.bundle import ReproducingCall
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import Table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT

_SITE = PublicSiteSettings(public_site_url="https://site.example", site_dataset_links=True)

_CRS = "provenance/OECDCreditorReportingSystem"
_CRS_SPEC_ID = "spec_24e7ab259dcf221c56d7e76edeb977fd"
_CRS_VERSION = "cv_5c9dfd37e657d3a3c7000cc6acf49c43"
_CRS_STAT_VARS = tuple(
    "dcid:Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110"
    f"_CRSCooperationModalityB01_CRSPurposeCode{purpose}_Nominal"
    for purpose in ("11110", "11120", "11130")
)
_CRS_SPEC: dict[str, object] = {
    "schema_version": 1,
    "spec_id": _CRS_SPEC_ID,
    "identity": {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {"property": "crsPurpose", "binding": {"state": "unbound"}},
            {"property": "crsTypeOfFlow", "binding": {"state": "value", "ref": "CRSTypeOfFlow10"}},
        ],
        "entities": {"origin": ["country/DEU"], "destination": ["country/NGA"]},
        "requested_window": {"kind": "all"},
    },
}
_CRS_COLUMNS = (
    "stat_var",
    "financingProvider",
    "financingRecipient",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)
_CRS_TABLE = Table(
    columns=_CRS_COLUMNS,
    rows=tuple(
        (
            stat_var,
            "country/DEU",
            "country/NGA",
            year,
            1000.0 + index,
            "CurrencyUSD",
            None,
            _CRS,
            None,
        )
        for index, stat_var in enumerate(_CRS_STAT_VARS)
        for year in ("2015", "2016")
    ),
)
_CRS_CITATION = CitationSource(
    ref=_CRS,
    label="OECD Creditor Reporting System (CRS)",
    license="OECD Terms and Conditions",
    license_url="https://www.oecd.org/en/about/terms-conditions.html",
    url="https://data-explorer.oecd.org/vis?tm=crs",
    source=CitedSourceNode(
        ref="dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
        label="Organisation for Economic Co-operation and Development (OECD)",
    ),
    documentation=CitedDocumentation(
        publisher="Organisation for Economic Co-operation and Development (OECD)",
        description="The OECD Development Assistance Committee's activity-level database.",
        source_data_url="https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",
        attribution="OECD (2026), Creditor Reporting System (CRS), OECD Data Explorer",
        update_cadence="Quarterly",
        last_update="2026-10-02",
        years_covered="1973-2024",
        granularity="Annual, by provider and recipient",
        currencies=["USD"],
        transformation_log=[
            TransformationStepModel(
                transformation="Aggregate totals",
                description="Activities sharing the same year, provider and recipient are summed.",
            )
        ],
    ),
)

_SDG_STAT_VAR = "dcid:undata/sdg/SH_ACS_UNHC_25"
_GDP_STAT_VAR = "dcid:Adjusted_GrowthRate_GrossDomesticProduct_EconomicActivity_QuarterOnChange"

_NAMES = {
    **{
        stat_var.removeprefix("dcid:"): f"Variable {n}" for n, stat_var in enumerate(_CRS_STAT_VARS)
    },
    _SDG_STAT_VAR.removeprefix("dcid:"): "Proportion of health facilities with water",
    _GDP_STAT_VAR.removeprefix("dcid:"): "GDP growth, quarter on quarter",
    "country/DEU": "Germany",
    "country/NGA": "Nigeria",
    "CurrencyUSD": "US dollar",
    "undata/UNIT_MEASURE-INDEX": "Index",
    "Percent": "Percent",
    "ChainLinkedVolume": "Chain-linked volume",
}
_GRAPH_ARCS = {
    dcid: {"name": (name,), "typeOf": ("Country",) if dcid.startswith("country/") else ()}
    for dcid, name in _NAMES.items()
}


class _FailingGraph(FakeMintGraphReader):
    """A graph whose batched node read fails, as a read past the mint deadline does."""

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        raise GraphReadError("graph read past its deadline")


def _render_crs(*, graph: FakeMintGraphReader | None = None) -> dict[str, bytes]:
    store = FakeArtifactStore()
    options = MintOptions(format="csv", bundle=True)
    rendered = render_and_store(
        frame_chunks=iter((_CRS_TABLE,)),
        options=options,
        spec=_CRS_SPEC,
        provenance_refs=[_CRS],
        citations={_CRS: _CRS_CITATION},
        reproducing_call=ReproducingCall(
            path=f"/api/datasets/{_CRS_SPEC_ID}", options={"format": "csv", "bundle": "true"}
        ),
        content_version=_CRS_VERSION,
        prepared_at=PREPARED_AT,
        site=_SITE,
        artifact_store=store,
        entity_columns=("financingProvider", "financingRecipient"),
        enrichment=EnrichmentLookup(
            graph or FakeMintGraphReader(arcs=_GRAPH_ARCS), type_kinds={}, parties={}
        ),
        pivot_max_output_cells=5_000_000,
        code_index=get_index(),
        resolved_stat_var_refs=(),
    )
    return _unzip(store.fetch(rendered.artifact.url))


def _unzip(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _validate(tmp_path: Path, files: Mapping[str, bytes]) -> dict[str, object]:
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    report = frictionless.validate(str(tmp_path / "datapackage.json"))
    assert report.valid, report.flatten(["type", "note"])
    return json.loads(files["datapackage.json"])


def _codes(files: Mapping[str, bytes]) -> dict[str, tuple[str, str]]:
    rows = csv.DictReader(io.StringIO(files["data/codes.csv"].decode()))
    return {row["code"]: (row["label"], row["kind"]) for row in rows}


def _foreign_keys(resource: Mapping[str, object]) -> list[str]:
    schema = resource["schema"]
    assert isinstance(schema, Mapping)
    keys = schema.get("foreignKeys", [])
    assert isinstance(keys, Sequence)
    return [key["fields"][0] for key in keys]


def test_a_crs_bundle_validates_and_names_every_code_column_and_link(tmp_path):
    files = _render_crs()
    descriptor = _validate(tmp_path, files)

    assert set(files) == {
        "README.md",
        "SOURCES.md",
        "CITATION.cff",
        "datapackage.json",
        "data/oecd-creditor-reporting-system.csv",
        "data/codes.csv",
    }
    data, codes_resource = descriptor["resources"]
    assert data["name"] == "oecd-creditor-reporting-system"
    assert codes_resource["name"] == "codes"
    assert _foreign_keys(data) == [
        "stat_var",
        "financingProvider",
        "financingRecipient",
        "crsCooperationModality",
        "crsPurpose",
        "crsTypeOfFinance",
        "crsTypeOfFlow",
        "unit",
        "provenance",
    ]
    fields = {field["name"]: field for field in data["schema"]["fields"]}
    assert fields["financingProvider"]["description"] == "The provider the value flows from."
    assert fields["financingProvider_label"]["description"] == "The name of the provider."
    assert fields["crsPurpose"]["description"] == "The sector of the variable, as a code."
    assert fields["crsPurpose_label"]["description"] == "The sector of the variable, in words."
    assert fields["currency"]["description"] == ("The ISO 4217 code of the currency `value` is in.")
    assert fields["provenance_label"]["description"] == "The name of the source dataset."

    assert _codes(files) == {
        "CurrencyUSD": ("US dollar", "unit"),
        "country/DEU": ("Germany", "place"),
        "country/NGA": ("Nigeria", "place"),
        "CRSCooperationModalityB01": (
            "core support to non-governmental organisations, other private bodies, "
            "public-private partnerships, and research institutes",
            "modality",
        ),
        "CRSPurposeCode11110": ("education policy and administrative management", "sector"),
        "CRSPurposeCode11120": ("education facility and training", "sector"),
        "CRSPurposeCode11130": ("teacher training", "sector"),
        "CRSTypeOfFinance110": ("standard grant", "instrument"),
        "CRSTypeOfFlow10": ("official development assistance", "flow"),
        **{stat_var: (f"Variable {n}", "variable") for n, stat_var in enumerate(_CRS_STAT_VARS)},
        _CRS: ("OECD Creditor Reporting System (CRS)", "source"),
    }
    summary = "6 rows · 3 variables · 1 provider, 1 recipient · 2015\u20132016 · Unit: US dollar"
    assert descriptor["title"] == "OECD Creditor Reporting System (CRS), 2015\u20132016"
    assert descriptor["description"] == summary
    assert descriptor["created"] == "2026-10-04T09:00:00Z"
    assert descriptor["dfo"]["links"] == {
        "version": f"https://site.example/api/datasets/{_CRS_SPEC_ID}/{_CRS_VERSION}",
        "latest": f"https://site.example/api/datasets/{_CRS_SPEC_ID}",
        "site": f"https://site.example/data?spec={_CRS_SPEC_ID}&v={_CRS_VERSION}",
        "documentation": {_CRS: f"https://site.example/api/graph/documentation/{_CRS}"},
    }
    assert descriptor["licenses"] == [
        {
            "path": "https://www.oecd.org/en/about/terms-conditions.html",
            "title": "OECD Terms and Conditions",
        }
    ]

    readme = files["README.md"].decode()
    assert summary in readme
    assert '"input": {"kind": "spec_id"' in readme
    assert "| `measurementMethod` | How the source measured the value. Empty in this file. |" in (
        readme
    )
    assert "OECD (2026), Creditor Reporting System (CRS), OECD Data Explorer" in readme
    sources = files["SOURCES.md"].decode()
    assert "Aggregate totals" in sources


_PINNED_CRS_SPEC: dict[str, object] = {
    **_CRS_SPEC,
    "identity": {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": _CRS_STAT_VARS[0].removeprefix("dcid:")},
        "slots": [],
        "entities": {"origin": ["country/DEU"], "destination": ["country/NGA"]},
        "requested_window": {"kind": "all"},
    },
}
_SITE_LINK = f"https://site.example/data?spec={_CRS_SPEC_ID}&v={_CRS_VERSION}"


@pytest.mark.parametrize(
    ("spec", "site", "printed"),
    [
        pytest.param(_CRS_SPEC, _SITE, True, id="described"),
        pytest.param(
            _CRS_SPEC,
            PublicSiteSettings(public_site_url="https://site.example"),
            False,
            id="flag-off",
        ),
        # `statvar_set` mints this pinned arm, which Search's
        # `dataset_resubmit` refuses, so the website cannot open it.
        pytest.param(_PINNED_CRS_SPEC, _SITE, False, id="pinned"),
    ],
)
def test_a_crs_bundle_prints_the_site_link_only_for_a_replayable_identity(
    spec: Mapping[str, object], site: PublicSiteSettings, printed: bool
):
    store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=iter((_CRS_TABLE,)),
        options=MintOptions(format="csv", bundle=True),
        spec=spec,
        provenance_refs=[_CRS],
        citations={_CRS: _CRS_CITATION},
        reproducing_call=None,
        content_version=_CRS_VERSION,
        prepared_at=PREPARED_AT,
        site=site,
        artifact_store=store,
        entity_columns=("financingProvider", "financingRecipient"),
        enrichment=EnrichmentLookup(
            FakeMintGraphReader(arcs=_GRAPH_ARCS), type_kinds={}, parties={}
        ),
        pivot_max_output_cells=5_000_000,
        resolved_stat_var_refs=(),
    )
    files = _unzip(store.fetch(rendered.artifact.url))

    links = json.loads(files["datapackage.json"])["dfo"]["links"]
    assert links.get("site") == (_SITE_LINK if printed else None)
    assert (_SITE_LINK in files["README.md"].decode()) is printed
    assert (_SITE_LINK in files["CITATION.cff"].decode()) is printed


def test_a_failed_label_read_labels_each_code_with_itself_and_still_validates(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="materialise.bundle_columns"):
        files = _render_crs(graph=_FailingGraph(arcs={}))
    _validate(tmp_path, files)

    codes = _codes(files)
    assert codes["country/DEU"] == ("country/DEU", "entity")
    assert codes[_CRS_STAT_VARS[0]] == (_CRS_STAT_VARS[0], "variable")
    assert codes[_CRS] == ("OECD Creditor Reporting System (CRS)", "source")
    assert "bundle code label read failed" in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        "ftp://site.example",
        "https://site.example/path",
        "site.example",
        "https://user:secret@site.example",
        "https://user@site.example",
        "https://site.example:bad",
        "https://site.example:99999",
    ],
)
def test_the_public_site_url_must_be_an_http_origin(url):
    with pytest.raises(ValueError, match="origin"):
        PublicSiteSettings(public_site_url=url)


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://site.example:8443", "https://site.example:8443"),
        ("http://[::1]:8080/", "http://[::1]:8080"),
    ],
)
def test_the_public_site_url_keeps_a_port_and_brackets_an_ipv6_host(url, origin):
    assert PublicSiteSettings(public_site_url=url).public_site_url == origin


def test_the_public_site_url_drops_a_trailing_slash():
    assert PublicSiteSettings(public_site_url="https://site.example/").public_site_url == (
        "https://site.example"
    )


_AGE_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-10-04T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="ages",
)
_AGE_BAND_LOW = "dcid:AgeBandLow"
_AGE_BAND_HIGH = "dcid:AgeBandHigh"


def _age_index(low_label: str) -> CodeIndex:
    """A code index decoding two age-band StatVars, the low band labelled `low_label`."""
    bands = {_AGE_BAND_LOW: low_label, _AGE_BAND_HIGH: "0 to 5"}
    return CodeIndex(
        stamp=_AGE_STAMP,
        declarations=(
            CodeSystemDeclaration(
                code_system="ageBand",
                hierarchical=False,
                disjoint_from_children=False,
                summable=False,
                absent_reading="not_declared",
                plain_name="age band",
            ),
        ),
        codes=tuple(
            CodeRow(
                dcid=dcid, code_system="ageBand", code=label, label=label, synonyms=(), parent=None
            )
            for dcid, label in bands.items()
        ),
        statvars=tuple(
            StatVarRow(
                dcid=f"Count_Person_{dcid.removeprefix('dcid:')}",
                population_type="Person",
                measured_property="count",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("observationAbout",),
                constraint_properties={"ageBand": dcid},
            )
            for dcid in bands
        ),
    )


_AGE_TABLE = Table(
    columns=("stat_var", "observationAbout", "time_period", "value"),
    rows=tuple(
        (f"dcid:Count_Person_{dcid.removeprefix('dcid:')}", "country/NGA", "2020", 1.0 + n)
        for n, dcid in enumerate(sorted((_AGE_BAND_LOW, _AGE_BAND_HIGH)))
    ),
)


@pytest.mark.parametrize("trigger", ["-", "+", "=", "@"])
@pytest.mark.parametrize("format_", ["csv", "xlsx", "parquet"])
def test_a_pivot_heading_a_spreadsheet_escapes_is_documented_by_its_written_name(
    format_, trigger, tmp_path
):
    """A decoded label starting with a formula trigger heads a pivot column. The writer escapes it
    with an apostrophe, and the descriptor and README name that written heading. Each row of
    this pivot fills one heading, so an Excel row ends in an empty cell, and still validates."""
    store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=iter((_AGE_TABLE,)),
        options=MintOptions(
            format=format_,
            bundle=True,
            pivot=PivotOptions(index="stat_var", columns="stat_var_ageBand"),
        ),
        spec={"spec_id": "spec_age_bands"},
        provenance_refs=[],
        citations={},
        reproducing_call=None,
        content_version="cv_age_bands",
        prepared_at=PREPARED_AT,
        site=_SITE,
        artifact_store=store,
        entity_columns=("observationAbout",),
        enrichment=EnrichmentLookup(
            FakeMintGraphReader(arcs=_GRAPH_ARCS), type_kinds={}, parties={}
        ),
        pivot_max_output_cells=5_000_000,
        code_index=_age_index(f"{trigger}5 to 0"),
        resolved_stat_var_refs=(),
    )
    files = _unzip(store.fetch(rendered.artifact.url))

    # Frictionless reads Parquet through pandas, which this environment lacks.
    descriptor = (
        json.loads(files["datapackage.json"])
        if format_ == "parquet"
        else _validate(tmp_path, files)
    )
    (data, *_) = descriptor["resources"]
    fields = {field["name"]: field for field in data["schema"]["fields"]}
    written = f"'{trigger}5 to 0"
    assert set(fields) == {"stat_var", written, "0 to 5"}
    assert fields[written]["description"].startswith("The value for the entry")
    assert f"| `{written}` |" in files["README.md"].decode()
