"""Tests manifest assembly and the separation between assembly and storage, and the encoded
version-record bodies.
"""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import CodeSystemDeclaration, IndexStamp
from dataset_store.snapshot import ManifestFacts, StoredVersion
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest import (
    assemble_manifest,
    build_citation,
    collapsed_head_manifest,
    coverage_model,
    definition_version_record,
    healed_citation,
    manifest_facts,
    persisted_manifest,
    record_collapsed_head_version,
    stored_output_profile,
    stored_series_conversions,
)
from datasets_api.manifest_models import (
    ArtifactEntryModel,
    CitationModel,
    CitationSource,
    CitedDataCommons,
    CitedDocumentation,
    CitedSourceNode,
    CoverageModel,
    DefinitionModel,
    Manifest,
    TransformationStepModel,
)
from datasets_api.render import DerivedVersion
from datasets_api.versions import ArtifactRecord, NullVersionRecordWriter, VersionRecord
from description.citation import citation_line
from dfo_platform.roles import FlowFamily, role_wording
from materialise import (
    EnrichmentLookup,
    MaterialisationResult,
)
from materialise.dependencies import Dependencies
from materialise.frame_build import build_frame
from materialise.graph import Facet
from materialise.profile import compute_output_profile
from materialise.provenance_doc import CoverageFacts, read_provenance_doc, read_source_citations
from materialise.series_conversion import SeriesConversion, SeriesKey
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace
from materialise.window import Coverage
from origin_catalogue import (
    BASE,
    CardFacts,
    CatalogueSnapshot,
    ProvenanceRow,
    SourceRef,
    bound_catalogue,
)
from origin_catalogue.catalogue import Licence, SourceDocumentation
from origin_catalogue.job import build_provenance_table
from origin_catalogue.registry import parse_registry
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader, StoredDefinition
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_PREPARED_AT = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


_STAT_VAR_A = "GrantA"
_STAT_VAR_B = "GrantB"
_REF_A = f"dcid:{_STAT_VAR_A}"
_RETRIEVED = date(2026, 9, 28)
_REF_B = f"dcid:{_STAT_VAR_B}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _csv_row(*, stat_var_ref: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var_ref},country/KEN,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


# The mixer uses bare StatVar ids in queries and CSV. Graph refs use the `dcid:` prefix.
# Role-bearing probes group rows by the bare `row.variable`.
_CSV_BY_STAT_VAR = {
    _STAT_VAR_A: _CSV_HEADER
    + _csv_row(stat_var_ref=_STAT_VAR_A, year="2015", value="1000")
    + _csv_row(stat_var_ref=_STAT_VAR_A, year="2016", value="1500"),
    _STAT_VAR_B: _CSV_HEADER
    + _csv_row(stat_var_ref=_STAT_VAR_B, year="2017", value="2000")
    + _csv_row(stat_var_ref=_STAT_VAR_B, year="2018", value="2500"),
}

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=2,
    earliest_date="2015",
    latest_date="2018",
)


def _sdmx_transport(*, csv_by_stat_var=None):
    csv_by_stat_var = _CSV_BY_STAT_VAR if csv_by_stat_var is None else csv_by_stat_var

    def handler(request: httpx.Request) -> httpx.Response:
        # Group requests list each same-scope member. Return all rows under one header.
        requested_vars = str(request.url.params.get("c[variableMeasured]", "")).split(",")
        rows = "".join(
            csv_by_stat_var[var].removeprefix(_CSV_HEADER)
            for var in requested_vars
            if var in csv_by_stat_var
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_service() -> FakeCoverageService:
    """Coverage for both StatVars, matching `_FACET` and spanning the CSV fixtures' 2015-2018 years."""
    index = build_coverage_index(
        [
            (_STAT_VAR_A, "facet1", ABSENT_ENTITY, "country/KEN", [2015, 2016, 2017, 2018]),
            (_STAT_VAR_B, "facet1", ABSENT_ENTITY, "country/KEN", [2015, 2016, 2017, 2018]),
        ],
        role_names={
            _STAT_VAR_A: RoleNames(donor=None, recipient="recipient"),
            _STAT_VAR_B: RoleNames(donor=None, recipient="recipient"),
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(*, artifact_store=None, csv_by_stat_var=None):
    from dc_sdmx import SdmxClient

    store = FakeDatasetStore()
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR_A: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            _STAT_VAR_B: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
        },
        facets={
            (_STAT_VAR_A, "country/KEN"): [_FACET],
            (_STAT_VAR_B, "country/KEN"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )
    sdmx = SdmxClient(
        base_url=_BASE_URL, transport=_sdmx_transport(csv_by_stat_var=csv_by_stat_var)
    )
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


_BODY = {
    "schema_version": 1,
    "input": {
        "kind": "statvar_set",
        "statvars": [_REF_A],
        "scope": {
            "entities": {"recipient": ["country/KEN"]},
            "facets": {
                "provenance": {"ref": "OECD_DAC2A"},
                "measurementMethod": {"ref": "GrantEquivalent"},
                "unit": {"ref": "USD"},
            },
        },
    },
    "options": {"bundle": False},
}


def _statistics_artifacts(manifest: dict) -> list[dict]:
    return [a for a in manifest["artifacts"] if a["rendering"].get("statistics") == "series"]


def _statistics_rows_by_var(manifest: dict, artifact_store: FakeArtifactStore) -> dict[str, dict]:
    """Reads the statistics artifact and returns its row keyed by `stat_var`."""
    rows: dict[str, dict] = {}
    for artifact in _statistics_artifacts(manifest):
        table = pq.read_table(io.BytesIO(artifact_store.fetch(artifact["url"])))
        assert table.num_rows == 1
        row = table.to_pylist()[0]
        rows[row["stat_var"]] = row
    return rows


def test_single_observation_series_carries_a_null_change_in_its_statistics_artifact():
    """A's CSV carries one row, so `change_absolute`/`change_percent` are
    genuinely null in the single-observation series' own stored row.
    """
    csv_by_stat_var = {
        _STAT_VAR_A: _CSV_HEADER + _csv_row(stat_var_ref=_STAT_VAR_A, year="2015", value="1000"),
        _STAT_VAR_B: _CSV_BY_STAT_VAR[_STAT_VAR_B],
    }
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store, csv_by_stat_var=csv_by_stat_var)

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    stats_by_var = _statistics_rows_by_var(resp.json(), artifact_store)
    assert stats_by_var[_REF_A]["observations"] == 1
    assert stats_by_var[_REF_A]["change_absolute"] is None
    assert stats_by_var[_REF_A]["change_percent"] is None


def test_zero_baseline_series_carries_a_null_change_percent_but_keeps_absolute():
    """A's baseline (first) observation is zero, so `change_percent` is
    undefined and genuinely null, while `change_absolute` still carries the
    signal.
    """
    csv_by_stat_var = {
        _STAT_VAR_A: _CSV_HEADER
        + _csv_row(stat_var_ref=_STAT_VAR_A, year="2015", value="0")
        + _csv_row(stat_var_ref=_STAT_VAR_A, year="2016", value="500"),
        _STAT_VAR_B: _CSV_BY_STAT_VAR[_STAT_VAR_B],
    }
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store, csv_by_stat_var=csv_by_stat_var)

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    stats_by_var = _statistics_rows_by_var(resp.json(), artifact_store)
    assert stats_by_var[_REF_A]["change_absolute"] == 500.0
    assert stats_by_var[_REF_A]["change_percent"] is None


def test_change_absolute_overflow_serves_200_with_a_null_change_in_the_statistics_artifact():
    """A's two observations are both finite frame values, but their
    decimal-clean difference overflows past a finite `float`. The door
    still returns 200 with a serialisable body, and `change_absolute`/
    `change_percent` are genuinely null in A's own stored row, not an
    infinite value.
    """
    csv_by_stat_var = {
        _STAT_VAR_A: _CSV_HEADER
        + _csv_row(stat_var_ref=_STAT_VAR_A, year="2015", value="-1e308")
        + _csv_row(stat_var_ref=_STAT_VAR_A, year="2016", value="1e308"),
        _STAT_VAR_B: _CSV_BY_STAT_VAR[_STAT_VAR_B],
    }
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store, csv_by_stat_var=csv_by_stat_var)

    resp = client.post("/api/datasets", json=_BODY)

    assert resp.status_code == 200, resp.text
    stats_by_var = _statistics_rows_by_var(resp.json(), artifact_store)
    assert stats_by_var[_REF_A]["observations"] == 2
    assert stats_by_var[_REF_A]["change_absolute"] is None
    assert stats_by_var[_REF_A]["change_percent"] is None


def test_build_citation_keeps_one_entry_per_ref_and_processing_level_pair():
    """Citation entries deduplicate by `(ref, processing_level)`, not by `ref` alone."""
    citation, _warnings = build_citation(
        processed=False,
        provenance_refs=["OECD_DAC2A", "OECD_DAC2A", "OECD_DAC2A"],
        processing_levels=["minor", "minor", "significant"],
        source_citations={},
        retrieved=_RETRIEVED,
    )

    assert len(citation.sources) == 2
    by_level = {source.processing_level: source for source in citation.sources}
    assert set(by_level) == {"minor", "significant"}
    assert all(source.ref == "OECD_DAC2A" for source in citation.sources)
    assert citation.text == citation_line(retrieved=_RETRIEVED, sources=["OECD_DAC2A"])


def test_build_citation_names_a_catalogued_source_from_the_provenance_table():
    """A Provenance the origin catalogue lists takes its label, url, source, Data Commons and
    licence, as a display label, its dcid and its terms page, from the provenance table.
    `citation.text` is the citation line over the label."""
    wdi = "dc/base/WorldDevelopmentIndicators"
    graph = FakeMintGraphReader()
    row = ProvenanceRow(
        id=wdi,
        data_commons=BASE,
        source=SourceRef(ref="dc/s/WorldBank", label="World Bank"),
        name="World Development Indicators",
        url="https://datatopics.worldbank.org/world-development-indicators/",
        licence=Licence(
            type="CreativeCommonsAttributionLicense",
            label="Creative Commons Attribution License",
            url="https://creativecommons.org/licenses/by/4.0/",
        ),
    )
    refs = [wdi]

    citation, _warnings = build_citation(
        processed=False,
        provenance_refs=refs,
        source_citations=read_source_citations(
            graph, refs, catalogue=restrictive_catalogue(provenances=[row])
        ),
        retrieved=_RETRIEVED,
    )

    assert citation == CitationModel(
        text=citation_line(retrieved=_RETRIEVED, sources=["World Development Indicators"]),
        sources=[
            CitationSource(
                ref=wdi,
                label="World Development Indicators",
                processing_level=None,
                license="Creative Commons Attribution License",
                license_type="CreativeCommonsAttributionLicense",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                url="https://datatopics.worldbank.org/world-development-indicators/",
                source=CitedSourceNode(ref="dc/s/WorldBank", label="World Bank"),
                data_commons=CitedDataCommons(id="base", label="Data Commons by Google"),
            )
        ],
    )


def test_build_citation_snapshots_each_source_s_documentation():
    """The documentation read beside a citation lands on its source, mapped member by member;
    a source read without documentation carries none."""
    crs = "provenance/OECDCreditorReportingSystem"
    graph = FakeMintGraphReader(
        arcs={
            crs: {
                "typeOf": ("Provenance",),
                "name": ("OECD Creditor Reporting System",),
                "licenseAttribution": ("OECD (2026), Creditor Reporting System",),
                "sourceReleaseFrequency": ("P3M",),
                "sourceDataUrl": ("https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",),
                "lastDataRefreshDate": ("2026-10-02",),
                "earliestObservationDate": ("1973",),
                "latestObservationDate": ("2024",),
                "dataTransformationDescription": (
                    "## Download the source data\\n\\nThe complete CRS file is downloaded.",
                ),
            },
        }
    )

    documented, _ = build_citation(
        processed=False,
        provenance_refs=[crs],
        source_citations=read_source_citations(
            graph, [crs], catalogue=restrictive_catalogue(), documented=True
        ),
        retrieved=_RETRIEVED,
    )
    undocumented, _ = build_citation(
        processed=False,
        provenance_refs=[crs],
        source_citations=read_source_citations(graph, [crs], catalogue=restrictive_catalogue()),
        retrieved=_RETRIEVED,
    )

    assert documented.sources[0].documentation == CitedDocumentation(
        attribution="OECD (2026), Creditor Reporting System",
        update_cadence="Quarterly",
        source_data_url="https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",
        last_update="2026-10-02",
        years_covered="1973-2024",
        transformation_log=[
            TransformationStepModel(
                transformation="Download the source data",
                description="The complete CRS file is downloaded.",
            )
        ],
    )
    assert undocumented.sources[0].documentation is None


def test_build_citation_names_an_unlisted_source_after_its_graph_dataset():
    """A Provenance the catalogue does not list takes its parent Dataset's graph name and
    carries no source or Data Commons."""
    graph = FakeMintGraphReader(
        arcs={
            "dc/base/SomeImport": {
                "typeOf": ("Provenance",),
                "isPartOf": ("dc/d/SomeDataset",),
            },
            "dc/d/SomeDataset": {"typeOf": ("Dataset",), "name": ("Some Dataset",)},
        }
    )
    refs = ["dc/base/SomeImport"]

    citation, _warnings = build_citation(
        processed=False,
        provenance_refs=refs,
        source_citations=read_source_citations(graph, refs, catalogue=restrictive_catalogue()),
        retrieved=_RETRIEVED,
    )

    (source,) = citation.sources
    assert (source.label, source.source, source.data_commons) == ("Some Dataset", None, None)


def test_build_citation_falls_back_to_the_ref_for_a_source_with_no_resolved_entry():
    citation, _warnings = build_citation(
        processed=False,
        provenance_refs=["dc/base/NotHeld"],
        source_citations={},
        retrieved=_RETRIEVED,
    )

    assert citation.sources == [
        CitationSource(ref="dc/base/NotHeld", label="dc/base/NotHeld", license=None, url=None)
    ]
    assert citation.text == citation_line(retrieved=_RETRIEVED, sources=["dc/base/NotHeld"])


def test_build_citation_names_every_source_in_one_line_in_first_seen_order():
    """Two sources read as one citation line, in the order the refs arrive."""
    citation, _warnings = build_citation(
        processed=False,
        provenance_refs=["dc/base/B", "dc/base/A"],
        source_citations={},
        retrieved=_RETRIEVED,
    )

    assert citation.text == citation_line(retrieved=_RETRIEVED, sources=["dc/base/B", "dc/base/A"])


_WDI = "dc/base/WorldDevelopmentIndicators"


def _documented_wdi() -> CatalogueSnapshot:
    """A catalogue listing a licensed and documented source: a mint fills every citation
    field."""
    row = ProvenanceRow(
        id=_WDI,
        data_commons=BASE,
        source=SourceRef(ref="dc/s/WorldBank", label="World Bank"),
        name="World Development Indicators",
        url="https://datatopics.worldbank.org/world-development-indicators/",
        licence=Licence(
            type="CreativeCommonsAttributionLicense",
            label="Creative Commons Attribution License",
            url="https://creativecommons.org/licenses/by/4.0/",
        ),
    )
    return restrictive_catalogue(
        provenances=[row],
        dataset_facts={
            _WDI: CardFacts(
                license_type="Creative Commons Attribution License", update_cadence="P1Y"
            )
        },
        source_documentation={
            _WDI: SourceDocumentation(
                provenances=(_WDI,),
                citation="World Bank (2026), World Development Indicators",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
        },
    )


def test_healed_citation_fills_an_old_body_with_what_a_fresh_mint_stores():
    """A body stored before the citation carried a licence, url, source and Data Commons heals to
    the source a first mint builds from the same catalogue."""
    graph = FakeMintGraphReader()
    catalogue = _documented_wdi()
    fresh, _ = build_citation(
        processed=False,
        provenance_refs=[_WDI],
        source_citations=read_source_citations(graph, [_WDI], catalogue=catalogue, documented=True),
        retrieved=_RETRIEVED,
    )
    stored = fresh.model_copy(
        update={
            "sources": [
                fresh.sources[0].model_copy(
                    update={
                        "license": None,
                        "url": None,
                        "source": None,
                        "data_commons": None,
                    }
                )
            ]
        }
    )

    with bound_catalogue(catalogue):
        healed, complete = healed_citation(stored, graph=graph, deadline=None, coverage_index=None)

    assert fresh.sources[0].documentation is not None
    assert (healed, complete) == (fresh, True)
    assert graph.calls["node_arcs_many"] == graph.calls["node_subjects"] == 0


def test_healed_citation_reads_nothing_for_a_fully_populated_body():
    graph = FakeMintGraphReader()
    catalogue = _documented_wdi()
    stored, _ = build_citation(
        processed=False,
        provenance_refs=[_WDI],
        source_citations=read_source_citations(graph, [_WDI], catalogue=catalogue, documented=True),
        retrieved=_RETRIEVED,
    )
    reads_before = dict(graph.calls)

    with bound_catalogue(catalogue):
        healed, complete = healed_citation(stored, graph=graph, deadline=None, coverage_index=None)

    assert (healed, complete) == (stored, True)
    assert graph.calls == reads_before


# Citations from a catalogue the catalogue job built over the same graph.

_CRS = "provenance/OECDCreditorReportingSystem"
_CRS_DATASET = "dc/d/OECD_CreditorReportingSystem"
_OECD = "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"
_WORLD_BANK = "dc/s/WorldBank"
_SHARED_DATASET = "dc/d/WorldBank_Shared"
_SHARED_A = "dc/base/WorldBank_SharedA"
_SHARED_B = "dc/base/WorldBank_SharedB"
_CC_BY = "CreativeCommonsAttributionLicense"
_OECD_TERMS = "OECDTermsAndConditions"

_CATALOGUED_REGISTRY = f"""
[[base]]
source = "{_WORLD_BANK}"
label = "World Bank"
exclusions = []

[[un_system]]
provenance = "undata/p/WHO"
segment = "who"
label = "World Health Organization (WHO)"
"""


def _catalogued_graph() -> FakeMintGraphReader:
    """A documented DFO provenance under its Dataset, a licensed base orphan, and a base
    Dataset with two children, each carrying the citation's licence and documentation
    arcs."""
    return FakeMintGraphReader(
        arcs={
            _OECD: {"typeOf": ("Source",), "name": ("OECD",)},
            _WORLD_BANK: {"typeOf": ("Source",), "name": ("World Bank",)},
            _CRS_DATASET: {
                "typeOf": ("Dataset",),
                "name": ("Creditor Reporting System",),
                "url": ("https://www.oecd.org/en/data/datasets/creditor-reporting-system.html",),
                "isPartOf": (_OECD,),
            },
            _CRS: {
                "typeOf": ("Provenance",),
                "name": ("OECDCreditorReportingSystem",),
                "url": ("https://data-explorer.oecd.org/crs",),
                "source": (_OECD,),
                "isPartOf": (_CRS_DATASET,),
                "licenseType": (_OECD_TERMS,),
                "license": ("https://www.oecd.org/en/about/terms-conditions.html",),
                "licenseAttribution": ("OECD (2026), Creditor Reporting System",),
                "sourceReleaseFrequency": ("P3M",),
                "sourceDataUrl": ("https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",),
                "lastDataRefreshDate": ("2026-10-02",),
                "nextDataRefreshDate": ("2027-01-02",),
                "earliestObservationDate": ("1973",),
                "latestObservationDate": ("2024",),
                "description": ("Activity-level aid flows reported by DAC members.",),
                "dataTransformationDescription": (
                    "## Download the source data\\n\\nThe complete CRS file is downloaded.",
                ),
                "dataTransformationChangelog": (
                    "## 2026-08-20 First release\\n\\n### Added\\n\\n- Dataset added to the graph.\\n",
                ),
                "sourceMethodologyDescription": (
                    "The OECD collects activity-level reports\\nfrom each DAC member.\\n",
                ),
                "dataNotes": ("- Figures for the latest year are preliminary.\\n",),
            },
            _OECD_TERMS: {"name": ("OECD Terms and Conditions",)},
            _CC_BY: {"name": ("Creative Commons Attribution License",)},
            _WDI: {
                "typeOf": ("Provenance",),
                "name": ("WorldDevelopmentIndicators",),
                "url": ("https://datatopics.worldbank.org/world-development-indicators/",),
                "source": (_WORLD_BANK,),
                "licenseType": (_CC_BY,),
                "license": ("https://creativecommons.org/licenses/by/4.0/",),
                "licenseAttribution": ("World Bank (2026), World Development Indicators",),
                "sourceReleaseFrequency": ("P1Y",),
                "lastDataRefreshDate": ("2026-07-01",),
                "earliestObservationDate": ("1960-01",),
                "latestObservationDate": ("2025",),
            },
            _SHARED_DATASET: {
                "typeOf": ("Dataset",),
                "name": ("World Bank Shared",),
                "url": ("https://data.worldbank.org/shared",),
                "isPartOf": (_WORLD_BANK,),
            },
            _SHARED_A: {
                "typeOf": ("Provenance",),
                "url": ("https://data.worldbank.org/shared/a",),
                "source": (_WORLD_BANK,),
                "isPartOf": (_SHARED_DATASET,),
                "licenseType": (_CC_BY,),
                "licenseAttribution": ("World Bank (2026)",),
                "lastDataRefreshDate": ("2026-03-01",),
                "nextDataRefreshDate": ("2026-12-01",),
                "description": ("Child A.",),
                "dataNotes": ("- A note only a sole child would serve.\\n",),
            },
            _SHARED_B: {
                "typeOf": ("Provenance",),
                "url": ("https://data.worldbank.org/shared/b",),
                "source": (_WORLD_BANK,),
                "isPartOf": (_SHARED_DATASET,),
                "licenseType": (_CC_BY,),
                "license": ("https://creativecommons.org/licenses/by/4.0/",),
                "licenseAttribution": ("World Bank (2026)",),
                "lastDataRefreshDate": ("2026-05-01",),
                "nextDataRefreshDate": ("2026-11-01",),
            },
        }
    )


def _catalogue_built_from(graph: FakeMintGraphReader) -> CatalogueSnapshot:
    """The catalogue the job's provenance stage builds from `graph`."""
    stage = build_provenance_table(
        registry=parse_registry(_CATALOGUED_REGISTRY.encode()),
        dfo_provenances=frozenset({_CRS}),
        graph=graph,
    )
    assert stage.refusals == ()
    return restrictive_catalogue(
        provenances=stage.rows.values(),
        dataset_facts=stage.dataset_facts,
        source_documentation=stage.source_documentation,
    )


_CATALOGUED_REFS = [_CRS, _WDI, _SHARED_A]


def _provenances_as_currencies(provenances: tuple[str, ...]) -> CoverageFacts:
    return CoverageFacts(currencies=tuple(sorted(provenances)))


_CATALOGUED_CITATION = Path(__file__).parent / "fixtures" / "catalogued_citation.json"


def test_a_first_mint_s_catalogued_citation_is_the_wire_the_graph_reads_gave():
    """The citation of catalogued sources, licence and documentation included, is the bytes a
    first mint stored when it read them from the graph. The coverage lookup names the
    Provenances each document covers as its currencies, so the bytes pin that set too."""
    graph = _catalogued_graph()
    catalogue = _catalogue_built_from(graph)

    citation, _warnings = build_citation(
        processed=False,
        provenance_refs=_CATALOGUED_REFS,
        source_citations=read_source_citations(
            graph,
            _CATALOGUED_REFS,
            catalogue=catalogue,
            documented=True,
            coverage_facts_of=_provenances_as_currencies,
        ),
        retrieved=_RETRIEVED,
    )

    assert citation.model_dump_json(indent=2) + "\n" == _CATALOGUED_CITATION.read_text()


def test_a_first_mint_s_catalogued_citation_reads_no_graph():
    """Licence and documentation come from the catalogue, and each document is the one
    `read_provenance_doc` reads from the graph, members off the wire included."""
    graph = _catalogued_graph()
    catalogue = _catalogue_built_from(graph)
    reads_before = dict(graph.calls)

    citations = read_source_citations(
        graph,
        _CATALOGUED_REFS,
        catalogue=catalogue,
        documented=True,
        coverage_facts_of=_provenances_as_currencies,
    )

    assert graph.calls == reads_before
    assert {ref: citation.documentation for ref, citation in citations.items()} == {
        ref: read_provenance_doc(
            graph, ref, catalogue=catalogue, coverage_facts_of=_provenances_as_currencies
        )
        for ref in _CATALOGUED_REFS
    }


# Statistics placement in definition manifests.


def _statistics_fixture_frame():
    return build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _REF_A,
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2015",
                "value": 1000.0,
            },
            {
                "stat_var": _REF_A,
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2016",
                "value": 1500.0,
            },
        ],
    )


def _statistics_file(frame):
    """Builds the production `series-statistics/1` artifact.
    These tests cover manifest wiring.
    """
    sink = SeriesStatisticsSink(continuing=())
    sink.write(compute_series_statistics_table(frame))
    return sink.finish()


def _statistics_fixture_result(frame) -> MaterialisationResult:
    return MaterialisationResult(
        frame_chunks=lambda: iter((frame,)),
        frame_sha256="0" * 64,
        content_version="cv_fixture",
        trace=ManifestTrace(
            applied_window=None,
            graph_revision="rev1",
            source_revisions=(),
            resolved_stat_vars=(),
            facets=(),
            normalisations=(),
            conversion=None,
            frame_grammar_version="1.0.0",
            materialiser_build="test-build",
            capability_version="dset_test",
        ),
        coverage=Coverage(rows=len(frame.rows), dimensions=(), window=None),
        warnings=(),
        series_statistics=_statistics_file(frame),
        output_profile=compute_output_profile(
            frame,
            role_columns=("recipient",),
            enrichment=EnrichmentLookup(FakeMintGraphReader(), parties=lambda: {}),
        ),
        conversion_record=None,
        member_conversion_records={_REF_A: None},
        dependencies=Dependencies(
            members="sha256:fixture", sources=(), conversion_sources=(), recipes=(), window=None
        ),
    )


def test_assemble_manifest_carries_the_materialisations_own_series_count():
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored = StoredDefinition(
        spec_id="spec_fixture",
        schema_version=1,
        identity={},
        identity_bytes=b"{}",
        identity_digest="sha256:fixture",
    )

    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )

    assert manifest.series_count == result.series_statistics.series_count == 1


def test_assemble_manifest_carries_code_index_stamp_beside_graph_revision():
    # Assembly stamps the manifest from the `CodeIndex` used for capabilities.
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored = StoredDefinition(
        spec_id="spec_fixture",
        schema_version=1,
        identity={},
        identity_bytes=b"{}",
        identity_digest="sha256:fixture",
    )
    code_index = get_index()

    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=code_index,
        settings=DatasetsBuildSettings(),
    )

    assert manifest.trace["code_index_stamp"] == {
        "content_hash": code_index.stamp.content_hash,
        "exported_at": code_index.stamp.exported_at,
        "graph_endpoint": code_index.stamp.graph_endpoint,
    }
    assert manifest.trace["graph_revision"] == "rev1"


def _assemble_fixture_manifest() -> Manifest:
    return assemble_manifest(
        stored=_fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:f"),
        materialisation=_statistics_fixture_result(_statistics_fixture_frame()),
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )


def test_assemble_manifest_records_the_request_s_origin_catalogue_build(bind_catalogue):
    bind_catalogue(restrictive_catalogue(build_id="20260929T000000Z-abcd1234"))

    manifest = _assemble_fixture_manifest()

    assert manifest.trace["origin_catalogue_build_id"] == "20260929T000000Z-abcd1234"
    assert "origin_catalogue_stale" not in {warning.code for warning in manifest.warnings}


def test_assemble_manifest_reports_a_stale_origin_catalogue_as_a_diagnostic(bind_catalogue):
    built_at = datetime.now(UTC) - timedelta(days=15)
    bind_catalogue(restrictive_catalogue(build_id="old-build", built_at=built_at))

    manifest = _assemble_fixture_manifest()

    assert "origin_catalogue_stale" not in [w.code for w in manifest.warnings]
    (stale,) = [d for d in manifest.diagnostics if d.code == "origin_catalogue_stale"]
    assert stale.severity == "warn"
    assert "old-build" in stale.message


def _minimal_manifest_kwargs() -> dict[str, object]:
    return {
        "content_version": "cv1",
        "prepared_at": _PREPARED_AT,
        "coverage": CoverageModel(rows=0, dimensions=[]),
        "citation": CitationModel(text="", sources=[]),
        "trace": {},
        "artifacts": [],
    }


def test_top_level_series_count_is_allowed_on_a_definition_manifest():
    definition = DefinitionModel(schema_version=1, spec_id="spec_a", identity={})

    manifest = Manifest(definition=definition, series_count=1, **_minimal_manifest_kwargs())

    assert manifest.series_count == 1


# Assembly builds the response. Record functions write version bodies.


def _fixture_stored_definition(*, spec_id: str, identity_digest: str) -> StoredDefinition:
    return StoredDefinition(
        spec_id=spec_id,
        schema_version=1,
        identity={},
        identity_bytes=b"{}",
        identity_digest=identity_digest,
    )


def test_assemble_manifest_derives_output_options_and_writes_nothing():
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored = _fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:fixture")

    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )

    assert manifest.citation.text == citation_line(
        retrieved=_PREPARED_AT.date(), sources=["OECD_DAC2A"], processed=False
    )
    assert manifest.output_options is not None
    assert manifest.output_options.capability_version == "dset_test"
    assert [c.name for c in manifest.output_options.columns] == [
        c.name for c in result.output_profile.columns
    ]


_ARTIFACT_ENTRY = ArtifactEntryModel(
    artifact_digest="sha256:" + "ab" * 32,
    rendering={"format": "csv"},
    bytes=12,
    media_type="text/csv",
    created_at="2026-10-04T12:00:00Z",
    tier="working",
    url="https://example.test/a.csv",
)

_CONVERSION_ENTRY = {
    "recipe": "ConversionRecipeDAC",
    "rate_table_version": "ab" * 32,
    "members": [{"stat_var": _REF_A, "native_unit": "CurrencyEUR"}],
}


@pytest.mark.parametrize(
    ("version", "route"),
    [
        pytest.param("plain", "via", id="plain"),
        pytest.param("converted", "processed by", id="converted"),
        pytest.param("collapsed", "processed by", id="collapsed"),
        pytest.param("unbuildable", "via", id="unbuildable-steps"),
    ],
)
def test_a_manifest_citation_routes_through_the_dfo_as_the_version_s_steps_do(
    version: str, route: str, monkeypatch: pytest.MonkeyPatch
):
    """The manifest's line reads "processed by" the DFO where the version's description
    record holds a step that changes figures: a conversion, or a collapse's sum of variables.
    A version that passes its source's figures through reads "via", as does a converted
    version whose steps cannot be built. The mint still goes on."""
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    if version in {"converted", "unbuildable"}:
        result = replace(result, trace=replace(result.trace, conversion=(_CONVERSION_ENTRY,)))
    if version == "unbuildable":

        def unbuildable(**_: object) -> list[object]:
            raise ValueError("a trace the step models refuse")

        monkeypatch.setattr("datasets_api.manifest.mint_steps", unbuildable)
    manifest = assemble_manifest(
        stored=_fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:f"),
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )
    if version == "collapsed":
        assert manifest.output_options is not None
        manifest = collapsed_head_manifest(
            manifest=manifest,
            derived=DerivedVersion(
                frame_chunks=lambda: iter((frame,)),
                role_columns=(),
                rows=frame.num_rows,
                distinct={},
                frame_sha256="",
                content_version="cv_fixture_collapsed",
                summed=True,
                party_totals_left_out=(),
            ),
            artifact=_ARTIFACT_ENTRY,
            statistics_artifact=_ARTIFACT_ENTRY,
            series_count=1,
            output_options=manifest.output_options,
        )

    assert (
        manifest.citation.text == f"OECD_DAC2A (2026), {route} the Development Finance Observatory"
    )


_SERIES_CONVERSIONS = {
    (
        "Amount_FinancialFlow_DevelopmentAssistance_Nominal",
        (("financingProvider", "country/DEU"), ("financingRecipient", "country/KEN")),
    ): SeriesConversion(
        recipe_id="ConversionRecipeDAC",
        table_provenance_id="provenance/DACTables",
        exchange_rate_table_version="sha256:rates",
        deflator_table_version="sha256:deflators",
        fallback_count=2,
        fallback_examples=(("country/DEU", "DACCountriesTotal"),),
        projected_count=0,
        projected_examples=(),
    ),
    (
        "Amount_FinancialFlow_DevelopmentAssistance_Nominal",
        (("financingProvider", "country/FRA"), ("financingRecipient", "country/KEN")),
    ): SeriesConversion(
        recipe_id="ConversionRecipeDAC",
        table_provenance_id="provenance/DACTables",
        exchange_rate_table_version="sha256:rates",
        deflator_table_version=None,
        fallback_count=0,
        fallback_examples=(),
        projected_count=3,
        projected_examples=("country/FRA@2022",),
    ),
}


@pytest.mark.parametrize("series_conversions", [{}, _SERIES_CONVERSIONS], ids=["none", "two"])
@pytest.mark.parametrize(
    ("default_scope", "plural"),
    [((), False), (("Earth",), False), (("country/KEN", "country/GHA"), True)],
    ids=["bound", "worldwide", "plural"],
)
def test_definition_version_record_stores_its_facts(
    series_conversions: dict[SeriesKey, SeriesConversion],
    default_scope: tuple[str, ...],
    plural: bool,
):
    """The facts store each converted series' record and whether the default place scope was
    plural, alongside the Manifest fields, and assemble the manifest the mint built. Facts
    without them read no records and an unknown scope, and assemble the same manifest."""
    frame = _statistics_fixture_frame()
    result = replace(
        _statistics_fixture_result(frame),
        series_conversions=series_conversions,
        default_scope=default_scope,
    )
    stored = _fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:fixture")
    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )
    record = definition_version_record(
        manifest=manifest,
        stored=stored,
        materialisation=result,
        artifacts=[],
    )

    facts = record.facts
    assert facts == manifest_facts(
        manifest,
        output_profile=result.output_profile,
        series_conversions=series_conversions,
        default_scope=default_scope,
    )
    assert isinstance(facts, ManifestFacts)
    stored_version = _stored_version(record)
    assert persisted_manifest(stored=stored_version, served=manifest) == manifest
    assert stored_output_profile(facts) == result.output_profile
    assert stored_series_conversions(facts) == series_conversions
    assert (facts.series_conversions is not None) == bool(series_conversions)
    assert facts.default_scope_plural is plural
    without_records = manifest_facts(manifest, output_profile=result.output_profile)
    assert stored_series_conversions(without_records) == {}
    assert without_records.default_scope_plural is None
    assert persisted_manifest(
        stored=replace(stored_version, facts=without_records), served=manifest
    ) == persisted_manifest(stored=stored_version, served=manifest)


def _stored_version(record: VersionRecord) -> StoredVersion:
    return StoredVersion(
        subject_id=record.subject_id,
        subject_kind=record.subject_kind,
        content_version=record.content_version,
        schema_version=record.schema_version,
        facts=record.facts,
        base_content_version=record.base_content_version,
    )


def test_definition_version_record_carries_base_content_version_on_the_record():
    """`base_content_version` is the record's own field, outside the strict Manifest schema."""
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored = _fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:fixture")
    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )
    record = definition_version_record(
        manifest=manifest,
        stored=stored,
        materialisation=result,
        artifacts=[],
        base_content_version="cv_base",
    )

    assert record.base_content_version == "cv_base"


class _RecordedVersions(NullVersionRecordWriter):
    def __init__(self) -> None:
        self.records: list[VersionRecord] = []

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        self.records.append(record)
        return True


def test_a_version_record_links_its_artifacts_under_the_digest_they_were_rendered_under():
    """A derived or collapsed version links each file it lists under the digest the file was
    rendered under, its `enrichment_stamp`, which salts the link key of a rendering with a
    party-kind column. A record-only artifact keeps the digest it carries."""
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored = _fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:fixture")
    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=get_index(),
        settings=DatasetsBuildSettings(),
    )
    own = ArtifactRecord(
        artifact_digest="sha256:" + "cd" * 32,
        rendering={"format": "frame"},
        media_type="application/vnd.dfo.frame",
        enrichment_digest="own",
    )
    rendered = _ARTIFACT_ENTRY.model_copy(
        update={"rendering": {"format": "csv", "bundle": False}, "enrichment_stamp": "current"}
    )
    written = definition_version_record(
        manifest=manifest,
        stored=stored,
        materialisation=result,
        artifacts=[rendered],
        record_only_artifacts=[own],
    )
    recorder = _RecordedVersions()
    record_collapsed_head_version(
        manifest=manifest.model_copy(update={"artifacts": [rendered]}),
        subject_id="spec_fixture",
        schema_version=1,
        base_content_version="cv_base",
        output_profile=result.output_profile,
        frame_sha256=result.frame_sha256,
        record_only_artifacts=[own],
        version_writer=recorder,
    )

    (collapsed,) = recorder.records
    for record in (written, collapsed):
        assert [artifact.enrichment_digest for artifact in record.artifacts] == ["current", "own"]


# The caller's injected CodeIndex must also supply the manifest capability stamp.


def _fixture_code_index(*, code_system: str = "syntheticAxis") -> CodeIndex:
    """Builds an index with a synthetic axis absent from the packaged index."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-09-15T00:00:00Z",
            code_counts={code_system: 0},
            statvar_shape_counts={},
            content_hash="sha256:fixture",
        ),
        declarations=[
            CodeSystemDeclaration(
                code_system=code_system,
                hierarchical=False,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="synthetic axis",
            )
        ],
        codes=[],
        statvars=[],
    )


def test_assemble_manifest_never_reads_the_process_singleton(monkeypatch):
    # A singleton lookup would fail this test.
    monkeypatch.setattr(
        "code_index.shared.get_index",
        lambda: (_ for _ in ()).throw(AssertionError("must not read the process singleton")),
    )
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored = _fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:fixture")

    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=_fixture_code_index(),
        settings=DatasetsBuildSettings(),
    )

    assert manifest.output_options is not None
    assert manifest.output_options.groupby.axes == [{"name": "syntheticAxis"}]


def test_assemble_manifest_reports_the_injected_declarations_not_the_singletons():
    # `syntheticAxis` exists only in the injected index.
    assert "syntheticAxis" not in get_index().declarations
    frame = _statistics_fixture_frame()
    result = _statistics_fixture_result(frame)
    stored = _fixture_stored_definition(spec_id="spec_fixture", identity_digest="sha256:fixture")

    manifest = assemble_manifest(
        stored=stored,
        materialisation=result,
        artifacts=[],
        provenance_refs=["OECD_DAC2A"],
        source_citations={},
        prepared_at=_PREPARED_AT,
        capability_version="dset_test",
        materialiser_build="test-build",
        code_index=_fixture_code_index(),
        settings=DatasetsBuildSettings(),
    )

    axes = {axis["name"] for axis in manifest.output_options.groupby.axes}
    assert axes == {"syntheticAxis"}


@pytest.mark.parametrize("family", [None, "aid", "debt_service", "debt_stock", "debt_lending"])
def test_coverage_dimensions_name_their_flow_axis_and_its_members(
    family: FlowFamily | None,
) -> None:
    """The family words the members a flow dimension counts (`role_wording`), which its
    label does not: `financingProvider` counts providers of aid and creditors of debt."""
    coverage = Coverage(
        rows=3,
        dimensions=(
            {"label": "financingProvider", "count": 2},
            {"label": "financingRecipient", "count": 1},
            {"label": "years", "count": 3},
        ),
        window=None,
    )

    dumped = coverage_model(coverage, family=family).model_dump(mode="json")["dimensions"]

    wording = role_wording(family)
    assert dumped == [
        {
            "label": "financingProvider",
            "count": 2,
            "axis": "origin",
            "noun": {"singular": wording.origin.singular, "plural": wording.origin.plural},
        },
        {
            "label": "financingRecipient",
            "count": 1,
            "axis": "destination",
            "noun": {
                "singular": wording.destination.singular,
                "plural": wording.destination.plural,
            },
        },
        {"label": "years", "count": 3, "axis": None},
    ]
    assert (dumped[0]["noun"]["plural"] == "creditors") == (family in _DEBT_FAMILIES)


_DEBT_FAMILIES = ("debt_service", "debt_stock", "debt_lending")


def test_a_stored_coverage_dimension_without_an_axis_carries_it_once_loaded() -> None:
    stored = {
        "rows": 3,
        "dimensions": [
            {"label": "financingProvider", "count": 2},
            {"label": "financingRecipient", "count": 1, "axis": None},
            {"label": "years", "count": 3},
        ],
        "window": None,
    }

    model = CoverageModel.model_validate(stored)

    assert [(d.label, d.axis) for d in model.dimensions] == [
        ("financingProvider", "origin"),
        ("financingRecipient", "destination"),
        ("years", None),
    ]
