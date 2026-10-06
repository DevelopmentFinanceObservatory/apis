"""Tests manifest assembly and the separation between assembly and storage, and the encoded
version-record bodies.
"""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pyarrow.parquet as pq
import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import CodeSystemDeclaration, IndexStamp
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest import (
    assemble_manifest,
    build_citation,
    coverage_model,
    decode_default_scope_plural,
    decode_series_conversions,
    decode_version_body,
    encode_version_body,
    healed_citation,
    persisted_manifest,
    record_definition_version,
)
from datasets_api.manifest_models import (
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
from datasets_api.revision import SentinelRevisionReader
from datasets_api.versions import ArtifactRecord, VersionRecord
from materialise import (
    EnrichmentLookup,
    MaterialisationResult,
)
from materialise.content_version import MintPreimage
from materialise.frame_build import build_frame
from materialise.graph import Facet
from materialise.profile import compute_output_profile
from materialise.provenance_doc import read_source_citations
from materialise.series_conversion import SeriesConversion, SeriesKey
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace
from materialise.window import Coverage
from origin_catalogue import BASE, ProvenanceRow, SourceRef, bound_catalogue
from series_coverage.schema import FacetMetadata, RoleNames
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


class RecordingVersionRecordWriter:
    """A `VersionRecordWriter` that records every call and persists
    nothing, kept local here so this file's own unit tests of the
    assemble/record split don't need a door-level fixture.
    """

    def __init__(self) -> None:
        self.records: list[VersionRecord] = []

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> None:
        self.records.append(record)

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        return None


_STAT_VAR_A = "GrantA"
_STAT_VAR_B = "GrantB"
_REF_A = f"dcid:{_STAT_VAR_A}"
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
            (_STAT_VAR_A, "facet1", "country/USA", "country/KEN", [2015, 2016, 2017, 2018]),
            (_STAT_VAR_B, "facet1", "country/USA", "country/KEN", [2015, 2016, 2017, 2018]),
        ],
        role_names={
            _STAT_VAR_A: RoleNames(donor="provider", recipient="recipient"),
            _STAT_VAR_B: RoleNames(donor="provider", recipient="recipient"),
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
        revision_reader=SentinelRevisionReader(),
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
        provenance_refs=["OECD_DAC2A", "OECD_DAC2A", "OECD_DAC2A"],
        processing_levels=["minor", "minor", "significant"],
        source_citations={},
    )

    assert len(citation.sources) == 2
    by_level = {source.processing_level: source for source in citation.sources}
    assert set(by_level) == {"minor", "significant"}
    assert all(source.ref == "OECD_DAC2A" for source in citation.sources)


def test_build_citation_names_a_catalogued_source_from_the_provenance_table():
    """A Provenance the origin catalogue lists takes its label, url, source and Data Commons
    from the provenance table and its licence, as a display label, its dcid and its terms
    page, from the graph. `citation.text` joins the labels."""
    wdi = "dc/base/WorldDevelopmentIndicators"
    graph = FakeMintGraphReader(
        arcs={
            wdi: {
                "typeOf": ("Provenance",),
                "name": ("WorldDevelopmentIndicators",),
                "licenseType": ("CreativeCommonsAttributionLicense",),
                "license": ("https://creativecommons.org/licenses/by/4.0/",),
            },
            "CreativeCommonsAttributionLicense": {
                "name": ("Creative Commons Attribution License",)
            },
        }
    )
    row = ProvenanceRow(
        id=wdi,
        data_commons=BASE,
        source=SourceRef(ref="dc/s/WorldBank", label="World Bank"),
        name="World Development Indicators",
        url="https://datatopics.worldbank.org/world-development-indicators/",
    )
    refs = [wdi]

    citation, _warnings = build_citation(
        provenance_refs=refs,
        source_citations=read_source_citations(
            graph, refs, catalogue=restrictive_catalogue(provenances=[row])
        ),
    )

    assert citation == CitationModel(
        text="World Development Indicators",
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
        provenance_refs=[crs],
        source_citations=read_source_citations(
            graph, [crs], catalogue=restrictive_catalogue(), documented=True
        ),
    )
    undocumented, _ = build_citation(
        provenance_refs=[crs],
        source_citations=read_source_citations(graph, [crs], catalogue=restrictive_catalogue()),
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
        provenance_refs=refs,
        source_citations=read_source_citations(graph, refs, catalogue=restrictive_catalogue()),
    )

    (source,) = citation.sources
    assert (source.label, source.source, source.data_commons) == ("Some Dataset", None, None)


def test_build_citation_falls_back_to_the_ref_for_a_source_with_no_resolved_entry():
    citation, _warnings = build_citation(provenance_refs=["dc/base/NotHeld"], source_citations={})

    assert citation.sources == [
        CitationSource(ref="dc/base/NotHeld", label="dc/base/NotHeld", license=None, url=None)
    ]
    assert citation.text == "dc/base/NotHeld"


_WDI = "dc/base/WorldDevelopmentIndicators"


def _documented_wdi() -> tuple[FakeMintGraphReader, ProvenanceRow]:
    """A catalogued, licensed and documented source: a mint fills every citation field."""
    graph = FakeMintGraphReader(
        arcs={
            _WDI: {
                "typeOf": ("Provenance",),
                "name": ("WorldDevelopmentIndicators",),
                "licenseType": ("CreativeCommonsAttributionLicense",),
                "license": ("https://creativecommons.org/licenses/by/4.0/",),
                "licenseAttribution": ("World Bank (2026), World Development Indicators",),
                "sourceReleaseFrequency": ("P1Y",),
            },
            "CreativeCommonsAttributionLicense": {
                "name": ("Creative Commons Attribution License",)
            },
        }
    )
    row = ProvenanceRow(
        id=_WDI,
        data_commons=BASE,
        source=SourceRef(ref="dc/s/WorldBank", label="World Bank"),
        name="World Development Indicators",
        url="https://datatopics.worldbank.org/world-development-indicators/",
    )
    return graph, row


def test_healed_citation_fills_an_old_body_with_what_a_fresh_mint_stores():
    """A body stored before the citation carried a licence, url, source and Data Commons heals to
    the source a first mint builds from the same live read."""
    graph, row = _documented_wdi()
    catalogue = restrictive_catalogue(provenances=[row])
    fresh, _ = build_citation(
        provenance_refs=[_WDI],
        source_citations=read_source_citations(graph, [_WDI], catalogue=catalogue, documented=True),
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


def test_healed_citation_reads_nothing_for_a_fully_populated_body():
    graph, row = _documented_wdi()
    catalogue = restrictive_catalogue(provenances=[row])
    stored, _ = build_citation(
        provenance_refs=[_WDI],
        source_citations=read_source_citations(graph, [_WDI], catalogue=catalogue, documented=True),
    )
    reads_before = dict(graph.calls)

    with bound_catalogue(catalogue):
        healed, complete = healed_citation(stored, graph=graph, deadline=None, coverage_index=None)

    assert (healed, complete) == (stored, True)
    assert graph.calls == reads_before


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
    sink = SeriesStatisticsSink()
    sink.write(compute_series_statistics_table(frame))
    return sink.finish()


def _statistics_fixture_result(frame) -> MaterialisationResult:
    return MaterialisationResult(
        frame_chunks=lambda: iter((frame,)),
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
            frame, role_columns=("recipient",), enrichment=EnrichmentLookup(FakeMintGraphReader())
        ),
        conversion_record=None,
        member_conversion_records={_REF_A: None},
        mint_preimage=MintPreimage(graph_revision="rev1", source_revisions=(), facets=()),
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


def test_assemble_manifest_warns_when_the_origin_catalogue_is_stale(bind_catalogue):
    built_at = datetime.now(UTC) - timedelta(days=15)
    bind_catalogue(restrictive_catalogue(build_id="old-build", built_at=built_at))

    manifest = _assemble_fixture_manifest()

    (stale,) = [w for w in manifest.warnings if w.code == "origin_catalogue_stale"]
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

    assert manifest.output_options is not None
    assert manifest.output_options.capability_version == "dset_test"
    assert [c.name for c in manifest.output_options.columns] == [
        c.name for c in result.output_profile.columns
    ]


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
def test_record_definition_version_writes_the_encoded_body_exactly_once(
    series_conversions: dict[SeriesKey, SeriesConversion],
    default_scope: tuple[str, ...],
    plural: bool,
):
    """The body stores each converted series' record and whether the default place scope was
    plural, alongside the Manifest fields. Decoding the body returns both and the same
    manifest fields as a body without them. A body without the keys decodes to no records
    and an unknown scope."""
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
    writer = RecordingVersionRecordWriter()

    record_definition_version(
        manifest=manifest,
        stored=stored,
        materialisation=result,
        artifacts=[],
        version_writer=writer,
    )

    assert len(writer.records) == 1
    record = writer.records[0]
    assert record.immutable_body == encode_version_body(
        manifest,
        output_profile=result.output_profile,
        series_conversions=series_conversions,
        default_scope=default_scope,
    )
    assert "output_options" not in record.immutable_body
    assert "collection" not in record.immutable_body
    assert persisted_manifest(body=record.immutable_body, served=manifest) == manifest
    assert record.immutable_body["output_profile"] == result.output_profile.to_json()
    assert decode_series_conversions(record.immutable_body) == series_conversions
    assert ("series_conversions" in record.immutable_body) == bool(series_conversions)
    without_records = encode_version_body(manifest, output_profile=result.output_profile)
    assert decode_series_conversions(without_records) == {}
    assert decode_default_scope_plural(record.immutable_body) is plural
    assert decode_default_scope_plural(without_records) is None
    assert decode_version_body(record.immutable_body) == decode_version_body(without_records)


def test_record_definition_version_carries_base_content_version_off_the_stored_body():
    """`base_content_version` belongs on `VersionRecord`, outside `immutable_body` and its
    strict Manifest schema.
    """
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
    writer = RecordingVersionRecordWriter()

    record_definition_version(
        manifest=manifest,
        stored=stored,
        materialisation=result,
        artifacts=[],
        version_writer=writer,
        base_content_version="cv_base",
    )

    assert len(writer.records) == 1
    record = writer.records[0]
    assert record.base_content_version == "cv_base"
    assert "base_content_version" not in record.immutable_body


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


def test_coverage_dimensions_name_the_flow_axis_of_their_declared_role() -> None:
    coverage = Coverage(
        rows=3,
        dimensions=(
            {"label": "financingProvider", "count": 2},
            {"label": "financingRecipient", "count": 1},
            {"label": "years", "count": 3},
        ),
        window=None,
    )

    model = coverage_model(coverage)

    assert [(d.label, d.axis) for d in model.dimensions] == [
        ("financingProvider", "origin"),
        ("financingRecipient", "destination"),
        ("years", None),
    ]


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
