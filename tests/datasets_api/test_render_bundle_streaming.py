"""Pins streamed bundle contents, CSV byte identity, deterministic ZIP bytes, and citations."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from io import BytesIO

from datasets_api.manifest_models import CitationSource
from datasets_api.mint_models import MintOptions
from datasets_api.render import render_and_store
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import Table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_STAT_VAR = "dcid:MalariaGrantDisbursements"
_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit")
_GRAPH = FakeMintGraphReader(arcs={})
_ROWS = (
    (_STAT_VAR, "country/KEN", "2015", 1000.0, "USD"),
    (_STAT_VAR, "country/TGO", "2016", 2500.5, "USD"),
)
_TABLE = Table(columns=_COLUMNS, rows=_ROWS)


def _mint_bundle(
    *,
    spec_id: str = "bundle-streaming-test",
    provenance_refs: Sequence[str] = (),
    citations: Mapping[str, CitationSource] | None = None,
) -> tuple[bytes, str]:
    """Return one bundled CSV's bytes and digest from a fresh fake store."""
    artifact_store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=lambda: iter((_TABLE,)),
        options=MintOptions(format="csv", bundle=True),
        spec={"spec_id": spec_id},
        provenance_refs=list(provenance_refs),
        citations=citations or {},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=artifact_store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
        resolved_stat_var_refs=(),
        pivot_max_output_cells=5_000_000,
        groupby_fold_max_rows=2_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    fetched = artifact_store.fetch(rendered.artifact.url)
    return fetched, rendered.artifact.artifact_digest


def test_bundle_zip_streamed_via_open_put_is_a_valid_archive_with_the_expected_members():
    fetched, digest = _mint_bundle()

    with zipfile.ZipFile(BytesIO(fetched)) as zf:
        names = set(zf.namelist())
        assert zf.testzip() is None  # every member's own CRC checks out
        data_bytes = zf.read("data/dataset.csv")

    assert names == {
        "datapackage.json",
        "CITATION.cff",
        "README.md",
        "SOURCES.md",
        "data/dataset.csv",
        "data/codes.csv",
    }
    assert digest.startswith("sha256:")

    # ZIP_STORED preserves the CSV bytes the bare rendering writes.
    artifact_store = FakeArtifactStore()
    bare = render_and_store(
        frame_chunks=lambda: iter((_TABLE,)),
        options=MintOptions(format="csv", bundle=False),
        spec={"spec_id": "bundle-streaming-test"},
        provenance_refs=[],
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=artifact_store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
        resolved_stat_var_refs=(),
        pivot_max_output_cells=5_000_000,
        groupby_fold_max_rows=2_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    assert data_bytes == artifact_store.fetch(bare.artifact.url)


def test_bundle_zip_is_byte_identical_across_two_independent_renders():
    # Fixed entry order, timestamps, and zip64 flags make independent renders byte-identical.
    first_bytes, first_digest = _mint_bundle()
    second_bytes, second_digest = _mint_bundle()

    assert first_digest == second_digest
    assert first_bytes == second_bytes


def test_bundle_names_each_source_as_the_manifest_citation_does():
    """Cited sources use curated labels, urls and licences. Uncited refs use their identifiers.
    An Open Definition licence id gives a `name`, a terms URL a `path`, and a source with
    neither gets no `licenses` entry."""
    wdi = "dc/base/WorldDevelopmentIndicators"
    open_data = "dc/base/OpenDataSource"
    uncited = "dc/base/NotInTheCitation"
    fetched, _digest = _mint_bundle(
        provenance_refs=[wdi, open_data, uncited],
        citations={
            wdi: CitationSource(
                ref=wdi,
                label="World Development Indicators",
                license="Creative Commons Attribution License",
                license_type="CreativeCommonsAttributionLicense",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                url="https://datacatalog.worldbank.org/search/dataset/0037712",
            ),
            open_data: CitationSource(
                ref=open_data,
                label="Open Data Source",
                license="Creative Commons Zero",
                license_type="CreativeCommons0",
            ),
        },
    )

    with zipfile.ZipFile(BytesIO(fetched)) as zf:
        descriptor = json.loads(zf.read("datapackage.json"))
        citation_cff = zf.read("CITATION.cff").decode()

    assert descriptor["sources"] == [
        {
            "title": "World Development Indicators",
            "path": "https://datacatalog.worldbank.org/search/dataset/0037712",
        },
        {"title": "Open Data Source"},
        {"title": uncited},
    ]
    assert descriptor["licenses"] == [
        {
            "path": "https://creativecommons.org/licenses/by/4.0/",
            "title": "Creative Commons Attribution License",
        },
        {"name": "CC0-1.0", "title": "Creative Commons Zero"},
    ]
    assert "World Development Indicators" in citation_cff
    assert "WorldDevelopmentIndicators" not in citation_cff
