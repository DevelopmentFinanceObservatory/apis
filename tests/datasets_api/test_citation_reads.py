"""A serve call resolves each cited provenance and parent Dataset once."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Sequence
from io import BytesIO

import pytest

from datasets_api import manifest, serve
from dc_sdmx import SdmxClient
from materialise.graph import GraphReadError
from materialise.provenance_doc import (
    _CITATION_LINKED_PROPERTIES,
    _CITATION_NODE_PROPERTIES,
    _NODE_PROPERTIES,
    CoverageFacts,
)
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter
from .test_serve_bounds import (
    _FACET,
    _RECIPIENT,
    _coverage_service,
    _csv_body,
    _mint_body,
    _sdmx_transport,
)

# Shared by every member of the first two tests' mints.
_PROVENANCE_REF = "OECD_DAC2A"
_DATASET = "dc/d/OECD_DevelopmentAssistanceCommittee"
_LICENCE = "CreativeCommonsAttributionLicense"
# What a documentation read stores for a source the graph documents nothing about.
_UNDOCUMENTED = {"currencies": [], "transformation_log": [], "version_history": []}


class _RecordingGraphReader(FakeMintGraphReader):
    """Records graph reads so citation reads can be separated from facet probes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.arc_reads: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self.arc_reads.append((tuple(dcids), tuple(properties)))
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


def _graph_for(
    stat_vars: list[str], graph_type: type[_RecordingGraphReader] = _RecordingGraphReader
) -> _RecordingGraphReader:
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        stat_var.removeprefix("dcid:"): {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        }
        for stat_var in stat_vars
    }
    arcs[_PROVENANCE_REF] = {
        "typeOf": ("Provenance",),
        "name": ("OECD_DAC2A",),
        "isPartOf": (_DATASET,),
        "licenseType": (_LICENCE,),
    }
    arcs[_LICENCE] = {"name": ("Creative Commons Attribution License",)}
    arcs[_DATASET] = {
        "typeOf": ("Dataset",),
        "name": ("OECD Development Assistance Committee",),
        "url": ("https://data-explorer.oecd.org/",),
    }
    facets = {(stat_var.removeprefix("dcid:"), _RECIPIENT): [_FACET] for stat_var in stat_vars}
    return graph_type(arcs=arcs, facets=facets, labels={_RECIPIENT: "Kenya"})


_CITATION_PROPERTIES = {_CITATION_NODE_PROPERTIES, _CITATION_LINKED_PROPERTIES}


def _citation_reads(graph: _RecordingGraphReader) -> list[tuple[str, ...]]:
    """Returns citation-read dcids in call order, excluding facet refresh probes and
    documentation reads."""
    return [dcids for dcids, properties in graph.arc_reads if properties in _CITATION_PROPERTIES]


def _documentation_reads(graph: _RecordingGraphReader) -> list[tuple[str, ...]]:
    """Returns the dcids of each documentation read's first read, the documented node."""
    return [dcids for dcids, properties in graph.arc_reads if properties == _NODE_PROPERTIES]


class _DocumentationFailingGraphReader(_RecordingGraphReader):
    """Fails the child read a Dataset-parented provenance's documentation needs."""

    def node_subjects(self, dcid: str, **kwargs):
        raise GraphReadError(f"transport error reading the children of {dcid!r}")


def _bundled_mint_body(*, statvars: list[str]) -> dict:
    return {**_mint_body(statvars=statvars), "options": {"bundle": True}}


def _bundle_source_titles(artifact_store: FakeArtifactStore, manifest: dict) -> list[str]:
    artifact = manifest["artifacts"][0]  # the bundle, followed by series statistics
    with zipfile.ZipFile(BytesIO(artifact_store.fetch(artifact["url"]))) as zf:
        descriptor = json.loads(zf.read("datapackage.json"))
    return [source["title"] for source in descriptor["sources"]]


def _client(*, graph, sdmx, stat_vars: list[str], artifact_store: FakeArtifactStore):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service([(stat_var, _RECIPIENT) for stat_var in stat_vars]),
    )


def test_single_definition_mint_reads_the_cited_source_once():
    stat_var = "dcid:ProvenanceMemoDefinition"
    requests: list = []
    graph = _graph_for([stat_var])
    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=_sdmx_transport(
            {stat_var.removeprefix("dcid:"): _csv_body(stat_var=stat_var, num_rows=2)}, requests
        ),
    )
    artifact_store = FakeArtifactStore()
    client = _client(graph=graph, sdmx=sdmx, stat_vars=[stat_var], artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_bundled_mint_body(statvars=[stat_var]))

    assert resp.status_code == 200, resp.text
    assert _citation_reads(graph) == [(_PROVENANCE_REF,), (_DATASET, _LICENCE)]
    assert _documentation_reads(graph) == [(_PROVENANCE_REF,)]
    assert resp.json()["citation"]["sources"] == [
        {
            "ref": _PROVENANCE_REF,
            "label": "OECD Development Assistance Committee",
            "license": "Creative Commons Attribution License",
            "license_type": _LICENCE,
            "url": "https://data-explorer.oecd.org/",
            "documentation": _UNDOCUMENTED,
        }
    ]
    assert _bundle_source_titles(artifact_store, resp.json()) == [
        "OECD Development Assistance Committee"
    ]
    assert graph.calls["node_label"] == 0


# Two members, each served from its own facet with its own provenance.
_IMPORT_A = "dc/base/ImportA"
_IMPORT_B = "dc/base/ImportB"


def test_a_failed_documentation_read_mints_with_the_citation_and_no_documentation(caplog):
    stat_var = "dcid:ProvenanceMemoUndocumented"
    graph = _graph_for([stat_var], _DocumentationFailingGraphReader)
    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=_sdmx_transport(
            {stat_var.removeprefix("dcid:"): _csv_body(stat_var=stat_var, num_rows=2)}, []
        ),
    )
    client = _client(
        graph=graph, sdmx=sdmx, stat_vars=[stat_var], artifact_store=FakeArtifactStore()
    )

    with caplog.at_level(logging.WARNING, logger="materialise.provenance_doc"):
        resp = client.post("/api/datasets", json=_bundled_mint_body(statvars=[stat_var]))

    assert resp.status_code == 200, resp.text
    assert resp.json()["citation"]["sources"] == [
        {
            "ref": _PROVENANCE_REF,
            "label": "OECD Development Assistance Committee",
            "license": "Creative Commons Attribution License",
            "license_type": _LICENCE,
            "url": "https://data-explorer.oecd.org/",
        }
    ]
    assert [
        (record.name, record.levelno, type(record.exc_info[1]) if record.exc_info else None)
        for record in caplog.records
    ] == [
        ("materialise.provenance_doc", logging.WARNING, GraphReadError),
        ("datasets_api.serve", logging.WARNING, None),
    ]
    assert caplog.records[1].getMessage().startswith(serve.BUNDLE_LINKED_INCOMPLETE_MARKER)


def test_a_mint_documents_its_sources_from_the_mint_s_coverage_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    """Currencies and granularity come from the coverage index the mint admitted against."""
    looked_up: list[tuple[str, ...]] = []

    def coverage_facts(index, provenance_dcids: tuple[str, ...]) -> CoverageFacts:
        assert index is not None
        looked_up.append(provenance_dcids)
        return CoverageFacts(currencies=("USD",), granularity="Annual, by provider and recipient")

    monkeypatch.setattr(manifest, "coverage_facts_from_index", coverage_facts)
    stat_var = "dcid:ProvenanceMemoCoverageFacts"
    graph = _graph_for([stat_var])
    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=_sdmx_transport(
            {stat_var.removeprefix("dcid:"): _csv_body(stat_var=stat_var, num_rows=2)}, []
        ),
    )
    client = _client(
        graph=graph, sdmx=sdmx, stat_vars=[stat_var], artifact_store=FakeArtifactStore()
    )

    resp = client.post("/api/datasets", json=_bundled_mint_body(statvars=[stat_var]))

    assert resp.status_code == 200, resp.text
    (source,) = resp.json()["citation"]["sources"]
    assert source["documentation"]["currencies"] == ["USD"]
    assert source["documentation"]["granularity"] == "Annual, by provider and recipient"
    assert looked_up == [(_PROVENANCE_REF,)]
