"""A serve call resolves each cited provenance and parent Dataset once."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Sequence
from dataclasses import replace
from io import BytesIO
from typing import cast

import pytest

from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import manifest, serve
from datasets_api.revision import SentinelRevisionReader
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.graph import GraphReadError
from materialise.provenance_doc import (
    _CITATION_LICENCE_PROPERTIES,
    _CITATION_LINKED_PROPERTIES,
    _CITATION_NODE_PROPERTIES,
    _NODE_PROPERTIES,
    CoverageFacts,
)
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter
from .test_serve_bounds import (
    _CSV_HEADER,
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


_CITATION_PROPERTIES = {
    _CITATION_NODE_PROPERTIES,
    _CITATION_LICENCE_PROPERTIES,
    _CITATION_LINKED_PROPERTIES,
}


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
        revision_reader=SentinelRevisionReader(),
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


def test_collection_mint_reads_the_cited_source_once_regardless_of_member_count():
    stat_vars = [
        "dcid:ProvenanceMemoCollectionA",
        "dcid:ProvenanceMemoCollectionB",
        "dcid:ProvenanceMemoCollectionC",
    ]
    requests: list = []
    graph = _graph_for(stat_vars)
    bodies = {
        stat_var.removeprefix("dcid:"): _csv_body(stat_var=stat_var, num_rows=2)
        for stat_var in stat_vars
    }
    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=_sdmx_transport(bodies, requests),
    )
    artifact_store = FakeArtifactStore()
    client = _client(graph=graph, sdmx=sdmx, stat_vars=stat_vars, artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_bundled_mint_body(statvars=stat_vars))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert len(manifest["collection"]["members"]) == len(stat_vars)
    assert _citation_reads(graph) == [(_PROVENANCE_REF,), (_DATASET, _LICENCE)]
    assert _documentation_reads(graph) == [(_PROVENANCE_REF,)]
    assert [source["label"] for source in manifest["citation"]["sources"]] == [
        "OECD Development Assistance Committee"
    ]
    # All member resources use the same source title.
    assert _bundle_source_titles(artifact_store, manifest) == [
        "OECD Development Assistance Committee"
    ]
    assert graph.calls["node_label"] == 0


# Two members, each served from its own facet with its own provenance.
_IMPORT_A = "dc/base/ImportA"
_IMPORT_B = "dc/base/ImportB"
_DATASET_A = "dc/d/Publisher_DatasetA"
_MEMBER_FACETS = {
    "dcid:DistinctRefMemberA": ("facetA", _IMPORT_A),
    "dcid:DistinctRefMemberB": ("facetB", _IMPORT_B),
}


def _distinct_ref_csv(*, stat_var: str, facet_id: str, provenance: str) -> str:
    rows = "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var.removeprefix('dcid:')},{_RECIPIENT},USD,"
        f"GrantEquivalent,NotApplicable,{provenance},{2000 + i},1000,,{facet_id}\r\n"
        for i in range(2)
    )
    return _CSV_HEADER + rows


def _distinct_ref_coverage() -> FakeCoverageService:
    bare = {stat_var.removeprefix("dcid:"): facet for stat_var, facet in _MEMBER_FACETS.items()}
    return FakeCoverageService(
        build_coverage_index(
            [
                (stat_var, facet_id, "country/USA", _RECIPIENT, [2020])
                for stat_var, (facet_id, _provenance) in bare.items()
            ],
            role_names={
                stat_var: RoleNames(donor="provider", recipient="recipient") for stat_var in bare
            },
            facet_metadata={
                facet_id: FacetMetadata(
                    provenance_id=provenance, measurement_method="GrantEquivalent", unit="USD"
                )
                for facet_id, provenance in bare.values()
            },
        )
    )


def test_collection_members_with_distinct_refs_reuse_the_collection_s_one_citation_read(
    version_store,
):
    """The two refs are batched. Only `_IMPORT_A` has a parent Dataset. Member version
    records reuse the collection citation."""
    stat_vars = list(_MEMBER_FACETS)
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        stat_var.removeprefix("dcid:"): {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        }
        for stat_var in stat_vars
    }
    arcs[_IMPORT_A] = {
        "typeOf": ("Provenance",),
        "isPartOf": (_DATASET_A,),
        "licenseType": ("PublicDomain",),
    }
    arcs[_DATASET_A] = {"typeOf": ("Dataset",), "name": ("Dataset A",)}
    arcs[_IMPORT_B] = {"typeOf": ("Provenance",), "name": ("Import B",)}
    graph = _RecordingGraphReader(
        arcs=arcs,
        facets={
            (stat_var.removeprefix("dcid:"), _RECIPIENT): [
                replace(_FACET, facet_id=facet_id, provenance_id=provenance)
            ]
            for stat_var, (facet_id, provenance) in _MEMBER_FACETS.items()
        },
        labels={_RECIPIENT: "Kenya"},
    )
    requests: list = []
    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=_sdmx_transport(
            {
                stat_var.removeprefix("dcid:"): _distinct_ref_csv(
                    stat_var=stat_var, facet_id=facet_id, provenance=provenance
                )
                for stat_var, (facet_id, provenance) in _MEMBER_FACETS.items()
            },
            requests,
        ),
    )
    store = FakeDatasetStore()
    snapshot_reader = LiveSnapshotReader(store=version_store)
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=snapshot_reader,
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_distinct_ref_coverage(),
    )
    # The scope leaves the provenance axis unbound, so each member takes
    # its own facet's provenance.
    body = _bundled_mint_body(statvars=stat_vars)
    del body["input"]["scope"]["facets"]["provenance"]

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    collection_sources = {source["ref"]: source for source in manifest["citation"]["sources"]}
    assert collection_sources == {
        _IMPORT_A: {
            "ref": _IMPORT_A,
            "label": "Dataset A",
            "license": "PublicDomain",
            "license_type": "PublicDomain",
            "documentation": _UNDOCUMENTED,
        },
        _IMPORT_B: {"ref": _IMPORT_B, "label": "Import B", "documentation": _UNDOCUMENTED},
    }
    assert _citation_reads(graph) == [(_IMPORT_A, _IMPORT_B), (_DATASET_A, "PublicDomain")]
    assert _documentation_reads(graph) == [(_IMPORT_A,), (_IMPORT_B,)]

    member_refs = set()
    for member in manifest["collection"]["members"]:
        stored = snapshot_reader.read_version(
            member["definition"]["spec_id"], member["content_version"]
        )
        assert stored is not None
        (member_source,) = cast(list, cast(dict, stored.immutable_body["citation"])["sources"])
        member_refs.add(member_source["ref"])
        expected = {
            "processing_level": None,
            "license": None,
            "license_type": None,
            "license_url": None,
            "url": None,
            "source": None,
            "data_commons": None,
        } | collection_sources[member_source["ref"]]
        expected["documentation"] = {
            "publisher": None,
            "description": None,
            "source_data_url": None,
            "attribution": None,
            "update_cadence": None,
            "last_update": None,
            "years_covered": None,
            "granularity": None,
        } | _UNDOCUMENTED
        assert member_source == expected
    assert member_refs == {_IMPORT_A, _IMPORT_B}


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
    assert caplog.records[1].getMessage().startswith(serve.BUNDLE_NOT_LINKED_MARKER)


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
