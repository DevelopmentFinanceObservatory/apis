"""A file that can hold a party-kind cell is linked under a key salted with the code index's
`enrichment_digest`, so a change to a party's kind re-renders it; any other file keeps its key.

Two code indexes that differ in one party's kind serve the same version from one shared store.
The party-kind file each lists carries its index's digest as `enrichment_stamp`, and its
`rendering` is the same under both. A frame with no party role writes no party-kind cell, so
its file has one link under both.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import replace

import httpx
import pytest

from artifact_store import LiveArtifactInventory
from code_index import CodeIndex, parse_index
from code_index.shared import get_index
from dataset_store.snapshot import LiveSnapshotReader, StoredArtifactLocation
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.render import current_enrichment_locations
from datasets_api.version_store import (
    LiveHeadStore,
    LiveVersionLister,
    LiveVersionRecordWriter,
    enriched_rendering_key,
)
from datasets_api.versions import carries_party_kind
from dc_sdmx import SdmxClient
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.code_index.support import INDEX_FIXTURE
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from . import test_mint_single_role as subject_only
from ._app_factories import make_client
from ._graph_doubles import FakeMintGraphReader
from .conftest import FakeDatasetStore
from .test_mint_party_kind import (
    _DONOR,
    _FACET,
    _HEADER,
    _PROVENANCE,
    _RECIPIENTS,
    _STAT_VAR,
)
from .test_pin_door import FakeArtifactReferenceWriter
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_CHANGED_PARTY = "DACAfricaUnallocated"
_PUBLISHER = "publisher-principal"
_KIND_COLUMN_OPTIONS: dict[str, object] = {
    "format": "parquet",
    "bundle": False,
    "layout": "frame",
    "columns": ["party_kind"],
}
_LONG_BUNDLE_OPTIONS: dict[str, object] = {"format": "csv", "bundle": True}
_LONG_PARQUET_OPTIONS: dict[str, object] = {"format": "parquet", "bundle": False}
_FRAME_OPTIONS: dict[str, object] = {"format": "parquet", "bundle": False, "layout": "frame"}
_CAN_HOLD_A_KIND = pytest.mark.parametrize(
    "options",
    [_KIND_COLUMN_OPTIONS, _LONG_BUNDLE_OPTIONS, _LONG_PARQUET_OPTIONS],
    ids=["frame-layout-party-kind-column", "default-long-bundle", "default-long-parquet"],
)


@pytest.fixture(scope="module")
def index_x() -> CodeIndex:
    return get_index()


@pytest.fixture(scope="module")
def index_y(index_x: CodeIndex) -> CodeIndex:
    """The fixture index with one residual read as a party."""
    payload = json.loads(gzip.decompress(INDEX_FIXTURE.read_bytes()))
    for row in payload["parties"]:
        if row["dcid"] == _CHANGED_PARTY:
            assert row["kind"] == "residual"
            row["kind"] = "party"
            row["party_class"] = "official"
    index = parse_index(payload)
    assert index.parties[_CHANGED_PARTY] == replace(
        index_x.parties[_CHANGED_PARTY], kind="party", party_class="official"
    )
    return index


class _Rig:
    """One dataset store, version store and artifact store shared by every client."""

    def __init__(self, version_store) -> None:
        self.version_store = version_store
        self.dataset_store = FakeDatasetStore()
        self.version_writer = LiveVersionRecordWriter(store=version_store)
        self.artifact_store = _InventoryRecordingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        )

    def client(self, code_index: CodeIndex):
        csv_body = _HEADER + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{recipient},{_DONOR},USD,OECD_DAC2A,"
            f"NotApplicable,{_PROVENANCE},2022,{1000 + i},,facet1\r\n"
            for i, recipient in enumerate(_RECIPIENTS)
        )
        coverage = FakeCoverageService(
            build_coverage_index(
                [(_STAT_VAR, "facet1", _DONOR, recipient, [2022]) for recipient in _RECIPIENTS],
                role_names={
                    _STAT_VAR: RoleNames(donor="financingProvider", recipient="financingRecipient")
                },
                facet_metadata={
                    "facet1": FacetMetadata(
                        provenance_id=_PROVENANCE, measurement_method="OECD_DAC2A", unit="USD"
                    )
                },
            )
        )
        graph = FakeMintGraphReader(
            arcs={
                _STAT_VAR: {
                    "typeOf": ("StatisticalVariable",),
                    "observationProperties": ("financingProvider", "financingRecipient"),
                    "measuredProperty": ("amount",),
                },
                _DONOR: {"typeOf": ("Country",), "name": ("France",)},
                **{dcid: {"typeOf": types} for dcid, types in _RECIPIENTS.items()},
            },
            facets={(_STAT_VAR, recipient): [_FACET] for recipient in _RECIPIENTS},
            labels={_PROVENANCE: "OECD DAC2a"},
        )
        return make_client(
            registry=LiveSpecRegistry(store=self.dataset_store),
            spec_reader=LiveSpecReader(store=self.dataset_store),
            graph=graph,
            sdmx=SdmxClient(
                base_url="https://datacommons.one.org/core/api",
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        text=csv_body,
                        headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                    )
                ),
            ),
            artifact_store=self.artifact_store,
            version_writer=self.version_writer,
            snapshot_reader=LiveSnapshotReader(store=self.version_store),
            reference_writer=FakeArtifactReferenceWriter(),
            coverage=coverage,
            code_index=code_index,
            head_writer=LiveHeadStore(store=self.version_store),
        )

    def subject_mint(self, code_index: CodeIndex, options: dict[str, object]) -> dict:
        """A mint of a dataset whose frame holds only the subject role, no party role."""
        client = make_client(
            registry=LiveSpecRegistry(store=self.dataset_store),
            spec_reader=LiveSpecReader(store=self.dataset_store),
            graph=subject_only._graph(),
            sdmx=SdmxClient(
                base_url=subject_only._BASE_URL, transport=subject_only._sdmx_transport([])
            ),
            artifact_store=self.artifact_store,
            version_writer=self.version_writer,
            snapshot_reader=LiveSnapshotReader(store=self.version_store),
            reference_writer=FakeArtifactReferenceWriter(),
            code_index=code_index,
            head_writer=LiveHeadStore(store=self.version_store),
        )
        resp = client.post("/api/datasets", json={**subject_only._BODY, "options": options})
        assert resp.status_code == 200, resp.text
        return resp.json()

    def mint(self, code_index: CodeIndex, options: dict[str, object]) -> dict:
        body = {
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [f"dcid:{_STAT_VAR}"],
                "scope": {
                    "entities": {
                        "financingProvider": [_DONOR],
                        "financingRecipient": sorted(_RECIPIENTS),
                    },
                    "facets": {
                        "provenance": {"ref": _PROVENANCE},
                        "measurementMethod": {"ref": "OECD_DAC2A"},
                        "unit": {"ref": "USD"},
                    },
                },
            },
            "options": options,
        }
        resp = self.client(code_index).post("/api/datasets", json=body)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def listed_stamps(self, code_index: CodeIndex, spec_id: str) -> dict[str, str | None]:
        """The `enrichment_stamp` the versions route lists for each version of `spec_id`."""
        client = make_client(
            registry=LiveSpecRegistry(store=self.dataset_store),
            spec_reader=LiveSpecReader(store=self.dataset_store),
            snapshot_reader=LiveSnapshotReader(store=self.version_store),
            recipe_store=LiveRecipeStore(store=self.version_store),
            version_lister=LiveVersionLister(store=self.version_store),
            code_index=code_index,
            publish_allowed_principals=frozenset({_PUBLISHER}),
        )
        resp = client.get(
            f"/api/datasets/{spec_id}/versions", headers={"x-dfo-principal": _PUBLISHER}
        )
        assert resp.status_code == 200, resp.text
        return {
            version["content_version"]: version.get("enrichment_stamp")
            for version in resp.json()["versions"]
        }

    def locations(self, manifest: dict) -> list[StoredArtifactLocation]:
        return LiveSnapshotReader(store=self.version_store).read_artifact_locations(
            manifest["definition"]["spec_id"], manifest["content_version"]
        )


def _requested(manifest: dict) -> list[dict]:
    """The listed data artifacts: every artifact but the series statistics."""
    return [a for a in manifest["artifacts"] if a["rendering"].get("statistics") != "series"]


def _of_rendering(locations: list[StoredArtifactLocation], rendering: dict) -> list:
    return [loc for loc in locations if dict(loc.rendering) == rendering]


@_CAN_HOLD_A_KIND
def test_a_party_kind_file_is_relinked_under_the_new_digest_and_listed_once(
    version_store, index_x: CodeIndex, index_y: CodeIndex, options: dict[str, object]
):
    assert index_x.enrichment_digest != index_y.enrichment_digest
    rig = _Rig(version_store)

    under_x = rig.mint(index_x, options)
    under_y = rig.mint(index_y, options)

    assert under_x["content_version"] == under_y["content_version"]
    (file_x,) = _requested(under_x)
    (file_y,) = _requested(under_y)
    assert file_x["enrichment_stamp"] == index_x.enrichment_digest
    assert file_y["enrichment_stamp"] == index_y.enrichment_digest
    # The salt never enters the rendering a caller selects by.
    assert file_x["rendering"] == file_y["rendering"]
    assert "enrichment_digest" not in json.dumps(file_y["rendering"])
    # The file linked under X is not the one served under Y: it was rendered again.
    assert file_x["artifact_digest"] != file_y["artifact_digest"]

    linked = _of_rendering(rig.locations(under_y), file_y["rendering"])
    assert {loc.artifact_digest for loc in linked} == {
        file_x["artifact_digest"],
        file_y["artifact_digest"],
    }
    for stamp, served in ((index_x.enrichment_digest, file_x), (index_y.enrichment_digest, file_y)):
        (current,) = current_enrichment_locations(linked, enrichment_digest=stamp)
        assert current.artifact_digest == served["artifact_digest"]
        assert current.rendering_key == enriched_rendering_key(served["rendering"], stamp)


@_CAN_HOLD_A_KIND
def test_a_party_kind_file_linked_under_the_old_digest_is_served_again_when_it_returns(
    version_store, index_x: CodeIndex, index_y: CodeIndex, options: dict[str, object]
):
    rig = _Rig(version_store)
    first = rig.mint(index_x, options)
    rig.mint(index_y, options)

    back = rig.mint(index_x, options)

    assert _requested(back) == _requested(first)


def test_a_file_that_cannot_hold_a_party_kind_is_selected_under_both_digests(
    version_store, index_x: CodeIndex, index_y: CodeIndex
):
    rig = _Rig(version_store)

    under_x = rig.mint(index_x, _FRAME_OPTIONS)
    under_y = rig.mint(index_y, _FRAME_OPTIONS)

    (file_x,) = _requested(under_x)
    (file_y,) = _requested(under_y)
    assert file_x == file_y
    assert len(_of_rendering(rig.locations(under_y), file_y["rendering"])) == 1
    assert "enrichment_stamp" not in file_y
    linked = rig.locations(under_y)
    assert current_enrichment_locations(linked, enrichment_digest="any-digest") == linked


@pytest.mark.parametrize(
    "options", [_LONG_BUNDLE_OPTIONS, _LONG_PARQUET_OPTIONS], ids=["long-bundle", "long-parquet"]
)
def test_a_long_file_of_a_frame_with_no_party_role_keeps_its_one_link_under_a_new_digest(
    version_store, index_x: CodeIndex, index_y: CodeIndex, options: dict[str, object]
):
    rig = _Rig(version_store)

    under_x = rig.subject_mint(index_x, options)
    under_y = rig.subject_mint(index_y, options)

    (file_x,) = _requested(under_x)
    (file_y,) = _requested(under_y)
    assert file_x == file_y
    assert "enrichment_stamp" not in file_y
    assert len(_of_rendering(rig.locations(under_y), file_y["rendering"])) == 1


def test_the_versions_route_lists_the_stamp_each_minted_version_s_kind_file_carries(
    version_store, index_x: CodeIndex
):
    rig = _Rig(version_store)

    party_free = rig.subject_mint(index_x, _LONG_PARQUET_OPTIONS)
    party = rig.mint(index_x, _LONG_PARQUET_OPTIONS)

    (party_free_file,) = _requested(party_free)
    (party_file,) = _requested(party)
    assert "enrichment_stamp" not in party_free_file
    assert party_file["enrichment_stamp"] == index_x.enrichment_digest
    assert rig.listed_stamps(index_x, party_free["definition"]["spec_id"]) == {
        party_free["content_version"]: None
    }
    assert rig.listed_stamps(index_x, party["definition"]["spec_id"]) == {
        party["content_version"]: index_x.enrichment_digest
    }


_PARQUET = {"format": "parquet", "bundle": False}


@pytest.mark.parametrize(
    ("rendering", "carries"),
    [
        pytest.param({**_PARQUET, "layout": 4}, True, id="long-layout"),
        pytest.param({**_PARQUET, "layout": "frame"}, False, id="frame-no-token"),
        *(
            pytest.param({**_PARQUET, "layout": "frame", "columns": [token]}, True, id=token)
            for token in (
                "party_kind",
                "party_class",
                "party_whole",
                "financingRecipient_party_class",
                "financingProvider_party_whole",
            )
        ),
        pytest.param(
            {**_PARQUET, "layout": "frame", "columns": ["financingRecipient_iso3"]},
            False,
            id="graph-field-token",
        ),
        pytest.param({"format": "frame"}, False, id="canonical-frame"),
    ],
)
def test_a_rendering_carries_party_cells_when_its_layout_or_a_party_field_token_writes_them(
    rendering, carries
):
    assert carries_party_kind(rendering) is carries
