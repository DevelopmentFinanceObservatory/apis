"""Reads of a version after the code index rotated: the definition door's head hit and its
re-render from the stored frame, and the snapshot door.

Two clients over one store differ in one party's kind. The file with a party-kind column that
the first linked is not the one the second serves. Each read lists a rendering once and serves
the file rendered under its own client's digest, which carries that digest as its
`enrichment_stamp`. A file with no such column is the same under both and carries no stamp.
"""

from __future__ import annotations

from dataclasses import replace
from typing import override

import pytest

from artifact_store import LiveArtifactInventory, Tier
from code_index import CodeIndex
from datasets_api import current
from datasets_api.render import current_enrichment_locations
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord

from . import test_serve_party_totals as party_totals
from .conftest import FakeDatasetStore
from .test_definition_door import _client
from .test_enrichment_rotation import (
    _FRAME_OPTIONS,
    _of_rendering,
    _Rig,
    index_x,
    index_y,
)
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

# The fixtures imported from `test_enrichment_rotation` are used as test arguments.
# ruff: noqa: F401, F811

_KIND_QUERY = "format=parquet&bundle=false&layout=frame&columns=party_kind"
_NO_COLUMN_QUERY = "format=parquet&bundle=false&layout=frame"
_KIND_RENDERING = {
    "format": "parquet",
    "bundle": False,
    "columns": ["party_kind"],
    "layout": "frame",
}
_NO_COLUMN_RENDERING = {"format": "parquet", "bundle": False, "layout": "frame"}


def _get(client, spec_id: str, query: str = "") -> dict:
    resp = client.get(f"/api/datasets/{spec_id}?{query}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _snapshot(client, minted: dict) -> dict:
    resp = client.get(
        f"/api/datasets/{minted['definition']['spec_id']}/{minted['content_version']}"
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _file(manifest: dict, rendering: dict) -> dict:
    """The one artifact `manifest` lists for `rendering`."""
    (listed,) = [a for a in manifest["artifacts"] if a["rendering"] == rendering]
    return listed


class _Renders:
    """Counts the renders from the stored frame a head hit makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        render_and_store = current.render_and_store

        def counting(**kwargs):
            self.count += 1
            return render_and_store(**kwargs)

        monkeypatch.setattr(current, "render_and_store", counting)


def test_a_stored_rendering_re_render_after_a_rotation_serves_and_links_the_current_file(
    version_store, index_x: CodeIndex, index_y: CodeIndex
):
    rig = _Rig(version_store)
    minted = rig.mint(index_x, _FRAME_OPTIONS)
    spec_id = minted["definition"]["spec_id"]

    under_x = _get(rig.client(index_x), spec_id, _KIND_QUERY)
    under_y = _get(rig.client(index_y), spec_id, _KIND_QUERY)

    file_x = _file(under_x, _KIND_RENDERING)
    file_y = _file(under_y, _KIND_RENDERING)
    assert (file_x["enrichment_stamp"], file_y["enrichment_stamp"]) == (
        index_x.enrichment_digest,
        index_y.enrichment_digest,
    )
    assert file_x["artifact_digest"] != file_y["artifact_digest"]
    linked = _of_rendering(rig.locations(minted), _KIND_RENDERING)
    (current_link,) = current_enrichment_locations(
        linked, enrichment_digest=index_y.enrichment_digest
    )
    assert current_link.artifact_digest == file_y["artifact_digest"]


def test_a_head_hit_after_a_rotation_serves_the_linked_file_without_rendering(
    version_store, index_x: CodeIndex, index_y: CodeIndex, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig(version_store)
    minted = rig.mint(index_x, _FRAME_OPTIONS)
    spec_id = minted["definition"]["spec_id"]
    client_x, client_y = rig.client(index_x), rig.client(index_y)
    file_x = _file(_get(client_x, spec_id, _KIND_QUERY), _KIND_RENDERING)
    file_y = _file(_get(client_y, spec_id, _KIND_QUERY), _KIND_RENDERING)
    renders = _Renders(monkeypatch)

    again_y = _get(client_y, spec_id, _KIND_QUERY)
    again_x = _get(client_x, spec_id, _KIND_QUERY)

    assert renders.count == 0
    assert _file(again_y, _KIND_RENDERING) == file_y
    assert _file(again_x, _KIND_RENDERING) == file_x
    assert file_y["enrichment_stamp"] == index_y.enrichment_digest
    assert file_x["enrichment_stamp"] == index_x.enrichment_digest


def test_a_re_render_lists_the_file_a_concurrent_call_linked_under_the_current_digest(
    version_store, index_x: CodeIndex, index_y: CodeIndex
):
    """The response lists the file linked under the request's key, as a snapshot read does,
    even when another call linked a different one first."""
    rig = _Rig(version_store)
    minted = rig.mint(index_x, _FRAME_OPTIONS)
    rival = rig.artifact_store.put(b"the concurrent call's file", tier=Tier.working)

    class _RivalLinksFirst(LiveVersionRecordWriter):
        @override
        def record_artifact(self, *, artifact: ArtifactRecord, **kwargs) -> str:
            if artifact.enrichment_digest is not None:
                super().record_artifact(
                    artifact=replace(artifact, artifact_digest=rival.digest), **kwargs
                )
            return super().record_artifact(artifact=artifact, **kwargs)

    rig.version_writer = _RivalLinksFirst(store=version_store)

    under_y = _get(rig.client(index_y), minted["definition"]["spec_id"], _KIND_QUERY)

    assert _file(under_y, _KIND_RENDERING)["artifact_digest"] == rival.digest


def test_a_snapshot_read_after_a_rotation_lists_each_rendering_once_under_its_digest(
    version_store, index_x: CodeIndex, index_y: CodeIndex
):
    rig = _Rig(version_store)
    minted = rig.mint(index_x, _FRAME_OPTIONS)
    spec_id = minted["definition"]["spec_id"]
    file_x = _file(_get(rig.client(index_x), spec_id, _KIND_QUERY), _KIND_RENDERING)
    file_y = _file(_get(rig.client(index_y), spec_id, _KIND_QUERY), _KIND_RENDERING)

    snapshot_y = _snapshot(rig.client(index_y), minted)
    snapshot_x = _snapshot(rig.client(index_x), minted)

    for snapshot, listed, digest in (
        (snapshot_y, file_y, index_y.enrichment_digest),
        (snapshot_x, file_x, index_x.enrichment_digest),
    ):
        renderings = [a["rendering"] for a in snapshot["artifacts"]]
        assert all(renderings.count(rendering) == 1 for rendering in renderings)
        assert _file(snapshot, _KIND_RENDERING) == listed
        assert listed["enrichment_stamp"] == digest


def test_a_snapshot_read_states_no_stamp_for_a_file_rendered_under_another_digest(
    version_store, index_x: CodeIndex, index_y: CodeIndex
):
    """A party-kind file with no twin under the reading client's digest is listed, and
    nothing in the listing claims it holds that digest's kinds."""
    rig = _Rig(version_store)
    minted = rig.mint(index_x, _FRAME_OPTIONS)
    file_x = _file(
        _get(rig.client(index_x), minted["definition"]["spec_id"], _KIND_QUERY), _KIND_RENDERING
    )

    snapshot_y = _snapshot(rig.client(index_y), minted)

    listed = _file(snapshot_y, _KIND_RENDERING)
    assert listed["artifact_digest"] == file_x["artifact_digest"]
    assert "enrichment_stamp" not in listed
    assert "enrichment_stamp" not in snapshot_y


def test_a_file_with_no_party_kind_column_is_unchanged_by_a_rotation(
    version_store, index_x: CodeIndex, index_y: CodeIndex, monkeypatch: pytest.MonkeyPatch
):
    rig = _Rig(version_store)
    minted = rig.mint(index_x, _FRAME_OPTIONS)
    spec_id = minted["definition"]["spec_id"]
    client_y = rig.client(index_y)
    first = _file(minted, _NO_COLUMN_RENDERING)
    renders = _Renders(monkeypatch)

    under_y = _get(client_y, spec_id, _NO_COLUMN_QUERY)
    snapshot_y = _snapshot(client_y, minted)

    assert renders.count == 0
    assert _file(under_y, _NO_COLUMN_RENDERING) == first
    assert _file(snapshot_y, _NO_COLUMN_RENDERING) == first
    assert "enrichment_stamp" not in first
    assert len(_of_rendering(rig.locations(minted), _NO_COLUMN_RENDERING)) == 1


def _cut_rig(version_store):
    """Clients over one store, one per code index, for a flow dataset with an open `provider`
    side whose DAC total a cut leaves out, so the cut is a version of its own."""
    store = FakeDatasetStore()
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    head_writer = LiveHeadStore(store=version_store)
    index_x = party_totals._code_index()
    index_y = CodeIndex(
        stamp=index_x.stamp,
        declarations=index_x.declarations.values(),
        codes=index_x.codes,
        statvars=index_x.statvars,
        parties=[
            replace(party, kind="residual") if party.dcid == party_totals._KENYA else party
            for party in index_x.parties.values()
        ],
    )

    def client(code_index: CodeIndex):
        return _client(
            graph=party_totals._graph(),
            sdmx_transport=party_totals._sdmx_transport(),
            store=store,
            artifact_store=artifact_store,
            version_store=version_store,
            code_index=code_index,
            coverage=party_totals._coverage_service(),
            head_writer=head_writer,
        )

    return index_x, index_y, client, *party_totals._spec_id_and_sidecar()


def test_a_cut_version_read_after_a_rotation_serves_and_lists_the_current_file(version_store):
    index_x, index_y, client, spec_id, sidecar = _cut_rig(version_store)
    client_x, client_y = client(index_x), client(index_y)
    assert index_x.enrichment_digest != index_y.enrichment_digest
    party_totals._mint(client_x, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    query = _KIND_QUERY + "&without_party_totals=true"
    rendering = {**_KIND_RENDERING, "without_party_totals": True}

    under_x = _get(client_x, spec_id, query)
    under_y = _get(client_y, spec_id, query)

    assert under_x["trace"]["party_totals_left_out"]
    assert under_x["content_version"] == under_y["content_version"]
    file_x, file_y = _file(under_x, rendering), _file(under_y, rendering)
    assert file_x["artifact_digest"] != file_y["artifact_digest"]
    assert file_y["enrichment_stamp"] == index_y.enrichment_digest
    for client_of, listed in ((client_y, file_y), (client_x, file_x)):
        snapshot = client_of.get(f"/api/datasets/{spec_id}/{under_y['content_version']}").json()
        assert _file(snapshot, rendering)["artifact_digest"] == listed["artifact_digest"]
