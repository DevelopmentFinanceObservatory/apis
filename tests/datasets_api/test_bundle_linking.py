"""A bundle render whose graph metadata reads fail is returned but never linked, so a later
render reads again; a render of a version already on record fills its stored citation's empty
fields from a live read, within a floor budget of its own."""

from __future__ import annotations

import time
import zipfile
from collections.abc import Mapping, Sequence
from io import BytesIO

import pytest

from artifact_store import ArtifactNotFoundError, LiveArtifactInventory, Tier
from datasets_api import serve
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import definition as definition_door
from datasets_api.mint_models import MintOptions
from datasets_api.render import artifact_rendering
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from dc_graph import GraphReadError
from materialise import provenance_doc
from materialise.enrichment import _PROPERTIES as _LABEL_PROPERTIES
from materialise.provenance_doc import SourceDocumentNotFound

from . import test_serve_groupby_collapse as groupby_fixture
from ._graph_doubles import FakeMintGraphReader
from .test_definition_door import _client as definition_door_client
from .test_remint_returns_persisted_version import (
    _FIRST_ATTRIBUTION,
    _PROVENANCE_REF,
    _REVISED_ATTRIBUTION,
    _arcs,
    _bundled,
    _client,
    _graph,
    _snapshot,
)
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore


class _FlakyGraph(FakeMintGraphReader):
    """Fails documentation reads (the Dataset's child read) or a bundle render's code label
    read while the matching flag is set, and records the seconds each of those reads had left.
    Materialisation reads labels too, for the output profile, so a label read counts as the
    bundle's only while `rendering` is set (`_mark_renders`)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_documentation = False
        self.fail_labels = False
        self.rendering = False
        self.documentation_budgets: list[float | None] = []
        self.label_budgets: list[float | None] = []

    def node_subjects(self, dcid: str, *, deadline: float | None = None, **kwargs):
        self.documentation_budgets.append(_budget(deadline))
        if self.fail_documentation:
            raise GraphReadError(f"transport error reading the children of {dcid!r}")
        return super().node_subjects(dcid, deadline=deadline, **kwargs)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        if self.rendering and tuple(properties) == _LABEL_PROPERTIES:
            self.label_budgets.append(_budget(deadline))
            if self.fail_labels:
                raise GraphReadError("transport error reading code labels")
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


def _mark_renders(monkeypatch: pytest.MonkeyPatch, graph: _FlakyGraph) -> None:
    """Sets `graph.rendering` for the length of each definition render, a mint's or a head
    hit's."""
    render = serve.render_and_store

    def marked(**kwargs):
        graph.rendering = True
        try:
            return render(**kwargs)
        finally:
            graph.rendering = False

    monkeypatch.setattr(serve, "render_and_store", marked)
    monkeypatch.setattr(definition_door, "render_and_store", marked)


class _ReadableArtifactStore(_InventoryRecordingArtifactStore):
    """Also serves `get`, which a head hit uses to read frame1."""

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        key = (artifact_digest, tier.value)
        if key not in self._objects:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        return self._objects[key]


def _budget(deadline: float | None) -> float | None:
    return None if deadline is None else deadline - time.monotonic()


def _bundle_text(artifact_store, manifest: Mapping) -> str:
    """Every document in the bundle: the Markdown files, `datapackage.json` and the CFF."""
    with zipfile.ZipFile(BytesIO(artifact_store.fetch(manifest["artifacts"][0]["url"]))) as zf:
        return "\n".join(
            zf.read(name).decode("utf-8")
            for name in zf.namelist()
            if name.endswith((".md", ".json", ".cff"))
        )


def _linked_digests(client, manifest: dict) -> list[str]:
    return [artifact["artifact_digest"] for artifact in _snapshot(client, manifest)["artifacts"]]


def _bundle_digest(manifest: Mapping) -> str:
    return manifest["artifacts"][0]["artifact_digest"]


def test_a_bundle_whose_documentation_read_failed_is_unlinked_until_a_remint_heals_it(
    version_store,
):
    """The first mint stores its citation without documentation and links only the series
    statistics. The re-mint documents the bundle from a live read, links it, and leaves the
    stored citation as it was."""
    stat_vars = ["dcid:LinkingDocumentation"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    client, artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs, graph=graph
    )
    graph.fail_documentation = True

    first = client.post("/api/datasets", json=_bundled(stat_vars))
    unlinked = _linked_digests(client, first.json())
    graph.fail_documentation = False
    second = client.post("/api/datasets", json=_bundled(stat_vars)).json()

    assert first.status_code == 200, first.text
    assert _FIRST_ATTRIBUTION not in _bundle_text(artifact_store, first.json())
    assert unlinked == [first.json()["artifacts"][1]["artifact_digest"]]
    assert _FIRST_ATTRIBUTION in _bundle_text(artifact_store, second)
    assert _bundle_digest(second) != _bundle_digest(first.json())
    assert _bundle_digest(second) in _linked_digests(client, second)
    assert second["citation"] == first.json()["citation"]
    assert "documentation" not in _snapshot(client, second)["citation"]["sources"][0]


def test_a_bundle_whose_label_read_failed_is_unlinked_until_a_read_answers(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    stat_vars = ["dcid:LinkingLabels"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    _mark_renders(monkeypatch, graph)
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs, graph=graph
    )
    graph.fail_labels = True

    first = client.post("/api/datasets", json=_bundled(stat_vars))
    unlinked = _linked_digests(client, first.json())
    graph.fail_labels = False
    second = client.post("/api/datasets", json=_bundled(stat_vars)).json()

    assert first.status_code == 200, first.text
    assert graph.label_budgets
    assert _bundle_digest(first.json()) not in unlinked
    assert unlinked == [first.json()["artifacts"][1]["artifact_digest"]]
    assert _bundle_digest(second) in _linked_digests(client, second)


def test_a_source_the_graph_documents_nothing_for_links_its_bundle(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    stat_vars = ["dcid:LinkingUndocumented"]
    arcs = _arcs(stat_vars)
    client, _artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)

    def undocumented(_graph, dcid: str, **_kwargs):
        raise SourceDocumentNotFound(dcid)

    monkeypatch.setattr(provenance_doc, "read_provenance_doc", undocumented)

    minted = client.post("/api/datasets", json=_bundled(stat_vars))

    assert minted.status_code == 200, minted.text
    assert _bundle_digest(minted.json()) in _linked_digests(client, minted.json())


def test_a_remint_of_a_fully_cited_version_and_the_options_route_read_no_documentation(
    version_store,
):
    """A stored citation with documentation and both licence fields set needs no live read,
    and the options route carries no citation."""
    stat_vars = ["dcid:LinkingNoRereads"]
    arcs = _arcs(stat_vars)
    arcs[_PROVENANCE_REF] = {**arcs[_PROVENANCE_REF], "license": ("https://example.org/terms",)}
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs, graph=graph
    )
    first = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    reads_after_first_mint = len(graph.documentation_budgets)

    second = client.post("/api/datasets", json=_bundled(stat_vars))
    options = client.get(f"/api/datasets/{first['definition']['spec_id']}/options")

    assert reads_after_first_mint == 1
    assert second.status_code == 200, second.text
    assert options.status_code == 200, options.text
    assert options.json()["trace"]["served_from"] != "head"
    assert len(graph.documentation_budgets) == reads_after_first_mint
    assert _bundle_digest(second.json()) == _bundle_digest(first)


def test_bundle_metadata_reads_keep_their_floor_past_the_request_deadline(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """With a one-second mint deadline and a 60-second floor, the code label reads, the first
    mint's documentation read and a re-mint's documentation read each start with far more
    than the deadline leaves."""
    stat_vars = ["dcid:LinkingFloor"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    _mark_renders(monkeypatch, graph)
    client, _artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        graph=graph,
        settings=DatasetsBuildSettings(mint_timeout_seconds=1, metadata_read_seconds=60),
    )
    first = client.post("/api/datasets", json=_bundled(stat_vars))
    first_mint_budgets = list(graph.documentation_budgets)

    # A format not yet linked, whose stored citation has no licence URL, reads its source live.
    second = client.post(
        "/api/datasets",
        json={**_bundled(stat_vars), "options": {"bundle": True, "format": "parquet"}},
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first_mint_budgets
    assert all(budget is not None and budget > 30 for budget in first_mint_budgets)
    heal_budgets = graph.documentation_budgets[len(first_mint_budgets) :]
    assert heal_budgets
    assert all(budget is not None and budget > 30 for budget in heal_budgets)
    assert graph.label_budgets
    assert all(budget is not None and budget > 30 for budget in graph.label_budgets)


@pytest.mark.parametrize(
    "stat_vars",
    [["dcid:LinkingLinkedNoHeal"], ["dcid:LinkingLinkedNoHealA", "dcid:LinkingLinkedNoHealB"]],
    ids=["definition", "collection"],
)
def test_a_remint_of_a_linked_bundle_reads_no_documentation_to_fill_its_citation(
    version_store, stat_vars: list[str]
):
    """The stored citation has no licence URL, but the version's bundle is linked and
    servable, so the re-mint lists that bundle and reads no documentation, for a definition
    and for a collection."""
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs, graph=graph
    )
    first = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    reads_after_first_mint = len(graph.documentation_budgets)

    second = client.post("/api/datasets", json=_bundled(stat_vars))

    assert "license_url" not in first["citation"]["sources"][0]
    assert reads_after_first_mint == 1
    assert second.status_code == 200, second.text
    assert len(graph.documentation_budgets) == reads_after_first_mint
    assert _bundle_digest(second.json()) == _bundle_digest(first)


def test_a_remint_fills_only_the_stored_citation_s_empty_fields_from_the_graph(version_store):
    """The stored citation has documentation but no licence URL. After the graph gains a
    licence URL and revises its attribution, a re-mint in a format not yet linked carries the
    new URL beside the stored attribution, and the stored citation stays as it was."""
    stat_vars = ["dcid:LinkingFillEmpty"]
    arcs = _arcs(stat_vars)
    client, artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)
    first = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    terms = "https://example.org/terms"
    arcs[_PROVENANCE_REF] = {
        **arcs[_PROVENANCE_REF],
        "license": (terms,),
        "licenseAttribution": (_REVISED_ATTRIBUTION,),
    }

    second = client.post(
        "/api/datasets",
        json={**_bundled(stat_vars), "options": {"bundle": True, "format": "parquet"}},
    ).json()

    assert terms not in _bundle_text(artifact_store, first)
    bundle = _bundle_text(artifact_store, second)
    assert terms in bundle
    assert _FIRST_ATTRIBUTION in bundle
    assert _REVISED_ATTRIBUTION not in bundle
    assert second["citation"] == first["citation"]
    assert "license_url" not in second["citation"]["sources"][0]


def test_a_head_hit_bundle_whose_label_read_failed_is_unlinked_until_a_read_answers(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """A format the head has not stored renders from frame1 on a head hit, and links only once
    its code label read answers."""
    stat_vars = ["dcid:LinkingHeadHit"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    _mark_renders(monkeypatch, graph)
    client, _artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        graph=graph,
        artifact_store=_ReadableArtifactStore(inventory=LiveArtifactInventory(store=version_store)),
        head_writer=LiveHeadStore(store=version_store),
    )
    minted = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    path = f"/api/datasets/{minted['definition']['spec_id']}?bundle=true&format=parquet"
    graph.fail_labels = True

    first = client.get(path)
    unlinked = _linked_digests(client, minted)
    graph.fail_labels = False
    second = client.get(path).json()

    assert first.status_code == 200, first.text
    assert first.json()["trace"]["served_from"] == "head"
    assert first.json()["content_version"] == minted["content_version"]
    assert _bundle_digest(first.json()) not in unlinked
    assert _bundle_digest(minted) in unlinked
    assert second["trace"]["served_from"] == "head"
    assert _bundle_digest(second) in _linked_digests(client, minted)


class _RivalLinksAfterTheCollapsedInsert:
    """A `VersionRecordWriter` over `delegate` that, once a collapsed version's insert has
    committed, links `rival` under it, as a concurrent head hit of that version would."""

    def __init__(self, delegate: LiveVersionRecordWriter, *, rival: ArtifactRecord) -> None:
        self._delegate = delegate
        self._rival = rival

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        created = self._delegate.record_version(record, trace_id=trace_id)
        if created and record.base_content_version is not None:
            self._delegate.record_artifact(
                subject_id=record.subject_id,
                content_version=record.content_version,
                artifact=self._rival,
            )
        return created

    def record_artifact(self, **kwargs) -> None:
        self._delegate.record_artifact(**kwargs)


def test_a_head_hit_collapse_whose_label_read_failed_lists_the_bundle_a_concurrent_hit_linked(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """A head hit renders a groupby collapse whose code label read fails, so it creates the
    collapsed version without linking its bundle. A concurrent head hit links its bundle under
    that version before this one responds, and the response lists that linked bundle, as a
    snapshot read does."""
    graph = _FlakyGraph(
        arcs={
            groupby_fixture._MEMBER_A: {**groupby_fixture._MEMBER_ARCS, "testAxis": ("TestAxisA",)},
            groupby_fixture._MEMBER_B: {**groupby_fixture._MEMBER_ARCS, "testAxis": ("TestAxisB",)},
        },
        labels={"OECD_DAC2A": "OECD DAC2a", groupby_fixture._RECIPIENT: "Kenya"},
    )
    _mark_renders(monkeypatch, graph)
    artifact_store = _ReadableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    groupby_options = {"bundle": True, "groupby": {"by": ["recipient"], "op": "sum"}}
    rival = artifact_store.put(b"the concurrent head hit's bundle", tier=Tier.working)
    client = definition_door_client(
        graph=graph,
        sdmx_transport=groupby_fixture._sdmx_transport(),
        version_store=version_store,
        version_writer=_RivalLinksAfterTheCollapsedInsert(
            LiveVersionRecordWriter(store=version_store),
            rival=ArtifactRecord(
                artifact_digest=rival.digest,
                rendering=artifact_rendering(
                    MintOptions.model_validate(groupby_options), site=DatasetsBuildSettings().site
                ),
                media_type="application/zip",
            ),
        ),
        head_writer=LiveHeadStore(store=version_store),
        artifact_store=artifact_store,
        code_index=groupby_fixture._code_index(),
        coverage=groupby_fixture._coverage_service(),
    )
    spec_id, sidecar = groupby_fixture._spec_id_and_sidecar()
    base = groupby_fixture._mint(
        client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False}
    )
    graph.fail_labels = True

    got = client.get(f"/api/datasets/{spec_id}?bundle=true&groupby.by=recipient&groupby.op=sum")

    assert got.status_code == 200, got.text
    collapsed = got.json()
    assert collapsed["trace"]["served_from"] == "head"
    assert collapsed["content_version"] != base["content_version"]
    assert graph.label_budgets
    bundles = [a["artifact_digest"] for a in collapsed["artifacts"] if a["rendering"].get("bundle")]
    assert bundles == [rival.digest]
    assert rival.digest in _linked_digests(client, collapsed)
