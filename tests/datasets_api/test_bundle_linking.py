"""A bundle render whose graph metadata reads fail is linked as incomplete, under a rendering
no request matches, so a later render reads again; a render of a version already on record fills its stored citation's empty
fields from a live read, within a floor budget of its own."""

from __future__ import annotations

import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from io import BytesIO

import pytest

from artifact_store import ArtifactNotFoundError, LiveArtifactInventory, Tier
from dataset_store.clock import utc_now
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import current, serve
from datasets_api import deps as deps_module
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.snapshot import rendering_servable
from datasets_api.mint_models import MintOptions
from datasets_api.render import (
    METADATA_INCOMPLETE,
    METADATA_KEY,
    artifact_rendering,
    artifact_rendering_key,
)
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import (
    ArtifactRecord,
    SubjectHead,
    VersionRecord,
    VersionWrite,
    WrittenVersion,
)
from dc_graph import GraphReadError
from materialise import EnrichmentLookup, provenance_doc
from materialise.enrichment import _PROPERTIES as _LABEL_PROPERTIES
from materialise.provenance_doc import SourceDocumentNotFound

from . import test_serve_groupby_collapse as groupby_fixture
from ._graph_doubles import FakeMintGraphReader
from .test_definition_door import _client as definition_door_client
from .test_download_links import _RenewingArtifactStore
from .test_remint_returns_persisted_version import (
    _FIRST_ATTRIBUTION,
    _PIN_PRINCIPAL,
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


class _UncachedEnrichment(EnrichmentLookup):
    """Reads every label from the graph. The surface's one lookup otherwise answers a head
    hit's label read from what the mint cached, and the read under test never reaches the
    graph."""

    def _store(self, resolved: Mapping[str, Mapping[str, str]]) -> None:
        pass


def _uncache_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps_module, "EnrichmentLookup", _UncachedEnrichment)


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
    monkeypatch.setattr(current, "render_and_store", marked)


class _ReadableArtifactStore(_InventoryRecordingArtifactStore):
    """Also serves `get`, which a head hit uses to read the stored frame."""

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


def _linked(client, manifest: dict) -> list[dict]:
    """The artifacts a snapshot lists besides the canonical frame, which every version links."""
    return [
        artifact
        for artifact in _snapshot(client, manifest)["artifacts"]
        if artifact["rendering"] != {"format": "frame"}
    ]


def _linked_digests(client, manifest: dict) -> list[str]:
    return [artifact["artifact_digest"] for artifact in _linked(client, manifest)]


def _bundle_digest(manifest: Mapping) -> str:
    return manifest["artifacts"][0]["artifact_digest"]


def _is_incomplete(artifact: Mapping) -> bool:
    return artifact["rendering"].get(METADATA_KEY) == METADATA_INCOMPLETE


def _download_status(client, manifest: Mapping) -> int:
    """The status the first artifact's download link answers."""
    return client.get(manifest["artifacts"][0]["url"], follow_redirects=False).status_code


def test_a_bundle_whose_documentation_read_failed_is_linked_incomplete_until_a_remint_heals_it(
    version_store,
):
    """The first mint stores its citation without documentation and links its bundle as
    incomplete: the download link resolves, the request's rendering is not servable, and the
    snapshot lists only the series statistics. The re-mint documents the bundle from a live
    read, links it beside the incomplete one, and leaves the stored citation as it was. The
    incomplete bundle is listed once pinned, after the complete one."""
    stat_vars = ["dcid:LinkingDocumentation"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    client, artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        graph=graph,
        pin_allowed_principals=frozenset({_PIN_PRINCIPAL}),
    )
    graph.fail_documentation = True

    first = client.post("/api/datasets", json=_bundled(stat_vars))
    listed_first = _linked_digests(client, first.json())
    download = _download_status(client, first.json())
    settings = DatasetsBuildSettings()
    servable = rendering_servable(
        subject_id=first.json()["definition"]["spec_id"],
        content_version=first.json()["content_version"],
        rendering_key=artifact_rendering_key(MintOptions(bundle=True), site=settings.site),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        now=utc_now,
        settings=settings,
    )
    graph.fail_documentation = False
    second = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    listed_second = _linked_digests(client, second)
    pin = client.post(
        f"/api/datasets/{second['definition']['spec_id']}/{second['content_version']}/pin",
        json={"schema_version": 1, "artifact_digest": _bundle_digest(first.json())},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )

    assert first.status_code == 200, first.text
    assert _FIRST_ATTRIBUTION not in _bundle_text(artifact_store, first.json())
    assert _is_incomplete(first.json()["artifacts"][0])
    assert download == 302
    assert not servable
    statistics = first.json()["artifacts"][1]["artifact_digest"]
    assert listed_first == [statistics]
    assert _FIRST_ATTRIBUTION in _bundle_text(artifact_store, second)
    assert not _is_incomplete(second["artifacts"][0])
    assert _bundle_digest(second) != _bundle_digest(first.json())
    assert listed_second == [_bundle_digest(second), statistics]
    assert second["citation"] == first.json()["citation"]
    assert "documentation" not in _snapshot(client, second)["citation"]["sources"][0]
    assert pin.status_code == 200, pin.text
    pinned = _linked(client, second)
    assert [artifact["artifact_digest"] for artifact in pinned] == [
        _bundle_digest(second),
        _bundle_digest(first.json()),
        statistics,
    ]
    assert _is_incomplete(pinned[1])


def test_a_bundle_whose_label_read_failed_is_linked_incomplete_until_a_read_answers(
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
    listed = _linked_digests(client, first.json())
    download = _download_status(client, first.json())
    graph.fail_labels = False
    second = client.post("/api/datasets", json=_bundled(stat_vars)).json()

    assert first.status_code == 200, first.text
    assert graph.label_budgets
    assert _is_incomplete(first.json()["artifacts"][0])
    assert download == 302
    assert listed == [first.json()["artifacts"][1]["artifact_digest"]]
    assert not _is_incomplete(second["artifacts"][0])
    assert _bundle_digest(second) != _bundle_digest(first.json())
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


@pytest.mark.parametrize("format", ["parquet", "xlsx"])
def test_bytes_linked_incomplete_and_complete_resolve_past_retention_and_are_listed_once(
    version_store, format: str
):
    """The graph documents nothing more for these sources once it answers, so the render
    after a failed documentation read gives the incomplete render's bytes and one digest is
    linked under both renderings. Past retention with its object gone, the link renders the
    complete rendering again and redirects. Pinned, the digest is listed once, without the
    marker."""
    stat_vars = [f"dcid:LinkingSharedDigest{format}"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    artifact_store = _RenewingArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client, _artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        graph=graph,
        artifact_store=artifact_store,
        head_writer=LiveHeadStore(store=version_store),
        pin_allowed_principals=frozenset({_PIN_PRINCIPAL}),
    )
    body = {**_bundled(stat_vars), "options": {"bundle": True, "format": format}}
    # The first mint cites the sources, so the next render reads only their documentation.
    cited = client.post("/api/datasets", json=_bundled(stat_vars))
    graph.fail_documentation = True
    incomplete = client.post("/api/datasets", json=body)
    graph.fail_documentation = False
    complete = client.post("/api/datasets", json=body).json()
    digest = _bundle_digest(complete)
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (utc_now() - timedelta(days=31),),
        )
    # The lifecycle rule deleted the bundle's object.
    del artifact_store._objects[(digest, Tier.working.value)]

    download = client.get(complete["artifacts"][0]["url"], follow_redirects=False)
    pin = client.post(
        f"/api/datasets/{complete['definition']['spec_id']}/{complete['content_version']}/pin",
        json={"schema_version": 1, "artifact_digest": digest},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )
    listed = [a for a in _snapshot(client, complete)["artifacts"] if a["artifact_digest"] == digest]

    assert cited.status_code == 200, cited.text
    assert incomplete.status_code == 200, incomplete.text
    assert _is_incomplete(incomplete.json()["artifacts"][0])
    assert not _is_incomplete(complete["artifacts"][0])
    assert _bundle_digest(incomplete.json()) == digest
    assert download.status_code == 302, download.text
    assert pin.status_code == 200, pin.text
    assert [artifact["rendering"] for artifact in listed] == [complete["artifacts"][0]["rendering"]]


@pytest.mark.parametrize(
    ("options", "marked"),
    [
        pytest.param({"format": "parquet", "bundle": False}, True, id="long-layout"),
        pytest.param(
            {"format": "parquet", "bundle": False, "layout": "frame"}, False, id="frame-layout"
        ),
    ],
)
def test_only_a_file_that_reads_names_is_linked_incomplete_when_the_label_read_fails(
    version_store, monkeypatch: pytest.MonkeyPatch, options: dict[str, object], marked: bool
):
    """A frame-layout file has no name columns and reads no names, so a failing label read
    leaves its rendering the options its request sent, which a client matches by equality."""
    stat_vars = [f"dcid:LinkingBareLabels{len(options)}"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    _mark_renders(monkeypatch, graph)
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs, graph=graph
    )
    graph.fail_labels = True

    minted = client.post("/api/datasets", json={**_bundled(stat_vars), "options": options})

    assert minted.status_code == 200, minted.text
    artifact = minted.json()["artifacts"][0]
    assert _is_incomplete(artifact) is marked
    if not marked:
        assert artifact["rendering"] == options
        assert graph.label_budgets == []


class _AgedHeadStore(LiveHeadStore):
    """A head store whose heads read as two days old, past the maximum age, so a re-mint
    rebuilds the version its head names."""

    def read_head(self, subject_id: str) -> SubjectHead | None:
        head = super().read_head(subject_id)
        if head is None:
            return None
        aged = head.materialised_at - timedelta(days=2)
        return replace(head, confirmed_at=aged, materialised_at=aged)


def test_a_remint_of_a_fully_cited_version_and_the_options_route_read_no_documentation(
    version_store,
):
    """A re-mint rebuilding the version its head names finds a stored citation with
    documentation and both licence fields set, which needs no live read, and the options route
    carries no citation."""
    stat_vars = ["dcid:LinkingNoRereads"]
    arcs = _arcs(stat_vars)
    arcs[_PROVENANCE_REF] = {**arcs[_PROVENANCE_REF], "license": ("https://example.org/terms",)}
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    client, _artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        graph=graph,
        head_writer=_AgedHeadStore(store=version_store),
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


def test_bundle_metadata_reads_end_at_the_request_deadline(version_store, monkeypatch):
    """With a one-second mint deadline, the code label reads, the first mint's documentation
    read and a re-mint's documentation read each end at that deadline, so none runs past the
    door's wait."""
    stat_vars = ["dcid:LinkingRequestDeadline"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    _mark_renders(monkeypatch, graph)
    client, _artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        graph=graph,
        settings=DatasetsBuildSettings(mint_timeout_seconds=1),
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
    heal_budgets = graph.documentation_budgets[len(first_mint_budgets) :]
    for budgets in (first_mint_budgets, heal_budgets, graph.label_budgets):
        assert budgets
        assert all(budget is not None and budget <= 1 for budget in budgets)


def test_a_remint_of_a_linked_bundle_reads_no_documentation_to_fill_its_citation(version_store):
    """The stored citation has no licence URL, but the version's bundle is linked and
    servable, so a re-mint rebuilding the version its head names lists that bundle and reads
    no documentation."""
    stat_vars = ["dcid:LinkingLinkedNoHeal"]
    arcs = _arcs(stat_vars)
    graph = _graph(stat_vars, arcs, _FlakyGraph)
    client, _artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        graph=graph,
        head_writer=_AgedHeadStore(store=version_store),
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


def test_a_head_hit_bundle_whose_label_read_failed_is_linked_incomplete_until_a_read_answers(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """A format the head has not stored renders from the stored frame on a head hit. A failed
    code label read links it as incomplete, downloadable and unlisted, and the next head hit
    renders again and links a complete one."""
    _uncache_labels(monkeypatch)
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
    listed = _linked_digests(client, minted)
    download = _download_status(client, first.json())
    graph.fail_labels = False
    second = client.get(path).json()

    assert first.status_code == 200, first.text
    assert first.json()["trace"]["served_from"] == "head"
    assert first.json()["content_version"] == minted["content_version"]
    assert _is_incomplete(first.json()["artifacts"][0])
    assert download == 302
    assert _bundle_digest(first.json()) not in listed
    assert _bundle_digest(minted) in listed
    assert second["trace"]["served_from"] == "head"
    assert not _is_incomplete(second["artifacts"][0])
    assert _bundle_digest(second) != _bundle_digest(first.json())
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

    def record_artifact(self, **kwargs) -> str:
        return self._delegate.record_artifact(**kwargs)

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        return self._delegate.write_version(write, trace_id=trace_id)


def test_a_head_hit_collapse_whose_label_read_failed_lists_the_bundle_a_concurrent_hit_linked(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """A head hit renders a groupby collapse whose code label read fails, so it creates the
    collapsed version with its bundle linked as incomplete. A concurrent head hit links its
    bundle under that version before this one responds, and the response lists that complete
    bundle, as a snapshot read does."""
    _uncache_labels(monkeypatch)
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


@pytest.mark.parametrize("head_hit", [False, True], ids=["mint", "head-hit"])
def test_a_groupby_collapse_whose_label_read_failed_is_linked_incomplete_under_its_own_version(
    version_store, monkeypatch: pytest.MonkeyPatch, head_hit: bool
):
    """A multi-contributor collapse has a content version of its own. A mint and a head hit
    each link its incomplete bundle under that version, so the download link resolves."""
    _uncache_labels(monkeypatch)
    graph = _FlakyGraph(
        arcs={
            groupby_fixture._MEMBER_A: {**groupby_fixture._MEMBER_ARCS, "testAxis": ("TestAxisA",)},
            groupby_fixture._MEMBER_B: {**groupby_fixture._MEMBER_ARCS, "testAxis": ("TestAxisB",)},
        },
        labels={"OECD_DAC2A": "OECD DAC2a", groupby_fixture._RECIPIENT: "Kenya"},
    )
    _mark_renders(monkeypatch, graph)
    client = definition_door_client(
        graph=graph,
        sdmx_transport=groupby_fixture._sdmx_transport(),
        version_store=version_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=LiveHeadStore(store=version_store),
        artifact_store=_ReadableArtifactStore(inventory=LiveArtifactInventory(store=version_store)),
        code_index=groupby_fixture._code_index(),
        coverage=groupby_fixture._coverage_service(),
    )
    spec_id, sidecar = groupby_fixture._spec_id_and_sidecar()
    base = groupby_fixture._mint(
        client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False}
    )
    graph.fail_labels = True

    if head_hit:
        collapsed = client.get(
            f"/api/datasets/{spec_id}?bundle=true&groupby.by=recipient&groupby.op=sum"
        ).json()
    else:
        collapsed = groupby_fixture._mint(
            client,
            spec_id=spec_id,
            sidecar=sidecar,
            options={"bundle": True, "groupby": {"by": ["recipient"], "op": "sum"}},
        )

    assert collapsed["content_version"] != base["content_version"]
    assert graph.label_budgets
    assert _is_incomplete(collapsed["artifacts"][0])
    assert _download_status(client, collapsed) == 302
