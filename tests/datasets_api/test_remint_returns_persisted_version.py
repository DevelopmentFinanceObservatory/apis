"""A re-mint after a source rename returns the stored citation because `content_version`
hashes the frame, not the citation.

The response, snapshot, and bundle use stored names through a shared
`LiveVersionRecordWriter` and `LiveSnapshotReader`.
"""

from __future__ import annotations

import itertools
import json
import zipfile
from datetime import UTC, datetime, timedelta
from io import BytesIO

import pytest

from artifact_store import ArtifactWriteError, LiveArtifactInventory, Tier
from dataset_store.snapshot import WORKING_TIER_RETENTION, LiveSnapshotReader
from datasets_api import current, serve
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest_models import WarningModel
from datasets_api.render import RENDITION_KEY
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import VersionRecord, VersionWrite, WrittenVersion
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dc_sdmx import SdmxClient
from description.citation import citation_line
from dfo_platform.config import PublicSiteSettings
from origin_catalogue.service import get_catalogue_service
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from . import test_serve_groupby_collapse as groupby_fixture
from ._app_factories import make_client
from ._graph_doubles import FakeMintGraphReader
from .conftest import FakeDatasetStore
from .test_definition_door import _client as definition_door_client
from .test_pin_door import FakeArtifactReferenceWriter
from .test_serve_bounds import (
    _FACET,
    _RECIPIENT,
    _coverage_service,
    _csv_body,
    _mint_body,
    _sdmx_transport,
)
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_PROVENANCE_REF = "OECD_DAC2A"
_DATASET = "dc/d/OECD_DevelopmentAssistanceCommittee"
_FIRST_NAME = "OECD Development Assistance Committee"
_RENAMED = "OECD DAC (renamed)"
_FIRST_ATTRIBUTION = "OECD (2026), DAC2a"
_REVISED_ATTRIBUTION = "OECD (2027), DAC2a, revised"

_FIRST_PREPARED = datetime(2026, 10, 4, 9, 0, tzinfo=UTC)


def _prepared(hours: int) -> str:
    """The wire `prepared_at` of the version first minted `hours` ticks of `clock` in."""
    return (_FIRST_PREPARED + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each mint's candidate `prepared_at` is an hour after the previous one's."""
    ticks = itertools.count()
    monkeypatch.setattr(
        serve, "_prepared_now", lambda: _FIRST_PREPARED + timedelta(hours=next(ticks))
    )


class _StaleCatalogue(FakeCatalogue):
    """A `FakeCatalogue` past `origin_catalogue.STALE_AFTER`."""

    def is_stale(self, *, now=None) -> bool:
        return True


_NAMED_NODES: dict[str, dict[str, tuple[str, ...]]] = {
    _RECIPIENT: {"name": ("Kenya",)},
    "GrantEquivalent": {"name": ("Grant equivalent",)},
}
"""Names for the download's default name columns, so a render adds no `enrichment_unavailable`
warning and a re-mint's response matches the first mint's. `_arcs` adds them to the arcs the
graph reads, which tests mutate in place."""


def _arcs(stat_vars: list[str]) -> dict[str, dict[str, tuple[str, ...]]]:
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        stat_var.removeprefix("dcid:"): {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
            "name": (stat_var,),
        }
        for stat_var in stat_vars
    }
    arcs[_PROVENANCE_REF] = {
        "typeOf": ("Provenance",),
        "isPartOf": (_DATASET,),
        "licenseType": ("CreativeCommonsAttributionLicense",),
        "licenseAttribution": (_FIRST_ATTRIBUTION,),
    }
    arcs[_DATASET] = {"typeOf": ("Dataset",), "name": (_FIRST_NAME,)}
    arcs.update(_NAMED_NODES)
    return arcs


def _attributions(manifest: dict) -> list[str | None]:
    return [source["documentation"]["attribution"] for source in manifest["citation"]["sources"]]


def _graph(stat_vars: list[str], arcs, graph_type=FakeMintGraphReader):
    return graph_type(
        arcs=arcs,
        facets={(stat_var.removeprefix("dcid:"), _RECIPIENT): [_FACET] for stat_var in stat_vars},
        labels={_RECIPIENT: "Kenya"},
    )


def _client(
    *,
    version_store,
    stat_vars: list[str],
    arcs,
    graph=None,
    artifact_store=None,
    version_writer=None,
    snapshot_reader=None,
    settings=None,
    head_writer=None,
    pin_allowed_principals=None,
):
    """A composed app over `version_store`. Clients sharing an `artifact_store` share one
    bucket, as every instance of one deployment does."""
    dataset_store = FakeDatasetStore()
    if artifact_store is None:
        artifact_store = _InventoryRecordingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        )
    if graph is None:
        graph = _graph(stat_vars, arcs)
    requests: list = []
    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=_sdmx_transport(
            {
                stat_var.removeprefix("dcid:"): _csv_body(stat_var=stat_var, num_rows=2)
                for stat_var in stat_vars
            },
            requests,
        ),
    )
    client = make_client(
        registry=LiveSpecRegistry(store=dataset_store),
        spec_reader=LiveSpecReader(store=dataset_store),
        graph=graph,
        sdmx=sdmx,
        artifact_store=artifact_store,
        version_writer=version_writer or LiveVersionRecordWriter(store=version_store),
        snapshot_reader=snapshot_reader or LiveSnapshotReader(store=version_store),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service([(stat_var, _RECIPIENT) for stat_var in stat_vars]),
        settings=settings,
        head_writer=head_writer,
        pin_allowed_principals=pin_allowed_principals,
    )
    return client, artifact_store


def _bundled(stat_vars: list[str]) -> dict:
    return {**_mint_body(statvars=stat_vars), "options": {"bundle": True}}


def _bundle_source_titles(artifact_store, manifest: dict) -> list[str]:
    artifact = manifest["artifacts"][0]  # The bundle is listed before series statistics.
    with zipfile.ZipFile(BytesIO(artifact_store.fetch(artifact["url"]))) as zf:
        descriptor = json.loads(zf.read("datapackage.json"))
    return [source["title"] for source in descriptor["sources"]]


def _snapshot(client, manifest: dict) -> dict:
    resp = client.get(
        f"/api/datasets/{manifest['definition']['spec_id']}/{manifest['content_version']}"
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# Every stored field a snapshot read serves unchanged. `artifacts` differs
# by contract (a mint lists the rendering it made, a snapshot every
# servable one), and `output_options` is derived per read.
_STORED_FIELDS = (
    "content_version",
    "prepared_at",
    "coverage",
    "citation",
    "trace",
    "warnings",
    "series_count",
)


# How one call was served, which `serve_current` stamps onto the stored trace.
_SERVING_STAMP = ("served_from", "confirmed_at")


def _stored_fields(manifest: dict) -> dict:
    fields = {field: manifest.get(field) for field in _STORED_FIELDS}
    fields["trace"] = {
        key: value for key, value in fields["trace"].items() if key not in _SERVING_STAMP
    }
    return fields


def _remint_after_a_rename(*, version_store, stat_vars: list[str]) -> tuple[dict, dict, object]:
    arcs = _arcs(stat_vars)
    client, artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)

    first = client.post("/api/datasets", json=_bundled(stat_vars))
    assert first.status_code == 200, first.text
    arcs[_DATASET] = {"typeOf": ("Dataset",), "name": (_RENAMED,)}
    second = client.post("/api/datasets", json=_bundled(stat_vars))
    assert second.status_code == 200, second.text

    first_manifest, second_manifest = first.json(), second.json()
    assert second_manifest["content_version"] == first_manifest["content_version"]
    return first_manifest, second_manifest, (client, artifact_store)


def test_a_definition_remint_after_a_rename_returns_the_stored_citation(version_store):
    first, second, (client, artifact_store) = _remint_after_a_rename(
        version_store=version_store, stat_vars=["dcid:RemintDefinition"]
    )

    assert [source["label"] for source in second["citation"]["sources"]] == [_FIRST_NAME]
    assert second["citation"] == first["citation"]
    assert _stored_fields(second) == _stored_fields(_snapshot(client, second))
    # The bundle is rendered from the stored citation: same bytes, same
    # digest as the one the first mint linked under this version.
    assert _bundle_source_titles(artifact_store, second) == [_FIRST_NAME]
    assert second["artifacts"][0]["artifact_digest"] == first["artifacts"][0]["artifact_digest"]


@pytest.mark.parametrize(
    "stat_vars",
    [["dcid:RemintOriginDefinition"]],
    ids=["definition"],
)
def test_a_remint_of_a_stored_version_keeps_the_warnings_on_record(
    version_store, monkeypatch: pytest.MonkeyPatch, stat_vars: list[str]
):
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=_arcs(stat_vars)
    )
    first = client.post("/api/datasets", json=_bundled(stat_vars))
    assert first.status_code == 200, first.text
    _title, detail, severity = DATASETS_WARNING_CATALOGUE["un_system_unavailable"]
    unavailable = WarningModel(
        code="un_system_unavailable", severity=severity, message=f"{detail} country/GRC."
    )
    monkeypatch.setattr(
        serve, "un_unavailable_warnings", lambda _blocked, *, request=None: (unavailable,)
    )

    second = client.post("/api/datasets", json=_bundled(stat_vars))

    assert second.status_code == 200, second.text
    assert second.json()["content_version"] == first.json()["content_version"]
    # A version's facts are fixed at its mint, so a re-mint serves the warnings on record.
    assert "un_system_unavailable" not in [w["code"] for w in first.json()["warnings"]]
    assert second.json()["warnings"] == first.json()["warnings"]


@pytest.mark.parametrize(
    "stat_vars",
    [["dcid:RemintStaleDefinition"]],
    ids=["definition"],
)
def test_a_remint_of_a_stored_version_carries_this_call_s_stale_catalogue_diagnostic(
    version_store, monkeypatch: pytest.MonkeyPatch, stat_vars: list[str]
):
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=_arcs(stat_vars)
    )
    first = client.post("/api/datasets", json=_bundled(stat_vars))
    assert first.status_code == 200, first.text
    monkeypatch.setattr(get_catalogue_service(), "snapshot", lambda: _StaleCatalogue())

    second = client.post("/api/datasets", json=_bundled(stat_vars))

    assert second.status_code == 200, second.text
    assert second.json()["content_version"] == first.json()["content_version"]
    assert "origin_catalogue_stale" not in [w["code"] for w in first.json()["diagnostics"]]
    assert "origin_catalogue_stale" in [w["code"] for w in second.json()["diagnostics"]]


def _stale_messages(manifest: dict) -> list[str]:
    return [w["message"] for w in manifest["diagnostics"] if w["code"] == "origin_catalogue_stale"]


@pytest.mark.parametrize(
    "stat_vars",
    [["dcid:RemintRestaleDefinition"]],
    ids=["definition"],
)
def test_a_remint_under_another_stale_build_carries_only_that_build_s_diagnostic(
    version_store, monkeypatch: pytest.MonkeyPatch, stat_vars: list[str]
):
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=_arcs(stat_vars)
    )
    service = get_catalogue_service()
    monkeypatch.setattr(service, "snapshot", lambda: _StaleCatalogue(build_id="build-a"))
    first = client.post("/api/datasets", json=_bundled(stat_vars))
    assert first.status_code == 200, first.text
    monkeypatch.setattr(service, "snapshot", lambda: _StaleCatalogue(build_id="build-b"))

    second = client.post("/api/datasets", json=_bundled(stat_vars))

    assert second.status_code == 200, second.text
    assert second.json()["content_version"] == first.json()["content_version"]
    assert ["'build-a'" in message for message in _stale_messages(first.json())] == [True]
    assert ["'build-b'" in message for message in _stale_messages(second.json())] == [True]


@pytest.mark.parametrize(
    "stat_vars",
    [["dcid:RemintFreshDefinition"]],
    ids=["definition"],
)
def test_a_remint_under_a_fresh_build_carries_no_stale_diagnostic(
    version_store, monkeypatch: pytest.MonkeyPatch, stat_vars: list[str]
):
    client, _artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=_arcs(stat_vars)
    )
    service = get_catalogue_service()
    monkeypatch.setattr(service, "snapshot", lambda: _StaleCatalogue(build_id="build-a"))
    first = client.post("/api/datasets", json=_bundled(stat_vars))
    assert first.status_code == 200, first.text
    monkeypatch.setattr(service, "snapshot", lambda: FakeCatalogue(build_id="build-b"))

    second = client.post("/api/datasets", json=_bundled(stat_vars))

    assert second.status_code == 200, second.text
    assert second.json()["content_version"] == first.json()["content_version"]
    assert "origin_catalogue_stale" in [w["code"] for w in first.json()["diagnostics"]]
    assert "origin_catalogue_stale" not in [w["code"] for w in second.json()["diagnostics"]]


@pytest.mark.parametrize(
    "stat_vars",
    [["dcid:RemintDocumentedDefinition"]],
    ids=["definition"],
)
def test_a_remint_after_a_documentation_change_keeps_the_first_mint_s_snapshot_and_date(
    version_store, clock, stat_vars: list[str]
):
    """The source documentation and `prepared_at` captured at first mint are what a re-mint
    and a snapshot read serve after the graph's documentation changes."""
    arcs = _arcs(stat_vars)
    client, _artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)
    first = client.post("/api/datasets", json=_bundled(stat_vars))
    assert first.status_code == 200, first.text
    arcs[_PROVENANCE_REF] = {
        **arcs[_PROVENANCE_REF],
        "licenseAttribution": (_REVISED_ATTRIBUTION,),
    }

    second = client.post("/api/datasets", json=_bundled(stat_vars)).json()

    assert _attributions(first.json()) == [_FIRST_ATTRIBUTION]
    assert first.json()["prepared_at"] == _prepared(0)
    assert _attributions(second) == [_FIRST_ATTRIBUTION]
    assert second["prepared_at"] == _prepared(0)
    assert _stored_fields(second) == _stored_fields(first.json())
    assert _stored_fields(second) == _stored_fields(_snapshot(client, second))
    assert (
        second["artifacts"][0]["artifact_digest"] == first.json()["artifacts"][0]["artifact_digest"]
    )


def test_a_first_mint_after_a_rename_cites_the_new_name(version_store):
    """Control for the two tests above: with no version on record, the
    write returns what it assembled, so the rename is visible and the
    stored-citation assertions above are not vacuous."""
    stat_vars = ["dcid:RemintControl"]
    arcs = _arcs(stat_vars)
    arcs[_DATASET] = {"typeOf": ("Dataset",), "name": (_RENAMED,)}
    arcs[_PROVENANCE_REF] = {
        **arcs[_PROVENANCE_REF],
        "licenseAttribution": (_REVISED_ATTRIBUTION,),
    }
    client, artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)

    resp = client.post("/api/datasets", json=_bundled(stat_vars))

    assert resp.status_code == 200, resp.text
    assert [source["label"] for source in resp.json()["citation"]["sources"]] == [_RENAMED]
    assert _attributions(resp.json()) == [_REVISED_ATTRIBUTION]
    assert _bundle_source_titles(artifact_store, resp.json()) == [_RENAMED]


def test_a_collapse_minted_over_a_renamed_base_cites_the_base_s_stored_citation(
    version_store, clock
):
    """A collapse after a source rename uses the base version's stored sources and
    `prepared_at` for its record, bundle, and repeat-mint response. Its line reads "processed
    by" the DFO, which summed the base's variables, where the base's reads "via"."""
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        **_NAMED_NODES,
        groupby_fixture._MEMBER_A: {
            **groupby_fixture._MEMBER_ARCS,
            "testAxis": ("TestAxisA",),
            "name": ("Member A",),
        },
        groupby_fixture._MEMBER_B: {
            **groupby_fixture._MEMBER_ARCS,
            "testAxis": ("TestAxisB",),
            "name": ("Member B",),
        },
        groupby_fixture._PROVENANCE: {"typeOf": ("Provenance",), "isPartOf": (_DATASET,)},
        _DATASET: {"typeOf": ("Dataset",), "name": (_FIRST_NAME,)},
    }
    graph = FakeMintGraphReader(arcs=arcs, labels={groupby_fixture._RECIPIENT: "Kenya"})
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    client = definition_door_client(
        graph=graph,
        sdmx_transport=groupby_fixture._sdmx_transport(),
        version_store=version_store,
        artifact_store=artifact_store,
        code_index=groupby_fixture._code_index(),
        coverage=groupby_fixture._coverage_service(),
    )
    spec_id, sidecar = groupby_fixture._spec_id_and_sidecar()
    groupby_options = {"bundle": True, "groupby": {"by": ["recipient"], "op": "sum"}}

    base = groupby_fixture._mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": True})
    arcs[_DATASET] = {"typeOf": ("Dataset",), "name": (_RENAMED,)}
    collapsed = groupby_fixture._mint(
        client, spec_id=spec_id, sidecar=sidecar, options=groupby_options
    )
    again = groupby_fixture._mint(client, spec_id=spec_id, sidecar=sidecar, options=groupby_options)

    assert collapsed["content_version"] != base["content_version"]
    base_citation = _snapshot(client, base)["citation"]
    assert [source["label"] for source in base_citation["sources"]] == [_FIRST_NAME]
    assert base_citation["text"] == citation_line(
        retrieved=_FIRST_PREPARED.date(), sources=[_FIRST_NAME], processed=False
    )
    collapsed_citation = {
        **base_citation,
        "text": citation_line(
            retrieved=_FIRST_PREPARED.date(), sources=[_FIRST_NAME], processed=True
        ),
    }
    assert collapsed["citation"] == collapsed_citation
    assert _snapshot(client, collapsed)["citation"] == collapsed_citation
    assert _bundle_source_titles(artifact_store, collapsed) == [_FIRST_NAME]
    assert again["content_version"] == collapsed["content_version"]
    assert _stored_fields(again) == _stored_fields(_snapshot(client, collapsed))
    assert base["prepared_at"] == _prepared(0)
    assert collapsed["prepared_at"] == _prepared(0)


def test_a_collapse_reminted_under_a_changed_catalogue_keeps_the_base_s_recorded_warnings(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """A collapse whose base version is on record under a head that rebuilds renders the
    derived version from a fresh materialisation and serves the warnings the base recorded,
    whatever origin warnings the call's catalogue finds."""
    client = definition_door_client(
        graph=groupby_fixture._graph(),
        sdmx_transport=groupby_fixture._sdmx_transport(),
        version_store=version_store,
        head_writer=LiveHeadStore(store=version_store),
        code_index=groupby_fixture._code_index(),
        coverage=groupby_fixture._coverage_service(),
    )
    spec_id, sidecar = groupby_fixture._spec_id_and_sidecar()
    _title, detail, severity = DATASETS_WARNING_CATALOGUE["un_system_unavailable"]

    def unavailable_at(place: str) -> None:
        warning = WarningModel(
            code="un_system_unavailable", severity=severity, message=f"{detail} {place}."
        )
        monkeypatch.setattr(
            serve, "un_unavailable_warnings", lambda _blocked, *, request=None: (warning,)
        )

    unavailable_at("country/GRC")
    base = groupby_fixture._mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": True})
    monkeypatch.setattr(get_catalogue_service(), "snapshot", lambda: FakeCatalogue(build_id="b"))
    unavailable_at("country/ITA")
    # A head past its maximum age rebuilds, and its version is on record.
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.subject_head SET confirmed_at = %s, materialised_at = %s "
            "WHERE subject_id = %s",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", spec_id),
        )

    collapsed = groupby_fixture._mint(
        client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": True, "groupby": {"by": ["recipient"], "op": "sum"}},
    )

    assert collapsed["content_version"] != base["content_version"]
    (recorded,) = _snapshot(client, base)["warnings"]
    assert recorded["message"].endswith("country/GRC.")
    assert [w for w in collapsed["warnings"] if w["code"] == "un_system_unavailable"] == [recorded]


class _RaceObservingWriter:
    """A `VersionRecordWriter` over `delegate` that keeps what each `write_version` stored.

    Clients here write no head, so a call never reads the stored version before rendering; a
    version a concurrent mint recorded first is found by the write transaction, which links
    only the option-free artifacts (`rendered` false), and the call renders again from the
    stored record.
    """

    def __init__(self, delegate: LiveVersionRecordWriter) -> None:
        self._delegate = delegate
        self.written: list[WrittenVersion] = []

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        return self._delegate.record_version(record, trace_id=trace_id)

    def record_artifact(self, **kwargs) -> None:
        self._delegate.record_artifact(**kwargs)

    def write_version(self, write: VersionWrite, *, trace_id: str | None = None) -> WrittenVersion:
        written = self._delegate.write_version(write, trace_id=trace_id)
        self.written.append(written)
        return written


# A write that found the winner's record, then the write of the render from it.
_LOST_THEN_RENDERED = [False, True]


def test_a_write_that_loses_a_concurrent_insert_returns_the_winner_s_record(version_store):
    """A concurrent first mint recorded the version before this call's write; the response
    uses its stored record.
    """
    stat_vars = ["dcid:RemintRace"]
    arcs = _arcs(stat_vars)
    winner, artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)
    won = winner.post("/api/datasets", json=_bundled(stat_vars))
    assert won.status_code == 200, won.text
    arcs[_DATASET] = {"typeOf": ("Dataset",), "name": (_RENAMED,)}

    writer = _RaceObservingWriter(LiveVersionRecordWriter(store=version_store))
    loser, _ = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        artifact_store=artifact_store,
        version_writer=writer,
    )

    lost = loser.post("/api/datasets", json=_bundled(stat_vars))

    assert lost.status_code == 200, lost.text
    assert [written.rendered for written in writer.written] == _LOST_THEN_RENDERED
    assert lost.json()["content_version"] == won.json()["content_version"]
    assert [source["label"] for source in lost.json()["citation"]["sources"]] == [_FIRST_NAME]
    assert _stored_fields(lost.json()) == _stored_fields(won.json())


def test_a_write_that_loses_a_concurrent_first_mint_returns_the_winner_s_bundle(
    version_store, clock
):
    """The losing call first rendered under its own `prepared_at`, so its bytes differ from the
    winner's; it renders again from the stored record and lists the bundle the winner linked,
    dated as the stored record is."""
    stat_vars = ["dcid:RemintRaceBundle"]
    arcs = _arcs(stat_vars)
    winner, artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)
    won = winner.post("/api/datasets", json=_bundled(stat_vars)).json()
    writer = _RaceObservingWriter(LiveVersionRecordWriter(store=version_store))
    loser, _ = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        artifact_store=artifact_store,
        version_writer=writer,
    )

    lost = loser.post("/api/datasets", json=_bundled(stat_vars))

    assert lost.status_code == 200, lost.text
    assert [written.rendered for written in writer.written] == _LOST_THEN_RENDERED
    assert lost.json()["prepared_at"] == won["prepared_at"] == _prepared(0)
    assert lost.json()["artifacts"][0]["artifact_digest"] == won["artifacts"][0]["artifact_digest"]
    assert _bundle_created(artifact_store, lost.json()) == _prepared(0)


def test_a_collapse_whose_base_a_concurrent_mint_wrote_first_takes_the_base_s_prepared_at(
    version_store, clock
):
    """The base is on record before this call's write, which renders the collapse under its own
    `prepared_at`. The write finds the base, records nothing rendered, and the call renders the
    collapse again from the stored base, so the collapsed record, its response, its linked
    bundle and its next mint all carry the base's stored `prepared_at`."""
    graph = groupby_fixture._graph()
    spec_store = FakeDatasetStore()
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )

    def client(version_writer=None):
        return definition_door_client(
            graph=graph,
            store=spec_store,
            sdmx_transport=groupby_fixture._sdmx_transport(),
            version_store=version_store,
            artifact_store=artifact_store,
            version_writer=version_writer,
            code_index=groupby_fixture._code_index(),
            coverage=groupby_fixture._coverage_service(),
        )

    spec_id, sidecar = groupby_fixture._spec_id_and_sidecar()
    groupby_options = {"bundle": True, "groupby": {"by": ["recipient"], "op": "sum"}}
    base = groupby_fixture._mint(
        client(), spec_id=spec_id, sidecar=sidecar, options={"bundle": False}
    )
    writer = _RaceObservingWriter(LiveVersionRecordWriter(store=version_store))

    collapsed = groupby_fixture._mint(
        client(writer), spec_id=spec_id, sidecar=sidecar, options=groupby_options
    )
    again = groupby_fixture._mint(
        client(), spec_id=spec_id, sidecar=sidecar, options=groupby_options
    )

    assert [written.rendered for written in writer.written] == _LOST_THEN_RENDERED
    assert collapsed["content_version"] != base["content_version"]
    assert base["prepared_at"] == _prepared(0)
    assert collapsed["prepared_at"] == _prepared(0)
    assert _bundle_created(artifact_store, collapsed) == _prepared(0)
    snapshot = _snapshot(client(), collapsed)
    assert snapshot["prepared_at"] == _prepared(0)
    assert _digests(_bundle_entries(snapshot)) == _digests(_bundle_entries(collapsed))
    assert again["content_version"] == collapsed["content_version"]
    assert _stored_fields(again) == _stored_fields(snapshot)
    assert _digests(_bundle_entries(again)) == _digests(_bundle_entries(collapsed))


@pytest.mark.parametrize(
    "stat_vars",
    [["dcid:RaceBundleDefinition"]],
    ids=["definition"],
)
def test_a_bundle_minted_by_the_loser_of_a_first_mint_is_dated_as_the_winner_s_record(
    version_store, clock, stat_vars: list[str]
):
    """The winner mints without a bundle, so it links no bundle. The loser's write finds the
    winner's record, and the loser renders its bundle again from it and links it. That
    bundle, the response and a snapshot read all carry the winner's stored `prepared_at`."""
    arcs = _arcs(stat_vars)
    winner, artifact_store = _client(version_store=version_store, stat_vars=stat_vars, arcs=arcs)
    won = winner.post("/api/datasets", json=_mint_body(statvars=stat_vars)).json()
    writer = _RaceObservingWriter(LiveVersionRecordWriter(store=version_store))
    loser, _ = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        artifact_store=artifact_store,
        version_writer=writer,
    )

    lost = loser.post("/api/datasets", json=_bundled(stat_vars)).json()
    snapshot = _snapshot(loser, lost)

    assert [written.rendered for written in writer.written] == _LOST_THEN_RENDERED
    assert lost["content_version"] == won["content_version"]
    assert won["prepared_at"] == lost["prepared_at"] == snapshot["prepared_at"] == _prepared(0)
    assert _bundle_created(artifact_store, lost) == _prepared(0)
    (linked,) = _bundle_entries(snapshot)
    assert _digests(_bundle_entries(lost)) == [linked["artifact_digest"]]


@pytest.mark.parametrize(
    ("stat_vars", "render_name", "subject_kind"),
    [
        (["dcid:FailedRenderDefinition"], "render_and_store", "spec"),
    ],
    ids=["definition"],
)
def test_a_rendering_that_fails_after_the_record_leaves_the_version_with_its_statistics(
    version_store, monkeypatch: pytest.MonkeyPatch, stat_vars, render_name, subject_kind
):
    """The record is written before the requested rendering, so a rendering that fails leaves a
    version on record. Its series statistics and canonical frame were linked first, so a
    snapshot read lists the statistics and the store holds the canonical frame."""
    client, _ = _client(version_store=version_store, stat_vars=stat_vars, arcs=_arcs(stat_vars))

    def failing_render(**_kwargs):
        raise ArtifactWriteError(object_name="bundle", digest="sha256:bundle")

    monkeypatch.setattr(serve, render_name, failing_render)

    failed = client.post("/api/datasets", json=_bundled(stat_vars))

    assert failed.status_code >= 500, failed.text
    with version_store.transaction() as tx:
        (row,) = tx.fetch_all(
            "SELECT subject_id, content_version FROM datasets.version_record "
            "WHERE subject_kind = %s",
            (subject_kind,),
        )
        renderings = [
            link["rendering"]
            for link in tx.fetch_all("SELECT rendering FROM datasets.version_artifact", ())
        ]
    snapshot = client.get(f"/api/datasets/{row['subject_id']}/{row['content_version']}")
    assert snapshot.status_code == 200, snapshot.text
    statistics = [
        artifact
        for artifact in snapshot.json()["artifacts"]
        if artifact["rendering"].get("statistics") == "series"
    ]
    assert len(statistics) == len(stat_vars)
    assert {"format": "frame"} in renderings


def _digests(artifacts: list[dict]) -> list[str]:
    return [artifact["artifact_digest"] for artifact in artifacts]


def _bundle_created(artifact_store, manifest: dict) -> str:
    with zipfile.ZipFile(BytesIO(artifact_store.fetch(manifest["artifacts"][0]["url"]))) as zf:
        return json.loads(zf.read("datapackage.json"))["created"]


def _bundle_codes(artifact_store, manifest: dict) -> str:
    with zipfile.ZipFile(BytesIO(artifact_store.fetch(manifest["artifacts"][0]["url"]))) as zf:
        return zf.read("data/codes.csv").decode("utf-8")


def test_a_remint_returns_the_linked_bundle_until_it_expires_then_its_own_render(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """A recipient renamed in the graph changes a fresh render's codes table. A re-mint lists
    the bundle linked at first mint while its unpinned working copy is servable, and its own
    render once that copy has aged out."""
    stat_vars = ["dcid:RemintLinkedBundle"]
    arcs = _arcs(stat_vars)
    arcs[_RECIPIENT] = {"typeOf": ("Country",), "name": ("Kenya",)}
    first_client, artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs
    )
    first = first_client.post("/api/datasets", json=_bundled(stat_vars)).json()
    arcs[_RECIPIENT] = {"typeOf": ("Country",), "name": ("Republic of Kenya",)}
    # A new client, so no label cache holds the old name.
    client, _ = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs, artifact_store=artifact_store
    )

    linked = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    after_expiry = datetime.now(UTC) + timedelta(days=31)
    monkeypatch.setattr(serve, "utc_now", lambda: after_expiry)
    expired = client.post("/api/datasets", json=_bundled(stat_vars)).json()

    assert linked["content_version"] == expired["content_version"] == first["content_version"]
    assert linked["artifacts"][0]["artifact_digest"] == first["artifacts"][0]["artifact_digest"]
    assert "Republic of Kenya" not in _bundle_codes(artifact_store, linked)
    assert expired["artifacts"][0]["artifact_digest"] != first["artifacts"][0]["artifact_digest"]
    assert "Republic of Kenya" in _bundle_codes(artifact_store, expired)


def _age_working_copy(version_store, artifact_digest: str) -> None:
    """Move the digest's working-tier copy past `WORKING_TIER_RETENTION`."""
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s "
            "WHERE artifact_digest = %s AND tier = 'working'",
            (datetime.now(UTC) - WORKING_TIER_RETENTION - timedelta(days=1), artifact_digest),
        )


_PIN_PRINCIPAL = "first-party-key"


def test_a_remint_after_the_linked_bundle_expires_links_its_render_for_snapshot_and_pin(
    version_store,
):
    """Once the linked bundle's unpinned working copy has expired, a re-mint whose render
    differs (a recipient renamed in the graph) replaces the link. The bundle it returns is the
    one a snapshot read lists and the pin door accepts, and the next re-mint returns it again."""
    stat_vars = ["dcid:RemintRelinkedBundle"]
    arcs = _arcs(stat_vars)
    arcs[_RECIPIENT] = {"typeOf": ("Country",), "name": ("Kenya",)}
    first_client, artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs
    )
    first = first_client.post("/api/datasets", json=_bundled(stat_vars)).json()
    (first_bundle,) = _bundle_entries(first)
    _age_working_copy(version_store, first_bundle["artifact_digest"])
    arcs[_RECIPIENT] = {"typeOf": ("Country",), "name": ("Republic of Kenya",)}
    client, _ = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        artifact_store=artifact_store,
        pin_allowed_principals=frozenset({_PIN_PRINCIPAL}),
    )

    relinked = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    again = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    (relinked_bundle,) = _bundle_entries(relinked)
    pin = client.post(
        f"/api/datasets/{relinked['definition']['spec_id']}/{relinked['content_version']}/pin",
        json={"schema_version": 1, "artifact_digest": relinked_bundle["artifact_digest"]},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )

    assert relinked["content_version"] == first["content_version"]
    assert relinked_bundle["artifact_digest"] != first_bundle["artifact_digest"]
    assert [
        bundle["artifact_digest"] for bundle in _bundle_entries(_snapshot(client, relinked))
    ] == [relinked_bundle["artifact_digest"]]
    assert [bundle["artifact_digest"] for bundle in _bundle_entries(again)] == [
        relinked_bundle["artifact_digest"]
    ]
    assert pin.status_code == 200, pin.text


def test_a_remint_keeps_a_pinned_bundle_linked_however_old_its_working_copy(version_store):
    """A pinned bundle stays servable, so a re-mint after its working copy expired and the
    graph changed still returns it and links nothing new."""
    stat_vars = ["dcid:RemintPinnedBundle"]
    arcs = _arcs(stat_vars)
    arcs[_RECIPIENT] = {"typeOf": ("Country",), "name": ("Kenya",)}
    client, artifact_store = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        pin_allowed_principals=frozenset({_PIN_PRINCIPAL}),
    )
    first = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    (first_bundle,) = _bundle_entries(first)
    pin = client.post(
        f"/api/datasets/{first['definition']['spec_id']}/{first['content_version']}/pin",
        json={"schema_version": 1, "artifact_digest": first_bundle["artifact_digest"]},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )
    assert pin.status_code == 200, pin.text
    _age_working_copy(version_store, first_bundle["artifact_digest"])
    arcs[_RECIPIENT] = {"typeOf": ("Country",), "name": ("Republic of Kenya",)}
    renamed, _ = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs, artifact_store=artifact_store
    )

    again = renamed.post("/api/datasets", json=_bundled(stat_vars)).json()

    assert [bundle["artifact_digest"] for bundle in _bundle_entries(again)] == [
        first_bundle["artifact_digest"]
    ]
    assert [bundle["artifact_digest"] for bundle in _bundle_entries(_snapshot(renamed, again))] == [
        first_bundle["artifact_digest"]
    ]


_MOVED_SITE = "https://moved.example"


def _bundle_entries(manifest: dict) -> list[dict]:
    return [artifact for artifact in manifest["artifacts"] if artifact["rendering"].get("bundle")]


def _mint_then_move_the_site(version_store):
    """Mint a bundle, then mint it again from a client whose public site URL differs, over
    the same store and bucket."""
    stat_vars = ["dcid:RenditionFlip"]
    arcs = _arcs(stat_vars)
    first_client, artifact_store = _client(
        version_store=version_store, stat_vars=stat_vars, arcs=arcs
    )
    first = first_client.post("/api/datasets", json=_bundled(stat_vars)).json()
    moved_client, _ = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=arcs,
        artifact_store=artifact_store,
        settings=DatasetsBuildSettings(site=PublicSiteSettings(public_site_url=_MOVED_SITE)),
    )
    moved = moved_client.post("/api/datasets", json=_bundled(stat_vars)).json()
    return first, moved, moved_client, artifact_store


def test_a_site_url_change_renders_and_links_a_new_bundle(version_store):
    """The rendition token moves with the public site URL, so the next mint misses the old
    link, renders a bundle printing the new URL, and links it under its own key."""
    first, moved, moved_client, artifact_store = _mint_then_move_the_site(version_store)
    (first_bundle,) = _bundle_entries(first)
    (moved_bundle,) = _bundle_entries(moved)

    assert moved["content_version"] == first["content_version"]
    assert moved_bundle["artifact_digest"] != first_bundle["artifact_digest"]
    assert moved_bundle["rendering"][RENDITION_KEY] != first_bundle["rendering"][RENDITION_KEY]
    with zipfile.ZipFile(BytesIO(artifact_store.fetch(moved_bundle["url"]))) as zf:
        assert _MOVED_SITE in zf.read("README.md").decode("utf-8")
    assert moved_bundle["artifact_digest"] in {
        bundle["artifact_digest"] for bundle in _bundle_entries(_snapshot(moved_client, moved))
    }


def test_a_bundle_linked_under_an_old_rendition_is_listed_only_once_pinned(version_store):
    """After a site URL change, a snapshot read lists only the bundle carrying the new token.
    Once the old bundle is pinned, the read lists it too, after the new one, and signs its
    bytes."""
    first, moved, moved_client, artifact_store = _mint_then_move_the_site(version_store)
    (first_bundle,) = _bundle_entries(first)
    (moved_bundle,) = _bundle_entries(moved)

    before_pin = _bundle_entries(_snapshot(moved_client, moved))
    pinned = artifact_store.copy(
        first_bundle["artifact_digest"], from_tier=Tier.working, to_tier=Tier.pinned
    )
    LiveArtifactInventory(store=version_store).record(
        digest=pinned.digest,
        tier=Tier.pinned,
        bucket=pinned.bucket,
        object_name=pinned.object_name,
        size_bytes=pinned.size_bytes,
        uploaded=True,
    )
    after_pin = _snapshot(moved_client, moved)["artifacts"]

    assert [bundle["artifact_digest"] for bundle in before_pin] == [moved_bundle["artifact_digest"]]
    assert [artifact["artifact_digest"] for artifact in after_pin[:2]] == [
        moved_bundle["artifact_digest"],
        first_bundle["artifact_digest"],
    ]
    assert after_pin[-1]["rendering"].get("statistics") == "series"
    old = after_pin[1]
    assert old["rendering"] == first_bundle["rendering"]
    assert old["tier"] == "pinned"
    assert artifact_store.fetch(old["url"]) == artifact_store.fetch(first_bundle["url"])


def test_a_head_hit_under_the_same_settings_serves_the_linked_bundle(
    version_store, monkeypatch: pytest.MonkeyPatch
):
    """The definition door keys a bundle request with the rendering the mint linked, so a GET
    after a mint under the same settings finds that link and renders nothing."""
    stat_vars = ["dcid:RenditionHeadHit"]
    client, _ = _client(
        version_store=version_store,
        stat_vars=stat_vars,
        arcs=_arcs(stat_vars),
        head_writer=LiveHeadStore(store=version_store),
    )
    minted = client.post("/api/datasets", json=_bundled(stat_vars)).json()
    (minted_bundle,) = _bundle_entries(minted)

    def _no_render(**_kwargs):
        raise AssertionError("a head hit on a linked bundle rendered it again")

    monkeypatch.setattr(current, "render_and_store", _no_render)
    got = client.get(f"/api/datasets/{minted['definition']['spec_id']}?bundle=true")

    assert got.status_code == 200, got.text
    assert got.json()["trace"]["served_from"] == "head"
    assert [bundle["artifact_digest"] for bundle in _bundle_entries(got.json())] == [
        minted_bundle["artifact_digest"]
    ]
