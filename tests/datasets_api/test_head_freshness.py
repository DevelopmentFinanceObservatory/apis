"""Tests head freshness in `serve_current` by dependency fingerprint.

A head is current for `head_recheck_interval` with no read. Past it, one graph read of the
stored source refs' `lastDataRefreshDate` builds the current fingerprint and
`dependency_changes` compares it with the head's: a match confirms the head with no SDMX
request, and a mismatch rebuilds. A head older than `head_max_age` rebuilds regardless, and a
head stored with no fingerprint rebuilds once. A head whose sources are all DFO's records the
import the bound graph release copied (`Manifest.imported_data`), serves past `head_max_age`
while the bound release carries that import, whatever its builder or DC serving revision, up
to `head_imported_data_max_age_seconds`, and rebuilds once it carries another. Each serve's
`datasets.timings` labels name the `rebuild_reason` and count `sdmx_requests` and
`graph_requests`.

The materialiser, `serve_current` and the head store are real. The graph is
`FakeMintGraphReader`, SDMX a counting mock transport, and the store
`FakeVersionRecordDatasetStore`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Generator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from artifact_store import LiveArtifactInventory
from code_index.shared import get_index
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.current import DefinitionDeps, serve_current
from datasets_api.manifest_models import Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_graph import GraphClient
from dc_sdmx import SdmxClient
from dfo_platform.timings import PhaseTimings, collecting
from graph_generation import FakeGeneration
from graph_release.context import bound_release, current_release
from graph_release.format import PLATFORM
from graph_release.service import Release
from materialise import EnrichmentLookup, MaterialisationRequest
from materialise.content_version import recipe_dependency_fingerprint
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.window import RequestedWindow
from origin_catalogue import Catalogue
from origin_catalogue.origins import DFO
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader, StoredDefinition
from statvar_members import MemberEnumerator
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row

from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_materialisation_record import _registered
from .test_mint_statvar_set import _CSV_BODY, _FACET, _STAT_VAR, _coverage_service
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_PROVENANCE = "OECD_DAC2A"
_REFRESH_DATE = "2026-10-07"
_RECHECK = timedelta(seconds=900)
_MAX_AGE = timedelta(hours=24)
_START = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
_DFO_SOURCE = FakeCatalogue(provenances={_PROVENANCE: provenance_row(_PROVENANCE, DFO)})
_RECIPE_ID = next(iter(sorted(RECIPE_BODIES)))


class _Clock:
    def __init__(self) -> None:
        self.at = _START

    def __call__(self) -> datetime:
        return self.at


class _CountingSdmx(httpx.MockTransport):
    def __init__(self) -> None:
        self.requests = 0

        def _respond(_request: httpx.Request) -> httpx.Response:
            self.requests += 1
            return httpx.Response(
                200,
                text=_CSV_BODY,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )

        super().__init__(_respond)


class _Rig:
    """One definition served through `serve_current` against shared fakes and a clock."""

    def __init__(self, version_store: FakeVersionRecordDatasetStore, *, dated: bool) -> None:
        self.clock = _Clock()
        self.transport = _CountingSdmx()
        self.version_store = version_store
        self.arcs: dict[str, dict[str, tuple[str, ...]]] = {
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            _PROVENANCE: {"lastDataRefreshDate": (_REFRESH_DATE,)} if dated else {},
        }
        self.graph = FakeMintGraphReader(
            arcs=self.arcs,
            facets={(_STAT_VAR, "country/KEN"): [_FACET]},
            labels={_PROVENANCE: "OECD DAC2a", "country/KEN": "Kenya"},
        )
        store = FakeDatasetStore()
        self.deps = DefinitionDeps(
            registry=LiveSpecRegistry(store=store),
            spec_reader=LiveSpecReader(store=store),
            graph=self.graph,
            sdmx=SdmxClient(
                base_url="https://datacommons.one.org/core/api", transport=self.transport
            ),
            artifact_store=_InventoryRecordingArtifactStore(
                inventory=LiveArtifactInventory(store=version_store)
            ),
            materialiser_build="test-build",
            capability_version="dset_test",
            ceiling=1_000_000,
            member_concurrency=4,
            enumerator=MemberEnumerator(),
            enrichment=EnrichmentLookup(self.graph, parties=lambda: {}),
            code_index=get_index(),
            rate_cache=RateTableCache(),
            coverage=_coverage_service(),
            version_writer=LiveVersionRecordWriter(store=version_store),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            head_store=LiveHeadStore(store=version_store, now=self.clock),
            head_recheck_interval=_RECHECK,
            head_max_age=_MAX_AGE,
            now=self.clock,
        )
        self.stored: StoredDefinition
        self.stored, self.request = _registered(self.deps)

    def serve(
        self, request: MaterialisationRequest | None = None
    ) -> tuple[Manifest, dict[str, str]]:
        """Serve a GET of the definition, returning the manifest and the call's timing labels
        and counts."""
        timings = PhaseTimings()
        resolved = request if request is not None else self.request
        with collecting(timings):
            manifest = serve_current(
                self.stored,
                resolve_requests=lambda: [resolved],
                options=MintOptions(bundle=False),
                trigger="get",
                deps=self.deps,
                trace_id=None,
            )
        fields = dict(item.split("=", 1) for item in timings.as_log_field().split())
        return manifest, fields

    def graph_reads(self) -> int:
        return sum(self.graph.calls.values())

    def reset_counts(self) -> None:
        self.transport.requests = 0
        for method in self.graph.calls:
            self.graph.calls[method] = 0

    def set_head_dependencies(self, dependencies: Mapping[str, object] | None) -> None:
        with self.version_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.subject_head SET dependencies = %s WHERE subject_id = %s",
                (dependencies, self.stored.spec_id),
            )

    def head_dependencies(self) -> dict[str, object]:
        head = self.deps.head_store.read_head(self.stored.spec_id)
        assert head is not None and head.dependencies is not None
        return dict(head.dependencies)


@pytest.fixture
def version_store() -> Generator[FakeVersionRecordDatasetStore]:
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


def test_a_first_serve_rebuilds_with_reason_no_head(version_store):
    rig = _Rig(version_store, dated=True)

    manifest, fields = rig.serve()

    assert manifest.trace["served_from"] == "materialise"
    assert fields["rebuild_reason"] == "no_head"
    assert int(fields["sdmx_requests"]) == rig.transport.requests > 0


def test_a_head_inside_the_recheck_window_serves_with_no_read(version_store):
    rig = _Rig(version_store, dated=True)
    rig.serve()
    rig.reset_counts()
    rig.clock.at += _RECHECK - timedelta(seconds=1)

    manifest, fields = rig.serve()

    assert manifest.trace["served_from"] == "head"
    assert "rebuild_reason" not in fields
    assert fields["sdmx_requests"] == "0"
    assert fields["graph_requests"] == "0"
    assert rig.graph_reads() == 0
    assert rig.transport.requests == 0


def test_past_the_recheck_window_an_unchanged_fingerprint_costs_one_graph_read(version_store):
    rig = _Rig(version_store, dated=True)
    minted, _ = rig.serve()
    rig.reset_counts()
    rig.clock.at += _RECHECK + timedelta(seconds=1)

    manifest, fields = rig.serve()

    assert manifest.trace["served_from"] == "head"
    assert manifest.content_version == minted.content_version
    assert "rebuild_reason" not in fields
    assert rig.graph.calls["node_arcs_many"] == 1
    assert rig.graph_reads() == 1
    assert rig.transport.requests == 0
    assert fields["sdmx_requests"] == "0"
    head = rig.deps.head_store.read_head(rig.stored.spec_id)
    assert head is not None and head.confirmed_at == rig.clock.at


def _source_token_moved(rig: _Rig) -> MaterialisationRequest | None:
    rig.arcs[_PROVENANCE] = {"lastDataRefreshDate": ("2026-10-10",)}
    return None


def _member_digest_differs(rig: _Rig) -> MaterialisationRequest | None:
    rig.set_head_dependencies({**rig.head_dependencies(), "members": "sha256:other"})
    return None


def _recipe_digest_differs(rig: _Rig) -> MaterialisationRequest | None:
    rig.set_head_dependencies(
        {**rig.head_dependencies(), "recipes": [{"id": _RECIPE_ID, "digest": "sha256:stale"}]}
    )
    return None


def _window_end_moves(rig: _Rig) -> MaterialisationRequest | None:
    return replace(
        rig.request,
        window=RequestedWindow(kind="range", start_year=2015, end_year=2027, end_unstated=True),
    )


_OPEN_2026 = RequestedWindow(kind="range", start_year=2015, end_year=2026, end_unstated=True)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(_source_token_moved, f"sources:{_PROVENANCE}", id="source-token"),
        pytest.param(_member_digest_differs, "members", id="member-digest"),
        pytest.param(_recipe_digest_differs, f"recipes:{_RECIPE_ID}", id="recipe-digest"),
        pytest.param(_window_end_moves, "window", id="window-end"),
    ],
)
def test_a_changed_fingerprint_member_rebuilds_and_names_it(
    version_store, change: Callable[[_Rig], MaterialisationRequest | None], reason: str
):
    assert recipe_dependency_fingerprint(_RECIPE_ID) is not None
    rig = _Rig(version_store, dated=True)
    opened = replace(rig.request, window=_OPEN_2026)
    rig.serve(opened)
    rig.clock.at += _RECHECK + timedelta(seconds=1)
    request = change(rig) or opened
    rig.reset_counts()

    manifest, fields = rig.serve(request)

    assert manifest.trace["served_from"] == "materialise"
    assert fields["rebuild_reason"] == reason
    assert rig.transport.requests > 0
    assert fields["sdmx_requests"] == str(rig.transport.requests)


def test_a_source_with_no_refresh_date_never_rebuilds_on_its_token(version_store):
    rig = _Rig(version_store, dated=False)
    rig.serve()
    assert rig.head_dependencies()["sources"] == [{"ref": _PROVENANCE, "token": None}]

    for _ in range(3):
        rig.clock.at += _RECHECK + timedelta(seconds=1)
        rig.reset_counts()
        manifest, fields = rig.serve()

        assert manifest.trace["served_from"] == "head"
        assert "rebuild_reason" not in fields
        assert rig.transport.requests == 0


def _import_max_age(rig: _Rig) -> timedelta:
    return timedelta(seconds=rig.deps.settings.head_imported_data_max_age_seconds)


def _rebuilt_release(**changes: object) -> Release:
    """The bound release with its manifest changed by `changes`, served by no service."""
    release = current_release()
    manifest = dataclasses.replace(release.loaded.manifest, **changes)
    loaded = dataclasses.replace(release.loaded, manifest=manifest)
    return dataclasses.replace(release, loaded=loaded, service=None)


def _redeployed() -> Release:
    """A release of the same import from another builder and DC serving revision: every DFO
    part rebuilt, every platform part as it was."""
    bound = current_release()
    redeployed = _rebuilt_release(
        dc_serving_revision="dc-serving-next",
        builder_code_hash="b" * 64,
        declarations_hash="d" * 64,
        parts=tuple(
            ref if ref.layer == PLATFORM else dataclasses.replace(ref, sha256="f" * 64)
            for ref in bound.manifest.parts
        ),
    )
    assert any(ref.layer != PLATFORM for ref in bound.manifest.parts)
    assert redeployed.release_id != bound.release_id
    return redeployed


def _imported() -> Release:
    return _rebuilt_release(import_execution_id="import-next")


def _base_source(rig: _Rig) -> timedelta:
    """The fake catalogue lists no provenance, so the head is keyed on no import."""
    assert "imported_data" not in rig.head_dependencies()
    return _MAX_AGE


def _keyed_past_the_import_backstop(rig: _Rig) -> timedelta:
    assert rig.head_dependencies()["imported_data"] == current_release().manifest.imported_data
    return _import_max_age(rig)


def _written_before_import_keys(rig: _Rig) -> timedelta:
    dependencies = rig.head_dependencies()
    del dependencies["imported_data"]
    rig.set_head_dependencies(dependencies)
    return _MAX_AGE


@pytest.mark.parametrize(
    ("catalogue", "age_limit"),
    [
        pytest.param(None, _base_source, id="unlisted-source"),
        pytest.param(_DFO_SOURCE, _keyed_past_the_import_backstop, id="import-backstop"),
        pytest.param(_DFO_SOURCE, _written_before_import_keys, id="head-without-import"),
    ],
)
def test_a_head_older_than_the_maximum_age_rebuilds(
    version_store,
    bind_catalogue,
    catalogue: Catalogue | None,
    age_limit: Callable[[_Rig], timedelta],
):
    if catalogue is not None:
        bind_catalogue(catalogue)
    rig = _Rig(version_store, dated=True)
    rig.serve()
    rig.clock.at += age_limit(rig) + timedelta(seconds=1)
    rig.reset_counts()

    manifest, fields = rig.serve()

    assert manifest.trace["served_from"] == "materialise"
    assert fields["rebuild_reason"] == "max_age"
    assert rig.transport.requests > 0


@pytest.mark.parametrize(
    "release", [current_release, _redeployed], ids=["same-release", "redeployed-release"]
)
def test_a_head_keyed_on_the_bound_import_serves_past_the_maximum_age(
    version_store, bind_catalogue, release: Callable[[], Release]
):
    bind_catalogue(_DFO_SOURCE)
    rig = _Rig(version_store, dated=True)
    minted, _ = rig.serve()
    assert rig.head_dependencies()["imported_data"] == current_release().manifest.imported_data
    rig.clock.at += _import_max_age(rig) - timedelta(seconds=1)
    rig.reset_counts()

    with bound_release(release()):
        manifest, fields = rig.serve()

    assert manifest.trace["served_from"] == "head"
    assert manifest.content_version == minted.content_version
    assert "rebuild_reason" not in fields
    assert rig.transport.requests == 0


@pytest.mark.parametrize(
    "elapsed",
    [_RECHECK - timedelta(seconds=1), _MAX_AGE + timedelta(seconds=1)],
    ids=["inside-recheck-window", "past-maximum-age"],
)
def test_a_head_keyed_on_a_replaced_import_rebuilds_and_records_the_new_one(
    version_store, bind_catalogue, elapsed: timedelta
):
    bind_catalogue(_DFO_SOURCE)
    rig = _Rig(version_store, dated=True)
    rig.serve()
    rig.clock.at += elapsed
    rig.reset_counts()

    with bound_release(_imported()) as published:
        manifest, fields = rig.serve()

    assert manifest.trace["served_from"] == "materialise"
    assert fields["rebuild_reason"] == "imported_data"
    assert rig.transport.requests > 0
    assert published.manifest.imported_data != current_release().manifest.imported_data
    assert rig.head_dependencies()["imported_data"] == published.manifest.imported_data


def _drop_fingerprint(rig: _Rig) -> None:
    rig.set_head_dependencies(None)


def _name_a_run_of_another_version(rig: _Rig) -> None:
    """The head's run produced another version, as when a writer moved `content_version`
    without naming its run."""
    with rig.version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record "
            "(subject_id, subject_kind, content_version, schema_version, immutable_body, "
            "recorded_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (rig.stored.spec_id, "spec", "cv-other", 1, {}, _START),
        )
        tx.execute(
            "UPDATE datasets.materialisation SET content_version = %s WHERE id = "
            "(SELECT materialisation_id FROM datasets.subject_head WHERE subject_id = %s)",
            ("cv-other", rig.stored.spec_id),
        )


@pytest.mark.parametrize(
    "lose_fingerprint",
    [_drop_fingerprint, _name_a_run_of_another_version],
    ids=["no-dependencies", "run-of-another-version"],
)
def test_a_head_with_no_fingerprint_rebuilds_once(version_store, lose_fingerprint):
    rig = _Rig(version_store, dated=True)
    rig.serve()
    lose_fingerprint(rig)
    rig.clock.at += _RECHECK + timedelta(seconds=1)

    rebuilt, rebuilt_fields = rig.serve()
    rig.clock.at += _RECHECK + timedelta(seconds=1)
    rig.reset_counts()
    again, again_fields = rig.serve()

    assert rebuilt.trace["served_from"] == "materialise"
    assert rebuilt_fields["rebuild_reason"] == "schema"
    assert again.trace["served_from"] == "head"
    assert "rebuild_reason" not in again_fields
    assert rig.transport.requests == 0


def test_a_refresh_date_read_is_never_served_from_a_cache_under_one_generation():
    """A source's refresh date can move while the graph generation holds still, so the
    freshness read must reach the graph each time. `node_arcs_many` sends every read, even
    with the arcs cache on and the generation unchanged, and returns the date the graph
    holds now."""
    dates = iter(("2026-10-07", "2026-10-10"))
    posted: list[object] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        posted.append(request.content)
        date = next(dates)
        return httpx.Response(
            200,
            json={
                "data": {
                    _PROVENANCE: {"arcs": {"lastDataRefreshDate": {"nodes": [{"value": date}]}}}
                }
            },
        )

    client = GraphClient(
        base_url="http://fake/core/api",
        transport=httpx.MockTransport(_respond),
        generation=FakeGeneration(token="generation-1"),
        arcs_cache_size=10_000,
    )

    first = client.node_arcs_many([_PROVENANCE], properties=("lastDataRefreshDate",))
    second = client.node_arcs_many([_PROVENANCE], properties=("lastDataRefreshDate",))

    assert first[_PROVENANCE] == {"lastDataRefreshDate": ("2026-10-07",)}
    assert second[_PROVENANCE] == {"lastDataRefreshDate": ("2026-10-10",)}
    assert len(posted) == 2
