"""Pins `ReleaseService`: load at start, a request-triggered recheck, the swap
inside its guards, one release per request across a swap, file cleanup after
the last holder, the serving views built as a release opens, the Datasets
surface and worker wiring with and without `GRAPH_RELEASE_URI`, and the Graph,
Presentation and Search surfaces' binding under it."""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from artifact_publication.uri import write_uri_bytes
from code_index.shared import get_index
from datasets_api.admission import MaterialisePermits
from datasets_api.app import datasets_surface
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsOverrides, open_deps
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.oidc import FakeOidcVerifier
from datasets_api.operations.worker_app import RUN_ROUTE_PATH, build_worker_app
from graph_api.app import graph_surface
from graph_release import context
from graph_release.context import ReleaseNotBound, bound_release, current_release
from graph_release.format import Manifest, Pointer, manifest_uri, open_release, pointer_uri
from graph_release.service import (
    Release,
    ReleaseService,
    ReleaseUriMissingError,
    _LocalFiles,
    get_release_service,
    optional_release_service,
    reset_release_service,
)
from graph_release.views import ViewRefused
from materialise.rate_tables import RateTableCache
from origin_catalogue import current_catalogue
from origin_catalogue.service import CatalogueService
from presentation_api.app import presentation_surface
from search.engine.app import search_surface
from series_coverage.shared import get_coverage_service
from tests.datasets_api._artifact_doubles import FakeArtifactStore
from tests.datasets_api._operation_store import FakeOperationDatasetStore
from tests.datasets_api.test_worker_app import _complete_run, _submit, _task_body
from tests.graph_release.support import build_test_release, build_view_release, sample_tables
from tests.graph_release.test_format import _publish_file
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_RECHECK = 60.0


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _views_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most releases here hold no view parts, so the process service builds no views unless a
    test sets `GRAPH_RELEASE_COVERAGE` itself."""
    monkeypatch.delenv("GRAPH_RELEASE_COVERAGE", raising=False)


@pytest.fixture
def root(tmp_path: Path) -> str:
    return (tmp_path / "releases").as_uri()


@pytest.fixture
def local_dir(tmp_path: Path) -> Path:
    return tmp_path / "local"


def _other_tables() -> dict[str, pa.Table]:
    """`sample_tables` with different `nodes`, so the release shares every part but one."""
    tables = sample_tables()
    nodes = tables["nodes"]
    tables["nodes"] = nodes.set_column(
        nodes.schema.get_field_index("name"), "name", pa.array([f"Other {n}" for n in range(10)])
    )
    return tables


def _service(
    root: str, local_dir: Path, *, clock: _Clock, loader=None, **kwargs: object
) -> ReleaseService:
    if loader is not None:
        kwargs["loader"] = loader
    return ReleaseService(
        uri=root, local_dir=local_dir, recheck_seconds=_RECHECK, clock=clock, **kwargs
    )


def _recheck(service: ReleaseService, clock: _Clock) -> Release:
    """Moves the clock past the interval, runs the recheck it starts, and
    returns the release served after it."""
    clock.now += _RECHECK
    service.snapshot()
    service.close()
    return service.snapshot()


def _files(manifest: Manifest) -> set[str]:
    return {ref.file_name for ref in manifest.parts}


def _catalogue_service() -> CatalogueService:
    return CatalogueService(
        artifact_uri="fake://origin-catalogue",
        recheck_seconds=0,
        loader=lambda _uri: FakeCatalogue(build_id="fake-catalogue"),
    )


class TestService:
    def test_construction_serves_the_published_release_with_its_parts_mapped(
        self, root: str, local_dir: Path
    ):
        manifest = build_test_release(root, tables=sample_tables())

        release = _service(root, local_dir, clock=_Clock()).snapshot()

        assert release.release_id == manifest.release_id
        assert set(release.loaded.tables) == {"series", "facets", "edges", "nodes", "imports"}
        assert {path.name for path in local_dir.iterdir()} == _files(manifest)

    def test_construction_refuses_when_nothing_is_published(self, root: str, local_dir: Path):
        with pytest.raises(FileNotFoundError):
            _service(root, local_dir, clock=_Clock())

    def test_the_pointer_is_rechecked_only_once_the_interval_has_passed(
        self, root: str, local_dir: Path
    ):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()
        service = _service(root, local_dir, clock=clock)
        second = build_test_release(root, tables=_other_tables())

        clock.now += _RECHECK - 1
        service.snapshot()
        service.close()
        assert service.snapshot().release_id == first.release_id

        assert _recheck(service, clock).release_id == second.release_id

    def test_a_failed_recheck_keeps_the_served_release(self, root: str, local_dir: Path):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()
        calls: list[str | None] = []

        def loader(root_uri: str, *, local_dir: Path, release_id: str | None = None):
            calls.append(release_id)
            if len(calls) > 1:
                raise OSError("bucket unreachable")
            return open_release(root_uri, local_dir=local_dir, release_id=release_id)

        service = _service(root, local_dir, clock=clock, loader=loader)
        second = build_test_release(root, tables=_other_tables())

        assert _recheck(service, clock).release_id == first.release_id
        assert calls == [None, second.release_id]

    def test_a_swap_loads_then_flips_inside_every_registered_guard_then_calls_back(
        self, root: str, local_dir: Path
    ):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()
        events: list[str] = []

        def loader(root_uri: str, *, local_dir: Path, release_id: str | None = None):
            events.append("load")
            return open_release(root_uri, local_dir=local_dir, release_id=release_id)

        service = _service(root, local_dir, clock=clock, loader=loader)
        events.clear()

        @contextmanager
        def guard() -> Iterator[None]:
            events.append(f"enter serving {service._served.release.release_id[:6]}")
            yield
            events.append(f"exit serving {service._served.release.release_id[:6]}")

        unregister = service.register_swap_guard(guard)
        service.on_swap(lambda release: events.append(f"swapped to {release.release_id[:6]}"))
        second = build_test_release(root, tables=_other_tables())

        _recheck(service, clock)

        assert events == [
            "load",
            f"enter serving {first.release_id[:6]}",
            f"exit serving {second.release_id[:6]}",
            f"swapped to {second.release_id[:6]}",
        ]

        unregister()
        events.clear()
        third = build_test_release(root, tables=sample_tables(), dc_serving_revision="third")
        _recheck(service, clock)
        assert events == ["load", f"swapped to {third.release_id[:6]}"]

    def test_the_guard_is_free_while_the_new_release_loads(self, root: str, local_dir: Path):
        build_test_release(root, tables=sample_tables())
        clock = _Clock()
        guard_lock = threading.Lock()
        held_during_load: list[bool] = []

        def loader(root_uri: str, *, local_dir: Path, release_id: str | None = None):
            held_during_load.append(guard_lock.locked())
            return open_release(root_uri, local_dir=local_dir, release_id=release_id)

        service = _service(root, local_dir, clock=clock, loader=loader)
        service.register_swap_guard(lambda: guard_lock)
        second = build_test_release(root, tables=_other_tables())

        assert _recheck(service, clock).release_id == second.release_id
        assert held_during_load == [False, False]

    def test_a_request_reads_one_release_across_a_swap(self, root: str, local_dir: Path):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()
        service = _service(root, local_dir, clock=clock)
        second = build_test_release(root, tables=_other_tables())

        with bound_release(service.snapshot()):
            before = current_release()
            assert _recheck(service, clock).release_id == second.release_id
            after = current_release()

        assert before is after
        assert after.release_id == first.release_id

    @pytest.mark.unbound_release
    def test_unbound_current_release_raises(self):
        with pytest.raises(ReleaseNotBound):
            current_release()

    def test_a_replaced_release_unlinks_its_own_files_once_its_last_holder_drops_it(
        self, root: str, local_dir: Path
    ):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()
        service = _service(root, local_dir, clock=clock)
        second = build_test_release(root, tables=_other_tables())
        held = service.snapshot()

        _recheck(service, clock)
        on_disk = {path.name for path in local_dir.iterdir()}
        assert on_disk == _files(first) | _files(second)

        del held
        gc.collect()
        assert {path.name for path in local_dir.iterdir()} == _files(second)
        assert _files(first) - _files(second)

    def test_a_release_never_unlinks_the_file_of_a_part_its_loader_skipped(
        self, root: str, local_dir: Path
    ):
        first = build_test_release(root, tables=sample_tables())
        unknown = _publish_file(root, pa.table({"future_column": [7]}), name="future")
        with_unknown = dataclasses.replace(first, parts=(*first.parts, unknown))
        write_uri_bytes(manifest_uri(root, with_unknown.release_id), with_unknown.to_json())
        write_uri_bytes(pointer_uri(root), Pointer(with_unknown.release_id).to_json())
        local_dir.mkdir()
        (local_dir / unknown.file_name).write_bytes(b"another owner's file")
        clock = _Clock()
        service = _service(root, local_dir, clock=clock)
        assert unknown.file_name not in service.snapshot().loaded.file_names
        second = build_test_release(root, tables=_other_tables())

        _recheck(service, clock)
        gc.collect()

        on_disk = {path.name for path in local_dir.iterdir()}
        assert on_disk == _files(second) | {unknown.file_name}

    def test_a_failed_load_unlinks_what_it_downloaded(self, root: str, local_dir: Path):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()

        def failing_after_download(root_uri: str, *, local_dir: Path, release_id: str | None):
            loaded = open_release(root_uri, local_dir=local_dir, release_id=release_id)
            if release_id is not None:
                raise OSError("refused after the download")
            return loaded

        service = _service(root, local_dir, clock=clock, loader=failing_after_download)
        second = build_test_release(root, tables=_other_tables())

        assert _recheck(service, clock).release_id == first.release_id
        assert _files(second) - _files(first)
        assert {path.name for path in local_dir.iterdir()} == _files(first)

    def test_a_drop_during_a_load_unlinks_only_what_the_load_did_not_take_up(self, tmp_path: Path):
        files = _LocalFiles(tmp_path)
        for name in ("shared", "old"):
            (tmp_path / name).write_bytes(b"part")
        files.hold(["shared", "old"])

        with files.loading():
            files.drop(["shared", "old"])
            assert {path.name for path in tmp_path.iterdir()} == {"shared", "old"}
            files.hold(["shared"])

        assert {path.name for path in tmp_path.iterdir()} == {"shared"}


class TestViews:
    @pytest.mark.parametrize("build_views", [False, True])
    def test_views_are_built_as_a_release_opens_only_when_asked(
        self, root: str, local_dir: Path, build_views: bool
    ):
        manifest = build_view_release(root)

        release = _service(root, local_dir, clock=_Clock(), build_views=build_views).snapshot()

        if not build_views:
            assert release.views is None
            return
        assert release.views is not None
        assert release.views.code_index.stamp.coverage_build == manifest.release_id
        assert release.views.measure_index.axes
        assert release.views.coverage.num_rows > 0

    def test_construction_refuses_a_release_whose_views_do_not_build(
        self, root: str, local_dir: Path
    ):
        build_test_release(root, tables=sample_tables())

        with pytest.raises(ViewRefused):
            _service(root, local_dir, clock=_Clock(), build_views=True)
        assert list(local_dir.iterdir()) == []

    def test_a_recheck_keeps_the_served_release_when_the_new_views_do_not_build(
        self, root: str, local_dir: Path
    ):
        first = build_view_release(root)
        clock = _Clock()
        service = _service(root, local_dir, clock=clock, build_views=True)
        build_test_release(root, tables=sample_tables())

        served = _recheck(service, clock)

        assert served.release_id == first.release_id
        assert served.views is not None


class TestProcessService:
    @pytest.fixture(autouse=True)
    def _fresh_singleton(self) -> Iterator[None]:
        reset_release_service()
        yield
        reset_release_service()

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_unset_uri_builds_no_service(self, monkeypatch: pytest.MonkeyPatch, value: str | None):
        if value is None:
            monkeypatch.delenv("GRAPH_RELEASE_URI", raising=False)
        else:
            monkeypatch.setenv("GRAPH_RELEASE_URI", value)

        assert optional_release_service() is None
        assert context.optional_release_service() is None
        with pytest.raises(ReleaseUriMissingError):
            get_release_service()

    @pytest.mark.parametrize("flag", [None, "1"])
    def test_the_coverage_flag_builds_the_served_releases_views(
        self, root: str, monkeypatch: pytest.MonkeyPatch, flag: str | None
    ):
        build_view_release(root)
        monkeypatch.setenv("GRAPH_RELEASE_URI", root)
        if flag is None:
            monkeypatch.delenv("GRAPH_RELEASE_COVERAGE", raising=False)
        else:
            monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", flag)

        views = get_release_service().snapshot().views

        assert (views is not None) == (flag is not None)

    def test_a_set_uri_loads_its_release_once_per_process(
        self, root: str, monkeypatch: pytest.MonkeyPatch
    ):
        manifest = build_test_release(root, tables=sample_tables())
        monkeypatch.setenv("GRAPH_RELEASE_URI", root)

        service = optional_release_service()

        assert service is get_release_service()
        assert service is not None and service.snapshot().release_id == manifest.release_id
        assert context.optional_release_service() is service

    @pytest.mark.parametrize(
        "build_surface",
        [graph_surface, presentation_surface, search_surface],
        ids=["graph", "presentation", "search"],
    )
    def test_a_request_binds_the_served_release_beside_the_catalogue(
        self, root: str, monkeypatch: pytest.MonkeyPatch, build_surface
    ):
        manifest = build_test_release(root, tables=sample_tables())
        monkeypatch.setenv("GRAPH_RELEASE_URI", root)
        surface = build_surface()

        assert surface.request_scope is not None
        with surface.request_scope():
            assert current_release().release_id == manifest.release_id
            current_catalogue()


class TestDatasetsSurface:
    @pytest.mark.unbound_release
    def test_without_a_release_a_request_binds_the_catalogue_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("GRAPH_RELEASE_URI", raising=False)
        surface = datasets_surface(catalogue=_catalogue_service())

        assert surface.request_scope is not None
        with surface.request_scope():
            assert current_catalogue().build_id == "fake-catalogue"
            with pytest.raises(ReleaseNotBound):
                current_release()

    def test_a_request_binds_the_served_release_beside_the_catalogue(
        self, root: str, local_dir: Path
    ):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()
        service = _service(root, local_dir, clock=clock)
        surface = datasets_surface(catalogue=_catalogue_service(), release=service)
        second = build_test_release(root, tables=_other_tables())

        assert surface.request_scope is not None
        with surface.request_scope():
            assert current_catalogue().build_id == "fake-catalogue"
            assert current_release().release_id == first.release_id
            _recheck(service, clock)
            assert current_release().release_id == first.release_id
        with surface.request_scope():
            assert current_release().release_id == second.release_id

    def test_a_swap_waits_for_every_materialise_permit_while_the_deps_are_open(
        self, root: str, local_dir: Path
    ):
        first = build_test_release(root, tables=sample_tables())
        clock = _Clock()
        loaded = threading.Event()

        def loader(root_uri: str, *, local_dir: Path, release_id: str | None = None):
            opened = open_release(root_uri, local_dir=local_dir, release_id=release_id)
            loaded.set()
            return opened

        service = _service(root, local_dir, clock=clock, loader=loader)
        loaded.clear()
        second = build_test_release(root, tables=_other_tables())
        overrides = DepsOverrides(
            coverage=get_coverage_service(),
            rate_cache=RateTableCache(),
            code_index=get_index(),
            release=service,
            **_every_dependency_injected(),
        )

        async def swap_while_a_permit_is_held() -> None:
            async with open_deps(
                settings=DatasetsBuildSettings.from_env(), overrides=overrides, catalogue=None
            ) as deps:
                permits: MaterialisePermits = deps.materialise_semaphore
                assert permits.acquire(timeout=0)
                clock.now += _RECHECK
                service.snapshot()
                # The new release opens while the permit is held, and the flip waits for it.
                assert loaded.wait(5)
                await asyncio.sleep(0.2)
                assert service.snapshot().release_id == first.release_id
                permits.release()
                await asyncio.to_thread(service.close)

        asyncio.run(swap_while_a_permit_is_held())

        assert service.snapshot().release_id == second.release_id


def _every_dependency_injected() -> dict[str, object]:
    """Doubles for every dependency `open_deps` would build live, so it builds nothing."""
    return {
        name: object()
        for name in (
            "registry",
            "spec_reader",
            "snapshot_reader",
            "version_writer",
            "head_store",
            "version_lister",
            "operation_store",
            "recipe_store",
            "description_store",
            "reference_writer",
            "graph",
            "sdmx",
            "task_client",
        )
    } | {"artifact_store": FakeArtifactStore()}


class TestWorker:
    def _bound_release(self, release: ReleaseService | None) -> list[str]:
        operation_store = LiveOperationStore(store=FakeOperationDatasetStore())
        operation_id = _submit(operation_store)
        bound: list[str] = []

        def run(*, operation, deps, operation_store, **_) -> None:
            try:
                bound.append(current_release().release_id)
            except ReleaseNotBound:
                bound.append("unbound")
            _complete_run(operation=operation, deps=deps, operation_store=operation_store)

        app = build_worker_app(
            operation_store=operation_store,
            deps=object(),  # never read: `run` stands in for the runner
            oidc_verifier=FakeOidcVerifier(),
            worker_url="https://worker.example.org",
            worker_invoker_service_account="invoker@example.iam.gserviceaccount.com",
            run_in_thread=run,
            catalogue=_catalogue_service(),
            release=release,
        )
        response = TestClient(app).post(
            RUN_ROUTE_PATH.format(operation_id=operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t"},
        )
        assert response.status_code == 200
        return bound

    def test_an_operation_runs_on_the_release_served_when_it_is_delivered(
        self, root: str, local_dir: Path
    ):
        build_test_release(root, tables=sample_tables())
        clock = _Clock()
        service = _service(root, local_dir, clock=clock)
        second = build_test_release(root, tables=_other_tables())
        _recheck(service, clock)

        assert self._bound_release(service) == [second.release_id]

    @pytest.mark.unbound_release
    def test_without_a_release_an_operation_binds_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("GRAPH_RELEASE_URI", raising=False)

        assert self._bound_release(None) == ["unbound"]
