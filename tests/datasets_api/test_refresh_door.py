"""Frame refresh tests: resolution from stored records, each currency branch, the three answers,
the queued rebuild, the shared rebuild, and admission on `POST /api/datasets/frames/refresh`.
"""

from __future__ import annotations

import contextvars
import functools
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import datasets_api.current as current_module
import datasets_api.doors.refresh as refresh_module
import datasets_api.serve as serve_module
from artifact_store import LiveArtifactInventory, Tier
from code_index import CodeIndex, DatasetDeclaration
from code_index.shared import get_index
from dataset_store.snapshot import (
    WORKING_TIER_RETENTION,
    LiveSnapshotReader,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.mint_models import GroupbyOptions
from datasets_api.operation_store import LiveOperationStore, OperationBacklogFull
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker import run_operation
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.recorded_rendering import (
    MintingOptions,
    RenderingNotRecorded,
    StoredRendering,
    recorded_mintings,
)
from datasets_api.render import CANONICAL_FRAME_RENDERING, servable_frame_location
from datasets_api.version_store import (
    LiveVersionLister,
    LiveVersionRecordWriter,
    rendering_key,
)
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup
from materialise.rate_tables import RateTableCache
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from statvar_members import MemberEnumerator

from . import test_serve_groupby_collapse as collapse_fixture
from . import test_serve_party_totals as party_fixture
from . import test_version_file_selector as selector_fixture
from ._app_factories import make_client
from ._frame_doubles import link_frame1_only
from ._operation_store import FakeOperationDatasetStore
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_concurrency import _SETTINGS_ENV
from .test_definition_door import _CSV_HEADER, _RECIPIENT, _STAT_VAR, _STAT_VAR_REF, _mint_one_spec
from .test_definition_door_currency import (
    _FACET,
    _FACET_CHANGED,
    _PAST_INTERVAL,
    _PAST_MAX_AGE,
    _CoverageFromFacets,
    _FakeHeadStore,
    _mutable_graph,
    _RenderableArtifactStore,
    _row,
)
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_ROUTE = "/api/datasets/frames/refresh"
_LIVE_PRINCIPAL = "live-refresh-principal"
_PUBLISH_PRINCIPAL = "publish-principal"
_LIVE = {"x-dfo-principal": _LIVE_PRINCIPAL}
_PUBLISH = {"x-dfo-principal": _PUBLISH_PRINCIPAL}
_GROUPBY = {"by": ["recipient"], "op": "sum"}


@dataclass
class _Rig:
    client: TestClient
    head_store: _FakeHeadStore
    version_store: FakeVersionRecordDatasetStore
    artifact_store: _RenderableArtifactStore
    reference_writer: FakeArtifactReferenceWriter
    task_client: FakeTaskClient
    operation_store: LiveOperationStore
    spec_store: FakeDatasetStore
    sdmx_requests: list[httpx.Request]
    # The SDMX response body, replaced to move the data.
    body: dict[str, str]

    def refresh(self, *, spec_id: str, content_version: str) -> httpx.Response:
        return self.client.post(
            _ROUTE, json={"spec_id": spec_id, "content_version": content_version}, headers=_LIVE
        )

    def frame_servable(self, *, spec_id: str, content_version: str) -> bool:
        """Whether the version links a servable canonical frame, the one Presentation loads."""
        locations = LiveSnapshotReader(store=self.version_store).read_artifact_locations(
            spec_id, content_version
        )
        return servable_frame_location(locations, now=datetime.now(UTC)) is not None

    def expire_frame(self, *, spec_id: str, content_version: str) -> None:
        """Age the version's canonical frame working copy past `WORKING_TIER_RETENTION` and drop its
        object, as the bucket's lifecycle rule does."""
        with self.version_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT artifact_digest FROM datasets.version_artifact "
                "WHERE subject_id = %s AND content_version = %s AND rendering_key = %s",
                (spec_id, content_version, rendering_key(CANONICAL_FRAME_RENDERING)),
            )
            assert rows
            for row in rows:
                tx.execute(
                    "UPDATE datasets.artifact_location SET stored_at = %s "
                    "WHERE artifact_digest = %s AND tier = 'working'",
                    (
                        datetime.now(UTC) - WORKING_TIER_RETENTION - timedelta(days=1),
                        row["artifact_digest"],
                    ),
                )
                self.artifact_store._objects.pop((row["artifact_digest"], Tier.working.value), None)
        assert not self.frame_servable(spec_id=spec_id, content_version=content_version)

    def listed_versions(self, spec_id: str) -> dict[str, dict]:
        """The versions route's entries for the dataset, by content version."""
        resp = self.client.get(f"/api/datasets/{spec_id}/versions", headers=_PUBLISH)
        assert resp.status_code == 200, resp.text
        return {version["content_version"]: version for version in resp.json()["versions"]}


def _rig(
    *,
    graph,
    coverage,
    body: str,
    code_index=None,
    live_refresh_principal: str = _LIVE_PRINCIPAL,
    materialise_max_concurrency: int | None = None,
    work_max_concurrency: int | None = None,
    stores_of: _Rig | None = None,
    **settings: object,
) -> _Rig:
    """A composed Datasets app over one SQLite version store, with a head store tests can age
    and an SDMX body tests can replace. `stores_of` names a rig whose stores, head store and
    queue the app shares, so a test can serve them under another code index."""
    if stores_of is None:
        spec_store = FakeDatasetStore()
        version_store = FakeVersionRecordDatasetStore()
        artifact_store = _RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        )
        head_store = _FakeHeadStore()
        reference_writer = FakeArtifactReferenceWriter()
        task_client = FakeTaskClient()
        operation_store = LiveOperationStore(store=FakeOperationDatasetStore())
    else:
        spec_store = stores_of.spec_store
        version_store = stores_of.version_store
        artifact_store = stores_of.artifact_store
        head_store = stores_of.head_store
        reference_writer = stores_of.reference_writer
        task_client = stores_of.task_client
        operation_store = stores_of.operation_store
    body_ref = {"text": body}
    sdmx_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sdmx_requests.append(request)
        return httpx.Response(
            200,
            text=body_ref["text"],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = make_client(
        registry=LiveSpecRegistry(store=spec_store),
        spec_reader=LiveSpecReader(store=spec_store),
        graph=graph,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        head_writer=head_store,
        reference_writer=reference_writer,
        recipe_store=LiveRecipeStore(store=version_store),
        version_lister=LiveVersionLister(store=version_store),
        code_index=code_index,
        coverage=coverage,
        operation_store=operation_store,
        task_client=task_client,
        materialise_max_concurrency=materialise_max_concurrency,
        work_max_concurrency=work_max_concurrency,
        settings=DatasetsBuildSettings(
            live_refresh_principal=live_refresh_principal,
            first_party_principal=_PUBLISH_PRINCIPAL,
            **settings,
        ),
    )
    return _Rig(
        client=client,
        head_store=head_store,
        version_store=version_store,
        artifact_store=artifact_store,
        reference_writer=reference_writer,
        task_client=task_client,
        operation_store=operation_store,
        spec_store=spec_store,
        sdmx_requests=sdmx_requests,
        body=body_ref,
    )


@dataclass
class _PlainRig:
    rig: _Rig
    facets: dict
    arcs: dict
    graph: object
    coverage: _CoverageFromFacets
    spec_id: str
    content_version: str

    def move_data(self) -> None:
        """Move the source's refresh date, the facet and the rows a rebuild fetches."""
        self.arcs["OECD_DAC2A"] = {"lastDataRefreshDate": ("2026-02-01",)}
        self.facets[(_STAT_VAR, _RECIPIENT)] = [_FACET_CHANGED]
        self.rig.body["text"] = _CSV_HEADER + _row(
            stat_var_ref=_STAT_VAR_REF, obs_value="2000", period="2021"
        )


def _plain_rig(**kwargs) -> _PlainRig:
    """A rig holding one minted single-member dataset, its head primed by the mint."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "OECD_DAC2A": {"lastDataRefreshDate": ("2026-01-01",)}
    }
    graph = _mutable_graph(facets, arcs)
    coverage = _CoverageFromFacets(facets)
    rig = _rig(
        graph=graph,
        coverage=coverage,
        body=_CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000"),
        **kwargs,
    )
    spec_id = _mint_one_spec(rig.client)
    return _PlainRig(
        rig=rig,
        facets=facets,
        arcs=arcs,
        graph=graph,
        coverage=coverage,
        spec_id=spec_id,
        content_version=rig.head_store.head_for(spec_id).content_version,
    )


def _count_materialise_slots(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record each materialise slot `serve_current` takes, which a render or rebuild does."""
    taken: list[int] = []
    real = current_module.bounded_materialise_slot

    def counting(*args, **kwargs):
        taken.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(current_module, "bounded_materialise_slot", counting)
    return taken


# --- Plain frames: one branch of the freshness check per test ---


def test_a_head_inside_the_recheck_interval_answers_current_without_a_read(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    slots = _count_materialise_slots(monkeypatch)
    graph_calls = dict(plain.graph.calls)
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": plain.spec_id, "content_version": plain.content_version},
    }
    assert resp.headers["cache-control"] == "no-store"
    assert slots == []
    assert plain.graph.calls == graph_calls
    assert len(plain.rig.sdmx_requests) == sdmx_requests
    assert plain.rig.head_store.confirm_head_calls == 0


def test_an_unchanged_fingerprint_past_the_interval_confirms_the_head_and_answers_current(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_INTERVAL)
    slots = _count_materialise_slots(monkeypatch)
    arc_reads = plain.graph.calls["node_arcs_many"]
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "current"
    # One refresh-date read confirmed the head with no slot. Nothing was fetched.
    assert slots == []
    assert plain.graph.calls["node_arcs_many"] == arc_reads + 1
    assert plain.rig.head_store.confirm_head_calls == 1
    assert len(plain.rig.sdmx_requests) == sdmx_requests


def test_a_moved_source_date_rebuilds_and_answers_moved_then_and_on_later_calls():
    plain = _plain_rig()
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_INTERVAL)
    plain.move_data()
    sdmx_requests = len(plain.rig.sdmx_requests)

    rebuilt = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert rebuilt.status_code == 200, rebuilt.text
    answer = rebuilt.json()
    assert answer["status"] == "moved"
    moved_to = answer["frame"]["content_version"]
    assert answer["frame"]["spec_id"] == plain.spec_id
    assert moved_to != plain.content_version
    assert plain.rig.head_store.head_for(plain.spec_id).content_version == moved_to
    assert len(plain.rig.sdmx_requests) == sdmx_requests + 1
    # The rebuilt version holds the canonical frame Presentation reads.
    locations = LiveSnapshotReader(store=plain.rig.version_store).read_artifact_locations(
        plain.spec_id, moved_to
    )
    assert CANONICAL_FRAME_RENDERING in [location.rendering for location in locations]

    # The old frame is still moved on a call that rebuilds nothing.
    later = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
    assert later.json() == answer
    assert len(plain.rig.sdmx_requests) == sdmx_requests + 1

    # The frame the head now holds is current.
    current = plain.rig.refresh(spec_id=plain.spec_id, content_version=moved_to)
    assert current.json()["status"] == "current"

    # The refresh records no reference and pins nothing.
    assert plain.rig.reference_writer.records == {}
    assert all(tier != Tier.pinned.value for _digest, tier in plain.rig.artifact_store._objects)


def test_a_head_past_the_maximum_age_rebuilds_despite_an_unchanged_fingerprint():
    plain = _plain_rig()
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_MAX_AGE)
    materialised_at = plain.rig.head_store.head_for(plain.spec_id).materialised_at
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    # The same rows rebuild to the same version, so the named frame stays current.
    assert resp.json()["status"] == "current"
    assert len(plain.rig.sdmx_requests) == sdmx_requests + 1
    assert plain.rig.head_store.head_for(plain.spec_id).materialised_at > materialised_at


def test_a_head_whose_stored_window_differs_rebuilds_past_the_recheck_interval():
    plain = _plain_rig()
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_INTERVAL)
    head = plain.rig.head_store.head_for(plain.spec_id)
    # The head read an open window that ended in 2021, which a read now does not.
    plain.rig.head_store._heads[plain.spec_id] = replace(
        head,
        dependencies={**head.dependencies, "window": {"start_year": 2020, "end_year": 2021}},
    )
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "current"
    assert len(plain.rig.sdmx_requests) == sdmx_requests + 1


def test_a_head_stored_as_a_frame1_document_alone_answers_current_with_no_rebuild():
    plain = _plain_rig()
    link_frame1_only(
        plain.rig.version_store,
        plain.rig.artifact_store,
        subject_id=plain.spec_id,
        content_version=plain.content_version,
    )
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "current"
    # The frame1 document is the servable stored frame, so nothing was fetched.
    assert len(plain.rig.sdmx_requests) == sdmx_requests


def test_a_head_whose_canonical_frame_expired_rebuilds_before_it_answers_current():
    plain = _plain_rig()
    plain.rig.expire_frame(spec_id=plain.spec_id, content_version=plain.content_version)
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "current"
    # Nothing stored renders the head's frame, so the dataset was fetched again.
    assert len(plain.rig.sdmx_requests) == sdmx_requests + 1
    assert plain.rig.frame_servable(spec_id=plain.spec_id, content_version=plain.content_version)


# --- Resolution refusals ---


@pytest.mark.parametrize(
    ("frame", "status", "code"),
    [
        pytest.param(
            {"spec_id": "spec_unregistered", "content_version": "cv_any"},
            404,
            "spec_not_found",
            id="unknown-spec",
        ),
        pytest.param(
            {"spec_id": None, "content_version": "cv_unknown"},
            404,
            "snapshot_not_found",
            id="unknown-version",
        ),
        pytest.param(
            {"spec_id": "ops_0123456789abcdef", "content_version": "ocv_0123456789abcdef"},
            422,
            "frame_not_refreshable",
            id="derived-frame",
        ),
        pytest.param(
            {"spec_id": "not-an-id", "content_version": "cv_any"},
            400,
            "request_invalid",
            id="malformed-id",
        ),
    ],
)
def test_a_frame_the_records_cannot_resolve_refuses(frame: dict, status: int, code: str):
    plain = _plain_rig()
    body = {**frame, "spec_id": frame["spec_id"] or plain.spec_id}

    resp = plain.rig.client.post(_ROUTE, json=body, headers=_LIVE)

    assert resp.status_code == status, resp.text
    assert resp.json()["code"] == code


# --- Collapsed frames ---


@dataclass
class _CollapseRig:
    rig: _Rig
    spec_id: str
    base_content_version: str
    collapsed_content_version: str

    def move_data(self) -> None:
        """Change the rows a rebuild fetches and age the head past its maximum age, which
        forces the rebuild under this fixture's static coverage."""
        self.rig.body["text"] = collapse_fixture._CSV_BODY.replace(",1000,", ",3000,")
        self.rig.head_store.age_head(self.spec_id, age=_PAST_MAX_AGE)

    def version_rows(self) -> dict[str, str | None]:
        with self.rig.version_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT content_version, base_content_version FROM datasets.version_record "
                "WHERE subject_id = %s",
                (self.spec_id,),
            )
        return {row["content_version"]: row["base_content_version"] for row in rows}

    def recorded_groupby(self, content_version: str) -> list[object]:
        locations = LiveSnapshotReader(store=self.rig.version_store).read_artifact_locations(
            self.spec_id, content_version
        )
        return [
            location.rendering["groupby"]
            for location in locations
            if "groupby" in location.rendering
        ]


def _family_rig(*, body: str, code_index: CodeIndex | None = None) -> _Rig:
    """A rig over the two-member family dataset, serving `body` as its observations."""
    return _rig(
        graph=collapse_fixture._graph(),
        coverage=collapse_fixture._coverage_service(),
        body=body,
        code_index=code_index if code_index is not None else collapse_fixture._code_index(),
    )


def _collapse_rig() -> _CollapseRig:
    """A rig holding a two-member dataset minted with a groupby that sums both members, so the
    collapse has its own content version and the head holds the raw one."""
    rig = _family_rig(body=collapse_fixture._CSV_BODY)
    spec_id, sidecar = collapse_fixture._spec_id_and_sidecar()
    minted = collapse_fixture._mint(
        rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _GROUPBY},
    )
    base_content_version = rig.head_store.head_for(spec_id).content_version
    assert minted["content_version"] != base_content_version
    return _CollapseRig(
        rig=rig,
        spec_id=spec_id,
        base_content_version=base_content_version,
        collapsed_content_version=minted["content_version"],
    )


def test_a_collapsed_frame_of_the_head_is_current_and_renders_nothing():
    collapsed = _collapse_rig()
    puts = collapsed.rig.artifact_store.put_calls

    resp = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {
            "spec_id": collapsed.spec_id,
            "content_version": collapsed.collapsed_content_version,
        },
    }
    assert collapsed.rig.artifact_store.put_calls == puts


def test_a_collapse_another_caller_recorded_from_the_head_is_answered_without_a_render(
    monkeypatch: pytest.MonkeyPatch,
):
    """A frame recorded under one groupby answers `moved` to the collapse another caller's
    mint recorded from the current head, read from the stored versions with no render."""
    collapsed = _collapse_rig()
    collapsed.rig.body["text"] = (
        collapse_fixture._CSV_HEADER + _MEMBER_A_ROW.replace(",1000,", ",3000,") + _MEMBER_B_ROW
    )
    collapsed.rig.head_store.age_head(collapsed.spec_id, age=_PAST_MAX_AGE)
    spec_id, sidecar = collapse_fixture._spec_id_and_sidecar()
    recorded = collapse_fixture._mint(
        collapsed.rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _GROUPBY},
    )["content_version"]
    assert recorded != collapsed.collapsed_content_version
    real = refresh_module.head_rendering
    renders: list[str] = []

    def counting(**kwargs):
        renders.append(kwargs["content_version"])
        return real(**kwargs)

    monkeypatch.setattr(refresh_module, "head_rendering", counting)

    resp = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "moved",
        "frame": {"spec_id": collapsed.spec_id, "content_version": recorded},
    }
    assert renders == []


def test_a_rebuild_moves_a_collapsed_frame_to_the_collapse_of_the_new_head():
    collapsed = _collapse_rig()
    collapsed.move_data()

    rebuilt = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert rebuilt.status_code == 200, rebuilt.text
    answer = rebuilt.json()
    assert answer["status"] == "moved"
    moved_to = answer["frame"]["content_version"]
    new_head = collapsed.rig.head_store.head_for(collapsed.spec_id).content_version
    assert new_head != collapsed.base_content_version
    # The answer is the collapse of the new head under the frame's own groupby.
    assert collapsed.version_rows()[moved_to] == new_head
    assert collapsed.recorded_groupby(moved_to) == [_GROUPBY]

    # A later call finds that collapse on record and renders nothing.
    puts = collapsed.rig.artifact_store.put_calls
    sdmx_requests = len(collapsed.rig.sdmx_requests)
    later = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )
    assert later.json() == answer
    assert collapsed.rig.artifact_store.put_calls == puts
    assert len(collapsed.rig.sdmx_requests) == sdmx_requests


def test_a_head_moved_by_another_rendering_mints_the_collapse_from_its_stored_frame():
    collapsed = _collapse_rig()
    collapsed.move_data()
    # A plain read rebuilds the dataset, so the new head holds no collapse yet.
    plain = collapsed.rig.client.get(f"/api/datasets/{collapsed.spec_id}?bundle=false")
    assert plain.status_code == 200, plain.text
    new_head = collapsed.rig.head_store.head_for(collapsed.spec_id).content_version
    assert new_head != collapsed.base_content_version
    assert new_head not in collapsed.version_rows().values()
    sdmx_requests = len(collapsed.rig.sdmx_requests)

    resp = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["status"] == "moved"
    moved_to = answer["frame"]["content_version"]
    assert collapsed.version_rows()[moved_to] == new_head
    assert collapsed.recorded_groupby(moved_to) == [_GROUPBY]
    # The collapse came from the head's stored frame, with no fetch.
    assert len(collapsed.rig.sdmx_requests) == sdmx_requests


def test_a_recorded_collapse_whose_canonical_frame_expired_is_rendered_again_from_the_head():
    collapsed = _collapse_rig()
    collapsed.move_data()
    answer = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    ).json()
    assert answer["status"] == "moved"
    moved_to = answer["frame"]["content_version"]
    collapsed.rig.expire_frame(spec_id=collapsed.spec_id, content_version=moved_to)
    sdmx_requests = len(collapsed.rig.sdmx_requests)

    resp = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == answer
    # The collapse came again from the head's stored frame, with no fetch.
    assert collapsed.rig.frame_servable(spec_id=collapsed.spec_id, content_version=moved_to)
    assert len(collapsed.rig.sdmx_requests) == sdmx_requests


def test_a_collapsed_frame_with_no_file_recording_its_groupby_refuses():
    collapsed = _collapse_rig()
    # The versions route reads the groupby the refresh resolves.
    # Both frames hold a party role, so each lists the serving index's digest.
    digest = collapse_fixture._code_index().enrichment_digest
    assert collapsed.rig.listed_versions(collapsed.spec_id) == {
        collapsed.collapsed_content_version: {
            "content_version": collapsed.collapsed_content_version,
            "base_content_version": collapsed.base_content_version,
            "rendering": "groupby",
            "groupby": _GROUPBY,
            "enrichment_stamp": digest,
        },
        collapsed.base_content_version: {
            "content_version": collapsed.base_content_version,
            "base_content_version": None,
            "rendering": "plain",
            "groupby": None,
            "enrichment_stamp": digest,
        },
    }
    with collapsed.rig.version_store.transaction() as tx:
        # A link's head rows go first, as the foreign key on them requires.
        for table in ("version_artifact_head", "version_artifact"):
            tx.execute(
                f"DELETE FROM datasets.{table} WHERE subject_id = %s "
                "AND content_version = %s AND rendering_key != %s",
                (
                    collapsed.spec_id,
                    collapsed.collapsed_content_version,
                    rendering_key(CANONICAL_FRAME_RENDERING),
                ),
            )

    resp = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "rendering_not_recorded"
    listed = collapsed.rig.listed_versions(collapsed.spec_id)[collapsed.collapsed_content_version]
    assert (listed["rendering"], listed["groupby"]) == ("not_recorded", None)


# --- Frames minted without their party totals ---


_CUT_FRAME = {"format": "parquet", "bundle": False, "layout": "frame", "without_party_totals": True}


def test_a_rebuild_moves_a_cut_frame_to_the_cut_of_the_new_head():
    """A frame minted without its party totals refreshes to the cut of the current data, which
    the refresh mints with the option again and records under the new head."""
    rig = _rig(
        graph=party_fixture._graph(),
        coverage=party_fixture._coverage_service(),
        body=party_fixture._CSV_BODY,
        code_index=party_fixture._code_index(),
    )
    spec_id, sidecar = party_fixture._spec_id_and_sidecar()
    minted = party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options=_CUT_FRAME)
    base = rig.head_store.head_for(spec_id).content_version
    assert minted["content_version"] != base

    current = rig.refresh(spec_id=spec_id, content_version=minted["content_version"])
    assert current.json()["status"] == "current", current.text

    rig.body["text"] = party_fixture._CSV_BODY.replace(",100,", ",300,")
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    moved = rig.refresh(spec_id=spec_id, content_version=minted["content_version"])

    assert moved.status_code == 200, moved.text
    assert moved.json()["status"] == "moved"
    moved_to = moved.json()["frame"]["content_version"]
    new_head = rig.head_store.head_for(spec_id).content_version
    assert new_head != base
    reader = LiveSnapshotReader(store=rig.version_store)
    version = reader.read_version(spec_id, moved_to)
    assert version is not None and version.base_content_version == new_head
    assert recorded_mintings(
        location.rendering for location in reader.read_artifact_locations(spec_id, moved_to)
    ) == [MintingOptions(without_party_totals=True)]
    snapshot = rig.client.get(f"/api/datasets/{spec_id}/{moved_to}")
    assert snapshot.json()["trace"]["party_totals_left_out"] == party_fixture._LEFT_OUT
    assert snapshot.json()["coverage"]["rows"] == 1


def _totals_alone_rig(*, code_index: CodeIndex, stores_of: _Rig | None = None) -> _Rig:
    """A rig over the party-totals dataset whose DAC total stands alone in 2021, under
    `code_index`."""
    return _rig(
        graph=party_fixture._graph(),
        coverage=selector_fixture._totals_alone_coverage(),
        body=selector_fixture._TOTALS_ALONE_CSV,
        code_index=code_index,
        stores_of=stores_of,
    )


def test_a_recorded_cut_whose_frame_expired_re_renders_to_its_own_version():
    """A cut minted when the cut left out every total, so the DAC total of 2021 too, keeps
    answering `current` once its own frame expires under an index that marks that total whole:
    the refresh replays the cut its trace records on the head's frame, where a fresh cut now
    keeps the DAC total in 2021."""
    minting = _totals_alone_rig(code_index=party_fixture._code_index())
    spec_id, sidecar = party_fixture._spec_id_and_sidecar()
    cut = party_fixture._mint(minting.client, spec_id=spec_id, sidecar=sidecar, options=_CUT_FRAME)
    assert cut["coverage"]["rows"] == 1
    fresh = party_fixture._mint(
        _totals_alone_rig(code_index=selector_fixture._dac_total_whole()).client,
        spec_id=spec_id,
        sidecar=sidecar,
        options=_CUT_FRAME,
    )
    assert fresh["content_version"] != cut["content_version"]
    minting.expire_frame(spec_id=spec_id, content_version=cut["content_version"])
    serving = _totals_alone_rig(code_index=selector_fixture._dac_total_whole(), stores_of=minting)

    resp = serving.refresh(spec_id=spec_id, content_version=cut["content_version"])

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": spec_id, "content_version": cut["content_version"]},
    }
    assert serving.frame_servable(spec_id=spec_id, content_version=cut["content_version"])


# --- A sum that left the frame unchanged shares the raw version ---

_MEMBER_A_ROW, _MEMBER_B_ROW = collapse_fixture._CSV_BODY.removeprefix(
    collapse_fixture._CSV_HEADER
).splitlines(keepends=True)
_PLAIN_MINT = {"bundle": False}
_SUMMED_MINT = {"bundle": False, "groupby": _GROUPBY}


def _absent_reads_zero_code_index() -> CodeIndex:
    """The family's code index with its dataset declared to read a missing observation as
    zero, so a sum over the family is allowed while one member has no rows."""
    base = collapse_fixture._code_index()
    return CodeIndex(
        stamp=base.stamp,
        declarations=base.declarations.values(),
        codes=base.codes,
        statvars=base.statvars,
        datasets=(
            DatasetDeclaration(
                provenance=collapse_fixture._PROVENANCE,
                label="OECD DAC2a",
                description="Aid disbursements by recipient.",
                opt_in=False,
                rank=0,
                absent_observation="zero",
                account="provider",
            ),
        ),
    )


def _one_contributor_rig(*mints: dict) -> _CollapseRig:
    """A rig holding the family dataset while only one member has rows, minted once per entry
    of `mints`. A groupby sum leaves that frame unchanged, so every mint is linked under the
    head's raw version, which the rig names as both of its versions."""
    rig = _family_rig(
        body=collapse_fixture._CSV_HEADER + _MEMBER_A_ROW,
        code_index=_absent_reads_zero_code_index(),
    )
    spec_id, sidecar = collapse_fixture._spec_id_and_sidecar()
    versions = {
        collapse_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options=options)[
            "content_version"
        ]
        for options in mints
    }
    content_version = rig.head_store.head_for(spec_id).content_version
    assert versions == {content_version}
    return _CollapseRig(
        rig=rig,
        spec_id=spec_id,
        base_content_version=content_version,
        collapsed_content_version=content_version,
    )


def _move_to(shared: _CollapseRig, *rows: str) -> None:
    """Serve `rows` as the dataset's observations and force the next refresh to rebuild."""
    shared.rig.body["text"] = collapse_fixture._CSV_HEADER + "".join(rows)
    shared.rig.head_store.age_head(shared.spec_id, age=_PAST_MAX_AGE)


@pytest.mark.parametrize(
    ("mints", "moved_to_a_collapse"),
    [
        pytest.param((_PLAIN_MINT,), False, id="only-a-plain-file"),
        pytest.param((_SUMMED_MINT,), True, id="only-a-groupby-file"),
    ],
)
def test_a_raw_version_linking_one_rendering_refreshes_as_that_rendering(
    mints: tuple[dict, ...], moved_to_a_collapse: bool
):
    shared = _one_contributor_rig(*mints)
    _move_to(shared, _MEMBER_A_ROW, _MEMBER_B_ROW)

    resp = shared.rig.refresh(spec_id=shared.spec_id, content_version=shared.base_content_version)

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["status"] == "moved"
    new_head = shared.rig.head_store.head_for(shared.spec_id).content_version
    assert new_head != shared.base_content_version
    moved_to = answer["frame"]["content_version"]
    if moved_to_a_collapse:
        # Two members now contribute, so the sum has a version of its own.
        assert shared.version_rows()[moved_to] == new_head
        assert shared.recorded_groupby(moved_to) == [_GROUPBY]
    else:
        assert moved_to == new_head


def test_a_raw_version_linking_both_renderings_refuses_once_they_differ():
    shared = _one_contributor_rig(_PLAIN_MINT, _SUMMED_MINT)
    named = {"spec_id": shared.spec_id, "content_version": shared.base_content_version}

    # The head is the named version, where the two renderings are one frame.
    unmoved = shared.rig.refresh(**named)
    assert unmoved.status_code == 200, unmoved.text
    assert unmoved.json() == {"status": "current", "frame": named}

    # New data with one member contributing still sums to the plain frame.
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"))
    one_frame = shared.rig.refresh(**named)
    assert one_frame.status_code == 200, one_frame.text
    assert one_frame.json()["status"] == "moved"
    moved_to = one_frame.json()["frame"]["content_version"]
    assert moved_to == shared.rig.head_store.head_for(shared.spec_id).content_version
    assert moved_to != shared.base_content_version
    # The current frame records both renderings too, so it is as ambiguous as the named one.
    assert shared.version_rows()[moved_to] is None
    assert shared.recorded_groupby(moved_to) == [_GROUPBY]

    # A second member contributes: the sum is no longer the plain frame.
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"), _MEMBER_B_ROW)
    for frame in (named, {**named, "content_version": moved_to}):
        refused = shared.rig.refresh(**frame)
        assert refused.status_code == 422, refused.text
        assert refused.json()["code"] == "rendering_not_recorded"
        assert refused.json()["retryable"] is False


_CUT_MINT = {"bundle": False, "without_party_totals": True}
_CUT_SUMMED_MINT = {**_CUT_MINT, "groupby": _GROUPBY}


def _draws_on(rig: _Rig, *, spec_id: str, stored: str, target: str) -> bool:
    """Whether the render routes draw a card made on `stored`, recording its rendering as a
    card revision does, on `target` (`refresh_can_answer`)."""
    reader = LiveSnapshotReader(store=rig.version_store)
    recorded = refresh_module.stored_rendering(
        spec_id=spec_id, content_version=stored, snapshot_reader=reader
    )
    return refresh_module.refresh_can_answer(
        stored=refresh_module.StoredRendering.from_record(recorded.to_record()),
        target=refresh_module.stored_rendering(
            spec_id=spec_id, content_version=target, snapshot_reader=reader
        ),
    )


def test_a_raw_version_linking_a_cut_and_a_summed_cut_refreshes_until_they_differ():
    """A raw version linking a party-totals cut and a cut with a groupby, and no plain file,
    refreshes while both render the current data as one frame, and refuses once they differ."""
    shared = _one_contributor_rig(_CUT_MINT, _CUT_SUMMED_MINT)
    named = {"spec_id": shared.spec_id, "content_version": shared.base_content_version}

    unmoved = shared.rig.refresh(**named)
    assert unmoved.status_code == 200, unmoved.text
    assert unmoved.json() == {"status": "current", "frame": named}

    # New data with one member contributing: the cut leaves nothing out and the sum is the frame.
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"))
    one_frame = shared.rig.refresh(**named)
    assert one_frame.status_code == 200, one_frame.text
    assert one_frame.json()["status"] == "moved"
    moved_to = one_frame.json()["frame"]["content_version"]
    assert moved_to == shared.rig.head_store.head_for(shared.spec_id).content_version
    assert moved_to != shared.base_content_version
    assert shared.recorded_groupby(moved_to) == [_GROUPBY]
    assert _draws_on(
        shared.rig, spec_id=shared.spec_id, stored=shared.base_content_version, target=moved_to
    )

    # Its frame file expires: the refresh rebuilds and answers the same frame, as for any frame.
    shared.rig.expire_frame(spec_id=shared.spec_id, content_version=moved_to)
    rebuilt = shared.rig.refresh(**named)
    assert rebuilt.status_code == 200, rebuilt.text
    assert rebuilt.json() == one_frame.json()
    assert shared.rig.frame_servable(spec_id=shared.spec_id, content_version=moved_to)

    # A second member contributes: the summed cut is no longer the cut frame.
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"), _MEMBER_B_ROW)
    for frame in (named, {**named, "content_version": moved_to}):
        refused = shared.rig.refresh(**frame)
        assert refused.status_code == 422, refused.text
        assert refused.json()["code"] == "rendering_not_recorded"


def test_a_rendering_the_rebuilt_head_cannot_make_refuses(monkeypatch: pytest.MonkeyPatch):
    """A frame recorded under a cut and a summed cut whose summed cut cannot be made from the
    head rebuilds, and when it cannot be made from the rebuilt head either the refresh refuses
    `rendering_not_recorded`."""
    shared = _one_contributor_rig(_CUT_MINT, _CUT_SUMMED_MINT)
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"))
    real = refresh_module._rendering_frame
    summed_cut = MintingOptions(
        groupby=GroupbyOptions.model_validate(_GROUPBY), without_party_totals=True
    )

    def cannot_make_the_summed_cut(**kwargs):
        return None if kwargs["minting"] == summed_cut else real(**kwargs)

    monkeypatch.setattr(refresh_module, "_rendering_frame", cannot_make_the_summed_cut)
    sdmx_requests = len(shared.rig.sdmx_requests)

    resp = shared.rig.refresh(spec_id=shared.spec_id, content_version=shared.base_content_version)

    assert resp.status_code == 422, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == ("rendering_not_recorded", False)
    # The head read sent the call to a rebuild, which fetched.
    assert len(shared.rig.sdmx_requests) > sdmx_requests


@pytest.mark.parametrize(
    ("mints", "answers_current"),
    [
        pytest.param((_CUT_MINT,), True, id="one-rendering"),
        pytest.param((_CUT_MINT, _CUT_SUMMED_MINT), False, id="two-renderings"),
    ],
)
def test_a_head_linking_every_rendering_still_makes_each_one(
    mints: tuple[dict, ...], answers_current: bool, monkeypatch: pytest.MonkeyPatch
):
    """A head version that links a file of each rendering a frame is recorded under answers
    only once each rendering makes it again from the head: a file may have been linked there
    from another head. When the summed cut now makes another frame, the refresh refuses
    `rendering_not_recorded`; a frame recorded under the cut alone stays current."""
    shared = _one_contributor_rig(*mints)
    summed_cut = MintingOptions(
        groupby=GroupbyOptions.model_validate(_GROUPBY), without_party_totals=True
    )
    real = refresh_module._rendering_frame

    def the_summed_cut_makes_another_frame(**kwargs):
        frame = real(**kwargs)
        if frame is None or kwargs["minting"] != summed_cut:
            return frame
        return replace(
            frame, manifest=frame.manifest.model_copy(update={"content_version": "cv_other"})
        )

    monkeypatch.setattr(refresh_module, "_rendering_frame", the_summed_cut_makes_another_frame)
    named = {"spec_id": shared.spec_id, "content_version": shared.base_content_version}

    resp = shared.rig.refresh(**named)

    if answers_current:
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "current", "frame": named}
    else:
        assert resp.status_code == 422, resp.text
        assert (resp.json()["code"], resp.json()["retryable"]) == ("rendering_not_recorded", False)


def _raised_refusals(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """The case keywords of each `RenderingNotRecorded` the refresh door builds, in order. The
    door builds one only to raise it, so these name the case each refusal raised."""
    cases: list[dict[str, object]] = []

    class Recording(RenderingNotRecorded):
        def __init__(self, *, spec_id: str, content_version: str, **case: Any) -> None:
            cases.append(case)
            super().__init__(spec_id=spec_id, content_version=content_version, **case)

    monkeypatch.setattr(refresh_module, "RenderingNotRecorded", Recording)
    return cases


def test_a_rebuild_to_the_named_frame_that_cannot_make_its_renderings_refuses(
    monkeypatch: pytest.MonkeyPatch,
):
    """A frame recorded under a cut and a summed cut whose head frame expired rebuilds to the
    same version, and when neither rendering can be made from the rebuilt head the refresh
    refuses `rendering_not_recorded` as a rendering the rebuild cannot make."""
    shared = _one_contributor_rig(_CUT_MINT, _CUT_SUMMED_MINT)
    shared.rig.expire_frame(spec_id=shared.spec_id, content_version=shared.base_content_version)
    monkeypatch.setattr(refresh_module, "_rendering_frame", lambda **kwargs: None)
    named = {"spec_id": shared.spec_id, "content_version": shared.base_content_version}
    raised = _raised_refusals(monkeypatch)

    resp = shared.rig.refresh(**named)

    assert resp.status_code == 422, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == ("rendering_not_recorded", False)
    assert raised == [{"renderings": 2, "agreed_on": None, "unmade": True, "plain": False}]
    assert shared.rig.frame_servable(**named)


def test_a_rebuild_is_checked_against_its_own_head_when_another_writer_moves_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """A frame recorded under the plain rendering and a groupby that left it unchanged rebuilds
    on new data, and another writer then moves the head: the refresh checks the renderings
    against the version the rebuild materialised and answers `moved` to it."""
    shared = _one_contributor_rig(_PLAIN_MINT, _SUMMED_MINT)
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"))
    real_serve_spec = refresh_module.serve_spec

    def then_another_writer(**kwargs):
        served = real_serve_spec(**kwargs)
        head = shared.rig.head_store.head_for(shared.spec_id)
        shared.rig.head_store._heads[shared.spec_id] = replace(
            head, content_version=shared.base_content_version
        )
        return served

    monkeypatch.setattr(refresh_module, "serve_spec", then_another_writer)

    resp = shared.rig.refresh(spec_id=shared.spec_id, content_version=shared.base_content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "moved"
    moved_to = resp.json()["frame"]["content_version"]
    assert moved_to != shared.base_content_version
    assert shared.version_rows()[moved_to] is None
    assert shared.recorded_groupby(moved_to) == [_GROUPBY]


_SUMMED = MintingOptions(groupby=GroupbyOptions.model_validate(_GROUPBY))


def test_a_servable_head_that_cannot_make_the_groupby_of_a_plain_frame_refuses_at_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """A raw version linking a plain file and a groupby file, whose head another caller's plain
    mint moved, refuses `rendering_not_recorded` without a rebuild when the servable head
    cannot make the groupby."""
    shared = _one_contributor_rig(_PLAIN_MINT, _SUMMED_MINT)
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"))
    spec_id, sidecar = collapse_fixture._spec_id_and_sidecar()
    collapse_fixture._mint(shared.rig.client, spec_id=spec_id, sidecar=sidecar, options=_PLAIN_MINT)
    assert shared.rig.head_store.head_for(spec_id).content_version != shared.base_content_version
    real = refresh_module._rendering_frame
    monkeypatch.setattr(
        refresh_module,
        "_rendering_frame",
        lambda **kwargs: None if kwargs["minting"] == _SUMMED else real(**kwargs),
    )
    sdmx_requests = len(shared.rig.sdmx_requests)
    raised = _raised_refusals(monkeypatch)

    resp = shared.rig.refresh(spec_id=shared.spec_id, content_version=shared.base_content_version)

    assert resp.status_code == 422, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == ("rendering_not_recorded", False)
    assert raised == [{"renderings": 2, "unmade": False, "unmade_at_head": True, "plain": True}]
    assert len(shared.rig.sdmx_requests) == sdmx_requests


def test_a_rebuilt_plain_frame_that_cannot_make_the_groupby_refuses(
    monkeypatch: pytest.MonkeyPatch,
):
    """A raw version linking a plain file and a groupby file whose head frame expired rebuilds
    on new data, and when the rebuilt version cannot make the groupby the refresh refuses
    `rendering_not_recorded`."""
    shared = _one_contributor_rig(_PLAIN_MINT, _SUMMED_MINT)
    shared.rig.expire_frame(spec_id=shared.spec_id, content_version=shared.base_content_version)
    _move_to(shared, _MEMBER_A_ROW.replace(",1000,", ",3000,"))
    real = refresh_module._rendering_frame
    monkeypatch.setattr(
        refresh_module,
        "_rendering_frame",
        lambda **kwargs: None if kwargs["minting"] == _SUMMED else real(**kwargs),
    )
    sdmx_requests = len(shared.rig.sdmx_requests)
    raised = _raised_refusals(monkeypatch)

    resp = shared.rig.refresh(spec_id=shared.spec_id, content_version=shared.base_content_version)

    assert resp.status_code == 422, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == ("rendering_not_recorded", False)
    assert raised == [{"renderings": 2, "unmade": True, "unmade_at_head": False, "plain": True}]
    assert len(shared.rig.sdmx_requests) > sdmx_requests


def test_renderings_that_agree_on_a_frame_the_rebuild_did_not_serve_refuse(
    monkeypatch: pytest.MonkeyPatch,
):
    """When the renderings of a frame a rebuild served agree on another version, the refresh
    refuses `rendering_not_recorded`."""
    shared = _one_contributor_rig(_CUT_MINT, _CUT_SUMMED_MINT)
    # The head is fresh, so the rebuild runs only because the head read finds nothing.
    shared.rig.body["text"] = collapse_fixture._CSV_HEADER + _MEMBER_A_ROW.replace(
        ",1000,", ",3000,"
    )
    real = refresh_module._agreed_frame
    calls: list[str] = []

    def agreed_on_the_old_head(**kwargs):
        calls.append(kwargs["head_content_version"])
        if len(calls) == 1:
            return None
        return real(**{**kwargs, "head_content_version": shared.base_content_version})

    monkeypatch.setattr(refresh_module, "_agreed_frame", agreed_on_the_old_head)

    resp = shared.rig.refresh(spec_id=shared.spec_id, content_version=shared.base_content_version)

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "rendering_not_recorded"
    new_head = shared.rig.head_store.head_for(shared.spec_id).content_version
    assert calls == [shared.base_content_version, new_head]


# The groupby that keeps `testAxis` sums `flowAxis`, so it has a version of its own.
_KEEPS_TEST_AXIS = {"by": ["recipient", "testAxis"], "op": "sum"}


def _cut_family_index(*, dac_total_kind: str) -> CodeIndex:
    """The cut family's index with `flowAxis` summable, the DAC total of `dac_total_kind`: a
    `total` the cut leaves out, or a `party` it keeps."""
    index = selector_fixture._cut_family_code_index(flow_summable=True)
    return CodeIndex(
        stamp=index.stamp,
        declarations=index.declarations.values(),
        codes=index.codes,
        statvars=index.statvars,
        parties=[
            replace(row, kind=dac_total_kind) if row.dcid == party_fixture._DAC_TOTAL else row
            for row in index.parties.values()
        ],
    )


def _cut_family_rig(*, dac_total_kind: str, stores_of: _Rig | None = None) -> _Rig:
    return _rig(
        graph=selector_fixture._cut_family_graph(),
        coverage=selector_fixture._cut_family_coverage(),
        body=selector_fixture._CUT_FAMILY_CSV,
        code_index=_cut_family_index(dac_total_kind=dac_total_kind),
        stores_of=stores_of,
    )


def _move_cut_family(rig: _Rig, *, spec_id: str, by: int) -> None:
    """Add `by` to every value the source serves and force the next refresh to rebuild."""
    rig.body["text"] = selector_fixture._CUT_FAMILY_CSV.replace(
        ",2020,9", f",2020,{9 + by}"
    ).replace(",2020,1", f",2020,{1 + by}")
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)


def test_a_derived_version_recorded_under_two_renderings_refreshes_until_they_differ():
    """A groupby and a groupby with a party-totals cut that left nothing out share one derived
    version. It refreshes while both render the current data as one frame, and refuses once a
    code index that makes the DAC total a total has the cut leave it out at a later head."""
    minting = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    summed = party_fixture._mint(
        minting.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _KEEPS_TEST_AXIS},
    )
    both = party_fixture._mint(
        minting.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS},
    )
    assert both["content_version"] == summed["content_version"]
    assert summed["content_version"] != minting.head_store.head_for(spec_id).content_version
    named = {"spec_id": spec_id, "content_version": summed["content_version"]}

    unmoved = minting.refresh(**named)
    assert unmoved.status_code == 200, unmoved.text
    assert unmoved.json() == {"status": "current", "frame": named}

    _move_cut_family(minting, spec_id=spec_id, by=1)
    one_frame = minting.refresh(**named)
    assert one_frame.status_code == 200, one_frame.text
    assert one_frame.json()["status"] == "moved"
    moved_to = one_frame.json()["frame"]["content_version"]
    reader = LiveSnapshotReader(store=minting.version_store)
    version = reader.read_version(spec_id, moved_to)
    assert version is not None
    assert version.base_content_version == minting.head_store.head_for(spec_id).content_version
    assert recorded_mintings(
        location.rendering for location in reader.read_artifact_locations(spec_id, moved_to)
    ) == [
        MintingOptions(groupby=GroupbyOptions.model_validate(_KEEPS_TEST_AXIS)),
        MintingOptions(
            groupby=GroupbyOptions.model_validate(_KEEPS_TEST_AXIS), without_party_totals=True
        ),
    ]
    assert _draws_on(minting, spec_id=spec_id, stored=summed["content_version"], target=moved_to)

    serving = _cut_family_rig(dac_total_kind="total", stores_of=minting)
    _move_cut_family(serving, spec_id=spec_id, by=2)
    for frame in (named, {**named, "content_version": moved_to}):
        refused = serving.refresh(**frame)
        assert refused.status_code == 422, refused.text
        assert refused.json()["code"] == "rendering_not_recorded"


def test_a_rebuild_checks_the_renderings_again_at_the_version_it_materialised(
    monkeypatch: pytest.MonkeyPatch,
):
    """A head read agrees the groupby and the summed cut on one derived version, and the call
    rebuilds to count the head's year coverage. The rebuild materialises new data, with its
    flows swapped, that the groupby sums to that version again, and where the summed cut makes
    another frame: the refresh refuses `rendering_not_recorded`."""
    rig = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    named = party_fixture._mint(
        rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _KEEPS_TEST_AXIS},
    )
    party_fixture._mint(
        rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS},
    )
    rig.body["text"] = _cut_family_body(_flows_value)
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    agreed_head = rig.head_store.head_for(spec_id).content_version
    # The head read finds the head's year coverage uncounted, once.
    uncounted = iter([True])
    real_counted = current_module.year_coverage_counted
    monkeypatch.setattr(
        current_module,
        "year_coverage_counted",
        lambda coverage: False if next(uncounted, False) else real_counted(coverage),
    )
    real_materialise = current_module._materialise_and_record

    def on_swapped_flows(*args, **kwargs):
        rig.body["text"] = _cut_family_body(functools.partial(_flows_value, swapped=True))
        return real_materialise(*args, **kwargs)

    monkeypatch.setattr(current_module, "_materialise_and_record", on_swapped_flows)
    summed_cut = MintingOptions(
        groupby=GroupbyOptions.model_validate(_KEEPS_TEST_AXIS), without_party_totals=True
    )
    real_frame = refresh_module._rendering_frame
    served: list[str] = []

    def the_cut_leaves_rows_out_at_a_new_head(**kwargs):
        frame = real_frame(**kwargs)
        if kwargs["minting"] == summed_cut and kwargs["head_content_version"] != agreed_head:
            # The summed cut of the rebuilt data is another frame: the head's raw one stands in.
            return refresh_module._servable_frame(
                spec_id=spec_id, content_version=kwargs["head_content_version"], deps=kwargs["deps"]
            )
        return frame

    monkeypatch.setattr(refresh_module, "_rendering_frame", the_cut_leaves_rows_out_at_a_new_head)
    real_serve_spec = refresh_module.serve_spec

    def recording_serve_spec(**kwargs):
        manifest = real_serve_spec(**kwargs)
        served.append(manifest.content_version)
        return manifest

    monkeypatch.setattr(refresh_module, "serve_spec", recording_serve_spec)

    resp = rig.refresh(spec_id=spec_id, content_version=named["content_version"])

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "rendering_not_recorded"
    # The rebuild materialised new data and served the version the head read agreed on.
    assert rig.head_store.head_for(spec_id).content_version != agreed_head
    agreed_on = LiveSnapshotReader(store=rig.version_store).read_version(spec_id, served[0])
    assert agreed_on is not None
    assert agreed_on.base_content_version == agreed_head


_SUMMED_KEEPING_TEST_AXIS = {"bundle": False, "groupby": _KEEPS_TEST_AXIS}


@pytest.mark.parametrize(
    ("mints", "dac_total_kind", "answers_current"),
    [
        pytest.param((_SUMMED_KEEPING_TEST_AXIS,), "total", True, id="one-rendering"),
        pytest.param(
            (_SUMMED_KEEPING_TEST_AXIS, {**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS}),
            "party",
            True,
            id="renderings-agree",
        ),
        pytest.param(
            (_SUMMED_KEEPING_TEST_AXIS, {**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS}),
            "total",
            False,
            id="summed-cut-moved",
        ),
    ],
)
def test_a_rebuild_that_remakes_the_named_frame_answers_current_once_its_renderings_agree(
    mints: tuple[dict, ...], dac_total_kind: str, answers_current: bool
):
    """A rebuild on new data, with its flows swapped, that the groupby sums to the named derived
    version again answers `current` when every rendering the frame is recorded under makes that
    version from the rebuilt head. When a code index that makes the DAC total a total has the
    summed cut leave it out, the summed cut makes another frame and the refresh refuses
    `rendering_not_recorded`; a frame recorded under the groupby alone stays current."""
    minting = _cut_family_rig(dac_total_kind="party")
    minting.body["text"] = _cut_family_body(_flows_value)
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    versions = {
        party_fixture._mint(minting.client, spec_id=spec_id, sidecar=sidecar, options=options)[
            "content_version"
        ]
        for options in mints
    }
    assert len(versions) == 1
    named = {"spec_id": spec_id, "content_version": versions.pop()}
    head = minting.head_store.head_for(spec_id).content_version
    serving = _cut_family_rig(dac_total_kind=dac_total_kind, stores_of=minting)
    serving.body["text"] = _cut_family_body(functools.partial(_flows_value, swapped=True))
    serving.head_store.age_head(spec_id, age=_PAST_MAX_AGE)

    resp = serving.refresh(**named)

    assert serving.head_store.head_for(spec_id).content_version != head
    if answers_current:
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "current", "frame": named}
    else:
        assert resp.status_code == 422, resp.text
        assert (resp.json()["code"], resp.json()["retryable"]) == ("rendering_not_recorded", False)


@dataclass(frozen=True)
class _LinkedFromAnotherHead:
    """A dataset whose groupby, at head A, makes Ag and whose groupby with the summed cut makes
    D, both recorded from A, while at head B, where the DAC total has no rows, the groupby makes
    D too, so the groupby's file is linked under D while D's record keeps base A. The head is
    back at A and fresh. `groupby_at_a` is `None` when only the summed cut was minted."""

    rig: _Rig
    spec_id: str
    head_a: str
    groupby_at_a: str | None
    summed_cut_at_a: str


def _bodies_a_and_b() -> tuple[str, str]:
    """The cut family's observations at head A, and at head B, where the DAC total has no rows."""
    body_a = _cut_family_body(_flows_value)
    header, *rows = body_a.splitlines(keepends=True)
    body_b = header + "".join(row for row in rows if f",{party_fixture._DAC_TOTAL}," not in row)
    return body_a, body_b


def _linked_from_another_head(
    *,
    summed_cut_first: bool,
    unheaded_at_a: bool = False,
    summed_cut_at_b: bool = False,
    unheaded_after_b: bool = False,
    summed_cut_again_at_a: bool = False,
    summed_cut_alone: bool = False,
) -> _LinkedFromAnotherHead:
    """`unheaded_at_a` clears the heads A's links name, as on links written before links named
    heads, and `summed_cut_at_b` mints the summed cut at B too, which makes D there as well.
    `unheaded_after_b` clears every link's heads once B's mints are linked, and
    `summed_cut_again_at_a` mints the summed cut at A once the head is back there, which links
    it under D again naming A. `summed_cut_alone` mints the summed cut alone at A, clears its
    link's heads, and mints the summed cut alone at B, which makes D there and names B on that
    link."""
    rig = _cut_family_rig(dac_total_kind="total")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    groupby = {"bundle": False, "groupby": _KEEPS_TEST_AXIS}
    summed_cut = {**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS}

    def mint(options: dict) -> str:
        return party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options=options)[
            "content_version"
        ]

    body_a, body_b = _bodies_a_and_b()
    rig.body["text"] = body_a
    if summed_cut_alone:
        summed_cut_at_a = mint(summed_cut)
        head_a = rig.head_store.head_for(spec_id).content_version
        _unhead_links(rig, spec_id=spec_id)
        rig.body["text"] = body_b
        rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
        assert mint(summed_cut) == summed_cut_at_a
        assert rig.head_store.head_for(spec_id).content_version != head_a
        rig.body["text"] = body_a
        rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
        mint({"bundle": False})
        assert rig.head_store.head_for(spec_id).content_version == head_a
        return _LinkedFromAnotherHead(
            rig=rig,
            spec_id=spec_id,
            head_a=head_a,
            groupby_at_a=None,
            summed_cut_at_a=summed_cut_at_a,
        )
    made_at_a = {
        name: mint(options)
        for name, options in (
            [("summed_cut", summed_cut), ("groupby", groupby)]
            if summed_cut_first
            else [("groupby", groupby), ("summed_cut", summed_cut)]
        )
    }
    head_a = rig.head_store.head_for(spec_id).content_version
    assert made_at_a["groupby"] != made_at_a["summed_cut"]
    if unheaded_at_a:
        _unhead_links(rig, spec_id=spec_id)
    rig.body["text"] = body_b
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    assert mint(groupby) == made_at_a["summed_cut"]
    if summed_cut_at_b:
        assert mint(summed_cut) == made_at_a["summed_cut"]
    assert rig.head_store.head_for(spec_id).content_version != head_a
    if unheaded_after_b:
        _unhead_links(rig, spec_id=spec_id)
    # Another caller's plain mint brings the head back to A, fresh, so the refresh reads it.
    rig.body["text"] = body_a
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    mint({"bundle": False})
    assert rig.head_store.head_for(spec_id).content_version == head_a
    if summed_cut_again_at_a:
        assert mint(summed_cut) == made_at_a["summed_cut"]
    return _LinkedFromAnotherHead(
        rig=rig,
        spec_id=spec_id,
        head_a=head_a,
        groupby_at_a=made_at_a["groupby"],
        summed_cut_at_a=made_at_a["summed_cut"],
    )


def _counted_renders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The head each `head_rendering` call of the refresh door renders from, in call order."""
    real = refresh_module.head_rendering
    renders: list[str] = []

    def counting(**kwargs):
        renders.append(kwargs["content_version"])
        return real(**kwargs)

    monkeypatch.setattr(refresh_module, "head_rendering", counting)
    return renders


def _counted_real_renders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The version each render from a stored frame (`current._render_stored_rendering`) renders
    from, in call order. A `head_rendering` that finds its file linked renders nothing."""
    real = current_module._render_stored_rendering
    renders: list[str] = []

    def counting(**kwargs):
        renders.append(kwargs["content_version"])
        return real(**kwargs)

    monkeypatch.setattr(current_module, "_render_stored_rendering", counting)
    return renders


def _refresh_as_groupby_card(
    linked: _LinkedFromAnotherHead,
    *,
    content_version: str,
    monkeypatch: pytest.MonkeyPatch,
    summed_cut: bool = False,
) -> httpx.Response:
    """Refresh `content_version` as a card recorded under the groupby alone, or the groupby with
    the summed cut when `summed_cut`, with the base its version's record keeps, A."""
    recorded = StoredRendering(
        base_content_version=linked.head_a,
        collapsed=True,
        mintings=(
            MintingOptions(
                groupby=GroupbyOptions.model_validate(_KEEPS_TEST_AXIS),
                without_party_totals=summed_cut,
            ),
        ),
    )
    monkeypatch.setattr(
        refresh_module,
        "answer_frame_refresh",
        functools.partial(refresh_module.answer_frame_refresh, recorded=recorded),
    )
    return linked.rig.refresh(spec_id=linked.spec_id, content_version=content_version)


def _unhead_links(rig: _Rig, *, spec_id: str) -> None:
    """Clear the heads every link under the dataset names, as on links written before links
    named heads."""
    with rig.version_store.transaction() as tx:
        tx.execute("DELETE FROM datasets.version_artifact_head WHERE subject_id = %s", (spec_id,))


@pytest.mark.parametrize("summed_cut_first", [False, True], ids=["groupby-first", "cut-first"])
def test_a_rendering_linked_under_a_derived_version_from_another_head_is_made_again(
    summed_cut_first: bool,
):
    """Back at A, the groupby makes Ag, not D, so a refresh of D, recorded under the groupby and
    the summed cut, refuses `rendering_not_recorded`."""
    linked = _linked_from_another_head(summed_cut_first=summed_cut_first)

    resp = linked.rig.refresh(spec_id=linked.spec_id, content_version=linked.summed_cut_at_a)

    assert resp.status_code == 422, resp.text
    assert (resp.json()["code"], resp.json()["retryable"]) == ("rendering_not_recorded", False)


@pytest.mark.parametrize("summed_cut_first", [False, True], ids=["groupby-first", "cut-first"])
def test_a_groupby_card_on_a_version_its_groupby_made_at_another_head_moves_to_this_heads(
    summed_cut_first: bool, monkeypatch: pytest.MonkeyPatch
):
    """A card made at B as the groupby alone is D, whose record keeps base A. Back at A its
    refresh moves to Ag, the version the groupby made from A, read from the stored versions
    with no render: D's groupby file names B as the head that made it."""
    linked = _linked_from_another_head(summed_cut_first=summed_cut_first)
    renders = _counted_renders(monkeypatch)

    resp = _refresh_as_groupby_card(
        linked, content_version=linked.summed_cut_at_a, monkeypatch=monkeypatch
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "moved",
        "frame": {"spec_id": linked.spec_id, "content_version": linked.groupby_at_a},
    }
    assert renders == []


@pytest.mark.parametrize(
    ("named", "answer"),
    [("groupby_at_a", "current"), ("summed_cut_at_a", "moved")],
    ids=["one-minting-version-counts", "two-minting-version-does-not"],
)
def test_a_link_naming_no_head_counts_only_under_a_one_minting_version_of_that_base(
    named: str, answer: str, monkeypatch: pytest.MonkeyPatch
):
    """With no link naming a head, a groupby card's refresh at A counts Ag, recorded from A
    and linking files of the groupby alone, as the groupby's version from A, and answers it
    with no render. D, recorded from A but linking files of the groupby and the summed cut,
    does not count, so a card on D moves to Ag rather than answering `current`."""
    linked = _linked_from_another_head(summed_cut_first=False)
    _unhead_links(linked.rig, spec_id=linked.spec_id)
    renders = _counted_renders(monkeypatch)

    resp = _refresh_as_groupby_card(
        linked, content_version=getattr(linked, named), monkeypatch=monkeypatch
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": answer,
        "frame": {"spec_id": linked.spec_id, "content_version": linked.groupby_at_a},
    }
    assert renders == []


@pytest.mark.parametrize(
    ("unheaded_at_a", "summed_cut_at_b", "renders_at_a"),
    [(False, True, False), (True, False, True)],
    ids=["made-from-two-heads", "legacy-link-beside-a-later-head"],
)
def test_a_summed_cut_card_on_d_answers_current_at_a(
    unheaded_at_a: bool,
    summed_cut_at_b: bool,
    renders_at_a: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """A card on D recorded under the summed cut alone answers `current` at A, though D's
    groupby file names B. When the summed cut made D at B too, its file names both heads, so the
    stored versions answer with no render. When A's links name no head, D's links record two
    mintings, so its link naming none does not count for A and the summed cut is made again from
    A, which makes D."""
    linked = _linked_from_another_head(
        summed_cut_first=False, unheaded_at_a=unheaded_at_a, summed_cut_at_b=summed_cut_at_b
    )
    renders = _counted_renders(monkeypatch)

    resp = _refresh_as_groupby_card(
        linked, content_version=linked.summed_cut_at_a, monkeypatch=monkeypatch, summed_cut=True
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": linked.spec_id, "content_version": linked.summed_cut_at_a},
    }
    assert renders == ([linked.head_a] if renders_at_a else [])


def test_a_one_minting_legacy_version_another_head_made_again_still_counts_for_its_base(
    monkeypatch: pytest.MonkeyPatch,
):
    """D, recorded from A, links the summed cut alone, and its link names no head. The summed
    cut makes D again at B, which names B on that link. D's links still record the one minting,
    so they count for D's base A, and a summed-cut card on D answers `current` at A with no
    render."""
    linked = _linked_from_another_head(summed_cut_first=False, summed_cut_alone=True)
    renders = _counted_renders(monkeypatch)
    real_renders = _counted_real_renders(monkeypatch)

    resp = _refresh_as_groupby_card(
        linked, content_version=linked.summed_cut_at_a, monkeypatch=monkeypatch, summed_cut=True
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": linked.spec_id, "content_version": linked.summed_cut_at_a},
    }
    assert (renders, real_renders) == ([], [])


def test_a_link_naming_no_head_beside_one_that_gained_a_head_does_not_count_for_the_base(
    monkeypatch: pytest.MonkeyPatch,
):
    """D, recorded from A, links the summed cut made from A and the groupby made from B, and
    neither link names a head. The summed cut made again at A names A on its link, leaving the
    groupby's link the only one naming none. D's links still record two mintings, so that link
    does not count for A, and a groupby card on D moves to Ag rather than answering `current`
    D with no render."""
    linked = _linked_from_another_head(
        summed_cut_first=False, unheaded_after_b=True, summed_cut_again_at_a=True
    )
    renders = _counted_renders(monkeypatch)

    resp = _refresh_as_groupby_card(
        linked, content_version=linked.summed_cut_at_a, monkeypatch=monkeypatch
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "moved",
        "frame": {"spec_id": linked.spec_id, "content_version": linked.groupby_at_a},
    }
    assert renders == []


@pytest.mark.parametrize("unheaded", [False, True], ids=["raw-version", "pre-migration-derived"])
def test_a_one_minting_frame_at_the_head_it_was_made_from_answers_current_with_no_render(
    unheaded: bool, monkeypatch: pytest.MonkeyPatch
):
    """A frame recorded under one groupby answers `current` at the head it was made from with
    no render: a raw version the groupby leaves unchanged, whose files link only from its own
    frame, and a derived version whose links name no head and record the one groupby."""
    if unheaded:
        rig = _cut_family_rig(dac_total_kind="party")
        spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
        sidecar = {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": selector_fixture._CUT_FAMILY_IDENTITY,
        }
        named = party_fixture._mint(
            rig.client,
            spec_id=spec_id,
            sidecar=sidecar,
            options={"bundle": False, "groupby": _KEEPS_TEST_AXIS},
        )["content_version"]
        assert named != rig.head_store.head_for(spec_id).content_version
        _unhead_links(rig, spec_id=spec_id)
    else:
        shared = _one_contributor_rig(_SUMMED_MINT)
        rig, spec_id, named = shared.rig, shared.spec_id, shared.base_content_version
        assert named == rig.head_store.head_for(spec_id).content_version
    renders = _counted_renders(monkeypatch)

    resp = rig.refresh(spec_id=spec_id, content_version=named)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": spec_id, "content_version": named},
    }
    assert renders == []


def test_a_named_version_whose_frame_expired_is_made_again_with_its_recorded_cut(
    monkeypatch: pytest.MonkeyPatch,
):
    """The summed cut made V1 from the head under a code index that keeps the DAC total and V2
    under one that leaves it out. With V1's frame file expired, a refresh of V1 under the second
    index makes V1 again from the head, replaying the cut its trace records, and answers
    `current`, rather than move to V2."""
    minting = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    summed_cut = {**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS}
    v1 = party_fixture._mint(minting.client, spec_id=spec_id, sidecar=sidecar, options=summed_cut)[
        "content_version"
    ]
    serving = _cut_family_rig(dac_total_kind="total", stores_of=minting)
    v2 = party_fixture._mint(serving.client, spec_id=spec_id, sidecar=sidecar, options=summed_cut)[
        "content_version"
    ]
    head = serving.head_store.head_for(spec_id).content_version
    assert len({v1, v2, head}) == 3
    serving.expire_frame(spec_id=spec_id, content_version=v1)
    renders = _counted_renders(monkeypatch)

    resp = serving.refresh(spec_id=spec_id, content_version=v1)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": spec_id, "content_version": v1},
    }
    assert renders == [head]
    assert serving.frame_servable(spec_id=spec_id, content_version=v1)


def test_a_cut_card_on_a_raw_version_answers_current_at_its_head_after_a_code_index_change(
    monkeypatch: pytest.MonkeyPatch,
):
    """A party-totals cut that leaves nothing out under one code index is linked under the raw
    head H. Under a code index that makes the DAC total a total, a refresh at the same head still
    answers `current` H with no render: the code index is outside the rule, and the cut's file
    linked under H records that the cut made H from H."""
    minting = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    named = party_fixture._mint(
        minting.client, spec_id=spec_id, sidecar=sidecar, options=_CUT_MINT
    )["content_version"]
    assert named == minting.head_store.head_for(spec_id).content_version
    serving = _cut_family_rig(dac_total_kind="total", stores_of=minting)
    renders = _counted_renders(monkeypatch)
    real_renders = _counted_real_renders(monkeypatch)

    resp = serving.refresh(spec_id=spec_id, content_version=named)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": spec_id, "content_version": named},
    }
    assert (renders, real_renders) == ([], [])


def test_a_raw_version_card_answers_from_its_own_file_though_a_link_under_it_names_a_head(
    monkeypatch: pytest.MonkeyPatch,
):
    """A groupby that leaves the raw head H unchanged answers a card on H with no render, also
    once a link under H names another head, as a derived write from that head whose rows hash to
    H leaves it: the groupby's file linked under H records that the groupby made H from H."""
    shared = _one_contributor_rig(_SUMMED_MINT)
    rig, spec_id, named = shared.rig, shared.spec_id, shared.base_content_version
    with rig.version_store.transaction() as tx:
        linked = tx.fetch_all(
            "SELECT rendering_key, rendering FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, named),
        )
        (key,) = [row["rendering_key"] for row in linked if "groupby" in row["rendering"]]
        tx.execute(
            "INSERT INTO datasets.version_artifact_head "
            "(subject_id, content_version, rendering_key, head_content_version) "
            "VALUES (%s, %s, %s, %s)",
            (spec_id, named, key, "cv_another_head"),
        )
    renders = _counted_renders(monkeypatch)
    real_renders = _counted_real_renders(monkeypatch)

    resp = rig.refresh(spec_id=spec_id, content_version=named)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": spec_id, "content_version": named},
    }
    assert (renders, real_renders) == ([], [])


def test_a_version_recorded_from_another_head_is_made_again_with_this_heads_cut(
    monkeypatch: pytest.MonkeyPatch,
):
    """The summed cut makes D at B, where the DAC total has no rows and the cut leaves nothing
    out, and makes D again at A, where it leaves the DAC total out. D's record keeps base B and
    its trace B's empty cut. With D's frame file expired, a refresh of D at A makes the summed
    cut again from A with the cut A decides, which makes D, and answers `current` with D."""
    rig = _cut_family_rig(dac_total_kind="total")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    summed_cut = {**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS}

    def mint(options: dict) -> str:
        return party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options=options)[
            "content_version"
        ]

    body_a, body_b = _bodies_a_and_b()
    rig.body["text"] = body_b
    made = mint(summed_cut)
    head_b = rig.head_store.head_for(spec_id).content_version
    rig.body["text"] = body_a
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    assert mint(summed_cut) == made
    head_a = rig.head_store.head_for(spec_id).content_version
    assert head_a != head_b
    rig.expire_frame(spec_id=spec_id, content_version=made)
    renders = _counted_renders(monkeypatch)

    resp = rig.refresh(spec_id=spec_id, content_version=made)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": spec_id, "content_version": made},
    }
    assert renders == [head_a]


def test_a_link_of_an_unreadable_groupby_keeps_its_version_from_counting_for_its_base(
    monkeypatch: pytest.MonkeyPatch,
):
    """V, recorded from head H, links first a groupby this build's grammar refuses, and later a
    summed-cut file another head made. The unreadable link records a minting the rule cannot
    read, so V counts for H only under case 2, which its summed-cut link's head rules out, and a
    summed-cut card on V renders the summed cut from H."""
    rig = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    named = party_fixture._mint(
        rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _KEEPS_TEST_AXIS},
    )["content_version"]
    head = rig.head_store.head_for(spec_id).content_version
    assert named != head
    summed_cut = {"format": "csv", "bundle": False, "groupby": _KEEPS_TEST_AXIS}
    summed_cut["without_party_totals"] = True
    with rig.version_store.transaction() as tx:
        linked = tx.fetch_all(
            "SELECT artifact_digest, rendering_key, rendering FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, named),
        )
        groupby_links = [row for row in linked if "groupby" in row["rendering"]]
        assert groupby_links
        for row in groupby_links:
            tx.execute(
                "UPDATE datasets.version_artifact SET rendering = %s "
                "WHERE subject_id = %s AND content_version = %s AND rendering_key = %s",
                (
                    {**row["rendering"], "groupby": {**_KEEPS_TEST_AXIS, "op": "median"}},
                    spec_id,
                    named,
                    row["rendering_key"],
                ),
            )
        tx.execute(
            "INSERT INTO datasets.version_artifact "
            "(subject_id, content_version, artifact_digest, rendering_key, rendering, media_type) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                spec_id,
                named,
                groupby_links[0]["artifact_digest"],
                rendering_key(summed_cut),
                summed_cut,
                "text/csv",
            ),
        )
        tx.execute(
            "INSERT INTO datasets.version_artifact_head "
            "(subject_id, content_version, rendering_key, head_content_version) "
            "VALUES (%s, %s, %s, %s)",
            (spec_id, named, rendering_key(summed_cut), "cv_another_head"),
        )
    recorded = StoredRendering(
        base_content_version=head,
        collapsed=True,
        mintings=(
            MintingOptions(
                groupby=GroupbyOptions.model_validate(_KEEPS_TEST_AXIS), without_party_totals=True
            ),
        ),
    )
    monkeypatch.setattr(
        refresh_module,
        "answer_frame_refresh",
        functools.partial(refresh_module.answer_frame_refresh, recorded=recorded),
    )
    renders = _counted_renders(monkeypatch)

    resp = rig.refresh(spec_id=spec_id, content_version=named)

    assert resp.status_code == 200, resp.text
    assert renders == [head]


def test_a_frame_of_two_renderings_at_a_fresh_head_that_made_it_answers_with_no_render(
    monkeypatch: pytest.MonkeyPatch,
):
    """A groupby and a groupby with a party-totals cut that left nothing out make one derived
    version from the head, and each links its file there naming that head. A refresh at the
    same fresh head answers `current` from the stored versions and renders nothing."""
    rig = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    made = [
        party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options=options)[
            "content_version"
        ]
        for options in (
            {"bundle": False, "groupby": _KEEPS_TEST_AXIS},
            {**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS},
        )
    ]
    assert made[0] == made[1]
    assert refresh_module.stored_rendering(
        spec_id=spec_id,
        content_version=made[0],
        snapshot_reader=LiveSnapshotReader(store=rig.version_store),
    ).ambiguous
    renders = _counted_renders(monkeypatch)

    resp = rig.refresh(spec_id=spec_id, content_version=made[0])

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": spec_id, "content_version": made[0]},
    }
    assert renders == []


def test_a_refresh_finds_the_version_its_rebuild_recorded(monkeypatch: pytest.MonkeyPatch):
    """A derived frame recorded under two renderings whose head and frame expired rebuilds on
    new data, and the check after the rebuild finds the groupby the rebuild recorded instead of
    rendering it again."""
    minting = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    summed = party_fixture._mint(
        minting.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _KEEPS_TEST_AXIS},
    )
    party_fixture._mint(
        minting.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS},
    )
    head = minting.head_store.head_for(spec_id).content_version
    minting.expire_frame(spec_id=spec_id, content_version=head)
    minting.expire_frame(spec_id=spec_id, content_version=summed["content_version"])
    # The head is fresh, so the rebuild runs only because the head read cannot render.
    minting.body["text"] = selector_fixture._CUT_FAMILY_CSV.replace(",2020,9", ",2020,19")
    rendered: list[tuple[str, MintingOptions]] = []
    real = refresh_module.head_rendering

    def recording(**kwargs):
        options = kwargs["options"]
        if options is not None and options.groupby is not None:
            rendered.append(
                (
                    kwargs["content_version"],
                    MintingOptions(
                        groupby=options.groupby,
                        without_party_totals=options.without_party_totals,
                    ),
                )
            )
        return real(**kwargs)

    monkeypatch.setattr(refresh_module, "head_rendering", recording)

    resp = minting.refresh(spec_id=spec_id, content_version=summed["content_version"])

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "moved"
    new_head = minting.head_store.head_for(spec_id).content_version
    assert new_head != head
    groupby = GroupbyOptions.model_validate(_KEEPS_TEST_AXIS)
    # Only the summed cut renders from the new head; the rebuild recorded the groupby.
    assert [entry for entry in rendered if entry[0] == new_head] == [
        (new_head, MintingOptions(groupby=groupby, without_party_totals=True))
    ]


def _cut_family_body(value) -> str:
    """The cut family's observations with `value(member, provider)` as each row's value."""
    header = selector_fixture._CUT_FAMILY_CSV.splitlines(keepends=True)[0]
    return header + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{member},{provider},country/KEN,USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2020,{value(member, provider)},,facet1\r\n"
        for member in selector_fixture._CUT_FAMILY
        for provider in (party_fixture._DAC_TOTAL, party_fixture._FRANCE)
    )


def _flows_value(member: str, provider: str, *, swapped: bool = False) -> int:
    """A value per member and provider whose flows `X` and `Y` trade places when `swapped`, which
    leaves every sum over `flowAxis` as it was."""
    axis, flow = selector_fixture._CUT_FAMILY[member]
    if swapped:
        flow = {"FlowX": "FlowY", "FlowY": "FlowX"}[flow]
    base = 900 if provider == party_fixture._DAC_TOTAL else 100
    return base + {"TestAxisA": 10, "TestAxisB": 20}[axis] + {"FlowX": 1, "FlowY": 5}[flow]


@dataclass
class _RelinkedRig:
    """A derived frame `named` recorded under the groupby and the groupby with the cut, and a
    later collapse `relinked` recorded under the groupby alone from `relinked_base`, whose
    rows the source's current data, with its flows swapped, sums to again."""

    rig: _Rig
    spec_id: str
    named: str
    relinked: str
    relinked_base: str


def _relinked_rig() -> _RelinkedRig:
    rig = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    groupby = {"bundle": False, "groupby": _KEEPS_TEST_AXIS}
    named = party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options=groupby)
    party_fixture._mint(
        rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS},
    )
    rig.body["text"] = _cut_family_body(_flows_value)
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    relinked = party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options=groupby)
    relinked_base = rig.head_store.head_for(spec_id).content_version
    assert relinked["content_version"] != named["content_version"]
    rig.body["text"] = _cut_family_body(functools.partial(_flows_value, swapped=True))
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    return _RelinkedRig(
        rig=rig,
        spec_id=spec_id,
        named=named["content_version"],
        relinked=relinked["content_version"],
        relinked_base=relinked_base,
    )


def _record_head_renderings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, MintingOptions]]:
    """Record the head and minting options of each groupby render the refresh asks for."""
    rendered: list[tuple[str, MintingOptions]] = []
    real = refresh_module.head_rendering

    def recording(**kwargs):
        options = kwargs["options"]
        if options is not None and options.groupby is not None:
            rendered.append(
                (
                    kwargs["content_version"],
                    MintingOptions(
                        groupby=options.groupby, without_party_totals=options.without_party_totals
                    ),
                )
            )
        return real(**kwargs)

    monkeypatch.setattr(refresh_module, "head_rendering", recording)
    return rendered


def test_a_rebuild_is_checked_against_the_head_it_wrote():
    """A rebuild that sums the new data to a collapse first recorded from an older head, whose
    frame has expired, is checked against the head the rebuild wrote, where both renderings
    make that collapse, and answers `moved` to it."""
    relinked = _relinked_rig()
    relinked.rig.expire_frame(spec_id=relinked.spec_id, content_version=relinked.relinked_base)

    resp = relinked.rig.refresh(spec_id=relinked.spec_id, content_version=relinked.named)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "moved",
        "frame": {"spec_id": relinked.spec_id, "content_version": relinked.relinked},
    }
    reader = LiveSnapshotReader(store=relinked.rig.version_store)
    version = reader.read_version(relinked.spec_id, relinked.relinked)
    assert version is not None and version.base_content_version == relinked.relinked_base
    assert relinked.rig.head_store.head_for(relinked.spec_id).content_version != (
        relinked.relinked_base
    )


def test_a_collapse_a_rebuild_relinks_from_the_head_is_found(monkeypatch: pytest.MonkeyPatch):
    """The check after a rebuild that sums the new data to a collapse first recorded from an
    older head finds that collapse as the groupby of the new head and renders only the summed
    cut, from the new head."""
    relinked = _relinked_rig()
    rendered = _record_head_renderings(monkeypatch)

    resp = relinked.rig.refresh(spec_id=relinked.spec_id, content_version=relinked.named)

    assert resp.status_code == 200, resp.text
    assert resp.json()["frame"]["content_version"] == relinked.relinked
    new_head = relinked.rig.head_store.head_for(relinked.spec_id).content_version
    groupby = GroupbyOptions.model_validate(_KEEPS_TEST_AXIS)
    assert rendered == [(new_head, MintingOptions(groupby=groupby, without_party_totals=True))]


def test_a_second_head_read_sees_what_another_rebuild_recorded(
    monkeypatch: pytest.MonkeyPatch,
):
    """A head read that finds the head's frame expired, followed by another caller's rebuild of
    the same head that records its collapse, and a second head read: the second read finds that
    collapse and renders only the summed cut."""
    rig = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    named = party_fixture._mint(
        rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _KEEPS_TEST_AXIS},
    )
    party_fixture._mint(
        rig.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS},
    )
    rig.body["text"] = _cut_family_body(_flows_value)
    rig.head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    head = rig.head_store.head_for(spec_id).content_version
    rig.expire_frame(spec_id=spec_id, content_version=head)
    groupby = GroupbyOptions.model_validate(_KEEPS_TEST_AXIS)
    rendered = _record_head_renderings(monkeypatch)
    real_serve_spec = refresh_module.serve_spec

    def two_head_reads(**kwargs):
        deps = kwargs["deps"]
        load_head = kwargs["load_head"]
        assert load_head(head, deps.coverage.snapshot()) is None
        # Another caller's groupby read rebuilds the head and records its collapse.
        real_serve_spec(
            stored=kwargs["stored"],
            options=refresh_module.frame_options(MintingOptions(groupby=groupby)),
            trigger="refresh",
            deps=deps,
            trace_id=None,
        )
        rendered.clear()
        second = load_head(head, deps.coverage.snapshot())
        assert second is not None
        return second

    monkeypatch.setattr(refresh_module, "serve_spec", two_head_reads)

    resp = rig.refresh(spec_id=spec_id, content_version=named["content_version"])

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "moved"
    assert rendered == [(head, MintingOptions(groupby=groupby, without_party_totals=True))]


def test_a_second_head_read_finds_the_collapse_the_first_read_made(
    monkeypatch: pytest.MonkeyPatch,
):
    """A head read that renders the groupby to a collapse first recorded from an older head,
    then cannot make the summed cut, sends the call to a rebuild. The operations backlog is
    full, so the head is read again, and that read finds the collapse the first one made and
    renders only the summed cut."""
    relinked = _relinked_rig()
    rig, spec_id = relinked.rig, relinked.spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    # The head's raw version holds the swapped flows, and nothing is derived from it yet.
    party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    head = rig.head_store.head_for(spec_id).content_version
    groupby = GroupbyOptions.model_validate(_KEEPS_TEST_AXIS)
    summed_cut = MintingOptions(groupby=groupby, without_party_totals=True)
    rendered = _record_head_renderings(monkeypatch)
    real_frame = refresh_module._rendering_frame
    summed_cut_calls: list[int] = []

    def first_summed_cut_unmade(**kwargs):
        if kwargs["minting"] == summed_cut:
            summed_cut_calls.append(1)
            if len(summed_cut_calls) == 1:
                return None
        return real_frame(**kwargs)

    def backlog_full(*args, **kwargs):
        raise OperationBacklogFull(in_flight=20, limit=20)

    monkeypatch.setattr(refresh_module, "_rendering_frame", first_summed_cut_unmade)
    monkeypatch.setattr(current_module, "_materialise_and_record", backlog_full)

    resp = rig.refresh(spec_id=spec_id, content_version=relinked.named)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "moved",
        "frame": {"spec_id": spec_id, "content_version": relinked.relinked},
    }
    assert rendered == [
        (head, MintingOptions(groupby=groupby)),
        (head, summed_cut),
    ]


def test_a_rebuild_replaces_the_version_a_head_read_made_from_the_same_head(
    monkeypatch: pytest.MonkeyPatch,
):
    """A head read renders the groupby to one version and cannot make the summed cut, so the
    call rebuilds. The rebuild materialises the same head and serves the groupby as another
    version, and the refresh checks the renderings against the version the rebuild served."""
    relinked = _relinked_rig()
    rig, spec_id = relinked.rig, relinked.spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    party_fixture._mint(rig.client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    head = rig.head_store.head_for(spec_id).content_version
    groupby = MintingOptions(groupby=GroupbyOptions.model_validate(_KEEPS_TEST_AXIS))
    summed_cut = MintingOptions(groupby=groupby.groupby, without_party_totals=True)
    real_frame = refresh_module._rendering_frame
    first_read: list[MintingOptions] = []

    def first_read_makes_another_version(**kwargs):
        minting = kwargs["minting"]
        if minting in (groupby, summed_cut) and len(first_read) < 2:
            first_read.append(minting)
            if minting == summed_cut:
                return None
            # The head read made the groupby as a version, here the older collapse's base, that
            # records the groupby alone.
            kwargs["derived"].add(
                base_content_version=head, minting=groupby, content_version=relinked.relinked_base
            )
            return refresh_module._servable_frame(
                spec_id=spec_id, content_version=relinked.relinked_base, deps=kwargs["deps"]
            )
        return real_frame(**kwargs)

    monkeypatch.setattr(refresh_module, "_rendering_frame", first_read_makes_another_version)

    resp = rig.refresh(spec_id=spec_id, content_version=relinked.named)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "moved",
        "frame": {"spec_id": spec_id, "content_version": relinked.relinked},
    }
    # The rebuild materialised the head the first read was made from.
    assert rig.head_store.head_for(spec_id).content_version == head


@pytest.mark.parametrize("renderings", [1, 2], ids=["one-rendering", "two-renderings"])
def test_a_refresh_lists_the_dataset_versions_once(
    renderings: int, monkeypatch: pytest.MonkeyPatch
):
    """A refresh of a derived frame, which looks up a derived version of the new head for each
    rendering, reads the versions made from that head once and lists no other version: one
    recorded under one rendering at a head another caller's plain mint moved, read from that
    head, and one recorded under two renderings through a rebuild."""
    minting = _cut_family_rig(dac_total_kind="party")
    spec_id = recompute_identity(selector_fixture._CUT_FAMILY_IDENTITY).spec_id
    sidecar = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": selector_fixture._CUT_FAMILY_IDENTITY,
    }
    summed = party_fixture._mint(
        minting.client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": _KEEPS_TEST_AXIS},
    )
    if renderings == 2:
        party_fixture._mint(
            minting.client,
            spec_id=spec_id,
            sidecar=sidecar,
            options={**_CUT_MINT, "groupby": _KEEPS_TEST_AXIS},
        )
    _move_cut_family(minting, spec_id=spec_id, by=1)
    if renderings == 1:
        # Another caller's plain mint moves the head, so the refresh reads it rather than
        # rebuild.
        party_fixture._mint(
            minting.client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False}
        )
    listed: list[tuple[str, str | None]] = []
    real_derived = LiveVersionLister.list_head_versions
    real_all = LiveVersionLister.list_versions

    def counting_derived(self, subject_id, *, head_content_version):
        listed.append((subject_id, head_content_version))
        return real_derived(self, subject_id, head_content_version=head_content_version)

    def counting_all(self, subject_id):
        listed.append((subject_id, None))
        return real_all(self, subject_id)

    monkeypatch.setattr(LiveVersionLister, "list_head_versions", counting_derived)
    monkeypatch.setattr(LiveVersionLister, "list_versions", counting_all)

    resp = minting.refresh(spec_id=spec_id, content_version=summed["content_version"])

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "moved"
    new_head = minting.head_store.head_for(spec_id).content_version
    assert listed == [(spec_id, new_head)]


def test_a_refresh_under_the_recorded_rendering_ignores_a_file_linked_later(
    monkeypatch: pytest.MonkeyPatch,
):
    """A card made on a raw version linking only its groupby sum records that rendering. A plain
    file linked under the version later makes its records ambiguous, so once the renderings
    differ a refresh from the records refuses, and one under the recorded rendering moves to the
    sum of the new data."""
    shared = _one_contributor_rig(_SUMMED_MINT)
    reader = LiveSnapshotReader(store=shared.rig.version_store)
    recorded = refresh_module.stored_rendering(
        spec_id=shared.spec_id,
        content_version=shared.base_content_version,
        snapshot_reader=reader,
    )
    assert not recorded.ambiguous
    _, sidecar = collapse_fixture._spec_id_and_sidecar()
    linked = collapse_fixture._mint(
        shared.rig.client, spec_id=shared.spec_id, sidecar=sidecar, options=_PLAIN_MINT
    )
    assert linked["content_version"] == shared.base_content_version
    assert refresh_module.stored_rendering(
        spec_id=shared.spec_id, content_version=shared.base_content_version, snapshot_reader=reader
    ).ambiguous
    _move_to(shared, _MEMBER_A_ROW, _MEMBER_B_ROW)
    named = {"spec_id": shared.spec_id, "content_version": shared.base_content_version}

    from_records = shared.rig.refresh(**named)
    assert from_records.status_code == 422, from_records.text
    assert from_records.json()["code"] == "rendering_not_recorded"

    monkeypatch.setattr(
        refresh_module,
        "answer_frame_refresh",
        functools.partial(refresh_module.answer_frame_refresh, recorded=recorded),
    )
    resp = shared.rig.refresh(**named)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "moved"
    moved_to = resp.json()["frame"]["content_version"]
    assert (
        shared.version_rows()[moved_to]
        == shared.rig.head_store.head_for(shared.spec_id).content_version
    )
    assert shared.recorded_groupby(moved_to) == [_GROUPBY]


# --- A rebuild over the inline budget ---


def _worker_deps(plain: _PlainRig) -> DefinitionDeps:
    """The worker's deps over the rig's stores, graph, SDMX body, and coverage. The worker
    advances the rig's head, as the deployed worker does."""
    rig = plain.rig
    settings = DatasetsBuildSettings()
    return DefinitionDeps(
        registry=LiveSpecRegistry(store=rig.spec_store),
        spec_reader=LiveSpecReader(store=rig.spec_store),
        graph=plain.graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=rig.body["text"],
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
        ),
        artifact_store=rig.artifact_store,
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=4,
        enumerator=MemberEnumerator(),
        enrichment=EnrichmentLookup(plain.graph),
        code_index=get_index(),
        rate_cache=RateTableCache(),
        coverage=plain.coverage,
        version_writer=LiveVersionRecordWriter(store=rig.version_store),
        snapshot_reader=LiveSnapshotReader(store=rig.version_store),
        head_store=rig.head_store,
        head_recheck_interval=timedelta(seconds=settings.head_recheck_seconds),
        head_max_age=timedelta(seconds=settings.head_max_age_seconds),
    )


def _queue_a_rebuild(plain: _PlainRig, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Move the data and put every rebuild over the inline budget, then refresh once."""
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_INTERVAL)
    plain.move_data()
    monkeypatch.setattr(current_module, "within_sync_budget", lambda estimate, **_: False)
    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_rebuild_over_the_inline_budget_answers_pending_then_moved(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    sdmx_requests = len(plain.rig.sdmx_requests)

    queued = _queue_a_rebuild(plain, monkeypatch)

    assert queued["status"] == "pending"
    assert queued["retry_after_s"] == 1.5
    operation_id = queued["operation_id"]
    assert plain.rig.task_client.enqueued == [(operation_id, 0)]
    assert len(plain.rig.sdmx_requests) == sdmx_requests

    # A call while the operation is queued names it and enqueues nothing.
    again = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
    assert again.json() == queued
    assert plain.rig.task_client.enqueued == [(operation_id, 0)]

    claimed = plain.rig.operation_store.claim(
        operation_id, submission_count=0, lease_seconds=60, final_delivery=False
    )
    assert claimed is not None
    run_operation(
        operation=claimed,
        deps=_worker_deps(plain),
        operation_store=plain.rig.operation_store,
    )
    completed = plain.rig.operation_store.get(operation_id)
    assert completed is not None and completed.state == "done"

    done = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
    assert done.status_code == 200, done.text
    assert done.json() == {
        "status": "moved",
        "frame": {"spec_id": plain.spec_id, "content_version": completed.content_version},
    }
    assert completed.content_version != plain.content_version
    assert plain.rig.task_client.enqueued == [(operation_id, 0)]


def test_a_done_rebuild_of_a_kind_column_frame_is_checked_under_the_current_digest(
    monkeypatch: pytest.MonkeyPatch,
):
    """A rebuild whose rendering `carries_party_kind` is linked by the worker under the
    key salted with the serving code index's digest. The refresh reads the head the worker
    advanced under the same key, so it answers `moved`; a key without the digest would find
    nothing and submit the operation again."""
    real_frame_options = refresh_module.frame_options
    monkeypatch.setattr(
        refresh_module,
        "frame_options",
        lambda minting: real_frame_options(minting).model_copy(update={"columns": ["party_kind"]}),
    )
    plain = _plain_rig()
    operation_id = _queue_a_rebuild(plain, monkeypatch)["operation_id"]
    claimed = plain.rig.operation_store.claim(
        operation_id, submission_count=0, lease_seconds=60, final_delivery=False
    )
    assert claimed is not None
    run_operation(
        operation=claimed,
        deps=_worker_deps(plain),
        operation_store=plain.rig.operation_store,
    )
    completed = plain.rig.operation_store.get(operation_id)
    assert completed is not None and completed.state == "done"

    done = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert done.status_code == 200, done.text
    assert done.json() == {
        "status": "moved",
        "frame": {"spec_id": plain.spec_id, "content_version": completed.content_version},
    }
    assert plain.rig.task_client.enqueued == [(operation_id, 0)]


def test_a_done_rebuild_whose_files_left_storage_runs_again_before_it_is_answered(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    rig = plain.rig
    operation_id = _queue_a_rebuild(plain, monkeypatch)["operation_id"]

    def run_queued_rebuild(*, submission_count: int) -> str:
        claimed = rig.operation_store.claim(
            operation_id, submission_count=submission_count, lease_seconds=60, final_delivery=False
        )
        assert claimed is not None
        run_operation(
            operation=claimed,
            deps=_worker_deps(plain),
            operation_store=rig.operation_store,
        )
        completed = rig.operation_store.get(operation_id)
        assert completed is not None and completed.state == "done"
        assert completed.content_version is not None
        return completed.content_version

    rebuilt = run_queued_rebuild(submission_count=0)
    # The rebuilt version's files leave storage: the store holds no object and no location.
    with rig.version_store.transaction() as tx:
        gone = tx.fetch_all(
            "SELECT artifact_digest FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (plain.spec_id, rebuilt),
        )
        for row in gone:
            tx.execute(
                "DELETE FROM datasets.artifact_location WHERE artifact_digest = %s",
                (row["artifact_digest"],),
            )
    assert gone
    for digest, tier in list(rig.artifact_store._objects):
        if digest in {row["artifact_digest"] for row in gone}:
            del rig.artifact_store._objects[(digest, tier)]

    resp = rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    # The submission resets the done row and enqueues it again, so the answer is the rerun.
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "pending", "operation_id": operation_id, "retry_after_s": 1.5}
    assert rig.task_client.enqueued == [(operation_id, 0), (operation_id, 1)]

    assert run_queued_rebuild(submission_count=1) == rebuilt
    answered = rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
    assert answered.json() == {
        "status": "moved",
        "frame": {"spec_id": plain.spec_id, "content_version": rebuilt},
    }


def test_a_queued_rebuild_that_failed_for_good_is_an_internal_error(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    queued = _queue_a_rebuild(plain, monkeypatch)
    claimed = plain.rig.operation_store.claim(
        queued["operation_id"], submission_count=0, lease_seconds=60, final_delivery=False
    )
    assert claimed is not None
    plain.rig.operation_store.fail(
        claimed.id, attempt=claimed.attempt_count, refusal_code="over_ceiling", retryable=False
    )

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 500, resp.text
    assert resp.json()["code"] == "internal_error"
    assert resp.json()["retryable"] is False


# --- One rebuild for concurrent callers ---


# The slot and flight wait the concurrency tests configure. A rebuild held longer than this
# shows whether a waiting caller gave up at it.
_SHORT_WAIT_SECONDS = 0.1


@dataclass
class _HeldRebuild:
    """Four callers at one stale head: three refreshes and a definition read. The first call
    into materialisation is held until `release` is set, and `entries` counts every call."""

    plain: _PlainRig
    entered: threading.Event
    release: threading.Event
    entries: list[int]
    responses: list[httpx.Response]
    head_reads: int

    def run(self, *, hold_seconds: float = 0.0) -> None:
        """Start the callers, hold the first rebuild until all four have read the stale head,
        then `hold_seconds` longer. The rebuilding caller took its turn without waiting, so it
        rebuilds from the head it read."""
        rig, plain = self.plain.rig, self.plain

        def refresh() -> None:
            self.responses.append(
                rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
            )

        def read_definition() -> None:
            self.responses.append(rig.client.get(f"/api/datasets/{plain.spec_id}?bundle=false"))

        # The client context keeps every request on one portal loop.
        callers = [
            threading.Thread(target=contextvars.copy_context().run, args=(caller,))
            for caller in (refresh, refresh, refresh, read_definition)
        ]
        for caller in callers:
            caller.start()
        assert self.entered.wait(timeout=5.0), "no caller reached materialisation"
        waited_until = time.monotonic() + 5.0
        while (
            rig.head_store.read_head_calls < self.head_reads + 4 and time.monotonic() < waited_until
        ):
            time.sleep(0.01)
        assert rig.head_store.read_head_calls == self.head_reads + 4
        time.sleep(hold_seconds)
        self.release.set()
        for caller in callers:
            caller.join(timeout=10.0)
        assert len(self.responses) == 4


def _held_rebuild(
    monkeypatch: pytest.MonkeyPatch, *, first_rebuild_raises: Exception | None = None
) -> _HeldRebuild:
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    # A slot for every caller, so only the shared rebuild keeps them out of materialisation.
    plain = _plain_rig(materialise_max_concurrency=4, materialise_wait_seconds=_SHORT_WAIT_SECONDS)
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_INTERVAL)
    plain.move_data()
    held = _HeldRebuild(
        plain=plain,
        entered=threading.Event(),
        release=threading.Event(),
        entries=[],
        responses=[],
        head_reads=plain.rig.head_store.read_head_calls,
    )
    real_materialise_group = serve_module.materialise_group

    def counting_materialise_group(*args, **kwargs):
        held.entries.append(1)
        if len(held.entries) == 1:
            held.entered.set()
            assert held.release.wait(timeout=10.0)
            if first_rebuild_raises is not None:
                raise first_rebuild_raises
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", counting_materialise_group)
    return held


def test_concurrent_refreshes_and_a_definition_read_enter_materialisation_once(
    monkeypatch: pytest.MonkeyPatch,
):
    held = _held_rebuild(monkeypatch)
    plain, rig = held.plain, held.plain.rig

    # The rebuild outlasts the materialise wait, which is slept through here. Each waiter's
    # bound is its own request deadline.
    held.run(hold_seconds=3 * _SHORT_WAIT_SECONDS)

    responses = held.responses
    assert [resp.status_code for resp in responses] == [200] * 4, [r.text for r in responses]
    assert len(held.entries) == 1
    new_head = rig.head_store.head_for(plain.spec_id).content_version
    assert new_head != plain.content_version
    answers = [resp.json() for resp in responses]
    refreshed = [answer for answer in answers if "status" in answer]
    assert (
        refreshed
        == [{"status": "moved", "frame": {"spec_id": plain.spec_id, "content_version": new_head}}]
        * 3
    )
    (manifest,) = [answer for answer in answers if "status" not in answer]
    assert manifest["content_version"] == new_head


def test_a_failed_shared_rebuild_leaves_each_waiter_to_rebuild_and_answer_for_itself(
    monkeypatch: pytest.MonkeyPatch,
):
    held = _held_rebuild(monkeypatch, first_rebuild_raises=RuntimeError("upstream fell over"))
    plain, rig = held.plain, held.plain.rig

    held.run()

    # The caller whose rebuild failed answers its own failure. No waiter inherits it.
    assert sorted(resp.status_code for resp in held.responses) == [200, 200, 200, 500]
    assert 2 <= len(held.entries) <= 4
    new_head = rig.head_store.head_for(plain.spec_id).content_version
    assert new_head != plain.content_version
    for answer in (resp.json() for resp in held.responses if resp.status_code == 200):
        frame = answer["frame"] if "status" in answer else answer
        assert frame["content_version"] == new_head


def test_a_saturated_work_executor_refuses_a_refresh_and_still_serves_a_head_hit(
    monkeypatch: pytest.MonkeyPatch,
):
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    # One work thread, so one held refresh fills the work executor. The definition read is a
    # head hit on the read executor.
    plain = _plain_rig(work_max_concurrency=1)
    rig = plain.rig
    entered = threading.Event()
    release = threading.Event()
    real_stored_rendering = refresh_module.stored_rendering

    def held_stored_rendering(**kwargs):
        entered.set()
        assert release.wait(timeout=10.0)
        return real_stored_rendering(**kwargs)

    monkeypatch.setattr(refresh_module, "stored_rendering", held_stored_rendering)
    held: list[httpx.Response] = []

    def refresh() -> None:
        held.append(rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version))

    holder = threading.Thread(target=contextvars.copy_context().run, args=(refresh,))
    holder.start()
    assert entered.wait(timeout=5.0), "the refresh never reached its worker"
    refused = rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
    definition = rig.client.get(f"/api/datasets/{plain.spec_id}?bundle=false")
    release.set()
    holder.join(timeout=10.0)

    assert (refused.status_code, refused.json()["code"]) == (503, "service_unavailable")
    assert refused.json()["retryable"] is True
    assert definition.status_code == 200, definition.text
    assert definition.json()["content_version"] == plain.content_version
    assert [resp.status_code for resp in held] == [200]


# --- Admission ---


def test_a_call_without_the_principal_is_refused_before_the_body_is_read():
    plain = _plain_rig()

    # A body that is not JSON would refuse `request_invalid` if it were read.
    resp = plain.rig.client.post(
        _ROUTE, content=b"not json", headers={"content-type": "application/json"}
    )

    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "refresh_not_permitted"


def test_the_publish_and_live_refresh_principals_open_only_their_own_routes():
    plain = _plain_rig()
    frame = {"spec_id": plain.spec_id, "content_version": plain.content_version}
    publish = {"x-dfo-principal": _PUBLISH_PRINCIPAL}

    refused = plain.rig.client.post(_ROUTE, json=frame, headers=publish)
    assert refused.status_code == 403, refused.text
    assert refused.json()["code"] == "refresh_not_permitted"

    recipe_path = "/api/datasets/frames/ops_0123456789ab/ocv_0123456789ab/recipe"
    for method, path in (
        ("POST", "/api/datasets/frames/publish"),
        ("GET", recipe_path),
        ("PUT", recipe_path),
    ):
        refused = plain.rig.client.request(method, path, json={}, headers=_LIVE)
        assert refused.status_code == 403, (method, path, refused.text)
        assert refused.json()["code"] == "publish_not_permitted"

    # Each principal opens its own surface.
    assert plain.rig.client.post(_ROUTE, json=frame, headers=_LIVE).status_code == 200
    recipe = plain.rig.client.get(recipe_path, headers=publish)
    assert recipe.status_code == 404, recipe.text
    assert recipe.json()["code"] == "recipe_not_found"


@pytest.mark.parametrize("presented", ["", _LIVE_PRINCIPAL], ids=["empty", "any-value"])
def test_an_unset_principal_refuses_every_caller(presented: str):
    plain = _plain_rig(live_refresh_principal="")

    resp = plain.rig.client.post(
        _ROUTE,
        json={"spec_id": plain.spec_id, "content_version": plain.content_version},
        headers={"x-dfo-principal": presented},
    )

    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "refresh_not_permitted"


def test_a_repeated_principal_header_is_refused():
    plain = _plain_rig()

    resp = plain.rig.client.post(
        _ROUTE,
        json={"spec_id": plain.spec_id, "content_version": plain.content_version},
        headers=[("x-dfo-principal", _LIVE_PRINCIPAL), ("x-dfo-principal", _LIVE_PRINCIPAL)],
    )

    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "refresh_not_permitted"


def test_the_principal_is_read_from_the_unprefixed_environment_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LIVE_REFRESH_PRINCIPAL", raising=False)
    assert DatasetsBuildSettings().live_refresh_principal == ""

    monkeypatch.setenv("LIVE_REFRESH_PRINCIPAL", "from-env")
    monkeypatch.setenv("DATASETS_LIVE_REFRESH_PRINCIPAL", "prefixed")
    assert DatasetsBuildSettings().live_refresh_principal == "from-env"
