"""Frame refresh tests: resolution from stored records, each currency branch, the three answers,
the queued rebuild, the shared rebuild, and admission on `POST /api/datasets/frames/refresh`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

import datasets_api.doors.definition as definition_module
import datasets_api.doors.refresh as refresh_module
import datasets_api.serve as serve_module
from artifact_store import LiveArtifactInventory, Tier
from code_index import CodeIndex, DatasetDeclaration
from dataset_store.snapshot import (
    WORKING_TIER_RETENTION,
    LiveSnapshotReader,
    servable_location_of_media_type,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker import run_operation
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.revision import SentinelRevisionReader
from datasets_api.version_store import LiveVersionLister, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.rate_tables import RateTableCache
from series_coverage.shared import CoverageSnapshot
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from statvar_members import MemberEnumerator

from . import test_serve_groupby_collapse as collapse_fixture
from ._app_factories import make_client
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


class _Coverage(_CoverageFromFacets):
    """Adds the worker's `snapshot_current()` read to the mutable coverage double."""

    def snapshot_current(self) -> CoverageSnapshot:
        return self.snapshot()


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

    def frame1_servable(self, *, spec_id: str, content_version: str) -> bool:
        """Whether the version links a servable frame1 file, the one Presentation loads."""
        locations = LiveSnapshotReader(store=self.version_store).read_artifact_locations(
            spec_id, content_version
        )
        return (
            servable_location_of_media_type(
                locations, media_type=FRAME1_MEDIA_TYPE, now=datetime.now(UTC)
            )
            is not None
        )

    def expire_frame1(self, *, spec_id: str, content_version: str) -> None:
        """Age the version's frame1 working copy past `WORKING_TIER_RETENTION` and drop its
        object, as the bucket's lifecycle rule does."""
        with self.version_store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT artifact_digest FROM datasets.version_artifact "
                "WHERE subject_id = %s AND content_version = %s AND media_type = %s",
                (spec_id, content_version, FRAME1_MEDIA_TYPE),
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
        assert not self.frame1_servable(spec_id=spec_id, content_version=content_version)

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
    definition_pool_size: int | None = None,
    **settings: object,
) -> _Rig:
    """A composed Datasets app over one SQLite version store, with a head store tests can age
    and an SDMX body tests can replace."""
    spec_store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    head_store = _FakeHeadStore()
    reference_writer = FakeArtifactReferenceWriter()
    task_client = FakeTaskClient()
    operation_store = LiveOperationStore(store=FakeOperationDatasetStore())
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
        revision_reader=SentinelRevisionReader(),
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
        definition_pool_size=definition_pool_size,
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
    graph: object
    coverage: _Coverage
    spec_id: str
    content_version: str

    def move_data(self) -> None:
        """Change the facet the probe reads and the rows a rebuild fetches."""
        self.facets[(_STAT_VAR, _RECIPIENT)] = [_FACET_CHANGED]
        self.rig.body["text"] = _CSV_HEADER + _row(
            stat_var_ref=_STAT_VAR_REF, obs_value="2000", period="2021"
        )


def _plain_rig(**kwargs) -> _PlainRig:
    """A rig holding one minted single-member dataset, its head primed by the mint."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    coverage = _Coverage(facets)
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
        graph=graph,
        coverage=coverage,
        spec_id=spec_id,
        content_version=rig.head_store.head_for(spec_id).content_version,
    )


def _count_probe_slots(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record each materialise slot the definition module takes, which a facet probe does."""
    taken: list[int] = []
    real = definition_module.bounded_materialise_slot

    def counting(*args, **kwargs):
        taken.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(definition_module, "bounded_materialise_slot", counting)
    return taken


# --- Plain frames: one branch of the currency check per test ---


def test_a_head_inside_the_recheck_interval_answers_current_without_a_probe(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    probe_slots = _count_probe_slots(monkeypatch)
    graph_calls = dict(plain.graph.calls)
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "current",
        "frame": {"spec_id": plain.spec_id, "content_version": plain.content_version},
    }
    assert resp.headers["cache-control"] == "no-store"
    assert probe_slots == []
    assert plain.graph.calls == graph_calls
    assert len(plain.rig.sdmx_requests) == sdmx_requests
    assert plain.rig.head_store.confirm_head_calls == 0


def test_an_unchanged_probe_past_the_interval_confirms_the_head_and_answers_current(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_INTERVAL)
    probe_slots = _count_probe_slots(monkeypatch)
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "current"
    # The probe ran under one materialise slot and confirmed the head. Nothing was fetched.
    assert probe_slots == [1]
    assert plain.rig.head_store.confirm_head_calls == 1
    assert len(plain.rig.sdmx_requests) == sdmx_requests


def test_a_moved_signal_rebuilds_and_answers_moved_then_and_on_later_calls():
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
    # The rebuilt version holds the frame1 file Presentation reads.
    locations = LiveSnapshotReader(store=plain.rig.version_store).read_artifact_locations(
        plain.spec_id, moved_to
    )
    assert FRAME1_MEDIA_TYPE in {location.media_type for location in locations}

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


def test_a_head_past_the_maximum_age_rebuilds_despite_an_unchanged_probe():
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


def test_a_turned_clock_window_rebuilds_inside_the_recheck_interval():
    plain = _plain_rig()
    head = plain.rig.head_store.head_for(plain.spec_id)
    # The head read an open window that ended in 2021, and a read now ends it this year.
    plain.rig.head_store._heads[plain.spec_id] = replace(
        head, signal=tuple(replace(member, clock_window=(2020, 2021)) for member in head.signal)
    )
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "current"
    assert len(plain.rig.sdmx_requests) == sdmx_requests + 1


def test_a_head_whose_frame1_expired_rebuilds_before_it_answers_current():
    plain = _plain_rig()
    plain.rig.expire_frame1(spec_id=plain.spec_id, content_version=plain.content_version)
    sdmx_requests = len(plain.rig.sdmx_requests)

    resp = plain.rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "current"
    # Nothing stored renders the head's frame1, so the dataset was fetched again.
    assert len(plain.rig.sdmx_requests) == sdmx_requests + 1
    assert plain.rig.frame1_servable(spec_id=plain.spec_id, content_version=plain.content_version)


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
    # The collapse came from the head's stored frame1, with no fetch.
    assert len(collapsed.rig.sdmx_requests) == sdmx_requests


def test_a_recorded_collapse_whose_frame1_expired_is_rendered_again_from_the_head():
    collapsed = _collapse_rig()
    collapsed.move_data()
    answer = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    ).json()
    assert answer["status"] == "moved"
    moved_to = answer["frame"]["content_version"]
    collapsed.rig.expire_frame1(spec_id=collapsed.spec_id, content_version=moved_to)
    sdmx_requests = len(collapsed.rig.sdmx_requests)

    resp = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == answer
    # The collapse came again from the head's stored frame1, with no fetch.
    assert collapsed.rig.frame1_servable(spec_id=collapsed.spec_id, content_version=moved_to)
    assert len(collapsed.rig.sdmx_requests) == sdmx_requests


def test_a_collapsed_frame_with_no_file_recording_its_groupby_refuses():
    collapsed = _collapse_rig()
    # The versions route reads the groupby the refresh resolves.
    assert collapsed.rig.listed_versions(collapsed.spec_id) == {
        collapsed.collapsed_content_version: {
            "content_version": collapsed.collapsed_content_version,
            "base_content_version": collapsed.base_content_version,
            "rendering": "groupby",
            "groupby": _GROUPBY,
        },
        collapsed.base_content_version: {
            "content_version": collapsed.base_content_version,
            "base_content_version": None,
            "rendering": "plain",
            "groupby": None,
        },
    }
    with collapsed.rig.version_store.transaction() as tx:
        tx.execute(
            "DELETE FROM datasets.version_artifact WHERE subject_id = %s "
            "AND content_version = %s AND media_type != %s",
            (collapsed.spec_id, collapsed.collapsed_content_version, FRAME1_MEDIA_TYPE),
        )

    resp = collapsed.rig.refresh(
        spec_id=collapsed.spec_id, content_version=collapsed.collapsed_content_version
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "rendering_not_recorded"
    listed = collapsed.rig.listed_versions(collapsed.spec_id)[collapsed.collapsed_content_version]
    assert (listed["rendering"], listed["groupby"]) == ("not_recorded", None)


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


# --- A rebuild over the inline budget ---


def _worker_deps(plain: _PlainRig) -> DefinitionDeps:
    """The worker's deps over the rig's stores, graph, SDMX body, and coverage."""
    rig = plain.rig
    settings = DatasetsBuildSettings()
    return DefinitionDeps(
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
        revision_reader=SentinelRevisionReader(),
        artifact_store=rig.artifact_store,
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=4,
        enumerator=MemberEnumerator(),
        enrichment=EnrichmentLookup(plain.graph),
        code_index=definition_module.get_index(),
        rate_cache=RateTableCache(),
        coverage=plain.coverage,
        version_writer=LiveVersionRecordWriter(store=rig.version_store),
        snapshot_reader=LiveSnapshotReader(store=rig.version_store),
        head_recheck_interval=timedelta(seconds=settings.head_recheck_seconds),
        head_max_age=timedelta(seconds=settings.head_max_age_seconds),
    )


def _queue_a_rebuild(plain: _PlainRig, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Move the data and put every rebuild over the inline budget, then refresh once."""
    plain.rig.head_store.age_head(plain.spec_id, age=_PAST_INTERVAL)
    plain.move_data()
    monkeypatch.setattr(definition_module, "within_sync_budget", lambda estimate, **_: False)
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

    claimed = plain.rig.operation_store.claim_lease(
        operation_id, submission_count=0, lease_seconds=60
    )
    assert claimed is not None
    run_operation(
        operation=claimed,
        deps=_worker_deps(plain),
        operation_store=plain.rig.operation_store,
        lease_seconds=60,
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


def test_a_done_rebuild_whose_files_left_storage_runs_again_before_it_is_answered(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _plain_rig()
    rig = plain.rig
    operation_id = _queue_a_rebuild(plain, monkeypatch)["operation_id"]

    def run_queued_rebuild(*, submission_count: int) -> str:
        claimed = rig.operation_store.claim_lease(
            operation_id, submission_count=submission_count, lease_seconds=60
        )
        assert claimed is not None
        run_operation(
            operation=claimed,
            deps=_worker_deps(plain),
            operation_store=rig.operation_store,
            lease_seconds=60,
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

    # The done row is failed and submitted again, so the answer is the rebuild, not its version.
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
    claimed = plain.rig.operation_store.claim_lease(
        queued["operation_id"], submission_count=0, lease_seconds=60
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
        """Start the callers, hold the first rebuild until all four have read the stale head
        and the rebuilding one has read it again on taking its turn, then `hold_seconds`
        longer."""
        rig, plain = self.plain.rig, self.plain

        def refresh() -> None:
            self.responses.append(
                rig.refresh(spec_id=plain.spec_id, content_version=plain.content_version)
            )

        def read_definition() -> None:
            self.responses.append(rig.client.get(f"/api/datasets/{plain.spec_id}?bundle=false"))

        # The client context keeps every request on one portal loop.
        with rig.client:
            callers = [threading.Thread(target=refresh) for _ in range(3)]
            callers.append(threading.Thread(target=read_definition))
            for caller in callers:
                caller.start()
            assert self.entered.wait(timeout=5.0), "no caller reached materialisation"
            waited_until = time.monotonic() + 5.0
            while (
                rig.head_store.read_head_calls < self.head_reads + 5
                and time.monotonic() < waited_until
            ):
                time.sleep(0.01)
            assert rig.head_store.read_head_calls == self.head_reads + 5
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


def test_a_saturated_refresh_door_refuses_a_refresh_and_still_serves_a_definition_read(
    monkeypatch: pytest.MonkeyPatch,
):
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    # One thread for each door, so one held refresh fills the refresh door.
    plain = _plain_rig(definition_pool_size=1, refresh_max_concurrency=1)
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

    with rig.client:
        holder = threading.Thread(target=refresh)
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
        ("GET", f"/api/datasets/{plain.spec_id}/versions"),
    ):
        refused = plain.rig.client.request(method, path, json={}, headers=_LIVE)
        assert refused.status_code == 403, (method, path, refused.text)
        assert refused.json()["code"] == "publish_not_permitted"

    # Each principal opens its own surface.
    assert plain.rig.client.post(_ROUTE, json=frame, headers=_LIVE).status_code == 200
    versions = plain.rig.client.get(f"/api/datasets/{plain.spec_id}/versions", headers=publish)
    assert versions.status_code == 200, versions.text


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
