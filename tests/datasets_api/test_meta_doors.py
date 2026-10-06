"""Tests health, schema, capabilities, and operation routes through the composed app.
Schema and capability checks cover document validity, stable ids, model agreement, and caching.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jsonschema
import pytest
from fastapi.responses import JSONResponse

from artifact_store import Tier
from code_index.shared import get_index
from dataset_store.snapshot import StoredArtifactLocation
from datasets_api.capabilities import build_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.meta import (
    OperationFailed,
    OperationNotFound,
    _resolve_operation,
    build_meta_door,
)
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.operation_store import (
    MAX_REDISPATCHES,
    REDISPATCH_GRACE_SECONDS,
    LiveOperationStore,
    Operation,
    lease_stale_after_seconds,
    never_claimed_stale_after_seconds,
)
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.render import artifact_rendering
from datasets_api.schema import (
    Manifest,
    MintRequest,
    OperationResource,
    PinRequest,
    PinResponse,
    build_schema_document,
)
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from datasets_api.version_store import rendering_key
from datasets_api.versions import NullVersionRecordWriter
from dfo_platform import Alternative, ProblemIssue, ProblemSpec
from dfo_platform.build import build_commit
from dfo_platform.config import PublicSiteSettings
from dfo_platform.problem import Cause
from series_coverage.index import Estimate
from spec_registry import LiveSpecRegistry

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._operation_store import FakeOperationDatasetStore
from ._snapshot_doubles import FakeSnapshotReader, FakeSpecReader, _store_and_locate, _version
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_OP_NOW = datetime(2026, 6, 1, tzinfo=UTC)
# The same safe-default settings `LiveOperationStore(store=..., now=...)`
# below resolves for itself (no `settings=`/`backlog_max=`/
# `dispatch_concurrency=` passed), so these thresholds match what it
# actually computes.
_DEFAULT_SETTINGS = DatasetsBuildSettings()
_STALE_AFTER_SECONDS = lease_stale_after_seconds(
    max_retry_duration_seconds=_DEFAULT_SETTINGS.operation_max_retry_duration_seconds
)
_NEVER_CLAIMED_STALE_AFTER_SECONDS = never_claimed_stale_after_seconds(
    backlog_max=_DEFAULT_SETTINGS.operation_backlog_max,
    dispatch_concurrency=_DEFAULT_SETTINGS.operation_dispatch_concurrency,
    max_retry_duration_seconds=_DEFAULT_SETTINGS.operation_max_retry_duration_seconds,
)
_DEFAULT_ESTIMATE = Estimate(rows=1_000_000, requests=40)
# `bundle` defaults to true, so a plain parquet mint is a bundle.
_PARQUET_BUNDLE = MintOptions(format="parquet")


class _FixedOperationStore:
    """Returns operations from a fixed map for route tests."""

    def __init__(self, operations: dict[str, Operation]) -> None:
        self._operations = operations

    def create_or_get(self, **kwargs):
        raise NotImplementedError

    def claim_dispatch(self, id: str):
        return None

    def confirm_dispatch(self, id: str, *, submission_count: int, dispatch_count: int) -> None:
        return None

    def claim_lease(self, id: str, *, submission_count: int, lease_seconds: float):
        return None

    def progress(
        self, id: str, *, attempt: int, rows_written: int, lease_seconds: float | None = None
    ):
        return None

    def complete(self, id: str, *, attempt: int, subject_id: str, content_version: str):
        return None

    def fail(self, id: str, *, attempt: int, refusal_code: str, retryable: bool, refusal=None):
        return None

    def release(self, id: str, *, attempt: int):
        return None

    def expire_if_dead(self, id: str):
        return None

    def fail_unservable_done(self, id: str, *, attempt: int):
        return None

    def get(self, id: str) -> Operation | None:
        return self._operations.get(id)


def _operation(
    *,
    id: str = "op_1",
    state: str,
    rows_written: int = 0,
    estimate: Estimate = _DEFAULT_ESTIMATE,
    subject_id: str | None = None,
    content_version: str | None = None,
    refusal_code: str | None = None,
    retryable: bool | None = None,
    refusal: ProblemSpec | None = None,
    created_at: datetime = _OP_NOW,
    options: MintOptions = _PARQUET_BUNDLE,
) -> Operation:
    return Operation(
        id=id,
        request_identity={"input": {"spec_id": "spec_a"}},
        options=options.model_dump(mode="json"),
        coverage_stamp="cov_stamp_1",
        state=state,
        estimate=estimate,
        rows_written=rows_written,
        subject_id=subject_id,
        content_version=content_version,
        refusal_code=refusal_code,
        attempt_count=1,
        lease_until=None,
        created_at=created_at,
        updated_at=_OP_NOW,
        submission_count=0,
        retryable=retryable,
        dispatch_count=1,
        dispatched_at=created_at,
        dispatch_confirmed=1,
        execution_coverage_stamp=None,
        submitted_at=created_at,
        origin_catalogue_build_id="fake-catalogue",
        refusal=refusal,
    )


def test_healthz_is_contract_shaped() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/healthz")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "build": build_commit(),
        "capability_version": DATASETS_CAPABILITY_VERSION,
    }


def test_capabilities_carries_the_real_computed_version() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/capabilities")

    assert resp.status_code == 200
    body = resp.json()
    assert body["capability_version"] == DATASETS_CAPABILITY_VERSION
    assert body["capability_version"].startswith("dset_")
    assert body["build"] == build_commit()


def test_schema_route_serves_the_built_schema_document() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/schema")

    assert resp.status_code == 200
    assert resp.json() == build_schema_document()


def test_schema_document_validates_against_each_model_and_is_stable() -> None:
    body = build_schema_document()

    assert body["schema_version"] == 1
    assert body["capabilities"] == "/api/datasets/capabilities"

    # Every schema this document carries must itself be a syntactically
    # valid JSON Schema document (Draft 2020-12, pydantic v2's default).
    for key in (
        "mint_request",
        "manifest",
        "pin_request",
        "pin_response",
        "operation_resource",
        "problem_details_extensions",
    ):
        jsonschema.Draft202012Validator.check_schema(body[key])
    jsonschema.Draft202012Validator.check_schema(body["register_response"])

    # Published schema ids are fixed contract values.
    assert (
        body["mint_request"]["$id"] == "https://schemas.dfo.one.org/datasets/v1/mint-request.json"
    )
    assert body["manifest"]["$id"] == "https://schemas.dfo.one.org/datasets/v1/manifest.json"
    assert body["pin_request"]["$id"] == "https://schemas.dfo.one.org/datasets/v1/pin-request.json"
    assert (
        body["pin_response"]["$id"] == "https://schemas.dfo.one.org/datasets/v1/pin-response.json"
    )
    assert (
        body["operation_resource"]["$id"]
        == "https://schemas.dfo.one.org/datasets/v1/operation-resource.json"
    )


def test_schema_document_matches_the_live_pydantic_models() -> None:
    body = build_schema_document()

    assert body["mint_request"] == MintRequest.model_json_schema()
    assert body["manifest"] == Manifest.model_json_schema()
    assert body["pin_request"] == PinRequest.model_json_schema()
    assert body["pin_response"] == PinResponse.model_json_schema()
    assert body["operation_resource"] == OperationResource.model_json_schema()


def test_operation_not_found_when_the_door_lifespan_never_ran() -> None:
    """Without lifespan startup, `operation_store` is `None` and the poll route returns
    `operation_not_found`.
    """
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/operations/op_whatever")

    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "operation_not_found"
    assert body["retryable"] is False
    assert "op_whatever" in body["detail"]


def test_unknown_operation_id_against_a_real_store_refuses_operation_not_found() -> None:
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=_FixedOperationStore({}),
    )

    resp = client.get("/api/datasets/operations/op_never_created")

    assert resp.status_code == 404
    assert resp.json()["code"] == "operation_not_found"


def test_pending_operation_reports_progress() -> None:
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=_FixedOperationStore({"op_1": _operation(state="pending", rows_written=0)}),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"]["operation_id"] == "op_1"
    assert body["operation"]["status"] == "pending"
    assert body["operation"]["progress"] == {"rows_written": 0, "estimate_rows": 1_000_000}
    assert "manifest" not in body
    assert "error" not in body


def test_running_operation_reports_progress_against_the_estimate() -> None:
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=_FixedOperationStore(
            {"op_1": _operation(state="running", rows_written=250_000)}
        ),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"]["status"] == "running"
    assert body["operation"]["progress"] == {
        "rows_written": 250_000,
        "estimate_rows": 1_000_000,
    }


def test_failed_operation_refuses_operation_failed_at_the_registrys_own_status() -> None:
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=_FixedOperationStore(
            {"op_1": _operation(state="failed", refusal_code="excel_limit")}
        ),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["retryable"] is False
    assert "excel_limit" in body["detail"]


def test_failed_operation_poll_names_its_stored_refusal_code_and_retryable_flag() -> None:
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=_FixedOperationStore(
            {"op_1": _operation(state="failed", refusal_code="entities_unbound", retryable=True)}
        ),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["retryable"] is True
    assert body["cause"] == {"code": "entities_unbound", "entities": [], "truncated": False}


_STORED_REFUSALS = {
    "facet_not_available": ProblemSpec(
        status=422,
        code="facet_not_available",
        retryable=False,
        detail="member dcid:sv_a has no facet in the requested unit; choose one of the listed",
        pointer="/input/scope/facets",
        alternatives=[Alternative(ref="top", label="Top")],
        truncated=True,
        issues=[
            ProblemIssue(
                pointer="/input/scope/facets/0",
                member="dcid:sv_a",
                axis="unit",
                alternatives=[
                    Alternative(ref="USD", label="US dollars"),
                    Alternative(ref="EUR", label="Euros"),
                ],
                truncated=False,
            )
        ],
        constraint_id="unit_required",
        headers={"Retry-After": "1"},
    ),
    "over_ceiling": ProblemSpec(
        status=422,
        code="over_ceiling",
        retryable=False,
        detail="the candidate window was crossed",
        ceiling_kind="candidate_window",
        count=4000,
        cap=4000,
    ),
}


def _poll_failed(refusal: ProblemSpec | None, *, refusal_code: str):
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=_FixedOperationStore(
            {"op_1": _operation(state="failed", refusal_code=refusal_code, refusal=refusal)}
        ),
    )
    return client.get("/api/datasets/operations/op_1")


def test_failed_operation_poll_carries_the_stored_refusals_detail_pointer_and_issues() -> None:
    stored = _STORED_REFUSALS["facet_not_available"]

    resp = _poll_failed(stored, refusal_code="facet_not_available")

    # The outer status, code and retryable flag are the operation's own.
    assert resp.status_code == 500
    assert "retry-after" not in resp.headers
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["type"] == "/codes/operation_failed"
    assert body["status"] == 500
    assert body["retryable"] is False
    assert body["cause"] == {"code": "facet_not_available", "entities": [], "truncated": False}
    assert body["detail"] == stored.detail
    assert body["pointer"] == "/input/scope/facets"
    assert body["alternatives"] == [{"ref": "top", "label": "Top"}]
    assert body["truncated"] is True
    assert body["issues"] == [
        {
            "pointer": "/input/scope/facets/0",
            "member": "dcid:sv_a",
            "axis": "unit",
            "alternatives": [
                {"ref": "USD", "label": "US dollars"},
                {"ref": "EUR", "label": "Euros"},
            ],
            "truncated": False,
        }
    ]
    assert body["constraint_id"] == "unit_required"


def test_failed_operation_poll_carries_the_stored_refusals_ceiling_fields() -> None:
    stored = _STORED_REFUSALS["over_ceiling"]

    resp = _poll_failed(stored, refusal_code="over_ceiling")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["cause"]["code"] == "over_ceiling"
    assert body["detail"] == stored.detail
    assert (body["ceiling_kind"], body["count"], body["cap"]) == ("candidate_window", 4000, 4000)


def test_failed_operation_poll_keeps_the_stored_causes_entities_under_the_refusal_code() -> None:
    stored = ProblemSpec(
        status=422,
        code="rate_coverage_gap",
        retryable=False,
        detail="no rate covers two entities",
        cause=Cause(code="rate_missing", entities=["country/KEN@2020"], truncated=True),
    )

    resp = _poll_failed(stored, refusal_code="rate_coverage_gap")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["cause"] == {
        "code": "rate_coverage_gap",
        "entities": ["country/KEN@2020"],
        "truncated": True,
    }


@pytest.mark.parametrize("corruption", [{"severity": "high"}, {"count": "many"}])
def test_failed_operation_poll_with_an_unparseable_stored_refusal_names_the_code_alone(
    corruption: dict[str, object],
) -> None:
    fake = FakeOperationDatasetStore()
    try:
        operation_store = LiveOperationStore(store=fake, now=lambda: _OP_NOW)
        op = operation_store.create_or_get(
            request_identity={"subject_id": "spec_a"},
            options={"format": "parquet"},
            coverage_stamp="cov_stamp_1",
            origin_catalogue_build_id="fake-catalogue",
            estimate=_DEFAULT_ESTIMATE,
        )
        operation_store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        stored = _STORED_REFUSALS["over_ceiling"]
        operation_store.fail(
            op.id, attempt=1, refusal_code="over_ceiling", retryable=False, refusal=stored
        )
        with fake.transaction() as tx:
            tx.execute(
                "UPDATE datasets.operation SET refusal = %s WHERE id = %s",
                ({**stored.to_canonical_data(), **corruption}, op.id),
            )
        client = make_client(
            registry=LiveSpecRegistry(store=FakeDatasetStore()), operation_store=operation_store
        )

        resp = client.get(f"/api/datasets/operations/{op.id}")
    finally:
        fake.close()

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["cause"] == {"code": "over_ceiling", "entities": [], "truncated": False}
    assert body["detail"] == f"operation {op.id!r} did not complete successfully: over_ceiling"
    assert "ceiling_kind" not in body


def test_failed_operation_poll_with_no_stored_refusal_names_the_code_alone() -> None:
    resp = _poll_failed(None, refusal_code="facet_not_available")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["detail"] == "operation 'op_1' did not complete successfully: facet_not_available"
    assert body["cause"] == {"code": "facet_not_available", "entities": [], "truncated": False}
    assert not body.keys() & {"pointer", "alternatives", "issues", "constraint_id", "count"}


def test_failed_operation_poll_keeps_its_own_detail_when_the_stored_refusal_has_none() -> None:
    stored = ProblemSpec(status=404, code="spec_not_found", retryable=False)

    resp = _poll_failed(stored, refusal_code="spec_not_found")

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "operation 'op_1' did not complete successfully: spec_not_found"
    assert body["cause"]["code"] == "spec_not_found"


def test_done_operation_serves_the_manifest_with_a_freshly_signed_url_without_a_principal() -> None:
    artifact_store = FakeArtifactStore()
    location = _stored_rendering(artifact_store, _PARQUET_BUNDLE, data=b"a,b\n1,2\n")
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=_SITE_SETTINGS,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=FakeSnapshotReader(
            versions={("spec_a", "cv1"): _version()},
            locations={("spec_a", "cv1"): [location]},
        ),
        artifact_store=artifact_store,
        reference_writer=FakeArtifactReferenceWriter(),
        version_writer=NullVersionRecordWriter(),
        operation_store=_FixedOperationStore(
            {"op_1": _operation(state="done", subject_id="spec_a", content_version="cv1")}
        ),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"] == {"operation_id": "op_1", "status": "done"}
    assert body["manifest"]["content_version"] == "cv1"
    assert body["manifest"]["definition"]["spec_id"] == "spec_a"
    assert len(body["manifest"]["artifacts"]) == 1
    assert body["manifest"]["artifacts"][0]["artifact_digest"] == location.artifact_digest
    assert "error" not in body


# A site other than the test environment's, so a key derived under any other settings misses.
_SITE_SETTINGS = DatasetsBuildSettings(
    site=PublicSiteSettings(public_site_url="https://data.example.org")
)
_BARE_PARQUET = MintOptions(format="parquet", bundle=False)
_GROUPBY = MintOptions(
    format="parquet", bundle=False, groupby=GroupbyOptions(by=["stat_var"], op="sum")
)
_STATISTICS = {"statistics": "series"}
_MEDIA_TYPES = {"csv": "text/csv", "parquet": "application/vnd.apache.parquet"}


def _stored_rendering(
    artifact_store: FakeArtifactStore,
    options: MintOptions,
    *,
    data: bytes,
) -> StoredArtifactLocation:
    """A pinned location (so retention against the real clock never applies) stored under the
    rendering and key the mint links `options` under."""
    rendering = artifact_rendering(options, site=_SITE_SETTINGS.site)
    return _store_and_locate(
        artifact_store,
        data,
        tier=Tier.pinned,
        rendering=rendering,
        rendering_key=rendering_key(rendering),
        media_type="application/zip" if options.bundle else _MEDIA_TYPES[options.format],
    )


def _stored_statistics(artifact_store: FakeArtifactStore, *, data: bytes) -> StoredArtifactLocation:
    return _store_and_locate(
        artifact_store,
        data,
        tier=Tier.pinned,
        rendering=_STATISTICS,
        rendering_key=rendering_key(_STATISTICS),
        media_type="application/vnd.apache.parquet",
    )


@pytest.mark.parametrize(
    "renderings",
    [
        pytest.param((_BARE_PARQUET, _GROUPBY), id="bare-parquet-vs-groupby"),
        pytest.param((MintOptions(), _GROUPBY), id="bundle-vs-groupby"),
    ],
)
def test_a_done_operation_lists_only_its_own_rendering_and_statistics(
    renderings: tuple[MintOptions, ...],
) -> None:
    """#1172: two renderings and statistics stored under one `(subject_id, content_version)`.
    Each done operation's poll lists the artifacts of the rendering its request posted plus
    statistics, the list the synchronous mint answers with."""
    artifact_store = FakeArtifactStore()
    own: dict[int, set[str]] = {}
    locations: list[StoredArtifactLocation] = []
    for index, options in enumerate(renderings):
        stored = _stored_rendering(artifact_store, options, data=f"{index}".encode())
        own[index] = {stored.artifact_digest}
        locations.append(stored)
    statistics = _stored_statistics(artifact_store, data=b"stats")
    locations.append(statistics)
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=_SITE_SETTINGS,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=FakeSnapshotReader(
            versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): locations}
        ),
        artifact_store=artifact_store,
        reference_writer=FakeArtifactReferenceWriter(),
        version_writer=NullVersionRecordWriter(),
        operation_store=_FixedOperationStore(
            {
                f"op_{index}": _operation(
                    id=f"op_{index}",
                    state="done",
                    subject_id="spec_a",
                    content_version="cv1",
                    options=options,
                )
                for index, options in enumerate(renderings)
            }
        ),
    )

    for index in own:
        resp = client.get(f"/api/datasets/operations/op_{index}")

        assert resp.status_code == 200, resp.json()
        listed = {artifact["artifact_digest"] for artifact in resp.json()["manifest"]["artifacts"]}
        assert listed == own[index] | {statistics.artifact_digest}


def test_a_done_operation_whose_rendering_is_gone_fails_retryable_and_a_resubmit_resets_it() -> (
    None
):
    """Only another rendering is stored under the operation's version. The poll fails the row
    retryable `internal_error` and answers `operation_failed`, and a resubmit resets the row."""
    artifact_store = FakeArtifactStore()
    locations = [
        _stored_rendering(artifact_store, _GROUPBY, data=b"groupby"),
        _stored_statistics(artifact_store, data=b"stats"),
    ]
    operation_store = LiveOperationStore(store=FakeOperationDatasetStore(), now=lambda: _OP_NOW)
    submission = {
        "request_identity": {"input": {"spec_id": "spec_a"}},
        "options": _BARE_PARQUET.model_dump(mode="json"),
        "coverage_stamp": "cov_stamp_1",
        "origin_catalogue_build_id": "fake-catalogue",
        "estimate": _DEFAULT_ESTIMATE,
    }
    op = operation_store.create_or_get(**submission)
    assert operation_store.claim_lease(op.id, submission_count=0, lease_seconds=60) is not None
    assert operation_store.complete(op.id, attempt=1, subject_id="spec_a", content_version="cv1")
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=_SITE_SETTINGS,
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=FakeSnapshotReader(
            versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): locations}
        ),
        artifact_store=artifact_store,
        reference_writer=FakeArtifactReferenceWriter(),
        version_writer=NullVersionRecordWriter(),
        operation_store=operation_store,
    )

    resp = client.get(f"/api/datasets/operations/{op.id}")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "operation_failed"
    assert body["retryable"] is True
    assert body["cause"]["code"] == "internal_error"
    failed = operation_store.get(op.id)
    assert failed is not None
    assert (failed.state, failed.refusal_code, failed.retryable) == (
        "failed",
        "internal_error",
        True,
    )

    resubmitted = operation_store.create_or_get(**submission)

    assert resubmitted.id == op.id
    assert resubmitted.state == "pending"
    assert resubmitted.submission_count == 1


class _RacedOperationStore(_FixedOperationStore):
    """Reads a done row first, then `current`, as when a concurrent poll or resubmit moves the
    row before this poll's `fail_unservable_done`, which therefore returns `None`."""

    def __init__(self, done: Operation, current: Operation | None) -> None:
        super().__init__({})
        self._reads = [done, current]

    def get(self, id: str) -> Operation | None:
        return self._reads.pop(0)


@pytest.mark.parametrize(
    ("current", "status", "refusal_code"),
    [
        pytest.param(
            _operation(state="failed", refusal_code="upstream_unavailable", retryable=True),
            None,
            "upstream_unavailable",
            id="failed",
        ),
        pytest.param(_operation(state="pending"), "pending", None, id="pending"),
        pytest.param(_operation(state="running"), "running", None, id="running"),
        pytest.param(None, None, None, id="missing"),
        pytest.param(
            _operation(state="done", subject_id="spec_a", content_version="cv2"),
            None,
            "internal_error",
            id="done-again",
        ),
    ],
)
def test_a_poll_that_loses_the_demotion_race_answers_from_the_rows_current_state(
    current: Operation | None, status: str | None, refusal_code: str | None
) -> None:
    artifact_store = FakeArtifactStore()
    locations = [_stored_rendering(artifact_store, _GROUPBY, data=b"groupby")]
    done = _operation(
        state="done", subject_id="spec_a", content_version="cv1", options=_BARE_PARQUET
    )

    def _poll() -> OperationResource:
        return _resolve_operation(
            operation_id=done.id,
            operation_store=_RacedOperationStore(done, current),
            task_client=None,
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=FakeSnapshotReader(
                versions={("spec_a", "cv1"): _version()}, locations={("spec_a", "cv1"): locations}
            ),
            artifact_store=artifact_store,
            now=lambda: _OP_NOW,
            settings=_SITE_SETTINGS,
        )

    if current is None:
        with pytest.raises(OperationNotFound):
            _poll()
    elif status is not None:
        assert _poll().operation.status == status
    else:
        with pytest.raises(OperationFailed) as excinfo:
            _poll()
        assert (excinfo.value.refusal_code, excinfo.value.retryable) == (refusal_code, True)


def _stale_operation_store(now_box: dict[str, datetime]) -> LiveOperationStore:
    return LiveOperationStore(store=FakeOperationDatasetStore(), now=lambda: now_box["t"])


class TestDeadWorkerDetection:
    """Tests stale-operation expiry with a fake clock.
    Claimed rows age from `lease_until`. Pending rows without a lease use `created_at`.
    """

    def test_persists_failed_for_a_running_operation_whose_lease_expired_past_the_window(
        self,
    ) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = operation_store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options={"format": "csv"},
            coverage_stamp="cov_stamp_1",
            origin_catalogue_build_id="fake-catalogue",
            estimate=Estimate(rows=1_000_000, requests=40),
        )
        claimed = operation_store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        assert claimed is not None and claimed.lease_until is not None
        # Expiry starts at `lease_until`, the end of the lease.
        now_box["t"] = claimed.lease_until + timedelta(seconds=_STALE_AFTER_SECONDS + 1)

        with pytest.raises(OperationFailed) as excinfo:
            _resolve_operation(
                operation_id=op.id,
                operation_store=operation_store,
                task_client=None,
                spec_reader=None,
                snapshot_reader=None,
                artifact_store=None,
                now=lambda: now_box["t"],
                settings=DatasetsBuildSettings(),
            )
        assert excinfo.value.refusal_code == "internal_error"

        # The poll persists the failure in the operation store.
        stored = operation_store.get(op.id)
        assert stored is not None
        assert stored.state == "failed"
        assert stored.refusal_code == "internal_error"

    def test_reports_running_just_under_the_stale_window_past_lease_expiry(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = operation_store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options={"format": "csv"},
            coverage_stamp="cov_stamp_1",
            origin_catalogue_build_id="fake-catalogue",
            estimate=Estimate(rows=1_000_000, requests=40),
        )
        claimed = operation_store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        assert claimed is not None and claimed.lease_until is not None
        now_box["t"] = claimed.lease_until + timedelta(seconds=_STALE_AFTER_SECONDS - 1)

        resource = _resolve_operation(
            operation_id=op.id,
            operation_store=operation_store,
            task_client=None,
            spec_reader=None,
            snapshot_reader=None,
            artifact_store=None,
            now=lambda: now_box["t"],
            settings=DatasetsBuildSettings(),
        )

        assert resource.operation.status == "running"
        assert operation_store.get(op.id).state == "running"

    def test_never_claimed_pending_operation_does_not_expire_at_the_claimed_row_threshold(
        self,
    ) -> None:
        """A never-claimed row stays pending past the claimed-row threshold."""
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = operation_store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options={"format": "csv"},
            coverage_stamp="cov_stamp_1",
            origin_catalogue_build_id="fake-catalogue",
            estimate=Estimate(rows=1_000_000, requests=40),
        )
        now_box["t"] = _OP_NOW + timedelta(seconds=_STALE_AFTER_SECONDS + 1)

        resource = _resolve_operation(
            operation_id=op.id,
            operation_store=operation_store,
            task_client=None,
            spec_reader=None,
            snapshot_reader=None,
            artifact_store=None,
            now=lambda: now_box["t"],
            settings=DatasetsBuildSettings(),
        )

        assert resource.operation.status == "pending"
        assert operation_store.get(op.id).state == "pending"

    def test_never_claimed_pending_operation_expires_past_its_own_longer_threshold(self) -> None:
        """A never-claimed pending row expires after its `created_at` threshold."""
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = operation_store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options={"format": "csv"},
            coverage_stamp="cov_stamp_1",
            origin_catalogue_build_id="fake-catalogue",
            estimate=Estimate(rows=1_000_000, requests=40),
        )
        now_box["t"] = _OP_NOW + timedelta(seconds=_NEVER_CLAIMED_STALE_AFTER_SECONDS + 1)

        with pytest.raises(OperationFailed) as excinfo:
            _resolve_operation(
                operation_id=op.id,
                operation_store=operation_store,
                task_client=None,
                spec_reader=None,
                snapshot_reader=None,
                artifact_store=None,
                now=lambda: now_box["t"],
                settings=DatasetsBuildSettings(),
            )
        assert excinfo.value.refusal_code == "internal_error"

        stored = operation_store.get(op.id)
        assert stored is not None
        assert stored.state == "failed"
        assert stored.refusal_code == "internal_error"
        assert stored.retryable is True

    def test_released_pending_operation_whose_lease_expired_past_the_window_expires(
        self,
    ) -> None:
        """Release sets `lease_until` to now, so the claimed-row expiry rule still applies."""
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = operation_store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options={"format": "csv"},
            coverage_stamp="cov_stamp_1",
            origin_catalogue_build_id="fake-catalogue",
            estimate=Estimate(rows=1_000_000, requests=40),
        )
        claimed = operation_store.claim_lease(op.id, submission_count=0, lease_seconds=60)
        assert claimed is not None
        released = operation_store.release(op.id, attempt=1)
        assert released is not None and released.state == "pending"
        assert released.lease_until is not None
        now_box["t"] = released.lease_until + timedelta(seconds=_STALE_AFTER_SECONDS + 1)

        with pytest.raises(OperationFailed) as excinfo:
            _resolve_operation(
                operation_id=op.id,
                operation_store=operation_store,
                task_client=None,
                spec_reader=None,
                snapshot_reader=None,
                artifact_store=None,
                now=lambda: now_box["t"],
                settings=DatasetsBuildSettings(),
            )
        assert excinfo.value.refusal_code == "internal_error"

        stored = operation_store.get(op.id)
        assert stored is not None
        assert stored.state == "failed"
        assert stored.refusal_code == "internal_error"


def test_capabilities_says_the_operation_envelope_is_active() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/capabilities")

    body = resp.json()
    assert body["operation_envelope_active"] is True
    assert body["caps"]["operation_retention_days"] > 0


def test_schema_route_carries_cache_headers() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/schema", headers={"accept-encoding": "identity"})

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=300"
    assert resp.headers["vary"] == "Accept-Encoding, Origin"
    assert "etag" in resp.headers


def test_schema_route_serves_gzip_when_accepted() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/schema", headers={"accept-encoding": "gzip"})

    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "gzip"
    # httpx decodes `.content`. The `content-encoding` header confirms the gzip wire response.
    assert resp.content == JSONResponse(build_schema_document()).body


def test_schema_route_304s_on_if_none_match() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    etag = client.get("/api/datasets/schema", headers={"accept-encoding": "identity"}).headers[
        "etag"
    ]

    resp = client.get(
        "/api/datasets/schema",
        headers={"accept-encoding": "identity", "if-none-match": etag},
    )

    assert resp.status_code == 304
    assert resp.content == b""


def test_capabilities_route_carries_cache_headers() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.get("/api/datasets/capabilities", headers={"accept-encoding": "identity"})

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=300"
    assert resp.headers["vary"] == "Accept-Encoding, Origin"
    assert "etag" in resp.headers


def test_capabilities_route_serves_gzip_when_accepted() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    expected = JSONResponse(
        build_capabilities(
            capability_version=DATASETS_CAPABILITY_VERSION,
            build=build_commit(),
            code_index_stamp=get_index().stamp,
            declarations=get_index().declarations,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
    ).body

    resp = client.get("/api/datasets/capabilities", headers={"accept-encoding": "gzip"})

    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.content == expected


def test_capabilities_route_304s_on_if_none_match() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    etag = client.get(
        "/api/datasets/capabilities", headers={"accept-encoding": "identity"}
    ).headers["etag"]

    resp = client.get(
        "/api/datasets/capabilities",
        headers={"accept-encoding": "identity", "if-none-match": etag},
    )

    assert resp.status_code == 304
    assert resp.content == b""


def test_capabilities_route_identity_bytes_match_the_builder() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    expected = JSONResponse(
        build_capabilities(
            capability_version=DATASETS_CAPABILITY_VERSION,
            build=build_commit(),
            code_index_stamp=get_index().stamp,
            declarations=get_index().declarations,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
    ).body

    resp = client.get("/api/datasets/capabilities", headers={"accept-encoding": "identity"})

    assert resp.content == expected


def test_schema_document_not_built_when_the_route_is_gated_off(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "datasets_api.doors.meta.build_schema_document",
        lambda: calls.append(1) or {},
    )
    monkeypatch.setattr("datasets_api.doors.meta.SCHEMA_ROUTE_LIVE", False)

    build_meta_door()

    assert calls == []


class _RaisingTaskClient:
    def __init__(self) -> None:
        self.calls = 0

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        self.calls += 1
        raise TimeoutError("create_task deadline exceeded")


def _poll(operation_store: LiveOperationStore, operation_id: str, task_client) -> OperationResource:
    return _resolve_operation(
        operation_id=operation_id,
        operation_store=operation_store,
        task_client=task_client,
        spec_reader=None,
        snapshot_reader=None,
        artifact_store=None,
        now=lambda: _OP_NOW,
        settings=DatasetsBuildSettings(),
    )


class TestPollDispatchReconciliation:
    """Polling dispatches due pending operations through `dispatch_if_due`."""

    def _operation(self, operation_store: LiveOperationStore):
        return operation_store.create_or_get(
            request_identity={"input": {"spec_id": "spec_a"}},
            options={"format": "csv"},
            coverage_stamp="cov_stamp_1",
            origin_catalogue_build_id="fake-catalogue",
            estimate=Estimate(rows=1_000_000, requests=40),
        )

    def test_a_poll_redispatches_a_lost_enqueue_after_the_grace(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = self._operation(operation_store)
        assert operation_store.claim_dispatch(op.id) is not None  # its enqueue was lost
        task_client = FakeTaskClient()

        inside = _poll(operation_store, op.id, task_client)
        assert inside.operation.status == "pending"
        assert task_client.enqueued == []

        now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
        resource = _poll(operation_store, op.id, task_client)

        assert resource.operation.status == "pending"
        assert task_client.enqueued == [(op.id, 0)]
        stored = operation_store.get(op.id)
        assert stored is not None
        assert stored.dispatch_count == 2
        assert stored.dispatch_confirmed == 2

    def test_a_poll_dispatches_a_row_that_was_never_dispatched(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = self._operation(operation_store)
        task_client = FakeTaskClient()

        _poll(operation_store, op.id, task_client)

        assert task_client.enqueued == [(op.id, 0)]

    def test_a_poll_never_redispatches_a_confirmed_dispatch(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = self._operation(operation_store)
        task_client = FakeTaskClient()
        _poll(operation_store, op.id, task_client)
        assert task_client.enqueued == [(op.id, 0)]

        now_box["t"] = now_box["t"] + timedelta(hours=2)
        _poll(operation_store, op.id, task_client)

        assert task_client.enqueued == [(op.id, 0)]

    def test_an_enqueue_failure_on_a_poll_still_answers_the_poll(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = self._operation(operation_store)
        task_client = _RaisingTaskClient()

        resource = _poll(operation_store, op.id, task_client)

        assert task_client.calls == 1
        assert resource.operation.status == "pending"
        stored = operation_store.get(op.id)
        assert stored is not None
        assert stored.dispatch_count == 1
        assert stored.dispatch_confirmed == 0

    def test_a_submission_that_used_every_dispatch_fails_once_its_last_task_is_stale(
        self,
    ) -> None:
        """After the dispatch cap, polls wait for the stale threshold from the last dispatch."""
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = self._operation(operation_store)
        task_client = _RaisingTaskClient()
        for _ in range(MAX_REDISPATCHES + 1):
            _poll(operation_store, op.id, task_client)
            now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
        assert task_client.calls == MAX_REDISPATCHES + 1
        exhausted = operation_store.get(op.id)
        assert exhausted is not None and exhausted.dispatched_at is not None
        last_dispatched_at = exhausted.dispatched_at

        now_box["t"] = last_dispatched_at + timedelta(seconds=_NEVER_CLAIMED_STALE_AFTER_SECONDS)
        resource = _poll(operation_store, op.id, task_client)
        assert resource.operation.status == "pending"
        assert task_client.calls == MAX_REDISPATCHES + 1

        now_box["t"] = now_box["t"] + timedelta(seconds=1)
        with pytest.raises(OperationFailed) as excinfo:
            _poll(operation_store, op.id, task_client)

        assert excinfo.value.refusal_code == "internal_error"
        assert task_client.calls == MAX_REDISPATCHES + 1

    def test_a_running_operation_is_never_dispatched_by_a_poll(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = self._operation(operation_store)
        assert operation_store.claim_lease(op.id, submission_count=0, lease_seconds=600)
        task_client = FakeTaskClient()

        resource = _poll(operation_store, op.id, task_client)

        assert resource.operation.status == "running"
        assert task_client.enqueued == []


def test_the_composed_poll_route_reconciles_through_the_surfaces_task_client() -> None:
    operation_store = LiveOperationStore(store=FakeOperationDatasetStore())
    op = operation_store.create_or_get(
        request_identity={"input": {"spec_id": "spec_a"}},
        options={"format": "csv"},
        coverage_stamp="cov_stamp_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=1_000_000, requests=40),
    )
    task_client = FakeTaskClient()
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=operation_store,
        task_client=task_client,
    )

    resp = client.get(f"/api/datasets/operations/{op.id}")

    assert resp.status_code == 200
    assert resp.json()["operation"]["status"] == "pending"
    assert task_client.enqueued == [(op.id, 0)]
