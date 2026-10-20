"""Tests health, schema, capabilities, and operation routes through the composed app.
Schema and capability checks cover document validity, stable ids, model agreement, and caching.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import jsonschema
import pytest
from fastapi.responses import JSONResponse

from artifact_store import Tier
from code_index.shared import get_index
from dataset_store.snapshot import Revocation, StoredArtifactLocation
from datasets_api.capabilities import build_capabilities
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.meta import (
    OperationFailed,
    OperationNotFound,
    _resolve_operation,
    build_meta_door,
)
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.operation_store import LiveOperationStore, Operation
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS
from datasets_api.operations.lease import LEASE_SECONDS
from datasets_api.operations.submit import (
    OPERATION_POLL_MAX_MS,
    OPERATION_SUBMIT_RETRY_AFTER_MS,
    poll_interval_ms,
    retry_after_headers,
)
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.render import artifact_rendering
from datasets_api.schema import (
    Manifest,
    MintRequest,
    OperationEnvelope,
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
from spec_registry import IdentityQuarantined, LiveSpecRegistry

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._operation_store import FakeOperationDatasetStore
from ._snapshot_doubles import FakeSnapshotReader, FakeSpecReader, _store_and_locate, _version
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_CONFIRMED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_OP_NOW = datetime(2026, 6, 1, tzinfo=UTC)
_DEFAULT_ESTIMATE = Estimate(rows=1_000_000, requests=40)
# `bundle` defaults to true, so a plain parquet mint is a bundle.
_PARQUET_BUNDLE = MintOptions(format="parquet")


class _FixedOperationStore:
    """Returns operations from a fixed map for route tests."""

    def __init__(self, operations: dict[str, Operation]) -> None:
        self._operations = operations

    def create_or_get(self, **kwargs):
        raise NotImplementedError

    def mark_dispatched(self, id: str, *, submission_count: int):
        return None

    def claim(self, id: str, *, submission_count: int, lease_seconds: int, final_delivery: bool):
        return None

    def progress(self, id: str, *, attempt: int, rows_written: int):
        return None

    def record_estimate(self, id: str, *, attempt: int, estimate: Estimate):
        return None

    def complete(
        self, id: str, *, attempt: int, subject_id: str, content_version: str, confirmed_at
    ):
        return None

    def fail(self, id: str, *, attempt: int, refusal_code: str, retryable: bool, refusal=None):
        return None

    def release(self, id: str, *, attempt: int):
        return None

    def abandon(self, id: str, *, attempt: int):
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
    confirmed_at: datetime | None = None,
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
        claimed_at=_OP_NOW if state == "running" else None,
        lease_seconds=1_500 if state == "running" else None,
        created_at=created_at,
        updated_at=_OP_NOW,
        submission_count=0,
        retryable=retryable,
        dispatched_at=created_at,
        submitted_at=created_at,
        origin_catalogue_build_id="fake-catalogue",
        refusal=refusal,
        confirmed_at=confirmed_at,
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
    assert body["manifest"] == Manifest.model_json_schema(mode="serialization")
    assert body["pin_request"] == PinRequest.model_json_schema()
    assert body["pin_response"] == PinResponse.model_json_schema(mode="serialization")
    assert body["operation_resource"] == OperationResource.model_json_schema(mode="serialization")


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


@pytest.mark.parametrize("state", ["pending", "running"])
def test_an_unfinished_poll_paces_the_caller_in_the_body_and_the_header(state: str) -> None:
    """The row is months old, so the pacing sits at its cap."""
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        operation_store=_FixedOperationStore({"op_1": _operation(state=state)}),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 200
    assert resp.json()["operation"]["retry_after_ms"] == OPERATION_POLL_MAX_MS
    assert resp.headers["retry-after"] == str(OPERATION_POLL_MAX_MS // 1000)


@pytest.mark.parametrize("state", ["pending", "running"])
def test_an_unfinished_poll_paces_by_the_age_of_the_operation(state: str) -> None:
    operation_store = _FixedOperationStore({"op_1": _operation(state=state)})
    now_box = {"t": _OP_NOW + timedelta(seconds=20)}

    resource = _resolve_operation(
        operation_id="op_1",
        operation_store=operation_store,
        task_client=None,
        spec_reader=None,
        snapshot_reader=None,
        now=lambda: now_box["t"],
        settings=DatasetsBuildSettings(),
    )

    assert resource.operation.retry_after_ms == 5_000


@pytest.mark.parametrize(
    ("age_seconds", "expected_ms"),
    [
        pytest.param(-5, OPERATION_SUBMIT_RETRY_AFTER_MS, id="clock-skew-floors"),
        pytest.param(0, OPERATION_SUBMIT_RETRY_AFTER_MS, id="new-floors"),
        pytest.param(6, OPERATION_SUBMIT_RETRY_AFTER_MS, id="floor-edge"),
        pytest.param(8, 2_000, id="just-above-the-floor"),
        pytest.param(20, 5_000, id="a-quarter-of-the-age"),
        pytest.param(32, OPERATION_POLL_MAX_MS, id="cap-edge"),
        pytest.param(3_600, OPERATION_POLL_MAX_MS, id="old-caps"),
    ],
)
def test_poll_interval_is_a_quarter_of_the_age_between_the_floor_and_the_cap(
    age_seconds: int, expected_ms: int
) -> None:
    operation = _operation(state="running")

    interval = poll_interval_ms(operation, now=_OP_NOW + timedelta(seconds=age_seconds))

    assert interval == expected_ms


def test_a_done_operation_is_paced_at_the_floor_whatever_its_age() -> None:
    operation = _operation(state="done")

    interval = poll_interval_ms(operation, now=_OP_NOW + timedelta(hours=1))

    assert interval == OPERATION_SUBMIT_RETRY_AFTER_MS


@pytest.mark.parametrize(
    ("retry_after_ms", "expected"),
    [
        pytest.param(1_500, {"Retry-After": "2"}, id="rounds-up"),
        pytest.param(2_000, {"Retry-After": "2"}, id="whole-second-stays"),
        pytest.param(2_001, {"Retry-After": "3"}, id="one-millisecond-over-rounds-up"),
        pytest.param(8_000, {"Retry-After": "8"}, id="cap"),
        pytest.param(None, {}, id="no-pacing-no-header"),
    ],
)
def test_retry_after_header_is_whole_seconds_rounded_up(
    retry_after_ms: int | None, expected: dict[str, str]
) -> None:
    envelope = OperationEnvelope(
        operation_id="op_1", status="pending", retry_after_ms=retry_after_ms
    )

    assert retry_after_headers(envelope) == expected


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
        operation_store.claim(op.id, submission_count=0, lease_seconds=1_500, final_delivery=False)
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


@pytest.mark.parametrize(
    ("confirmed_at", "currency"),
    [
        pytest.param(
            _CONFIRMED_AT,
            {"served_from": "materialise", "confirmed_at": "2026-01-01T00:00:00+00:00"},
            id="confirmed",
        ),
        pytest.param(None, {}, id="completed-by-an-image-recording-no-instant"),
    ],
)
def test_done_operation_serves_the_manifest_with_a_freshly_signed_url_without_a_principal(
    confirmed_at: datetime | None, currency: dict[str, str]
) -> None:
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
            {
                "op_1": _operation(
                    state="done",
                    subject_id="spec_a",
                    content_version="cv1",
                    confirmed_at=confirmed_at,
                )
            }
        ),
    )

    resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"] == {"operation_id": "op_1", "status": "done"}
    assert "retry-after" not in resp.headers
    trace = body["manifest"]["trace"]
    assert {key: trace[key] for key in ("served_from", "confirmed_at") if key in trace} == currency
    assert body["manifest"]["content_version"] == "cv1"
    assert body["manifest"]["definition"]["spec_id"] == "spec_a"
    assert len(body["manifest"]["artifacts"]) == 1
    assert body["manifest"]["artifacts"][0]["artifact_digest"] == location.artifact_digest
    assert "error" not in body


class _QuarantinedSpecReader(FakeSpecReader):
    def read_definition(self, spec_id: str, *, schema_version: int | None = None):
        raise IdentityQuarantined(subject_id=spec_id, id_kind="spec")


@pytest.mark.parametrize(
    "failure", ["spec_not_found", "snapshot_not_found", "snapshot_revoked", "identity_quarantined"]
)
def test_a_done_operation_whose_manifest_cannot_be_read_answers_internal_error(
    failure: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Each snapshot failure of a done operation's subject answers the route's
    `internal_error`, classified by the door without the platform's fallback."""
    spec_reader = {
        "spec_not_found": FakeSpecReader(),
        "identity_quarantined": _QuarantinedSpecReader(),
    }.get(failure, FakeSpecReader(spec_ids=["spec_a"]))
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=_SITE_SETTINGS,
        spec_reader=spec_reader,
        snapshot_reader=FakeSnapshotReader(
            versions={} if failure == "snapshot_not_found" else {("spec_a", "cv1"): _version()},
            revocations=(
                {("spec_a", "cv1"): Revocation(reason=None, revoked_at=_OP_NOW)}
                if failure == "snapshot_revoked"
                else {}
            ),
        ),
        artifact_store=FakeArtifactStore(),
        reference_writer=FakeArtifactReferenceWriter(),
        version_writer=NullVersionRecordWriter(),
        operation_store=_FixedOperationStore(
            {"op_1": _operation(state="done", subject_id="spec_a", content_version="cv1")}
        ),
    )

    with caplog.at_level(logging.ERROR):
        resp = client.get("/api/datasets/operations/op_1")

    assert resp.status_code == 500, resp.text
    assert resp.json()["code"] == "internal_error"
    assert not [r for r in caplog.records if r.funcName == "_classify_and_envelope"]


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
    assert (
        operation_store.claim(op.id, submission_count=0, lease_seconds=1_500, final_delivery=False)
        is not None
    )
    assert operation_store.complete(
        op.id, attempt=1, subject_id="spec_a", content_version="cv1", confirmed_at=_CONFIRMED_AT
    )
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


def _create(operation_store: LiveOperationStore) -> Operation:
    return operation_store.create_or_get(
        request_identity={"input": {"spec_id": "spec_a"}},
        options={"format": "csv"},
        coverage_stamp="cov_stamp_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=1_000_000, requests=40),
    )


def _poll_at(operation_store: LiveOperationStore, operation_id: str, at: datetime):
    return _resolve_operation(
        operation_id=operation_id,
        operation_store=operation_store,
        task_client=None,
        spec_reader=None,
        snapshot_reader=None,
        now=lambda: at,
        settings=DatasetsBuildSettings(),
    )


class TestAbandonedRun:
    """A poll fails a row nothing will finish: a running row whose lease ended with no later
    delivery due, and any unfinished row past the operation deadline."""

    def test_a_final_delivery_s_row_past_its_lease_fails_retryable_and_persists(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        assert operation_store.claim(
            op.id, submission_count=0, lease_seconds=60, final_delivery=True
        )
        now_box["t"] = _OP_NOW + timedelta(seconds=61)

        with pytest.raises(OperationFailed) as excinfo:
            _poll_at(operation_store, op.id, now_box["t"])

        assert (excinfo.value.refusal_code, excinfo.value.retryable) == ("internal_error", True)
        stored = operation_store.get(op.id)
        assert stored is not None
        assert (stored.state, stored.refusal_code, stored.retryable) == (
            "failed",
            "internal_error",
            True,
        )

    def test_a_running_row_at_its_lease_end_still_reports_running(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        assert operation_store.claim(
            op.id, submission_count=0, lease_seconds=60, final_delivery=False
        )
        now_box["t"] = _OP_NOW + timedelta(seconds=60)

        assert _poll_at(operation_store, op.id, now_box["t"]).operation.status == "running"
        assert operation_store.get(op.id).state == "running"

    def test_a_poll_past_the_lease_leaves_the_row_to_the_delivery_still_due(self) -> None:
        """The first delivery dies holding its claim. The queue's retry lands 1,680 s after it,
        past the 1,620 s lease, and a poll between the two leaves the row for it."""
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        assert operation_store.mark_dispatched(op.id, submission_count=0)
        claimed = operation_store.claim(
            op.id, submission_count=0, lease_seconds=LEASE_SECONDS, final_delivery=False
        )
        assert claimed is not None

        now_box["t"] = _OP_NOW + timedelta(seconds=LEASE_SECONDS + 1)
        assert _poll_at(operation_store, op.id, now_box["t"]).operation.status == "running"
        assert operation_store.get(op.id) == claimed

        now_box["t"] = _OP_NOW + timedelta(seconds=1_680)
        retried = operation_store.claim(
            op.id, submission_count=0, lease_seconds=LEASE_SECONDS, final_delivery=False
        )
        assert retried is not None
        assert (retried.state, retried.attempt_count) == ("running", 2)

    @pytest.mark.parametrize(
        "history", ["never-claimed", "released", "never-dispatched", "claimed-delivery-due"]
    )
    @pytest.mark.parametrize("past_deadline", [0, 1])
    def test_an_unfinished_row_past_the_operation_deadline_fails_retryable(
        self, history: str, past_deadline: int
    ) -> None:
        """The deadline runs from the dispatch, or from the submission when the task create
        never returned. It ends the wait of a pending row and of a running row whose lease
        ended with a later delivery due."""
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        if history != "never-dispatched":
            assert operation_store.mark_dispatched(op.id, submission_count=0)
        if history in ("released", "claimed-delivery-due"):
            assert operation_store.claim(
                op.id, submission_count=0, lease_seconds=60, final_delivery=False
            )
        if history == "released":
            assert operation_store.release(op.id, attempt=1)
        waiting = "running" if history == "claimed-delivery-due" else "pending"
        status = "failed" if past_deadline else waiting
        deadline = DatasetsBuildSettings().operation_deadline_seconds
        now_box["t"] = _OP_NOW + timedelta(seconds=deadline + past_deadline)

        if status == waiting:
            assert _poll_at(operation_store, op.id, now_box["t"]).operation.status == waiting
        else:
            with pytest.raises(OperationFailed) as excinfo:
                _poll_at(operation_store, op.id, now_box["t"])
            assert (excinfo.value.refusal_code, excinfo.value.retryable) == (
                "internal_error",
                True,
            )
        stored = operation_store.get(op.id)
        assert stored is not None and stored.state == status

    @pytest.mark.parametrize(("idle", "status"), [(0, "running"), (1, "failed")])
    def test_a_running_row_with_no_claim_time_ends_its_lease_when_its_heartbeat_stops(
        self, idle: int, status: str
    ) -> None:
        """An image that renewed its lease through `updated_at` and recorded no delivery
        claimed the row, so the row is abandoned once `updated_at` is `LEASE_SECONDS` old."""
        now_box = {"t": _OP_NOW}
        fake = FakeOperationDatasetStore()
        operation_store = LiveOperationStore(store=fake, now=lambda: now_box["t"])
        op = _create(operation_store)
        assert operation_store.claim(
            op.id, submission_count=0, lease_seconds=60, final_delivery=False
        )
        with fake.transaction() as tx:
            tx.execute(
                "UPDATE datasets.operation SET claimed_at = NULL, lease_seconds = NULL, "
                "final_delivery = NULL WHERE id = %s",
                (op.id,),
            )
        now_box["t"] = _OP_NOW + timedelta(seconds=LEASE_SECONDS + idle)

        if status == "running":
            assert _poll_at(operation_store, op.id, now_box["t"]).operation.status == "running"
        else:
            with pytest.raises(OperationFailed):
                _poll_at(operation_store, op.id, now_box["t"])
        stored = operation_store.get(op.id)
        assert stored is not None and stored.state == status


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

    build_meta_door(deps=fake_deps_cell(), settings=DatasetsBuildSettings.from_env())

    assert calls == []


class _RaisingTaskClient:
    def __init__(self) -> None:
        self.calls = 0

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        self.calls += 1
        raise TimeoutError("create_task deadline exceeded")


def _poll(
    operation_store: LiveOperationStore,
    operation_id: str,
    task_client,
    now_box: dict[str, datetime],
) -> OperationResource:
    return _resolve_operation(
        operation_id=operation_id,
        operation_store=operation_store,
        task_client=task_client,
        spec_reader=None,
        snapshot_reader=None,
        now=lambda: now_box["t"],
        settings=DatasetsBuildSettings(),
    )


class TestPollEnqueueRecovery:
    """A poll creates the task for an undispatched pending row past the recovery window."""

    def test_a_poll_recovers_an_undispatched_row_only_past_the_window(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        task_client = FakeTaskClient()

        now_box["t"] = _OP_NOW + timedelta(seconds=ENQUEUE_RECOVERY_SECONDS)
        assert _poll(operation_store, op.id, task_client, now_box).operation.status == "pending"
        assert task_client.enqueued == []

        now_box["t"] += timedelta(seconds=1)
        assert _poll(operation_store, op.id, task_client, now_box).operation.status == "pending"
        assert task_client.enqueued == [(op.id, 0)]
        stored = operation_store.get(op.id)
        assert stored is not None and stored.dispatched_at == now_box["t"]

    def test_a_poll_never_recreates_a_recorded_dispatch(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        assert operation_store.mark_dispatched(op.id, submission_count=0) is not None
        task_client = FakeTaskClient()

        now_box["t"] += timedelta(hours=2)
        _poll(operation_store, op.id, task_client, now_box)

        assert task_client.enqueued == []

    def test_an_enqueue_failure_on_a_poll_still_answers_the_poll(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        task_client = _RaisingTaskClient()
        now_box["t"] += timedelta(seconds=ENQUEUE_RECOVERY_SECONDS + 1)

        resource = _poll(operation_store, op.id, task_client, now_box)

        assert task_client.calls == 1
        assert resource.operation.status == "pending"
        stored = operation_store.get(op.id)
        assert stored is not None and stored.dispatched_at is None

    def test_a_running_operation_is_never_dispatched_by_a_poll(self) -> None:
        now_box = {"t": _OP_NOW}
        operation_store = _stale_operation_store(now_box)
        op = _create(operation_store)
        assert operation_store.claim(
            op.id, submission_count=0, lease_seconds=600, final_delivery=False
        )
        task_client = FakeTaskClient()
        now_box["t"] += timedelta(seconds=ENQUEUE_RECOVERY_SECONDS + 1)

        resource = _poll(operation_store, op.id, task_client, now_box)

        assert resource.operation.status == "running"
        assert task_client.enqueued == []


def test_the_composed_poll_route_reconciles_through_the_surfaces_task_client() -> None:
    # Submitted long before the route's own clock, so the enqueue is overdue.
    operation_store = LiveOperationStore(store=FakeOperationDatasetStore(), now=lambda: _OP_NOW)
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
