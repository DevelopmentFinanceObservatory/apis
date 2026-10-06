"""Direct tests for `classify_datasets_exception`, including its `None` fallback for
exception types this surface does not classify.
"""

from __future__ import annotations

import pytest
from dfo_spec_kit.identity import IdentityError

from dataset_store.snapshot import SnapshotRevoked
from datasets_api.app import RegistryBackpressure, classify_datasets_exception
from datasets_api.codes_materialise import RenderedArtifactSigningFailed
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    SnapshotNotFound,
    SnapshotSigningFailed,
    SpecNotFound,
)
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.doors.pin import classify_pin_exception
from datasets_api.doors.snapshot import classify_snapshot_exception
from spec_registry import (
    IdentityCollision,
    IdentityQuarantined,
    SpecIdMismatch,
    SpecRegistryUnavailable,
)


def test_classify_datasets_exception_maps_spec_id_mismatch():
    exc = SpecIdMismatch(supplied="urn:spec:supplied", computed="urn:spec:computed")
    spec = classify_datasets_exception(exc)
    assert spec is not None
    assert spec.status == 422
    assert spec.code == "spec_id_mismatch"
    assert spec.retryable is False


def test_classify_datasets_exception_maps_identity_collision():
    exc = IdentityCollision(subject_id="urn:spec:collision")
    spec = classify_datasets_exception(exc)
    assert spec is not None
    assert spec.status == 500
    assert spec.code == "identity_collision"
    assert spec.retryable is False
    assert "urn:spec:collision" in spec.detail


def test_classify_datasets_exception_maps_identity_quarantined():
    exc = IdentityQuarantined(subject_id="urn:spec:quarantined", id_kind="spec")
    spec = classify_datasets_exception(exc)
    assert spec is not None
    assert spec.status == 500
    assert spec.code == "identity_quarantined"
    assert spec.retryable is False
    assert "urn:spec:quarantined" in spec.detail


def test_classify_datasets_exception_maps_registry_unavailable():
    exc = SpecRegistryUnavailable("registry write pool exhausted")
    spec = classify_datasets_exception(exc)
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


def test_classify_datasets_exception_maps_registry_backpressure():
    exc = RegistryBackpressure()
    spec = classify_datasets_exception(exc)
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


def test_classify_datasets_exception_maps_identity_error():
    exc = IdentityError("schema_violation", "identity document failed schema validation")
    spec = classify_datasets_exception(exc)
    assert spec is not None
    assert spec.status == 400
    assert spec.code == "request_invalid"
    assert spec.retryable is False
    assert spec.detail == str(exc)


def test_classify_datasets_exception_returns_none_for_an_undeclared_type():
    assert classify_datasets_exception(RuntimeError("not this surface's own")) is None


@pytest.mark.parametrize(
    "exc",
    [
        SpecNotFound(id="spec_exemplar"),
        SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
        SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
        ArtifactExpired(id="spec_exemplar", content_version="cv_exemplar"),
    ],
)
def test_pin_doors_first_four_branches_match_the_snapshot_doors_own(exc):
    """Pin-only deployments compose this classifier without the snapshot door, so its
    overlapping branches must match the snapshot classifier."""
    assert classify_snapshot_exception(exc) == classify_pin_exception(exc)


def test_signing_failure_classifies_identically_on_snapshot_and_mint_doors():
    """Unclassified signing failures must map to the same retryable 503 on both doors.
    Their details differ, so compare status, code, and retryability."""
    snapshot_exc = SnapshotSigningFailed(
        bucket="exemplar-bucket",
        object_name="sha256:exemplar",
        digest="sha256:exemplar",
        error_class_name=None,
        status_code=None,
    )
    mint_exc = RenderedArtifactSigningFailed(
        bucket="exemplar-bucket",
        object_name="sha256:exemplar",
        digest="sha256:exemplar",
        error_class_name=None,
        status_code=None,
    )
    snapshot_spec = classify_snapshot_exception(snapshot_exc)
    mint_spec = classify_mint_exception(mint_exc)
    assert snapshot_spec is not None
    assert mint_spec is not None
    assert snapshot_spec.status == mint_spec.status
    assert snapshot_spec.code == mint_spec.code
    assert snapshot_spec.retryable == mint_spec.retryable
