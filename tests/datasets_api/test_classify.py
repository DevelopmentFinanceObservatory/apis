"""Direct tests for `classify_datasets_exception`, including its `None` fallback for
exception types this surface does not classify.
"""

from __future__ import annotations

from dfo_spec_kit.identity import IdentityError

from datasets_api.app import RegistryBackpressure, classify_datasets_exception
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
