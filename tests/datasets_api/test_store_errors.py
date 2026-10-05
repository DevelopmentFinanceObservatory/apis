"""Tests SQLSTATE parsing and dataset-store error classification."""

from __future__ import annotations

import pytest

from datasets_api.store_errors import DatasetsStoreUnavailable, classify_store_exception
from dfo_platform.store_errors import extract_sqlstate, store_failure_is_permanent


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("bare driver/network failure, no SQLSTATE"),
        RuntimeError("not a dict shape", "extra positional arg"),
    ],
)
def test_extract_sqlstate_returns_none_for_a_non_pg8000_shaped_exception(exc):
    assert extract_sqlstate(exc) is None


def test_extract_sqlstate_returns_none_for_an_exception_with_no_args():
    assert extract_sqlstate(RuntimeError()) is None


def test_extract_sqlstate_parses_the_pg8000_error_response_shape():
    exc = RuntimeError({"C": "42702", "M": "column reference is ambiguous"})
    assert extract_sqlstate(exc) == "42702"


@pytest.mark.parametrize(
    "sqlstate",
    [
        "42702",  # syntax_error_or_access_rule_violation: ambiguous_column
        "42601",  # syntax_error_or_access_rule_violation: syntax_error
        "22P02",  # data_exception: invalid_text_representation
        "23503",  # integrity_constraint_violation: foreign_key_violation
        "23505",  # integrity_constraint_violation: unique_violation
        "0A000",  # feature_not_supported
    ],
)
def test_store_failure_is_permanent_for_permanent_sqlstate_classes(sqlstate):
    assert store_failure_is_permanent(sqlstate) is True


@pytest.mark.parametrize(
    "sqlstate",
    [
        "08006",  # connection_failure
        "53300",  # too_many_connections (insufficient_resources)
        "57014",  # query_canceled (operator_intervention)
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "99999",  # unrecognised class
    ],
)
def test_store_failure_is_permanent_for_transient_sqlstate_classes(sqlstate):
    assert store_failure_is_permanent(sqlstate) is False


def test_store_failure_is_permanent_defaults_transient_for_no_sqlstate():
    assert store_failure_is_permanent(None) is False


def test_classify_store_exception_maps_a_transient_failure_retryable_503():
    exc = DatasetsStoreUnavailable("exemplar")
    spec = classify_store_exception(exc)
    assert spec is not None
    assert spec.status == 503
    assert spec.code == "service_unavailable"
    assert spec.retryable is True


def test_classify_store_exception_maps_a_permanent_failure_non_retryable_500():
    exc = DatasetsStoreUnavailable("exemplar", sqlstate="23503")
    spec = classify_store_exception(exc)
    assert spec is not None
    assert spec.status == 500
    assert spec.code == "internal_error"
    assert spec.retryable is False


def test_classify_store_exception_returns_none_for_an_undeclared_type():
    assert classify_store_exception(RuntimeError("not this module's own")) is None
