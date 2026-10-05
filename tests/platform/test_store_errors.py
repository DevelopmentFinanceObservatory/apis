"""Checks SQLSTATE classification and log-safe exception diagnostics."""

from __future__ import annotations

import pytest

from dfo_platform.store_errors import safe_diagnostics, store_failure_is_permanent


@pytest.mark.parametrize(
    "sqlstate",
    [
        "42601",  # syntax_error_or_access_rule_violation: syntax_error
        "22P02",  # data_exception: invalid_text_representation
        "23505",  # integrity_constraint_violation: unique_violation
        "0A000",  # feature_not_supported
    ],
)
def test_store_failure_is_permanent_true_for_one_member_of_each_permanent_class(sqlstate):
    assert store_failure_is_permanent(sqlstate) is True


def test_store_failure_is_permanent_false_for_a_transient_sqlstate_class():
    assert store_failure_is_permanent("08006") is False  # connection_failure


def test_store_failure_is_permanent_false_for_no_sqlstate():
    assert store_failure_is_permanent(None) is False


def test_safe_diagnostics_omits_the_exception_message():
    exc = RuntimeError("bound parameter value that must not reach a log line")
    assert "bound parameter" not in safe_diagnostics(exc)
    assert "RuntimeError" in safe_diagnostics(exc)


def test_safe_diagnostics_includes_the_sqlstate_when_present():
    exc = RuntimeError({"C": "23505", "M": "duplicate key value"})
    assert safe_diagnostics(exc) == "builtins.RuntimeError sqlstate=23505"


def test_safe_diagnostics_omits_sqlstate_when_absent():
    assert safe_diagnostics(RuntimeError("bare failure")) == "builtins.RuntimeError"
