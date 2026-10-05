"""Tests pin the permanent/transient boundary for artifact signing failures, including
Google Auth class-name handling and HTTP status codes.
"""

from __future__ import annotations

import google.auth.exceptions
import pytest

from datasets_api.codes_materialise import (
    _PERMANENT_SIGNING_FAILURE_CLASS_NAMES,
    signing_failure_is_permanent,
)


@pytest.mark.parametrize(
    ("error_class_name", "status_code", "expected"),
    [
        # `status_code` alone deciding, with no class name to fall back on:
        # 401 and 403 are permanent. 429 and 5xx responses are transient.
        (None, 401, True),
        (None, 403, True),
        (None, 429, False),
        (None, 502, False),
        # `error_class_name` alone deciding, with no `status_code` at all --
        # the shape a malformed credential/certificate defect carries in
        # production (no HTTP status at all).
        ("MalformedError", None, True),
        ("InvalidValue", None, True),
        # `RefreshError`/`DefaultCredentialsError` with no status is the
        # realistic shape of a TRANSIENT metadata-server outage, not a
        # permanent one: `google.auth.compute_engine.credentials`'s own
        # `refresh()` wraps a metadata-server `TransportError` as a bare
        # `RefreshError`, and `google.auth._default`'s own ADC discovery
        # ends in `DefaultCredentialsError` when the metadata server is
        # unreachable -- neither name is in
        # `_PERMANENT_SIGNING_FAILURE_CLASS_NAMES` for exactly this reason.
        ("RefreshError", None, False),
        ("DefaultCredentialsError", None, False),
        # An unrecognised class name, an unrecognised status code, and both
        # fields absent all default to transient.
        ("SomeOtherError", None, False),
        (None, None, False),
        (None, 404, False),
    ],
    ids=[
        "401-alone",
        "403-alone",
        "429-alone",
        "502-alone",
        "MalformedError-no-status",
        "InvalidValue-no-status",
        "RefreshError-no-status-is-transient",
        "DefaultCredentialsError-no-status-is-transient",
        "unrecognised-class-no-status",
        "both-fields-absent",
        "unrecognised-status-404",
    ],
)
def test_signing_failure_is_permanent(
    error_class_name: str | None, status_code: int | None, expected: bool
):
    assert (
        signing_failure_is_permanent(error_class_name=error_class_name, status_code=status_code)
        == expected
    )


def test_permanent_signing_failure_class_names_are_real_google_auth_exceptions():
    """Each allowlisted name must match a `google.auth.exceptions` class's `__qualname__`."""
    for name in _PERMANENT_SIGNING_FAILURE_CLASS_NAMES:
        cls = getattr(google.auth.exceptions, name, None)
        assert cls is not None, f"google.auth.exceptions has no attribute {name!r}"
        assert cls.__qualname__ == name
