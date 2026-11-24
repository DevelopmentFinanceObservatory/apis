"""Read the structured `json_fields` of the log records a test captured."""

from __future__ import annotations

import pytest

from dfo_platform.cloud_logging import JSON_FIELDS_ATTR


def json_fields(caplog: pytest.LogCaptureFixture, key: str) -> list[dict[str, object]]:
    """The `json_fields` of each captured record that carries `key`, in log order."""
    fields = (getattr(record, JSON_FIELDS_ATTR, None) or {} for record in caplog.records)
    return [one for one in fields if key in one]
