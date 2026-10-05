"""Pins `GraphClient` constructor bounds and exception-class validation without network calls."""

from __future__ import annotations

import pytest

from dc_graph import (
    GraphCapExceededError,
    GraphClient,
    GraphDeadlineExceededError,
    GraphReadError,
)


class _UnrelatedError(Exception):
    pass


def test_read_error_cls_must_be_a_graph_read_error_subclass() -> None:
    with pytest.raises(TypeError, match="read_error_cls"):
        GraphClient(base_url="http://fake/core/api", read_error_cls=_UnrelatedError)


def test_deadline_error_cls_must_be_a_graph_deadline_exceeded_error_subclass() -> None:
    with pytest.raises(TypeError, match="deadline_error_cls"):
        GraphClient(base_url="http://fake/core/api", deadline_error_cls=GraphReadError)


def test_cap_error_cls_must_be_a_graph_cap_exceeded_error_subclass() -> None:
    with pytest.raises(TypeError, match="cap_error_cls"):
        GraphClient(base_url="http://fake/core/api", cap_error_cls=_UnrelatedError)


def test_a_deadline_error_cls_subclass_is_accepted_as_a_read_error_cls_too() -> None:
    """A `GraphReadError` subclass can serve as both read and deadline error type."""
    GraphClient(
        base_url="http://fake/core/api",
        read_error_cls=GraphDeadlineExceededError,
        deadline_error_cls=GraphDeadlineExceededError,
        cap_error_cls=GraphCapExceededError,
    )
