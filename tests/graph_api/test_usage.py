"""Tests for `graph_api.usage.usage_document`."""

from __future__ import annotations

from graph_api.source_models import Usage
from graph_api.usage import usage_document


def test_both_entries_are_null():
    assert usage_document() == Usage(api_access=None, python=None)


def test_takes_no_dcid_and_is_stable_across_calls():
    """Usage data is static and independent of request inputs."""
    assert usage_document() == usage_document()
