"""Pin RFC 7386 merge semantics and create-only field refusals."""

from __future__ import annotations

import pytest

from assembler.errors import SchemaInvalid
from assembler.merge import merge_intent


def test_absent_field_is_untouched():
    current = {
        "type": "line_chart",
        "title": "Health ODA to Ethiopia",
        "window": {"start": "2015", "end": "2024"},
    }
    merged = merge_intent(current=current, patch={"title": "New title"})
    assert merged["window"] == {"start": "2015", "end": "2024"}
    assert merged["title"] == "New title"


def test_null_clears_an_optional_field():
    current = {"type": "line_chart", "window": {"start": "2015", "end": "2024"}}
    merged = merge_intent(current=current, patch={"window": None})
    assert "window" not in merged


def test_select_replaces_whole_never_merges_key_by_key():
    current = {
        "type": "line_chart",
        "select": {"statvar": "health_oda", "recipient": "country/ETH"},
    }
    patch_select = {"statvar": "health_oda", "recipient": "country/GHA"}
    merged = merge_intent(current=current, patch={"select": patch_select})
    assert merged["select"] == patch_select


def test_window_object_replaces_whole_never_merges_key_by_key():
    current = {"type": "line_chart", "window": {"start": "2015", "end": "2024"}}
    merged = merge_intent(current=current, patch={"window": {"end": "2020"}})
    # A deep merge would have kept start="2015"; RFC 7386 replaces the object whole.
    assert merged["window"] == {"end": "2020"}


def test_every_field_is_absolute_resending_cannot_compound():
    current = {"type": "line_chart", "title": "A"}
    once = merge_intent(current=current, patch={"title": "B"})
    twice = merge_intent(current=once, patch={"title": "B"})
    assert once == twice


@pytest.mark.parametrize("field", ["type", "frame_ref"])
def test_create_only_fields_refuse_on_revise(field):
    current = {
        "type": "line_chart",
        "frame_ref": {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "b" * 32},
    }
    with pytest.raises(SchemaInvalid) as excinfo:
        merge_intent(current=current, patch={field: "something else"})
    assert excinfo.value.pointer == f"/{field}"
    assert excinfo.value.legal_fields is not None
    assert field not in excinfo.value.legal_fields


def test_create_only_field_refusal_reports_before_touching_other_keys():
    """A create-only field refusal also discards legal fields in the same patch."""
    current = {"type": "line_chart", "title": "A"}
    with pytest.raises(SchemaInvalid):
        merge_intent(current=current, patch={"title": "B", "type": "stat_card"})
