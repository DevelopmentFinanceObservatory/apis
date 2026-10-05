"""Ensures event schemas define `variable_index` as the final response parts-list index."""

from __future__ import annotations

from search.engine.events import SentenceEvent, ShapeEvent, StepEvent


def test_variable_index_field_descriptions_name_the_parts_list():
    for model in (ShapeEvent, SentenceEvent, StepEvent):
        description = model.model_fields["variable_index"].description or ""
        assert "parts list" in description, (
            f"{model.__name__}.variable_index description does not name the "
            "final response's parts list"
        )
