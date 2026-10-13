"""`strip_schema_annotations` drops annotation keywords and keeps field names and literals."""

from __future__ import annotations

from dfo_platform.schema_annotations import strip_schema_annotations


def test_strips_annotations_and_keeps_names_and_literals():
    schema = {
        "title": "Node",
        "description": "a node",
        "type": "object",
        "properties": {
            "title": {"title": "Title", "type": "string"},
            "properties": {"description": "nested", "anyOf": [{"type": "null"}]},
            "kind": {
                "enum": ["title", "description"],
                "const": {"title": "x"},
                "default": {"description": "y"},
                "examples": [{"title": "z"}],
            },
        },
        "discriminator": {"propertyName": "kind", "mapping": {"title": "#/$defs/description"}},
        "$defs": {"description": {"title": "Description", "type": "integer"}},
        "required": ["title"],
    }

    assert strip_schema_annotations(schema) == {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "properties": {"anyOf": [{"type": "null"}]},
            "kind": {
                "enum": ["title", "description"],
                "const": {"title": "x"},
                "default": {"description": "y"},
                "examples": [{"title": "z"}],
            },
        },
        "discriminator": {"propertyName": "kind", "mapping": {"title": "#/$defs/description"}},
        "$defs": {"description": {"type": "integer"}},
        "required": ["title"],
    }
