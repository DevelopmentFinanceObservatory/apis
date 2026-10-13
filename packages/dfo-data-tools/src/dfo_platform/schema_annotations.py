"""Strip prose annotations from a JSON Schema before a capability token hashes it.

Every surface's `version.py` hashes its served schemas through `strip_schema_annotations`, so
rewording a description leaves the token alone while any structural change moves it.
"""

from __future__ import annotations

# Keywords whose text annotates the schema they sit in and never reaches a response body.
_ANNOTATION_KEYWORDS = frozenset({"description", "title"})

# Keywords holding a map of names to subschemas. The names are wire field names and stay,
# so a field called `description` or `title` keeps its key and its type in the hash.
_NAMED_SUBSCHEMA_KEYWORDS = frozenset({"properties", "patternProperties", "$defs", "definitions"})

# Keywords holding instance data or a tag mapping. Their values are hashed verbatim, so a
# `const` or `enum` value spelled `title` survives.
_LITERAL_KEYWORDS = frozenset({"enum", "const", "default", "examples", "discriminator"})


def strip_schema_annotations(schema: object) -> object:
    """Return `schema` without `description` and `title` keywords at any schema position.

    Property names under `properties`, `patternProperties`, `$defs` and `definitions` stay,
    and values under `enum`, `const`, `default`, `examples` and `discriminator` are copied
    unchanged. Every other keyword's value is walked as a schema or list of schemas.
    """
    if isinstance(schema, list):
        return [strip_schema_annotations(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    stripped: dict[object, object] = {}
    for keyword, value in schema.items():
        if keyword in _ANNOTATION_KEYWORDS:
            continue
        if keyword in _LITERAL_KEYWORDS:
            stripped[keyword] = value
        elif keyword in _NAMED_SUBSCHEMA_KEYWORDS and isinstance(value, dict):
            stripped[keyword] = {
                name: strip_schema_annotations(subschema) for name, subschema in value.items()
            }
        else:
            stripped[keyword] = strip_schema_annotations(value)
    return stripped
