"""The `spec-kit` repository's descriptions distribution, `dfo_descriptions`, as the tests reach it.

`validate_description` checks a served description against the kit's description schema and
its structure rules for a description.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from dfo_descriptions import check_description, description_schema, steps_schema
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from referencing import Registry, Resource


@cache
def _validator() -> Validator:
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema))
        for schema in (steps_schema(), description_schema())
    )
    return Draft202012Validator(description_schema(), registry=registry)


def validate_description(served: dict[str, Any]) -> None:
    """Check `served` against the kit's `description.schema.json` and its structure rules for a
    description."""
    _validator().validate(served)
    check_description(served)
