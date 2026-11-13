"""The base of every response model a surface publishes a JSON Schema for."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict
from pydantic.config import JsonDict


def is_none(value: object) -> bool:
    """An `exclude_if` predicate for a field the server leaves out of a body when it is `None`."""
    return value is None


def optional_in_schema(*names: str) -> Callable[[JsonDict], None]:
    """A model's `json_schema_extra` that leaves `names` out of the schema's `required`, for
    members the server writes on every body, though bodies recorded before the members existed
    lack them, such as a computed field added beside an older spelling."""

    def drop_required(schema: JsonDict) -> None:
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [name for name in required if name not in names]

    return drop_required


class ResponseModel(BaseModel):
    """A response model. The server serialises every field, defaulted or not, so the schema
    built in serialization mode lists each one as required. A field the server leaves out of
    some bodies declares `exclude_if=is_none`, which keeps it optional. A member the server
    emits on every body, though bodies recorded before it existed lack it, stays out of
    `required` through `optional_in_schema`."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)
