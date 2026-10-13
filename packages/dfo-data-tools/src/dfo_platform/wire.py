"""The base of every response model a surface publishes a JSON Schema for."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def is_none(value: object) -> bool:
    """An `exclude_if` predicate for a field the server leaves out of a body when it is `None`."""
    return value is None


class ResponseModel(BaseModel):
    """A response model. The server serialises every field, defaulted or not, so the schema
    built in serialization mode lists each one as required. A field the server leaves out of
    some bodies declares `exclude_if=is_none`, which keeps it optional."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)
