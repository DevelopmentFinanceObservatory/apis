from __future__ import annotations

import types
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, SerializationInfo, SerializerFunctionWrapHandler
from pydantic_core import SchemaSerializer, core_schema

type _Serialize = Callable[[Any, SerializerFunctionWrapHandler, SerializationInfo], Any]


def required_nullable_fields(cls: type[BaseModel]) -> tuple[str, ...]:
    """The fields of `cls` that are required and whose annotation admits None."""
    return tuple(
        name
        for name, field in cls.model_fields.items()
        if field.is_required() and _admits_none(field.annotation)
    )


def _admits_none(annotation: object) -> bool:
    """Whether a field annotation accepts None: `None` itself or a union naming it."""
    if annotation is None or annotation is type(None):
        return True
    origin = typing.get_origin(annotation)
    return origin in (typing.Union, types.UnionType) and type(None) in typing.get_args(annotation)


def _keep_required_nulls(cls: type[StrictModel]) -> None:
    """Wrap `cls`'s serializer, around any the model declares itself, so a dump under
    `exclude_none` keeps each required, nullable field whose value is None.

    The wrap goes on the model node of the class's own core schema, which every model that
    nests `cls` builds from, and carries the declared serializer's return schema, so the
    JSON schema stays as pydantic generates it."""
    names = required_nullable_fields(cls)
    if not names:
        return
    node = _model_node(cls.__pydantic_core_schema__, cls)
    declared = node.get("serialization")
    if declared is not None and declared["type"] != "function-wrap":
        raise TypeError(f"{cls.__name__} declares a plain serializer, so its nulls cannot be kept")
    serialize = _declared_serialize(declared)

    def keep_nulls(
        value: StrictModel, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Any:
        data = serialize(value, handler, info)
        if info.exclude_none:
            for name in names:
                if name not in data and getattr(value, name) is None:
                    data[name] = None
        return data

    node["serialization"] = core_schema.wrap_serializer_function_ser_schema(
        keep_nulls,
        info_arg=True,
        return_schema=declared.get("return_schema") if declared is not None else None,
    )
    # The class's prebuilt serializer is the one being replaced, so build without it.
    cls.__pydantic_serializer__ = SchemaSerializer(
        cls.__pydantic_core_schema__, node.get("config"), _use_prebuilt=False
    )


def _declared_serialize(declared: Any) -> _Serialize:
    """The model's own wrap serializer as a call that takes info, else the handler alone."""
    if declared is None:
        return lambda value, handler, _info: handler(value)
    function = declared["function"]
    if declared.get("info_arg"):
        return function
    return lambda value, handler, _info: function(value, handler)


def _model_node(schema: core_schema.CoreSchema, cls: type) -> dict[str, Any]:
    """The `model` node of `cls` in its core schema, which pydantic may wrap in a
    `definitions` schema or a validator function."""
    stack: list[Any] = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "model" and node.get("cls") is cls:
                return node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    raise LookupError(f"{cls.__name__}'s core schema holds no model node for it")


class StrictModel(BaseModel):
    """Base class for every model in the v1 contract, request and response
    alike: both directions forbid unknown fields, so a misspelled field in
    server-side construction of a response fails loudly at validation time
    rather than shipping with the intended field silently unset. The one
    exception is `Resolution`, which stays a plain `BaseModel`: its `status`
    is a computed field echoed in every serialised response body, so
    forbidding unknown fields there would reject the model's own
    `model_dump()` output on the next `model_validate` round-trip. The other
    two response kinds share its base and forbid unknown fields themselves.

    A required field that admits null keeps its null under `exclude_none`, which every
    response dump takes: dropping the key would leave a payload the model refuses on the
    way back. Each model with such fields gets a serializer that restores them
    (`_keep_required_nulls`). A model without them serialises as pydantic builds it.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        super().__pydantic_on_complete__()
        _keep_required_nulls(cls)
