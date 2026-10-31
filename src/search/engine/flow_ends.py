"""One permutation of a flow's two ends, applied to a whole raw-text parse.

The query's own wording can show that the extraction read a flow backwards: "Kenya's debt
service" makes Kenya the debtor, the destination, where the extraction put Kenya on origin.
The repair is a permutation of ends, and it holds for every field of the parse that names
an end, so an operation that ranks "over destination" in the extraction's reading ranks over
origin once the ends swap. `END_FIELDS` lists those fields by model, and `permute_ends`
rebuilds a parse with each of them mapped, whatever object holds it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import cache
from typing import Any, Final, TypeAliasType, cast, get_args, get_type_hints

from pydantic import BaseModel

from search.engine.extract import ExtractedEntity, ExtractedOperation
from search.models import Operation, RoleAxis

type Ends = Mapping[RoleAxis, RoleAxis]

SWAPPED: Final[Ends] = {"origin": "destination", "destination": "origin"}

END_FIELDS: Final[Mapping[type, frozenset[str]]] = {
    ExtractedEntity: frozenset({"flow_role"}),
    ExtractedOperation: frozenset({"over"}),
    Operation: frozenset({"over"}),
}
"""Every field of a raw-text parse (`extract.ExtractionResult`) whose value can name a flow
end, by the model that holds it."""


def end_permutation(moves: Mapping[str, tuple[RoleAxis, RoleAxis]]) -> Ends:
    """The permutation the query's wording makes of the extraction's ends: `SWAPPED` when
    every place the wording moves (`moves`, its extracted axis then the worded one) goes to
    the other end, else none. A wording that moves one place and keeps another permutes
    nothing, since no one permutation maps both."""
    if moves and all(extracted != worded for extracted, worded in moves.values()):
        return SWAPPED
    return {}


def permute_ends[T](value: T, ends: Ends) -> T:
    """`value` with every `END_FIELDS` field inside it mapped through `ends`, at any depth
    of models, dataclasses, sequences and mappings. Only the fields that can hold an end
    (`_carrying_fields`) are rebuilt; every other field keeps its object."""
    if not ends:
        return value
    return cast(T, _permuted(value, ends))


def _permuted(value: object, ends: Ends) -> object:
    if isinstance(value, list):
        return [_permuted(item, ends) for item in value]
    if isinstance(value, tuple):
        return tuple(_permuted(item, ends) for item in value)
    if isinstance(value, Mapping):
        return {key: _permuted(item, ends) for key, item in value.items()}
    if not isinstance(value, BaseModel) and not dataclasses.is_dataclass(value):
        return value
    fields = _carrying_fields(type(value))
    if not fields:
        return value
    update = {name: _permuted(getattr(value, name), ends) for name in fields}
    for name in END_FIELDS.get(type(value), ()):
        update[name] = _mapped(update[name], ends)
    if isinstance(value, BaseModel):
        return value.model_copy(update=update)
    return dataclasses.replace(cast(Any, value), **update)


@cache
def _carrying_fields(model: type) -> frozenset[str]:
    """The fields of `model` that name an end (`END_FIELDS`) or can hold a model that does,
    the only fields `permute_ends` rebuilds. Parse models hold no cycle of types."""
    own = END_FIELDS.get(model, frozenset())
    return own | frozenset(
        name
        for name, annotation in _field_types(model).items()
        if any(_carrying_fields(held) for held in _models(annotation))
    )


def _field_types(model: type) -> dict[str, object]:
    if issubclass(model, BaseModel):
        return {name: field.annotation for name, field in model.model_fields.items()}
    return get_type_hints(model)


def _models(annotation: object) -> list[type]:
    """The models and dataclasses an annotation can hold, at any depth of its arguments."""
    if isinstance(annotation, TypeAliasType):
        return _models(annotation.__value__)
    if isinstance(annotation, type) and (
        issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation)
    ):
        return [annotation]
    return [held for arg in get_args(annotation) for held in _models(arg)]


def _mapped(value: object, ends: Ends) -> object:
    return ends.get(cast(RoleAxis, value), value) if isinstance(value, str) else value
