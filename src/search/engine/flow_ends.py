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
from typing import Final, cast

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
    of models, dataclasses, sequences and mappings. Values `ends` does not name, and every
    other field, stay as they are."""
    if not ends:
        return value
    return cast(T, _permuted(value, ends))


def _permuted(value: object, ends: Ends) -> object:
    fields = END_FIELDS.get(type(value), frozenset())
    if isinstance(value, BaseModel):
        update = {name: _permuted(getattr(value, name), ends) for name in type(value).model_fields}
        update.update({name: _mapped(update[name], ends) for name in fields})
        return value.model_copy(update=update)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        update = {
            f.name: _permuted(getattr(value, f.name), ends)
            for f in dataclasses.fields(value)
            if f.init
        }
        update.update({name: _mapped(update[name], ends) for name in fields})
        return dataclasses.replace(value, **update)
    if isinstance(value, list | tuple):
        return type(value)(_permuted(item, ends) for item in value)
    if isinstance(value, Mapping):
        return {key: _permuted(item, ends) for key, item in value.items()}
    return value


def _mapped(value: object, ends: Ends) -> object:
    return ends.get(cast(RoleAxis, value), value) if isinstance(value, str) else value
