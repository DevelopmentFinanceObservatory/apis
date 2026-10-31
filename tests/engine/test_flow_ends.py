"""Checks the permutation of ends a raw-text parse takes as one, over every field naming an end."""

from __future__ import annotations

import dataclasses
import typing
from typing import Literal, TypeAliasType, get_args, get_origin

from pydantic import BaseModel

from search.engine.extract import ExtractedEntity, ExtractedOperation, Extraction, ExtractionResult
from search.engine.flow_ends import END_FIELDS, SWAPPED, end_permutation, permute_ends
from search.models import Operation


def _names_an_end(annotation: object) -> bool:
    if isinstance(annotation, TypeAliasType):
        return _names_an_end(annotation.__value__)
    if get_origin(annotation) is Literal:
        return bool({"origin", "destination"} & set(get_args(annotation)))
    return any(_names_an_end(arg) for arg in get_args(annotation))


def _models(annotation: object) -> list[type]:
    if isinstance(annotation, TypeAliasType):
        return _models(annotation.__value__)
    if isinstance(annotation, type) and (
        issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation)
    ):
        return [annotation]
    return [model for arg in get_args(annotation) for model in _models(arg)]


def _field_types(model: type) -> dict[str, object]:
    if issubclass(model, BaseModel):
        return {name: field.annotation for name, field in model.model_fields.items()}
    return typing.get_type_hints(model)


def _end_fields() -> dict[type, set[str]]:
    """Every field naming an end on a model a parse holds, reached from `ExtractionResult`."""
    found: dict[type, set[str]] = {}
    queue, seen = [ExtractionResult], set()
    while queue:
        model = queue.pop()
        if model in seen:
            continue
        seen.add(model)
        for name, annotation in _field_types(model).items():
            if _names_an_end(annotation):
                found.setdefault(model, set()).add(name)
            queue.extend(_models(annotation))
    return found


def test_every_field_naming_an_end_is_permuted():
    assert _end_fields() == {model: set(names) for model, names in END_FIELDS.items()}


def test_a_swap_maps_every_end_of_the_parse_and_nothing_else():
    entities = [
        ExtractedEntity(text="Kenya", flow_role="origin", role_span="'s"),
        ExtractedEntity(text="Africa"),
    ]
    operations = (
        Operation(verb="rank", phrase="top", over="destination"),
        Operation(verb="rank", phrase="by year", over="year"),
    )
    result = ExtractionResult(
        extraction=Extraction(
            kind="resolution",
            variables=["debt"],
            entities=entities,
            operations=[ExtractedOperation(verb="rank", variable=0, phrase="top", over="origin")],
        ),
        qualifiers={},
        operations={"debt": operations},
    )

    swapped = permute_ends(result, SWAPPED)

    assert [e.flow_role for e in swapped.extraction.entities] == ["destination", "unspecified"]
    assert [op.over for op in swapped.extraction.operations] == ["destination"]
    assert [op.over for op in swapped.operations["debt"]] == ["origin", "year"]
    assert permute_ends(swapped, SWAPPED) == result


def test_only_a_wording_that_moves_every_place_across_swaps():
    assert end_permutation({"Kenya": ("origin", "destination")}) == SWAPPED
    assert (
        end_permutation({"Kenya": ("origin", "destination"), "China": ("destination", "origin")})
        == SWAPPED
    )
    assert (
        end_permutation({"Kenya": ("origin", "destination"), "China": ("origin", "origin")}) == {}
    )
    assert end_permutation({}) == {}
