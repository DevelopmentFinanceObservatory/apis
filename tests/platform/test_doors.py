"""Tests door flattening and classifier composition."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.routing import APIRoute

from dfo_platform.doors import DoorContribution, combine_classifiers, flatten


def _door(path: str) -> DoorContribution:
    router = APIRouter()

    @router.get(path)
    async def handler() -> dict[str, str]:
        return {"path": path}

    return DoorContribution(router=router, classify=lambda _exc: None)


def test_flatten_concatenates_every_doors_routes_in_order() -> None:
    doors = (_door("/a"), _door("/b"), _door("/c"))

    router = flatten(doors)

    assert [route.path for route in router.routes] == ["/a", "/b", "/c"]


def test_flatten_returns_a_router_whose_routes_are_real_apiroute_objects() -> None:
    """Flattened routes remain `APIRoute` objects."""
    router = flatten((_door("/a"),))

    assert len(router.routes) == 1
    assert isinstance(router.routes[0], APIRoute)


def test_flatten_of_no_doors_returns_an_empty_router() -> None:
    assert flatten(()).routes == []


def test_combine_classifiers_returns_the_first_non_none_result() -> None:
    def refuses_value_error(exc: BaseException) -> str | None:
        return "value_error" if isinstance(exc, ValueError) else None

    def refuses_type_error(exc: BaseException) -> str | None:
        return "type_error" if isinstance(exc, TypeError) else None

    classify = combine_classifiers(refuses_value_error, refuses_type_error)

    assert classify(ValueError()) == "value_error"
    assert classify(TypeError()) == "type_error"


def test_combine_classifiers_returns_none_when_no_classifier_recognizes_it() -> None:
    classify = combine_classifiers(lambda _exc: None, lambda _exc: None)

    assert classify(RuntimeError()) is None
