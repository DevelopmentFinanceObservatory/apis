"""Every Datasets refusal names a code the code registry gives one HTTP status on each endpoint
whose route raises it, and every exemplar a door declares goes out on the wire at that status.

A route raises the refusal classes in its inventory and their subclasses: a door's
`exception_types`, or for the definition door the inventory of each of its three routes
(`ROUTE_EXCEPTION_TYPES`). The mint door's inventory holds what the worker raises running the
operations a mint queues, since the worker classifies under `datasets.mint`. Every `Refusal` subclass with a code belongs to some door, so none
escapes the sweep. The registry maps every (code, endpoint) a route's inventory declares to the
status the wire sends, so any gap fails here.
"""

from __future__ import annotations

import pytest
from dfo_spec_kit.registry import REGISTRY

import datasets_api.app  # noqa: F401 -- imports every door module, so every refusal class loads
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.doors.analysis import build_analysis_door
from datasets_api.doors.definition import ROUTE_EXCEPTION_TYPES, build_definition_door
from datasets_api.doors.description import build_description_door
from datasets_api.doors.files import build_files_door
from datasets_api.doors.meta import build_meta_door
from datasets_api.doors.mint import build_mint_door
from datasets_api.doors.observations import build_observations_door
from datasets_api.doors.pin import build_pin_door
from datasets_api.doors.publish import build_publish_door
from datasets_api.doors.recipe import build_recipe_door
from datasets_api.doors.refresh import build_refresh_door
from datasets_api.doors.register import build_register_door
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.doors.versions import build_versions_door
from datasets_api.operation_store import classify_operation_store_exception
from datasets_api.store_errors import classify_store_exception
from dfo_platform.codes import bound_registry_endpoint, registered_status
from dfo_platform.doors import DoorContribution, combine_classifiers
from dfo_platform.problem import Refusal
from spec_registry import IdentityQuarantined

# The registry endpoints each door's routes serve.
_ENDPOINTS = {
    "meta": ("datasets.operations",),
    "register": ("datasets.register",),
    "mint": ("datasets.mint",),
    "definition": ("datasets.definition", "datasets.options", "datasets.members"),
    "refresh": ("datasets.refresh",),
    "analysis": ("datasets.analysis",),
    "recipe": ("datasets.recipe",),
    "snapshot": ("datasets.snapshot",),
    "versions": ("datasets.snapshot",),
    "description": ("datasets.description",),
    "files": ("datasets.files",),
    "observations": ("datasets.observations",),
    "pin": ("datasets.pin",),
    "publish": ("datasets.publish",),
}

# Refusal classes only another surface raises, as Search's subclasses of materialise refusals.
_OTHER_SURFACES = ("search.", "tests.")


@pytest.fixture(scope="module")
def doors() -> dict[str, DoorContribution]:
    # Doors read their dependencies when they serve, so an empty cell builds every inventory.
    cell, settings = DepsCell(), DatasetsBuildSettings.from_env()
    return {
        "meta": build_meta_door(deps=cell, settings=settings),
        "register": build_register_door(deps=cell),
        "mint": build_mint_door(deps=cell),
        "definition": build_definition_door(deps=cell),
        "refresh": build_refresh_door(deps=cell, settings=settings),
        "analysis": build_analysis_door(deps=cell),
        "recipe": build_recipe_door(deps=cell, settings=settings),
        "snapshot": build_snapshot_door(deps=cell),
        "versions": build_versions_door(deps=cell),
        "description": build_description_door(deps=cell),
        "files": build_files_door(deps=cell),
        "observations": build_observations_door(deps=cell),
        "pin": build_pin_door(deps=cell, settings=settings),
        "publish": build_publish_door(deps=cell, settings=settings),
    }


def _subclasses(cls: type) -> set[type]:
    found = set()
    for sub in cls.__subclasses__():
        found |= {sub, *_subclasses(sub)}
    return found


def _statuses(code: str, endpoint: str) -> set[int]:
    """The statuses the registry gives `code` on `endpoint`'s problem details."""
    entry = REGISTRY[code] if code in REGISTRY else None  # noqa: SIM401 -- CodeRegistry has no .get()
    return {
        mapping.http_status
        for mapping in (entry.mappings if entry is not None else ())
        if mapping.endpoint == endpoint
        and mapping.channel == "problem_details"
        and mapping.http_status is not None
    }


def _inventories(doors: dict[str, DoorContribution]) -> dict[str, dict[str, tuple[type, ...]]]:
    """Each door's routes, by registry endpoint, with the refusal classes each declares."""
    return {
        name: dict(ROUTE_EXCEPTION_TYPES)
        if name == "definition"
        else dict.fromkeys(_ENDPOINTS[name], door.exception_types)
        for name, door in doors.items()
    }


def _raised(doors: dict[str, DoorContribution]) -> dict[type, set[str]]:
    """Each refusal class with a code, and the endpoints whose routes raise it."""
    raised: dict[type, set[str]] = {}
    for routes in _inventories(doors).values():
        for endpoint, declared_types in routes.items():
            for declared in declared_types:
                # A declared base without a code, as `Refusal` itself, names no refusal of its own.
                if not issubclass(declared, Refusal) or not hasattr(declared, "code"):
                    continue
                for cls in {declared, *_subclasses(declared)}:
                    if hasattr(cls, "code") and not cls.__module__.startswith(_OTHER_SURFACES):
                        raised.setdefault(cls, set()).add(endpoint)
    return raised


def test_the_definition_door_declares_an_inventory_for_each_of_its_endpoints(doors):
    assert set(ROUTE_EXCEPTION_TYPES) == set(_ENDPOINTS["definition"])


def test_every_declared_refusal_has_a_registered_status_on_each_endpoint_raising_it(doors):
    missing = {
        (cls.code, endpoint)
        for cls, endpoints in _raised(doors).items()
        for endpoint in endpoints
        if not _statuses(cls.code, endpoint)
    }
    assert sorted(missing) == []


def test_every_declared_refusal_code_has_one_status_on_each_endpoint_declaring_it(doors):
    """A route renders a refusal at `registered_status(code)` with its endpoint bound. A pin bump
    giving a code a second status on an endpoint that declares it fails here, as does one that
    leaves an unmapped pair with no single status to fall back to."""
    ambiguous = {
        (cls.code, endpoint)
        for cls, endpoints in _raised(doors).items()
        for endpoint in endpoints
        if len(_statuses(cls.code, endpoint)) > 1
    }
    assert sorted(ambiguous) == []
    unresolved = []
    for cls, endpoints in sorted(_raised(doors).items(), key=lambda item: item[0].code):
        for endpoint in sorted(endpoints):
            with bound_registry_endpoint(endpoint):
                try:
                    registered_status(cls.code)
                except LookupError:
                    unresolved.append((cls.code, endpoint))
    assert unresolved == []


def test_every_endpoint_the_registry_maps_identity_quarantined_on_declares_it(doors):
    """A route that reads a registered id raises `IdentityQuarantined`, so the registry maps the
    code on it and the route's inventory declares the class."""
    mapped = {
        mapping.endpoint
        for mapping in REGISTRY["identity_quarantined"].mappings
        if mapping.channel == "problem_details"
    }
    assert sorted(mapped - _raised(doors)[IdentityQuarantined]) == []


def test_every_datasets_route_binds_the_registry_endpoint_its_door_serves(doors):
    """Each route declares its registry endpoint, so its refusals render at that endpoint's
    status. The meta door's health, schema and capabilities routes serve no endpoint."""
    for name, door in doors.items():
        declared = set(door.registry_endpoints.values())
        assert declared == set(_ENDPOINTS[name]), name
        routed = {getattr(route, "path", None) for route in door.router.routes}
        assert set(door.registry_endpoints) <= routed, name
    assert all(
        getattr(route, "path", None) in door.registry_endpoints
        for name, door in doors.items()
        if name != "meta"
        for route in door.router.routes
    )


def test_the_refusal_class_sweep_covers_every_door_and_route(doors):
    assert set(_ENDPOINTS) == set(doors)
    assert all(_inventories(doors)[name].keys() == set(_ENDPOINTS[name]) for name in doors)


def test_every_refusal_class_belongs_to_a_door(doors):
    refusals = {
        cls
        for cls in _subclasses(Refusal)
        if hasattr(cls, "code") and not cls.__module__.startswith(_OTHER_SURFACES)
    }
    assert refusals, "no refusal class loaded"
    assert sorted(c.__qualname__ for c in refusals - set(_raised(doors))) == []


def test_every_refusal_exemplar_goes_out_at_its_registered_status(doors):
    """Each exemplar is classified by its own door's `classify` with each endpoint bound whose
    route declares its class, as the wire is, and goes out at the one status the registry gives
    its code on that endpoint."""
    sent: set[tuple[str, str, int]] = set()
    for name, door in doors.items():
        for exemplar in door.exception_exemplars:
            for endpoint, declared in _inventories(doors)[name].items():
                if not isinstance(exemplar, declared):
                    continue
                with bound_registry_endpoint(endpoint):
                    spec = door.classify(exemplar)
                if spec is not None:
                    sent.add((spec.code, endpoint, spec.status))
    disagreements = {
        (code, endpoint, status)
        for code, endpoint, status in sent
        if _statuses(code, endpoint) != {status}
    }
    assert sorted(disagreements) == []


def _surface_classifier(doors: dict[str, DoorContribution]):
    """The classifier `datasets_api.app.datasets_surface` composes, in its door order."""
    return combine_classifiers(
        *(door.classify for door in doors.values()),
        classify_store_exception,
        classify_operation_store_exception,
    )


def _raising_types(declared_types: tuple[type, ...]) -> tuple[type, ...]:
    """The declared types that name failures: a refusal base without a code, as `Refusal`
    itself, names none."""
    return tuple(
        declared
        for declared in declared_types
        if not issubclass(declared, Refusal) or hasattr(declared, "code")
    )


def _surface_exemplars(doors: dict[str, DoorContribution]) -> tuple[BaseException, ...]:
    return tuple(exemplar for door in doors.values() for exemplar in door.exception_exemplars)


def test_every_type_a_route_raises_has_a_surface_exemplar(doors):
    """The status sweep below classifies exemplars, so a raised type with none escapes it."""
    exemplars = _surface_exemplars(doors)
    uncovered = {
        (declared.__qualname__, endpoint)
        for routes in _inventories(doors).values()
        for endpoint, declared_types in routes.items()
        for declared in _raising_types(declared_types)
        if not any(isinstance(exemplar, declared) for exemplar in exemplars)
    }
    assert sorted(uncovered) == []


def test_every_exemplar_a_route_raises_maps_on_its_endpoint_under_the_surface_classifier(doors):
    """Each route renders through the whole surface's classifier with its endpoint bound, which
    recognises graph, upstream and store errors beside the doors' refusals. Every exemplar of a
    type the route's inventory names classifies to a code the registry maps on that endpoint at
    the status sent; a code with no status there raises `LookupError` and goes out as 500."""
    classify = _surface_classifier(doors)
    exemplars = _surface_exemplars(doors)
    unmapped = set()
    for routes in _inventories(doors).values():
        for endpoint, declared_types in routes.items():
            raising = _raising_types(declared_types)
            for exemplar in exemplars:
                if not isinstance(exemplar, raising):
                    continue
                with bound_registry_endpoint(endpoint):
                    try:
                        spec = classify(exemplar)
                    except LookupError:
                        unmapped.add((type(exemplar).__qualname__, endpoint))
                        continue
                if spec is not None and _statuses(spec.code, endpoint) != {spec.status}:
                    unmapped.add((type(exemplar).__qualname__, endpoint))
    assert sorted(unmapped) == []
