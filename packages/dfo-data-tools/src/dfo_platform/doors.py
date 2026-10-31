"""FastAPI helpers for composing route families into a `dfo_platform.app.Surface`."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

from fastapi import APIRouter, FastAPI, Request
from fastapi.routing import APIRoute

from dfo_platform import ProblemSpec
from dfo_platform.codes import bound_endpoint, bound_registry_endpoint

# A route as `registry_endpoints` names it: its method, and its path as the router declares it.
type RouteKey = tuple[str, str]


@dataclass(frozen=True)
class DoorContribution:
    """One route family's contribution to a composed surface.

    Routers carry full paths. Classifiers return `None` for unrecognised exceptions. Lifespans enter
    together. Middleware checks run synchronously before FastAPI parses the request body. Each
    check decides for itself, for example by path, whether a request concerns it.
    `endpoint_exemplars` holds, for a door that declares them, an exemplar of every refusal the
    routes of each code registry endpoint raise (`check_endpoint_exemplars`).
    `registry_endpoints` maps each route's `RouteKey` to the code registry endpoint it serves
    (`dfo_platform.app.Surface.registry_endpoints`), and `public_paths` names the route paths any
    origin may read (`dfo_platform.app.Surface.public_paths`).
    """

    router: APIRouter
    classify: Callable[[BaseException], ProblemSpec | None]
    exception_types: tuple[type[Exception], ...] = ()
    exception_exemplars: tuple[BaseException, ...] = ()
    endpoint_exemplars: Mapping[str, tuple[BaseException, ...]] = field(default_factory=dict)
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None
    middleware: tuple[Callable[[Request], ProblemSpec | None], ...] = ()
    registry_endpoints: Mapping[RouteKey, str] = field(default_factory=dict)
    public_paths: frozenset[str] = frozenset()


def combine_classifiers(
    *classifiers: Callable[[BaseException], ProblemSpec | None],
) -> Callable[[BaseException], ProblemSpec | None]:
    """Try classifiers in order; return the first match or `None` for the platform default."""

    def classify(exc: BaseException) -> ProblemSpec | None:
        for classifier in classifiers:
            spec = classifier(exc)
            if spec is not None:
                return spec
        return None

    return classify


def classify_by_door(
    doors: Iterable[DoorContribution],
    *leading: Callable[[BaseException], ProblemSpec | None],
) -> Callable[[BaseException], ProblemSpec | None]:
    """Try `leading`, then the classifier of the door whose route the request matched, then
    every door's in order.

    The matched route is the one whose code registry endpoint the platform binds while it
    classifies (`codes.bound_endpoint`), so an exception several doors recognise is classified
    and logged by the door that raised it. With no endpoint bound, or one no door names, the
    doors are tried in order.
    """
    doors = tuple(doors)
    by_endpoint = {
        endpoint: door.classify for door in doors for endpoint in door.registry_endpoints.values()
    }
    in_order = tuple(door.classify for door in doors)

    def classify(exc: BaseException) -> ProblemSpec | None:
        own = by_endpoint.get(bound_endpoint() or "")
        first = () if own is None else (own,)
        return combine_classifiers(*leading, *first, *in_order)(exc)

    return classify


def flatten(doors: Iterable[DoorContribution]) -> APIRouter:
    """Combine routes in registration order while keeping `.routes` populated with `APIRoute`
    objects. This FastAPI version wraps `include_router` results in a lazy `_IncludedRouter`.
    """
    router = APIRouter()
    for door in doors:
        router.routes.extend(door.router.routes)
    return router


def route_keys(router: APIRouter) -> frozenset[RouteKey]:
    """The `RouteKey` of each method each route on `router` answers."""
    return frozenset(
        (method, route.path)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods or ()
    )


def route_registry_endpoints(router: APIRouter, endpoint: str) -> dict[RouteKey, str]:
    """Map every route on `router` to the one code registry `endpoint` its door serves."""
    return dict.fromkeys(route_keys(router), endpoint)


def path_registry_endpoints(router: APIRouter, endpoints: Mapping[str, str]) -> dict[RouteKey, str]:
    """Map every method `router` declares on each path of `endpoints` to that path's code
    registry endpoint, so a method added to a path is bound with it.

    Raises:
        ValueError: `endpoints` names a path no route on `router` declares.
    """
    keys = {key: endpoints[key[1]] for key in route_keys(router) if key[1] in endpoints}
    unrouted = sorted(set(endpoints) - {path for _, path in keys})
    if unrouted:
        raise ValueError(f"no route declares the registry-bound paths {unrouted!r}")
    return keys


def check_endpoint_exemplars(
    doors: Iterable[DoorContribution],
    classify: Callable[[BaseException], ProblemSpec | None],
) -> None:
    """Classify each door's `endpoint_exemplars` with their endpoint bound, as a request is, so
    a refusal with no one registered status there fails at startup. Classifiers log what they
    recognise, and an exemplar is no fault, so logging is off for the sweep.

    Raises:
        LookupError: The code registry maps no one status for a code on the endpoint whose
            routes raise it, or `classify` does not recognise an exemplar.
    """
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for door in doors:
            for endpoint, exemplars in door.endpoint_exemplars.items():
                for exemplar in exemplars:
                    with bound_registry_endpoint(endpoint):
                        spec = classify(exemplar)
                    if spec is None:
                        raise LookupError(f"nothing classifies {exemplar!r} on {endpoint}")
    finally:
        logging.disable(previous)
