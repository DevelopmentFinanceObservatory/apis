"""FastAPI helpers for composing route families into a `dfo_platform.app.Surface`."""

from __future__ import annotations

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


def endpoint_exemplars(doors: Iterable[DoorContribution]) -> tuple[BaseException, ...]:
    """Every exemplar `doors` declare by endpoint, each once."""
    return tuple(
        dict.fromkeys(
            exemplar
            for door in doors
            for exemplars in door.endpoint_exemplars.values()
            for exemplar in exemplars
        )
    )


def check_endpoint_exemplars(
    doors: Iterable[DoorContribution],
    classify: Callable[[BaseException], ProblemSpec | None],
) -> None:
    """Refuse a surface whose doors do not prove their statuses: each door declares exemplars for
    exactly the endpoints it binds, covers every exception type it declares, and keeps no
    `exception_exemplars` beside them, and `classify` gives every exemplar a status with its
    endpoint bound, as a request is. Classifiers are pure, so this logs nothing.

    Raises:
        LookupError: A door breaks one of those rules, or the code registry maps no one status
            for a code on the endpoint whose routes raise it.
    """
    for door in doors:
        name = sorted(door.registry_endpoints.values()) or door.router.routes
        if door.exception_exemplars:
            raise LookupError(f"door {name} keeps exception_exemplars; use endpoint_exemplars")
        if set(door.endpoint_exemplars) != set(door.registry_endpoints.values()):
            raise LookupError(
                f"door {name} declares exemplars for {sorted(door.endpoint_exemplars)}"
            )
        exemplars = endpoint_exemplars((door,))
        for declared in door.exception_types:
            if not any(isinstance(exemplar, declared) for exemplar in exemplars):
                raise LookupError(f"door {name} has no exemplar of {declared.__qualname__}")
        for endpoint, declared_exemplars in door.endpoint_exemplars.items():
            for exemplar in declared_exemplars:
                with bound_registry_endpoint(endpoint):
                    spec = classify(exemplar)
                if spec is None:
                    raise LookupError(f"nothing classifies {exemplar!r} on {endpoint}")
