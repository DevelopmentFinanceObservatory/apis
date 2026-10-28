"""FastAPI helpers for composing route families into a `dfo_platform.app.Surface`."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

from fastapi import APIRouter, FastAPI, Request
from fastapi.routing import APIRoute

from dfo_platform import ProblemSpec
from dfo_platform.codes import bound_endpoint


@dataclass(frozen=True)
class DoorContribution:
    """One route family's contribution to a composed surface.

    Routers carry full paths. Classifiers return `None` for unrecognised exceptions. Lifespans enter
    together. Middleware checks run synchronously before FastAPI parses the request body. Each
    check decides for itself, for example by path, whether a request concerns it.
    `registry_endpoints` maps each route path to the code registry endpoint it serves
    (`dfo_platform.app.Surface.registry_endpoints`), and `public_paths` names the route paths any
    origin may read (`dfo_platform.app.Surface.public_paths`).
    """

    router: APIRouter
    classify: Callable[[BaseException], ProblemSpec | None]
    exception_types: tuple[type[Exception], ...] = ()
    exception_exemplars: tuple[BaseException, ...] = ()
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None
    middleware: tuple[Callable[[Request], ProblemSpec | None], ...] = ()
    registry_endpoints: Mapping[str, str] = field(default_factory=dict)
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


def route_registry_endpoints(router: APIRouter, endpoint: str) -> dict[str, str]:
    """Map every route on `router` to the one code registry `endpoint` its door serves."""
    return {route.path: endpoint for route in router.routes if isinstance(route, APIRoute)}
