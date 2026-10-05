"""FastAPI helpers for composing route families into a `dfo_platform.app.Surface`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, Request

from dfo_platform import ProblemSpec


@dataclass(frozen=True)
class DoorContribution:
    """One route family's contribution to a composed surface.

    Routers carry full paths. Classifiers return `None` for unrecognised exceptions. Lifespans enter
    together. Middleware checks run synchronously before FastAPI parses the request body. Each
    check decides for itself, for example by path, whether a request concerns it.
    """

    router: APIRouter
    classify: Callable[[BaseException], ProblemSpec | None]
    exception_types: tuple[type[Exception], ...] = ()
    exception_exemplars: tuple[BaseException, ...] = ()
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None
    middleware: tuple[Callable[[Request], ProblemSpec | None], ...] = ()


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


def flatten(doors: Iterable[DoorContribution]) -> APIRouter:
    """Combine routes in registration order while keeping `.routes` populated with `APIRoute`
    objects. This FastAPI version wraps `include_router` results in a lazy `_IncludedRouter`.
    """
    router = APIRouter()
    for door in doors:
        router.routes.extend(door.router.routes)
    return router
