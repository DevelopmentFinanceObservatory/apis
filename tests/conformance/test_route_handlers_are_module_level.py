"""Require module-level route handlers on the graph, search, and skills surfaces."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.routing import APIRoute

from dfo_platform.app import Surface
from graph_api.app import graph_surface
from search.engine.app import search_surface
from skills_api.app import skills_surface

_SURFACE_FACTORIES: tuple[Callable[[], Surface], ...] = (
    graph_surface,
    search_surface,
    skills_surface,
)


@pytest.mark.parametrize("build_surface", _SURFACE_FACTORIES, ids=lambda f: f.__module__)
def test_every_route_handler_is_a_module_level_function(
    build_surface: Callable[[], Surface],
) -> None:
    surface = build_surface()
    routes = [route for route in surface.router.routes if isinstance(route, APIRoute)]

    assert routes, f"{surface.name!r} surface mounted no APIRoute at all"
    for route in routes:
        qualname = route.endpoint.__qualname__
        assert "<locals>" not in qualname, (
            f"{surface.name!r} surface's own {route.path!r} route handler "
            f"{qualname} is a closure, not a module-level function"
        )
