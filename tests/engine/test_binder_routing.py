"""A request routes to its binder once, in `core`, and every part binds through that one;
the app refuses to start when the configured binder asks Jev without a key."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import SecretStr

from search.engine import app, binders, config, core
from search.engine.binders import GeminiBinder
from tests.engine._harness import make_request, offline_resolve

_SRC = Path(__file__).parents[2] / "src"
_BINDERS = _SRC / "search" / "engine" / "binders"


def _configured_binder_callers() -> list[str]:
    """The source files outside `binders/` that call `configured_binder`."""
    return [
        str(path.relative_to(_SRC))
        for path in sorted(_SRC.rglob("*.py"))
        if _BINDERS not in path.parents
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "configured_binder"
    ]


def test_only_core_routes_a_request_to_its_binder():
    assert _configured_binder_callers() == ["search/engine/core.py"]


class _CountingBinder(GeminiBinder):
    def __init__(self) -> None:
        self.binds = 0

    async def bind(self, request, *, llm):
        self.binds += 1
        return await super().bind(request, llm=llm)


def test_the_request_routes_once_and_its_parts_bind_through_that_binder(monkeypatch):
    routed: list[str | None] = []
    binder = _CountingBinder()

    def route(query: str | None) -> GeminiBinder:
        routed.append(query)
        return binder

    monkeypatch.setattr(core, "configured_binder", route)

    response = offline_resolve(make_request("fertility rate in Kenya"))

    assert response.status == "definite"
    assert routed == ["fertility rate in Kenya"]
    assert binder.binds == 1


def test_the_default_binder_keys_as_before_the_routing():
    """Under the default settings the request routes to the Gemini binder, whose name and
    model are the ones the response cache key held before the routing."""
    binder = binders.configured_binder("fertility rate in Kenya")

    assert (binder.name, binder.model, binder.reads_query_text) == (
        "gemini",
        config.settings().engine_model,
        False,
    )


@pytest.mark.parametrize("binder", ["jev", "shadow"])
def test_the_app_refuses_to_start_a_jev_binder_without_a_key(search_settings, binder):
    search_settings(binder=binder)

    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        app.search_surface()


@pytest.mark.parametrize(("binder", "key"), [("gemini", ""), ("jev", "key"), ("shadow", "key")])
def test_the_key_check_passes_a_binder_that_has_what_it_needs(monkeypatch, binder, key):
    update = {"binder": binder, "jev_api_key": SecretStr(key)}
    monkeypatch.setattr(config, "_SETTINGS", config.settings().model_copy(update=update))

    app._require_jev_key()
