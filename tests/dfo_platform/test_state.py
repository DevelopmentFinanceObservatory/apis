"""`dfo_platform.state.StateSlot`: the typed `app.state` accessor every
surface's own lifespan uses to publish its dependency bundle for `Depends`,
and every route handler uses to read it back.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from dfo_platform.state import StateNotSet, StateSlot


def test_set_then_call_returns_the_stored_value() -> None:
    slot: StateSlot[str] = StateSlot("widget")
    app = FastAPI()

    slot.set(app, "hello")

    @app.get("/echo")
    def echo(value: str = Depends(slot)) -> dict[str, str]:
        return {"value": value}

    with TestClient(app) as client:
        resp = client.get("/echo")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"value": "hello"}


def test_call_before_set_raises_state_not_set() -> None:
    slot: StateSlot[str] = StateSlot("widget")
    app = FastAPI()

    with pytest.raises(StateNotSet, match="widget"):
        slot(_request_for(app))


def test_clear_removes_the_stored_value() -> None:
    slot: StateSlot[str] = StateSlot("widget")
    app = FastAPI()
    slot.set(app, "hello")

    slot.clear(app)

    with pytest.raises(StateNotSet):
        slot(_request_for(app))


def test_clear_before_any_set_is_a_no_op() -> None:
    slot: StateSlot[str] = StateSlot("widget")
    app = FastAPI()

    slot.clear(app)

    with pytest.raises(StateNotSet):
        slot(_request_for(app))


class _NotReady(Exception):
    """Zero-argument stand-in for a surface's not-ready exception."""


def test_call_before_set_with_unset_exception_raises_it_instead_of_state_not_set() -> None:
    slot: StateSlot[str] = StateSlot("widget", unset_exception=_NotReady)
    app = FastAPI()

    with pytest.raises(_NotReady):
        slot(_request_for(app))


def test_two_slots_on_one_app_never_collide() -> None:
    """Each slot reads and clears only its own app-state entry."""
    graph_slot: StateSlot[str] = StateSlot("graph_deps")
    skills_slot: StateSlot[str] = StateSlot("skills_deps")
    app = FastAPI()

    graph_slot.set(app, "graph-bundle")
    skills_slot.set(app, "skills-bundle")

    assert graph_slot(_request_for(app)) == "graph-bundle"
    assert skills_slot(_request_for(app)) == "skills-bundle"

    graph_slot.clear(app)

    with pytest.raises(StateNotSet):
        graph_slot(_request_for(app))
    assert skills_slot(_request_for(app)) == "skills-bundle"


def _request_for(app: FastAPI) -> Request:
    """Build a request with `app` set for a direct `StateSlot` call."""
    return Request({"type": "http", "app": app, "headers": []})
