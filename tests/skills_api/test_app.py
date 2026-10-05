"""Pins Skills surface construction, lifespan ownership, and the environment-gated mount."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import skills_api.app as app_module
from dfo_apis.app import all_surfaces, deployed_surfaces
from dfo_platform import HealthResponse
from dfo_platform.app import create_app
from skills_api.app import skills_surface
from skills_api.version import SKILLS_CAPABILITY_VERSION


def test_surface_name_prefix_and_capability_version():
    surface = skills_surface(skills_index=object())

    assert surface.name == "skills"
    assert surface.prefix == "/api/skills"
    assert surface.capability_version == SKILLS_CAPABILITY_VERSION


def test_healthz_answers_200_through_the_composed_app():
    app = create_app([skills_surface(skills_index=object())])

    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.get("/api/skills/healthz")

    assert resp.status_code == 200, resp.text
    body = HealthResponse.model_validate(resp.json())
    assert body.capability_version == SKILLS_CAPABILITY_VERSION


class TestLifespanSkillsIndexOwnership:
    """The shared `skills_index.SharedSkillsIndex` this surface's own
    lifespan acquires when none is injected, and never touches when one is.
    `acquire_shared_index`/`release_shared_index` are stubbed doubles here
    (real acquire/release, and the background refresh task they own, are
    already exercised by `tests/skills_index/` and
    `tests/engine/test_lifecycle.py::TestSkillsIndexLifespanOwnership`);
    this class proves only that `skills_surface`'s own lifespan calls them
    the right number of times, with the right arguments, at the right
    lifecycle points.
    """

    def test_self_acquired_index_passes_graph_none_and_releases_on_shutdown(self):
        sentinel = object()
        acquire_calls: list[dict[str, object]] = []
        release_calls: list[object] = []

        async def _spy_acquire(**kwargs):
            acquire_calls.append(kwargs)
            return sentinel

        async def _spy_release(shared):
            release_calls.append(shared)

        with (
            patch.object(app_module, "acquire_shared_index", _spy_acquire),
            patch.object(app_module, "release_shared_index", _spy_release),
        ):
            app = create_app([skills_surface()])
            with TestClient(app, raise_server_exceptions=True) as client:
                resp = client.get("/api/skills/healthz")
                assert resp.status_code == 200, resp.text
                # Not yet released while the app is still serving.
                assert release_calls == []

        assert acquire_calls == [{"graph": None}]
        assert release_calls == [sentinel]

    def test_injected_index_is_never_acquired_or_released(self):
        def _fail_acquire(*_args, **_kwargs):
            raise AssertionError("acquire_shared_index must not run when skills_index is injected")

        def _fail_release(*_args, **_kwargs):
            raise AssertionError("release_shared_index must not run when skills_index is injected")

        with (
            patch.object(app_module, "acquire_shared_index", _fail_acquire),
            patch.object(app_module, "release_shared_index", _fail_release),
        ):
            app = create_app([skills_surface(skills_index=object())])
            with TestClient(app, raise_server_exceptions=True) as client:
                resp = client.get("/api/skills/healthz")
                assert resp.status_code == 200, resp.text


class TestSkillsSurfaceGate:
    """`dfo_apis.app`'s env-gated mount of the Skills surface.

    `DFO_SKILLS_SURFACE` must be truthy ("1" or "true", case-insensitive)
    for `deployed_surfaces()` to include the Skills surface -- unset or any
    other value leaves it out. `all_surfaces()` always includes it (see
    `tests/platform/test_surface_conformance.py`, which parametrizes off
    that list instead).
    """

    @pytest.fixture(autouse=True)
    def _clear_skills_surface_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DFO_SKILLS_SURFACE", raising=False)

    def test_all_surfaces_always_includes_skills_regardless_of_the_flag(self) -> None:
        assert "skills" in [s.name for s in all_surfaces()]

    @pytest.mark.parametrize(
        "value",
        (
            pytest.param("1", id="one"),
            pytest.param("true", id="true"),
            pytest.param("True", id="title-true"),
            pytest.param("TRUE", id="upper-true"),
            pytest.param(" 1 ", id="padded-one"),
            pytest.param(" true ", id="padded-true"),
        ),
    )
    def test_truthy_value_mounts_the_skills_surface(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("DFO_SKILLS_SURFACE", value)
        assert "skills" in [s.name for s in deployed_surfaces()]

    @pytest.mark.parametrize(
        "value",
        (
            pytest.param("0", id="zero"),
            pytest.param("false", id="false"),
            pytest.param("False", id="title-false"),
            pytest.param("yes", id="yes"),
            pytest.param("on", id="on"),
            pytest.param("", id="empty"),
            pytest.param(None, id="unset"),
        ),
    )
    def test_non_truthy_or_unset_value_omits_the_skills_surface(
        self, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        if value is None:
            monkeypatch.delenv("DFO_SKILLS_SURFACE", raising=False)
        else:
            monkeypatch.setenv("DFO_SKILLS_SURFACE", value)
        assert "skills" not in [s.name for s in deployed_surfaces()]
