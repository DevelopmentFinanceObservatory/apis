"""Every Presentation route but the meta routes binds the code registry endpoint its row of
`ROUTE_ENDPOINTS` names, and every refusal a door declares for an endpoint renders, through the
surface's classifier with that endpoint bound, at the one status the registry gives its code.
"""

from __future__ import annotations

import logging
from collections import Counter

import pytest
from fastapi import APIRouter

from assembler.errors import FrameUnavailable, WriteRefused
from assembler.fakes import FakeReleaseRegistry
from assembler.ports import FrameUnavailableReason
from dfo_platform import codes
from dfo_platform.codes import bound_registry_endpoint
from dfo_platform.doors import DoorContribution, check_endpoint_exemplars
from presentation_api.app import presentation_surface
from presentation_api.codes import _UNCHANGED_RETRY_MODES, CODE_RETRY_MODE
from presentation_api.doors.components import classify_components_exception
from presentation_api.refusals import frame_unavailable_spec
from presentation_api.routes import RENDER_ENDPOINT, ROUTE_ENDPOINTS
from tests.code_registry import problem_statuses, registered_endpoints, unbound_routes

# Routes that serve no registry endpoint: they refuse only with platform codes.
_UNBOUND_PATHS = frozenset(
    {
        "/api/presentation/healthz",
        "/api/presentation/capabilities",
        "/api/presentation/release",
    }
)


def test_every_presentation_route_binds_the_registry_endpoint_it_serves(
    release_registry: FakeReleaseRegistry,
):
    """The surface binds every Presentation endpoint the registry maps through
    `ROUTE_ENDPOINTS`, and leaves only the routes in `_UNBOUND_PATHS` unbound."""
    surface = presentation_surface(release_registry=release_registry)

    assert surface.registry_endpoints == ROUTE_ENDPOINTS
    assert set(surface.registry_endpoints.values()) == registered_endpoints("presentation.")
    assert {path for _, path in unbound_routes(surface)} == _UNBOUND_PATHS


def test_each_presentation_endpoint_belongs_to_one_door(doors: tuple[DoorContribution, ...]):
    """`classify_by_door` tries the door that owns the bound endpoint first, so a shared endpoint
    would hand one door's routes to the other."""
    owners = Counter(
        endpoint for door in doors for endpoint in set(door.registry_endpoints.values())
    )
    assert [endpoint for endpoint, count in owners.items() if count > 1] == []


def test_a_door_declares_exemplars_for_each_endpoint_and_every_type_it_raises(
    doors: tuple[DoorContribution, ...],
):
    for door in doors:
        assert set(door.endpoint_exemplars) == set(door.registry_endpoints.values())
        exemplars = tuple(
            exemplar for declared in door.endpoint_exemplars.values() for exemplar in declared
        )
        uncovered = [
            declared.__qualname__
            for declared in door.exception_types
            if not any(isinstance(exemplar, declared) for exemplar in exemplars)
        ]
        assert uncovered == []


def test_every_presentation_refusal_goes_out_at_its_registered_status(
    release_registry: FakeReleaseRegistry, door_refusals: list[tuple[str, BaseException]]
):
    """On a render route `retryable` holds only for a code the same request may pass later."""
    surface = presentation_surface(release_registry=release_registry)
    disagreements: set[tuple[str, str, str, int]] = set()
    swept: set[str] = set()
    for endpoint, exemplar in door_refusals:
        with bound_registry_endpoint(endpoint):
            spec = surface.classify(exemplar)
        assert spec is not None, repr(exemplar)
        swept.add(endpoint)
        if problem_statuses(spec.code, endpoint) != {spec.status}:
            disagreements.add((type(exemplar).__qualname__, endpoint, spec.code, spec.status))
        if endpoint == RENDER_ENDPOINT and spec.retryable:
            assert CODE_RETRY_MODE[spec.code] in _UNCHANGED_RETRY_MODES, repr(exemplar)
    assert swept == registered_endpoints("presentation.")
    assert sorted(disagreements) == []


def test_a_registry_that_drops_a_raised_code_fails_the_surface_build(
    monkeypatch: pytest.MonkeyPatch, release_registry: FakeReleaseRegistry
):
    real = codes._problem_statuses
    monkeypatch.setattr(
        codes, "_problem_statuses", lambda code: {} if code == "write_refused" else real(code)
    )

    with pytest.raises(LookupError):
        presentation_surface(release_registry=release_registry)


def test_a_refusal_its_endpoint_leaves_unmapped_fails_at_startup():
    door = DoorContribution(
        router=APIRouter(),
        classify=lambda _exc: None,
        endpoint_exemplars={
            RENDER_ENDPOINT: (
                WriteRefused(subject_kind="component", subject_id="c", detail="d", fix="f"),
            )
        },
    )

    with pytest.raises(LookupError):
        check_endpoint_exemplars((door,), classify_components_exception)

    assert logging.root.manager.disable == logging.NOTSET


def test_an_exemplar_nothing_classifies_fails_at_startup():
    door = DoorContribution(
        router=APIRouter(),
        classify=lambda _exc: None,
        endpoint_exemplars={RENDER_ENDPOINT: (ValueError("exemplar"),)},
    )

    with pytest.raises(LookupError):
        check_endpoint_exemplars((door,), door.classify)


@pytest.mark.parametrize(
    ("reason", "logged"),
    [("io", True), ("recipe_unreadable", True), ("recipe_pending", False), ("expired", False)],
)
def test_a_frame_refusal_that_names_a_server_fault_logs_an_error(
    caplog: pytest.LogCaptureFixture, reason: FrameUnavailableReason, logged: bool
):
    with bound_registry_endpoint("presentation.shares"), caplog.at_level(logging.ERROR):
        frame_unavailable_spec(FrameUnavailable(reason=reason, detail="exemplar"))

    assert bool(caplog.records) is logged
