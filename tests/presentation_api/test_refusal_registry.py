"""Every Presentation route but the meta routes binds the code registry endpoint it serves, and
every exemplar and `FrameUnavailable` reason a door's routes raise goes out, through the surface's
classifier with the route's endpoint bound, at the one status the registry gives its code there.
"""

from __future__ import annotations

import logging
from collections import Counter

import pytest

from assembler.errors import (
    BindingInvalid,
    ConstraintViolation,
    FrameUnavailable,
    ReleaseMismatch,
    ResourceExhausted,
    SchemaInvalid,
    WriteRefused,
)
from assembler.fakes import FakeReleaseRegistry
from assembler.pipeline import ComponentHoldsNoFrame
from assembler.ports import FrameUnavailableReason
from datasets_api.allowance import AllowanceExhausted
from datasets_api.door_executor import build_door_executor
from datasets_api.live_refresh import RefreshNotPermitted
from dfo_platform.codes import bound_registry_endpoint
from dfo_platform.doors import DoorContribution
from presentation_api.app import presentation_surface
from presentation_api.codes import (
    _UNCHANGED_RETRY_MODES,
    CODE_RETRY_MODE,
    RENDER_ENDPOINT,
    served_status,
)
from presentation_api.doors.components import build_components_door
from presentation_api.doors.frames import build_frames_door
from presentation_api.doors.render import RenderTargetMismatch
from presentation_api.doors.shares import build_shares_door
from presentation_api.refusals import _FRAME_UNAVAILABLE_OUTCOME, frame_unavailable_spec
from tests.code_registry import problem_statuses, registered_endpoints, unbound_routes

# Routes that serve no registry endpoint: they refuse only with platform codes.
_UNBOUND_PATHS = frozenset(
    {
        "/api/presentation/healthz",
        "/api/presentation/capabilities",
        "/api/presentation/release",
    }
)

# The endpoints whose routes load or pin a frame.
_FRAME_ENDPOINTS = frozenset(
    {
        "presentation.write",
        "presentation.component_data",
        "presentation.frames",
        "presentation.render",
        "presentation.shares",
    }
)

_RENDER_ONLY = (RefreshNotPermitted, RenderTargetMismatch, ComponentHoldsNoFrame)
_WRITES_AND_DRAWS = (
    AllowanceExhausted,
    WriteRefused,
    SchemaInvalid,
    ConstraintViolation,
    ResourceExhausted,
)

# What each endpoint's routes do not raise, of the exemplars of the door that serves them.
_NOT_RAISED: dict[str, tuple[type[BaseException], ...]] = {
    "presentation.write": _RENDER_ONLY,
    "presentation.component_read": (*_RENDER_ONLY, *_WRITES_AND_DRAWS, BindingInvalid),
    "presentation.component_data": (*_RENDER_ONLY, *_WRITES_AND_DRAWS, ReleaseMismatch),
    "presentation.render": (AllowanceExhausted, WriteRefused),
    "presentation.frames": (),
    "presentation.shares": (RefreshNotPermitted,),
}


@pytest.fixture
def doors(release_registry: FakeReleaseRegistry) -> tuple[DoorContribution, ...]:
    reads = build_door_executor(1, "test-frames-bare", surface="presentation")
    return (
        build_components_door(release_registry=release_registry, bare_reads=reads),
        build_frames_door(bare_reads=reads),
        build_shares_door(release_registry=release_registry),
    )


def test_every_presentation_route_binds_the_registry_endpoint_it_serves(
    release_registry: FakeReleaseRegistry,
):
    """The surface binds every Presentation endpoint the registry maps, each to a route it
    serves, and leaves only the routes in `_UNBOUND_PATHS` unbound."""
    surface = presentation_surface(release_registry=release_registry)

    assert set(surface.registry_endpoints.values()) == registered_endpoints("presentation.")
    assert {path for _, path in unbound_routes(surface)} == _UNBOUND_PATHS


def test_each_presentation_endpoint_belongs_to_one_door(doors: tuple[DoorContribution, ...]):
    """`classify_by_door` tries the door that owns the bound endpoint first, so a shared endpoint
    would hand one door's routes to the other."""
    owners = Counter(
        endpoint for door in doors for endpoint in set(door.registry_endpoints.values())
    )
    assert [endpoint for endpoint, count in owners.items() if count > 1] == []


def _raised_on(endpoint: str, exemplar: BaseException) -> bool:
    """Whether a route of `endpoint` raises `exemplar`: a door's exemplars cover every route it
    serves."""
    return not isinstance(exemplar, _NOT_RAISED[endpoint])


def test_every_presentation_refusal_goes_out_at_its_registered_status(
    release_registry: FakeReleaseRegistry, doors: tuple[DoorContribution, ...]
):
    """A code a route raises and the registry leaves off its endpoint fails here, and on a render
    route `retryable` holds only for a code the same request may pass later."""
    surface = presentation_surface(release_registry=release_registry)
    reasons = tuple(
        FrameUnavailable(reason=reason, detail="exemplar") for reason in _FRAME_UNAVAILABLE_OUTCOME
    )
    disagreements: set[tuple[str, str, str, int]] = set()
    swept: set[str] = set()
    for door in doors:
        for endpoint in set(door.registry_endpoints.values()):
            frame_refusals = reasons if endpoint in _FRAME_ENDPOINTS else ()
            for exemplar in (*door.exception_exemplars, *frame_refusals):
                if not _raised_on(endpoint, exemplar):
                    continue
                with bound_registry_endpoint(endpoint):
                    spec = surface.classify(exemplar)
                assert spec is not None, repr(exemplar)
                swept.add(endpoint)
                if problem_statuses(spec.code, endpoint) != {spec.status}:
                    disagreements.add(
                        (type(exemplar).__qualname__, endpoint, spec.code, spec.status)
                    )
                if endpoint == RENDER_ENDPOINT and spec.retryable:
                    assert CODE_RETRY_MODE[spec.code] in _UNCHANGED_RETRY_MODES, repr(exemplar)
    assert swept == registered_endpoints("presentation.")
    assert sorted(disagreements) == []


@pytest.mark.parametrize(
    ("reason", "logged"),
    [("io", True), ("recipe_unreadable", True), ("pending", False), ("expired", False)],
)
def test_a_frame_refusal_that_names_a_server_fault_logs_an_error(
    caplog: pytest.LogCaptureFixture, reason: FrameUnavailableReason, logged: bool
):
    with bound_registry_endpoint("presentation.shares"), caplog.at_level(logging.ERROR):
        frame_unavailable_spec(FrameUnavailable(reason=reason, detail="exemplar"))

    assert bool(caplog.records) is logged


def test_a_code_the_bound_endpoint_leaves_unmapped_keeps_its_refusal_and_logs(
    caplog: pytest.LogCaptureFixture,
):
    with bound_registry_endpoint(RENDER_ENDPOINT), caplog.at_level(logging.ERROR):
        status = served_status("write_refused")

    assert status == 403
    assert "write_refused" in caplog.text
