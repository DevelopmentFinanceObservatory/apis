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
from dfo_platform.doors import DoorContribution, check_endpoint_exemplars, combine_classifiers
from presentation_api.app import presentation_surface
from presentation_api.codes import _UNCHANGED_RETRY_MODES, CODE_RETRY_MODE
from presentation_api.doors.components import classify_components_exception
from presentation_api.refusals import frame_unavailable_spec
from presentation_api.routes import RENDER_ENDPOINT, ROUTE_ENDPOINTS
from tests.code_registry import problem_statuses, registered_endpoints, unbound_routes

# The route of a real registry endpoint the exemplar contract tests bind or leave unbound.
_ROUTE = next(iter(ROUTE_ENDPOINTS))

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


def _write_refused() -> WriteRefused:
    return WriteRefused(subject_kind="component", subject_id="c", detail="d", fix="f")


def test_a_refusal_its_endpoint_leaves_unmapped_fails_at_startup():
    door = DoorContribution(
        router=APIRouter(),
        classify=lambda _exc: None,
        registry_endpoints={("GET", "/exemplar"): RENDER_ENDPOINT},
        endpoint_exemplars={RENDER_ENDPOINT: (_write_refused(),)},
    )

    with pytest.raises(LookupError, match=RENDER_ENDPOINT):
        check_endpoint_exemplars((door,), classify_components_exception)


@pytest.mark.parametrize(
    ("door", "rule"),
    [
        pytest.param(
            DoorContribution(
                router=APIRouter(),
                classify=classify_components_exception,
                registry_endpoints={("GET", "/exemplar"): RENDER_ENDPOINT},
                endpoint_exemplars={RENDER_ENDPOINT: (ValueError("exemplar"),)},
            ),
            "nothing classifies",
            id="an-exemplar-nothing-classifies",
        ),
        pytest.param(
            DoorContribution(
                router=APIRouter(),
                classify=classify_components_exception,
                registry_endpoints={("GET", "/exemplar"): RENDER_ENDPOINT},
                endpoint_exemplars={RENDER_ENDPOINT: ()},
            ),
            "declares no exemplars",
            id="an-empty-exemplar-tuple",
        ),
        pytest.param(
            DoorContribution(
                router=APIRouter(),
                classify=classify_components_exception,
                exception_exemplars=(_write_refused(),),
            ),
            "keeps exception_exemplars",
            id="exemplars-tied-to-no-endpoint",
        ),
        pytest.param(
            DoorContribution(
                router=APIRouter(),
                classify=classify_components_exception,
                registry_endpoints={("GET", "/exemplar"): ROUTE_ENDPOINTS[_ROUTE]},
            ),
            "declares exemplars",
            id="an-endpoint-without-exemplars",
        ),
        pytest.param(
            DoorContribution(
                router=APIRouter(),
                classify=classify_components_exception,
                endpoint_exemplars={ROUTE_ENDPOINTS[_ROUTE]: (_write_refused(),)},
            ),
            "declares exemplars",
            id="exemplars-for-an-unbound-endpoint",
        ),
        pytest.param(
            DoorContribution(
                router=APIRouter(),
                classify=classify_components_exception,
                exception_types=(WriteRefused, ValueError),
                registry_endpoints={("GET", "/exemplar"): ROUTE_ENDPOINTS[_ROUTE]},
                endpoint_exemplars={ROUTE_ENDPOINTS[_ROUTE]: (_write_refused(),)},
            ),
            "has no exemplar",
            id="a-declared-type-without-an-exemplar",
        ),
    ],
)
def test_a_door_that_breaks_the_exemplar_contract_fails_at_startup(
    door: DoorContribution, rule: str
):
    with pytest.raises(LookupError, match=rule):
        check_endpoint_exemplars((door,), door.classify)


def test_door_classifiers_leave_logging_to_the_platform(
    door_refusals: list[tuple[str, BaseException]],
    doors: tuple[DoorContribution, ...],
    caplog: pytest.LogCaptureFixture,
):
    """The startup sweep runs every classifier, so one that logs would log at every boot."""
    classify = combine_classifiers(*(door.classify for door in doors))
    with caplog.at_level(logging.DEBUG):
        for endpoint, exemplar in door_refusals:
            with bound_registry_endpoint(endpoint):
                classify(exemplar)

    assert caplog.records == []


@pytest.mark.parametrize(
    ("reason", "logged"),
    [
        ("io", True),
        ("recipe_unreadable", True),
        ("digest_mismatch", True),
        ("recipe_pending", False),
        ("expired", False),
        ("busy", False),
    ],
)
def test_a_frame_refusal_that_names_a_server_fault_carries_a_log_line(
    reason: FrameUnavailableReason, logged: bool
):
    with bound_registry_endpoint("presentation.shares"):
        spec = frame_unavailable_spec(FrameUnavailable(reason=reason, detail="exemplar"))

    assert (spec.log is not None) is logged
