"""A refusal renders at the code registry status of the endpoint its route serves."""

from __future__ import annotations

import pytest
from fastapi import APIRouter, Request
from fastapi.testclient import TestClient

from dfo_platform import ProblemSpec, codes
from dfo_platform.app import Surface, SurfaceConfigurationError, create_app
from dfo_platform.codes import bound_registry_endpoint, registered_status
from dfo_platform.problem import Refusal, classify_refusal

_CODE = "facet_binding_unsupported"
# A code the registry gives two statuses, one per endpoint.
_STATUSES = {"t.a": frozenset({422}), "t.b": frozenset({501})}


class _Unsupported(Refusal):
    code = _CODE


@pytest.fixture(autouse=True)
def _two_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    real = codes._problem_statuses
    monkeypatch.setattr(
        codes, "_problem_statuses", lambda code: _STATUSES if code == _CODE else real(code)
    )


def _admit(request: Request) -> ProblemSpec | None:
    if request.url.path != "/api/t/a/admitted":
        return None
    return classify_refusal(_Unsupported())


def _client(**overrides: object) -> TestClient:
    router = APIRouter()

    @router.get("/api/t/a")
    async def route_a() -> None:
        raise _Unsupported()

    @router.post("/api/t/a")
    async def route_a_posted() -> None:
        raise _Unsupported()

    @router.get("/api/t/a/admitted")
    async def route_admitted() -> None:
        return None

    @router.get("/api/t/b/{item}")
    async def route_b(item: str) -> None:
        raise _Unsupported()

    @router.get("/api/t/unbound")
    async def route_unbound() -> None:
        raise _Unsupported()

    surface = Surface(
        name="t",
        prefix="/api/t",
        router=router,
        capability_version="v-t",
        codes={},
        classify=classify_refusal,
        exception_types=(_Unsupported,),
        middleware=(_admit,),
        registry_endpoints=overrides.get(
            "registry_endpoints",
            {
                ("GET", "/api/t/a"): "t.a",
                ("POST", "/api/t/a"): "t.b",
                ("GET", "/api/t/a/admitted"): "t.a",
                ("GET", "/api/t/b/{item}"): "t.b",
            },
        ),
    )
    return TestClient(create_app([surface]), raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("path", "status"),
    [
        pytest.param("/api/t/a", 422, id="raised-on-a"),
        pytest.param("/api/t/a/admitted", 422, id="admission-check-on-a"),
        pytest.param("/api/t/b/x", 501, id="raised-on-b"),
        # No endpoint is bound and the code has two statuses, so classification fails loudly.
        pytest.param("/api/t/unbound", 500, id="unbound"),
    ],
)
def test_a_refusal_renders_at_the_status_of_its_routes_endpoint(path: str, status: int) -> None:
    response = _client().get(path)

    assert response.status_code == status, response.text


def test_two_methods_on_one_path_serve_their_own_endpoints() -> None:
    assert _client().post("/api/t/a").status_code == 501


def test_a_registry_endpoint_naming_no_route_is_refused() -> None:
    with pytest.raises(SurfaceConfigurationError):
        _client(registry_endpoints={("GET", "/api/t/missing"): "t.a"})


@pytest.mark.parametrize(
    ("statuses", "endpoint", "expected"),
    [
        pytest.param(_STATUSES, "t.b", 501, id="the-bound-endpoints-status"),
        pytest.param({"t.a": frozenset({410})}, "t.b", LookupError, id="unmapped-on-the-bound"),
        pytest.param({"t.a": frozenset({410})}, None, 410, id="unbound-one-status"),
        pytest.param(_STATUSES, None, LookupError, id="unbound-two-statuses"),
        pytest.param({"t.a": frozenset({410, 422})}, "t.a", LookupError, id="two-on-endpoint"),
        pytest.param({}, "t.a", LookupError, id="unmapped"),
    ],
)
def test_registered_status_reads_the_bound_endpoint_or_unbound_the_one_status_overall(
    monkeypatch: pytest.MonkeyPatch,
    statuses: dict[str, frozenset[int]],
    endpoint: str | None,
    expected: int | type[Exception],
) -> None:
    monkeypatch.setattr(codes, "_problem_statuses", lambda _code: statuses)

    with bound_registry_endpoint(endpoint):
        if isinstance(expected, int):
            assert registered_status(_CODE) == expected
        else:
            with pytest.raises(expected):
                registered_status(_CODE)
