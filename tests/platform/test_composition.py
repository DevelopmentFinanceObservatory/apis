"""Tests surface routing, platform fallback, the shared code endpoint, and composition errors."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from dfo_platform.app import (
    Surface,
    SurfaceConfigurationError,
    _matching_surface,
    _within_prefix,
    create_app,
)
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, CodeCollision
from dfo_platform.problem import ProblemSpec
from dfo_platform.version import PLATFORM_CAPABILITY_VERSION


class DummyError(Exception):
    """A surface-specific exception type, classified differently per surface."""


def _make_surface(name: str, prefix: str, code: str, capability_version: str) -> Surface:
    """Builds a surface that classifies `DummyError` under its prefix."""
    router = APIRouter()

    @router.get(f"{prefix}/ping")
    async def ping():
        return {"surface": name}

    @router.get(f"{prefix}/boom")
    async def boom():
        raise DummyError(f"{name} boom")

    catalogue = {code: (f"{name.title()} Title", f"{name} default detail")}

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, DummyError):
            return ProblemSpec(status=400, code=code, retryable=False)
        return None

    return Surface(
        name=name,
        prefix=prefix,
        router=router,
        capability_version=capability_version,
        codes=catalogue,
        classify=classify,
        exception_types=(DummyError,),
    )


@pytest.fixture
def surface_a() -> Surface:
    return _make_surface("a", "/api/a", "a_code", "va")


@pytest.fixture
def surface_b() -> Surface:
    return _make_surface("b", "/api/b", "b_code", "vb")


def test_each_surfaces_own_handler_runs_for_its_own_prefix(surface_a, surface_b):
    app = create_app([surface_a, surface_b])
    client = TestClient(app, raise_server_exceptions=False)

    resp_a = client.get("/api/a/boom")
    assert resp_a.status_code == 400
    body_a = resp_a.json()
    assert body_a["code"] == "a_code"
    assert body_a["capability_version"] == "va"
    assert body_a["title"] == "A Title"
    assert body_a["detail"] == "a default detail"

    resp_b = client.get("/api/b/boom")
    assert resp_b.status_code == 400
    body_b = resp_b.json()
    assert body_b["code"] == "b_code"
    assert body_b["capability_version"] == "vb"
    assert body_b["title"] == "B Title"
    assert body_b["detail"] == "b default detail"


def test_no_matching_surface_falls_back_to_platform_default():
    """A route outside every surface prefix uses the platform's `internal_error` envelope."""
    router = APIRouter()

    @router.get("/api/a/ping")
    async def ping():
        return {"ok": True}

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"a_code": ("A Title", "a default detail")},
    )
    app = create_app([surface])

    async def boom():
        raise RuntimeError("boom")

    app.add_api_route("/unmatched/boom", boom, methods=["GET"])

    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/unmatched/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    # Outside every surface's prefix, so the response carries the platform's
    # own version.
    assert body["capability_version"] == PLATFORM_CAPABILITY_VERSION


def test_codes_endpoint_serves_every_mounted_catalogue(surface_a, surface_b):
    app = create_app([surface_a, surface_b])
    client = TestClient(app)

    resp_a = client.get("/codes/a_code")
    assert resp_a.status_code == 200
    assert resp_a.json() == {
        "code": "a_code",
        "classification": None,
        "definition": None,
        "lifecycle": None,
        "mappings": [],
        "title": "A Title",
        "detail": "a default detail",
    }

    resp_b = client.get("/codes/b_code")
    assert resp_b.status_code == 200
    assert resp_b.json() == {
        "code": "b_code",
        "classification": None,
        "definition": None,
        "lifecycle": None,
        "mappings": [],
        "title": "B Title",
        "detail": "b default detail",
    }

    resp_platform = client.get("/codes/internal_error")
    assert resp_platform.status_code == 200
    assert resp_platform.json()["code"] == "internal_error"


def test_codes_endpoint_404_carries_the_full_problem_body(surface_a, surface_b):
    """An unknown code returns the full uncached problem-details envelope."""
    app = create_app([surface_a, surface_b])
    client = TestClient(app)

    resp = client.get("/codes/nonexistent_code")
    assert resp.status_code == 404
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    for key in (
        "code",
        "type",
        "title",
        "detail",
        "status",
        "trace_id",
        "capability_version",
        "retryable",
    ):
        assert key in body
    assert body["code"] == "code_not_found"
    assert body["type"] == "/codes/code_not_found"


def test_duplicate_code_across_surfaces_raises_codecollision_at_compose_time():
    surface_a = _make_surface("a", "/api/a", "shared_code", "va")
    surface_b = _make_surface("b", "/api/b", "shared_code", "vb")
    with pytest.raises(CodeCollision):
        create_app([surface_a, surface_b])


def test_create_app_refuses_a_route_outside_its_surfaces_prefix():
    """Every route must start with its surface prefix."""
    router = APIRouter()

    @router.get("/api/wrong/ping")
    async def ping():
        return {"ok": True}

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"a_code": ("A Title", "a default detail")},
    )
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([surface])
    message = str(exc_info.value)
    assert "'a'" in message
    assert "/api/wrong/ping" in message


def test_create_app_refuses_two_surfaces_sharing_a_prefix(surface_a):
    router = APIRouter()

    @router.get("/api/a/other")
    async def other():
        return {"ok": True}

    colliding = Surface(
        name="also-a",
        prefix="/api/a",
        router=router,
        capability_version="vb",
        codes={"other_code": ("Other", "other detail")},
    )
    with pytest.raises(SurfaceConfigurationError, match="also-a"):
        create_app([surface_a, colliding])


class _NonDefaultOnlyError(Exception):
    """An exception classified only by a non-default surface."""


def test_exception_type_declared_only_by_non_default_surface_falls_back_to_platform_default():
    """An exception uses the matched surface's handler set, then the platform fallback."""
    default_router = APIRouter()

    @default_router.get("/api/default/ping")
    async def default_ping():
        return {"ok": True}

    # The route belongs to the default surface. The other surface classifies this error.
    @default_router.get("/api/default/boom")
    async def default_boom():
        raise _NonDefaultOnlyError("raised outside the declaring surface's prefix")

    non_default_router = APIRouter()

    @non_default_router.get("/api/nondefault/ping")
    async def nondefault_ping():
        return {"ok": True}

    def classify_non_default(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _NonDefaultOnlyError):
            return ProblemSpec(status=422, code="nondefault_handled", retryable=False)
        return None

    default_surface = Surface(
        name="default",
        prefix="/api/default",
        router=default_router,
        capability_version="v-default",
        codes={"default_code": ("Default", "default detail")},
    )
    non_default_surface = Surface(
        name="non-default",
        prefix="/api/nondefault",
        router=non_default_router,
        capability_version="v-nondefault",
        codes={
            "nondefault_code": ("Nondefault", "nondefault detail"),
            "nondefault_handled": ("Nondefault Handled", "nondefault handled detail"),
        },
        classify=classify_non_default,
        exception_types=(_NonDefaultOnlyError,),
    )

    app = create_app([default_surface, non_default_surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/default/boom")

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    assert body["capability_version"] == "v-default"


def test_matched_surface_with_no_validation_handler_gets_its_own_capability_version():
    router = APIRouter()

    @router.get("/api/a/ping")
    async def ping():
        return {"ok": True}

    @router.post("/api/a/echo")
    async def echo(payload: dict):
        return payload

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="v-a",
        codes={"a_code": ("A", "a detail")},
        # No RequestValidationError handler of its own -- relies on the
        # platform default.
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    # Malformed JSON body triggers RequestValidationError.
    resp = client.post(
        "/api/a/echo", content=b"not json", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 400
    assert resp.json()["capability_version"] == "v-a"


def test_matched_surface_with_no_timeout_handler_gets_its_own_capability_version():
    router = APIRouter()

    @router.get("/api/a/boom")
    async def boom():
        raise TimeoutError("slow")

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="v-a",
        codes={"a_code": ("A", "a detail")},
    )
    other_router = APIRouter()

    @other_router.get("/api/b/ping")
    async def ping():
        return {"ok": True}

    other = Surface(
        name="b",
        prefix="/api/b",
        router=other_router,
        capability_version="v-b",
        codes={"b_code": ("B", "b detail")},
    )
    app = create_app([other, surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/boom")
    assert resp.status_code == 504
    assert resp.json()["capability_version"] == "v-a"


def test_matched_surface_with_no_catchall_handler_gets_its_own_capability_version():
    router = APIRouter()

    @router.get("/api/a/boom")
    async def boom():
        raise RuntimeError("boom")

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="v-a",
        codes={"a_code": ("A", "a detail")},
    )
    other_router = APIRouter()

    @other_router.get("/api/b/ping")
    async def ping():
        return {"ok": True}

    other = Surface(
        name="b",
        prefix="/api/b",
        router=other_router,
        capability_version="v-b",
        codes={"b_code": ("B", "b detail")},
    )
    app = create_app([other, surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/boom")
    assert resp.status_code == 500
    assert resp.json()["capability_version"] == "v-a"


class _BaseError(Exception):
    """A surface-classified base exception type."""


class _ChildError(_BaseError):
    """A subclass another surface's own `classify` recognizes."""


def _mro_surface_a() -> Surface:
    """Classifies the base error, including subclasses, on surface A."""
    router = APIRouter()

    @router.get("/api/a/raise-child")
    async def raise_child():
        raise _ChildError("child raised on A's path")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _BaseError):
            return ProblemSpec(status=409, code="a_refused", retryable=False)
        return None

    return Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"a_refused": ("A refused", "detail")},
        classify=classify,
        exception_types=(_BaseError,),
    )


def _mro_surface_b() -> Surface:
    """Classifies the subclass on surface B."""
    router = APIRouter()

    @router.get("/api/b/raise-child")
    async def raise_child():
        raise _ChildError("child raised on B's path")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ChildError):
            return ProblemSpec(status=422, code="b_refused", retryable=False)
        return None

    return Surface(
        name="b",
        prefix="/api/b",
        router=router,
        capability_version="vb",
        codes={"b_refused": ("B refused", "detail")},
        classify=classify,
        exception_types=(_ChildError,),
    )


def test_mounting_a_sibling_subclass_handler_does_not_change_the_other_surfaces_dispatch():
    """Mounting a sibling surface preserves the first surface's exception response."""
    surface_a = _mro_surface_a()
    surface_b = _mro_surface_b()

    alone = TestClient(create_app([surface_a]), raise_server_exceptions=False)
    resp_alone = alone.get("/api/a/raise-child")

    both = TestClient(create_app([surface_a, surface_b]), raise_server_exceptions=False)
    resp_both = both.get("/api/a/raise-child")

    assert resp_alone.status_code == resp_both.status_code == 409
    body_alone = resp_alone.json()
    body_both = resp_both.json()
    # Trace IDs differ per request; the remaining fields must match.
    del body_alone["trace_id"], body_both["trace_id"]
    assert body_alone == body_both
    assert body_alone["code"] == "a_refused"
    assert body_alone["capability_version"] == "va"


def test_subclass_raised_on_its_declaring_surfaces_own_path_gets_its_own_handler():
    surface_a = _mro_surface_a()
    surface_b = _mro_surface_b()
    app = create_app([surface_a, surface_b])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/b/raise-child")
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "b_refused"
    assert body["capability_version"] == "vb"


def test_surface_declaring_both_base_and_subclass_handlers_prefers_the_subclass():
    """Within one classifier, the first matching `isinstance` branch wins."""
    router = APIRouter()

    @router.get("/api/a/raise-child")
    async def raise_child():
        raise _ChildError("child raised on A's path")

    @router.get("/api/a/raise-base")
    async def raise_base():
        raise _BaseError("base raised on A's path")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, _ChildError):
            return ProblemSpec(status=422, code="child_handler", retryable=False)
        if isinstance(exc, _BaseError):
            return ProblemSpec(status=409, code="base_handler", retryable=False)
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={
            "a_code": ("A", "a detail"),
            "base_handler": ("Base", "base detail"),
            "child_handler": ("Child", "child detail"),
        },
        classify=classify,
        exception_types=(_BaseError, _ChildError),
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp_child = client.get("/api/a/raise-child")
    assert resp_child.status_code == 422
    assert resp_child.json()["code"] == "child_handler"

    resp_base = client.get("/api/a/raise-base")
    assert resp_base.status_code == 409
    assert resp_base.json()["code"] == "base_handler"


def test_within_prefix_respects_segment_boundaries():
    assert _within_prefix("/api/search", "/api/search")
    assert _within_prefix("/api/search/resolve", "/api/search")
    assert _within_prefix("/api/search/resolve", "/api/search/")
    # The shared string prefix has no segment boundary.
    assert not _within_prefix("/api/search-admin/purge", "/api/search")
    assert not _within_prefix("/api/searc", "/api/search")


def test_create_app_refuses_a_route_outside_its_prefix_by_segment_boundary():
    """A string-prefix match without a path-segment boundary is outside the surface prefix."""
    router = APIRouter()

    @router.get("/api/search-admin/purge")
    async def purge():
        return {"ok": True}

    surface = Surface(
        name="search",
        prefix="/api/search",
        router=router,
        capability_version="v1",
        codes={"search_code": ("Search", "search detail")},
    )
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([surface])
    message = str(exc_info.value)
    assert "'search'" in message
    assert "/api/search-admin/purge" in message


def test_create_app_refuses_nested_surface_prefixes():
    """Nested prefixes can make FastAPI route dispatch and exception ownership disagree."""
    broad_router = APIRouter()

    @broad_router.get("/api/alpha/resolve")
    async def broad_resolve():
        return {"ok": True}

    narrow_router = APIRouter()

    @narrow_router.get("/api/alpha/inner/resolve")
    async def narrow_resolve():
        return {"ok": True}

    broad = Surface(
        name="broad",
        prefix="/api/alpha",
        router=broad_router,
        capability_version="v-broad",
        codes={"broad_code": ("Broad", "broad detail")},
    )
    narrow = Surface(
        name="narrow",
        prefix="/api/alpha/inner",
        router=narrow_router,
        capability_version="v-narrow",
        codes={"narrow_code": ("Narrow", "narrow detail")},
    )
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([broad, narrow])
    message = str(exc_info.value)
    assert "nested prefixes" in message
    assert "'broad'" in message
    assert "'narrow'" in message
    assert "/api/alpha" in message
    assert "/api/alpha/inner" in message


def test_matching_surface_still_picks_the_longest_genuine_match():
    """Direct calls select the longest matching prefix; app composition rejects nested prefixes."""
    broad = Surface(
        name="broad",
        prefix="/api/alpha",
        router=APIRouter(),
        capability_version="v-broad",
        codes={},
    )
    narrow = Surface(
        name="narrow",
        prefix="/api/alpha/inner",
        router=APIRouter(),
        capability_version="v-narrow",
        codes={},
    )
    assert _matching_surface("/api/alpha/inner/resolve", [broad, narrow]) is narrow
    assert _matching_surface("/api/alpha/inner/resolve", [narrow, broad]) is narrow
    assert _matching_surface("/api/alpha/other", [broad, narrow]) is broad
    assert _matching_surface("/unrelated", [broad, narrow]) is None


def test_create_app_refuses_an_empty_surface_list():
    with pytest.raises(SurfaceConfigurationError, match="at least one surface"):
        create_app([])


def test_create_app_refuses_a_prefix_claiming_the_platform_codes_namespace():
    """The `/codes` prefix is reserved for the platform registry endpoint."""
    surface = _make_surface("codes-thief", "/codes", "thief_code", "v1")
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([surface])
    message = str(exc_info.value)
    assert "'codes-thief'" in message
    assert "/codes" in message


def test_create_app_refuses_the_root_prefix():
    surface = _make_surface("root", "/", "root_code", "v1")
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([surface])
    message = str(exc_info.value)
    assert "'root'" in message
    assert "'/'" in message


def test_create_app_refuses_a_bare_api_prefix_with_no_segment():
    surface = Surface(
        name="bare",
        prefix="/api",
        router=APIRouter(),
        capability_version="v1",
        codes={},
    )
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([surface])
    assert "'bare'" in str(exc_info.value)
    assert "'/api'" in str(exc_info.value)


def test_create_app_refuses_a_prefix_with_a_trailing_slash():
    surface = Surface(
        name="trailing",
        prefix="/api/x/",
        router=APIRouter(),
        capability_version="v1",
        codes={},
    )
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([surface])
    assert "'trailing'" in str(exc_info.value)
    assert "/api/x/" in str(exc_info.value)


def test_create_app_refuses_a_prefix_with_an_empty_segment():
    surface = Surface(
        name="doubled",
        prefix="/api//x",
        router=APIRouter(),
        capability_version="v1",
        codes={},
    )
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([surface])
    assert "'doubled'" in str(exc_info.value)
    assert "/api//x" in str(exc_info.value)


def test_create_app_refuses_two_surfaces_sharing_a_name():
    first = _make_surface("dup", "/api/one", "one_code", "v1")
    second = _make_surface("dup", "/api/two", "two_code", "v2")
    with pytest.raises(SurfaceConfigurationError) as exc_info:
        create_app([first, second])
    assert "'dup'" in str(exc_info.value)


def test_create_app_composes_two_surfaces_under_the_api_namespace():
    """Distinct, non-nesting surface prefixes under `/api` compose successfully."""
    search = _make_surface("search", "/api/search", "search_code", "v-search")
    datasets = _make_surface("datasets", "/api/datasets", "datasets_code", "v-datasets")
    app = create_app([search, datasets])
    client = TestClient(app)
    assert client.get("/api/search/ping").json() == {"surface": "search"}
    assert client.get("/api/datasets/ping").json() == {"surface": "datasets"}


def test_surface_cannot_shadow_the_platform_codes_registry():
    """Surface routes cannot claim `/codes`; lookups reach the platform registry."""
    router = APIRouter()

    @router.get("/codes/{code}")
    async def shadow_registry(code: str):
        return {"shadowed": code}

    shadow = Surface(
        name="shadow",
        prefix="/codes",
        router=router,
        capability_version="v1",
        codes={"shadow_code": ("Shadow", "shadow detail")},
    )
    with pytest.raises(SurfaceConfigurationError):
        create_app([shadow])

    legit = _make_surface("legit", "/api/legit", "legit_code", "v1")
    app = create_app([legit])
    client = TestClient(app)
    resp = client.get("/codes/legit_code")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "legit_code"
    assert body["title"] == "Legit Title"


def test_platform_owned_response_carries_the_platform_version_not_a_surfaces(surface_a, surface_b):
    """Platform-owned code responses carry `PLATFORM_CAPABILITY_VERSION`."""
    app = create_app([surface_a, surface_b])
    client = TestClient(app)

    resp = client.get("/codes/nonexistent_code")
    assert resp.status_code == 404
    assert resp.json()["capability_version"] == PLATFORM_CAPABILITY_VERSION


def test_platform_owned_response_is_unaffected_by_surface_order():
    """Platform-owned response bodies are independent of surface order."""
    forward = create_app(
        [
            _make_surface("a", "/api/a", "a_code", "va"),
            _make_surface("b", "/api/b", "b_code", "vb"),
        ]
    )
    backward = create_app(
        [
            _make_surface("b", "/api/b", "b_code", "vb"),
            _make_surface("a", "/api/a", "a_code", "va"),
        ]
    )

    resp_forward = TestClient(forward).get("/codes/nonexistent_code")
    resp_backward = TestClient(backward).get("/codes/nonexistent_code")

    body_forward = resp_forward.json()
    body_backward = resp_backward.json()
    del body_forward["trace_id"], body_backward["trace_id"]
    assert body_forward == body_backward


def test_routing_404_outside_every_surface_prefix_carries_the_platform_version(
    surface_a, surface_b
):
    app = create_app([surface_a, surface_b])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["type"] == "/codes/not_found"
    assert body["detail"] == "No route exists for the requested path."
    assert isinstance(body["trace_id"], str) and len(body["trace_id"]) == 32
    int(body["trace_id"], 16)  # genuinely hex, not merely 32 characters
    assert body["capability_version"] == PLATFORM_CAPABILITY_VERSION


def test_routing_404_under_a_surfaces_prefix_carries_that_surfaces_version(surface_a, surface_b):
    app = create_app([surface_a, surface_b])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/nope")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["type"] == "/codes/not_found"
    assert body["detail"] == "No route exists for the requested path."
    assert isinstance(body["trace_id"], str) and len(body["trace_id"]) == 32
    int(body["trace_id"], 16)
    assert body["capability_version"] == "va"


def test_routing_405_carries_the_matched_surfaces_version_and_a_real_allow_header():
    """The two routes expose different `Allow` values."""
    router_a = APIRouter()

    @router_a.get("/api/a/ping")
    async def ping_a():
        return {"ok": True}

    router_b = APIRouter()

    @router_b.post("/api/b/echo")
    async def echo_b(payload: dict):
        return payload

    surface_a = Surface(
        name="a",
        prefix="/api/a",
        router=router_a,
        capability_version="va",
        codes={"a_code": ("A", "a detail")},
    )
    surface_b = Surface(
        name="b",
        prefix="/api/b",
        router=router_b,
        capability_version="vb",
        codes={"b_code": ("B", "b detail")},
    )
    app = create_app([surface_a, surface_b])
    client = TestClient(app, raise_server_exceptions=False)

    resp_a = client.delete("/api/a/ping")
    assert resp_a.status_code == 405
    assert resp_a.headers["content-type"] == "application/problem+json"
    assert resp_a.headers["cache-control"] == "no-store"
    assert resp_a.headers["allow"] == "GET"
    body_a = resp_a.json()
    assert body_a["code"] == "method_not_allowed"
    assert body_a["type"] == "/codes/method_not_allowed"
    assert (
        body_a["detail"]
        == "The requested method is not allowed for this path; the Allow header lists the methods that are."
    )
    assert body_a["capability_version"] == "va"

    resp_b = client.get("/api/b/echo")
    assert resp_b.status_code == 405
    assert resp_b.headers["allow"] == "POST"
    body_b = resp_b.json()
    assert body_b["code"] == "method_not_allowed"
    assert (
        body_b["detail"]
        == "The requested method is not allowed for this path; the Allow header lists the methods that are."
    )
    assert body_b["capability_version"] == "vb"


def test_hand_raised_http_exception_keeps_its_own_detail_and_maps_to_request_invalid():
    """A custom HTTP detail survives rendering, and status 403 maps to `request_invalid`."""
    router = APIRouter()

    @router.get("/api/a/forbidden")
    async def forbidden():
        raise HTTPException(status_code=403, detail="You may not do that here.")

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"a_code": ("A", "a detail")},
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/forbidden")
    assert resp.status_code == 403
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["detail"] == "You may not do that here."
    assert body["capability_version"] == "va"


def test_hand_raised_http_exception_with_dict_detail_does_not_500():
    """Structured HTTP details use the catalogue detail and preserve the raised status."""
    router = APIRouter()

    @router.get("/api/probe/dict-detail")
    async def dict_detail():
        raise HTTPException(status_code=403, detail={"reason": "denied"})

    surface = Surface(
        name="probe",
        prefix="/api/probe",
        router=router,
        capability_version="v-probe",
        codes={"probe_code": ("Probe", "probe detail")},
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/probe/dict-detail")
    assert resp.status_code == 403
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["detail"] == PLATFORM_CODE_CATALOGUE["request_invalid"][1]
    assert body["retryable"] is False


def test_hand_raised_http_exception_with_list_detail_keeps_the_raised_status():
    """A list detail preserves the raised status."""
    router = APIRouter()

    @router.get("/api/probe/list-detail")
    async def list_detail():
        raise HTTPException(status_code=422, detail=["a", "b"])

    surface = Surface(
        name="probe",
        prefix="/api/probe",
        router=router,
        capability_version="v-probe",
        codes={"probe_code": ("Probe", "probe detail")},
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/probe/list-detail")
    assert resp.status_code == 422


def test_hand_raised_http_exception_with_dict_detail_and_headers_keeps_headers():
    """Structured details preserve headers from the raised `HTTPException`."""
    router = APIRouter()

    @router.get("/api/probe/dict-detail-with-headers")
    async def dict_detail_with_headers():
        raise HTTPException(
            status_code=403,
            detail={"reason": "denied"},
            headers={"X-Probe": "carried-through"},
        )

    surface = Surface(
        name="probe",
        prefix="/api/probe",
        router=router,
        capability_version="v-probe",
        codes={"probe_code": ("Probe", "probe detail")},
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/probe/dict-detail-with-headers")
    assert resp.status_code == 403
    assert resp.headers["x-probe"] == "carried-through"


def test_new_platform_codes_dereference():
    """The endpoint returns the exact title and detail for each platform code."""
    surface = _make_surface("solo", "/api/solo", "solo_code", "v1")
    app = create_app([surface])
    client = TestClient(app)

    expected = {
        "not_found": {
            "title": "Not Found",
            "detail": "No route exists for the requested path.",
        },
        "method_not_allowed": {
            "title": "Method Not Allowed",
            "detail": (
                "The requested method is not allowed for this path; the Allow header "
                "lists the methods that are."
            ),
        },
    }
    for code, expected_fields in expected.items():
        resp = client.get(f"/codes/{code}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == expected_fields["title"]
        assert body["detail"] == expected_fields["detail"]


def test_surface_overriding_starlette_http_exception_wins_over_the_platform_default():
    """A surface classifier can override a platform-default exception on its own prefix."""
    router = APIRouter()

    @router.get("/api/a/ping")
    async def ping():
        return {"ok": True}

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, StarletteHTTPException):
            return ProblemSpec(status=404, code="a_custom_not_found", retryable=False)
        return None

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"a_code": ("A", "a detail"), "a_custom_not_found": ("Custom", "custom detail")},
        classify=classify,
        exception_types=(StarletteHTTPException,),
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/a/nope")
    assert resp.status_code == 404
    assert resp.json()["code"] == "a_custom_not_found"


def test_surface_catchall_classify_does_not_swallow_routing_refusals():
    """Routing refusals reach the platform handler when the surface classifier does not claim them."""
    router = APIRouter()

    @router.get("/api/a/ping")
    async def ping():
        return {"ok": True}

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, (StarletteHTTPException, RequestValidationError, asyncio.TimeoutError)):
            return None
        return ProblemSpec(status=500, code="a_catchall", retryable=False)

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"a_code": ("A", "a detail"), "a_catchall": ("Catchall", "catchall detail")},
        classify=classify,
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp_404 = client.get("/api/a/nope")
    assert resp_404.status_code == 404
    assert resp_404.headers["content-type"] == "application/problem+json"
    assert resp_404.headers["cache-control"] == "no-store"
    body_404 = resp_404.json()
    assert body_404["code"] == "not_found"
    assert body_404["capability_version"] == "va"

    resp_405 = client.post("/api/a/ping", json={})
    assert resp_405.status_code == 405
    assert resp_405.headers["allow"] == "GET"
    assert resp_405.headers["content-type"] == "application/problem+json"
    body_405 = resp_405.json()
    assert body_405["code"] == "method_not_allowed"
    assert body_405["capability_version"] == "va"


def test_surface_catchall_classify_does_not_swallow_validation_or_timeout():
    """Validation and timeout errors use platform handlers unless the surface claims them."""
    router = APIRouter()

    @router.post("/api/a/echo")
    async def echo(payload: dict):
        return payload

    @router.get("/api/a/slow")
    async def slow():
        raise TimeoutError("slow")

    def classify(exc: BaseException) -> ProblemSpec | None:
        if isinstance(exc, (StarletteHTTPException, RequestValidationError, asyncio.TimeoutError)):
            return None
        return ProblemSpec(status=500, code="a_catchall", retryable=False)

    surface = Surface(
        name="a",
        prefix="/api/a",
        router=router,
        capability_version="va",
        codes={"a_code": ("A", "a detail"), "a_catchall": ("Catchall", "catchall detail")},
        classify=classify,
    )
    app = create_app([surface])
    client = TestClient(app, raise_server_exceptions=False)

    resp_validation = client.post(
        "/api/a/echo", content=b"not json", headers={"content-type": "application/json"}
    )
    assert resp_validation.status_code == 400
    body_validation = resp_validation.json()
    assert body_validation["code"] == "request_invalid"
    assert body_validation["capability_version"] == "va"

    resp_timeout = client.get("/api/a/slow")
    assert resp_timeout.status_code == 504
    body_timeout = resp_timeout.json()
    assert body_timeout["code"] == "timeout"
    assert body_timeout["capability_version"] == "va"
