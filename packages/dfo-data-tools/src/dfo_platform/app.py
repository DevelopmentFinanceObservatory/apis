"""Compose FastAPI surfaces with shared problem responses, code registry, and trace IDs.

Surface routers carry full paths. `prefix` selects exception and admission handling.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import Executor
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from dataclasses import dataclass, field
from typing import Any

from dfo_spec_kit.registry import REGISTRY
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.routing import BaseRoute, Match
from starlette.types import ASGIApp, Scope
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from dfo_platform import codes, refusal
from dfo_platform.build import build_commit
from dfo_platform.caller import log_request
from dfo_platform.codes import (
    PLATFORM_CODE_CATALOGUE,
    CodeCatalogue,
    CodeRegistryEntry,
    merge_catalogues,
)
from dfo_platform.compression import BufferedGzipMiddleware
from dfo_platform.config import executor_max_workers as _default_executor_max_workers
from dfo_platform.executor import SaturationTrackingExecutor, current_thread_is_owned_by
from dfo_platform.problem import (
    PROBLEM_JSON,
    Envelope,
    ProblemDetails,
    ProblemSpec,
    UnregisteredCode,
)
from dfo_platform.trace import (
    TRACE_ID_HEADER,
    cloud_trace_id_from_headers,
    get_trace_id,
    new_trace_id,
    reset_cloud_trace_id,
    reset_trace_id,
    set_cloud_trace_id,
    set_trace_id,
)
from dfo_platform.validation import (
    BodyParams,
    reset_request_body_model,
    set_request_body_model,
    walkable_body_annotation,
)
from dfo_platform.version import PLATFORM_CAPABILITY_VERSION

logger = logging.getLogger(__name__)


def _classify_nothing(_exc: BaseException) -> ProblemSpec | None:
    """Default classifier. `None` delegates to the platform."""
    return None


@dataclass(frozen=True)
class Surface:
    """One router and its exception, admission, and lifespan rules for the composed app.

    `prefix` selects dispatch for the router's full paths. It must be a
    canonical `/api/<segment>` path and disjoint from other surfaces. Platform
    routes and generated docs use paths outside `/api/`.

    `classify` synchronously maps recognised exceptions to `ProblemSpec` or
    returns `None` for the platform default. Its `isinstance` order controls
    specificity. `exception_types` registers handled types with
    `ExceptionMiddleware`. Omitted types render through the catch-all, then
    Starlette re-raises them. `exception_exemplars` supplies one real
    input per declared type.

    `middleware` contains synchronous admission checks that return a
    `ProblemSpec` or `None`. Checks run in order before dependency resolution
    or body parsing. The first spec returns through this surface's envelope
    before `call_next`. Returning a spec as data avoids the
    `ServerErrorMiddleware` re-raise caused by exceptions in this layer.

    `registry_endpoints` maps a route's path, as the router declares it, to the
    code registry endpoint it serves, such as `datasets.mint`. Admission checks
    and exception classification for a request that route matches run with that
    endpoint bound (`codes.bound_registry_endpoint`), so `registered_status`
    reads the route's own status for a code.

    `request_scope` returns a context manager entered around each request the
    surface serves, after its admission checks and before dependencies run. It
    binds per-request context variables, such as the origin catalogue, which
    handlers, dependencies and `asyncio.to_thread` hops then read.

    A surface lifespan runs beside the others. The app installs one shared
    default executor and probes its ownership after startup and shutdown.
    Surfaces pass private pools to `run_in_executor`. Changing the loop's
    default executor is rejected. The guard covers the serving loop. Separate
    loops in other threads keep their own executors.
    """

    name: str
    prefix: str
    router: APIRouter
    capability_version: str
    codes: CodeCatalogue
    classify: Callable[[BaseException], ProblemSpec | None] = _classify_nothing
    exception_types: tuple[type[Exception], ...] = ()
    exception_exemplars: tuple[BaseException, ...] = ()
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None
    middleware: tuple[Callable[[Request], ProblemSpec | None], ...] = ()
    registry_endpoints: Mapping[str, str] = field(default_factory=dict)
    request_scope: Callable[[], AbstractContextManager[object]] | None = None


class SurfaceConfigurationError(Exception):
    """Raised when a surface or composed app violates a configuration invariant.

    Composition rejects empty surface lists, invalid or nested prefixes,
    duplicate names, and routes outside their prefix.

    Also raised when a surface replaces the event loop's default executor
    or the startup ownership probe times out or fails. The shutdown probe
    logs executor failures because the app is already tearing down.
    """


_API_NAMESPACE = "/api/"

# Startup fails closed if the executor probe does not answer within this bound.
_OWNERSHIP_PROBE_TIMEOUT_S = 10.0

# Shutdown probes are diagnostic and use a shorter bound so pool cleanup can proceed.
_SHUTDOWN_OWNERSHIP_PROBE_TIMEOUT_S = 2.0

# Log each route with an unusable body annotation once. Worker threads share this set.
_unusable_body_model_paths: set[str] = set()
_unusable_body_model_lock = threading.Lock()


def _within_prefix(path: str, prefix: str) -> bool:
    """Match `prefix` on path-segment boundaries."""
    stripped = prefix.rstrip("/")
    return path == stripped or path.startswith(stripped + "/")


def _validate_prefix_shape(surface: Surface) -> None:
    """Require a canonical `/api/<segment>` prefix with no trailing or empty segments.

    The platform reserves `/codes/{code}` and FastAPI's documentation paths.
    """
    prefix = surface.prefix
    if not prefix.startswith(_API_NAMESPACE):
        reason = f"does not start with {_API_NAMESPACE!r}"
    elif len(prefix) == len(_API_NAMESPACE):
        reason = f"has no segment after {_API_NAMESPACE!r}"
    elif prefix.endswith("/"):
        reason = "ends with a trailing slash"
    elif "" in prefix.split("/")[1:]:
        reason = "contains an empty path segment"
    else:
        return
    raise SurfaceConfigurationError(
        f"surface {surface.name!r} declares prefix {prefix!r}, which {reason} -- "
        "the platform reserves every path outside /api/ for its own routes "
        "(/codes/{code} and FastAPI's generated documentation), and a surface "
        "mounted there would shadow them, since surface routers are included "
        "before the platform's own."
    )


def _validate_surfaces(surfaces: Sequence[Surface]) -> None:
    """Validate surface names, prefixes, and routes before composing the app.

    Prefixes must be canonical, unique, and non-nested. Each route must lie
    under its surface prefix. Shape checks run first so malformed prefixes
    get a direct error. Disjoint prefixes keep route registration and
    exception ownership aligned and protect platform routes.
    """
    seen_names: set[str] = set()
    seen_prefixes: dict[str, str] = {}
    for surface in surfaces:
        _validate_prefix_shape(surface)
        if surface.name in seen_names:
            raise SurfaceConfigurationError(
                f"surface {surface.name!r} is declared more than once -- names identify "
                "a surface in every SurfaceConfigurationError message and parametrize "
                "the conformance suite, so duplicates make both ambiguous."
            )
        seen_names.add(surface.name)

        for other_prefix, other_name in seen_prefixes.items():
            if surface.prefix == other_prefix:
                raise SurfaceConfigurationError(
                    f"surface {surface.name!r} declares prefix {surface.prefix!r}, but "
                    f"surface {other_name!r} already claims that same prefix -- two "
                    "surfaces sharing a prefix would make exception dispatch ambiguous "
                    "between them."
                )
            if _within_prefix(surface.prefix, other_prefix) or _within_prefix(
                other_prefix, surface.prefix
            ):
                raise SurfaceConfigurationError(
                    f"surface {surface.name!r} (prefix {surface.prefix!r}) and surface "
                    f"{other_name!r} (prefix {other_prefix!r}) have nested prefixes -- "
                    "FastAPI dispatches an overlapping route by registration order while "
                    "_matching_surface resolves exception ownership by longest-prefix-"
                    "wins, so the two can disagree about which surface's handler actually "
                    "owns a request under the narrower prefix."
                )
        seen_prefixes[surface.prefix] = surface.name

        for route in surface.router.routes:
            path = getattr(route, "path", None)
            if path is None:
                continue
            if not _within_prefix(path, surface.prefix):
                raise SurfaceConfigurationError(
                    f"surface {surface.name!r} declares prefix {surface.prefix!r}, but its "
                    f"router mounts {path!r}, which does not live under that prefix -- "
                    "exception dispatch for that route would silently fall through to "
                    "another surface's handler or the platform default instead of this "
                    "surface's own."
                )
        route_paths = {getattr(route, "path", None) for route in surface.router.routes}
        unrouted = sorted(set(surface.registry_endpoints) - route_paths)
        if unrouted:
            raise SurfaceConfigurationError(
                f"surface {surface.name!r} names registry endpoints for {unrouted!r}, which "
                "no route on its router declares, so no request would bind them."
            )


def _matching_surface(path: str, surfaces: Sequence[Surface]) -> Surface | None:
    """The surface whose `prefix` is the longest match for `path` (by
    `_within_prefix`, on path-segment boundaries), or None when no surface's
    prefix matches at all.
    """
    best: Surface | None = None
    best_len = -1
    for surface in surfaces:
        if _within_prefix(path, surface.prefix) and len(surface.prefix) > best_len:
            best = surface
            best_len = len(surface.prefix)
    return best


_OPENAPI_HTTP_METHODS = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)


def _openapi_operations(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield HTTP operation objects, skipping non-method Path Item keys."""
    for path_item in schema.get("paths", {}).values():
        for method, operation in path_item.items():
            if method in _OPENAPI_HTTP_METHODS:
                yield operation


def _openapi_refs(node: object) -> Iterator[str]:
    """Every `$ref` string reachable from `node`, walking dicts and lists."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref
        for value in node.values():
            yield from _openapi_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _openapi_refs(item)


def _normalize_component_refs(node: object) -> object:
    """Return `node` with internal `$ref` prefixes reduced to component names.

    Recurse through dictionaries and lists. Preserve all other values.
    """
    if isinstance(node, dict):
        normalized = {key: _normalize_component_refs(value) for key, value in node.items()}
        ref = normalized.get("$ref")
        if isinstance(ref, str) and "/" in ref:
            normalized["$ref"] = ref.rsplit("/", 1)[-1]
        return normalized
    if isinstance(node, list):
        return [_normalize_component_refs(item) for item in node]
    return node


def _same_component_schema(existing: Any, injected: Any) -> bool:
    """Compare schema shapes after normalising internal `$ref` formats."""
    return _normalize_component_refs(existing) == _normalize_component_refs(injected)


def _strip_stock_validation_error_schema(schema: dict[str, Any]) -> None:
    """Remove FastAPI's stock 422 schemas after checking for surviving references.

    Raises `AssertionError` if another operation references either removed
    component, which would leave a dangling `$ref`.
    """
    stock_names = ("HTTPValidationError", "ValidationError")
    stock_refs = {f"#/components/schemas/{name}" for name in stock_names}

    for operation in _openapi_operations(schema):
        responses = operation.get("responses", {})
        response = responses.get("422")
        if response is not None and stock_refs & set(_openapi_refs(response)):
            del responses["422"]

    schemas = schema.get("components", {}).get("schemas", {})
    survivor_schemas = {name: body for name, body in schemas.items() if name not in stock_names}
    survivors = {"paths": schema.get("paths", {}), "components": {"schemas": survivor_schemas}}
    referenced = set(_openapi_refs(survivors))
    dangling = stock_refs & referenced
    if dangling:
        raise AssertionError(
            f"{sorted(dangling)} still referenced outside FastAPI's stock validation-error "
            "responses; refusing to delete those component schemas out from under a live $ref"
        )

    for name in stock_names:
        schemas.pop(name, None)


def _problem_details_component_schemas() -> dict[str, Any]:
    """Return the ProblemDetails and nested schemas with OpenAPI component references."""
    raw = ProblemDetails.model_json_schema(ref_template="#/components/schemas/{model}")
    nested = raw.pop("$defs", {})
    return {"ProblemDetails": raw, **nested}


def _problem_details_response(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            PROBLEM_JSON: {"schema": {"$ref": "#/components/schemas/ProblemDetails"}},
        },
    }


def _install_problem_details_openapi(app: FastAPI) -> None:
    """Document the responses this app emits in its OpenAPI schema.

    Every operation gets a default RFC 9457 `ProblemDetails` response.
    Operations with request bodies or parameters also get `400` validation
    responses. FastAPI's stock `HTTPValidationError`/422 schema is removed.
    Registered refusal codes use the default response.
    """

    def _openapi() -> dict[str, Any]:
        # Match FastAPI's private route-version cache invalidation check.
        routes_version = app.router._get_routes_version()
        if app.openapi_schema and app._openapi_routes_version == routes_version:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            summary=app.summary,
            description=app.description,
            terms_of_service=app.terms_of_service,
            contact=app.contact,
            license_info=app.license_info,
            routes=app.routes,
            webhooks=app.webhooks.routes,
            tags=app.openapi_tags,
            servers=app.servers,
            separate_input_output_schemas=app.separate_input_output_schemas,
            external_docs=app.openapi_external_docs,
        )

        _strip_stock_validation_error_schema(schema)
        injected_schemas = _problem_details_component_schemas()
        existing_schemas = schema.setdefault("components", {}).setdefault("schemas", {})
        mismatched = {
            name
            for name in set(existing_schemas) & set(injected_schemas)
            if not _same_component_schema(existing_schemas[name], injected_schemas[name])
        }
        if mismatched:
            raise AssertionError(
                f"{sorted(mismatched)} already present among the generated schema's "
                "components.schemas with a DIFFERENT shape; refusing to silently "
                "overwrite with the ProblemDetails envelope's own component schemas. "
                "A route's own model sharing a name with a ProblemDetails-nested one "
                "(e.g. both named `Cause`) is fine when the two schemas are identical -- "
                "only a genuine shape mismatch reaches this error."
            )
        existing_schemas.update(injected_schemas)
        for operation in _openapi_operations(schema):
            responses = operation.setdefault("responses", {})
            responses["default"] = _problem_details_response(
                "any non-2xx response uses this envelope"
            )
            if "requestBody" in operation or operation.get("parameters"):
                responses["400"] = _problem_details_response(
                    "the request body or parameters failed validation (request_invalid)"
                )

        app.openapi_schema = schema
        app._openapi_routes_version = routes_version
        return app.openapi_schema

    app.openapi = _openapi  # ty: ignore[invalid-assignment]


def _install_cors_outermost_middleware_stack(app: FastAPI, *, cors_kwargs: dict[str, Any]) -> None:
    """Put CORS outside the full Starlette stack, including `ServerErrorMiddleware`.

    This lets browser clients read catch-all 500 responses. `app.add_middleware`
    places CORS inside the error layer, where those responses bypass its
    send wrapper. Install before the first request, while FastAPI's
    middleware stack is lazy. CORS exceptions bypass the problem
    envelope. Its settings come from trusted app configuration. The trace-ID
    layer wraps this stack outside CORS.
    """
    base_build_middleware_stack = app.build_middleware_stack

    def _build_middleware_stack() -> ASGIApp:
        return CORSMiddleware(base_build_middleware_stack(), **cors_kwargs)

    app.build_middleware_stack = _build_middleware_stack  # ty: ignore[invalid-assignment]


def _install_trace_id_outermost_middleware_stack(app: FastAPI, *, commit: str) -> None:
    """Wrap CORS with trace-ID middleware so preflight responses are stamped.

    CORS returns preflight responses before entering its inner stack. An
    outer trace layer sees those responses and all other paths. `commit` is
    bound once from `build_commit()`. Install the override before the first
    request, while FastAPI's middleware stack remains lazy.
    """
    base_build_middleware_stack = app.build_middleware_stack
    dispatch = functools.partial(_trace_id_dispatch, commit=commit)

    def _build_middleware_stack() -> ASGIApp:
        return BaseHTTPMiddleware(base_build_middleware_stack(), dispatch=dispatch)

    app.build_middleware_stack = _build_middleware_stack  # ty: ignore[invalid-assignment]


def _is_shared_cacheable(response: Response) -> bool:
    """Return whether a `Cache-Control` token equals `public`, case-insensitively.

    Tokens are comma-delimited.
    """
    cache_control = ", ".join(response.headers.getlist("cache-control"))
    tokens = (token.strip().lower() for token in cache_control.split(","))
    return "public" in tokens


def _is_event_stream(response: Response) -> bool:
    """Return whether the response media type is `text/event-stream`, ignoring parameters."""
    content_type = response.headers.get("content-type", "")
    return content_type.split(";", 1)[0].strip().lower() == "text/event-stream"


HSTS_HEADER = "Strict-Transport-Security"
HSTS_VALUE = "max-age=31536000; includeSubDomains"
"""`includeSubDomains` is inert on this service's own `*.run.app` host,
since an API service runs only on a subdomain, so it costs nothing to
carry. `preload` requires a list submission the operator makes to the
browser vendors' own preload list. A header value alone does not register
it, and the browser ignores an unsubmitted `preload` flag, so it stays
off."""

COMMIT_HEADER = "x-dfo-commit"
"""Response header carrying the running image's own build commit (see
`dfo_platform.build.build_commit`), on every response including one marked
shared-cacheable. The value is constant for the process's whole lifetime,
so a cache replaying it to a later request still replays the correct
value, the same reasoning `HSTS_HEADER` above carries."""

SERVER_TIMING_HEADER = "Server-Timing"
"""Response header this middleware stamps with a `total;dur=<ms>` metric
(one decimal place) whenever the response carries no `Server-Timing`
header of its own, measured around `call_next`. A surface that writes its
own richer `Server-Timing` header (e.g. datasets' `PhaseTimings`, which
puts `total` first) is left untouched. `total` is present on every
response regardless of whether the surface that produced it wrote its own
timing. Skipped on a `Cache-Control: public` response (the value would be
replayed by a cache to every later request that response serves,
misreporting that request's own timing) and on a `text/event-stream`
response (`BaseHTTPMiddleware`'s own `call_next` returns at response
start, so the elapsed time would be time-to-headers on a stream, not the
whole response)."""


async def _trace_id_dispatch(
    request: Request, call_next: RequestResponseEndpoint, *, commit: str
) -> Response:
    """Bind one trace ID to the request context and stamp response headers.

    Also binds the Cloud Trace ID parsed from the request's trace headers,
    which `dfo_platform.cloud_logging` stamps on every log record, and logs one
    line per request with the caller's claimed identity (`dfo_platform.caller`).

    This outermost layer also runs on CORS preflights. Only non-cacheable
    responses receive the request's trace ID. HSTS and build commit stamp
    every response. Server timing is added around `call_next` when the
    response has no timing header and is neither shared-cacheable nor an
    event stream.

    Exception handlers run inside `call_next` before the context resets.
    `_dispatch` can rebind the ID from `request.state` as a fallback.
    """
    trace_id = new_trace_id()
    request.state.trace_id = trace_id
    token = set_trace_id(trace_id)
    cloud_token = set_cloud_trace_id(cloud_trace_id_from_headers(request.headers))
    start = time.perf_counter()
    status: int | None = None
    try:
        response = await call_next(request)
        status = response.status_code
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers[HSTS_HEADER] = HSTS_VALUE
        response.headers[COMMIT_HEADER] = commit
        if not _is_shared_cacheable(response):
            response.headers[TRACE_ID_HEADER] = trace_id
        if (
            SERVER_TIMING_HEADER not in response.headers
            and not _is_shared_cacheable(response)
            and not _is_event_stream(response)
        ):
            response.headers[SERVER_TIMING_HEADER] = f"total;dur={elapsed_ms:.1f}"
        return response
    finally:
        try:
            log_request(
                method=request.method,
                path=request.url.path,
                status=status,
                headers=request.headers,
            )
        except Exception as exc:
            # A logging failure must not replace the response. The header value stays
            # out of the record, as it can carry a credential.
            logger.warning("request log line failed: %s", type(exc).__name__)
        finally:
            # The inner task retains its copied trace context after this outer task resets.
            reset_trace_id(token)
            reset_cloud_trace_id(cloud_token)


async def _probe_default_executor(
    loop: asyncio.AbstractEventLoop, *, platform_executor: Executor, timeout: float
) -> tuple[str, bool]:
    """Probe the loop's default executor and report whether `platform_executor` started
    the worker that answered.

    `timeout` bounds the probe. It raises `asyncio.TimeoutError` on expiry
    and `RuntimeError` if submission fails.
    """
    return await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: (
                threading.current_thread().name,
                current_thread_is_owned_by(platform_executor),
            ),
        ),
        timeout=timeout,
    )


async def _assert_default_executor_is_ours(
    loop: asyncio.AbstractEventLoop, *, platform_executor: Executor
) -> None:
    """Fail startup if the executor probe times out, fails, or finds a foreign worker.

    Each worker records the executor that started it, and the check compares that
    owner with `platform_executor` by identity. It covers the serving event loop.
    Loops created in other threads have separate pools.
    """
    try:
        observed, owned = await _probe_default_executor(
            loop, platform_executor=platform_executor, timeout=_OWNERSHIP_PROBE_TIMEOUT_S
        )
    except TimeoutError as exc:
        raise SurfaceConfigurationError(
            f"executor ownership could not be verified within "
            f"{_OWNERSHIP_PROBE_TIMEOUT_S:.0f}s -- failing closed rather "
            "than assuming the platform's own pool is still the loop's "
            "default executor. Some mounted surface's own lifespan or a "
            "request handler may have replaced it via "
            "set_default_executor with a wedged or overloaded pool."
        ) from exc
    except RuntimeError as exc:
        raise SurfaceConfigurationError(
            "the executor ownership probe could not submit its check "
            f"call ({exc}) -- either a replacement executor installed by "
            "some mounted surface's own lifespan or a request handler "
            "(via set_default_executor) is already shut down or "
            "otherwise broken, or the platform's own pool itself failed "
            "to start its workers."
        ) from exc
    if not owned:
        raise SurfaceConfigurationError(
            f"the event loop's default executor no longer routes to the "
            f"platform's own pool (observed thread {observed!r}) -- some "
            "mounted surface's own lifespan or a request handler must "
            "have replaced it via set_default_executor. At most one "
            "owner may set the loop's default executor, and the "
            "platform is that owner; a surface needing its own pool "
            "must pass it explicitly to loop.run_in_executor(pool, ...) "
            "instead."
        )


async def _log_default_executor_ownership_at_shutdown(
    loop: asyncio.AbstractEventLoop, *, platform_executor: Executor
) -> None:
    """Log shutdown probe failures. Teardown continues so the platform executor can close."""
    try:
        observed, owned = await _probe_default_executor(
            loop,
            platform_executor=platform_executor,
            timeout=_SHUTDOWN_OWNERSHIP_PROBE_TIMEOUT_S,
        )
    except TimeoutError:
        logger.warning(
            "executor ownership probe did not complete within %.0fs "
            "during shutdown; continuing so the platform's own "
            "executor.shutdown() still runs (this is a timeout, not "
            "a confirmed hijack)",
            _SHUTDOWN_OWNERSHIP_PROBE_TIMEOUT_S,
        )
        return
    except RuntimeError as exc:
        logger.error(
            "the executor ownership probe could not submit its check "
            "call at shutdown (%s) -- either a replacement executor "
            "installed by some mounted surface's own lifespan or a "
            "request handler (via set_default_executor) is already shut "
            "down or otherwise broken, or the platform's own pool "
            "itself failed to start its workers",
            exc,
        )
        return
    if not owned:
        logger.error(
            "the event loop's default executor no longer routed to the "
            "platform's own pool by shutdown (observed thread %r) -- "
            "some mounted surface's own lifespan or a request handler "
            "replaced it via set_default_executor",
            observed,
        )


def _reject_default_executor_override(_new_executor: object) -> None:
    # The shadow rejects replacements early. Startup and shutdown probes verify ownership.
    raise SurfaceConfigurationError(
        "a mounted surface's own lifespan (or request handler) called "
        "asyncio.get_running_loop().set_default_executor(...), but the "
        "event loop's default executor is platform-owned for the whole "
        "composed app's lifetime (see dfo_platform.app.create_app's own "
        "lifespan) -- at most one owner may set it, and the platform is "
        "that owner. A surface needing its own pool must pass it "
        "explicitly to loop.run_in_executor(pool, ...) instead."
    )


@asynccontextmanager
async def _lifespan(
    _app: FastAPI, *, surfaces: Sequence[Surface], executor_max_workers: int | None
) -> AsyncIterator[None]:
    """Install the shared executor, enter surface lifespans, and verify executor ownership.

    Lifespans enter in surface order and unwind in reverse. Resolve
    `executor_max_workers` at startup so configuration loaded by a surface
    import is available. After teardown, restore the prior executor and shut
    down the platform pool with `wait=False, cancel_futures=True` because
    running `to_thread` calls are uninterruptible.
    """
    loop = asyncio.get_running_loop()
    # Save the existing executor so the host loop can be restored after the app lifespan.
    prior_executor = getattr(loop, "_default_executor", None)
    resolved_max_workers = (
        executor_max_workers
        if executor_max_workers is not None
        else _default_executor_max_workers()
    )
    executor = SaturationTrackingExecutor(
        max_workers=resolved_max_workers, thread_name_prefix="dfo-io"
    )
    shadow_installed = False
    try:
        loop.set_default_executor(executor)
        try:
            loop.set_default_executor = _reject_default_executor_override  # ty: ignore[invalid-assignment]
            shadow_installed = True
        except AttributeError:
            logger.warning(
                "cannot install the set_default_executor shadow on this "
                "event loop implementation; immediate rejection of "
                "executor replacement is unavailable, so enforcement "
                "rests entirely on the startup and shutdown ownership "
                "probes"
            )
        try:
            async with AsyncExitStack() as stack:
                for surface in surfaces:
                    if surface.lifespan is not None:
                        await stack.enter_async_context(surface.lifespan(_app))
                await _assert_default_executor_is_ours(loop, platform_executor=executor)
                yield
        finally:
            # Probe after surface teardown so the check cannot delay resource cleanup.
            await _log_default_executor_ownership_at_shutdown(loop, platform_executor=executor)
    finally:
        try:
            if shadow_installed:
                del loop.set_default_executor
            # Assign directly. CPython rejects set_default_executor(None).
            # A prior None is valid, and run_in_executor creates a pool lazily.
            loop._default_executor = prior_executor  # ty: ignore[unresolved-attribute]
        except AttributeError as exc:
            # Preserve the active exception and always reach executor shutdown.
            logger.warning(
                "could not fully restore the event loop's prior state "
                "after the composed app's own lifespan (%s); the "
                "platform's own executor is still shut down below "
                "regardless",
                exc,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class _AppContext:
    """Request-time state resolved once by `create_app`.

    It holds the surfaces, the platform envelope, and each surface's merged
    envelope keyed by name.
    """

    surfaces: Sequence[Surface]
    registry: CodeCatalogue
    envelope: Envelope
    surface_envelopes: dict[str, Envelope]


def _envelope_for(request: Request, ctx: _AppContext) -> Envelope:
    """Return the matched surface's envelope, or the platform envelope when no prefix matches."""
    surface = _matching_surface(request.url.path, ctx.surfaces)
    if surface is None:
        return ctx.envelope
    return ctx.surface_envelopes[surface.name]


def _registry_endpoint(surface: Surface, scope: Scope) -> str | None:
    """The code registry endpoint of the first surface route that fully matches `scope`, as
    the router would pick it, or `None` when that route names none."""
    for route in surface.router.routes:
        if isinstance(route, BaseRoute) and route.matches(scope)[0] == Match.FULL:
            return surface.registry_endpoints.get(getattr(route, "path", ""))
    return None


async def _admission_dispatch(
    request: Request, call_next: RequestResponseEndpoint, *, ctx: _AppContext
) -> Response:
    """Run the matched surface's admission checks before dependencies or body parsing.

    Checks run in surface order. The first `ProblemSpec` returns through
    that surface's envelope. `None` passes to the next check or `call_next`,
    which runs inside the surface's `request_scope`. `call_next` starts the
    downstream app as a task, which copies the context the scope bound.
    """
    surface = _matching_surface(request.url.path, ctx.surfaces)
    if surface is not None:
        endpoint = _registry_endpoint(surface, request.scope) if surface.middleware else None
        for check in surface.middleware:
            with codes.bound_registry_endpoint(endpoint):
                spec = check(request)
            if spec is not None:
                response = _envelope_for(request, ctx).render(spec)
                trace_id = getattr(request.state, "trace_id", None)
                response.headers[TRACE_ID_HEADER] = (
                    trace_id if trace_id is not None else get_trace_id()
                )
                return response
        if surface.request_scope is not None:
            with surface.request_scope():
                return await call_next(request)
    return await call_next(request)


def _classify_and_envelope(
    request: Request, exc: Exception, ctx: _AppContext
) -> tuple[ProblemSpec, Envelope, str | None]:
    """Classify an exception and select its envelope and owning surface.

    A surface classifier that raises is logged and falls through to the
    platform default. Recognised exception types absent from
    `exception_types` render, then `ServerErrorMiddleware` re-raises
    the original exception.
    """
    surface = _matching_surface(request.url.path, ctx.surfaces)
    envelope = _envelope_for(request, ctx)
    if surface is not None:
        try:
            with codes.bound_registry_endpoint(_registry_endpoint(surface, request.scope)):
                spec = surface.classify(exc)
        except Exception:
            logger.exception(
                "surface %r's classify() raised while classifying %s; "
                "falling back to the platform default",
                surface.name,
                type(exc).__name__,
            )
            spec = None
        if spec is not None:
            if set(type(exc).__mro__).isdisjoint(surface.exception_types):
                logger.warning(
                    "surface %r's classify() recognized %s, which is absent "
                    "from its own declared exception_types; that type's "
                    "exception-handler registration falls to the bare-"
                    "Exception handler ServerErrorMiddleware owns, so its "
                    "response is followed by ServerErrorMiddleware's own "
                    "unconditional re-raise of the original exception",
                    surface.name,
                    type(exc).__name__,
                )
            return spec, envelope, surface.name
    return refusal.classify_default(exc), envelope, None


def _request_body_model(request: Request) -> object | None:
    """Return the matched route's body annotation for validation-error pointers.

    Return `None` when the route has no body. Starlette sets
    `request.scope["route"]` before dependencies run. `BodyParams` retains
    `FieldInfo` so discriminated-union tags reach the pointer walk. Routes
    without a walkable annotation are logged once and use the model-free
    fallback, which can point one segment too deep when a union tag matches a
    sibling key. This helper leaves validation responses at 400.
    """
    route = request.scope.get("route")
    dependant = getattr(route, "dependant", None)
    body_params = getattr(dependant, "body_params", None)
    if not body_params:
        return None
    # Pydantic stores an `Annotated` union's discriminator on `FieldInfo`.
    # Its annotation remains a bare union.
    by_name = {body_param.alias: body_param.field_info for body_param in body_params}
    embedded = len(body_params) > 1 or bool(getattr(route, "_embed_body_fields", False))
    fallback = body_params[0].field_info if not embedded else None
    if not any(walkable_body_annotation(a) for a in (*by_name.values(), fallback)):
        path = getattr(route, "path", None) or request.url.path
        with _unusable_body_model_lock:
            first = path not in _unusable_body_model_paths
            _unusable_body_model_paths.add(path)
        if first:
            logger.warning(
                "route %s declares request body parameters but exposes no "
                "annotation a JSON Pointer walk can read (%s); the pointer "
                "for a refused body falls back to the model-free body walk",
                path,
                ", ".join(f"{name}={annotation!r}" for name, annotation in by_name.items()),
            )
        return None
    return BodyParams(by_name=by_name, fallback=fallback, embedded=embedded)


async def _dispatch(request: Request, exc: Exception, *, ctx: _AppContext) -> JSONResponse:
    """Render a surface classification or the platform's default refusal."""
    # Rebind the request trace ID so exception handlers share its context.
    trace_id = getattr(request.state, "trace_id", None)
    token = set_trace_id(trace_id) if trace_id is not None else None
    # Bind the body model for validation_problem's JSON Pointer construction.
    model_token = set_request_body_model(_request_body_model(request))
    try:
        spec, surface_envelope, surface_name = _classify_and_envelope(request, exc, ctx)
        # Supply an ID if this handler ran outside `_trace_id_dispatch`.
        header_trace_id = trace_id if trace_id is not None else get_trace_id()
        try:
            response = surface_envelope.render(spec)
            response.headers[TRACE_ID_HEADER] = header_trace_id
            return response
        except UnregisteredCode as unregistered:
            # Render the platform default for an unregistered surface code.
            # This preserves the original error.
            logger.error(
                "surface %r's classify() returned code %r, which is not "
                "registered in its own catalogue; rendering the platform "
                "default for %s instead",
                surface_name,
                unregistered.code,
                type(exc).__name__,
                exc_info=exc,
            )
            response = ctx.envelope.render(refusal.render_unhandled_refusal())
            response.headers[TRACE_ID_HEADER] = header_trace_id
            return response
    finally:
        reset_request_body_model(model_token)
        if token is not None:
            reset_trace_id(token)


def _code_registry_route(ctx: _AppContext) -> Callable[[str], Any]:
    """Build the `/codes/{code}` handler from the merged platform registry.

    Registry entries return code metadata without an envelope. Unknown codes
    render through the envelope and carry `PLATFORM_CAPABILITY_VERSION`.
    """

    async def code_registry_entry(code: str) -> CodeRegistryEntry | JSONResponse:
        catalogue_entry = ctx.registry.get(code)
        registry_entry = REGISTRY[code] if code in REGISTRY else None  # noqa: SIM401 -- CodeRegistry has no .get()
        if catalogue_entry is None and registry_entry is None:
            return refusal.render_code_not_found_refusal(ctx.envelope, code=code)
        return codes.registry_entry_body(
            code=code, registry_entry=registry_entry, catalogue_entry=catalogue_entry
        )

    return code_registry_entry


def create_app(
    surfaces: Sequence[Surface],
    *,
    cors_origins: Sequence[str] = (),
    cors_methods: Sequence[str] = ("GET", "POST", "OPTIONS"),
    cors_headers: Sequence[str] = ("Content-Type",),
    cors_expose_headers: Sequence[str] = (TRACE_ID_HEADER, COMMIT_HEADER, "Server-Timing"),
    executor_max_workers: int | None = None,
    title: str = "DFO APIs",
    version: str = "1.0",
) -> FastAPI:
    """Build the shared FastAPI app from mounted surfaces.

    Resolve `executor_max_workers` at call time so configuration loaded by
    surface imports is available. Platform-level refusals use the platform
    capability version. Matched surfaces use their own catalogue and version.

    The dispatcher tries the matched surface's `classify`, then the platform
    default. Defaults map validation errors to 400, timeouts to retryable
    504, routing errors to 404/405, and other exceptions to logged 500
    responses whose details stay off the wire.

    Admission checks run before FastAPI parses request bodies. They return a
    `ProblemSpec` directly, separate from exception classification.
    """
    if not surfaces:
        raise SurfaceConfigurationError(
            "create_app requires at least one surface to mount; an app with no "
            "surfaces would serve nothing beyond the platform's own "
            "/codes/{code}, and tests/platform/test_surface_conformance.py is "
            "parametrized off the surface list -- an empty list would make "
            "every conformance check there pass vacuously rather than "
            "actually exercising anything."
        )
    _validate_surfaces(surfaces)
    registry = merge_catalogues(PLATFORM_CODE_CATALOGUE, *(s.codes for s in surfaces))
    envelope = Envelope(catalogue=registry, capability_version=PLATFORM_CAPABILITY_VERSION)
    ctx = _AppContext(
        surfaces=surfaces,
        registry=registry,
        envelope=envelope,
        surface_envelopes={
            surface.name: Envelope(
                catalogue=merge_catalogues(PLATFORM_CODE_CATALOGUE, surface.codes),
                capability_version=surface.capability_version,
            )
            for surface in surfaces
        },
    )

    app = FastAPI(
        title=title,
        version=version,
        lifespan=functools.partial(
            _lifespan, surfaces=surfaces, executor_max_workers=executor_max_workers
        ),
    )
    _install_problem_details_openapi(app)

    # Empty origins disable CORS. `expose_headers` controls which response
    # headers browser scripts can read. `allow_headers` controls request headers.
    # Install CORS outside ServerErrorMiddleware to cover catch-all responses.
    _install_cors_outermost_middleware_stack(
        app,
        cors_kwargs={
            "allow_origins": list(cors_origins),
            "allow_methods": list(cors_methods),
            "allow_headers": list(cors_headers),
            "expose_headers": list(cors_expose_headers),
        },
    )

    # Put trace IDs outside CORS so preflight responses are stamped. The image's
    # build commit is fixed, so resolve it once for the app's lifetime.
    _install_trace_id_outermost_middleware_stack(app, commit=build_commit())

    # Add gzip first so it sits innermost, wrapping only the router and its exception
    # handlers. The admission and trace-ID layers re-emit every response as a stream and
    # gzip leaves streams unencoded, so gzip has to see responses before they do. CORS
    # and the trace-ID layer then stamp their headers on the encoded response.
    app.add_middleware(BufferedGzipMiddleware)

    # Cloud Run is the sole ingress and terminates TLS, so its
    # X-Forwarded-Proto is trusted. The leftmost X-Forwarded-For entry is
    # Callers control the leftmost X-Forwarded-For entry. Treat `request.client`
    # as untrusted for authentication and rate limits.
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

    app.add_middleware(BaseHTTPMiddleware, dispatch=functools.partial(_admission_dispatch, ctx=ctx))

    exception_types = {
        RequestValidationError,
        asyncio.TimeoutError,
        StarletteHTTPException,
        Exception,
    } | {exc_type for surface in surfaces for exc_type in surface.exception_types}

    dispatch = functools.partial(_dispatch, ctx=ctx)
    for exc_type in exception_types:
        app.add_exception_handler(exc_type, dispatch)

    for surface in surfaces:
        app.include_router(surface.router)

    platform_router = APIRouter()
    platform_router.add_api_route(
        "/codes/{code}",
        _code_registry_route(ctx),
        methods=["GET"],
        response_model=CodeRegistryEntry,
    )
    app.include_router(platform_router)

    return app
