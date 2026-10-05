"""Tests shared executor installation, ownership checks, restoration, and shutdown."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

import dfo_platform.app as platform_app
from dfo_platform.app import Surface, SurfaceConfigurationError, create_app


def _thread_name_surface(prefix: str) -> Surface:
    """Builds a route that reports the executor thread name."""
    router = APIRouter()

    @router.get(f"{prefix}/thread-name")
    async def thread_name():
        name = await asyncio.to_thread(lambda: threading.current_thread().name)
        return {"name": name}

    return Surface(
        name=prefix.strip("/").replace("/", "-"),
        prefix=prefix,
        router=router,
        capability_version="v1",
        codes={},
    )


def _loop_capturing_surface(prefix: str, captured: dict) -> Surface:
    """Captures the event loop and its default executor during startup."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        captured["loop"] = loop
        # Captures the installed pool through the loop's `_default_executor` attribute.
        captured["executor"] = loop._default_executor
        yield

    return Surface(
        name=prefix.strip("/").replace("/", "-"),
        prefix=prefix,
        router=APIRouter(),
        capability_version="v1",
        codes={},
        lifespan=lifespan,
    )


class TestInstalledAsRealLoopDefault:
    def test_probe_route_observes_the_platform_prefixed_thread(self):
        surface = _thread_name_surface("/api/probe")
        app = create_app([surface])
        with TestClient(app) as client:
            resp = client.get("/api/probe/thread-name")
        assert resp.status_code == 200
        assert resp.json()["name"].startswith("dfo-io")


class TestCapacityIsAnOrdinaryParameter:
    def test_executor_max_workers_sizes_the_installed_pool(self, caplog):
        """Three concurrent calls saturate the pool configured with two workers."""
        release = threading.Event()
        router = APIRouter()

        @router.get("/api/probe/fanout")
        async def fanout():
            async def one() -> None:
                await asyncio.to_thread(release.wait, 5)

            tasks = [asyncio.create_task(one()) for _ in range(3)]
            await asyncio.sleep(0.1)  # let all three reach the (2-worker) pool
            release.set()
            await asyncio.gather(*tasks)
            return {"ok": True}

        surface = Surface(
            name="probe", prefix="/api/probe", router=router, capability_version="v1", codes={}
        )
        app = create_app([surface], executor_max_workers=2)

        with (
            caplog.at_level(logging.WARNING, logger="dfo_platform.executor"),
            TestClient(app) as client,
        ):
            resp = client.get("/api/probe/fanout")

        assert resp.status_code == 200
        assert any("saturated" in r.message for r in caplog.records)


class TestEnforcementFiresOnHijack:
    def test_surface_lifespan_replacing_default_executor_raises(self):
        @asynccontextmanager
        async def bad_lifespan(_app: FastAPI) -> AsyncIterator[None]:
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
            yield  # pragma: no cover -- the line above always raises first

        bad_surface = Surface(
            name="bad",
            prefix="/api/bad",
            router=APIRouter(),
            capability_version="v1",
            codes={},
            lifespan=bad_lifespan,
        )
        app = create_app([bad_surface])

        with (
            pytest.raises(SurfaceConfigurationError) as exc_info,
            TestClient(app, raise_server_exceptions=True),
        ):
            pass  # startup should raise before reaching here

        # Whichever of the two mechanisms actually caught it (the best-effort
        # shadow, or the run_in_executor probe) -- both name what happened in
        # terms of set_default_executor.
        assert "set_default_executor" in str(exc_info.value)


class TestStartupFailureRestoresOwnership:
    def test_unrelated_lifespan_failure_still_restores_executor(self):
        """A surface startup error restores the host loop and executor."""
        captured: dict = {}
        capture_surface = _loop_capturing_surface("/api/capture", captured)

        @asynccontextmanager
        async def failing_lifespan(_app: FastAPI) -> AsyncIterator[None]:
            raise RuntimeError("boom before yield")
            yield  # pragma: no cover

        failing_surface = Surface(
            name="failing",
            prefix="/api/failing",
            router=APIRouter(),
            capability_version="v1",
            codes={},
            lifespan=failing_lifespan,
        )
        # Capture the host loop before the second surface raises during startup.
        app = create_app([capture_surface, failing_surface])

        with (
            pytest.raises(RuntimeError, match="boom before yield"),
            TestClient(app, raise_server_exceptions=True),
        ):
            pass

        loop = captured["loop"]
        executor = captured["executor"]

        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))

        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)


class TestEnforcementHoldsForWholeLifetime:
    def test_mid_request_hijack_raises_through_the_catchall(self):
        router = APIRouter()

        @router.get("/api/hijack/mid-request")
        async def hijack():
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
            return {"unreachable": True}  # pragma: no cover

        surface = Surface(
            name="hijack", prefix="/api/hijack", router=router, capability_version="v1", codes={}
        )
        app = create_app([surface])

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/hijack/mid-request")

        assert resp.status_code == 500
        assert resp.json()["code"] == "internal_error"


class TestSharedPoolAcrossSurfaces:
    def test_two_composed_surfaces_observe_the_same_thread_prefix(self):
        """Both surfaces use the same platform thread pool."""
        first = _thread_name_surface("/api/first")
        second = _thread_name_surface("/api/second")
        app = create_app([first, second])

        with TestClient(app) as client:
            resp_first = client.get("/api/first/thread-name")
            resp_second = client.get("/api/second/thread-name")

        assert resp_first.json()["name"].startswith("dfo-io")
        assert resp_second.json()["name"].startswith("dfo-io")


class TestCleanShutdown:
    def test_exiting_testclient_shuts_down_the_executor(self):
        captured: dict = {}
        surface = _loop_capturing_surface("/api/capture", captured)
        app = create_app([surface])

        with TestClient(app):
            pass  # exiting this block must not raise

        executor = captured["executor"]
        # A shut-down executor rejects submissions.
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)


class _NoShadowLoop:
    """Wraps a loop and blocks instance assignment to `set_default_executor`."""

    __slots__ = ("_loop",)

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        object.__setattr__(self, "_loop", loop)

    def __getattr__(self, name):
        return getattr(self._loop, name)

    def __setattr__(self, name, value):
        if name == "set_default_executor":
            raise AttributeError(
                "this loop implementation has no instance slot for set_default_executor"
            )
        setattr(self._loop, name, value)


def _install_no_shadow_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes `dfo_platform.app` receive a loop that rejects the shadow assignment."""
    real_get_running_loop = asyncio.get_running_loop

    class _FakeAsyncioModule:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        @staticmethod
        def get_running_loop():
            return _NoShadowLoop(real_get_running_loop())

    monkeypatch.setattr(platform_app, "asyncio", _FakeAsyncioModule())


class _ExecutorAttrWriteBlockedLoop:
    """Wraps a loop and blocks writes to `_default_executor`."""

    __slots__ = ("_loop",)

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        object.__setattr__(self, "_loop", loop)

    def __getattr__(self, name):
        return getattr(self._loop, name)

    def __setattr__(self, name, value):
        if name == "_default_executor":
            raise AttributeError(
                "this loop implementation does not permit writing _default_executor directly"
            )
        setattr(self._loop, name, value)

    def __delattr__(self, name):
        delattr(self._loop, name)


def _install_executor_attr_write_blocked_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes `dfo_platform.app` receive a loop that rejects `_default_executor` writes."""
    real_get_running_loop = asyncio.get_running_loop

    class _FakeAsyncioModule:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        @staticmethod
        def get_running_loop():
            return _ExecutorAttrWriteBlockedLoop(real_get_running_loop())

    monkeypatch.setattr(platform_app, "asyncio", _FakeAsyncioModule())


class TestShadowInstallFallback:
    """The probe fallback enforces executor ownership when shadow installation raises `AttributeError`."""

    def test_boots_and_logs_warning_when_shadow_install_fails(self, monkeypatch, caplog):
        _install_no_shadow_loop(monkeypatch)
        surface = _thread_name_surface("/api/probe")
        app = create_app([surface])

        with caplog.at_level(logging.WARNING, logger="dfo_platform.app"), TestClient(app) as client:
            resp = client.get("/api/probe/thread-name")

        assert resp.status_code == 200
        assert resp.json()["name"].startswith("dfo-io")
        assert any(
            "set_default_executor" in r.message and r.levelname == "WARNING" for r in caplog.records
        )

    def test_probe_still_raises_on_hijack_when_shadow_install_fails(self, monkeypatch):
        _install_no_shadow_loop(monkeypatch)

        @asynccontextmanager
        async def bad_lifespan(_app: FastAPI) -> AsyncIterator[None]:
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
            yield  # pragma: no cover -- the line above always raises first

        bad_surface = Surface(
            name="bad",
            prefix="/api/bad",
            router=APIRouter(),
            capability_version="v1",
            codes={},
            lifespan=bad_lifespan,
        )
        app = create_app([bad_surface])

        with (
            pytest.raises(SurfaceConfigurationError) as exc_info,
            TestClient(app, raise_server_exceptions=True),
        ):
            pass  # startup should raise before reaching here

        # The probe catches the hijack when shadow installation fails.
        assert "set_default_executor" in str(exc_info.value)


class TestExecutorRestoration:
    """The app lifespan restores the host executor or the loop's default behavior."""

    def test_to_thread_works_before_and_after_and_a_prior_host_executor_is_restored(self):
        async def scenario() -> None:
            loop = asyncio.get_running_loop()
            host_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="host-io")
            loop.set_default_executor(host_executor)

            pre_name = await asyncio.to_thread(lambda: threading.current_thread().name)
            assert pre_name.startswith("host-io")

            surface = _thread_name_surface("/api/probe")
            app = create_app([surface])
            async with app.router.lifespan_context(app):
                in_name = await asyncio.to_thread(lambda: threading.current_thread().name)
                assert in_name.startswith("dfo-io")

            post_name = await asyncio.to_thread(lambda: threading.current_thread().name)
            assert post_name.startswith("host-io")
            assert loop._default_executor is host_executor

            host_executor.shutdown(wait=True)

        asyncio.run(scenario())

    def test_to_thread_works_after_lifespan_when_there_was_no_prior_executor(self):
        async def scenario() -> None:
            loop = asyncio.get_running_loop()
            assert loop._default_executor is None

            surface = _thread_name_surface("/api/probe")
            app = create_app([surface])
            async with app.router.lifespan_context(app):
                pass

            # With no prior executor, asyncio creates a new default pool after shutdown.
            name = await asyncio.to_thread(lambda: threading.current_thread().name)
            assert not name.startswith("dfo-io")

        asyncio.run(scenario())


class TestOwnershipIsIdentityNotName:
    """An initializer-stamped thread-local sentinel identifies the executor."""

    def test_replacement_executor_sharing_the_dfo_io_prefix_family_is_rejected(self, monkeypatch):
        _install_no_shadow_loop(monkeypatch)

        @asynccontextmanager
        async def bad_lifespan(_app: FastAPI) -> AsyncIterator[None]:
            asyncio.get_running_loop().set_default_executor(
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="dfo-io-other")
            )
            yield  # pragma: no cover -- the line above always raises first

        bad_surface = Surface(
            name="bad",
            prefix="/api/bad",
            router=APIRouter(),
            capability_version="v1",
            codes={},
            lifespan=bad_lifespan,
        )
        app = create_app([bad_surface])

        with (
            pytest.raises(SurfaceConfigurationError) as exc_info,
            TestClient(app, raise_server_exceptions=True),
        ):
            pass  # startup should raise before reaching here

        # The thread-local ownership sentinel rejects this matching name prefix.
        assert "dfo-io-other" in str(exc_info.value)


class TestShutdownHijackIsLoggedNotRaised:
    """A shutdown-time executor hijack is logged after surface teardown."""

    def test_hijack_during_surface_teardown_on_a_shadowless_loop_logs_error_without_raising(
        self, monkeypatch, caplog
    ):
        _install_no_shadow_loop(monkeypatch)

        @asynccontextmanager
        async def hijacking_teardown_lifespan(_app: FastAPI) -> AsyncIterator[None]:
            yield
            # Installs the hijack during surface teardown.
            asyncio.get_running_loop().set_default_executor(
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="dfo-io-hijack")
            )

        surface = Surface(
            name="teardown-hijack",
            prefix="/api/teardown-hijack",
            router=APIRouter(),
            capability_version="v1",
            codes={},
            lifespan=hijacking_teardown_lifespan,
        )
        app = create_app([surface])

        with (
            caplog.at_level(logging.ERROR, logger="dfo_platform.app"),
            TestClient(app, raise_server_exceptions=True),
        ):
            pass  # must not raise -- the shutdown checkpoint is log-only

        assert any(
            "set_default_executor" in r.message and r.levelname == "ERROR" for r in caplog.records
        )


class TestOuterFinallyGuardsAgainstAttributeError:
    """A restore failure leaves the active error intact and shuts down the pool."""

    def test_restore_write_failure_logs_warning_without_masking_the_hijack_and_still_shuts_down(
        self, monkeypatch, caplog
    ):
        _install_executor_attr_write_blocked_loop(monkeypatch)

        captured: dict = {}
        capture_surface = _loop_capturing_surface("/api/capture", captured)

        @asynccontextmanager
        async def bad_lifespan(_app: FastAPI) -> AsyncIterator[None]:
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
            yield  # pragma: no cover -- the line above always raises first

        bad_surface = Surface(
            name="bad",
            prefix="/api/bad",
            router=APIRouter(),
            capability_version="v1",
            codes={},
            lifespan=bad_lifespan,
        )
        # capture_surface first, so its lifespan has already captured loop
        # and executor by the time bad_surface's own lifespan raises.
        app = create_app([capture_surface, bad_surface])

        with (
            caplog.at_level(logging.WARNING, logger="dfo_platform.app"),
            pytest.raises(SurfaceConfigurationError) as exc_info,
            TestClient(app, raise_server_exceptions=True),
        ):
            pass  # startup should raise before reaching here

        # The hijack error remains top-level. The restore error is logged.
        assert "set_default_executor" in str(exc_info.value)

        # The pool shuts down even when executor restoration fails.
        executor = captured["executor"]
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)

        assert any(
            "could not fully restore" in r.message and r.levelname == "WARNING"
            for r in caplog.records
        )
