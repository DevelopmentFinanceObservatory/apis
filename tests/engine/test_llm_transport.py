"""Pins Gemini's httpx transport, retry behaviour, timeouts, and connection limits on loopback."""

from __future__ import annotations

import asyncio
import contextlib
import json
import select
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from google import genai
from google.genai import _api_client
from google.genai.types import HttpOptions
from pydantic import BaseModel

import search.engine.clients.llm as llm_module
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.llm import LLM, LLMMemo, detached, generate_structured_async
from search.engine.errors import LLMInfraError, LLMOutputRefused

# Captured at import, before the autouse `_guard_outbound_http` fixture swaps
# the class attribute for its refusal.
_REAL_ASYNC_HANDLE = httpx.AsyncHTTPTransport.handle_async_request


class _Value(BaseModel):
    value: int


_GENERATE_BODY = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": '{"value": 1}'}]}}],
        "modelVersion": "loopback-model",
    }
).encode()


@pytest.fixture
def loopback_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real httpx async transport, for loopback requests only."""

    async def _loopback_only(self: httpx.AsyncHTTPTransport, request: httpx.Request):
        assert request.url.host == "127.0.0.1", request.url
        return await _REAL_ASYNC_HANDLE(self, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _loopback_only)


@pytest.fixture
def live_client_env(monkeypatch: pytest.MonkeyPatch, search_settings) -> None:
    """A fresh module singleton, built with a key, with no cached client."""
    search_settings(gemini_api_key="test-key", llm_timeout_s=5.0)
    monkeypatch.setattr(llm_module, "_CLIENT", None)
    monkeypatch.setattr(llm_module, "_DETACHED_CLIENT", None)


def _point_at(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{port}")


@pytest.fixture
def closing_listener() -> Iterator[tuple[int, list[int]]]:
    """A listener that accepts each connection and closes it at once.

    Yields the port and a list holding one entry per accepted connection.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.settimeout(0.1)
    accepted: list[int] = []
    stop = threading.Event()

    def _serve() -> None:
        while not stop.is_set():
            try:
                conn, _addr = sock.accept()
            except TimeoutError:
                continue
            accepted.append(1)
            conn.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        yield sock.getsockname()[1], accepted
    finally:
        stop.set()
        thread.join()
        sock.close()


def _generate_body(text: str) -> bytes:
    return json.dumps(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
            "modelVersion": "loopback-model",
        }
    ).encode()


@pytest.fixture
def answer_listener(request: pytest.FixtureRequest) -> Iterator[int]:
    """Answers every POST with a model reply whose text is the test's parameter."""
    body = _generate_body(request.param)

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


_EMPTY_VALUE_BINDING = json.dumps(
    {
        "chosen_shape_index": 0,
        "bindings": [
            {"axis": "how", "property_dcid": "gender", "kind": "value", "value_dcids": []}
        ],
    }
)


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
@pytest.mark.parametrize(
    ("answer_listener", "refused"),
    [
        (_EMPTY_VALUE_BINDING, True),
        (json.dumps({"chosen_shape_index": 0, "ask": "x" * 600}), False),
        ('{"chosen_shape_index": "one"}', False),
        ("not json", False),
    ],
    ids=["empty-value-binding", "ask-over-its-length", "wrong-type", "unreadable"],
    indirect=["answer_listener"],
)
def test_only_an_answer_the_schema_refuses_on_purpose_is_refused(
    monkeypatch: pytest.MonkeyPatch, answer_listener: int, refused: bool
):
    """Gemini leaves the parsed response empty for each. A binding that names no value
    breaks the rule the bind schema refuses on purpose (`SCHEMA_REFUSAL`) and raises
    `LLMOutputRefused`, which a caller may answer for, with the call's usage. Any other
    validation error, and an unreadable answer, stay an infrastructure failure, which the
    response cache never stores."""
    _point_at(monkeypatch, answer_listener)
    llm = LLM(model="gemini-3.5-flash-lite")
    with pytest.raises(LLMInfraError, match="failed schema validation") as caught:
        asyncio.run(llm.generate_structured(prompt="p", system="s", schema=_UniversalBindOutput))
    assert isinstance(caught.value, LLMOutputRefused) is refused


@pytest.fixture
def keepalive_listener() -> Iterator[tuple[int, set[tuple[str, int]]]]:
    """Answers POST requests and records one client address per keep-alive connection."""
    peers: set[tuple[str, int]] = set()

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            peers.add(self.client_address)
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(_GENERATE_BODY)))
            self.end_headers()
            self.wfile.write(_GENERATE_BODY)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], peers
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.usefixtures("live_client_env")
def test_built_clients_take_the_httpx_path():
    """Both builders select httpx, where a default client in the same
    environment selects aiohttp whenever google-genai can import it."""
    default = genai.Client(api_key="test-key", http_options=HttpOptions(timeout=5000))
    assert default._api_client._use_aiohttp() is bool(_api_client.has_aiohttp)

    assert llm_module._get_client()._api_client._use_aiohttp() is False
    assert LLM(api_key="override")._client()._api_client._use_aiohttp() is False


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_a_dropped_connection_fails_once_and_fast(
    monkeypatch: pytest.MonkeyPatch, closing_listener: tuple[int, list[int]]
):
    port, accepted = closing_listener
    _point_at(monkeypatch, port)
    llm = LLM(model="gemini-3.5-flash-lite")
    llm_module._get_client()  # built before the clock starts

    started = time.monotonic()
    with pytest.raises(LLMInfraError, match="Gemini API error"):
        asyncio.run(llm.generate_structured(prompt="p", system="s", schema=_Value))
    elapsed = time.monotonic() - started

    assert accepted == [1]
    assert elapsed < 1.0


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_one_llm_serves_calls_from_successive_event_loops(
    monkeypatch: pytest.MonkeyPatch, keepalive_listener: tuple[int, set[tuple[str, int]]]
):
    """Each `asyncio.run` closes its loop, so the next call must use a separate pool."""
    port, peers = keepalive_listener
    _point_at(monkeypatch, port)
    llm = LLM(model="gemini-3.5-flash-lite")

    for _ in range(3):
        parsed, _usage = asyncio.run(llm.generate_structured(prompt="p", system="s", schema=_Value))
        assert parsed == _Value(value=1)

    assert len(peers) == 3


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


class _HoldingListener:
    """Hold requests with whitespace chunks every 50 ms; record arrivals and disconnects."""

    def __init__(self) -> None:
        self.arrivals: list[float] = []
        self.disconnects: list[float] = []
        self.max_open = 0
        self._open = 0
        self._lock = threading.Lock()
        self._release = threading.Semaphore(0)
        self._stop = threading.Event()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def release(self, count: int = 1) -> None:
        self._release.release(count)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        listener = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", 0)))
                with listener._lock:
                    listener.arrivals.append(time.monotonic())
                    listener._open += 1
                    listener.max_open = max(listener.max_open, listener._open)
                try:
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("transfer-encoding", "chunked")
                    self.end_headers()
                    self.wfile.flush()
                    if listener._hold(self.connection, self.wfile):
                        self.wfile.write(_chunk(_GENERATE_BODY) + b"0\r\n\r\n")
                    else:
                        self.close_connection = True
                finally:
                    with listener._lock:
                        listener._open -= 1

            def log_message(self, format: str, *args: object) -> None:
                return None

        return _Handler

    def _hold(self, conn: socket.socket, wfile) -> bool:
        """True once released; False when the client disconnects first or
        the listener stops."""
        while not self._stop.is_set():
            if self._release.acquire(timeout=0.05):
                return True
            readable, _, _ = select.select([conn], [], [], 0)
            try:
                if readable and conn.recv(1, socket.MSG_PEEK) == b"":
                    raise ConnectionResetError
                wfile.write(_chunk(b" "))
                wfile.flush()
            except OSError:
                with self._lock:
                    self.disconnects.append(time.monotonic())
                return False
        return False

    def __enter__(self) -> _HoldingListener:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


@pytest.fixture
def holding_listener(monkeypatch: pytest.MonkeyPatch) -> Iterator[_HoldingListener]:
    with _HoldingListener() as listener:
        _point_at(monkeypatch, listener.port)
        yield listener


async def _until(predicate, *, timeout: float = 2.0) -> float:
    """Poll `predicate` until true; return the seconds it took."""
    started = time.monotonic()
    while not predicate():
        if time.monotonic() - started > timeout:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0.01)
    return time.monotonic() - started


def _memo_llm() -> LLM:
    return LLM(
        model="gemini-3.5-flash-lite",
        memo=LLMMemo(max_bytes=1024 * 1024, max_entry_bytes=64 * 1024, ttl_s=60.0),
    )


def _memo_call(llm: LLM):
    return generate_structured_async(llm, prompt="p", system="s", schema=_Value)


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_cancelling_an_orphaned_leader_closes_its_connection(holding_listener: _HoldingListener):
    """The memo leader's caller is cancelled with no follower parked, so the
    live request is cancelled and the listener sees its connection close."""
    llm = _memo_llm()

    async def scenario() -> float:
        leader = asyncio.create_task(_memo_call(llm))
        await _until(lambda: len(holding_listener.arrivals) == 1)
        cancelled_at = time.monotonic()
        await _cancel(leader)
        await _until(lambda: holding_listener.disconnects, timeout=1.0)
        return holding_listener.disconnects[0] - cancelled_at

    assert asyncio.run(scenario()) < 1.0


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_a_parked_follower_keeps_the_leaders_connection_open(
    holding_listener: _HoldingListener,
):
    """With a follower parked on the key, cancelling the leader's caller
    leaves the request open. The follower leaving then closes it."""
    llm = _memo_llm()
    memo = llm._memo
    assert memo is not None

    async def scenario() -> None:
        leader = asyncio.create_task(_memo_call(llm))
        await _until(lambda: len(holding_listener.arrivals) == 1)
        follower = asyncio.create_task(_memo_call(llm))
        await _until(lambda: memo.stats().misses >= 2)

        await _cancel(leader)
        await asyncio.sleep(0.5)
        assert holding_listener.disconnects == []

        await _cancel(follower)
        await _until(lambda: holding_listener.disconnects, timeout=1.0)

    asyncio.run(scenario())
    assert len(holding_listener.arrivals) == 1


def _call_task(llm: LLM, prompt: str) -> asyncio.Task:
    return asyncio.create_task(llm.generate_structured(prompt=prompt, system="s", schema=_Value))


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_concurrent_gemini_requests_stay_within_the_cap(
    search_settings, holding_listener: _HoldingListener
):
    """With a pool cap of 2, the third call waits for one of the first two."""
    search_settings(llm_max_concurrency=2)
    llm = LLM(model="gemini-3.5-flash-lite")

    async def scenario() -> list:
        tasks = [_call_task(llm, f"p{i}") for i in range(3)]
        await _until(lambda: len(holding_listener.arrivals) == 2)
        await asyncio.sleep(0.3)
        assert len(holding_listener.arrivals) == 2

        holding_listener.release()
        await _until(lambda: len(holding_listener.arrivals) == 3)
        holding_listener.release(2)
        return await asyncio.gather(*tasks)

    results = asyncio.run(scenario())

    assert [parsed for parsed, _usage in results] == [_Value(value=1)] * 3
    assert holding_listener.max_open == 2


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_a_wait_for_a_pool_slot_is_bounded_by_the_llm_timeout(
    search_settings, holding_listener: _HoldingListener
):
    """A call queued behind a full pool waits at most SEARCH_LLM_TIMEOUT_S
    and fails as a Gemini timeout does, with LLMInfraError."""
    search_settings(llm_max_concurrency=1, llm_timeout_s=0.5)
    llm = LLM(model="gemini-3.5-flash-lite")

    async def scenario() -> tuple[LLMInfraError, float]:
        holder = _call_task(llm, "held")
        await _until(lambda: len(holding_listener.arrivals) == 1)
        started = time.monotonic()
        with pytest.raises(LLMInfraError, match="Gemini API error") as caught:
            await llm.generate_structured(prompt="queued", system="s", schema=_Value)
        elapsed = time.monotonic() - started
        holding_listener.release()
        await holder
        return caught.value, elapsed

    error, elapsed = asyncio.run(scenario())

    assert isinstance(error.__cause__, httpx.PoolTimeout)
    assert 0.4 < elapsed < 2.0
    assert len(holding_listener.arrivals) == 1


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_a_cancelled_pool_waiter_leaves_the_pool_usable(
    search_settings, holding_listener: _HoldingListener
):
    """A call cancelled while it waits for a pool slot gives up its place:
    the next call takes the freed connection at once."""
    search_settings(llm_max_concurrency=1, llm_timeout_s=5.0)
    llm = LLM(model="gemini-3.5-flash-lite")

    async def scenario() -> float:
        holder = _call_task(llm, "held")
        await _until(lambda: len(holding_listener.arrivals) == 1)
        waiter = _call_task(llm, "waiting")
        await asyncio.sleep(0.2)
        await _cancel(waiter)

        holding_listener.release()
        await holder
        started = time.monotonic()
        nxt = _call_task(llm, "next")
        await _until(lambda: len(holding_listener.arrivals) == 2, timeout=1.0)
        holding_listener.release()
        await nxt
        return time.monotonic() - started

    assert asyncio.run(scenario()) < 1.0
    assert len(holding_listener.arrivals) == 2


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_a_held_detached_call_leaves_the_served_connection_free(
    search_settings, holding_listener: _HoldingListener
):
    """With one served connection and one detached, a detached call held at the listener
    takes no served capacity: a served call started after it reaches the listener at
    once, well inside the pool wait that would fail it."""
    search_settings(llm_max_concurrency=1, jev_shadow_max_inflight=1, llm_timeout_s=1.0)
    llm = _memo_llm()

    async def scenario() -> list:
        control = asyncio.create_task(_memo_call(detached(llm)))
        await _until(lambda: len(holding_listener.arrivals) == 1)
        served = asyncio.create_task(_memo_call(llm))
        await _until(lambda: len(holding_listener.arrivals) == 2, timeout=0.5)
        holding_listener.release(2)
        return await asyncio.gather(served, control)

    results = asyncio.run(scenario())

    assert [parsed for parsed, _usage in results] == [_Value(value=1)] * 2
    assert holding_listener.max_open == 2


@pytest.mark.usefixtures("live_client_env")
def test_an_api_key_llm_builds_each_client_on_its_first_call(monkeypatch: pytest.MonkeyPatch):
    """An `LLM(api_key=...)` builds no client until a call needs one, and builds each once:
    its stage and detached clones share the two it builds."""
    built: list[int] = []

    def build(api_key: str, *, max_connections: int) -> object:
        built.append(max_connections)
        return object()

    monkeypatch.setattr(llm_module, "_build_client", build)
    settings = llm_module.config.settings()
    llm = LLM(model="gemini-3.5-flash-lite", api_key="override")
    assert built == []

    served = llm._client()
    assert llm_module.for_stage(llm, "bind")._client() is served  # ty: ignore[unresolved-attribute]  # for_stage clones the LLM
    assert built == [settings.llm_max_concurrency]

    shadow = detached(llm)._client()  # ty: ignore[unresolved-attribute]  # detached clones the LLM
    assert detached(llm_module.for_stage(llm, "bind"))._client() is shadow  # ty: ignore[unresolved-attribute]
    assert built == [settings.llm_max_concurrency, settings.jev_shadow_max_inflight]


@pytest.mark.usefixtures("live_client_env", "loopback_transport")
def test_an_api_key_override_llm_reuses_one_client(
    monkeypatch: pytest.MonkeyPatch, keepalive_listener: tuple[int, set[tuple[str, int]]]
):
    """An `LLM(api_key=...)` builds its client once, so two calls on it, and
    on its stage clone, share one pool: both reach the listener over one
    keep-alive connection."""
    port, peers = keepalive_listener
    _point_at(monkeypatch, port)
    llm = LLM(model="gemini-3.5-flash-lite", api_key="override")
    clone = llm_module.for_stage(llm, "bind")
    assert clone is not llm
    assert llm._client() is llm._client()
    assert clone._client() is llm._client()  # ty: ignore[unresolved-attribute]  # for_stage clones the LLM

    async def scenario() -> None:
        for caller in (llm, clone):
            parsed, _usage = await caller.generate_structured(prompt="p", system="s", schema=_Value)
            assert parsed == _Value(value=1)

    asyncio.run(scenario())

    assert len(peers) == 1
