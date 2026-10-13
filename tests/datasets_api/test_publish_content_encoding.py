"""Tests the publish door's `Content-Encoding` handling through HTTP.

`_admit_publish_content_encoding` refuses unaccepted encodings in middleware.
The publish route class checks a gzip request's principal, takes the publish
permit, and decodes gzip bodies before FastAPI reads them, refusing corrupt,
truncated, trailing, or oversized gzip data.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
import types
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import publish
from datasets_api.doors.publish import (
    _INFLATE_STEP_BYTES,
    MAX_PUBLISH_BODY_BYTES,
    PublishBodyEncodingInvalid,
    PublishBodyTooLarge,
    PublishResult,
    build_publish_door,
    classify_publish_exception,
)
from datasets_api.version_store import LiveVersionRecordWriter
from tests.artifact_store.conftest import FakeGCSClient

from ._deps_doubles import fake_deps_cell
from ._snapshot_doubles import make_client_for_door
from ._version_record_store import FakeVersionRecordDatasetStore

_PUBLISH_ROUTE = "/api/datasets/frames/publish"
_PRINCIPAL = "publisher-principal"
_AUTH_HEADERS = {"x-dfo-principal": _PRINCIPAL}
_JSON = {"content-type": "application/json"}
_GZIP = {**_JSON, "content-encoding": "gzip"}


def _client(*, pool_size: int | None = None) -> TestClient:
    """Wires shared SQLite stores so a publish succeeds and a replay compares stored bytes."""
    store = FakeVersionRecordDatasetStore()
    artifact_store = LiveArtifactStore(
        buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
        inventory=LiveArtifactInventory(store=store),
        client=FakeGCSClient(),
    )
    settings = DatasetsBuildSettings.from_env()
    if pool_size is not None:
        settings = settings.model_copy(update={"work_max_concurrency": pool_size})
    door = build_publish_door(
        deps=fake_deps_cell(
            settings=settings,
            artifact_store=artifact_store,
            snapshot_reader=LiveSnapshotReader(store=store),
            version_writer=LiveVersionRecordWriter(store=store),
        ),
        settings=settings,
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    return make_client_for_door(door)


def _body_bytes() -> bytes:
    rows = [
        {
            "stat_var": "dcid:Exemplar",
            "time_period": str(2000 + offset),
            "value": float(offset),
            "recipient": "country/KEN",
        }
        for offset in range(50)
    ]
    return json.dumps(
        {
            "schema_version": 1,
            "spec_id": "ops_abc123",
            "content_version": "ocv_abc123",
            "role_keys": ["recipient"],
            "rows": rows,
            "input_content_versions": ["cv_00000000000000000000000000abc123"],
        }
    ).encode()


def _encoded(body: bytes, encoding: str) -> bytes:
    return gzip.compress(body) if encoding == "gzip" else body


async def _asgi_post(
    app: FastAPI,
    receive: Callable[[], Awaitable[Message]],
    *,
    content_length: int,
    encoding: str,
    principal: str | None = _PRINCIPAL,
) -> tuple[int, dict[str, object]]:
    """POST through the app's ASGI stack with a caller-driven `receive`, returning the status and JSON."""
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(content_length).encode()),
        (b"content-encoding", encoding.encode()),
    ]
    if principal is not None:
        headers.append((b"x-dfo-principal", principal.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": _PUBLISH_ROUTE,
        "raw_path": _PUBLISH_ROUTE.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
        "state": {},
    }
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    [status] = [message["status"] for message in sent if message["type"] == "http.response.start"]
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return status, json.loads(body)


@pytest.fixture
def inflations(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Records every decompressor the door creates, so a test can assert it never inflated."""
    created: list[object] = []

    def recording_decompressobj(*args: object, **kwargs: object) -> object:
        decompressor = zlib.decompressobj(*args, **kwargs)
        created.append(decompressor)
        return decompressor

    monkeypatch.setattr(
        publish,
        "zlib",
        types.SimpleNamespace(decompressobj=recording_decompressobj, error=zlib.error),
    )
    return created


def _zero_bomb(decoded_bytes: int) -> bytes:
    """Gzip `decoded_bytes` zero bytes without holding them, about 1,000 to 1 compressed."""
    compressor = zlib.compressobj(level=9, wbits=31)
    step = b"\0" * _INFLATE_STEP_BYTES
    parts = [compressor.compress(step) for _ in range(decoded_bytes // len(step))]
    parts.append(compressor.flush())
    return b"".join(parts)


@pytest.mark.parametrize("encoding", ["gzip", "GZIP", " gzip "])
def test_gzip_body_publishes_the_same_frame_as_its_identity_form(encoding: str) -> None:
    """A gzip replay of an identity publish is idempotent, so the door stored identical bytes."""
    client = _client()
    body = _body_bytes()

    identity = client.post(_PUBLISH_ROUTE, content=body, headers={**_JSON, **_AUTH_HEADERS})
    assert identity.status_code == 200, identity.text
    assert identity.json()["published"]["created"] is True

    gzipped = client.post(
        _PUBLISH_ROUTE,
        content=gzip.compress(body, compresslevel=1),
        headers={**_JSON, "content-encoding": encoding, **_AUTH_HEADERS},
    )
    assert gzipped.status_code == 200, gzipped.text
    assert gzipped.json()["published"]["created"] is False
    assert (
        gzipped.json()["published"]["artifact_digest"]
        == identity.json()["published"]["artifact_digest"]
    )


def test_gzip_body_publishes_on_a_fresh_pair() -> None:
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE, content=gzip.compress(_body_bytes()), headers={**_GZIP, **_AUTH_HEADERS}
    )

    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_explicit_identity_encoding_publishes() -> None:
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=_body_bytes(),
        headers={**_JSON, "content-encoding": "identity", **_AUTH_HEADERS},
    )

    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


@pytest.mark.parametrize("encoding", ["br", "deflate", "x-gzip", "gzip, identity", "gzip, gzip"])
def test_unaccepted_encoding_refuses_before_reading_the_body(encoding: str) -> None:
    """The refusal comes from middleware: no principal is sent and the body is not valid gzip."""
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=b"irrelevant -- never read",
        headers={**_JSON, "content-encoding": encoding},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False
    assert f"Content-Encoding {encoding.lower()!r} is not accepted" in body["detail"]


def test_repeated_content_encoding_headers_refuse() -> None:
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=gzip.compress(_body_bytes()),
        headers=[
            ("content-type", "application/json"),
            ("content-encoding", "gzip"),
            ("content-encoding", "gzip"),
            ("x-dfo-principal", _PRINCIPAL),
        ],
    )

    assert response.status_code == 400
    assert "is not accepted" in response.json()["detail"]


@pytest.mark.parametrize(
    ("content", "detail"),
    [
        (b"{not gzip at all}", "not valid gzip data"),
        (gzip.compress(b'{"spec_id": "ops_abc123"}')[:-12], "truncated"),
        (gzip.compress(b'{"spec_id": "ops_abc123"}') + b"trailing", "after the end of its gzip"),
        (gzip.compress(b"{}") + gzip.compress(b"{}"), "after the end of its gzip"),
    ],
    ids=["corrupt", "truncated", "trailing-bytes", "second-member"],
)
def test_invalid_gzip_refuses_request_invalid(content: bytes, detail: str) -> None:
    client = _client()

    response = client.post(_PUBLISH_ROUTE, content=content, headers={**_GZIP, **_AUTH_HEADERS})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False
    assert detail in body["detail"]


class _CountingDecompressor:
    """Wraps a real decompressor and counts the decoded bytes it hands the door."""

    def __init__(self, inner: zlib._Decompress) -> None:
        self._inner = inner
        self.decoded = 0
        self.largest_piece = 0

    def _count(self, piece: bytes) -> bytes:
        self.decoded += len(piece)
        self.largest_piece = max(self.largest_piece, len(piece))
        return piece

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        return self._count(self._inner.decompress(data, max_length))

    def flush(self) -> bytes:
        return self._count(self._inner.flush())

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def test_gzip_bomb_refuses_without_decoding_past_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body that inflates 256-fold past the bound stops within one step of it."""
    limit = 2 * _INFLATE_STEP_BYTES
    decoded_size = 256 * limit
    bomb = _zero_bomb(decoded_size)
    assert len(bomb) < limit  # passes the Content-Length check, so only inflation can refuse

    decompressors: list[_CountingDecompressor] = []

    def counting_decompressobj(*args: object, **kwargs: object) -> _CountingDecompressor:
        decompressor = _CountingDecompressor(zlib.decompressobj(*args, **kwargs))
        decompressors.append(decompressor)
        return decompressor

    monkeypatch.setattr(publish, "MAX_PUBLISH_BODY_BYTES", limit)
    monkeypatch.setattr(
        publish,
        "zlib",
        types.SimpleNamespace(decompressobj=counting_decompressobj, error=zlib.error),
    )
    client = _client()

    response = client.post(_PUBLISH_ROUTE, content=bomb, headers={**_GZIP, **_AUTH_HEADERS})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    spec = classify_publish_exception(
        PublishBodyTooLarge(content_length=len(bomb), limit=limit, inflated=True)
    )
    assert spec is not None
    assert body["detail"] == spec.detail

    [decompressor] = decompressors
    assert limit < decompressor.decoded <= limit + _INFLATE_STEP_BYTES
    assert decompressor.largest_piece <= _INFLATE_STEP_BYTES
    assert decompressor.decoded * 64 < decoded_size


def test_gzip_body_at_exactly_the_bound_is_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound is inclusive: a body decoding to exactly the limit reaches JSON decoding."""
    limit = 2 * _INFLATE_STEP_BYTES
    monkeypatch.setattr(publish, "MAX_PUBLISH_BODY_BYTES", limit)
    client = _client()
    body = _body_bytes()
    padded = body + b" " * (limit - len(body))

    response = client.post(
        _PUBLISH_ROUTE, content=gzip.compress(padded), headers={**_GZIP, **_AUTH_HEADERS}
    )

    assert response.status_code == 200, response.text

    over = client.post(
        _PUBLISH_ROUTE, content=gzip.compress(padded + b" "), headers={**_GZIP, **_AUTH_HEADERS}
    )
    assert over.status_code == 400
    assert "inflates past" in over.json()["detail"]


@pytest.mark.parametrize(
    "principal_headers",
    [{}, {"x-dfo-principal": "someone-else"}],
    ids=["missing", "unlisted"],
)
def test_unauthorized_gzip_body_refuses_before_inflating(
    principal_headers: dict[str, str], inflations: list[object]
) -> None:
    """The door decodes nothing for a caller it will refuse, even when the gzip body is valid."""
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=gzip.compress(_body_bytes()),
        headers={**_GZIP, **principal_headers},
    )

    assert response.status_code == 403
    assert response.json()["code"] == "publish_not_permitted"
    assert inflations == []


def test_gzip_body_with_content_length_over_the_bound_refuses_in_middleware(
    inflations: list[object],
) -> None:
    client = _client()

    response = client.post(
        _PUBLISH_ROUTE,
        content=gzip.compress(_body_bytes()),
        headers={
            **_GZIP,
            **_AUTH_HEADERS,
            "content-length": str(MAX_PUBLISH_BODY_BYTES + 1),
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "request_invalid"
    spec = classify_publish_exception(
        PublishBodyTooLarge(content_length=MAX_PUBLISH_BODY_BYTES + 1, limit=MAX_PUBLISH_BODY_BYTES)
    )
    assert spec is not None
    assert body["detail"] == spec.detail
    assert inflations == []


@pytest.mark.parametrize("encoding", ["identity", "gzip"])
def test_client_disconnect_mid_body_refuses_request_invalid_without_an_error_log(
    encoding: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A gzip read answers a disconnect with the 400 FastAPI gives an identity read."""
    app = _client().app
    body = _encoded(_body_bytes(), encoding)
    messages: list[Message] = [
        {"type": "http.request", "body": body[:20], "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive() -> Message:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    with caplog.at_level(logging.ERROR):
        status, problem = asyncio.run(
            _asgi_post(app, receive, content_length=len(body), encoding=encoding)
        )

    assert status == 400
    assert problem["code"] == "request_invalid"
    assert problem["detail"] == "There was an error parsing the body"
    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []


def _second_publish_while_first_holds_the_permit(
    *,
    encoding: str,
    first_holds_permit_in: str,
    second_principal: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, dict[str, object], int]:
    """Sends a second publish while the first holds the door's only permit, mid-body or mid-build.

    Returns the second publish's status, problem, and how often it read its body.
    """
    app = _client(pool_size=1).app
    body = _encoded(_body_bytes(), encoding)
    build_started = threading.Event()
    finish_build = threading.Event()

    def stalled_publish_frame(**kwargs: object) -> PublishResult:
        build_started.set()
        finish_build.wait(timeout=10)
        return PublishResult(
            spec_id="ops_abc123",
            content_version="ocv_abc123",
            artifact_digest="sha256:" + "0" * 64,
            created=True,
        )

    if first_holds_permit_in == "build":
        monkeypatch.setattr(publish, "publish_frame", stalled_publish_frame)

    async def scenario() -> tuple[int, int, dict[str, object], int]:
        first_reading = asyncio.Event()
        finish_body = asyncio.Event()
        first_messages: list[Message] = [{"type": "http.request", "body": body, "more_body": False}]

        async def first_receive() -> Message:
            first_reading.set()
            if first_holds_permit_in == "body":
                await finish_body.wait()
            if first_messages:
                return first_messages.pop()
            await asyncio.Event().wait()  # no further client message
            raise AssertionError("unreachable")

        second_reads = 0

        async def second_receive() -> Message:
            nonlocal second_reads
            second_reads += 1
            return {"type": "http.request", "body": body, "more_body": False}

        first = asyncio.create_task(
            _asgi_post(app, first_receive, content_length=len(body), encoding=encoding)
        )
        await first_reading.wait()
        if first_holds_permit_in == "build":
            assert await asyncio.to_thread(build_started.wait, 10)
        second_status, second_problem = await _asgi_post(
            app,
            second_receive,
            content_length=len(body),
            encoding=encoding,
            principal=second_principal,
        )
        finish_body.set()
        finish_build.set()
        first_status, _ = await first
        return first_status, second_status, second_problem, second_reads

    first_status, second_status, second_problem, second_reads = asyncio.run(scenario())
    assert first_status == 200
    return second_status, second_problem, second_reads


@pytest.mark.parametrize("encoding", ["identity", "gzip"])
@pytest.mark.parametrize("first_holds_permit_in", ["body", "build"])
def test_publish_without_a_free_permit_refuses_before_reading_its_body(
    encoding: str, first_holds_permit_in: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, problem, reads = _second_publish_while_first_holds_the_permit(
        encoding=encoding,
        first_holds_permit_in=first_holds_permit_in,
        second_principal=_PRINCIPAL,
        monkeypatch=monkeypatch,
    )

    assert status == 503
    assert problem["code"] == "service_unavailable"
    assert problem["retryable"] is True
    assert reads == 0


@pytest.mark.parametrize("encoding", ["identity", "gzip"])
@pytest.mark.parametrize("principal", [None, "someone-else"], ids=["missing", "unlisted"])
def test_unauthorized_publish_refuses_without_taking_a_permit_or_reading_its_body(
    encoding: str, principal: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the only permit held, an unauthorized caller gets 403 rather than the 503 a permit would give."""
    status, problem, reads = _second_publish_while_first_holds_the_permit(
        encoding=encoding,
        first_holds_permit_in="body",
        second_principal=principal,
        monkeypatch=monkeypatch,
    )

    assert status == 403
    assert problem["code"] == "publish_not_permitted"
    assert reads == 0


class _StepLimitedDecompressor:
    """Wraps a real decompressor and fails the test once the door calls it too often.

    A decode loop that stalls spins without awaiting, so no timeout can interrupt it.
    The call limit turns that spin into a failure.
    """

    def __init__(self, inner: zlib._Decompress, *, max_calls: int) -> None:
        self._inner = inner
        self._calls_left = max_calls

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        self._calls_left -= 1
        if self._calls_left < 0:
            pytest.fail("the gzip decode loop did not terminate")
        return self._inner.decompress(data, max_length)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class _ChunkedRequest:
    """The slice of `Request` that `_inflate_gzip_body` reads."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.headers = {"content-length": str(sum(map(len, chunks)))}

    async def stream(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.parametrize("trailing_in_own_chunk", [False, True], ids=["same-chunk", "later-chunk"])
def test_trailing_bytes_after_a_member_larger_than_one_step_refuse(
    trailing_in_own_chunk: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member that outgrows one inflate step, then trailing bytes, refuses instead of spinning."""
    monkeypatch.setattr(
        publish,
        "zlib",
        types.SimpleNamespace(
            decompressobj=lambda *a, **k: _StepLimitedDecompressor(
                zlib.decompressobj(*a, **k), max_calls=10
            ),
            error=zlib.error,
        ),
    )
    member = gzip.compress(b"x" * (_INFLATE_STEP_BYTES + 1))
    chunks = [member, b"trailing"] if trailing_in_own_chunk else [member + b"trailing"]
    request = _ChunkedRequest(chunks)

    with pytest.raises(PublishBodyEncodingInvalid, match="after the end of its gzip"):
        asyncio.run(
            asyncio.wait_for(
                publish._inflate_gzip_body(request, limit=4 * _INFLATE_STEP_BYTES),  # type: ignore[arg-type]
                timeout=10,
            )
        )
