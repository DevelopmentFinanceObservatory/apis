"""`CloudRunGenerationSource`/`FixedGenerationSource` parsing and failure
behaviour: every failure this source can produce (HTTP, JSON, or ADC
credentials/refresh) raises `GenerationSourceError`, so `GraphGeneration`'s
refresh loop never sees an ADC exception type it would need to know about
specifically.
"""

from __future__ import annotations

import google.auth.exceptions
import httpx
import pytest

from graph_generation.source import (
    CloudRunGenerationSource,
    FixedGenerationSource,
    GenerationSourceError,
)
from tests.graph_generation.conftest import FakeCredentials

_SERVICE = (
    "projects/dev-fin-observatory/locations/northamerica-northeast1/"
    "services/dfo-dev-dc-datacommons-service"
)

# Trimmed, verified-live real Cloud Run Admin API response shape.
_REAL_RESPONSE_BODY = {
    "name": f"{_SERVICE}",
    "uri": "https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app",
    "urls": [
        "https://dfo-dev-dc-datacommons-service-41292283861.northamerica-northeast1.run.app",
        "https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app",
    ],
    "latestReadyRevision": f"{_SERVICE}/revisions/dfo-dev-dc-datacommons-service-00169-r58",
    "traffic": [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST", "percent": 100}],
    "trafficStatuses": [
        {
            "type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST",
            "revision": "dfo-dev-dc-datacommons-service-00169-r58",
            "percent": 100,
        }
    ],
}


def _source(handler, *, credentials: FakeCredentials | None = None) -> CloudRunGenerationSource:
    return CloudRunGenerationSource(
        service=_SERVICE,
        transport=httpx.MockTransport(handler),
        credentials_factory=lambda: credentials or FakeCredentials(),
    )


def test_token_parsed_from_real_shaped_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v2/{_SERVICE}"
        assert request.headers["Authorization"] == "Bearer fake-bearer-token"
        return httpx.Response(200, json=_REAL_RESPONSE_BODY)

    result = _source(handler).read()
    assert result.token == "dfo-dev-dc-datacommons-service-00169-r58"
    assert result.urls == tuple(_REAL_RESPONSE_BODY["urls"])


def test_multiple_serving_revisions_are_sorted_and_joined() -> None:
    body = dict(_REAL_RESPONSE_BODY)
    body["trafficStatuses"] = [
        {"revision": "svc-00170-z", "percent": 40},
        {"revision": "svc-00169-r58", "percent": 60},
        {"revision": "svc-not-serving", "percent": 0},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    result = _source(handler).read()
    assert result.token == "svc-00169-r58+svc-00170-z"


def test_falls_back_to_latest_ready_revision_when_traffic_statuses_absent() -> None:
    body = {k: v for k, v in _REAL_RESPONSE_BODY.items() if k != "trafficStatuses"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    result = _source(handler).read()
    assert result.token == "dfo-dev-dc-datacommons-service-00169-r58"


def test_credentials_are_built_once_and_refreshed_only_while_not_valid() -> None:
    credentials = FakeCredentials()
    build_calls = 0

    def factory() -> FakeCredentials:
        nonlocal build_calls
        build_calls += 1
        return credentials

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_REAL_RESPONSE_BODY)

    source = CloudRunGenerationSource(
        service=_SERVICE, transport=httpx.MockTransport(handler), credentials_factory=factory
    )
    source.read()
    source.read()
    assert build_calls == 1
    # `refresh()` leaves the fake `valid`, so the second read must skip it.
    assert credentials.refresh_calls == 1


def test_already_valid_credentials_are_not_refreshed() -> None:
    credentials = FakeCredentials(valid=True)
    credentials.token = "already-valid-token"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer already-valid-token"
        return httpx.Response(200, json=_REAL_RESPONSE_BODY)

    _source(handler, credentials=credentials).read()
    assert credentials.refresh_calls == 0


def test_credentials_factory_failure_raises_generation_source_error() -> None:
    def factory() -> FakeCredentials:
        raise google.auth.exceptions.DefaultCredentialsError("no ADC found")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be reached when credentials cannot be built")

    source = CloudRunGenerationSource(
        service=_SERVICE, transport=httpx.MockTransport(handler), credentials_factory=factory
    )
    with pytest.raises(GenerationSourceError):
        source.read()


def test_credentials_refresh_failure_raises_generation_source_error() -> None:
    credentials = FakeCredentials()

    def raise_refresh_error(request: object) -> None:
        raise google.auth.exceptions.RefreshError("refresh failed")

    credentials.refresh = raise_refresh_error  # type: ignore[method-assign]

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be reached when refresh fails")

    with pytest.raises(GenerationSourceError):
        _source(handler, credentials=credentials).read()


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda request: httpx.Response(503, text="unavailable"), id="non_2xx"),
        pytest.param(
            lambda request: (_ for _ in ()).throw(httpx.ConnectTimeout("timed out")),
            id="timeout",
        ),
        pytest.param(lambda request: httpx.Response(200, text="not json {"), id="malformed_json"),
        pytest.param(lambda request: httpx.Response(200, json={"uri": "x"}), id="missing_field"),
    ],
)
def test_read_failures_raise_generation_source_error_never_anything_else(handler) -> None:
    with pytest.raises(GenerationSourceError):
        _source(handler).read()


def test_no_bearer_token_raises_generation_source_error() -> None:
    credentials = FakeCredentials(refreshed_token=None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be reached when ADC refresh yields no token")

    with pytest.raises(GenerationSourceError):
        _source(handler, credentials=credentials).read()


def test_fixed_generation_source_makes_no_http_call() -> None:
    source = FixedGenerationSource(token="local-mirror-token")
    result = source.read()
    assert result.token == "local-mirror-token"
    assert result.urls == ()
