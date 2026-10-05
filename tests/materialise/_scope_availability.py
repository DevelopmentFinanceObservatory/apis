"""The default place scope's SDMX availability reads, answered from test data.

`materialise.place_scope` asks availability for the `observationAbout` values of one StatVar,
unfiltered. `ScopeAvailability` answers those reads from the `(StatVar, place)` pairs a test says
hold data, as an `AvailabilityReader` for a probe called directly and as an `/availability`
response for an `SdmxClient` over a mock transport.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Mapping, Sequence

import httpx

from dc_sdmx import EntityFilter, SdmxCapExceededError

_AVAILABILITY_PATH = "/sdmx/v3/availability/"


class ScopeAvailability:
    """`data` holds the `(StatVar, place)` pairs with observations; each read answers the places
    any named StatVar holds data at, in `data`'s order. A read past `max_values` raises
    `SdmxCapExceededError`. `reads` records each read's variables."""

    def __init__(self, *, data: Collection[tuple[str, str]]) -> None:
        self.pairs = tuple(dict.fromkeys(data))
        self._lock = threading.Lock()
        self.reads: list[tuple[str, ...]] = []

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        assert component == "observationAbout", component
        assert not filters, filters
        places = self._places(variables)
        if max_values is not None and len(places) > max_values:
            raise SdmxCapExceededError(
                "too many places", cap_kind="values", cap=max_values, observed=len(places)
            )
        return places

    def close(self) -> None:
        return None

    def respond(self, request: httpx.Request) -> httpx.Response | None:
        """The `/availability` body for the default place scope's read, or `None` for any other
        request."""
        if _AVAILABILITY_PATH not in request.url.path:
            return None
        component = request.url.path.rsplit("/", 1)[-1]
        assert component == "observationAbout", component
        values = self._places(str(request.url.params["c[variableMeasured]"]).split(","))
        body = {
            "data": {
                "dataConstraints": [
                    {"cubeRegions": [{"keyValues": [{"id": component, "values": values}]}]}
                ]
            }
        }
        return httpx.Response(200, json=body)

    def _places(self, variables: Sequence[str]) -> list[str]:
        with self._lock:
            self.reads.append(tuple(variables))
        wanted = set(variables)
        return list(dict.fromkeys(place for variable, place in self.pairs if variable in wanted))


class ScopeTransport(httpx.BaseTransport):
    """Answers the default place scope's availability read from `availability` and passes every
    other request to `inner`."""

    def __init__(self, inner: httpx.BaseTransport, availability: ScopeAvailability) -> None:
        self._inner = inner
        self._availability = availability

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._availability.respond(request)
        return response if response is not None else self._inner.handle_request(request)


class _UnreadAvailability:
    """Availability for a probe that reads no default place scope, since every request binds a
    place or a role. A read fails the test."""

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(self, *, variables: Sequence[str], **_kwargs: object) -> list[str]:
        raise AssertionError(f"read the default place scope of {list(variables)!r}")


NO_SCOPE_READ = _UnreadAvailability()
"""The availability reader of a probe that must read no default place scope."""
