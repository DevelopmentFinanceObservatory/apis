"""Pins the two-hop currency-unit lookup, memo, pagination, and malformed responses offline."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from dc_graph import (
    CURRENCY_UNIT_OF_MEASURE_ISO_CODES,
    GraphClient,
    GraphDeadlineExceededError,
    GraphNotAdvancingError,
    GraphReadError,
)
from dc_graph import client as client_module


def _typeof_body(member_dcids: list[str], currency_dcids: tuple[str, ...] = ()) -> dict:
    data = {"CurrencyUnit": {"arcs": {"typeOf": {"nodes": [{"dcid": d} for d in member_dcids]}}}}
    if currency_dcids:
        data["Currency"] = {"arcs": {"typeOf": {"nodes": [{"dcid": d} for d in currency_dcids]}}}
    return {"data": data}


def _with_pinned(graph_codes: dict[str, str]) -> dict[str, str]:
    """The `currency_units()` result for `graph_codes`: the pinned map merged under them."""
    return {**CURRENCY_UNIT_OF_MEASURE_ISO_CODES, **graph_codes}


def _iso_body(codes: dict[str, list[str]]) -> dict:
    return {
        "data": {
            dcid: {"arcs": {"iso4217AlphaCode": {"nodes": [{"value": v} for v in values]}}}
            for dcid, values in codes.items()
        }
    }


def test_currency_units_two_hop_shape_and_result(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        if payload["property"] == "<-typeOf":
            assert payload["nodes"] == ["CurrencyUnit", "Currency"]
            return _typeof_body(["CurrencyUSD", "CurrencyEUR"])
        assert payload["property"] == "->iso4217AlphaCode"
        assert payload["nodes"] == ["CurrencyUSD", "CurrencyEUR"]
        return _iso_body({"CurrencyUSD": ["USD"], "CurrencyEUR": ["EUR"]})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    result = client.currency_units()

    assert result == _with_pinned({"CurrencyUSD": "USD", "CurrencyEUR": "EUR"})
    assert len(calls) == 2


def test_currency_units_second_call_makes_no_request(monkeypatch) -> None:
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        if payload["property"] == "<-typeOf":
            return _typeof_body(["CurrencyUSD"])
        return _iso_body({"CurrencyUSD": ["USD"]})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    first = client.currency_units()
    second = client.currency_units()

    assert first == _with_pinned({"CurrencyUSD": "USD"})
    assert second == _with_pinned({"CurrencyUSD": "USD"})
    assert call_count[0] == 2  # One lookup uses two hops. The cached call uses none.


def test_currency_units_cache_hit_returns_a_copy_not_the_memo(monkeypatch) -> None:
    """Mutating a returned mapping must never corrupt the memo a later call
    returns."""

    def fake_post(self, path, payload, *, deadline=None):
        if payload["property"] == "<-typeOf":
            return _typeof_body(["CurrencyUSD"])
        return _iso_body({"CurrencyUSD": ["USD"]})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    first = client.currency_units()
    first["CurrencyUSD"] = "TAMPERED"

    assert client.currency_units() == _with_pinned({"CurrencyUSD": "USD"})


def test_currency_units_drops_a_member_with_zero_iso_codes(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        if payload["property"] == "<-typeOf":
            return _typeof_body(["CurrencyUSD", "CurrencyXYZ"])
        return _iso_body({"CurrencyUSD": ["USD"], "CurrencyXYZ": []})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    result = client.currency_units()

    assert result == _with_pinned({"CurrencyUSD": "USD"})
    assert "CurrencyXYZ" not in result


def test_currency_units_drops_a_member_with_several_iso_codes(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        if payload["property"] == "<-typeOf":
            return _typeof_body(["CurrencyUSD", "CurrencyXYZ"])
        return _iso_body({"CurrencyUSD": ["USD"], "CurrencyXYZ": ["ABC", "DEF"]})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    result = client.currency_units()

    assert result == _with_pinned({"CurrencyUSD": "USD"})
    assert "CurrencyXYZ" not in result


def test_currency_units_a_member_absent_from_the_iso_response_is_dropped(monkeypatch) -> None:
    """A member omitted by the ISO-code hop is dropped."""

    def fake_post(self, path, payload, *, deadline=None):
        if payload["property"] == "<-typeOf":
            return _typeof_body(["CurrencyUSD", "CurrencyXYZ"])
        return _iso_body({"CurrencyUSD": ["USD"]})  # CurrencyXYZ omitted

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    assert client.currency_units() == _with_pinned({"CurrencyUSD": "USD"})


def test_currency_units_failed_read_raises_and_is_not_cached(monkeypatch) -> None:
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        raise GraphReadError("boom")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    with pytest.raises(GraphReadError):
        client.currency_units()
    with pytest.raises(GraphReadError):
        client.currency_units()

    assert call_count[0] == 2  # neither failed attempt was memoised, so both retried


def test_currency_units_page_budget_exhausted_raises_from_a_not_advancing_cause(
    monkeypatch,
) -> None:
    # Each page carries a fresh token, so only the page budget stops the walk.
    monkeypatch.setattr(client_module, "_NODE_ARCS_MANY_MAX_PAGES", 2)
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        return {"data": _typeof_body(["CurrencyUSD"])["data"], "nextToken": f"tok{len(calls)}"}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    with pytest.raises(GraphReadError, match="still reported a nextToken") as raised:
        client.currency_units()
    assert isinstance(raised.value.__cause__, GraphNotAdvancingError)
    assert len(calls) == 2


def test_currency_units_empty_roster_raises_without_second_post_and_retries(monkeypatch) -> None:
    """An empty member response is a fault and raises GraphReadError. The
    failure is not cached, enabling retries on the next call."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        return {"data": {"CurrencyUnit": {"arcs": {}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    with pytest.raises(GraphReadError):
        client.currency_units()
    assert len(calls) == 1  # no member dcids -> the second hop is never posted

    with pytest.raises(GraphReadError):
        client.currency_units()
    assert len(calls) == 2  # the failed read was not cached, so this call retried


def test_currency_units_paginates_both_hops_via_next_token(monkeypatch) -> None:
    """Both paged hops merge results by `(dcid, property)`."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        if payload["property"] == "<-typeOf":
            if payload.get("nextToken") is None:
                return {
                    "data": _typeof_body(["CurrencyUSD"])["data"],
                    "nextToken": "typeof-page-2",
                }
            assert payload["nextToken"] == "typeof-page-2"
            return _typeof_body(["CurrencyEUR"])
        assert payload["property"] == "->iso4217AlphaCode"
        assert payload["nodes"] == ["CurrencyUSD", "CurrencyEUR"]
        if payload.get("nextToken") is None:
            return {
                "data": _iso_body({"CurrencyUSD": ["USD"]})["data"],
                "nextToken": "iso-page-2",
            }
        assert payload["nextToken"] == "iso-page-2"
        return _iso_body({"CurrencyEUR": ["EUR"]})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    result = client.currency_units()

    assert result == _with_pinned({"CurrencyUSD": "USD", "CurrencyEUR": "EUR"})
    assert len(calls) == 4  # 2 pages per hop


def test_currency_units_pins_the_four_currency_unit_of_measure_currencies() -> None:
    """Base Data Commons types these four `CurrencyUnitOfMeasure` with no ISO code, so the map
    is the only source of their codes. Widening it is a deliberate, reviewed change."""
    assert dict(CURRENCY_UNIT_OF_MEASURE_ISO_CODES) == {
        "USDollar": "USD",
        "Euro": "EUR",
        "CanadianDollar": "CAD",
        "IndianRupee": "INR",
    }


def test_currency_units_merges_the_pinned_map_under_the_graph_codes(monkeypatch) -> None:
    """The graph's own `iso4217AlphaCode` wins where both name a dcid, so an ontology overlay
    that codes `USDollar` replaces the pinned entry."""

    def fake_post(self, path, payload, *, deadline=None):
        if payload["property"] == "<-typeOf":
            return _typeof_body(["CurrencyUSD", "USDollar"])
        return _iso_body({"CurrencyUSD": ["USD"], "USDollar": ["XUS"]})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    result = client.currency_units()

    assert result["USDollar"] == "XUS"
    assert result["CurrencyUSD"] == "USD"
    assert result["Euro"] == "EUR"


def test_currency_units_maps_each_currency_member_to_its_own_code(monkeypatch) -> None:
    """Base Data Commons names its `Currency` members by their ISO 4217 code and gives them
    no `iso4217AlphaCode`, so `INR` maps to itself. A four-letter legacy member is no code,
    and a `CurrencyUnit` member's own code wins over the pinned map."""

    def fake_post(self, path, payload, *, deadline=None):
        if payload["property"] == "<-typeOf":
            return _typeof_body(["CurrencyUSD"], ("INR", "KES", "MYAD"))
        assert payload["nodes"] == ["CurrencyUSD"]
        return _iso_body({"CurrencyUSD": ["USD"]})

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    assert client.currency_units() == _with_pinned(
        {"CurrencyUSD": "USD", "INR": "INR", "KES": "KES"}
    )


@pytest.mark.parametrize(
    ("expires_after", "requested"),
    [
        pytest.param(1, ["<-typeOf"], id="between-typeof-pages"),
        pytest.param(2, ["<-typeOf", "<-typeOf"], id="between-hops"),
    ],
)
def test_currency_units_stops_paging_at_the_deadline(monkeypatch, expires_after, requested) -> None:
    """Once the absolute deadline passes, no further page is requested, the deadline error
    raises, and the partial roster is not cached. The fake clock replaces the `time` module
    the client reads its deadline from."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(client_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload["property"])
        if len(seen) == expires_after:
            clock.now = 101.0
        if payload["property"] == "<-typeOf" and payload.get("nextToken") is None:
            body = {"data": _typeof_body(["CurrencyUSD"])["data"], "nextToken": "page-2"}
        elif payload["property"] == "<-typeOf":
            body = _typeof_body(["CurrencyEUR"])
        else:
            body = _iso_body({"CurrencyUSD": ["USD"], "CurrencyEUR": ["EUR"]})
        return httpx.Response(200, json=body)

    client = GraphClient(
        base_url="http://fake/core/api",
        transport=httpx.MockTransport(handler),
        single_value_cache_enabled=True,
        timer=lambda: clock.now,
    )

    with pytest.raises(GraphDeadlineExceededError):
        client.currency_units(deadline=100.0)
    assert seen == requested

    seen.clear()
    assert client.currency_units() == _with_pinned({"CurrencyUSD": "USD", "CurrencyEUR": "EUR"})
    assert seen == ["<-typeOf", "<-typeOf", "->iso4217AlphaCode"]
