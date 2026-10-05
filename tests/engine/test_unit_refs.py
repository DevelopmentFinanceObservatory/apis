"""Graph-mapped CurrencyUnit dcids satisfy current-price requests by ISO 4217 code, and
conversion makes every ISO code and role shape expressible at both price bases."""

from __future__ import annotations

import logging

import pytest

from dc_graph import GraphClient
from search.engine.errors import GraphDeadlineFailure
from search.engine.unit.refs import admits, expressible, satisfies, unit_currency
from search.models import IdentityCurrencyRole, IdentityUnitRequest
from tests.fixtures import FakeGraph

_UNITS = {"CurrencyUSD": "USD", "CurrencyEUR": "EUR", "CurrencyGBP": "GBP"}


def _graph(units: dict | None = None, *, raise_on_call: bool = False) -> FakeGraph:
    if raise_on_call:
        return FakeGraph(raise_on_call=True)
    return FakeGraph(currency_units=_UNITS if units is None else units)


def _request(
    currency: str | IdentityCurrencyRole, prices: str, base_year: int | None = None
) -> IdentityUnitRequest:
    kwargs = {"currency": currency, "prices": prices}
    if base_year is not None:
        kwargs["base_year"] = base_year
    return IdentityUnitRequest(**kwargs)


@pytest.mark.parametrize(
    ("dcid", "currency"),
    [("CurrencyUSD", "USD"), ("CurrencyEUR", "EUR"), ("CurrencyGBP", "GBP")],
)
def test_satisfies_a_graph_mapped_dcid_at_current_prices(dcid, currency):
    assert satisfies(_graph(), _request(currency, "current"), dcid, deadline=None)


def test_satisfies_rejects_a_mismatched_currency():
    """A USD dcid does not satisfy a current-price EUR request."""
    assert not satisfies(_graph(), _request("EUR", "current"), "CurrencyUSD", deadline=None)


@pytest.mark.parametrize(
    "dcid",
    ["dc/unknown/unit", "", "USD", "CurrencyUSD_2023", "NotInTheGraph"],
)
def test_satisfies_is_total_over_dcids_the_graph_does_not_map(dcid):
    """An unmapped dcid satisfies no current-price request for any mapped currency."""
    graph = _graph()
    for currency in _UNITS.values():
        assert not satisfies(graph, _request(currency, "current"), dcid, deadline=None)


def test_satisfies_never_honours_a_constant_price_request():
    """Native unit refs express current prices; constant-price requests always fail."""
    assert not satisfies(_graph(), _request("USD", "constant", 2023), "CurrencyUSD", deadline=None)


def test_satisfies_fails_closed_on_a_failed_currency_units_read():
    assert not satisfies(
        _graph(raise_on_call=True), _request("USD", "current"), "CurrencyUSD", deadline=None
    )


@pytest.mark.parametrize("prices", ["current", "constant"])
@pytest.mark.parametrize("currency", ["USD", "EUR", "JPY", "XOF"])
def test_expressible_admits_every_iso_code_at_both_price_bases(currency, prices):
    base_year = 2024 if prices == "constant" else None
    assert expressible(_request(currency, prices, base_year))


@pytest.mark.parametrize("prices", ["current", "constant"])
@pytest.mark.parametrize("role", ["observationAbout", "origin", "destination"])
def test_expressible_admits_every_role_shape_at_both_price_bases(role, prices):
    base_year = 2024 if prices == "constant" else None
    assert expressible(_request(IdentityCurrencyRole(role=role), prices, base_year))


def test_expressible_refuses_a_pattern_valid_code_outside_iso_4217():
    assert not expressible(_request("XYZ", "current"))


def test_admits_none_for_no_request():
    assert admits(_graph(), None, deadline=None) is None


def test_admits_reduces_a_request_to_a_satisfies_predicate():
    graph = _graph()
    predicate = admits(graph, _request("USD", "current"), deadline=None)
    assert predicate is not None
    assert predicate("CurrencyUSD")
    assert not predicate("CurrencyEUR")
    assert not predicate("dc/unknown/unit")


def test_satisfies_native_facets_the_graph_types_without_an_iso_code_through_the_client_roster(
    monkeypatch,
):
    """Base Data Commons `USDollar` and the `Currency` member `INR` carry no ISO code in the
    graph; the roster `GraphClient.currency_units` returns still maps them, so a USDollar
    facet satisfies a current-price USD request and an INR facet a current-price INR one,
    with no conversion."""

    def fake_post(self, path, payload, *, deadline=None):
        if payload["property"] == "<-typeOf":
            return {
                "data": {
                    "CurrencyUnit": {"arcs": {"typeOf": {"nodes": [{"dcid": "CurrencyUSD"}]}}},
                    "Currency": {"arcs": {"typeOf": {"nodes": [{"dcid": "INR"}]}}},
                }
            }
        return {
            "data": {"CurrencyUSD": {"arcs": {"iso4217AlphaCode": {"nodes": [{"value": "USD"}]}}}}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    roster = GraphClient(base_url="http://fake/core/api").currency_units()
    graph = _graph(roster)

    assert satisfies(graph, _request("USD", "current"), "USDollar", deadline=None)
    assert satisfies(graph, _request("EUR", "current"), "Euro", deadline=None)
    assert not satisfies(graph, _request("EUR", "current"), "USDollar", deadline=None)
    assert satisfies(graph, _request("INR", "current"), "INR", deadline=None)
    assert not satisfies(graph, _request("USD", "current"), "INR", deadline=None)
    assert not satisfies(graph, _request("INR", "constant", 2022), "INR", deadline=None)


@pytest.mark.parametrize(
    ("dcid", "roster", "currency", "reads_graph"),
    [
        pytest.param("CurrencyKES", {}, "KES", False, id="currency-unit-convention"),
        pytest.param("USDollar", {}, "USD", False, id="currency-unit-of-measure"),
        pytest.param("INR", {"INR": "INR"}, "INR", True, id="base-dc-currency-member"),
        pytest.param("Tonne", {"INR": "INR"}, None, True, id="no-currency"),
        pytest.param("Percent", {"INR": "INR"}, None, False, id="percent"),
        pytest.param("PPPInternationalDollar", {}, None, False, id="purchasing-power"),
        pytest.param("CurrencyXYZW", {}, None, True, id="prefix-without-an-iso-code"),
    ],
)
def test_unit_currency_reads_the_graph_only_past_the_currency_conventions(
    dcid, roster, currency, reads_graph, monkeypatch
):
    graph = _graph(roster)
    reads: list[float | None] = []
    read_roster = graph.currency_units

    def counted(*, deadline: float | None = None) -> dict[str, str]:
        reads.append(deadline)
        return read_roster()

    monkeypatch.setattr(graph, "currency_units", counted)
    assert unit_currency(graph, dcid, deadline=123.0) == currency
    assert reads == ([123.0] if reads_graph else [])


def test_unit_currency_names_no_currency_and_logs_the_unit_when_the_graph_read_fails(caplog):
    with caplog.at_level(logging.WARNING, logger="search.engine.unit.refs"):
        assert unit_currency(_graph(raise_on_call=True), "INR", deadline=None) is None
    assert [record.args[0] for record in caplog.records] == ["INR"]


def test_an_expired_deadline_raises_rather_than_naming_no_currency(monkeypatch):
    """A deadline failure is a `GraphInfraError`, yet classification must not read it as "no
    currency": that would serve a EUR unit as `other` after the request budget ran out."""
    graph = _graph()

    def expired(*, deadline: float | None = None) -> dict[str, str]:
        raise GraphDeadlineFailure("graph read deadline exceeded")

    monkeypatch.setattr(graph, "currency_units", expired)
    with pytest.raises(GraphDeadlineFailure):
        unit_currency(graph, "INR", deadline=0.0)
    with pytest.raises(GraphDeadlineFailure):
        satisfies(graph, _request("USD", "current"), "CurrencyUSD", deadline=0.0)
