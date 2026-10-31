"""Checks the byline after the heading: latest-year label, plain unit with its price basis,
source and bound measurement method, and the heading's period."""

from __future__ import annotations

import pytest

from search.models import (
    METHOD_REF,
    UNIT_REF,
    DateRange,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityCurrencyRole,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ResolvedFacet,
    StatVar,
)
from search.render.heading import LATEST_YEAR_LABEL, heading_from_parts
from search.render.plan import render_sentence_plan
from search.render.tokens import plan_tokens, sentence_from_plan
from tests.render._helpers import _ref, _spec

_CURRENCY_UNITS = {
    "CurrencyUSD": "USD",
    "USDollar": "USD",
    "CurrencyKES": "KES",
    "CurrencyEUR": "EUR",
    "CurrencyGTQ": "GTQ",
}
_CRS = _ref("OECD_CRS", "OECD Creditor Reporting System (CRS)")
_NOMINAL = _ref("Nominal", "Nominal")


def _unit(ref: str) -> dict[str, ResolvedFacet]:
    return {
        "unit": ResolvedFacet(
            basis="fallback", ref=ref, label=ref, currency=_CURRENCY_UNITS.get(ref)
        )
    }


def _served(start: str, end: str) -> list[StatVar]:
    return [
        StatVar(
            ref=_ref("sv1", "SV1"),
            shape_id="test-shape",
            slot_values=[],
            constraint_properties=[],
            data_date_range=DateRange(start=start, end=end),
        )
    ]


def _render(**kwargs) -> tuple[str, list]:
    spec = _spec(measured_property=_ref("disbursements", "Disbursements"), **kwargs)
    plan = render_sentence_plan(spec)
    return sentence_from_plan(plan), plan


def test_the_byline_follows_the_heading_period_in_its_own_segment():
    """The heading ends with its period; the byline joins the unit and the source with
    " · ". The unit is a slot token on the unit axis that `UNIT_REF` names."""
    sentence, plan = _render(
        measurement_qualifier=_NOMINAL,
        facets=_unit("CurrencyUSD"),
        stat_vars=_served("1973", "2024"),
        resolved_sources=[_CRS],
    )
    assert sentence == (
        "Disbursements, 1973 to 2024. Current US dollars · OECD Creditor Reporting System (CRS)."
    )
    heading_end = next(i for i, t in enumerate(plan) if t.text == ".")
    assert {t.segment for t in plan[: heading_end + 1]} == {"heading"}
    assert [(t.kind, t.text, t.segment) for t in plan[heading_end + 1 :]] == [
        ("text", " ", "byline"),
        ("slot", "Current US dollars", "byline"),
        ("text", " · ", "byline"),
        ("slot", "OECD Creditor Reporting System (CRS)", "byline"),
        ("text", ".", "byline"),
    ]
    unit = next(t for t in plan if t.axis == "unit")
    assert unit.binding_ref == UNIT_REF


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({"measurement_qualifier": _NOMINAL}, "Current US dollars", id="nominal"),
        pytest.param(
            {"measurement_qualifier": _ref("RealValue", "Real value")},
            "Constant US dollars",
            id="real-value-no-base-year",
        ),
        pytest.param({}, "US dollars", id="no-price-basis"),
        pytest.param(
            {
                "identity_facets": {
                    "unit": IdentityFacetValueRequest(
                        request=IdentityUnitRequest(
                            currency="USD", prices="constant", base_year=2022
                        )
                    )
                }
            },
            "Constant 2022 US dollars",
            id="unit-request",
        ),
    ],
)
def test_the_unit_token_states_the_price_basis(kwargs, expected: str):
    _, plan = _render(facets=_unit("USDollar"), **kwargs)
    assert next(t for t in plan if t.axis == "unit").text == expected


def _converted(currency, prices: str, base_year: int | None) -> dict:
    return {
        "identity_facets": {
            "unit": IdentityFacetValueRequest(
                request=IdentityUnitRequest(currency=currency, prices=prices, base_year=base_year)
            )
        }
    }


@pytest.mark.parametrize(
    ("unit", "kwargs", "expected", "states"),
    [
        pytest.param(
            "CurrencyEUR",
            _converted("EUR", "constant", 2024),
            "Constant 2024 euros",
            [UNIT_REF, "axis:measurementQualifier"],
            id="converted-euros",
        ),
        pytest.param(
            "CurrencyGTQ",
            _converted("GTQ", "current", None),
            "Current Guatemalan quetzales",
            [UNIT_REF, "axis:measurementQualifier"],
            id="converted-quetzales",
        ),
        pytest.param(
            "CurrencyUSD",
            {"measurement_qualifier": _NOMINAL},
            "Current US dollars",
            [UNIT_REF, "axis:measurementQualifier"],
            id="native-us-dollars",
        ),
        pytest.param(
            "CurrencyEUR",
            {"measurement_qualifier": _NOMINAL},
            "Current euros",
            [UNIT_REF, "axis:measurementQualifier"],
            id="native-euros",
        ),
        pytest.param("CurrencyEUR", {}, "Euros", [UNIT_REF], id="native-euros-no-price-basis"),
    ],
)
def test_every_currency_is_the_byline_unit_token(unit: str, kwargs, expected: str, states):
    """Any currency reads in the byline by its plural noun with its price basis, on a
    token keyed by `UNIT_REF` that states the unit and the price-basis axis. The heading
    names no currency."""
    _, plan = _render(facets=_unit(unit), **kwargs)
    (token,) = [t for t in plan if t.axis == "unit"]
    assert (token.text, token.segment, token.binding_ref) == (expected, "byline", UNIT_REF)
    assert token.states == states


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        pytest.param("origin", "Constant 2024 provider currency", id="origin"),
        pytest.param("observationAbout", "Constant 2024 national currency", id="subject"),
    ],
)
def test_a_role_currency_request_reads_as_the_roles_currency(role: str, expected: str):
    """A conversion into each entity's own currency names no single unit facet, so the
    byline names the role's currency."""
    _, plan = _render(**_converted(IdentityCurrencyRole(role=role), "constant", 2024))
    (token,) = [t for t in plan if t.axis == "unit"]
    assert (token.text, token.binding_ref) == (expected, UNIT_REF)


def test_a_chained_dollar_unit_reads_constant_prices_of_its_base_year():
    _, plan = _render(facets=_unit("ChainedUSDollarBaseYr2017"))
    assert next(t for t in plan if t.axis == "unit").text == "Constant 2017 US dollars"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {"facets": _unit("PPPInternationalDollar")},
            "Disbursements in PPP international dollars. OECD Creditor Reporting System (CRS).",
            id="purchasing-power",
        ),
        pytest.param(
            {"facets": _unit("Percent")},
            "Disbursements in percent. OECD Creditor Reporting System (CRS).",
            id="percent",
        ),
        pytest.param(
            {"facets": _unit("CurrencyKES")},
            "Disbursements. Kenyan shillings · OECD Creditor Reporting System (CRS).",
            id="national-currency",
        ),
        pytest.param(
            {"facets": {"unit": ResolvedFacet(basis="fallback", selection="per_series")}},
            "Disbursements. OECD Creditor Reporting System (CRS).",
            id="per-series",
        ),
        pytest.param(
            {"facets": _unit("MetricTonCO2e")},
            "Disbursements. OECD Creditor Reporting System (CRS).",
            id="other-unit",
        ),
        pytest.param(
            {
                "facets": _unit("CurrencyUSD"),
                "measurement_qualifier": _NOMINAL,
                "measurement_denominator": _ref("Count_Person", "Total population"),
            },
            "Disbursements per capita. Current US dollars · OECD Creditor Reporting System (CRS).",
            id="per-head-us-dollars",
        ),
    ],
)
def test_the_unit_reads_once(kwargs, expected: str):
    """A unit reads once. A unit that changes the magnitude reads in the heading, and a
    currency reads in the byline, beside a per-head rate too, where the heading's unit
    token says "per capita" and the byline's the currency."""
    sentence, plan = _render(resolved_sources=[_CRS], **kwargs)
    assert sentence == expected
    for segment in ("heading", "byline"):
        assert sum(1 for t in plan if t.axis == "unit" and t.segment == segment) <= 1


@pytest.mark.parametrize(
    ("measure_label", "expected"),
    [
        pytest.param(None, [], id="graph-name"),
        pytest.param("disbursements in US dollars", ["Current US dollars"], id="authored-noun"),
    ],
)
def test_only_a_graph_name_is_read_for_its_currency(measure_label, expected):
    """A graph name that names US dollars carries the unit, so the byline leaves it out;
    an authored noun or measure label is never read as text, so the byline states it."""
    spec = _spec(
        measured_property=_ref("disbursements", "Disbursements in US dollars"),
        measure_label=measure_label,
        measurement_qualifier=_NOMINAL,
        facets=_unit("USDollar"),
    )
    plan = render_sentence_plan(spec)
    assert [t.text for t in plan if t.axis == "unit" and t.segment == "byline"] == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        pytest.param("CurrencyUSD", ["US dollars"], id="registry-money"),
        pytest.param("CurrencyKES", ["Kenyan shillings"], id="registry-national-currency"),
        pytest.param("ExampleGraphCurrency", [], id="unregistered"),
    ],
)
def test_a_unit_facet_with_no_currency_reads_the_registry_currency(ref, expected):
    """The unit registry types a money unit's currency, so a facet without its ISO code
    (`ResolvedFacet.currency`) still reads it; a unit the registry does not type reads its
    currency from that code alone."""
    spec = _spec(
        facets={"unit": ResolvedFacet(basis="fallback", ref=ref, label="Label")},
        resolved_sources=[_CRS],
    )
    plan = render_sentence_plan(spec)
    assert [t.text for t in plan if t.axis == "unit"] == expected


def test_a_latest_request_states_its_year_and_opens_the_byline_with_the_label():
    sentence, plan = _render(
        measurement_qualifier=_NOMINAL,
        facets=_unit("CurrencyUSD"),
        stat_vars=_served("2024", "2024"),
        resolved_sources=[_CRS],
        requested_window="latest",
    )
    assert sentence == (
        "Disbursements in 2024. Latest year with data · Current US dollars · "
        "OECD Creditor Reporting System (CRS)."
    )
    label = next(t for t in plan if t.text == LATEST_YEAR_LABEL)
    assert (label.kind, label.axis, label.segment) == ("text", None, "byline")
    # The heading's period is the one editable `when` token.
    assert [(t.text, t.segment) for t in plan if t.axis == "when"] == [("in 2024", "heading")]


def test_each_byline_item_opens_with_a_capital():
    """A lower-case source label opens with a capital in the byline."""
    sentence, _ = _render(
        facets=_unit("CurrencyUSD"),
        measurement_qualifier=_NOMINAL,
        resolved_sources=[_ref("src", "imputed multilateral sectors")],
    )
    assert sentence == "Disbursements. Current US dollars · Imputed multilateral sectors."


_WB = "WorldBankSubnationalPopulationEstimate"
_CENSUS = "CensusPEPSurvey"


def _method(ref: str | None, label: str | None = None) -> dict[str, ResolvedFacet]:
    if ref is None:
        return {"measurementMethod": ResolvedFacet(basis="bound", selection="per_series")}
    return {"measurementMethod": ResolvedFacet(basis="bound", ref=ref, label=label)}


@pytest.mark.parametrize(
    ("binding", "served", "expected"),
    [
        pytest.param(
            IdentityBindingValue(ref=_WB),
            _method(_WB, "World Bank Subnational Population Estimates"),
            "World Bank Subnational Population Estimates",
            id="ref",
        ),
        pytest.param(IdentityFacetValueAbsent(ref=None), {}, "No stated method", id="null-ref"),
        pytest.param(
            IdentityBindingSet(refs=[_CENSUS, _WB]),
            _method(_WB, "World Bank Subnational Population Estimates"),
            "World Bank Subnational Population Estimates",
            id="set-served-alike",
        ),
        pytest.param(
            IdentityBindingSet(refs=[_CENSUS, _WB]),
            _method(None),
            "Census pep survey and world bank subnational population estimate",
            id="set-per-series",
        ),
        pytest.param(None, _method(_WB, "World Bank estimates"), None, id="unbound"),
    ],
)
def test_a_bound_method_is_the_last_byline_token(binding, served, expected):
    """A bound method follows the source as a slot token stating `METHOD_REF`, and the
    byline still ends with one period. An unbound method states nothing, though the
    answer's facets name one."""
    sentence, plan = _render(
        measurement_qualifier=_NOMINAL,
        facets={**_unit("CurrencyUSD"), **served},
        resolved_sources=[_CRS],
        identity_facets=None if binding is None else {"measurementMethod": binding},
    )
    byline = "Current US dollars · OECD Creditor Reporting System (CRS)"
    if expected is None:
        assert sentence == f"Disbursements. {byline}."
        assert not any(METHOD_REF in t.states for t in plan)
        return
    assert sentence == f"Disbursements. {byline} · {expected}."
    method = plan[-2]
    assert method.model_dump(exclude_defaults=True) == {
        "kind": "slot",
        "text": expected,
        "axis": "method",
        "segment": "byline",
        "binding_ref": METHOD_REF,
        "states": [METHOD_REF],
    }
    assert plan[-3].text == " · " and plan[-4].axis == "source"


def test_a_heading_built_before_materialise_has_no_period_and_no_byline():
    """The bind-time heading has no StatVars, facets or sources: no period, and the byline
    holds no text."""
    spec = _spec(facets=_unit("CurrencyUSD"), resolved_sources=[_CRS])
    plan = plan_tokens(
        heading_from_parts(
            spec.resolved.slots,
            [],
            spec.resolved.shape,
            None,
            pinned_properties=frozenset(),
        )
    )
    assert sentence_from_plan(plan) == "Development finance flow."
    assert all(t.segment == "heading" for t in plan if t.text)
