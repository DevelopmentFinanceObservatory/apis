"""A converted answer names its currency from its read plan, whatever its preview holds.

`facets.answer_facets` builds `resolved.facets["unit"]` at grounding from the facets the read
plan serves, each reporting the currency a unit request converts to, so an omitted wide
preview and a narrow one render the same label. The label comes from grounding's one label
batch; a currency the batch did not label takes its display name from the unit lexicon.
"""

from __future__ import annotations

from search.engine.clients.graph import Facet
from search.engine.region.facets import answer_facet_dcids, answer_facets
from search.engine.retrieval.read_plan import PairPlan, PlanSeries
from search.models import IdentityFacetValueRequest, IdentityUnitRequest, Spec
from search.render.plan import render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import _spec


def _heading(spec: Spec) -> str:
    return sentence_from_plan(render_sentence_plan(spec))


_EUR_CONSTANT = IdentityUnitRequest(currency="EUR", prices="constant", base_year=2024)
_BASIS = {"currency": "bound", "prices": "bound", "base_year": "bound"}
_LABELS = {"CurrencyEUR": "Euro"}
# The graph names EUR a currency, so the byline states it.
_CURRENCY_UNITS = {"CurrencyEUR": "EUR", "CurrencyUSD": "USD"}


def _native_usd() -> Facet:
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=1,
        facet_id="f-usd",
        provenance_id="dc/base/Src",
        unit="CurrencyUSD",
    )


def _narrow_plan() -> PairPlan:
    """One series in its native US dollars at Senegal."""
    return PairPlan(
        members=("Test_SV",),
        years=None,
        series=(
            PlanSeries(
                stat_var="Test_SV",
                role_tuple=(("observationAbout", "country/SEN"),),
                candidates=(_native_usd(),),
                group=(_native_usd(),),
            ),
        ),
    )


def _wide_plan() -> PairPlan:
    """A scope-wide read of the same native US dollar facet at three places, whose converted
    preview is withheld (`observations.fetch._bulk_scope_dispatch`)."""
    places = ("country/SEN", "country/MLI", "country/NER")
    return PairPlan(
        members=("Test_SV",),
        years=None,
        series=tuple(
            PlanSeries(
                stat_var="Test_SV",
                role_tuple=(("observationAbout", place),),
                candidates=(_native_usd(),),
                group=(_native_usd(),),
            )
            for place in places
        ),
        scope=True,
        scope_facet_id="f-usd",
    )


def _converted_spec(
    plan: PairPlan,
    unit_request: IdentityUnitRequest = _EUR_CONSTANT,
    labels: dict[str, str] = _LABELS,
) -> Spec:
    return _spec(
        measure_label="gross disbursements",
        identity_facets={"unit": IdentityFacetValueRequest(request=unit_request)},
        facets=answer_facets(
            plan,
            labels,
            bound_axes={"unit": "bound"},
            request_basis=_BASIS,
            unit_request=unit_request,
            currency_of=_CURRENCY_UNITS.get,
        ),
    )


def test_an_omitted_wide_preview_renders_the_narrow_converted_label():
    wide = _converted_spec(_wide_plan())
    narrow = _converted_spec(_narrow_plan())

    assert _heading(wide) == _heading(narrow) == "Gross disbursements. Constant 2024 euros."
    assert wide.resolved.facets == narrow.resolved.facets


def test_the_unit_facet_names_the_requested_currency():
    unit = _converted_spec(_wide_plan()).resolved.facets["unit"]

    assert unit.basis == "bound"
    assert unit.ref == "CurrencyEUR"
    assert unit.label == "Euro"
    assert unit.currency == "EUR"
    assert unit.request_basis == _BASIS


def test_a_role_shaped_request_supplies_no_unit_facet():
    """A role currency resolves per series, so it names no single unit."""
    request = IdentityUnitRequest.model_validate(
        {"currency": {"role": "destination"}, "prices": "current"}
    )

    assert "unit" not in _converted_spec(_wide_plan(), request).resolved.facets


def test_the_requested_currency_joins_grounding_label_batch():
    assert "CurrencyEUR" in answer_facet_dcids(_wide_plan(), _EUR_CONSTANT)


def test_an_unlabelled_currency_is_named_by_its_iso_code():
    """A currency the label batch missed degrades to its ISO code, never to the "Currency
    EUR" dcid tail."""
    spec = _converted_spec(_wide_plan(), labels={})

    unit = spec.resolved.facets["unit"]
    assert unit.ref == "CurrencyEUR"
    assert unit.label == "EUR"
    assert _heading(spec) == "Gross disbursements. Constant 2024 euros."


def test_an_unlabelled_currency_takes_the_lexicon_display_name_when_one_exists():
    request = IdentityUnitRequest(currency="CHF", prices="current")

    unit = _converted_spec(_wide_plan(), request, labels={}).resolved.facets["unit"]

    assert unit.ref == "CurrencyCHF"
    assert unit.label == "Swiss franc"
