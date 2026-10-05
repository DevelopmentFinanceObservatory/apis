"""Tests request-form unit convertibility in options-surface membership decisions.

The fixtures add a unit request after target resolution to isolate surface behaviour."""

from __future__ import annotations

import copy
import dataclasses
import time

import pytest

from materialise.graph import SubjectPage
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError
from search.engine.options.resolve import resolve_options_target
from search.engine.options.surface import build_options_response
from search.models import (
    IdentityBindingSet,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityShapeSchema,
    IdentityUnitRequest,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import materialise_double_from_obs
from tests.engine._options_fixtures import (
    _GDP_PLACE_CHILDREN,
    _GDP_PLACE_NODES,
    _gdp_sv_node,
)
from tests.engine._options_fixtures import _standard_place_request as _gdp_place_request
from tests.fixtures import FakeGraph

_STANDARD_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GTM = "country/GTM"
_USA = "country/USA"
_PROVENANCE = "dc/base/OptElectSrc"
_RECIPE = "ConversionRecipeDAC"

_CURRENCY_ARCS = {
    "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
}


_GDP_ACTIVITY_SOURCE = "GrossDomesticProduction"
# A fixture-only measured property keeps the shape out of the packaged index, so population
# comes from graph arcs. `FinancialFlow` is a population type the DAC recipe converts.
_MEASURED_PROPERTY = "optElectAmount"
_FLOW_SHAPE = IdentityShapeSchema(
    population_type="FinancialFlow",
    measured_property=_MEASURED_PROPERTY,
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
# _STANDARD_GDP_SV's own activitySource value, asserted on its graph node
# (tests/fixtures/graph_nodes.json).


def _standard_place_request(**kwargs):
    """The GDP place request over a `FinancialFlow` population, the type the DAC recipe converts."""
    req = _gdp_place_request(**kwargs)
    return req.model_copy(
        update={"identity": req.identity.model_copy(update={"shape": _FLOW_SHAPE})}
    )


def _gdp_sv_arcs(
    *, activity_source: str = _GDP_ACTIVITY_SOURCE, population_type: str = "FinancialFlow"
) -> dict[str, tuple[str, ...]]:
    """Builds a GDP StatVar's five-tuple and `activitySource` arcs for population derivation.

    The default `FinancialFlow` population type is one the DAC recipe converts."""
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": (population_type,),
        "measuredProperty": (_MEASURED_PROPERTY,),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "measurementDenominator": (),
        "constraintProperties": ("activitySource",),
        "observationProperties": (),
        "activitySource": (activity_source,),
    }


def _materialise_arcs(*, provenance_declares_recipe: bool) -> dict[str, dict]:
    """Builds GDP and currency arcs, with an optional recipe on the provenance."""
    return {
        _STANDARD_GDP_SV: _gdp_sv_arcs(),
        **_CURRENCY_ARCS,
        _PROVENANCE: {"conversionRecipe": (_RECIPE,)} if provenance_declares_recipe else {},
    }


_GDP_SUBJECTS = {
    (_MEASURED_PROPERTY, "measuredProperty"): SubjectPage(
        dcids=(_STANDARD_GDP_SV,), truncated=False
    )
}
# enumerate_members' own reverse query (node_subjects(_MEASURED_PROPERTY,
# property="measuredProperty", ...)) -- FakeGraphReader.node_subjects
# answers strictly from this explicit mapping rather than scanning `arcs`
# itself, so every population fixture in this file needs a matching entry.


def _obs_facet(*, unit: str, facet_id: str, method: str | None = None) -> dict:
    facet = {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": _PROVENANCE,
        "unit": unit,
        "dates": ["2020"],
        "values": [10.0],
        "facetId": facet_id,
    }
    if method is not None:
        facet["measurementMethod"] = method
    return facet


def _with_unit_request(
    target,
    unit_request: IdentityUnitRequest,
    other_facets: dict[str, IdentityFacetBinding] | None = None,
):
    """Adds a unit request, and any other facet pins, to a resolved target without changing
    its population, anchors, or window."""
    return dataclasses.replace(
        target,
        facets={"unit": IdentityFacetValueRequest(request=unit_request), **(other_facets or {})},
    )


def _present_dcids(
    *,
    obs: dict,
    provenance_declares_recipe: bool,
    unit_request: IdentityUnitRequest | None,
    other_facets: dict[str, IdentityFacetBinding] | None = None,
):
    graph = FakeGraph(
        nodes=_GDP_PLACE_NODES,
        obs=obs,
        detect={},
        resolve={},
        children=_GDP_PLACE_CHILDREN,
        enum_members={},
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        provenance_declares_recipe=provenance_declares_recipe
    )
    materialise_graph._subjects = _GDP_SUBJECTS
    req = _standard_place_request(entities={"observationAbout": [_GTM]})
    target = resolve_options_target(req, graph=graph, materialise_graph=materialise_graph)
    if unit_request is not None:
        target = _with_unit_request(target, unit_request, other_facets)
    response = build_options_response(
        req,
        target,
        graph=graph,
        start_ms=now_ms(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    return {item.value.ref.dcid for item in response.options.items}


def test_request_form_unit_counts_only_the_facet_a_mint_would_elect():
    """{USD, current} on a stat_var carrying a native CurrencyUSD facet
    (Guatemala) and a native CurrencyEUR facet (USA), no recipe needed on
    either side: the native USD match wins outright, so elected_units admits
    only Guatemala. The literal unit_refs.satisfies rule would admit both,
    the over-admission this decision fixes."""
    obs = {
        f"{_STANDARD_GDP_SV}|{_GTM}": [_obs_facet(unit="CurrencyUSD", facet_id="f_gtm")],
        f"{_STANDARD_GDP_SV}|{_USA}": [_obs_facet(unit="CurrencyEUR", facet_id="f_usa")],
    }
    present = _present_dcids(
        obs=obs,
        provenance_declares_recipe=False,
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
    )
    assert present == {_GTM}


def test_request_form_unit_with_a_recipe_counts_the_converting_native_facet():
    """{EUR, current} against a stat_var carrying only a native CurrencyUSD
    facet, under a provenance that declares a conversionRecipe: the decision
    elects the native USD facet (the one a mint would actually convert), so
    membership admits Guatemala."""
    obs = {f"{_STANDARD_GDP_SV}|{_GTM}": [_obs_facet(unit="CurrencyUSD", facet_id="f_gtm")]}
    present = _present_dcids(
        obs=obs,
        provenance_declares_recipe=True,
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
    )
    assert present == {_GTM}


def test_a_null_method_pin_decides_the_unit_among_the_facets_with_no_method():
    """GDP at Guatemala carries a recorded US dollar series with no method and an Estimated
    euro series, under a provenance that declares the DAC recipe. With the method pinned to
    the facets that declare none and euros requested, the decision converts the recorded
    series, as a mint would serve it, so Guatemala is offered. Deciding with the method open
    elects the native euro facet, which the null-method pin then rejects."""
    obs = {
        f"{_STANDARD_GDP_SV}|{_GTM}": [
            _obs_facet(unit="CurrencyUSD", facet_id="f_recorded"),
            _obs_facet(unit="CurrencyEUR", facet_id="f_estimated", method="Estimated"),
        ]
    }
    present = _present_dcids(
        obs=obs,
        provenance_declares_recipe=True,
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        other_facets={"measurementMethod": IdentityFacetValueAbsent(ref=None)},
    )
    assert present == {_GTM}


def test_a_method_set_pin_decides_the_unit_for_each_method_in_the_set():
    """GDP at Guatemala carries US dollars under Recorded and euros under Estimated, under a
    provenance that declares the DAC recipe. With euros requested and the method pinned to
    the set {Recorded, Estimated}, the decision runs per method: it converts the Recorded
    series and takes the Estimated one natively, so Guatemala is offered. Deciding with the
    method open elects only the euro facet and leaves Recorded with no unit."""
    obs = {
        f"{_STANDARD_GDP_SV}|{_GTM}": [
            _obs_facet(unit="CurrencyUSD", facet_id="f_recorded", method="Recorded"),
            _obs_facet(unit="CurrencyEUR", facet_id="f_estimated", method="Estimated"),
        ]
    }
    present = _present_dcids(
        obs=obs,
        provenance_declares_recipe=True,
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        other_facets={"measurementMethod": IdentityBindingSet(refs=["Recorded", "Estimated"])},
    )
    assert present == {_GTM}


def test_request_form_unit_with_no_recipe_refuses_with_the_doors_own_code_and_cause():
    """{EUR, current} against the same native-USD-only stat_var, under a
    provenance lacking a conversionRecipe: decide_part_unit raises
    RecipeNotDeclaredError, and the options surface raises the identical
    EngineInputError shape region/promote_ground.py's _phase_ground_build
    raises for it (code, cause.code, pointer)."""
    obs = {f"{_STANDARD_GDP_SV}|{_GTM}": [_obs_facet(unit="CurrencyUSD", facet_id="f_gtm")]}
    with pytest.raises(EngineInputError) as exc_info:
        _present_dcids(
            obs=obs,
            provenance_declares_recipe=False,
            unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        )
    exc = exc_info.value
    assert exc.code == "recipe_not_declared"
    assert exc.status == 422
    assert exc.pointer == "/identity/facets/unit"
    assert exc.cause is not None
    assert exc.cause.code == "recipe_undeclared"


def test_deadline_is_threaded_into_the_request_form_unit_decision():
    """The request deadline reaches `variable_facets_batch` in the unit-conversion probe."""
    obs = {f"{_STANDARD_GDP_SV}|{_GTM}": [_obs_facet(unit="CurrencyUSD", facet_id="f_gtm")]}
    graph = FakeGraph(
        nodes=_GDP_PLACE_NODES,
        obs=obs,
        detect={},
        resolve={},
        children=_GDP_PLACE_CHILDREN,
        enum_members={},
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(provenance_declares_recipe=False)
    materialise_graph._subjects = _GDP_SUBJECTS

    deadlines: list[float | None] = []
    original_variable_facets_batch = materialise_graph.variable_facets_batch

    def _capturing_variable_facets_batch(*, stat_var, entities, deadline=None):
        deadlines.append(deadline)
        return original_variable_facets_batch(
            stat_var=stat_var, entities=entities, deadline=deadline
        )

    materialise_graph.variable_facets_batch = _capturing_variable_facets_batch

    req = _standard_place_request(entities={"observationAbout": [_GTM]})
    target = resolve_options_target(req, graph=graph, materialise_graph=materialise_graph)
    target = _with_unit_request(target, IdentityUnitRequest(currency="USD", prices="current"))

    before = time.monotonic()
    build_options_response(
        req,
        target,
        graph=graph,
        start_ms=now_ms(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert deadlines, "the request-form unit decision's own probe must have been issued at all"
    assert all(d is not None for d in deadlines)
    assert all(d > before for d in deadlines)


def test_no_request_form_unit_is_unaffected():
    """Without a unit binding, both candidates remain regardless of currency."""
    obs = {
        f"{_STANDARD_GDP_SV}|{_GTM}": [_obs_facet(unit="CurrencyUSD", facet_id="f_gtm")],
        f"{_STANDARD_GDP_SV}|{_USA}": [_obs_facet(unit="CurrencyEUR", facet_id="f_usa")],
    }
    present = _present_dcids(obs=obs, provenance_declares_recipe=False, unit_request=None)
    assert present == {_GTM, _USA}


# ---------------------------------------------------------------------------
# Two resolved stat_vars: elected_units must narrow per stat_var, never a
# union pooled across stat_vars.
# ---------------------------------------------------------------------------

_SV_B = "Amount_EconomicActivity_GrossDomesticProduction_Nominal_ElectedTwoSv"
_SV_B_ACTIVITY_SOURCE = "OptElectTwoSvAltSource"
# _STANDARD_GDP_SV's own activitySource value, asserted on its graph node
# (tests/fixtures/graph_nodes.json).
_SV1_ACTIVITY_SOURCE = "GrossDomesticProduction"
_PROVENANCE_SV1 = "dc/base/OptElectTwoSvSrc1"
_PROVENANCE_SV2 = "dc/base/OptElectTwoSvSrc2"

_CURRENCY_ARCS_TWO_SV = {
    "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    "CurrencyGBP": {"typeOf": ("CurrencyUnit",)},
    # The request's currency (see the test docstring): not a native facet
    # in this fixture, but it must resolve as a currency for
    # decide_member_conversion's EUR-side lookup once each stat_var's native
    # facet converts into it.
    "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
}


def _sv_b_node() -> dict:
    """Mirrors `_SV_B` in the search graph with its distinct `activitySource` value."""
    node = copy.deepcopy(_gdp_sv_node())
    node["arcs"]["activitySource"]["nodes"][0]["dcid"] = _SV_B_ACTIVITY_SOURCE
    node["arcs"]["activitySource"]["nodes"][0]["name"] = _SV_B_ACTIVITY_SOURCE
    return node


def _materialise_arcs_two_sv() -> dict[str, dict]:
    """Builds two GDP members with the same constraint key and distinct `activitySource` values.

    Their equal key sets keep both StatVars in the population."""
    return {
        _STANDARD_GDP_SV: _gdp_sv_arcs(activity_source=_SV1_ACTIVITY_SOURCE),
        _SV_B: _gdp_sv_arcs(activity_source=_SV_B_ACTIVITY_SOURCE),
        **_CURRENCY_ARCS_TWO_SV,
        # Both provenances name the same real recipe body (RECIPE_BODIES):
        # what this test turns on is each stat_var electing a different
        # native currency through the obs_count ranking, which the
        # observation facets decide, not the recipe.
        _PROVENANCE_SV1: {"conversionRecipe": (_RECIPE,)},
        _PROVENANCE_SV2: {"conversionRecipe": (_RECIPE,)},
    }


_GDP_SUBJECTS_TWO_SV = {
    (_MEASURED_PROPERTY, "measuredProperty"): SubjectPage(
        dcids=(_STANDARD_GDP_SV, _SV_B), truncated=False
    )
}


def _obs_facet_ranked(*, unit: str, facet_id: str, provenance: str, obs_count: int) -> dict:
    """Adds provenance and `obsCount` to control each StatVar's elected facet."""
    return {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": obs_count,
        "provenanceId": provenance,
        "unit": unit,
        "dates": ["2020"],
        "values": [10.0],
        "facetId": facet_id,
    }


def test_request_form_unit_narrows_per_stat_var_not_pooled_across_stat_vars():
    """For EUR conversion, Guatemala's top-ranked USD and GBP facets are elected per StatVar.

    The USA has the opposite, lower-ranked facets. Per-StatVar filtering excludes it; pooling
    elected facets would admit it."""
    obs = {
        f"{_STANDARD_GDP_SV}|{_GTM}": [
            _obs_facet_ranked(
                unit="CurrencyUSD", facet_id="sv1_gtm", provenance=_PROVENANCE_SV1, obs_count=3
            )
        ],
        f"{_STANDARD_GDP_SV}|{_USA}": [
            _obs_facet_ranked(
                unit="CurrencyGBP", facet_id="sv1_usa", provenance=_PROVENANCE_SV1, obs_count=1
            )
        ],
        f"{_SV_B}|{_GTM}": [
            _obs_facet_ranked(
                unit="CurrencyGBP", facet_id="sv2_gtm", provenance=_PROVENANCE_SV2, obs_count=3
            )
        ],
        f"{_SV_B}|{_USA}": [
            _obs_facet_ranked(
                unit="CurrencyUSD", facet_id="sv2_usa", provenance=_PROVENANCE_SV2, obs_count=1
            )
        ],
    }
    graph = FakeGraph(
        nodes={**_GDP_PLACE_NODES, _SV_B: _sv_b_node()},
        obs=obs,
        detect={},
        resolve={},
        children=_GDP_PLACE_CHILDREN,
        enum_members={},
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs_two_sv()
    materialise_graph._subjects = _GDP_SUBJECTS_TWO_SV
    req = _standard_place_request(
        entities={"observationAbout": [_GTM]},
        activity_source_refs=[_SV1_ACTIVITY_SOURCE, _SV_B_ACTIVITY_SOURCE],
    )
    target = resolve_options_target(req, graph=graph, materialise_graph=materialise_graph)
    # Both StatVars must survive derivation; otherwise a one-member union would mask pooling.
    resolved_sv_dcids = set(target.population.stat_var_dcids)
    assert resolved_sv_dcids == {_STANDARD_GDP_SV, _SV_B}, (
        f"expected both {_STANDARD_GDP_SV!r} and {_SV_B!r} in the resolved family, "
        f"got {resolved_sv_dcids!r} -- the two-stat_var premise this test relies on "
        "no longer holds"
    )
    target = _with_unit_request(target, IdentityUnitRequest(currency="EUR", prices="current"))
    response = build_options_response(
        req,
        target,
        graph=graph,
        start_ms=now_ms(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    present = {item.value.ref.dcid for item in response.options.items}
    assert present == {_GTM}
    assert _USA not in present
