"""Facet elections must select a combination present on a probed facet."""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.errors import FacetNotAvailableError
from materialise.facets import (
    FacetBinding,
    UnitRequest,
    declared_default_policies,
    joint_dispatch_facet_ids,
    select_facet,
)
from materialise.generation import _generate_member, _select_member_facets
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.pipeline import MaterialisationRequest
from materialise.rate_tables import RateTableCache
from materialise.window import RequestedWindow

from .conftest import FakeGraphReader

_SDMX_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR = "dcid:Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_BARE_STAT_VAR = "Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_ENTITY = "country/FRA"
_ROLES: tuple[str, ...] = ("observationAbout",)
_USD_PROVENANCE = "provenance/P1"
_EUR_PROVENANCE = "provenance/P2"
_DAC_RECIPE = "ConversionRecipeDAC"

_SUBJECT_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

# EUR outranks USD by obs_count (1000 vs 1), but a native USD request
# must select the USD facet on every axis.
_FACET_USD = Facet(
    facet_id="f_usd",
    provenance_id=_USD_PROVENANCE,
    measurement_method="M1",
    unit="CurrencyUSD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)
_FACET_EUR = Facet(
    facet_id="f_eur",
    provenance_id=_EUR_PROVENANCE,
    measurement_method="M2",
    unit="CurrencyEUR",
    obs_count=1000,
    earliest_date="2020",
    latest_date="2020",
)


def _arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        _USD_PROVENANCE: {"conversionRecipe": (_DAC_RECIPE,), "isPartOf": ()},
        _EUR_PROVENANCE: {"conversionRecipe": (_DAC_RECIPE,), "isPartOf": ()},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
        _BARE_STAT_VAR: {
            "measurementQualifier": ("Nominal",),
            "populationType": ("FinancialFlow",),
        },
    }


def _request(*, unit: FacetBinding) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=_ROLES,
        facet_bindings={"unit": unit},
        window=RequestedWindow(kind="all"),
    )


def _obs_row(*, unit: str, provenance: str, method: str, facet_id: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_BARE_STAT_VAR},{_ENTITY},{unit},{method},"
        f"P1Y,{provenance},2020,100,,{facet_id}\r\n"
    )


def _router_sdmx(bodies_by_stat_var: dict[str, str]) -> tuple[SdmxClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = dict(httpx.QueryParams(request.url.query.decode()))
        stat_var = params["c[variableMeasured]"]
        return httpx.Response(
            200,
            text=bodies_by_stat_var[stat_var],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url=_SDMX_BASE_URL, transport=httpx.MockTransport(handler)), requests


def test_generate_member_recovers_the_native_currency_facet_over_a_higher_ranked_disagreeing_alternative():
    """The EUR facet leads with 1000 observations, while the USD/current
    request selects USD. Every axis must resolve to the facet with the real
    USD observation. Axes once ranked independently, elected a combination no facet carries,
    and the member returned a silent empty result.
    """
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, _ENTITY): [_FACET_USD, _FACET_EUR]}, arcs=_arcs()
    )
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("USD", "current")))
    sdmx, requests = _router_sdmx(
        {
            _BARE_STAT_VAR: _SUBJECT_HEADER
            + _obs_row(
                unit="CurrencyUSD", provenance=_USD_PROVENANCE, method="M1", facet_id="f_usd"
            )
        }
    )

    outcome = _generate_member(
        request,
        sdmx=sdmx,
        graph=graph,
        ceiling=DEFAULT_SYNC_CEILING_ROWS,
        rate_cache=RateTableCache(),
    )

    assert outcome.selections["provenance"].ref == _USD_PROVENANCE
    assert outcome.selections["measurementMethod"].ref == "M1"
    assert outcome.selections["unit"].ref == "CurrencyUSD"
    assert outcome.selections["unit"].selection_reason == "native_currency_preferred"
    assert outcome.coverage.rows == 1  # the real USD row, not an empty degrade
    assert requests  # a real SDMX fetch ran, not a silent skip


def test_select_facet_elects_a_coherent_combination_directly():
    """The same reproduction at `facets.select_facet`'s own level, matching
    `verify-p1-facet-tuple.md`'s own repro script: all three axes elect
    `f_usd`, and `joint_dispatch_facet_ids` finds it.
    """
    all_bindings = {"unit": FacetBinding(state="request", request=UnitRequest("USD", "current"))}
    candidates = [_FACET_USD, _FACET_EUR]
    selections = {
        axis: select_facet(
            axis,
            all_bindings.get(axis, FacetBinding(state="unbound")),
            candidates=candidates,
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            all_bindings=all_bindings,
            window=None,
            policies=declared_default_policies(
                all_bindings, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )
        for axis in ("provenance", "measurementMethod", "unit")
    }

    assert selections["provenance"].ref == _USD_PROVENANCE
    assert selections["measurementMethod"].ref == "M1"
    assert selections["unit"].ref == "CurrencyUSD"
    assert joint_dispatch_facet_ids(candidates, selections) == ("f_usd",)


def test_no_currency_match_falls_through_to_the_ordinary_disagreeing_ranking():
    """Without a USD candidate, all axes rank the same pool and choose the
    higher-count EUR facet.
    """
    facet_gbp = Facet(
        facet_id="f_gbp",
        provenance_id=_USD_PROVENANCE,
        measurement_method="M1",
        unit="CurrencyGBP",
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
    )
    all_bindings = {"unit": FacetBinding(state="request", request=UnitRequest("USD", "current"))}

    provenance_selection = select_facet(
        "provenance",
        FacetBinding(state="unbound"),
        candidates=[facet_gbp, _FACET_EUR],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        all_bindings=all_bindings,
        window=None,
        policies=declared_default_policies(
            all_bindings, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert provenance_selection.ref == _EUR_PROVENANCE  # f_eur wins by obs_count
    assert provenance_selection.selection_reason == "most_observations_over_disagreement"


def test_select_member_facets_refuses_rather_than_degrading_when_a_declared_policy_still_diverges():
    """A declared provenance policy can select EUR while the unit request
    selects USD. The incoherent selection must raise `FacetNotAvailableError`.
    """
    policy = FacetDefaultPolicy(
        resolved_ref=_EUR_PROVENANCE,
        policy_node="policy/provenance-default",
        policy_version="v1",
        selection_reason="declared for this StatVar",
    )
    graph = FakeGraphReader(policies={(_STAT_VAR, "provenance"): policy})
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("USD", "current")))

    with pytest.raises(FacetNotAvailableError) as excinfo:
        _select_member_facets(request, [_FACET_USD, _FACET_EUR], graph=graph)

    assert excinfo.value.code == "facet_not_available"
    assert excinfo.value.issues is not None
    assert {issue.axis for issue in excinfo.value.issues} == {
        "provenance",
        "measurementMethod",
        "unit",
    }
