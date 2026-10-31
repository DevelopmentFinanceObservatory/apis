"""Tests `decide_member_unit` selection and refusals before observation or rate-table loads.

Real StatVar and entity dcids are used. Several cases compare the decision with `_generate_member`.
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import decide_group_unit, decide_member_unit
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.errors import DefaultFacetUndefinedError
from materialise.facets import CoverageAdmission, FacetBinding, MemberProbe, UnitRequest
from materialise.generation import _generate_member
from materialise.graph import Facet
from materialise.pipeline import MaterialisationRequest
from materialise.rate_tables import RateTableCache
from materialise.recipes import (
    NonCurrencyUnitError,
    RecipeNotDeclaredError,
    RoleAxisNotDeclaredError,
)
from materialise.window import RequestedWindow
from tests.engine._fake_coverage import build_coverage_index

from .conftest import FakeGraphReader

_SDMX_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR = "dcid:Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_BARE_STAT_VAR = "Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_ENTITY = "country/FRA"
_ROLES: tuple[str, ...] = ("observationAbout",)
_MEMBER_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_OTHER_PROVENANCE = "provenance/OtherSource"
_TABLE_PROVENANCE = "provenance/ONEConversionTablesDAC"
_DAC_RECIPE = "ConversionRecipeDAC"
# Subject-only requests use graph candidates; the required coverage index is empty.
_SUBJECT_ONLY_COVERAGE = build_coverage_index([], role_names={})

_SUBJECT_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _obs_row(
    *,
    stat_var: str = _BARE_STAT_VAR,
    entity: str = _ENTITY,
    unit: str,
    provenance: str = _MEMBER_PROVENANCE,
    year: str,
    value: str,
    facet_id: str = "f1",
) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{entity},{unit},NotApplicable,"
        f"P1Y,{provenance},{year},{value},,{facet_id}\r\n"
    )


def _router_sdmx(bodies_by_stat_var: dict[str, str]) -> tuple[SdmxClient, list[httpx.Request]]:
    """One `httpx.MockTransport` keyed by `variableMeasured`, following
    `test_conversion_generation.py`'s own convention. Every test using
    subject-only `_ROLES` probes the graph, not `sdmx`
    (`facets.probe_facets`'s own default-declared-roles branch), so
    `requests` records only a mint's own observation fetch and table
    loads -- the calls `decide_member_unit` must never make.
    """
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


def _facet(
    *,
    facet_id: str,
    unit: str,
    provenance: str = _MEMBER_PROVENANCE,
    obs_count: int = 5,
) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method="NotApplicable",
        unit=unit,
        obs_count=obs_count,
        earliest_date="2020",
        latest_date="2022",
    )


def _base_arcs(
    *,
    qualifiers: tuple[str, ...] = ("Nominal",),
    member_provenance_arcs: dict[str, tuple[str, ...]] | None = None,
    other_provenance_arcs: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    statvar_arcs: dict[str, tuple[str, ...]] = {
        "measurementQualifier": qualifiers,
        "populationType": ("FinancialFlow",),
    }
    return {
        _MEMBER_PROVENANCE: (
            member_provenance_arcs
            if member_provenance_arcs is not None
            else {"conversionRecipe": (_DAC_RECIPE,), "isPartOf": ()}
        ),
        _OTHER_PROVENANCE: (
            other_provenance_arcs
            if other_provenance_arcs is not None
            else {"conversionRecipe": (_DAC_RECIPE,), "isPartOf": ()}
        ),
        _TABLE_PROVENANCE: {"lastDataRefreshDate": ("2026-01-01T00:00:00Z",)},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
        _BARE_STAT_VAR: statvar_arcs,
    }


def _graph(
    facets: list[Facet], arcs: dict[str, dict[str, tuple[str, ...]]] | None = None
) -> FakeGraphReader:
    return FakeGraphReader(
        facets={(_BARE_STAT_VAR, _ENTITY): facets},
        arcs=arcs if arcs is not None else _base_arcs(),
    )


def _request(
    *,
    unit: FacetBinding,
    declared_roles: tuple[str, ...] = _ROLES,
    role_bindings: dict[str, list[str]] | None = None,
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings=role_bindings
        if role_bindings is not None
        else {"observationAbout": [_ENTITY]},
        declared_roles=declared_roles,
        facet_bindings={"unit": unit},
        window=RequestedWindow(kind="all"),
    )


# ---------------------------------------------------------------------------
# The entry elects the same dispatch facet a mint elects
# ---------------------------------------------------------------------------


def test_entry_matches_the_minted_native_currency_preference():
    # EUR carries the higher `obs_count`, but `{currency: "USD", prices:
    # "current"}`'s own native preference overrides ranking on both paths.
    facets = [
        _facet(facet_id="f1", unit="CurrencyUSD", obs_count=5),
        _facet(facet_id="f2", unit="CurrencyEUR", obs_count=50),
    ]
    graph = _graph(facets)
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("USD", "current")))

    entry_sdmx, entry_requests = _router_sdmx({})
    selections = decide_member_unit(
        request, settle=True, graph=graph, sdmx=entry_sdmx, coverage=_SUBJECT_ONLY_COVERAGE
    )
    assert selections["unit"].ref == "CurrencyUSD"
    assert entry_requests == []

    mint_sdmx, mint_requests = _router_sdmx(
        {_BARE_STAT_VAR: _SUBJECT_HEADER + _obs_row(unit="CurrencyUSD", year="2022", value="100")}
    )
    outcome = _generate_member(
        request,
        sdmx=mint_sdmx,
        graph=graph,
        ceiling=DEFAULT_SYNC_CEILING_ROWS,
        rate_cache=RateTableCache(),
    )
    assert outcome.selections["unit"].ref == "CurrencyUSD"
    assert mint_requests  # the mint's own fetch made at least one real SDMX call


def test_entry_matches_the_minted_ranked_default_for_an_unbound_unit():
    facets = [
        _facet(facet_id="f1", unit="CurrencyUSD", obs_count=5),
        _facet(facet_id="f2", unit="CurrencyEUR", obs_count=50),
    ]
    graph = _graph(facets)
    request = _request(unit=FacetBinding(state="unbound"))

    entry_sdmx, entry_requests = _router_sdmx({})
    selections = decide_member_unit(
        request, settle=True, graph=graph, sdmx=entry_sdmx, coverage=_SUBJECT_ONLY_COVERAGE
    )
    assert selections["unit"].ref == "CurrencyEUR"  # ranked top by obs_count
    assert entry_requests == []

    mint_sdmx, _mint_requests = _router_sdmx(
        {
            _BARE_STAT_VAR: _SUBJECT_HEADER
            + _obs_row(unit="CurrencyEUR", facet_id="f2", year="2022", value="100")
        }
    )
    outcome = _generate_member(
        request, sdmx=mint_sdmx, graph=graph, ceiling=DEFAULT_SYNC_CEILING_ROWS, rate_cache=None
    )
    assert outcome.selections["unit"].ref == "CurrencyEUR"


# ---------------------------------------------------------------------------
# Natively satisfied: nominal check alone
# ---------------------------------------------------------------------------


def test_natively_satisfied_passes_with_no_measurement_qualifier_arc():
    # No arithmetic runs on this path, so a StatVar carrying no
    # measurementQualifier arc at all passes.
    facets = [_facet(facet_id="f1", unit="CurrencyUSD")]
    graph = _graph(facets, arcs=_base_arcs(qualifiers=()))
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("USD", "current")))
    sdmx, requests = _router_sdmx({})

    selections = decide_member_unit(
        request, settle=True, graph=graph, sdmx=sdmx, coverage=_SUBJECT_ONLY_COVERAGE
    )
    assert selections["unit"].ref == "CurrencyUSD"
    assert requests == []


def test_natively_satisfied_refuses_a_declared_non_nominal_qualifier():
    facets = [_facet(facet_id="f1", unit="CurrencyUSD")]
    graph = _graph(facets, arcs=_base_arcs(qualifiers=("RealValue",)))
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("USD", "current")))
    sdmx, requests = _router_sdmx({})

    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        decide_member_unit(
            request, settle=True, graph=graph, sdmx=sdmx, coverage=_SUBJECT_ONLY_COVERAGE
        )
    assert exc_info.value.cause == "nominal_undeclared"
    assert requests == []


# ---------------------------------------------------------------------------
# No recipe declared: refuses on the entry and on a mint alike
# ---------------------------------------------------------------------------


def test_no_recipe_declared_refuses_on_the_entry_and_on_a_mint_alike():
    facets = [_facet(facet_id="f1", unit="CurrencyUSD")]
    graph = _graph(
        facets, arcs=_base_arcs(member_provenance_arcs={"conversionRecipe": (), "isPartOf": ()})
    )
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))

    entry_sdmx, entry_requests = _router_sdmx({})
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        decide_member_unit(
            request, settle=True, graph=graph, sdmx=entry_sdmx, coverage=_SUBJECT_ONLY_COVERAGE
        )
    assert exc_info.value.cause == "recipe_undeclared"
    assert entry_requests == []

    mint_sdmx, mint_requests = _router_sdmx({})
    with pytest.raises(RecipeNotDeclaredError) as exc_info2:
        _generate_member(
            request,
            sdmx=mint_sdmx,
            graph=graph,
            ceiling=DEFAULT_SYNC_CEILING_ROWS,
            rate_cache=RateTableCache(),
        )
    assert exc_info2.value.cause == "recipe_undeclared"
    assert mint_requests == []  # refuses before any fetch, on both paths


def test_elected_facet_with_no_recipe_refuses_even_though_a_lower_ranked_one_has_it():
    # The elected facet's missing recipe triggers refusal even when a lower-ranked facet has one.
    elected = _facet(facet_id="f1", unit="CurrencyUSD", provenance=_MEMBER_PROVENANCE, obs_count=50)
    other = _facet(facet_id="f2", unit="CurrencyUSD", provenance=_OTHER_PROVENANCE, obs_count=5)
    graph = _graph(
        [elected, other],
        arcs=_base_arcs(member_provenance_arcs={"conversionRecipe": (), "isPartOf": ()}),
    )
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))

    entry_sdmx, entry_requests = _router_sdmx({})
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        decide_member_unit(
            request, settle=True, graph=graph, sdmx=entry_sdmx, coverage=_SUBJECT_ONLY_COVERAGE
        )
    assert exc_info.value.cause == "recipe_undeclared"
    assert entry_requests == []

    mint_sdmx, mint_requests = _router_sdmx({})
    with pytest.raises(RecipeNotDeclaredError) as exc_info2:
        _generate_member(
            request,
            sdmx=mint_sdmx,
            graph=graph,
            ceiling=DEFAULT_SYNC_CEILING_ROWS,
            rate_cache=RateTableCache(),
        )
    assert exc_info2.value.cause == "recipe_undeclared"
    assert mint_requests == []


# ---------------------------------------------------------------------------
# Role-shaped currency, undeclared axis
# ---------------------------------------------------------------------------


def test_role_shaped_currency_with_an_undeclared_axis_raises_the_named_error():
    facets = [_facet(facet_id="f1", unit="CurrencyUSD")]
    graph = _graph(facets)
    # Only `observationAbout` is declared; `destination` is outside the shape.
    request = _request(
        unit=FacetBinding(state="request", request=UnitRequest("destination", "current")),
        declared_roles=("observationAbout",),
    )
    sdmx, requests = _router_sdmx({})

    with pytest.raises(RoleAxisNotDeclaredError) as exc_info:
        decide_member_unit(
            request, settle=True, graph=graph, sdmx=sdmx, coverage=_SUBJECT_ONLY_COVERAGE
        )
    assert exc_info.value.cause == "role_undeclared"
    assert requests == []


# ---------------------------------------------------------------------------
# `NonCurrencyUnitError` propagation: unit-only vs. mixed with another axis
# ---------------------------------------------------------------------------


def test_unit_only_non_currency_failure_propagates_unchanged():
    # The sole candidate's unit is `Percent`, no currency, so the request's
    # default fallback (no `CurrencyEUR` candidate exists) elects it and then
    # refuses on the currency check. Provenance and measurementMethod both
    # resolve against the same sole candidate, so the unit axis is the only
    # one that fails.
    facets = [_facet(facet_id="f1", unit="Percent")]
    graph = _graph(facets)
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))

    entry_sdmx, entry_requests = _router_sdmx({})
    with pytest.raises(NonCurrencyUnitError) as exc_info:
        decide_member_unit(
            request, settle=True, graph=graph, sdmx=entry_sdmx, coverage=_SUBJECT_ONLY_COVERAGE
        )
    assert exc_info.value.code == "facet_binding_unsupported"
    assert exc_info.value.unit == "Percent"
    assert entry_requests == []

    mint_sdmx, mint_requests = _router_sdmx({})
    with pytest.raises(NonCurrencyUnitError) as exc_info2:
        _generate_member(
            request,
            sdmx=mint_sdmx,
            graph=graph,
            ceiling=DEFAULT_SYNC_CEILING_ROWS,
            rate_cache=RateTableCache(),
        )
    assert exc_info2.value.code == "facet_binding_unsupported"
    assert mint_requests == []  # refuses before any fetch, on both paths


def test_unit_failure_plus_a_provenance_failure_keeps_the_aggregate():
    # The sole candidate carries no provenance, so the unbound
    # provenance axis refuses `default_facet_undefined` on its own,
    # independently of the unit axis's own non-currency refusal, which
    # would otherwise propagate alone. With both axes failing,
    # `_select_member_facets` keeps the aggregate instead.
    facet = Facet(
        facet_id="f1",
        provenance_id=None,
        measurement_method="NotApplicable",
        unit="Percent",
        obs_count=5,
        earliest_date="2020",
        latest_date="2022",
    )
    graph = _graph([facet])
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))

    entry_sdmx, entry_requests = _router_sdmx({})
    with pytest.raises(DefaultFacetUndefinedError) as exc_info:
        decide_member_unit(
            request, settle=True, graph=graph, sdmx=entry_sdmx, coverage=_SUBJECT_ONLY_COVERAGE
        )
    assert exc_info.value.code == "default_facet_undefined"
    assert {issue.axis for issue in exc_info.value.issues} == {"provenance", "unit"}
    assert entry_requests == []


@pytest.mark.parametrize(
    ("currency", "requires_provenance", "expected"),
    [
        # A search read states no source: a native request selects the unsourced facet.
        ("USD", False, "absent_on_selected_facet"),
        # No provenance declares a `conversionRecipe` for a converting request.
        ("EUR", False, "recipe_undeclared"),
        # A mint's request keeps the default: it must cite a source, so it refuses.
        ("USD", None, "default_facet_undefined"),
    ],
)
def test_an_unsourced_top_facet_answers_as_the_request_requires_provenance(
    currency: str, requires_provenance: bool | None, expected: str
):
    facet = Facet(
        facet_id="f1",
        provenance_id=None,
        measurement_method="NotApplicable",
        unit="CurrencyUSD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2022",
    )
    request = _request(unit=FacetBinding(state="request", request=UnitRequest(currency, "current")))
    if requires_provenance is not None:
        request = dataclasses.replace(request, requires_provenance=requires_provenance)
    sdmx, requests = _router_sdmx({})

    try:
        selections = decide_member_unit(
            request, settle=True, graph=_graph([facet]), sdmx=sdmx, coverage=_SUBJECT_ONLY_COVERAGE
        )
    except RecipeNotDeclaredError as exc:
        outcome = exc.cause
    except DefaultFacetUndefinedError as exc:
        outcome = exc.code
        assert {issue.axis for issue in exc.issues or ()} == {"provenance"}
    else:
        assert selections["provenance"].ref is None
        assert selections["unit"].ref == "CurrencyUSD"
        outcome = selections["provenance"].selection_reason
    assert outcome == expected
    assert requests == []


def test_precomputed_candidates_decide_a_request_binding_no_entity_without_a_probe():
    # A decision fetches no rows, so precomputed candidates carry no place admission and a
    # request with no entity scope needs no default place scope read.
    facets = [
        _facet(facet_id="f1", unit="CurrencyUSD", obs_count=5),
        _facet(facet_id="f2", unit="CurrencyEUR", obs_count=50),
    ]
    graph = _graph([])
    request = _request(unit=FacetBinding(state="unbound"), role_bindings={})

    sdmx, sdmx_requests = _router_sdmx({})
    (selections,) = decide_group_unit(
        [request],
        [MemberProbe(candidates=tuple(facets), admission=CoverageAdmission())],
        settle=True,
        graph=graph,
        sdmx=sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
    )

    assert selections["unit"].ref == "CurrencyEUR"
    assert sdmx_requests == []
    for probe_call in (
        "child_places_batch",
        "variable_facets_batch",
        "observation_facets_batch_many",
    ):
        assert graph.calls[probe_call] == 0
