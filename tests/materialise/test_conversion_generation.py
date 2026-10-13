"""Conversion tests exercise public materialisation entry points with mock SDMX responses."""

from __future__ import annotations

import re

import httpx
import pytest
from dfo_frame_grammar.grammar import UnitRequest as FrameUnitRequest

from dc_sdmx import SdmxClient
from materialise import (
    ConcatenatedFrameChunks,
    MaterialisationRequest,
    materialise_group,
    materialise_preview,
    materialise_spec,
)
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.convert import merged_record
from materialise.enrichment import EnrichmentLookup
from materialise.errors import MaterialisationUsageError, OverCeilingError
from materialise.facets import FacetBinding, UnitRequest
from materialise.frame_build import Frame
from materialise.generation import _generate_group
from materialise.graph import Facet
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import (
    CONVERSION_NODE_PROPERTIES,
    RateCoverageGapError,
    RecipeNotDeclaredError,
)
from materialise.reshape import RollupNotSummableError, classify_columns, groupby_sum
from materialise.series_conversion import SeriesConversion
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._rate_table_support import RateTableServer, recipe_table_bodies
from .conftest import FakeGraphReader, only_chunk

_BASE_URL = "https://datacommons.one.org/core/api"

_ROLES = ("financingProvider", "financingRecipient")
_STAT_VAR = "dcid:Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_BARE_STAT_VAR = "Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_STAT_VAR_B = "dcid:Amount_FinancialFlow_DevelopmentAssistance_Nominal_B"
_BARE_STAT_VAR_B = "Amount_FinancialFlow_DevelopmentAssistance_Nominal_B"

_MEMBER_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_DAC_RECIPE = "ConversionRecipeDAC"
_TABLE_PROVENANCE = RECIPE_BODIES[_DAC_RECIPE].table_provenance
_EXCHANGE_VAR = RECIPE_BODIES[_DAC_RECIPE].exchange_rate_variable
_DEFLATOR_VAR = RECIPE_BODIES[_DAC_RECIPE].deflator_variable


_ROLE_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_SUBJECT_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _role_row(
    *,
    stat_var: str = _BARE_STAT_VAR,
    provider: str = "country/FRA",
    recipient: str = "country/KEN",
    year: str,
    value: str,
    unit: str = "CurrencyUSD",
    facet_id: str = "f1",
) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{provider},{recipient},{unit},"
        f"GrantEquivalent,NotApplicable,{_MEMBER_PROVENANCE},{year},{value},,{facet_id}\r\n"
    )


def _rate_row(*, stat_var: str, entity: str, year: str, value: str, facet_id: str = "f1") -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{entity},NotApplicable,NotApplicable,"
        f"P1Y,{_TABLE_PROVENANCE},{year},{value},,{facet_id}\r\n"
    )


def _router_sdmx(
    role_bodies: dict[str, str], subject_bodies: dict[str, str] | None = None
) -> tuple[SdmxClient, list[httpx.Request]]:
    """Serve role-bearing probe and fetch responses, and the rate tables in
    `subject_bodies` over `/availability` and `/data`. Comma-joined StatVars
    return combined rows under one header.
    """

    def role_handler(request: httpx.Request) -> httpx.Response:
        params = dict(httpx.QueryParams(request.url.query.decode()))
        names = params["c[variableMeasured]"].split(",")
        rows = "".join(role_bodies[name].split("\r\n", 1)[1] for name in names)
        return httpx.Response(
            200,
            text=_ROLE_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    server = RateTableServer(subject_bodies or {}, fallback=role_handler)
    return server.client(base_url=_BASE_URL), server.requests


def _base_arcs(
    *, bare_stat_vars: tuple[str, ...] = (_BARE_STAT_VAR,), provenance: str = _MEMBER_PROVENANCE
) -> dict[str, dict[str, tuple[str, ...]]]:
    statvar_arcs: dict[str, tuple[str, ...]] = {
        "observationProperties": _ROLES,
        "measurementQualifier": ("Nominal",),
        "populationType": ("FinancialFlow",),
    }
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        provenance: {"conversionRecipe": (_DAC_RECIPE,), "isPartOf": ()},
        _TABLE_PROVENANCE: {"lastDataRefreshDate": ("2026-01-01T00:00:00Z",)},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
    }
    for bare_stat_var in bare_stat_vars:
        arcs[bare_stat_var] = statvar_arcs
    return arcs


def _graph(arcs: dict[str, dict[str, tuple[str, ...]]] | None = None) -> FakeGraphReader:
    return FakeGraphReader(arcs=arcs if arcs is not None else _base_arcs())


def _request(
    *,
    stat_var: str = _STAT_VAR,
    unit: FacetBinding,
    window: RequestedWindow | None = None,
    role_bindings: dict[str, list[str]] | None = None,
) -> MaterialisationRequest:
    bindings = (
        role_bindings
        if role_bindings is not None
        else {"financingProvider": ["country/FRA"], "financingRecipient": ["country/KEN"]}
    )
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings=bindings,
        declared_roles=_ROLES,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref=_MEMBER_PROVENANCE),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": unit,
        },
        window=window if window is not None else RequestedWindow(kind="all"),
    )


def _coverage_for(
    entries: list[tuple[str, str]],
    *,
    donor: str = "country/FRA",
    facet_id: str = "f1",
    provenance: str = _MEMBER_PROVENANCE,
    method: str = "GrantEquivalent",
    unit: str = "CurrencyUSD",
    year: int = 2022,
) -> CoverageIndex:
    """Build coverage for each StatVar/recipient pair with `_role_row` metadata.
    The unbound provider value is a placeholder.
    """
    bare_stat_vars = {bare for bare, _ in entries}
    role_names = {
        bare: RoleNames(donor="financingProvider", recipient="financingRecipient")
        for bare in bare_stat_vars
    }
    facet_metadata = {
        facet_id: FacetMetadata(provenance_id=provenance, measurement_method=method, unit=unit)
    }
    rows = [(bare, facet_id, donor, recipient, [year]) for bare, recipient in entries]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _unit_cells(frame: Frame) -> list[object]:
    """Return `unit` cells from the grammar's row view, which unwraps the struct."""
    index = frame.columns.index("unit")
    return [row[index] for row in frame.rows]


# ---------------------------------------------------------------------------
# Uniform target: {EUR, current}
# ---------------------------------------------------------------------------


def test_member_converts_to_eur_current_and_unit_cells_are_the_struct():
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
        subject_bodies={
            _EXCHANGE_VAR: (
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
                + _rate_row(
                    stat_var=_EXCHANGE_VAR,
                    entity="CurrencyEUR",
                    year="2022",
                    value="0.9",
                )
            )
        },
    )

    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    frame = only_chunk(result)
    assert frame.table["value"].to_pylist() == [pytest.approx(100.0 * 0.9 / 1.0)]
    assert _unit_cells(frame) == [
        FrameUnitRequest(currency="EUR", prices="current", base_year=None)
    ]
    # One provenance per frame: no partition, no `recipes_conflict`.
    assert set(frame.table["provenance"].to_pylist()) == {_MEMBER_PROVENANCE}


def test_member_loads_both_tables_for_constant_prices():
    request = _request(
        unit=FacetBinding(state="request", request=UnitRequest("USD", "constant", base_year=2024))
    )
    sdmx, requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2024", value="100")},
        subject_bodies={
            _EXCHANGE_VAR: (
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2024", value="1.0")
            ),
            _DEFLATOR_VAR: (
                _SUBJECT_HEADER
                + _rate_row(
                    stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2024", value="100.0"
                )
            ),
        },
    )

    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    dispatched = {
        dict(httpx.QueryParams(r.url.query.decode()))["c[variableMeasured]"] for r in requests
    }
    assert _EXCHANGE_VAR in dispatched
    assert _DEFLATOR_VAR in dispatched
    assert _unit_cells(only_chunk(result)) == [
        FrameUnitRequest(currency="USD", prices="constant", base_year=2024)
    ]


def _constant_usd_member(
    *, years: tuple[str, ...], deflator_years: tuple[str, ...], ceiling: int
) -> int:
    """Mints one constant-USD member with a row per year in `years` and a France
    deflator for `deflator_years` only, and returns its served row count. A
    row whose year has no deflator is left out in conversion."""
    request = _request(
        unit=FacetBinding(state="request", request=UnitRequest("USD", "constant", base_year=2024))
    )
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER
            + "".join(_role_row(year=year, value="100") for year in years)
        },
        subject_bodies={
            _EXCHANGE_VAR: _SUBJECT_HEADER
            + "".join(
                _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year=year, value="1.0")
                for year in years
            ),
            _DEFLATOR_VAR: _SUBJECT_HEADER
            + "".join(
                _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year=year, value="100.0")
                for year in deflator_years
            ),
        },
    )
    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    mint_kwargs = {
        "sdmx": sdmx,
        "graph": graph,
        "materialiser_build": "test-build",
        "capability_version": "v1",
        "enrichment": EnrichmentLookup(graph, parties={}),
        "rate_cache": RateTableCache(),
        "ceiling": ceiling,
        "coverage": coverage,
    }
    return only_chunk(materialise_spec(request, **mint_kwargs)).table.num_rows


def test_the_ceiling_counts_rows_served_after_deflator_omissions() -> None:
    # Two rows arrive; 2023 has no deflator, so one row is served under a ceiling of one.
    served = _constant_usd_member(years=("2023", "2024"), deflator_years=("2024",), ceiling=1)

    assert served == 1


def test_served_rows_past_the_ceiling_after_deflator_omissions_still_refuse() -> None:
    # Three rows arrive and 2022 has no deflator, so two rows are served under a ceiling of one.
    with pytest.raises(OverCeilingError) as exc_info:
        _constant_usd_member(
            years=("2022", "2023", "2024"), deflator_years=("2023", "2024"), ceiling=1
        )

    assert (exc_info.value.count, exc_info.value.cap) == (2, 1)


def test_native_current_passes_values_through():
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("USD", "current")))
    sdmx, requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
    )

    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    frame = only_chunk(result)
    assert frame.table["value"].to_pylist() == [100.0]
    assert _unit_cells(frame) == [
        FrameUnitRequest(currency="USD", prices="current", base_year=None)
    ]
    dispatched_vars = {
        dict(httpx.QueryParams(r.url.query.decode())).get("c[variableMeasured]") for r in requests
    }
    # No rate/deflator table load at all: natively satisfied.
    assert dispatched_vars == {_BARE_STAT_VAR}


def test_ref_form_unit_binding_stays_unconverted():
    request = _request(unit=FacetBinding(state="bound", ref="CurrencyUSD"))
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
    )

    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        coverage=coverage,
    )

    # A ref-form (non-request) unit binding passes through unconverted: `unit`
    # stays a plain string ref, byte-identical to the pre-conversion tail.
    frame = only_chunk(result)
    assert _unit_cells(frame) == ["CurrencyUSD"]
    assert frame.table["value"].to_pylist() == [100.0]
    assert result.series_conversions == {}


# ---------------------------------------------------------------------------
# Refusal ahead of any data request
# ---------------------------------------------------------------------------


def test_recipe_not_declared_refuses_before_any_data_request():
    # Subject-only declared_roles (`("observationAbout",)`), so the facet
    # probe itself goes through `graph.observation_facets_batch` rather
    # than an SDMX `/data` call: this is the one shape where "no SDMX call
    # ran at all" is a meaningful, literal assertion.
    orphan_provenance = "provenance/Orphan"
    arcs = _base_arcs(provenance=orphan_provenance)
    arcs[orphan_provenance] = {"conversionRecipe": (), "isPartOf": ()}
    graph = FakeGraphReader(
        arcs=arcs,
        facets={
            (_BARE_STAT_VAR, "country/KEN"): [
                Facet(
                    facet_id="f1",
                    provenance_id=orphan_provenance,
                    measurement_method="GrantEquivalent",
                    unit="CurrencyUSD",
                    obs_count=1,
                    earliest_date="2022",
                    latest_date="2022",
                )
            ]
        },
    )
    sdmx, requests = _router_sdmx(role_bodies={})
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"observationAbout": ["country/KEN"]},
        declared_roles=("observationAbout",),
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref=orphan_provenance),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(state="request", request=UnitRequest("EUR", "current")),
        },
        window=RequestedWindow(kind="all"),
    )

    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties={}),
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "recipe_undeclared"
    assert requests == []


def test_member_with_request_unit_binding_and_no_rate_cache_raises_before_any_sdmx_request():
    graph = _subject_only_graph((_BARE_STAT_VAR,))
    sdmx, requests = _router_sdmx(role_bodies={})
    request = _subject_only_request(
        _STAT_VAR, unit=FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    )

    with pytest.raises(MaterialisationUsageError):
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties={}),
        )
    assert requests == []  # no SDMX call at all: refused before any fetch


# ---------------------------------------------------------------------------
# LATEST windows use the chosen year.
# ---------------------------------------------------------------------------


def test_latest_window_refuses_rate_coverage_gap_on_the_chosen_year_only():
    request = _request(
        unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")),
        window=RequestedWindow(kind="latest"),
    )
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: (
                _ROLE_HEADER
                + _role_row(year="2022", value="100")
                + _role_row(year="2023", value="120")
            )
        },
        subject_bodies={
            # The table covers 2022. The latest window chooses 2023.
            _EXCHANGE_VAR: (
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
                + _rate_row(
                    stat_var=_EXCHANGE_VAR,
                    entity="CurrencyEUR",
                    year="2022",
                    value="0.9",
                )
            )
        },
    )

    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    with pytest.raises(RateCoverageGapError) as exc_info:
        materialise_spec(
            request,
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties={}),
            rate_cache=RateTableCache(),
            coverage=coverage,
        )
    assert exc_info.value.cause == "rate_missing"
    # The latest window selects 2023, which has no exchange-rate row.
    assert any("@2023" in entity for entity in exc_info.value.entities)
    assert not any("@2022" in entity for entity in exc_info.value.entities)


# ---------------------------------------------------------------------------
# Collection: a chunk converts each of its own routed members
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Group and preview: `rate_cache=None` refuses before any fetch
# ---------------------------------------------------------------------------


def _subject_only_request(stat_var: str, *, unit: FacetBinding) -> MaterialisationRequest:
    """Subject-only requests use graph-based facet discovery, allowing tests
    to verify conversion refusals before any SDMX fetch.
    """
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"observationAbout": ["country/KEN"]},
        declared_roles=("observationAbout",),
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref=_MEMBER_PROVENANCE),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": unit,
        },
        window=RequestedWindow(kind="all"),
    )


def _subject_only_graph(bare_stat_vars: tuple[str, ...]) -> FakeGraphReader:
    facet = Facet(
        facet_id="f1",
        provenance_id=_MEMBER_PROVENANCE,
        measurement_method="GrantEquivalent",
        unit="CurrencyUSD",
        obs_count=1,
        earliest_date="2022",
        latest_date="2022",
    )
    return FakeGraphReader(
        facets={(bare_stat_var, "country/KEN"): [facet] for bare_stat_var in bare_stat_vars}
    )


def test_group_with_request_unit_binding_and_no_rate_cache_raises_before_any_sdmx_request():
    graph = _subject_only_graph((_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    sdmx, requests = _router_sdmx(role_bodies={})
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    members = [
        _subject_only_request(_STAT_VAR, unit=eur_request),
        _subject_only_request(_STAT_VAR_B, unit=eur_request),
    ]

    with pytest.raises(MaterialisationUsageError):
        materialise_group(
            members,
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties={}),
        )
    assert requests == []  # no SDMX call at all: refused before any fetch


def test_group_with_ref_unit_binding_stays_unconverted():
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    sdmx, requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100"),
            _BARE_STAT_VAR_B: (
                _ROLE_HEADER + _role_row(stat_var=_BARE_STAT_VAR_B, year="2022", value="200")
            ),
        },
    )
    ref_binding = FacetBinding(state="bound", ref="CurrencyUSD")
    members = [
        _request(stat_var=_STAT_VAR, unit=ref_binding),
        _request(stat_var=_STAT_VAR_B, unit=ref_binding),
    ]

    graph = _graph(arcs)
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])
    result = materialise_group(
        members,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        coverage=coverage,
    )

    # A ref-form (non-request) unit binding passes through unconverted, for a
    # group the same as for one member: `unit` stays a plain string ref, values
    # untouched.
    frame = only_chunk(result)
    values_by_stat_var = dict(
        zip(frame.table["stat_var"].to_pylist(), frame.table["value"].to_pylist(), strict=True)
    )
    assert values_by_stat_var == {_STAT_VAR: 100.0, _STAT_VAR_B: 200.0}
    assert set(_unit_cells(frame)) == {"CurrencyUSD"}
    assert requests

    # `_GroupOutcome` exposes per-member conversion records.
    # `MaterialisationResult` carries the group-level conversion.
    outcome = _generate_group(
        members,
        sdmx=sdmx,
        graph=graph,
        ceiling=DEFAULT_SYNC_CEILING_ROWS,
        member_concurrency=1,
        coverage_index=coverage,
    )
    assert outcome.member_conversion_records == {_STAT_VAR: None, _STAT_VAR_B: None}
    assert outcome.member_conversion_recipe_ids == {_STAT_VAR: None, _STAT_VAR_B: None}


def test_preview_with_request_unit_binding_and_no_rate_cache_raises_before_any_sdmx_request():
    graph = _subject_only_graph((_BARE_STAT_VAR,))
    sdmx, requests = _router_sdmx(role_bodies={})
    request = _subject_only_request(
        _STAT_VAR, unit=FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    )

    with pytest.raises(MaterialisationUsageError):
        materialise_preview(request, sdmx=sdmx, graph=graph, limit=10)
    assert requests == []  # no SDMX call at all: refused before any fetch


def test_preview_with_ref_unit_binding_stays_unconverted():
    request = _request(unit=FacetBinding(state="bound", ref="CurrencyUSD"))
    sdmx, requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
    )

    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_preview(request, sdmx=sdmx, graph=graph, coverage=coverage, limit=10)

    # A ref-form (non-request) unit binding passes through unconverted: `unit`
    # stays a plain string ref, byte-identical to the pre-conversion tail.
    assert _unit_cells(result.frame) == ["CurrencyUSD"]
    assert result.frame.table["value"].to_pylist() == [100.0]
    assert requests
    # A ref-form unit binding plans no conversion, so conversion_record and
    # conversion_recipe_id are both None.
    assert result.conversion_record is None
    assert result.conversion_recipe_id is None


# ---------------------------------------------------------------------------
# Group: converts a request-form unit binding
# ---------------------------------------------------------------------------


def _eur_exchange_rate_body(*, year: str = "2022") -> str:
    return (
        _SUBJECT_HEADER
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year=year, value="1.0")
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyEUR", year=year, value="0.9")
    )


def test_group_converts_both_members_to_eur_current_with_one_concat_schema():
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    sdmx, requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100"),
            _BARE_STAT_VAR_B: (
                _ROLE_HEADER + _role_row(stat_var=_BARE_STAT_VAR_B, year="2022", value="200")
            ),
        },
        subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()},
    )
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    members = [
        _request(stat_var=_STAT_VAR, unit=eur_request),
        _request(stat_var=_STAT_VAR_B, unit=eur_request),
    ]

    graph = _graph(arcs)
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])
    result = materialise_group(
        members,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    frame = only_chunk(result)
    values_by_stat_var = dict(
        zip(frame.table["stat_var"].to_pylist(), frame.table["value"].to_pylist(), strict=True)
    )
    assert values_by_stat_var == {
        _STAT_VAR: pytest.approx(100.0 * 0.9),
        _STAT_VAR_B: pytest.approx(200.0 * 0.9),
    }
    # The concatenated schema gives both `unit` columns the request struct.
    assert set(_unit_cells(frame)) == {
        FrameUnitRequest(currency="EUR", prices="current", base_year=None)
    }
    assert requests

    # `_GroupOutcome` records each member's recipe and table.
    # `MaterialisationResult` carries the group-level conversion.
    outcome = _generate_group(
        members,
        sdmx=sdmx,
        graph=graph,
        ceiling=DEFAULT_SYNC_CEILING_ROWS,
        member_concurrency=1,
        rate_cache=RateTableCache(),
        coverage_index=coverage,
    )
    assert outcome.member_conversion_recipe_ids == {
        _STAT_VAR: _DAC_RECIPE,
        _STAT_VAR_B: _DAC_RECIPE,
    }
    assert set(outcome.member_conversion_records) == {_STAT_VAR, _STAT_VAR_B}
    for record in outcome.member_conversion_records.values():
        assert record is not None
        assert record.recipe_id == _DAC_RECIPE
    # Each member's one series keeps its own record, keyed by bare StatVar and role tuple.
    kenya_from_france = (
        ("financingProvider", "country/FRA"),
        ("financingRecipient", "country/KEN"),
    )
    assert set(result.series_conversions) == {
        (_BARE_STAT_VAR, kenya_from_france),
        (_BARE_STAT_VAR_B, kenya_from_france),
    }


def test_group_with_an_empty_member_keeps_the_struct_unit_type():
    # `_STAT_VAR_B`'s own row is real (its facet probe -- always a LATEST
    # read, regardless of the member's own requested window -- finds a
    # genuine candidate, so facet selection succeeds), but its `"range"`
    # window excludes 2022 entirely, so its own per-member fetch survives
    # to zero rows: an empty member cleared facet selection but found no rows
    # after filtering.
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100"),
            _BARE_STAT_VAR_B: (
                _ROLE_HEADER + _role_row(stat_var=_BARE_STAT_VAR_B, year="2022", value="200")
            ),
        },
        subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()},
    )
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    members = [
        _request(stat_var=_STAT_VAR, unit=eur_request),
        _request(
            stat_var=_STAT_VAR_B,
            unit=eur_request,
            window=RequestedWindow(kind="range", start_year=2099, end_year=2099),
        ),
    ]

    graph = _graph(arcs)
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])
    result = materialise_group(
        members,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    # The empty member's zero-row table retains a struct-typed `unit` column,
    # matching the populated member for `pa.concat_tables`.
    frame = only_chunk(result)
    assert frame.table["stat_var"].to_pylist() == [_STAT_VAR]
    assert _unit_cells(frame) == [
        FrameUnitRequest(currency="EUR", prices="current", base_year=None)
    ]


# ---------------------------------------------------------------------------
# Group: groupby homogeneity over the group's own converted `unit` column
# ---------------------------------------------------------------------------


def _reshape_group_members(*, unit_request: UnitRequest) -> list[MaterialisationRequest]:
    """`_STAT_VAR` has Kenya and Britain rows to group by recipient.
    `_STAT_VAR_B` has one row because StatVar remains part of the grain.
    """
    unit = FacetBinding(state="request", request=unit_request)
    role_bindings_a = {
        "financingProvider": ["country/FRA"],
        "financingRecipient": ["country/KEN", "country/GBR"],
    }
    role_bindings_b = {
        "financingProvider": ["country/FRA"],
        "financingRecipient": ["country/KEN"],
    }
    return [
        _request(stat_var=_STAT_VAR, unit=unit, role_bindings=role_bindings_a),
        _request(stat_var=_STAT_VAR_B, unit=unit, role_bindings=role_bindings_b),
    ]


def _reshape_role_bodies() -> dict[str, str]:
    return {
        _BARE_STAT_VAR: (
            _ROLE_HEADER
            + _role_row(recipient="country/KEN", year="2022", value="100")
            + _role_row(recipient="country/GBR", year="2022", value="50")
        ),
        _BARE_STAT_VAR_B: (
            _ROLE_HEADER
            + _role_row(stat_var=_BARE_STAT_VAR_B, recipient="country/KEN", year="2022", value="10")
        ),
    }


def test_group_uniform_request_passes_groupby_homogeneity():
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    sdmx, _requests = _router_sdmx(
        role_bodies=_reshape_role_bodies(),
        subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()},
    )
    members = _reshape_group_members(unit_request=UnitRequest("EUR", "current"))

    graph = _graph(arcs)
    coverage = _coverage_for(
        [
            (_BARE_STAT_VAR, "country/KEN"),
            (_BARE_STAT_VAR, "country/GBR"),
            (_BARE_STAT_VAR_B, "country/KEN"),
        ]
    )
    result = materialise_group(
        members,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    # A uniform target gives an equal struct cell (`{EUR, current}`) on
    # every row regardless of which entity `financingRecipient` served, so
    # collapsing that axis passes the facet-homogeneity check.
    table = ConcatenatedFrameChunks.from_chunks(result.frame_chunks())
    classification = classify_columns(table.columns)
    grouped = groupby_sum(
        table,
        axis="financingRecipient",
        classification=classification,
        summable_axes=frozenset({"financingRecipient"}),
    )

    stat_var_pos = grouped.columns.index("stat_var")
    value_pos = grouped.columns.index("value")
    value_by_stat_var = {row[stat_var_pos]: row[value_pos] for row in grouped.rows}
    assert value_by_stat_var == {
        _STAT_VAR: pytest.approx((100.0 + 50.0) * 0.9),
        _STAT_VAR_B: pytest.approx(10.0 * 0.9),
    }


def test_group_role_shaped_request_refuses_groupby_homogeneity():
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    arcs["CurrencyKES"] = {"typeOf": ("CurrencyUnit",)}
    arcs["CurrencyGBP"] = {"typeOf": ("CurrencyUnit",)}
    arcs["CurrencyKES"] = {**arcs["CurrencyKES"], "nationalCurrencyOf": ("country/KEN",)}
    arcs["CurrencyGBP"] = {**arcs["CurrencyGBP"], "nationalCurrencyOf": ("country/GBR",)}
    sdmx, _requests = _router_sdmx(
        role_bodies=_reshape_role_bodies(),
        subject_bodies={
            _EXCHANGE_VAR: (
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
                + _rate_row(
                    stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2022", value="130.0"
                )
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyGBP", year="2022", value="0.8")
            )
        },
    )
    members = _reshape_group_members(unit_request=UnitRequest("destination", "current"))

    graph = FakeGraphReader(arcs=arcs)
    coverage = _coverage_for(
        [
            (_BARE_STAT_VAR, "country/KEN"),
            (_BARE_STAT_VAR, "country/GBR"),
            (_BARE_STAT_VAR_B, "country/KEN"),
        ]
    )
    result = materialise_group(
        members,
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    # Destination currencies differ (KES for Kenya, GBP for Britain) while
    # other grain values match. Grouping recipients raises `facet_homogeneity`.
    table = ConcatenatedFrameChunks.from_chunks(result.frame_chunks())
    classification = classify_columns(table.columns)
    with pytest.raises(RollupNotSummableError) as excinfo:
        groupby_sum(
            table,
            axis="financingRecipient",
            classification=classification,
            summable_axes=frozenset({"financingRecipient"}),
        )
    assert excinfo.value.precondition == "facet_homogeneity"


# ---------------------------------------------------------------------------
# Group: a member's own conversion refusal preempts the pooled ceiling check
# ---------------------------------------------------------------------------


def test_group_rate_coverage_gap_preempts_the_pooled_ceiling_check():
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: (
                _ROLE_HEADER
                + _role_row(year="2022", value="100")
                + _role_row(year="2023", value="120")
            ),
            _BARE_STAT_VAR_B: (
                _ROLE_HEADER
                + _role_row(stat_var=_BARE_STAT_VAR_B, year="2022", value="200")
                + _role_row(stat_var=_BARE_STAT_VAR_B, year="2023", value="220")
            ),
        },
        subject_bodies={
            # Covers 2022. Both latest windows choose 2023.
            _EXCHANGE_VAR: _eur_exchange_rate_body(year="2022")
        },
    )
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    members = [
        _request(stat_var=_STAT_VAR, unit=eur_request, window=RequestedWindow(kind="latest")),
        _request(stat_var=_STAT_VAR_B, unit=eur_request, window=RequestedWindow(kind="latest")),
    ]

    graph = _graph(arcs)
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])
    # With ceiling 1, both latest rows total 2. The missing 2023 rate is
    # checked per member before the pooled ceiling.
    with pytest.raises(RateCoverageGapError) as exc_info:
        materialise_group(
            members,
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties={}),
            rate_cache=RateTableCache(),
            ceiling=1,
            coverage=coverage,
        )
    assert exc_info.value.cause == "rate_missing"
    assert any("@2023" in entity for entity in exc_info.value.entities)


def test_group_role_shaped_rate_coverage_gap_preempts_the_pooled_ceiling_check():
    # Destination rates resolve per served entity after fetch. Kenya lacks
    # a 2023 rate, so that gap wins before the pooled ceiling (2 rows over 1).
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    arcs["CurrencyKES"] = {"typeOf": ("CurrencyUnit",)}
    arcs["CurrencyGBP"] = {"typeOf": ("CurrencyUnit",)}
    arcs["CurrencyKES"] = {**arcs["CurrencyKES"], "nationalCurrencyOf": ("country/KEN",)}
    arcs["CurrencyGBP"] = {**arcs["CurrencyGBP"], "nationalCurrencyOf": ("country/GBR",)}
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: (
                _ROLE_HEADER
                + _role_row(recipient="country/KEN", year="2022", value="100")
                + _role_row(recipient="country/KEN", year="2023", value="110")
            ),
            _BARE_STAT_VAR_B: (
                _ROLE_HEADER
                + _role_row(
                    stat_var=_BARE_STAT_VAR_B, recipient="country/GBR", year="2022", value="200"
                )
                + _role_row(
                    stat_var=_BARE_STAT_VAR_B, recipient="country/GBR", year="2023", value="220"
                )
            ),
        },
        subject_bodies={
            _EXCHANGE_VAR: (
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2023", value="1.0")
                + _rate_row(
                    stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2022", value="130.0"
                )
                # No 2023 row for country/KEN: the LATEST window's own
                # chosen year for _STAT_VAR has no rate.
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyGBP", year="2022", value="0.8")
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyGBP", year="2023", value="0.78")
            )
        },
    )
    destination_request = FacetBinding(
        state="request", request=UnitRequest("destination", "current")
    )
    members = [
        _request(
            stat_var=_STAT_VAR,
            unit=destination_request,
            role_bindings={
                "financingProvider": ["country/FRA"],
                "financingRecipient": ["country/KEN"],
            },
            window=RequestedWindow(kind="latest"),
        ),
        _request(
            stat_var=_STAT_VAR_B,
            unit=destination_request,
            role_bindings={
                "financingProvider": ["country/FRA"],
                "financingRecipient": ["country/GBR"],
            },
            window=RequestedWindow(kind="latest"),
        ),
    ]

    graph = FakeGraphReader(arcs=arcs)
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/GBR")])
    with pytest.raises(RateCoverageGapError) as exc_info:
        materialise_group(
            members,
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties={}),
            rate_cache=RateTableCache(),
            ceiling=1,
            coverage=coverage,
        )
    assert exc_info.value.cause == "rate_missing"
    assert any("@2023" in entity for entity in exc_info.value.entities)


# ---------------------------------------------------------------------------
# Preview: converts before truncation
# ---------------------------------------------------------------------------


def test_preview_converts_before_truncation():
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))
    sdmx, requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: (
                _ROLE_HEADER
                + _role_row(year="2020", value="100")
                + _role_row(year="2021", value="110")
                + _role_row(year="2022", value="120")
            )
        },
        subject_bodies={
            _EXCHANGE_VAR: (
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2020", value="1.0")
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2021", value="1.0")
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
                + _rate_row(
                    stat_var=_EXCHANGE_VAR,
                    entity="CurrencyEUR",
                    year="2020",
                    value="0.85",
                )
                + _rate_row(
                    stat_var=_EXCHANGE_VAR,
                    entity="CurrencyEUR",
                    year="2021",
                    value="0.9",
                )
                + _rate_row(
                    stat_var=_EXCHANGE_VAR,
                    entity="CurrencyEUR",
                    year="2022",
                    value="0.95",
                )
            )
        },
    )

    graph = _graph()
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_preview(
        request, sdmx=sdmx, graph=graph, coverage=coverage, limit=2, rate_cache=RateTableCache()
    )

    # `total_rows` counts all survivors (3), regardless of `limit`.
    assert result.total_rows == 3
    assert result.truncated is True
    # Truncation keeps the 2 rows with the greatest TIME_PERIOD (2021,
    # 2022), already converted: a preview shows only converted values for a
    # request-form unit binding, even for rows that end up truncated.
    delivered = dict(
        zip(
            result.frame.table["time_period"].to_pylist(),
            result.frame.table["value"].to_pylist(),
            strict=True,
        )
    )
    assert delivered == {
        "2021": pytest.approx(110.0 * 0.9),
        "2022": pytest.approx(120.0 * 0.95),
    }
    assert set(_unit_cells(result.frame)) == {
        FrameUnitRequest(currency="EUR", prices="current", base_year=None)
    }
    assert requests
    # The preview's own conversion record and recipe id reach `PreviewResult`,
    # so a caller such as the Search preview can learn which recipe and
    # table versions produced the converted points.
    assert result.conversion_recipe_id == _DAC_RECIPE
    assert result.conversion_record is not None
    assert result.conversion_record.recipe_id == _DAC_RECIPE
    # Each table is versioned by a SHA-256 digest of its rows.
    assert re.fullmatch(r"[0-9a-f]{64}", result.conversion_record.exchange_rate_table_version)
    assert result.conversion_record.native_currency == "CurrencyUSD"


def _conversion_reads(graph: FakeGraphReader) -> int:
    """The graph reads a request made, less the rate-table version guard's
    `lastDataRefreshDate` reads, which stay fresh on purpose."""
    return sum(
        1
        for _dcids, properties in graph.node_arcs_many_calls
        if properties != ("lastDataRefreshDate",)
    )


def _decision_reads(graph: FakeGraphReader) -> list[tuple[str, ...]]:
    """The dcids of each conversion decision's node read."""
    return [
        dcids
        for dcids, properties in graph.node_arcs_many_calls
        if properties == CONVERSION_NODE_PROPERTIES
    ]


@pytest.mark.parametrize(
    ("unit_request", "native_unit", "reads_provenance"),
    [
        pytest.param(UnitRequest("EUR", "current"), "CurrencyUSD", True, id="eur-current"),
        pytest.param(
            UnitRequest("USD", "constant", base_year=2024), "CurrencyUSD", True, id="usd-2024"
        ),
        pytest.param(UnitRequest("USD", "current"), "CurrencyUSD", False, id="usd-current-native"),
        pytest.param(UnitRequest("EUR", "current"), "USDollar", True, id="eur-from-usdollar"),
    ],
)
def test_a_converted_preview_reads_the_graph_once_more_than_an_unconverted_one(
    unit_request: UnitRequest, native_unit: str, reads_provenance: bool
):
    """A single-member preview's conversion decision reads its StatVar, provenance, and
    both currencies in one graph read. A natively satisfied request reads no provenance.
    A `USDollar` native unit is read as `CurrencyUSD`."""
    rate_body = _SUBJECT_HEADER + "".join(
        _rate_row(stat_var=_EXCHANGE_VAR, entity=currency, year="2024", value=value)
        for currency, value in (("CurrencyUSD", "1.0"), ("CurrencyEUR", "0.9"))
    )
    deflator_body = _SUBJECT_HEADER + _rate_row(
        stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2024", value="100.0"
    )

    def preview_graph(unit: FacetBinding) -> FakeGraphReader:
        sdmx, _requests = _router_sdmx(
            role_bodies={
                _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2024", value="100", unit=native_unit)
            },
            subject_bodies={_EXCHANGE_VAR: rate_body, _DEFLATOR_VAR: deflator_body},
        )
        graph = _graph()
        result = materialise_preview(
            _request(unit=unit),
            sdmx=sdmx,
            graph=graph,
            coverage=_coverage_for([(_BARE_STAT_VAR, "country/KEN")], unit=native_unit, year=2024),
            limit=10,
            rate_cache=RateTableCache(),
        )
        assert result.total_rows == 1
        return graph

    converted = preview_graph(FacetBinding(state="request", request=unit_request))
    unconverted = preview_graph(FacetBinding(state="unbound"))

    assert _conversion_reads(converted) - _conversion_reads(unconverted) == 1
    [decision_read] = _decision_reads(converted)
    assert (_MEMBER_PROVENANCE in decision_read) is reads_provenance


def test_a_converted_group_sharing_one_provenance_reads_the_graph_once_more_than_an_unconverted_one():
    """Two members sharing a provenance cost one conversion read together. The first
    decision's read also fetches the second member's StatVar."""
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    role_bodies = {
        _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100"),
        _BARE_STAT_VAR_B: _ROLE_HEADER
        + _role_row(stat_var=_BARE_STAT_VAR_B, year="2022", value="200"),
    }
    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])

    def group_graph(unit: FacetBinding) -> FakeGraphReader:
        sdmx, _requests = _router_sdmx(
            role_bodies=role_bodies, subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()}
        )
        graph = _graph(arcs)
        materialise_group(
            [_request(stat_var=_STAT_VAR, unit=unit), _request(stat_var=_STAT_VAR_B, unit=unit)],
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties={}),
            rate_cache=RateTableCache(),
            coverage=coverage,
        )
        return graph

    converted = group_graph(FacetBinding(state="request", request=UnitRequest("EUR", "current")))
    unconverted = group_graph(FacetBinding(state="unbound"))

    assert _conversion_reads(converted) - _conversion_reads(unconverted) == 1


# ---------------------------------------------------------------------------
# One table version per request
# ---------------------------------------------------------------------------


class _RefreshingGraphReader(FakeGraphReader):
    """Moves the table provenance's `lastDataRefreshDate` after every completed table load:
    each pair of reads (the version guard's before and after) sees one date, and the next pair
    sees a later one. A request that loaded the table once per member would admit two versions.
    """

    def __init__(self, arcs: dict[str, dict[str, tuple[str, ...]]]) -> None:
        super().__init__(arcs=arcs)
        self.refresh_reads = 0

    def node_arcs_many(
        self,
        dcids,
        *,
        properties,
        individual_probe_budget=None,
        deadline=None,
    ):
        if tuple(properties) == ("lastDataRefreshDate",) and tuple(dcids) == (_TABLE_PROVENANCE,):
            month = 1 + self.refresh_reads // 2
            self.refresh_reads += 1
            return {_TABLE_PROVENANCE: {"lastDataRefreshDate": (f"2026-{month:02d}-01",)}}
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


def _refreshing_fixture() -> tuple[SdmxClient, list[httpx.Request], _RefreshingGraphReader]:
    """Two members' rows, and an exchange-rate table whose second load would serve a moved EUR
    rate (0.8 in place of 0.9)."""
    sdmx, requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100"),
            _BARE_STAT_VAR_B: (
                _ROLE_HEADER + _role_row(stat_var=_BARE_STAT_VAR_B, year="2022", value="200")
            ),
        },
        subject_bodies={
            _EXCHANGE_VAR: [
                _eur_exchange_rate_body(),
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyEUR", year="2022", value="0.8"),
            ]
        },
    )
    graph = _RefreshingGraphReader(_base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B)))
    return sdmx, requests, graph


def _exchange_data_requests(requests: list[httpx.Request]) -> list[httpx.Request]:
    return [
        request
        for request in requests
        if "/data/" in request.url.path
        and dict(httpx.QueryParams(request.url.query.decode())).get("c[variableMeasured]")
        == _EXCHANGE_VAR
    ]


def test_group_members_share_one_table_when_the_refresh_date_moves_between_loads():
    """Without one table per request the second member would cite a second table version and
    the group's finish would refuse with a 500."""
    sdmx, requests, graph = _refreshing_fixture()
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))

    result = materialise_group(
        [
            _request(stat_var=_STAT_VAR, unit=eur_request),
            _request(stat_var=_STAT_VAR_B, unit=eur_request),
        ],
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=_coverage_for(
            [(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")]
        ),
    )

    frame = only_chunk(result)
    assert sorted(frame.table["value"].to_pylist()) == [pytest.approx(90.0), pytest.approx(180.0)]
    records = list(result.member_conversion_records.values())
    assert len(records) == 2
    assert {record.table_refresh_date for record in records if record} == {"2026-01-01"}
    assert len({record.exchange_rate_table_version for record in records if record}) == 1
    assert len(_exchange_data_requests(requests)) == 1
    assert graph.refresh_reads == 2


# ---------------------------------------------------------------------------
# A preview that leaves a role open and reads each of its dcids as one series.
# ---------------------------------------------------------------------------

_OPEN_PROVIDER_YEARS = {"country/FRA": ("2020", "2021", "2022"), "country/DEU": ("2019", "2020")}


_EUR_CURRENT = UnitRequest("EUR", "current")
_RATE_YEARS = (2019, 2020, 2021, 2022)


def _open_provider_sources() -> tuple[SdmxClient, CoverageIndex]:
    """Kenya's rows from France and Germany, each provider ending in its own year, with
    the DAC recipe's tables. The deflator table holds France, projected in
    2022, and the DAC total, so Germany's rows fall back to the total and every French row
    reads a projected base-year deflator."""
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER
            + "".join(
                _role_row(provider=provider, year=year, value="100")
                for provider, years in _OPEN_PROVIDER_YEARS.items()
                for year in years
            )
        },
        subject_bodies=recipe_table_bodies(
            _DAC_RECIPE,
            rates={
                "CurrencyUSD": dict.fromkeys(_RATE_YEARS, 1.0),
                "CurrencyEUR": dict.fromkeys(_RATE_YEARS, 0.9),
            },
            deflators={
                "country/FRA": dict.fromkeys(_RATE_YEARS, 100.0),
                "DACCountriesTotal": dict.fromkeys(_RATE_YEARS, 100.0),
            },
            projected=[("country/FRA", 2022)],
        ),
    )
    coverage = build_coverage_index(
        [
            (_BARE_STAT_VAR, "f1", provider, "country/KEN", [int(year) for year in years])
            for provider, years in _OPEN_PROVIDER_YEARS.items()
        ],
        role_names={
            _BARE_STAT_VAR: RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
        facet_metadata={
            "f1": FacetMetadata(
                provenance_id=_MEMBER_PROVENANCE,
                measurement_method="GrantEquivalent",
                unit="CurrencyUSD",
            )
        },
    )
    return sdmx, coverage


def _open_provider_request(unit: UnitRequest, window: RequestedWindow) -> MaterialisationRequest:
    return _request(
        unit=FacetBinding(state="request", request=unit),
        window=window,
        role_bindings={"financingRecipient": ["country/KEN"]},
    )


def _open_provider_preview(
    window: RequestedWindow,
    *,
    series_roles: tuple[str, ...],
    limit: int = 100,
    series_limit: int | None = 100,
    unit: UnitRequest = _EUR_CURRENT,
):
    """`_open_provider_sources` read with the provider open."""
    sdmx, coverage = _open_provider_sources()
    return materialise_preview(
        _open_provider_request(unit, window),
        sdmx=sdmx,
        graph=_graph(),
        coverage=coverage,
        limit=limit,
        rate_cache=RateTableCache(),
        series_roles=series_roles,
        series_limit=series_limit,
    )


def _periods_by_provider(frame: Frame) -> dict[str, list[str]]:
    provider = frame.columns.index("financingProvider")
    period = frame.columns.index("time_period")
    periods: dict[str, list[str]] = {}
    for row in frame.rows:
        periods.setdefault(str(row[provider]), []).append(str(row[period]))
    return {key: sorted(values) for key, values in periods.items()}


@pytest.mark.parametrize("kind", ["all", "latest"])
def test_a_preview_with_series_roles_records_and_dates_each_open_role_series(kind: str) -> None:
    """A preview naming the open provider in `series_roles` converts each provider's rows on
    their own: one record per provider, whose sum is the preview's record, and under
    `latest` each provider's own latest period. Its converted values are the ones the same
    read serves with no `series_roles`."""
    window = RequestedWindow(kind=kind)
    split = _open_provider_preview(window, series_roles=("financingProvider",))
    whole = _open_provider_preview(window, series_roles=())

    assert set(split.series_conversion_records) == {
        (provider,) for provider in _OPEN_PROVIDER_YEARS
    }
    assert whole.series_conversion_records == {}
    assert split.conversion_record == merged_record(list(split.series_conversion_records.values()))
    expected = {
        provider: list(years) if kind == "all" else [years[-1]]
        for provider, years in _OPEN_PROVIDER_YEARS.items()
    }
    assert _periods_by_provider(split.frame) == expected
    if kind == "all":
        assert sorted(split.frame.rows, key=repr) == sorted(whole.frame.rows, key=repr)
        assert split.conversion_record == whole.conversion_record
    else:
        # A mint keeps the member's one latest period across every provider.
        assert _periods_by_provider(whole.frame) == {"country/FRA": ["2022"]}


@pytest.mark.parametrize(
    "unit", [_EUR_CURRENT, UnitRequest("USD", "constant", base_year=2022)], ids=repr
)
def test_a_converted_mint_stores_the_records_a_series_roles_preview_gives(
    unit: UnitRequest,
) -> None:
    """A mint with the provider open records each series as the preview reading that
    provider in `series_roles` does. At constant prices Germany's two fallback rows and
    France's three projected rows stay with their own series. The member's record is the
    series' records merged."""
    window = RequestedWindow(kind="all")
    preview = _open_provider_preview(window, series_roles=("financingProvider",), unit=unit)
    sdmx, coverage = _open_provider_sources()
    graph = _graph()
    mint = materialise_spec(
        _open_provider_request(unit, window),
        sdmx=sdmx,
        graph=graph,
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties={}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    assert mint.series_conversions == {
        (
            _BARE_STAT_VAR,
            (("financingProvider", provider), ("financingRecipient", "country/KEN")),
        ): SeriesConversion.from_record(record)
        for (provider,), record in preview.series_conversion_records.items()
    }
    assert mint.conversion_record == merged_record(list(preview.series_conversion_records.values()))
    counts = {
        roles[0][1]: (record.fallback_count, record.projected_count)
        for (_stat_var, roles), record in mint.series_conversions.items()
    }
    if unit.prices == "constant":
        assert counts == {"country/DEU": (2, 0), "country/FRA": (0, 3)}
        assert mint.series_conversions[
            (
                _BARE_STAT_VAR,
                (("financingProvider", "country/DEU"), ("financingRecipient", "country/KEN")),
            )
        ].fallback_examples == (("country/DEU", "DACCountriesTotal"),)
    else:
        assert counts == {"country/DEU": (0, 0), "country/FRA": (0, 0)}


def test_a_preview_with_series_roles_keeps_series_limit_rows_of_each_series() -> None:
    """France holds three rows and Germany two. Under a series limit of two each provider
    keeps its two most recent rows and its whole count, however few rows `limit` allows the
    read, where the same read with no `series_roles` keeps the four most recent rows across
    both and drops Germany's 2019."""
    window = RequestedWindow(kind="all")
    split = _open_provider_preview(
        window, series_roles=("financingProvider",), limit=2, series_limit=2
    )
    whole = _open_provider_preview(window, series_roles=(), limit=4)

    assert _periods_by_provider(split.frame) == {
        "country/FRA": ["2021", "2022"],
        "country/DEU": ["2019", "2020"],
    }
    assert split.series_total_rows == {("country/FRA",): 3, ("country/DEU",): 2}
    assert (split.total_rows, split.truncated) == (5, True)
    assert _periods_by_provider(whole.frame) == {
        "country/FRA": ["2020", "2021", "2022"],
        "country/DEU": ["2020"],
    }
    assert whole.series_total_rows == {}
    assert (whole.total_rows, whole.truncated) == (5, True)


def test_a_preview_with_series_roles_is_whole_when_every_series_fits_its_limit() -> None:
    split = _open_provider_preview(
        RequestedWindow(kind="all"), series_roles=("financingProvider",), series_limit=3
    )

    assert _periods_by_provider(split.frame) == {
        provider: list(years) for provider, years in _OPEN_PROVIDER_YEARS.items()
    }
    assert split.series_total_rows == {("country/FRA",): 3, ("country/DEU",): 2}
    assert (split.total_rows, split.truncated) == (5, False)


@pytest.mark.parametrize("series_limit", [None, 0])
def test_a_preview_with_series_roles_requires_a_series_limit(series_limit: int | None) -> None:
    with pytest.raises(MaterialisationUsageError, match="series_limit"):
        _open_provider_preview(
            RequestedWindow(kind="all"),
            series_roles=("financingProvider",),
            series_limit=series_limit,
        )
