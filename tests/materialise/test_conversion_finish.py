"""Public materialisation calls test conversion traces and projected-rate warnings."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
import pytest
from dfo_frame_grammar import canonical_frame_bytes

from dc_sdmx import SdmxClient
from materialise import (
    elected_latest_year,
    materialise_group,
    materialise_preview,
    materialise_spec,
)
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.content_version import recompute_version
from materialise.convert import ConversionRecord
from materialise.enrichment import EnrichmentLookup
from materialise.errors import MaterialisationUsageError, OverCeilingError
from materialise.facets import FacetBinding, UnitRequest
from materialise.finish import (
    _conversion_trace,
    _sorted_converted_members,
)
from materialise.pipeline import (
    MaterialisationRequest,
    MaterialisationResult,
    PreviewResult,
)
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import RateCoverageGapError
from materialise.window import LatestYear, RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._rate_table_support import RateTableServer
from .conftest import FakeGraphReader, only_chunk

_BASE_URL = "https://datacommons.one.org/core/api"

_ROLES = ("financingProvider", "financingRecipient")
_STAT_VAR = "dcid:Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_BARE_STAT_VAR = "Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_STAT_VAR_B = "dcid:Amount_FinancialFlow_DevelopmentAssistance_Nominal_B"
_BARE_STAT_VAR_B = "Amount_FinancialFlow_DevelopmentAssistance_Nominal_B"

_MEMBER_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_DAC_RECIPE = "ConversionRecipeDAC"
_IDS_RECIPE = "ConversionRecipeIDS"
_TABLE_PROVENANCE = RECIPE_BODIES[_DAC_RECIPE].table_provenance
_EXCHANGE_VAR = RECIPE_BODIES[_DAC_RECIPE].exchange_rate_variable
_DEFLATOR_VAR = RECIPE_BODIES[_DAC_RECIPE].deflator_variable
_REFRESH_DATE = "2026-01-01T00:00:00Z"


class _Sha256Hex:
    """Equal to any SHA-256 hex digest: each table's version is a digest of its rows."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and re.fullmatch(r"[0-9a-f]{64}", other) is not None

    def __repr__(self) -> str:
        return "<sha256 hex digest>"


_TABLE_VERSION = _Sha256Hex()


_ROLE_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_SUBJECT_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
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
    """Build coverage for each StatVar/recipient pair with matching facet
    metadata and `CurrencyUSD` as the native unit.
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


def _role_row(
    *,
    stat_var: str = _BARE_STAT_VAR,
    provider: str = "country/FRA",
    recipient: str = "country/KEN",
    year: str,
    value: str,
    unit: str = "CurrencyUSD",
    facet_id: str = "f1",
    provenance: str = _MEMBER_PROVENANCE,
) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{provider},{recipient},{unit},"
        f"GrantEquivalent,NotApplicable,{provenance},{year},{value},,{facet_id}\r\n"
    )


def _rate_row(
    *,
    stat_var: str,
    entity: str,
    year: str,
    value: str,
    measurement_method: str = "NotApplicable",
    facet_id: str = "f1",
) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{entity},NotApplicable,{measurement_method},"
        f"P1Y,{_TABLE_PROVENANCE},{year},{value},,{facet_id}\r\n"
    )


def _router_sdmx(
    role_bodies: dict[str, str], subject_bodies: dict[str, str] | None = None
) -> tuple[SdmxClient, list[httpx.Request]]:
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
        _TABLE_PROVENANCE: {"lastDataRefreshDate": (_REFRESH_DATE,)},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
        # A second native currency, so a test can hold the exchange rate (and
        # therefore the converted value) fixed while moving only which
        # native unit dcid the member's own facet resolved to.
        "CurrencyGBP": {"typeOf": ("CurrencyUnit",)},
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
    provenance: str = _MEMBER_PROVENANCE,
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
            "provenance": FacetBinding(state="bound", ref=provenance),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": unit,
        },
        window=window if window is not None else RequestedWindow(kind="all"),
    )


def _eur_exchange_rate_body(
    *, year: str = "2022", eu_measurement_method: str = "NotApplicable"
) -> str:
    return (
        _SUBJECT_HEADER
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year=year, value="1.0")
        + _rate_row(
            stat_var=_EXCHANGE_VAR,
            entity="CurrencyEUR",
            year=year,
            value="0.9",
            measurement_method=eu_measurement_method,
        )
    )


# ---------------------------------------------------------------------------
# Member: trace conversion shape, and content_version tracks the native unit
# ---------------------------------------------------------------------------


def test_member_converted_trace_and_content_version_shape():
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
        subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()},
    )

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    assert result.trace.conversion == (
        {
            "recipe": _DAC_RECIPE,
            "rate_table_version": _TABLE_VERSION,
            "members": [{"stat_var": _BARE_STAT_VAR, "native_unit": "CurrencyUSD"}],
            "fallback": {"count": 0, "examples": [], "truncated": False},
            "projected": {"count": 0, "examples": [], "truncated": False},
        },
    )
    assert result.content_version.startswith("cv_")
    assert result.conversion_record is not None
    assert result.conversion_record.table_provenance_id == _TABLE_PROVENANCE
    assert result.member_conversion_records == {_STAT_VAR: result.conversion_record}


def test_content_version_stays_when_only_the_native_unit_dcid_changes():
    """Two runs whose converted rows are identical are one frame and one version. The trace's
    conversion entry names each run's own native unit."""
    request_usd = _request(
        unit=FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    )
    sdmx_usd, _ = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100", unit="CurrencyUSD")
        },
        subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()},
    )
    coverage_usd = _coverage_for([(_BARE_STAT_VAR, "country/KEN")], unit="CurrencyUSD")
    result_usd = materialise_spec(
        request_usd,
        sdmx=sdmx_usd,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage_usd,
    )

    # `CurrencyGBP` carries the identical 1.0 exchange rate `CurrencyUSD`
    # does, so the converted value below is byte-identical to the USD run's
    # -- only the native unit dcid the member's own facet resolved to moves.
    gbp_body = (
        _SUBJECT_HEADER
        # `_validate_table` requires a `CurrencyUSD` anchor row on every
        # exchange-rate table, whether or not USD is this member's own
        # native or target currency.
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyGBP", year="2022", value="1.0")
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyEUR", year="2022", value="0.9")
    )
    request_gbp = _request(
        unit=FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    )
    sdmx_gbp, _ = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100", unit="CurrencyGBP")
        },
        subject_bodies={_EXCHANGE_VAR: gbp_body},
    )
    coverage_gbp = _coverage_for([(_BARE_STAT_VAR, "country/KEN")], unit="CurrencyGBP")
    result_gbp = materialise_spec(
        request_gbp,
        sdmx=sdmx_gbp,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage_gbp,
    )

    assert (
        only_chunk(result_usd).table["value"].to_pylist()
        == only_chunk(result_gbp).table["value"].to_pylist()
    )
    (usd_entry,) = result_usd.trace.conversion
    (gbp_entry,) = result_gbp.trace.conversion
    assert usd_entry["members"] == [{"stat_var": _BARE_STAT_VAR, "native_unit": "CurrencyUSD"}]
    assert gbp_entry["members"] == [{"stat_var": _BARE_STAT_VAR, "native_unit": "CurrencyGBP"}]
    assert result_usd.content_version == result_gbp.content_version


def test_ref_form_spec_carries_no_conversion():
    request = _request(unit=FacetBinding(state="bound", ref="CurrencyUSD"))
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
    )

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        coverage=coverage,
    )

    assert result.trace.conversion is None
    assert "conversion" not in result.trace.to_dict()
    assert not any(code["code"] == "deflator_projected" for code in result.warnings)
    assert result.conversion_record is None
    assert result.member_conversion_records == {_STAT_VAR: None}


# ---------------------------------------------------------------------------
# Member: constant prices, deflator table version and a fallback marker
# ---------------------------------------------------------------------------


def _constant_price_member() -> MaterialisationResult:
    """A USD constant-2024 member whose deflator role entity falls back to the recipe's
    fallback entity."""
    request = _request(
        unit=FacetBinding(state="request", request=UnitRequest("USD", "constant", base_year=2024))
    )
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2024", value="100")},
        subject_bodies={
            _EXCHANGE_VAR: (
                _SUBJECT_HEADER
                + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2024", value="1.0")
            ),
            # `country/FRA` (the deflator role's own entity, the request's
            # own `financingProvider` binding) carries no row here, only
            # the recipe's own fallback entity does, so this member falls
            # back for it.
            _DEFLATOR_VAR: (
                _SUBJECT_HEADER
                + _rate_row(
                    stat_var=_DEFLATOR_VAR, entity="DACCountriesTotal", year="2024", value="100.0"
                )
            ),
        },
    )

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")], year=2024)
    return materialise_spec(
        request,
        sdmx=sdmx,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )


def test_constant_price_trace_carries_deflator_version_and_fallback_marker():
    result = _constant_price_member()

    (conversion,) = result.trace.conversion
    assert conversion["deflator_table_version"] == _TABLE_VERSION
    assert conversion["fallback"] == {
        "count": 1,
        "examples": [["country/FRA", "DACCountriesTotal"]],
        "truncated": False,
    }


# ---------------------------------------------------------------------------
# Group: one conversion, one entry per converted member
# ---------------------------------------------------------------------------


def test_group_with_two_converted_members_carries_one_conversion_entry_naming_both():
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
        _request(stat_var=_STAT_VAR_B, unit=eur_request),
    ]

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])
    result = materialise_group(
        members,
        sdmx=sdmx,
        graph=_graph(arcs),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(arcs), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    assert result.trace.conversion == (
        {
            "recipe": _DAC_RECIPE,
            "rate_table_version": _TABLE_VERSION,
            "members": [
                {"stat_var": _BARE_STAT_VAR, "native_unit": "CurrencyUSD"},
                {"stat_var": _BARE_STAT_VAR_B, "native_unit": "CurrencyUSD"},
            ],
            "fallback": {"count": 0, "examples": [], "truncated": False},
            "projected": {"count": 0, "examples": [], "truncated": False},
        },
    )
    assert result.conversion_record is None
    assert result.member_conversion_records.keys() == {_STAT_VAR, _STAT_VAR_B}
    for record in result.member_conversion_records.values():
        assert record is not None
        assert record.table_provenance_id == _TABLE_PROVENANCE


def test_us_dollar_group_members_trace_their_facet_unit_like_the_member_path():
    """`USDollar` facets convert on `CurrencyUSD` rates. The group trace names `USDollar` per
    member, as each member's own `materialise_spec` result does.

    A one-request group takes the member path, so the group here has two members."""
    bare_stat_vars = (_BARE_STAT_VAR, _BARE_STAT_VAR_B)
    arcs = _base_arcs(bare_stat_vars=bare_stat_vars)
    arcs["USDollar"] = {"typeOf": ("CurrencyUnit",)}
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    role_bodies = {
        bare: _ROLE_HEADER + _role_row(stat_var=bare, year="2022", value="100", unit="USDollar")
        for bare in bare_stat_vars
    }

    def _run(materialise, request, entries):
        sdmx, _ = _router_sdmx(
            role_bodies=role_bodies, subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()}
        )
        return materialise(
            request,
            sdmx=sdmx,
            graph=_graph(arcs),
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(_graph(arcs), parties=lambda: {}),
            rate_cache=RateTableCache(),
            coverage=_coverage_for(entries, unit="USDollar"),
        )

    members = {
        stat_var: _run(
            materialise_spec,
            _request(stat_var=stat_var, unit=eur_request),
            [(bare, "country/KEN")],
        )
        for stat_var, bare in ((_STAT_VAR, _BARE_STAT_VAR), (_STAT_VAR_B, _BARE_STAT_VAR_B))
    }
    group = _run(
        materialise_group,
        [_request(stat_var=stat_var, unit=eur_request) for stat_var in members],
        [(bare, "country/KEN") for bare in bare_stat_vars],
    )

    member_entries: list[object] = []
    for result in members.values():
        (member_entry,) = result.trace.conversion
        member_entries += member_entry["members"]
    assert member_entries == [
        {"stat_var": bare, "native_unit": "USDollar"} for bare in bare_stat_vars
    ]
    (group_entry,) = group.trace.conversion
    assert group_entry["members"] == member_entries
    for stat_var, record in group.member_conversion_records.items():
        assert record is not None
        assert record.native_currency == "CurrencyUSD"
        assert record.native_unit_ref == "USDollar"
        assert record == members[stat_var].member_conversion_records[stat_var]


def test_finish_group_raises_when_converted_members_disagree_on_recipe():
    from materialise.convert import ConversionRecord

    records = {
        _STAT_VAR: ConversionRecord(
            recipe_id="recipe/A",
            exchange_rate_table_version="v1",
            deflator_table_version=None,
            native_currency="CurrencyUSD",
            native_unit_ref="CurrencyUSD",
            table_provenance_id=_TABLE_PROVENANCE,
            table_refresh_date="2026-01-01",
            fallback_count=0,
            fallback_examples=(),
            projected_count=0,
            projected_examples=(),
        ),
        _STAT_VAR_B: ConversionRecord(
            recipe_id="recipe/B",
            exchange_rate_table_version="v1",
            deflator_table_version=None,
            native_currency="CurrencyUSD",
            native_unit_ref="CurrencyUSD",
            table_provenance_id=_TABLE_PROVENANCE,
            table_refresh_date="2026-01-01",
            fallback_count=0,
            fallback_examples=(),
            projected_count=0,
            projected_examples=(),
        ),
    }

    with pytest.raises(MaterialisationUsageError):
        _sorted_converted_members(records)


# ---------------------------------------------------------------------------
# deflator_missing_omitted: a constant-price conversion leaves out the rows whose deflator
# the table lacks, and refuses only when no row is left
# ---------------------------------------------------------------------------


def _deflator_gap_member(deflator_years: tuple[str, ...]) -> MaterialisationResult:
    """France gives to Kenya in 2023 and 2024, read in constant 2024 US dollars. France's
    deflator holds `deflator_years` alone."""
    request = _request(
        unit=FacetBinding(state="request", request=UnitRequest("USD", "constant", base_year=2024))
    )
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER
            + _role_row(year="2023", value="100")
            + _role_row(year="2024", value="100")
        },
        subject_bodies={
            _EXCHANGE_VAR: _SUBJECT_HEADER
            + "".join(
                _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year=year, value="1.0")
                for year in ("2023", "2024")
            ),
            _DEFLATOR_VAR: _SUBJECT_HEADER
            + "".join(
                _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year=year, value="100.0")
                for year in deflator_years
            ),
        },
    )
    coverage = build_coverage_index(
        [(_BARE_STAT_VAR, "f1", "country/FRA", "country/KEN", [2023, 2024])],
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
    return materialise_spec(
        request,
        sdmx=sdmx,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )


def test_a_year_missing_its_deflator_is_left_out_and_named_in_the_manifest():
    result = _deflator_gap_member(("2024",))

    assert result.coverage.rows == 1
    (warning,) = [w for w in result.warnings if w["code"] == "deflator_missing_omitted"]
    assert (warning["severity"], warning["omitted_entity_years"]) == (
        "warn",
        {"count": 1, "entities": [{"ref": "dcid:country/FRA", "years": ["2023"]}]},
    )
    # The trace counts what the conversion left out, which the description's step states.
    (conversion,) = result.trace.conversion or ()
    assert conversion["left_out"] == [
        {"reason": "deflator_missing", "count": 1, "of": "entities"},
        {"reason": "deflator_missing", "count": 1, "of": "entity_periods"},
    ]


def test_a_base_year_missing_its_deflator_refuses_when_no_row_is_left():
    with pytest.raises(RateCoverageGapError) as exc_info:
        _deflator_gap_member(("2023",))
    assert exc_info.value.cause == "deflator_missing"
    assert exc_info.value.entities == ("country/FRA@2023", "country/FRA@2024")


# ---------------------------------------------------------------------------
# projected terms: counted in the trace, never a materialisation warning, on member, group,
# and preview
# ---------------------------------------------------------------------------


def test_projected_terms_are_counted_in_the_trace_and_raise_no_warning_on_member():
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
        subject_bodies={
            _EXCHANGE_VAR: _eur_exchange_rate_body(eu_measurement_method="IMFProjection")
        },
    )

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_spec(
        request,
        sdmx=sdmx,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    (conversion,) = result.trace.conversion
    assert conversion["projected"]["count"] == 1
    assert conversion["projected"]["examples"] == ["CurrencyEUR@2022"]
    assert "left_out" not in conversion
    assert "deflator_projected" not in [w["code"] for w in result.warnings]


def test_projected_terms_are_counted_in_the_trace_and_raise_no_warning_on_group():
    arcs = _base_arcs(bare_stat_vars=(_BARE_STAT_VAR, _BARE_STAT_VAR_B))
    sdmx, _requests = _router_sdmx(
        role_bodies={
            _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100"),
            _BARE_STAT_VAR_B: (
                _ROLE_HEADER + _role_row(stat_var=_BARE_STAT_VAR_B, year="2022", value="200")
            ),
        },
        subject_bodies={
            _EXCHANGE_VAR: _eur_exchange_rate_body(eu_measurement_method="IMFProjection")
        },
    )
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    members = [
        _request(stat_var=_STAT_VAR, unit=eur_request),
        _request(stat_var=_STAT_VAR_B, unit=eur_request),
    ]

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])
    result = materialise_group(
        members,
        sdmx=sdmx,
        graph=_graph(arcs),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(arcs), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )

    assert "deflator_projected" not in [w["code"] for w in result.warnings]
    # Both members project `CurrencyEUR@2022`, so one example represents 2 rows, untruncated.
    (conversion,) = result.trace.conversion
    assert conversion["projected"] == {
        "count": 2,
        "examples": ["CurrencyEUR@2022"],
        "truncated": False,
    }


def test_projected_terms_raise_no_warning_on_preview():
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
        subject_bodies={
            _EXCHANGE_VAR: _eur_exchange_rate_body(eu_measurement_method="IMFProjection")
        },
    )

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")])
    result = materialise_preview(
        request, sdmx=sdmx, graph=_graph(), coverage=coverage, limit=10, rate_cache=RateTableCache()
    )

    assert result.conversion_record is not None
    assert result.conversion_record.projected_count == 1
    assert "deflator_projected" not in [w["code"] for w in result.warnings]


# ---------------------------------------------------------------------------
# currency_unmapped_omitted: a conversion into each recipient's own currency omits the
# recipients no currency names, and refuses only when no row is left
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DestinationCurrencyInputs:
    """One 2022 row per bound recipient for each StatVar, read in each recipient's currency.
    Only Kenya is named by a currency (`CurrencyKES`)."""

    sdmx: SdmxClient
    arcs: dict[str, dict[str, tuple[str, ...]]]
    coverage: CoverageIndex
    requests: list[MaterialisationRequest]


def _destination_currency_inputs(
    recipients_by_stat_var: dict[str, list[str]],
    *,
    unrated: dict[str, str] | None = None,
    unrated_type: str = "CurrencyUnit",
) -> _DestinationCurrencyInputs:
    """`unrated` names each further recipient's national currency, typed `unrated_type`,
    which the rate table holds no rate for."""
    stat_vars = tuple(recipients_by_stat_var)
    arcs = _base_arcs(bare_stat_vars=stat_vars)
    arcs["CurrencyKES"] = {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": ("country/KEN",)}
    for entity, currency in (unrated or {}).items():
        arcs[currency] = {"typeOf": (unrated_type,), "nationalCurrencyOf": (entity,)}
    rate_body = (
        _SUBJECT_HEADER
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2022", value="112.0")
    )
    sdmx, _requests = _router_sdmx(
        role_bodies={
            bare: _ROLE_HEADER
            + "".join(
                _role_row(stat_var=bare, recipient=r, year="2022", value="100") for r in recipients
            )
            for bare, recipients in recipients_by_stat_var.items()
        },
        subject_bodies={_EXCHANGE_VAR: rate_body},
    )
    unit = FacetBinding(state="request", request=UnitRequest("destination", "current"))
    return _DestinationCurrencyInputs(
        sdmx=sdmx,
        arcs=arcs,
        coverage=_coverage_for(
            [(bare, r) for bare, recipients in recipients_by_stat_var.items() for r in recipients]
        ),
        requests=[
            _request(
                stat_var=f"dcid:{bare}",
                unit=unit,
                role_bindings={"financingProvider": ["country/FRA"], "financingRecipient": rs},
            )
            for bare, rs in recipients_by_stat_var.items()
        ],
    )


def _destination_currency_result(
    entry: str,
    recipients: list[str],
    *,
    second: list[str] | None = None,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    unrated: dict[str, str] | None = None,
    unrated_type: str = "CurrencyUnit",
) -> MaterialisationResult | PreviewResult:
    """Materialise `_destination_currency_inputs` through `entry`, binding every recipient.
    A group has a second StatVar binding `second`, by default `recipients`."""
    by_stat_var = {_BARE_STAT_VAR: recipients}
    if entry == "group":
        by_stat_var[_BARE_STAT_VAR_B] = second if second is not None else recipients
    inputs = _destination_currency_inputs(by_stat_var, unrated=unrated, unrated_type=unrated_type)
    if entry == "preview":
        return materialise_preview(
            inputs.requests[0],
            sdmx=inputs.sdmx,
            graph=_graph(inputs.arcs),
            coverage=inputs.coverage,
            limit=100,
            ceiling=ceiling,
            rate_cache=RateTableCache(),
        )
    common = {
        "ceiling": ceiling,
        "sdmx": inputs.sdmx,
        "graph": _graph(inputs.arcs),
        "materialiser_build": "test-build",
        "capability_version": "v1",
        "enrichment": EnrichmentLookup(_graph(inputs.arcs), parties=lambda: {}),
        "rate_cache": RateTableCache(),
        "coverage": inputs.coverage,
    }
    if entry == "group":
        return materialise_group(inputs.requests, **common)
    return materialise_spec(inputs.requests[0], **common)


def _served_recipients(result: MaterialisationResult | PreviewResult) -> set[str]:
    frames = [result.frame] if isinstance(result, PreviewResult) else result.frame_chunks()
    return {
        recipient for frame in frames for recipient in frame.table["financingRecipient"].to_pylist()
    }


_ENTRIES = ("member", "group", "preview")
_MINT_ENTRIES = ("member", "group")


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize(
    ("regions", "counted"),
    [
        pytest.param(
            ["southOfSahara", "africaRegion"],
            {"count": 2, "refs": ["dcid:africaRegion", "dcid:southOfSahara"]},
            id="named",
        ),
        pytest.param(
            [f"region/R{i:02d}" for i in range(12)],
            {"count": 12, "refs": [f"dcid:region/R{i:02d}" for i in range(10)]},
            id="capped",
        ),
    ],
)
def test_destination_currency_omits_the_recipients_no_currency_names(
    entry: str, regions: list[str], counted: dict[str, object]
):
    """Kenya converts; the regions' rows are left out and one warning counts them and carries
    the first five by ref, sorted. Bound explicitly, they raise no second
    `coverage_below_scope` warning."""
    result = _destination_currency_result(entry, ["country/KEN", *regions])

    assert _served_recipients(result) == {"country/KEN"}
    omitted = [w for w in result.warnings if w["code"] == "currency_unmapped_omitted"]
    assert [w["severity"] for w in omitted] == ["warn"]
    assert omitted[0]["omitted_entities"] == counted
    assert not any(w["code"] == "coverage_below_scope" for w in result.warnings)
    if isinstance(result, MaterialisationResult):
        (conversion,) = result.trace.conversion or ()
        assert conversion["left_out"] == [
            {"reason": "currency_unmapped", "count": len(regions), "of": "entities"}
        ]


@pytest.mark.parametrize("entry", _ENTRIES)
def test_destination_currency_refuses_when_no_recipient_is_named_by_a_currency(entry: str):
    with pytest.raises(RateCoverageGapError) as exc_info:
        _destination_currency_result(entry, ["southOfSahara", "africaRegion"])
    assert exc_info.value.cause == "currency_unmapped"
    assert exc_info.value.entities == ("africaRegion", "southOfSahara")


@pytest.mark.parametrize("entry", _MINT_ENTRIES)
@pytest.mark.parametrize(
    "unrated_type",
    [
        pytest.param("CurrencyUnit", id="rate-gap"),
        pytest.param("UnitOfMeasure", id="not-a-currency"),
    ],
)
def test_an_over_ceiling_request_refuses_over_ceiling_before_a_conversion_refusal(
    entry: str, unrated_type: str
):
    """Kenya converts, and Senegal's franc has no rate or is not typed a currency. Two rows
    are read against a ceiling of one, so the request refuses `over_ceiling` before
    conversion meets the gap or the currency read meets the untyped unit."""
    with pytest.raises(OverCeilingError):
        _destination_currency_result(
            entry,
            ["country/KEN", "country/SEN"],
            second=["country/KEN", "country/SEN"],
            ceiling=1,
            unrated={"country/SEN": "CurrencyXOF"},
            unrated_type=unrated_type,
        )


@dataclass(frozen=True)
class _LatestMint:
    """A destination-currency `latest` read through one entry, its requests and index, and
    the `(recipient, period)` of every row it serves."""

    result: MaterialisationResult | PreviewResult
    requests: list[MaterialisationRequest]
    coverage: CoverageIndex
    served: list[tuple[str, str]]


def _destination_latest(
    entry: str,
    held: dict[str, tuple[int, ...]],
    *,
    currencies: dict[str, str],
    blank: frozenset[tuple[str, int]] = frozenset(),
    rate_years: tuple[int, ...] = (2022,),
    unindexed: frozenset[str] = frozenset(),
) -> _LatestMint:
    """Read each recipient of `held` from France over the years it holds, valued but at
    `blank`, converted into each recipient's own currency. `currencies` names the ones a
    currency names. The rate table holds `rate_years` alone, so a row of another year
    refuses. The coverage index holds every recipient's series but those of `unindexed`."""
    stat_vars = (_BARE_STAT_VAR, _BARE_STAT_VAR_B) if entry == "group" else (_BARE_STAT_VAR,)
    arcs = _base_arcs(bare_stat_vars=stat_vars)
    for entity, code in currencies.items():
        arcs[f"Currency{code}"] = {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": (entity,)}

    def role_handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        period = params.get("c[TIME_PERIOD]")
        body = _ROLE_HEADER
        for bare in params["c[variableMeasured]"].split(","):
            for recipient, years in held.items():
                if period == "LATEST":
                    years = years[-1:]
                elif period is not None:
                    years = tuple(year for year in years if str(year) in period.split(","))
                for year in years:
                    body += _role_row(
                        stat_var=bare,
                        recipient=recipient,
                        year=str(year),
                        value=".." if (recipient, year) in blank else "100",
                    )
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    rate_body = _SUBJECT_HEADER + "".join(
        _rate_row(stat_var=_EXCHANGE_VAR, entity=f"Currency{code}", year=str(year), value=rate)
        for year in rate_years
        for code, rate in (("USD", "1.0"), *((code, "2.0") for code in currencies.values()))
    )
    sdmx = RateTableServer({_EXCHANGE_VAR: rate_body}, fallback=role_handler).client(
        base_url=_BASE_URL
    )
    coverage = build_coverage_index(
        [
            (bare, "f1", "country/FRA", recipient, list(years))
            for bare in stat_vars
            for recipient, years in held.items()
            if recipient not in unindexed
        ],
        role_names=dict.fromkeys(
            stat_vars, RoleNames(donor="financingProvider", recipient="financingRecipient")
        ),
        facet_metadata={
            "f1": FacetMetadata(
                provenance_id=_MEMBER_PROVENANCE,
                measurement_method="GrantEquivalent",
                unit="CurrencyUSD",
            )
        },
    )
    requests = [
        _request(
            stat_var=f"dcid:{bare}",
            unit=FacetBinding(state="request", request=UnitRequest("destination", "current")),
            window=RequestedWindow(kind="latest"),
            role_bindings={"financingProvider": ["country/FRA"], "financingRecipient": [*held]},
        )
        for bare in stat_vars
    ]
    graph = _graph(arcs)
    result: MaterialisationResult | PreviewResult
    if entry == "preview":
        result = materialise_preview(
            requests[0],
            sdmx=sdmx,
            graph=graph,
            coverage=coverage,
            limit=100,
            rate_cache=RateTableCache(),
        )
        frames = [result.frame]
    else:
        result = materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            materialiser_build="test-build",
            capability_version="v1",
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            rate_cache=RateTableCache(),
            coverage=coverage,
        )
        frames = list(result.frame_chunks())
    served = sorted(
        (recipient, period)
        for frame in frames
        for recipient, period in zip(
            frame.table["financingRecipient"].to_pylist(),
            frame.table["time_period"].to_pylist(),
            strict=True,
        )
    )
    return _LatestMint(result=result, requests=requests, coverage=coverage, served=served)


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize(
    "kenya_2023", [pytest.param("100", id="newest-value"), pytest.param("..", id="newest-blank")]
)
def test_destination_currency_latest_elects_its_year_without_the_omitted_recipients(
    entry: str, kenya_2023: str
):
    """Kenya and a region no currency names hold 2021 to 2023, and Uganda and Tanzania 2021
    and 2022. The three converted series elect 2022, since one of three reaches 2023. With
    the region counted two of four would, and 2023 would stand. The read serves the three
    at 2022, reading it again for Kenya, and leaves the region's 2022 row out. Kenya's 2022
    row converts whether or not its 2023 value parses."""
    currencies = {"country/KEN": "KES", "country/UGA": "UGX", "country/TZA": "TZS"}
    mint = _destination_latest(
        entry,
        {
            "country/KEN": (2021, 2022, 2023),
            "country/UGA": (2021, 2022),
            "country/TZA": (2021, 2022),
            "africaRegion": (2021, 2022, 2023),
        },
        currencies=currencies,
        blank=frozenset({("country/KEN", 2023)}) if kenya_2023 == ".." else frozenset(),
    )

    if isinstance(mint.result, MaterialisationResult):
        assert set(mint.result.latest_years.values()) == {
            LatestYear(
                elected=2022,
                served=2022,
                facet_ids=("f1",),
                omitted=(("financingRecipient", "africaRegion"),),
            )
        }
        # The recorded facets and omissions elect the same year again from the same index.
        assert {
            elected_latest_year(
                request, latest.omitted, coverage=mint.coverage, facet_ids=latest.facet_ids
            )
            for request in mint.requests
            for latest in [mint.result.latest_years[request.stat_var]]
        } == {2022}
    per_stat_var = len(mint.requests)
    assert mint.served == sorted(
        (recipient, "2022") for recipient in currencies for _ in range(per_stat_var)
    )


_NAMED_BY_A_CURRENCY = {"country/KEN": "KES", "country/UGA": "UGX", "country/TZA": "TZS"}


@pytest.mark.parametrize("entry", _MINT_ENTRIES)
@pytest.mark.parametrize(
    ("held", "served", "warned"),
    [
        pytest.param({"africaRegion": (2023,)}, [], {"empty_result"}, id="alone"),
        pytest.param(
            {
                **dict.fromkeys(_NAMED_BY_A_CURRENCY, (2021, 2022)),
                "africaRegion": (2023,),
            },
            [(recipient, "2022") for recipient in _NAMED_BY_A_CURRENCY],
            {"coverage_below_scope"},
            id="beside-named-recipients",
        ),
    ],
)
def test_a_recipient_no_currency_names_and_holding_no_value_is_reported_as_not_served(
    entry: str, held: dict[str, tuple[int, ...]], served: list[tuple[str, str]], warned: set[str]
):
    """The region's one row holds no value, so it serves nothing and is reported as not
    served, not as left out for its currency. The year is elected without its series, which
    no conversion could serve, and the run records it as omitted. Beside three recipients
    holding 2022, 2022 stands."""
    mint = _destination_latest(
        entry, held, currencies=_NAMED_BY_A_CURRENCY, blank=frozenset({("africaRegion", 2023)})
    )

    assert isinstance(mint.result, MaterialisationResult)
    assert mint.served == sorted(served * len(mint.requests))
    assert {w["code"] for w in mint.result.warnings} == warned
    assert {latest.omitted for latest in mint.result.latest_years.values()} == {
        (("financingRecipient", "africaRegion"),)
    }


@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_recipient_no_currency_names_is_left_out_before_the_year_is_elected(entry: str):
    """Kenya and Uganda hold 2023, three more recipients hold 2022 alone, and a region no
    currency names holds 2023 in the index with no value in it. Without the region two of
    five series reach 2023, under half, so 2022 is elected and all five serve it, as
    Search plans; counting the region, three of six would elect 2023."""
    currencies = {
        "country/KEN": "KES",
        "country/UGA": "UGX",
        "country/TZA": "TZS",
        "country/RWA": "RWF",
        "country/ETH": "ETB",
    }
    mint = _destination_latest(
        entry,
        {
            "country/KEN": (2021, 2022, 2023),
            "country/UGA": (2021, 2022, 2023),
            "country/TZA": (2021, 2022),
            "country/RWA": (2021, 2022),
            "country/ETH": (2021, 2022),
            "africaRegion": (2021, 2022, 2023),
        },
        currencies=currencies,
        blank=frozenset({("africaRegion", 2023)}),
        rate_years=(2022, 2023),
    )

    assert mint.served == sorted((r, "2022") for r in currencies for _ in mint.requests)
    if isinstance(mint.result, MaterialisationResult):
        assert {
            (latest.elected, latest.served, latest.omitted)
            for latest in mint.result.latest_years.values()
        } == {(2022, 2022, (("financingRecipient", "africaRegion"),))}
        assert {w["code"] for w in mint.result.warnings} == {"currency_unmapped_omitted"}


def test_an_omitted_recipient_the_index_holds_no_series_of_is_logged(caplog):
    """The region holds a 2022 value the index holds no series of. The election counts the
    index's series, so it stands, and the region's row is left out and recorded."""
    with caplog.at_level("WARNING", logger="materialise.generation"):
        mint = _destination_latest(
            "member",
            {**dict.fromkeys(_NAMED_BY_A_CURRENCY, (2021, 2022)), "africaRegion": (2021, 2022)},
            currencies=_NAMED_BY_A_CURRENCY,
            unindexed=frozenset({"africaRegion"}),
        )

    assert isinstance(mint.result, MaterialisationResult)
    assert mint.served == sorted((recipient, "2022") for recipient in _NAMED_BY_A_CURRENCY)
    assert {(latest.elected, latest.omitted) for latest in mint.result.latest_years.values()} == {
        (2022, (("financingRecipient", "africaRegion"),))
    }
    assert [
        record.args[0] for record in caplog.records if record.name == "materialise.generation"
    ] == [1]


# ---------------------------------------------------------------------------
# `count` records affected rows. `truncated` tracks distinct examples against the cap.
# ---------------------------------------------------------------------------


def _record(
    *,
    recipe_id: str = _DAC_RECIPE,
    exchange_rate_table_version: str = "0" * 64,
    deflator_table_version: str | None = None,
    native_unit_ref: str = "CurrencyUSD",
    fallback_count: int = 0,
    fallback_examples=(),
    projected_count: int = 0,
    projected_examples=(),
) -> ConversionRecord:
    return ConversionRecord(
        recipe_id=recipe_id,
        exchange_rate_table_version=exchange_rate_table_version,
        deflator_table_version=deflator_table_version,
        native_currency="CurrencyUSD",
        native_unit_ref=native_unit_ref,
        table_provenance_id=_TABLE_PROVENANCE,
        table_refresh_date="2026-01-01",
        fallback_count=fallback_count,
        fallback_examples=fallback_examples,
        projected_count=projected_count,
        projected_examples=projected_examples,
    )


def test_one_entity_falling_back_across_two_years_is_not_truncated():
    record = _record(fallback_count=2, fallback_examples=(("country/FRA", "DACCountriesTotal"),))

    (conversion,) = _conversion_trace([(_STAT_VAR, record)])

    assert conversion["fallback"] == {
        "count": 2,
        "examples": [["country/FRA", "DACCountriesTotal"]],
        "truncated": False,
    }


def test_eleven_distinct_projected_entities_are_truncated_at_ten():
    examples = tuple(f"country/C{i:02d}@2022" for i in range(11))
    record = _record(projected_count=11, projected_examples=examples)

    (conversion,) = _conversion_trace([(_STAT_VAR, record)])

    assert conversion["projected"]["count"] == 11
    assert len(conversion["projected"]["examples"]) == 10
    assert conversion["projected"]["examples"] == sorted(examples)[:10]
    assert conversion["projected"]["truncated"] is True


# ---------------------------------------------------------------------------
# Preview (#662): a converted preview mints the mint's own version
# ---------------------------------------------------------------------------


def _converted_preview_and_mint(*, native_unit: str = "CurrencyUSD"):
    request = _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current")))
    # `CurrencyGBP`'s rate entity needs its own row in the exchange-rate
    # table, and carries the same 1.0 rate `CurrencyUSD`'s does, so the
    # converted value stays byte-identical while only the native unit dcid
    # the member's facet resolved to moves.
    if native_unit == "CurrencyUSD":
        rate_body = _eur_exchange_rate_body()
    else:
        rate_body = (
            _SUBJECT_HEADER
            # `_validate_table` requires a `CurrencyUSD` anchor row on every
            # exchange-rate table, whether or not USD is this member's own
            # native or target currency.
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyGBP", year="2022", value="1.0")
            + _rate_row(
                stat_var=_EXCHANGE_VAR,
                entity="CurrencyEUR",
                year="2022",
                value="0.9",
            )
        )

    def _sdmx() -> SdmxClient:
        sdmx, _ = _router_sdmx(
            role_bodies={
                _BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100", unit=native_unit)
            },
            subject_bodies={_EXCHANGE_VAR: rate_body},
        )
        return sdmx

    coverage = _coverage_for([(_BARE_STAT_VAR, "country/KEN")], unit=native_unit)
    minted = materialise_spec(
        request,
        sdmx=_sdmx(),
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=coverage,
    )
    previewed = materialise_preview(
        request,
        sdmx=_sdmx(),
        graph=_graph(),
        coverage=coverage,
        with_content_version=True,
        limit=10,
        rate_cache=RateTableCache(),
    )
    return minted, previewed


def test_converted_preview_content_version_equals_the_mint_version():
    minted, previewed = _converted_preview_and_mint()

    assert minted.conversion_record is not None
    assert previewed.conversion_record is not None
    assert previewed.content_version is not None
    assert previewed.content_version == minted.content_version


def test_conversion_trace_groups_members_by_recipe_and_table_versions():
    """Members sharing a recipe and both table versions share an entry; a different deflator
    version under the same recipe gets its own. Entries sort by recipe then table versions,
    members by StatVar, and counts sum within an entry only."""
    trace = _conversion_trace(
        [
            ("dcid:Z", _record(recipe_id=_IDS_RECIPE, projected_count=3)),
            ("dcid:B", _record(deflator_table_version="d2", fallback_count=1)),
            ("dcid:A", _record(native_unit_ref="USDollar", projected_count=2)),
            ("dcid:C", _record(projected_count=5)),
        ]
    )

    assert trace is not None
    assert [(entry["recipe"], entry.get("deflator_table_version")) for entry in trace] == [
        (_DAC_RECIPE, None),
        (_DAC_RECIPE, "d2"),
        (_IDS_RECIPE, None),
    ]
    assert [entry["members"] for entry in trace] == [
        [
            {"stat_var": "A", "native_unit": "USDollar"},
            {"stat_var": "C", "native_unit": "CurrencyUSD"},
        ],
        [{"stat_var": "B", "native_unit": "CurrencyUSD"}],
        [{"stat_var": "Z", "native_unit": "CurrencyUSD"}],
    ]
    assert [entry["projected"]["count"] for entry in trace] == [7, 0, 3]
    assert [entry["fallback"]["count"] for entry in trace] == [0, 1, 0]


def test_conversion_trace_is_absent_when_no_member_converted():
    assert _conversion_trace([]) is None


def test_conversion_trace_lists_a_shared_stat_var_once_and_counts_every_record():
    """Records over one StatVar and native unit name that member once. Each record counts its
    own rows, so the counts sum over every record."""
    (entry,) = _conversion_trace(
        [
            (_STAT_VAR, _record(fallback_count=1, fallback_examples=(("KEN", "2020"),))),
            (_STAT_VAR, _record(fallback_count=2, fallback_examples=(("UGA", "2020"),))),
            (_STAT_VAR, _record(native_unit_ref="USDollar", projected_count=4)),
        ]
    )

    assert entry["members"] == [
        {"stat_var": _BARE_STAT_VAR, "native_unit": "CurrencyUSD"},
        {"stat_var": _BARE_STAT_VAR, "native_unit": "USDollar"},
    ]
    assert entry["fallback"] == {
        "count": 3,
        "examples": [["KEN", "2020"], ["UGA", "2020"]],
        "truncated": False,
    }
    assert entry["projected"]["count"] == 4


def test_conversion_trace_members_do_not_depend_on_arrival_order():
    records = [
        (_STAT_VAR, _record(native_unit_ref="USDollar")),
        (_STAT_VAR_B, _record()),
        (_STAT_VAR, _record()),
    ]

    assert _conversion_trace(records) == _conversion_trace(list(reversed(records)))


# ---------------------------------------------------------------------------
# Canonical frame bytes alone reproduce a converted frame's content version
# ---------------------------------------------------------------------------


def _recompute_from_frame_bytes(result: MaterialisationResult) -> str:
    """Recomputes the token as a client does, from the canonical frame bytes alone."""
    return recompute_version(canonical_frame_bytes(only_chunk(result)))


def _current_price_member() -> MaterialisationResult:
    sdmx, _requests = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + _role_row(year="2022", value="100")},
        subject_bodies={_EXCHANGE_VAR: _eur_exchange_rate_body()},
    )
    return materialise_spec(
        _request(unit=FacetBinding(state="request", request=UnitRequest("EUR", "current"))),
        sdmx=sdmx,
        graph=_graph(),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=_coverage_for([(_BARE_STAT_VAR, "country/KEN")]),
    )


@pytest.mark.parametrize(
    "build", [_current_price_member, _constant_price_member], ids=["current", "constant"]
)
def test_converted_member_token_recomputes_from_its_frame_bytes_alone(build):
    result = build()

    assert result.trace.conversion is not None
    assert _recompute_from_frame_bytes(result) == result.content_version


def test_converted_group_token_recomputes_from_its_frame_bytes_alone():
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
    result = materialise_group(
        [
            _request(stat_var=_STAT_VAR, unit=eur_request),
            _request(stat_var=_STAT_VAR_B, unit=eur_request),
        ],
        sdmx=sdmx,
        graph=_graph(arcs),
        materialiser_build="test-build",
        capability_version="v1",
        enrichment=EnrichmentLookup(_graph(arcs), parties=lambda: {}),
        rate_cache=RateTableCache(),
        coverage=_coverage_for(
            [(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")]
        ),
    )

    assert _recompute_from_frame_bytes(result) == result.content_version
