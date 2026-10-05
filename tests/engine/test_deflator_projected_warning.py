"""Tests projected-row warnings from ``_attach_observations`` and materialise conversions."""

from __future__ import annotations

import httpx

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.facets import FacetSelection
from materialise.rate_tables import RateTableCache
from search.engine.region.codes import DEFLATOR_PROJECTED
from search.engine.region.observations.dispatch import _attach_observations
from search.models import (
    IdentityUnitRequest,
    ObservationSeries,
    ObservationSeriesConversion,
    SeriesEntityRef,
    SeriesFacetRef,
    SeriesStatVarRef,
    Spec,
    SpecObservations,
    Warning,
)
from tests.conftest import minimal_spec
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
    build_observations_planned,
)
from tests.fixtures import FakeGraph
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

# Fabricated conversions isolate warning handling from conversion arithmetic.


def _conversion(
    *, projected_count: int, projected_examples: list[str]
) -> ObservationSeriesConversion:
    return ObservationSeriesConversion(
        recipe_id="recipe/Test",
        table_provenance=SeriesFacetRef(ref="provenance/TestRateTables"),
        exchange_rate_table_version="2026-01-01T00:00:00Z",
        deflator_table_version=None,
        fallback_count=0,
        fallback_examples=[],
        projected_count=projected_count,
        projected_examples=projected_examples,
    )


def _series(entity: str, *, conversion: ObservationSeriesConversion | None) -> ObservationSeries:
    return ObservationSeries(
        stat_var=SeriesStatVarRef(ref="Test_SV"),
        entities=[SeriesEntityRef(role="observationAbout", ref=entity)],
        points=[],
        total_obs=0,
        source_refs=[],
        facet={},
        conversion=conversion,
    )


def _observations(series: list[ObservationSeries]) -> SpecObservations:
    return SpecObservations(series=series, total_obs=0, truncated=False)


def _attach(observations: SpecObservations, *, include_points: bool = True) -> list[Warning]:
    spec = Spec.model_validate(minimal_spec())
    warnings: list[Warning] = []
    _attach_observations(spec, (observations, None), warnings, include_points=include_points)
    return warnings


def test_single_projected_series_warns_with_count_and_example():
    """One projected row names its ``entity@year`` example; an unconverted series or zero
    projected rows emit no warning."""
    observations = _observations(
        [
            _series(
                "country/KEN",
                conversion=_conversion(projected_count=1, projected_examples=["country/KEN@2022"]),
            )
        ]
    )
    warnings = _attach(observations)
    (warn,) = [w for w in warnings if w.code == DEFLATOR_PROJECTED]
    assert warn.severity == "warn"
    assert "1 row(s)" in warn.message
    assert "country/KEN@2022" in warn.message

    # Perturbation. Sets projected_count and projected_examples to zero and
    # confirms the warning is absent.
    unprojected = _observations(
        [_series("country/KEN", conversion=_conversion(projected_count=0, projected_examples=[]))]
    )
    warnings_after = _attach(unprojected)
    assert not any(w.code == DEFLATOR_PROJECTED for w in warnings_after)

    unconverted = _observations([_series("country/KEN", conversion=None)])
    assert not any(w.code == DEFLATOR_PROJECTED for w in _attach(unconverted))


def test_two_projected_series_total_counts_and_cap_examples():
    """Counts sum across series.

    Examples are sorted, deduplicated, and capped at ten."""
    examples_a = [f"country/A{i}@2020" for i in range(1, 7)]
    examples_b = [f"country/B{i}@2020" for i in range(1, 7)]
    observations = _observations(
        [
            _series(
                "country/A",
                conversion=_conversion(projected_count=6, projected_examples=examples_a),
            ),
            _series(
                "country/B",
                conversion=_conversion(projected_count=6, projected_examples=examples_b),
            ),
        ]
    )
    warnings = _attach(observations)
    (warn,) = [w for w in warnings if w.code == DEFLATOR_PROJECTED]
    assert "12 row(s)" in warn.message
    assert "(truncated)" in warn.message
    expected_examples = sorted(set(examples_a + examples_b))[:10]
    for example in expected_examples:
        assert example in warn.message
    for example in sorted(set(examples_a + examples_b))[10:]:
        assert example not in warn.message


def test_include_points_false_never_warns():
    """A facets-only prefetch emits no projected-row warning."""
    observations = _observations(
        [
            _series(
                "country/KEN",
                conversion=_conversion(projected_count=1, projected_examples=["country/KEN@2022"]),
            )
        ]
    )
    warnings = _attach(observations, include_points=False)
    assert not any(w.code == DEFLATOR_PROJECTED for w in warnings)


# A production currency conversion exercises the attached-warning path.

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_BARE_STAT_VAR = "Test_DeflatorProjected_Amount"
_PROVENANCE = "provenance/TestDeflatorProjectedSource"
_RECIPE = "ConversionRecipeDAC"
_YEAR = "2022"
_ENTITY = "country/KEN"
_EUR = "CurrencyEUR"


def _recipe_arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    """Builds the conversion arcs for the DAC deflator recipe."""
    arcs = conversion_arcs(
        recipe_id=_RECIPE, provenance_id=_PROVENANCE, currencies=["CurrencyUSD", _EUR]
    )
    arcs[_PROVENANCE]["isPartOf"] = ()
    arcs[_BARE_STAT_VAR] = dict(CONVERTIBLE_STAT_VAR_ARCS)
    return arcs


def _usd_fixture() -> ObsFacetFixture:
    return ObsFacetFixture(
        facet_id="f1",
        points=(),
        provenance_id=_PROVENANCE,
        measurement_method="Actual",
        unit="CurrencyUSD",
        # The facet's one observation, the row each fixture serves.
        obs_count=1,
    )


def _elected_combo() -> dict[str, FacetSelection]:
    """Returns the elected provenance, measurement method, and unit selections."""

    def _selection(axis: str, ref: str) -> FacetSelection:
        return FacetSelection(
            axis=axis,
            ref=ref,
            basis="bound",
            default_policy=None,
            default_policy_version=None,
            selection_reason=None,
            dispatch_facet_id="f1",
        )

    return {
        "provenance": _selection("provenance", _PROVENANCE),
        "measurementMethod": _selection("measurementMethod", "Actual"),
        "unit": _selection("unit", "CurrencyUSD"),
    }


def _main_body(*, fixture: ObsFacetFixture, value: float) -> str:
    return _csv_header(["observationAbout"]) + _csv_row(
        stat_var=_BARE_STAT_VAR, role_dcids=[_ENTITY], fixture=fixture, date_str=_YEAR, value=value
    )


def _sdmx_router(*, eur_projected: bool, main_body: str) -> MaterialiseSdmxClient:
    def observations(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=main_body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(
        recipe_table_bodies(
            _RECIPE,
            rates={"CurrencyUSD": {int(_YEAR): 1.0}, _EUR: {int(_YEAR): 0.9}},
            projected={(_EUR, int(_YEAR))} if eur_projected else (),
        ),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _fetch_and_attach(*, eur_projected: bool) -> list[Warning]:
    """Builds and attaches a converted series with the requested rate method."""
    usd_fixture = _usd_fixture()
    graph = FakeGraphReader(
        arcs=_recipe_arcs(),
        facets={(_BARE_STAT_VAR, _ENTITY): [_facet_from_fixture(usd_fixture)]},
    )
    sdmx = _sdmx_router(
        eur_projected=eur_projected, main_body=_main_body(fixture=usd_fixture, value=100.0)
    )
    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [_ENTITY]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        elected={_BARE_STAT_VAR: (_elected_combo(),)},
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    spec = Spec.model_validate(minimal_spec())
    warnings: list[Warning] = []
    _attach_observations(spec, (observations, None), warnings)
    return warnings


def test_real_conversion_projected_rate_warns_on_the_text_door():
    """The IMF-projected EUR rate produces a warning.

    The actual EUR rate leaves the warning list empty."""
    warnings = _fetch_and_attach(eur_projected=True)
    matches = [w for w in warnings if w.code == DEFLATOR_PROJECTED]
    assert len(matches) == 1, f"expected exactly one deflator_projected warning, got {warnings}"
    assert "1 row(s)" in matches[0].message
    assert f"{_EUR}@{_YEAR}" in matches[0].message

    warnings_actual = _fetch_and_attach(eur_projected=False)
    assert not any(w.code == DEFLATOR_PROJECTED for w in warnings_actual)
