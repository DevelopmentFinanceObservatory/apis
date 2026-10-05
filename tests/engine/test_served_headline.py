"""Check that attaching delivered observations leaves the coverage-time headline alone."""

from __future__ import annotations

from search.engine.region.observations.dispatch import _attach_observations
from search.models import (
    Headline,
    HeadlineYear,
    ObservationPoint,
    ObservationSeries,
    SeriesEntityRef,
    SeriesStatVarRef,
    Spec,
    SpecObservations,
)
from tests.conftest import minimal_spec


def _observations(series: list[ObservationSeries]) -> SpecObservations:
    return SpecObservations(series=series, total_obs=0, truncated=False)


def _headline_spec() -> Spec:
    spec = Spec.model_validate(minimal_spec())
    headline = Headline(
        year=2032,
        year_named=False,
        observations=1,
        series=1,
        role_counts={"financingRecipient": 1, "financingProvider": 1},
        recent_years=[
            HeadlineYear(
                year=2032,
                observations=1,
                series=1,
                role_counts={"financingRecipient": 1, "financingProvider": 1},
            )
        ],
    )
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"headline": headline})}
    )


def _dated_series(sv: str, origin: str, dates: list[str]) -> ObservationSeries:
    return ObservationSeries(
        stat_var=SeriesStatVarRef(ref=sv),
        entities=[
            SeriesEntityRef(role="origin", ref=origin),
            SeriesEntityRef(role="destination", ref="country/KEN"),
        ],
        points=[ObservationPoint(date=date, value=1.0) for date in dates],
        total_obs=len(dates),
    )


def test_delivered_series_leave_the_headline_as_coverage_counted_it():
    """Coverage counts the facets the preview serves before any point is read, so a buffered
    response keeps the headline a streamed `response` event already carried, whatever cut
    of the population the preview delivers."""
    spec = _headline_spec()
    observations = _observations(
        [
            _dated_series("Test_SV", "country/CHN", ["2022", "2023", "2024"]),
            _dated_series("Test_SV", "country/FRA", ["2023-06", "2024-06"]),
        ]
    )

    attached = _attach_observations(spec, (observations, None), [])

    assert attached.resolved.observations is observations
    assert attached.resolved.headline == Headline(
        year=2032,
        year_named=False,
        observations=1,
        series=1,
        role_counts={"financingRecipient": 1, "financingProvider": 1},
        recent_years=[
            HeadlineYear(
                year=2032,
                observations=1,
                series=1,
                role_counts={"financingRecipient": 1, "financingProvider": 1},
            )
        ],
    )


def test_headline_is_unchanged_without_delivered_points():
    spec = _headline_spec()
    attached = _attach_observations(
        spec, (_observations([_dated_series("Test_SV", "country/CHN", [])]), None), []
    )
    assert attached.resolved.headline == spec.resolved.headline

    not_shipped = _attach_observations(
        spec,
        (_observations([_dated_series("Test_SV", "country/CHN", ["2024"])]), None),
        [],
        include_points=False,
    )
    assert not_shipped.resolved.headline == spec.resolved.headline
