"""A composite answer's observation preview composes from its components' own: each component
keeps the points dated in the years it serves, and an empty composite says why. Its download
facts count the composite file's rows in memory, zero rows included."""

from __future__ import annotations

from code_index.composites import ComponentIdentity
from code_index.declarations import COMPOSITES
from code_index.shared import get_index
from code_index.types import CompositeDeclaration
from search.models import (
    BaseIdentity,
    DownloadFacts,
    ObservationPoint,
    ObservationSeries,
    SeriesStatVarRef,
    SpecObservations,
    omitted_observations,
)
from search.scope.composite import (
    ServedComponent,
    YearCounts,
    compose_observations,
    composite_download,
)
from search.scope.tables import ScopeFacts
from series_coverage.schema import RoleNames
from tests.code_index.aid_identity import education_aid_identity
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

_CRS_INCLUSIVE, _DAC2A_INCLUSIVE, _HEADLINE = COMPOSITES


def _served(declaration: CompositeDeclaration, *years: range) -> list[ServedComponent]:
    return [
        ServedComponent(
            component=ComponentIdentity(role=recipe.role, recipe=recipe, identity={}),
            years=YearCounts(counts={}),
            served=frozenset(served),
            members=None,
        )
        for recipe, served in zip(declaration.components, years, strict=True)
    ]


def _series(stat_var: str, *years: int, total_obs: int | None = None) -> ObservationSeries:
    return ObservationSeries(
        stat_var=SeriesStatVarRef(ref=stat_var),
        entities=[],
        points=[ObservationPoint(date=str(year), value=float(year)) for year in years],
        total_obs=len(years) if total_obs is None else total_obs,
    )


def _read(*series: ObservationSeries, truncated: bool = False) -> SpecObservations:
    return SpecObservations(
        series=list(series), total_obs=sum(s.total_obs for s in series), truncated=truncated
    )


def _dates(observations: SpecObservations) -> dict[str, list[str]]:
    return {s.stat_var.ref: [p.date for p in s.points] for s in observations.series}


_SPLICED = _served(_HEADLINE, range(2015, 2018), range(2018, 2026))


def test_spliced_keeps_both_components_years_and_series() -> None:
    composed = compose_observations(
        _SPLICED,
        [_read(_series("net", 2015, 2016, 2017)), _read(_series("ge", *range(2018, 2026)))],
    )
    assert _dates(composed) == {
        "net": ["2015", "2016", "2017"],
        "ge": [str(year) for year in range(2018, 2026)],
    }
    assert (composed.total_obs, composed.truncated, composed.omitted) == (11, False, None)


def test_a_point_outside_the_served_years_is_dropped() -> None:
    # A `latest` read is not clipped to its component's periods. The net component's 2024
    # point lies past 2017, so the composite drops it and the series it alone filled.
    served = _served(_HEADLINE, range(2015, 2025), range(2018, 2025))
    composed = compose_observations(
        served,
        [_read(_series("net", 2016, 2024), _series("late", 2024)), _read(_series("ge", 2024))],
    )
    assert _dates(composed) == {"net": ["2016"], "ge": ["2024"]}
    assert [s.total_obs for s in composed.series] == [1, 1]
    assert composed.total_obs == 2


def test_a_kept_series_holds_its_estimates() -> None:
    estimated = _series("ge", 2018, 2019).stating_estimates(True)
    composed = compose_observations(_SPLICED, [None, _read(estimated)])
    assert [s.holds_estimates for s in composed.series] == [True]


def test_components_with_no_point_in_their_years_omit_no_points() -> None:
    composed = compose_observations(
        _SPLICED,
        [
            _read(_series("net", 2020)),
            omitted_observations("observations_no_points", "none", dispatched=3, empty=3),
        ],
    )
    assert composed.series == []
    assert composed.omitted is not None
    assert (composed.omitted.reason, composed.omitted.dispatched) == ("observations_no_points", 4)


def test_a_component_omission_explains_an_empty_composite() -> None:
    composed = compose_observations(
        _SPLICED,
        [None, omitted_observations("observations_timed_out", "late", dispatched=2)],
    )
    assert composed.omitted is not None
    assert composed.omitted.reason == "observations_timed_out"


def test_a_missing_read_omits_probe_failed() -> None:
    composed = compose_observations(_SPLICED, [None, _read(_series("ge", 2017))])
    assert composed.omitted is not None
    assert composed.omitted.reason == "observations_probe_failed"


def test_stacked_total_obs_subtracts_the_points_dropped_at_the_tail() -> None:
    # The tables share 2015-2022, and the bilateral table runs on to 2024. A truncated bilateral
    # series shows its latest four points of six, so the cut falls among the visible ones.
    served = _served(_DAC2A_INCLUSIVE, range(2015, 2023), range(2015, 2023))
    bilateral = _series("bilateral", 2021, 2022, 2023, 2024, total_obs=10)
    imputed = _series("imputed", 2021, 2022)
    composed = compose_observations(served, [_read(bilateral, truncated=True), _read(imputed)])
    assert _dates(composed) == {"bilateral": ["2021", "2022"], "imputed": ["2021", "2022"]}
    assert [s.total_obs for s in composed.series] == [8, 2]
    assert (composed.total_obs, composed.truncated) == (10, True)


_USA, _KEN, _UGA = "country/USA", "country/KEN", "country/UGA"


def _download(
    members: tuple[frozenset[str] | None, frozenset[str] | None],
    series: list[tuple[str, str, list[int]]],
    served: range,
) -> DownloadFacts | None:
    """`composite_download` of the provider-inclusive composite whose bilateral and imputed
    components serve `members`, each in `served`, over a snapshot holding each of `series`,
    a (StatVar, recipient, years) from the United States."""
    coverage = build_coverage_index(
        [(dcid, "facet", _USA, recipient, years) for dcid, recipient, years in series],
        role_names=dict.fromkeys(
            (dcid for dcid, _, _ in series),
            RoleNames(donor="financingProvider", recipient="financingRecipient"),
        ),
    )
    identity = {"entities": {"origin": [_USA]}}
    components = [
        ServedComponent(
            component=ComponentIdentity(role=recipe.role, recipe=recipe, identity=identity),
            years=YearCounts(counts={}),
            served=frozenset(served),
            members=held,
        )
        for recipe, held in zip(_CRS_INCLUSIVE.components, members, strict=True)
    ]
    return composite_download(
        components,
        BaseIdentity.model_validate(education_aid_identity(source=_CRS_INCLUSIVE.provenance)),
        facts=ScopeFacts(index=get_index(), coverage=FakeCoverageService(coverage).snapshot()),
    )


def test_a_zero_filling_component_writes_a_row_per_member_and_cell() -> None:
    # The bilateral rows fill three cells in the shared 2020-2021: Kenya in both years and
    # Uganda in 2020. Its 2019 row lies outside them. Each of the three imputed members, the
    # one holding a Kenya 2020 row among them, writes one row per cell.
    download = _download(
        (frozenset({"a", "b"}), frozenset({"c", "d", "e"})),
        [("a", _KEN, [2019, 2020, 2021]), ("b", _UGA, [2020]), ("c", _KEN, [2020])],
        range(2020, 2022),
    )
    assert download is not None
    assert (download.rows, download.series, download.rows_exact) == (
        2 + 1 + 3 * 3,
        2 + 3 * 2,
        False,
    )
    assert download.window is not None
    assert (download.window.start_year, download.window.end_year) == (2020, 2021)


def test_a_component_whose_members_are_not_settled_states_no_rows() -> None:
    # An `observationAbout` family's rows are counted from facet metadata, which a count in
    # memory does not read.
    download = _download(
        (frozenset({"a"}), None), [("a", _KEN, [2020]), ("c", _KEN, [2020])], range(2020, 2021)
    )
    assert download is not None
    assert (download.rows, download.rows_exact, download.series, download.window) == (
        None,
        False,
        None,
        None,
    )
