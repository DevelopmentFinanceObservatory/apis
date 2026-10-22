"""Checks the download facts a Spec states where the recorded mints
(`test_download_mint_contract`) do not reach: a member the release's coverage view does not hold,
a series whose facets carry no complete date list, and the licence."""

from __future__ import annotations

import pytest

from origin_catalogue import BASE, CardFacts, DatasetRef, bound_catalogue
from search.engine.clients.graph import Facet
from search.engine.retrieval.download import download_facts
from search.engine.retrieval.read_plan import PairPlan, PlanRoles, PlanSeries, plan_pairs
from search.models import DownloadLicence, TimeWindow
from tests.engine._fake_coverage import build_coverage_index
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

_SV = "Count_Person"
_PLACE = "country/KEN"
_PROVENANCE = "dc/base/Census"
_DATASET = "dc/d/Census"
_EMPTY_INDEX = build_coverage_index([], role_names={})


def _two_facet_plan(*, dated: bool) -> PairPlan:
    """Kenya's series in two facets of one provenance, method and unit, which one group
    serves: 2018 to 2020 and 2020 to 2022, five distinct years over six observations."""
    spans = {"annual": ("2018", "2019", "2020"), "yearly": ("2020", "2021", "2022")}
    facets = tuple(
        Facet(
            obs_count=3,
            earliest_date=dates[0],
            latest_date=dates[-1],
            facet_id=facet_id,
            provenance_id=_PROVENANCE,
            observation_period="P1Y" if facet_id == "annual" else None,
            dates=dates if dated else (),
        )
        for facet_id, dates in spans.items()
    )
    series = PlanSeries(stat_var=_SV, role_tuple=(("observationAbout", _PLACE),), candidates=facets)
    return plan_pairs([series], members=[_SV], years=None)


@pytest.mark.parametrize(
    ("dated", "rows", "exact"), [(True, 5, True), (False, 6, False)], ids=["dated", "undated"]
)
def test_a_series_in_two_facets_counts_its_dates_once(dated: bool, rows: int, exact: bool) -> None:
    """With every date listed the file's rows are the distinct dates; without, the facets'
    observation counts are an upper bound, since their dates can coincide."""
    download = download_facts(
        _two_facet_plan(dated=dated),
        _EMPTY_INDEX,
        requested_window=None,
        conversion=None,
        provenance=None,
    )

    assert (download.rows, download.rows_exact, download.series) == (rows, exact, 1)
    assert download.window == TimeWindow(start_year=2018, end_year=2022)


@pytest.mark.parametrize(
    ("earliest", "latest"),
    [(None, "2030"), ("2018", None), (None, None)],
    ids=["no_earliest", "no_latest", "no_dates"],
)
@pytest.mark.parametrize("dated_sibling", [False, True], ids=["alone", "beside_dated_facet"])
def test_a_series_with_an_unknown_bound_states_no_window(
    earliest: str | None, latest: str | None, dated_sibling: bool
) -> None:
    """A facet whose first or last date is unknown leaves the file's years unknown, also
    beside a facet whose span is known, whose bounds would otherwise stand for the series'."""
    facets = [
        Facet(
            obs_count=5,
            earliest_date=earliest,
            latest_date=latest,
            facet_id="partial",
            provenance_id=_PROVENANCE,
        )
    ]
    if dated_sibling:
        facets.append(
            Facet(
                obs_count=3,
                earliest_date="2018",
                latest_date="2020",
                facet_id="dated",
                provenance_id=_PROVENANCE,
                dates=("2018", "2019", "2020"),
            )
        )
    series = PlanSeries(
        stat_var=_SV, role_tuple=(("observationAbout", _PLACE),), candidates=tuple(facets)
    )
    plan = plan_pairs([series], members=[_SV], years=None)

    download = download_facts(
        plan, _EMPTY_INDEX, requested_window=None, conversion=None, provenance=None
    )

    assert download.series == 1
    assert download.window is None


def test_a_role_answer_reading_a_member_coverage_lacks_states_no_rows() -> None:
    """The mint reads the member from SDMX, so a count from coverage would understate."""
    plan = PairPlan(
        members=("Amount_Uncovered",),
        years=None,
        roles=PlanRoles(
            bound={"donor": frozenset({"country/FRA"})},
            role_names={"donor": "financingProvider", "recipient": "financingRecipient"},
        ),
    )

    download = download_facts(
        plan, _EMPTY_INDEX, requested_window=None, conversion=None, provenance=None
    )

    assert (download.rows, download.rows_exact, download.series, download.window) == (
        None,
        False,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("listed", "facts", "expected"),
    [
        (True, CardFacts(license_type="Public Domain"), DownloadLicence(label="Public Domain")),
        (True, CardFacts(), None),
        (False, CardFacts(license_type="Public Domain"), None),
    ],
    ids=["licence", "no_licence", "unlisted_provenance"],
)
def test_the_licence_is_the_provenance_entry_s_in_the_catalogue(
    listed: bool, facts: CardFacts, expected: DownloadLicence | None
) -> None:
    row = provenance_row(_PROVENANCE, BASE, dataset=DatasetRef(ref=_DATASET, label="Census"))
    catalogue = (
        restrictive_catalogue(provenances=[row], dataset_facts={_DATASET: facts})
        if listed
        else restrictive_catalogue()
    )

    with bound_catalogue(catalogue):
        download = download_facts(
            _two_facet_plan(dated=True),
            _EMPTY_INDEX,
            requested_window=None,
            conversion=None,
            provenance=_PROVENANCE,
        )

    assert download.licence == expected
