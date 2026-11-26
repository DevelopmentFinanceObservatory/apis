"""Checks the download facts a Spec states where the recorded mints
(`test_download_mint_contract`) do not reach: a member the release's coverage view does not hold,
a series whose facets carry no complete date list, and the licence. Also checks the years a
Spec's series hold data in (`answer_years`) on plans built directly."""

from __future__ import annotations

import pytest

from materialise.recipe_bodies import RECIPE_BODIES
from origin_catalogue import BASE, CardFacts, DatasetRef, bound_catalogue
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.retrieval.download import answer_years, download_facts
from search.engine.retrieval.facet_policy import NO_DECLARED_POLICIES
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    PlanRoles,
    PlanSeries,
    plan_pairs,
    role_years,
)
from search.engine.retrieval.role_resolve import _role_plan
from search.engine.unit.conversion import UNDECIDED, CountConversion, CountedConversion
from search.engine.unit.conversion_catalogue import RecipeTables
from search.models import DownloadLicence, TimeWindow, YearsWithData
from series_coverage.schema import RoleNames
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
    return plan_pairs([series], members=[_SV], years=None, declared=NO_DECLARED_POLICIES)


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
    plan = plan_pairs([series], members=[_SV], years=None, declared=NO_DECLARED_POLICIES)

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
        declared=NO_DECLARED_POLICIES,
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
    assert answer_years(plan, _EMPTY_INDEX, conversion=None) is None


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


def _facet(facet_id: str, years: list[int], *, dated: bool = True) -> Facet:
    return Facet(
        obs_count=len(years),
        earliest_date=str(min(years)),
        latest_date=str(max(years)),
        facet_id=facet_id,
        provenance_id=_PROVENANCE,
        observation_period="P1Y",
        dates=tuple(str(y) for y in years) if dated else (),
    )


def _standard_plan(series: dict[str, list[Facet]], *, years: TimeWindow | None = None) -> PairPlan:
    """One series per place in `series`, each in its facets, read under `years`."""
    planned = [
        PlanSeries(stat_var=_SV, role_tuple=(("observationAbout", place),), candidates=tuple(f))
        for place, f in series.items()
    ]
    return plan_pairs(planned, members=[_SV], years=years, declared=NO_DECLARED_POLICIES)


def _deflating(place: str, years: range) -> CountedConversion:
    """A constant-price conversion whose deflator for `place` holds `years`."""
    return CountedConversion(
        tables=RecipeTables(
            rate_years={},
            deflator_years={place: frozenset(years)},
            base_year_min=years[0],
            base_year_max=years[-1],
        ),
        recipe=RECIPE_BODIES["ConversionRecipeDAC"],
        natives=frozenset({"USD"}),
        base_year=None,
        deflator_role="observationAbout",
        target_role=None,
        named=(),
    )


_GAPPED = [2000, 2001, 2002, 2005, 2006]
_ZETA = "country/ZZZ"


@pytest.mark.parametrize(
    ("series", "years", "conversion", "stated"),
    [
        pytest.param(
            {_PLACE: [_facet("a", _GAPPED)]},
            None,
            None,
            YearsWithData(start_year=2000, end_year=2006, gap_years=[2003, 2004]),
            id="dated-gap",
        ),
        pytest.param(
            {_PLACE: [_facet("a", _GAPPED), _facet("b", [2003, 2004], dated=False)]},
            None,
            None,
            YearsWithData(start_year=2000, end_year=2006),
            id="undated-facet-covers-the-gap",
        ),
        pytest.param(
            {_PLACE: [_facet("a", _GAPPED), _facet("b", [2008, 2009], dated=False)]},
            None,
            None,
            YearsWithData(start_year=2000, end_year=2009, gap_years=[2003, 2004, 2007]),
            id="dated-gaps-beside-an-undated-span",
        ),
        pytest.param(
            {_PLACE: [_facet("a", _GAPPED)], _ZETA: [_facet("z", [1990, 1991])]},
            TimeWindow(start_year=2001, end_year=2002),
            None,
            YearsWithData(start_year=2000, end_year=2006, gap_years=[2003, 2004]),
            id="series-outside-the-window",
        ),
        pytest.param(
            {_PLACE: [_facet("a", _GAPPED)]},
            None,
            _deflating(_PLACE, range(2001, 2007)),
            YearsWithData(start_year=2001, end_year=2006, gap_years=[2003, 2004]),
            id="conversion-keeps-its-deflator-years",
        ),
        pytest.param({_PLACE: [_facet("a", _GAPPED)]}, None, UNDECIDED, None, id="undecided"),
    ],
)
@pytest.mark.parametrize("index", [_EMPTY_INDEX, None], ids=["index", "no_index"])
def test_a_standard_plan_states_the_years_its_facets_hold(
    series: dict[str, list[Facet]],
    years: TimeWindow | None,
    conversion: CountConversion,
    stated: YearsWithData | None,
    index,
) -> None:
    """A facet listing its dates keeps its gaps, a facet listing none counts every year of
    its own span, and the years are their union. A series with no point under the window
    counts no year, a conversion keeps its deflator's years, and an undecidable conversion
    states none. The facet metadata alone counts them, with or without a coverage index."""
    plan = _standard_plan(series, years=years)

    assert answer_years(plan, index, conversion=conversion) == stated


def test_a_standard_series_with_an_unknown_bound_states_no_years() -> None:
    """A facet whose first date is unknown leaves the years unknown, also beside a facet
    whose dates are known."""
    facet = Facet(
        obs_count=4,
        earliest_date=None,
        latest_date="2020",
        facet_id="a",
        provenance_id=_PROVENANCE,
        observation_period="P1Y",
    )
    plan = _standard_plan({_PLACE: [facet, _facet("b", _GAPPED)]})
    assert {f.facet_id for f in plan.held[0].group} == {"a", "b"}

    assert answer_years(plan, _EMPTY_INDEX, conversion=None) is None


_FLOW = "Amount_Flow"
_FRA, _DEU, _KEN = "country/FRA", "country/DEU", "country/KEN"
_FLOW_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")


@pytest.mark.parametrize(
    ("conversion", "omitted", "stated"),
    [
        pytest.param(
            None, (), YearsWithData(start_year=1995, end_year=2020, gap_years=[2001]), id="both"
        ),
        pytest.param(
            CountedConversion(
                tables=RecipeTables(
                    rate_years={}, deflator_years={}, base_year_min=0, base_year_max=0
                ),
                recipe=RECIPE_BODIES["ConversionRecipeDAC"],
                natives=frozenset({"USD"}),
                base_year=None,
                deflator_role=None,
                target_role="financingProvider",
                named=frozenset({_FRA}),
            ),
            (),
            YearsWithData(start_year=2000, end_year=2020, gap_years=[2001]),
            id="conversion-names-one-donor",
        ),
        pytest.param(
            None,
            ((_FLOW, "financingProvider", _DEU),),
            YearsWithData(start_year=2000, end_year=2020, gap_years=[2001]),
            id="donor-omitted",
        ),
        pytest.param(UNDECIDED, (), None, id="undecided"),
    ],
)
@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_role_plan_states_the_years_its_index_rows_hold(
    search_settings, bulk: bool, conversion: CountConversion, omitted: tuple, stated
) -> None:
    """France's flows to Kenya hold 2000, 2002 to 2020 and Germany's 1995 to 1999, read
    under 1998 to 2012. The years run past both ends of the window, less a donor the
    conversion names no currency for or leaves out, and none under an undecidable
    conversion. A plan with no coverage index states none."""
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    years = [2000, *range(2002, 2021)]
    index = build_coverage_index(
        [
            (_FLOW, "f1", _FRA, _KEN, years),
            (_FLOW, "f1", _DEU, _KEN, list(range(1995, 2000))),
        ],
        role_names={_FLOW: _FLOW_ROLES},
    )
    plan = _role_plan(
        [_FLOW],
        member_facts={},
        names={"destination": "financingRecipient", "origin": "financingProvider"},
        subject_dcids=[_KEN],
        origin_dcids=[_FRA, _DEU],
        bound_roles={"recipient": frozenset({_KEN})},
        years=role_years(
            DateRequest(window=TimeWindow(start_year=1998, end_year=2012), latest=False),
            index.horizon,
        ),
        coverage_index=index,
        declared=NO_DECLARED_POLICIES,
    ).omitting(frozenset(omitted))

    assert isinstance(plan, BulkPlan) is bulk
    assert answer_years(plan, index, conversion=conversion) == stated
    assert answer_years(plan, None, conversion=conversion) is None


def test_a_role_plan_reading_a_member_coverage_lacks_states_no_years() -> None:
    """A plan reading a second member the index does not hold states no years, since the
    mint reads that member from SDMX and the years could start or end elsewhere."""
    index = build_coverage_index(
        [(_FLOW, "f1", _FRA, _KEN, [2000, 2001])],
        role_names={_FLOW: _FLOW_ROLES, "Amount_Uncovered": _FLOW_ROLES},
    )
    plan = _role_plan(
        [_FLOW, "Amount_Uncovered"],
        member_facts={},
        names={"destination": "financingRecipient", "origin": "financingProvider"},
        subject_dcids=[_KEN],
        origin_dcids=[_FRA],
        bound_roles={"recipient": frozenset({_KEN})},
        years=role_years(
            DateRequest(window=TimeWindow(start_year=1995, end_year=2020), latest=False),
            index.horizon,
        ),
        coverage_index=index,
        declared=NO_DECLARED_POLICIES,
    )

    assert plan.members == (_FLOW, "Amount_Uncovered")
    assert answer_years(plan, index, conversion=None) is None
