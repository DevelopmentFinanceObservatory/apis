"""A variable card's `availability`: the coverage summary folds a role-bearing variable, the facet
profile every other one, the default is the profile's election (P4), and a profile miss or a
countries-only scope leaves the gap it should."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from code_index.types import PartyRow
from graph_api.availability import (
    COUNTRIES_ONLY_GAP,
    FacetFilter,
    build_availability,
    facet_items,
    held_facets,
)
from materialise.facet_profile import FacetProfile, read_facet_profiles
from materialise.graph import Facet
from origin_catalogue import BASE, DFO, Catalogue, DataCommons, bound_catalogue
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

_MAX_BYTES = 2 * 1024 * 1024
_CRS = "provenance/OECDCreditorReportingSystem"
_WEO = "provenance/IMFWorldEconomicOutlook"
_WDI = "dc/base/WorldDevelopmentIndicators"
_DAC = "provenance/OECDDAC1"
_UNADMITTED = "provenance/NotInTheCatalogue"
_EDUCATION = (
    "Amount_FinancialFlow_Net_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110"
    "_CRSPurposeCode11110_Nominal"
)
_FLOW_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")


class _ProfileGraph(FakeGraphReader):
    """`FakeGraphReader` taking the facet read's byte cap."""

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )


class _NoGraph:
    """A graph and availability reader a coverage-read profile must never call."""

    def __getattr__(self, name: str):
        raise AssertionError(f"a role-bearing profile read the graph: {name}")


def _catalogue(
    *provenances: tuple[str, DataCommons], dfo_variables: Sequence[str] = ()
) -> Catalogue:
    return restrictive_catalogue(
        provenances=[provenance_row(pid, origin) for pid, origin in provenances],
        dfo_variables=dfo_variables,
    )


def _empty_coverage() -> CoverageIndex:
    return build_coverage_index([], role_names={})


def _profile_of(stat_var: str, *, catalogue: Catalogue, coverage: CoverageIndex, graph=None):
    graph = graph or _NoGraph()
    availability = graph.scope_availability() if isinstance(graph, FakeGraphReader) else graph
    with bound_catalogue(catalogue):
        return read_facet_profiles(
            [stat_var],
            graph=graph,
            availability=availability,
            coverage=coverage,
            max_bytes=_MAX_BYTES,
            deadline=None,
        )[stat_var]


def _build(stat_var, profile, *, coverage, catalogue, parties=None, currencies=None, kinds=None):
    kinds = kinds or {}
    assert isinstance(profile, FacetProfile), profile
    return build_availability(
        held=held_facets(
            stat_var=stat_var, profile=profile, coverage=coverage, catalogue=catalogue
        ),
        catalogue=catalogue,
        parties=parties or {},
        national_currencies=currencies or {},
        place_kind=lambda place: kinds.get(place, "country"),
        labels={"Estimated": "estimated"},
    )


# --- the design's CRS example, from the coverage summary ---------------------------------------


def _crs_coverage() -> tuple[CoverageIndex, dict[str, PartyRow]]:
    """The education example as the 8 October build holds it: 29 donors (the last five
    private), 166 recipient parties and 17 residual entries, 1990 to 2012, 6,604 observations. A second facet sits
    under a provenance the catalogue does not admit."""
    donors = [f"donor/D{i:02d}" for i in range(29)]
    recipients = [f"recipient/R{i:03d}" for i in range(183)]
    rows = []
    for k in range(366):
        length = 19 if k < 16 else 18
        start = 1990 + k % 5
        rows.append(
            (
                _EDUCATION,
                "fCRS",
                donors[k % 29],
                recipients[k // 2],
                list(range(start, start + length)),
            )
        )
    rows.append((_EDUCATION, "fOther", donors[0], recipients[0], [2020]))
    coverage = build_coverage_index(
        rows,
        role_names={_EDUCATION: _FLOW_ROLES},
        facet_metadata={
            "fCRS": FacetMetadata(provenance_id=_CRS, unit="CurrencyUSD", observation_period="P1Y"),
            "fOther": FacetMetadata(
                provenance_id=_UNADMITTED, unit="CurrencyUSD", observation_period="P1Y"
            ),
        },
    )
    parties = {
        d: PartyRow(dcid=d, kind="private" if i >= 24 else "party") for i, d in enumerate(donors)
    }
    parties.update(
        {
            r: PartyRow(dcid=r, kind="residual" if i >= 166 else "party")
            for i, r in enumerate(recipients)
        }
    )
    return coverage, parties


def test_the_crs_example_folds_from_the_coverage_summary():
    coverage, parties = _crs_coverage()
    catalogue = _catalogue((_CRS, DFO), dfo_variables=[_EDUCATION])
    profile = _profile_of(_EDUCATION, catalogue=catalogue, coverage=coverage)

    read = _build(_EDUCATION, profile, coverage=coverage, catalogue=catalogue, parties=parties)

    assert (read.gaps, read.transient) == ((), False)
    card = read.availability.model_dump(mode="json")
    assert card["summary"] == {
        "places": None,
        "entities": {
            "financingProvider": {
                "parties": 24,
                "private": 5,
                "totals": 0,
                "residual": 0,
                "unclassified": 0,
            },
            "financingRecipient": {
                "parties": 166,
                "private": 0,
                "totals": 0,
                "residual": 17,
                "unclassified": 0,
            },
        },
        "first_year": "1990",
        "last_year": "2012",
        "sources": 1,
        "facets": 1,
    }
    # Every kind count adds up to the role's distinct entities, private donors included.
    assert {role: sum(counts.values()) for role, counts in card["summary"]["entities"].items()} == {
        "financingProvider": 29,
        "financingRecipient": 183,
    }
    assert card["default"] == {
        "facets": ["fCRS"],
        "source": {"ref": _CRS, "label": _CRS, "documentation": f"/api/graph/documentation/{_CRS}"},
        "unit": {"ref": "CurrencyUSD", "label": "US dollars"},
        "measurement_method": None,
        "observation_period": "P1Y",
        "first_year": "1990",
        "last_year": "2012",
        "places": None,
        "without_data": 0,
    }
    (source,) = card["sources"]
    assert (
        source["source"]["ref"],
        source["facets"],
        source["first_year"],
        source["last_year"],
    ) == (
        _CRS,
        1,
        "1990",
        "2012",
    )
    assert source["series"] == [
        {
            "measurement_method": None,
            "observation_period": "P1Y",
            "scaling": None,
            "first_year": "1990",
            "last_year": "2012",
            "places": None,
            "observations": 6604,
            "units": {
                "count": 1,
                "leading": [{"ref": "CurrencyUSD", "label": "US dollars", "places": None}],
                "more": 0,
                "national_currency": None,
            },
            "binding": {"provenance": _CRS, "measurementMethod": None, "unit": "CurrencyUSD"},
        }
    ]
    assert card["more_sources"] == 0


# --- the WEO GDP fold ------------------------------------------------------------------------

_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_COUNTRIES = tuple(f"country/C{i:03d}" for i in range(197))
# C000 uses US dollars, C001-C025 the euro, C026-C195 139 other currencies (some shared), and
# C196 has no mapping.
_NATIONAL = {
    _COUNTRIES[0]: "USD",
    **dict.fromkeys(_COUNTRIES[1:26], "EUR"),
    **{place: f"X{(i - 26) % 139:02d}" for i, place in enumerate(_COUNTRIES[26:196], start=26)},
}


def _weo_series(method: str | None, *, reach: int, first: str, last: str, currencies: int):
    """One WEO series: US dollars at the first `reach` countries, and each national currency
    among the first `currencies` other ones (the euro first) at its countries in reach."""
    places_by_unit = {"CurrencyUSD": frozenset(_COUNTRIES[:reach])}
    codes = ["EUR", *(f"X{i:02d}" for i in range(currencies - 1))]
    for code in codes:
        places_by_unit[f"Currency{code}"] = frozenset(
            p for p in _COUNTRIES[:reach] if _NATIONAL.get(p) == code
        )
    facets = {
        unit: Facet(
            facet_id=f"weo-{method or 'none'}-{unit}",
            provenance_id=_WEO,
            measurement_method=method,
            unit=unit,
            observation_period="P1Y",
            obs_count=len(places) * 10,
            earliest_date=first,
            latest_date=last,
        )
        for unit, places in places_by_unit.items()
    }
    by_id = {facet.facet_id: facet for facet in facets.values()}
    return by_id, {facet.facet_id: places_by_unit[unit] for unit, facet in facets.items()}


def _weo_profile() -> FacetProfile:
    reported, reported_places = _weo_series(
        None, reach=197, first="1980", last="2025", currencies=140
    )
    estimated, estimated_places = _weo_series(
        "Estimated", reach=193, first="2019", last="2031", currencies=137
    )
    return FacetProfile(
        stat_var=_GDP,
        candidates=(*reported.values(), *estimated.values()),
        breadth={},
        default=(reported["weo-none-CurrencyUSD"],),
        places_by_facet={**reported_places, **estimated_places},
        places=_COUNTRIES,
    )


def test_the_weo_gdp_fold_counts_units_and_national_currencies():
    catalogue = _catalogue((_WEO, DFO), dfo_variables=[_GDP])

    read = _build(
        _GDP, _weo_profile(), coverage=_empty_coverage(), catalogue=catalogue, currencies=_NATIONAL
    )

    card = read.availability.model_dump(mode="json")
    assert card["summary"] == {
        "places": {"country": 197},
        "entities": None,
        "first_year": "1980",
        "last_year": "2031",
        "sources": 1,
        "facets": 279,
    }
    default = card["default"]
    assert (default["unit"]["ref"], default["measurement_method"], default["places"]) == (
        "CurrencyUSD",
        None,
        197,
    )
    assert (default["first_year"], default["last_year"], default["without_data"]) == (
        "1980",
        "2025",
        0,
    )
    (source,) = card["sources"]
    assert (source["places"], source["facets"]) == ({"country": 197}, 279)
    reported, estimated = source["series"]
    assert reported["measurement_method"] is None
    assert (reported["places"], reported["first_year"], reported["last_year"]) == (
        197,
        "1980",
        "2025",
    )
    assert reported["units"] == {
        "count": 141,
        "leading": [
            {"ref": "CurrencyUSD", "label": "US dollars", "places": 197},
            {"ref": "CurrencyEUR", "label": "euros", "places": 25},
        ],
        "more": 139,
        # US dollars cover more than the one country using them, so they stay plain.
        "national_currency": {"currencies": 140, "places": 195},
    }
    assert reported["binding"] == {
        "provenance": _WEO,
        "measurementMethod": None,
        "unit": "CurrencyUSD",
    }
    assert estimated["measurement_method"] == {"ref": "Estimated", "label": "estimated"}
    assert (estimated["places"], estimated["first_year"], estimated["last_year"]) == (
        193,
        "2019",
        "2031",
    )
    # The euro's 25 countries, and the 164 of C026-C192 whose currency is among X00-X135.
    assert (estimated["units"]["count"], estimated["units"]["national_currency"]) == (
        138,
        {"currencies": 137, "places": 189},
    )
    assert estimated["binding"]["measurementMethod"] == "Estimated"


@pytest.mark.parametrize(
    ("currencies", "national"),
    [
        pytest.param(
            {"country/A": "EUR", "country/B": "EUR"}, {"currencies": 1, "places": 2}, id="mapped"
        ),
        pytest.param({"country/A": "EUR"}, None, id="a-place-without-a-mapping"),
        pytest.param({"country/A": "EUR", "country/B": "XOF"}, None, id="another-currency"),
    ],
)
def test_a_unit_folds_as_national_only_when_every_place_it_covers_names_it(currencies, national):
    facet = Facet(
        facet_id="fEUR",
        provenance_id=_WEO,
        unit="CurrencyEUR",
        obs_count=2,
        earliest_date="2020",
        latest_date="2021",
    )
    profile = FacetProfile(
        stat_var=_GDP,
        candidates=(facet,),
        breadth={},
        default=(facet,),
        places_by_facet={"fEUR": frozenset({"country/A", "country/B"})},
    )

    read = _build(
        _GDP,
        profile,
        coverage=_empty_coverage(),
        catalogue=_catalogue((_WEO, DFO)),
        currencies=currencies,
    )

    (series,) = read.availability.sources[0].series
    assert series.units.count == 1
    assert (
        series.units.national_currency.model_dump() if series.units.national_currency else None
    ) == national


# --- a roleless variable at entities no place scan reaches (design acceptance case 8) --------

_EXCHANGE = "DACExchangeRate"
_ENTITIES = ("DACCountriesTotal", "CurrencyEUR", "CurrencyJPY")


def test_a_roleless_variable_shows_its_facets_at_dac_countries_total_and_currencies():
    facet = Facet(
        facet_id="fRate",
        provenance_id=_DAC,
        unit=None,
        obs_count=30,
        earliest_date="2000",
        latest_date="2024",
        observation_period="P1Y",
    )
    graph = _ProfileGraph(facets={(_EXCHANGE, entity): [facet] for entity in _ENTITIES})
    catalogue = _catalogue((_DAC, DFO), dfo_variables=[_EXCHANGE])
    profile = _profile_of(_EXCHANGE, catalogue=catalogue, coverage=_empty_coverage(), graph=graph)

    read = _build(
        _EXCHANGE,
        profile,
        coverage=_empty_coverage(),
        catalogue=catalogue,
        kinds={"DACCountriesTotal": "aggregate", "CurrencyEUR": "unit", "CurrencyJPY": "unit"},
    )

    assert (read.gaps, read.transient) == ((), False)
    card = read.availability
    assert card.summary.places == {"aggregate": 1, "unit": 2}
    assert [s.source.ref for s in card.sources] == [_DAC]
    assert card.default.facets == ["fRate"]
    assert (card.default.places, card.default.without_data) == (3, 0)


# --- gaps ------------------------------------------------------------------------------------


def test_a_countries_only_scope_is_a_permanent_gap_and_the_card_still_folds():
    facet = Facet(
        facet_id="fB",
        provenance_id=_WDI,
        unit="Person",
        obs_count=60,
        earliest_date="1960",
        latest_date="2024",
    )
    profile = FacetProfile(
        stat_var="Count_Person",
        candidates=(facet,),
        breadth={"fB": 2},
        default=(facet,),
        places_by_facet={"fB": frozenset({"country/A", "country/B"})},
        places=("country/A", "country/B"),
        countries_fallback=True,
    )

    read = _build(
        "Count_Person", profile, coverage=_empty_coverage(), catalogue=_catalogue((_WDI, BASE))
    )

    assert (read.gaps, read.transient) == ((COUNTRIES_ONLY_GAP,), False)
    assert read.availability is not None
    assert read.availability.summary.places == {"country": 2}


# --- P4: the card's default is the profile's ----------------------------------------------------

_POPULATION = "Count_Person"
_PLACES = ("country/P1", "country/P2", "country/P3")


def _population_facet(facet_id: str, provenance: str, obs_count: int) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method="Census",
        unit="Person",
        obs_count=obs_count,
        earliest_date="1960",
        latest_date="2025",
    )


def test_a_subject_only_cards_default_is_the_profiles_and_names_the_broader_base_source():
    """DFO holds one place of three and the base source all three, so the base facet serves the
    request naming nothing (design acceptance case 9 at small scale), and the card says so."""
    dfo = _population_facet("fD", _DAC, 60)
    base = _population_facet("fB", _WDI, 66)
    graph = _ProfileGraph(
        facets={
            (_POPULATION, _PLACES[0]): [dfo, base],
            **{(_POPULATION, p): [base] for p in _PLACES[1:]},
        }
    )
    catalogue = restrictive_catalogue(
        provenances=[provenance_row(_DAC, DFO), provenance_row(_WDI, BASE)],
        dfo_variables=[_POPULATION],
        base_variables={_POPULATION: ([_WDI], [])},
    )
    profile = _profile_of(_POPULATION, catalogue=catalogue, coverage=_empty_coverage(), graph=graph)

    read = _build(_POPULATION, profile, coverage=_empty_coverage(), catalogue=catalogue)

    default = read.availability.default
    assert default.facets == [f.facet_id for f in profile.default] == ["fB"]
    assert (default.source.ref, default.places, default.without_data) == (_WDI, 3, 0)
    assert [s.source.ref for s in read.availability.sources] == [_WDI, _DAC]


def test_a_role_bearing_cards_default_is_the_profiles_coverage_election():
    """`fWide` covers three recipients in one year and `fLong` one over eleven, so breadth
    elects `fWide`."""
    flows = "Amount_Flows"
    recipients = ("country/R1", "country/R2", "country/R3")
    coverage = build_coverage_index(
        [
            *((flows, "fWide", "country/DONOR", r, [2015]) for r in recipients),
            (flows, "fLong", "country/DONOR", recipients[0], list(range(2010, 2021))),
        ],
        role_names={flows: _FLOW_ROLES},
        facet_metadata={
            "fWide": FacetMetadata(provenance_id=_CRS, unit="USD", observation_period="P1Y"),
            "fLong": FacetMetadata(provenance_id=_DAC, unit="USD", observation_period="P1Y"),
        },
    )
    catalogue = _catalogue((_CRS, DFO), (_DAC, DFO), dfo_variables=[flows])
    profile = _profile_of(flows, catalogue=catalogue, coverage=coverage)

    read = _build(flows, profile, coverage=coverage, catalogue=catalogue)

    default = read.availability.default
    assert default.facets == [f.facet_id for f in profile.default] == ["fWide"]
    assert (default.source.ref, default.without_data) == (_CRS, 0)


# --- the `facets` list relation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "count", "first"),
    [
        pytest.param(FacetFilter(), 279, "weo-Estimated-CurrencyEUR", id="open"),
        pytest.param(FacetFilter(unit="CurrencyEUR"), 2, "weo-Estimated-CurrencyEUR", id="unit"),
        pytest.param(
            FacetFilter(measurement_method="Estimated"),
            138,
            "weo-Estimated-CurrencyEUR",
            id="method",
        ),
        pytest.param(
            FacetFilter(source=_WEO, unit="CurrencyUSD"),
            2,
            "weo-Estimated-CurrencyUSD",
            id="source-and-unit",
        ),
        pytest.param(FacetFilter(source=_WDI), 0, None, id="another-source"),
    ],
)
def test_facet_items_filter_by_source_unit_or_method_in_facet_id_order(where, count, first):
    catalogue = _catalogue((_WEO, DFO))
    items = facet_items(
        held=held_facets(
            stat_var=_GDP, profile=_weo_profile(), coverage=_empty_coverage(), catalogue=catalogue
        ),
        catalogue=catalogue,
        labels={},
        where=where,
    )

    refs = [item.ref for item in items]
    assert len(refs) == count
    assert refs == sorted(refs)
    assert (refs[0] if refs else None) == first
