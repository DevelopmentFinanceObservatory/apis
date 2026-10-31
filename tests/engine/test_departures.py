"""Departure facts read off the Spec a part answers with and the operations it carries."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from search.engine.assemble import build_no_data_part, part_departure_facts
from search.engine.clients.graph import Facet
from search.engine.departures import (
    breakdown_not_applied,
    measure_not_applied,
    merged_entity_facts,
    price_basis_not_applied,
    qualifier_property,
    scope_limited_fact,
    spec_departure_facts,
    stated_facts,
    surviving_facts,
    warning_key,
    window_gap_fact,
    without_request_words,
)
from search.engine.entry.raw_text import _dropped_phrase_fact
from search.engine.retrieval.read_plan import WindowGap, data_years, plan_standard, served_dates
from search.engine.spec_build import with_members_left_out
from search.heading.departure_fact import DepartureFact, UnservedMembers, ordered_facts
from search.heading.words import content_words
from search.models import (
    BindingSet,
    BindingValue,
    DateRange,
    EntityGroup,
    GraphRef,
    IdentityWindowAll,
    IdentityWindowLatest,
    IdentityWindowRange,
    ObservationSeries,
    ObservationSeriesConversion,
    Operation,
    ResolvedEntity,
    ResolvedFacet,
    ResolvedFilter,
    SeriesEntityRef,
    SeriesFacetRef,
    SeriesStatVarRef,
    Slot,
    SlotKey,
    SlotValue,
    Spec,
    SpecObservations,
    TimeWindow,
    Warning,
)
from tests.engine._assemble_fixtures import _make_spec
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import build_observations_planned, materialise_double_from_obs
from tests.engine.test_observations_window_and_label import _recorded_and_estimated_obs
from tests.fixtures import FakeGraph


def _spec(
    *,
    served: Sequence[tuple[str, str]] = (("1973", "2024"),),
    window: IdentityWindowAll | IdentityWindowLatest | IdentityWindowRange | None = None,
    method: str | None = None,
    groups: dict[str, EntityGroup] | None = None,
    data_years: tuple[int, int] | None = None,
) -> Spec:
    """A Spec whose StatVars serve `served`, asked over `window`, with `method` as its one
    measurementMethod and `groups` as its entity groups. Its data holds `data_years`, by
    default the years `served` spans, as an unclipped read serves them."""
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    members = [
        spec.resolved.members[0].model_copy(
            update={"data_date_range": DateRange(start=start, end=end)}
        )
        for start, end in served
    ]
    resolved_update: dict[str, object] = {"entity_groups": groups}
    if method is not None:
        resolved_update["facets"] = {
            "measurementMethod": ResolvedFacet(basis="fallback", ref=method, label=method.lower())
        }
    identity = spec.definition.identity.model_copy(
        update={"requested_window": window or IdentityWindowAll()}
    )
    if data_years is None and served:
        data_years = (
            min(int(start[:4]) for start, _end in served),
            max(int(end[:4]) for _start, end in served),
        )
    return spec.model_copy(
        update={
            "definition": spec.definition.model_copy(update={"identity": identity}),
            "resolved": spec.resolved.model_copy(update=resolved_update)
            .stating_data_years(data_years)
            .stating_members(members),
        }
    )


_GAPPY = Facet(
    facet_id="fCRS",
    provenance_id="dc/base/CRS",
    obs_count=14,
    earliest_date="2010",
    latest_date="2024",
    dates=("2010", *(str(year) for year in range(2012, 2025))),
    dates_complete=True,
)
"""A bilateral pair series holding 2010, then 2012 to 2024: no point in 2011."""


@pytest.mark.parametrize(
    ("start", "served", "expected"),
    [
        # 2011 lies inside the data's years, so no year departs, though the read serves
        # 2012 to 2015 and would otherwise say the data starts in 2012.
        (2011, ("2012", "2015"), []),
        # 2005 lies before the data starts, in 2010.
        (
            2005,
            ("2010", "2015"),
            [
                DepartureFact(
                    code="year_unavailable",
                    binding_ref="when:",
                    asked="2005 to 2015",
                    shown="2010 to 2015",
                    requested_years=(2005, 2015),
                    data_years=(2010, 2024),
                )
            ],
        ),
    ],
)
def test_year_unavailable_states_where_the_data_starts_unclipped(start, served, expected):
    """The read plan clips each StatVar's served span to the window (`served_dates`); the
    years the data holds come from its unclipped span (`read_plan.data_years`)."""
    plan = plan_standard(
        {("Count_Person", "country/ETH"): [_GAPPY]},
        members=["Count_Person"],
        years=TimeWindow(start_year=start, end_year=2015),
        places=["country/ETH"],
    )
    spec = _spec(
        served=[served_dates(plan)["Count_Person"]],
        window=IdentityWindowRange(start_year=start, end_year=2015),
        data_years=data_years(plan),
    )

    assert spec.resolved.members[0].data_date_range == DateRange(start=served[0], end=served[1])
    assert spec_departure_facts(spec, operations=()) == expected


@pytest.mark.parametrize(
    ("window", "served", "asked", "shown"),
    [
        # A named year after the data ends shows the whole span.
        (IdentityWindowRange(start_year=2030, end_year=2030), [("1973", "2024")], "2030", None),
        # A range reaching past the served span shows the years it overlaps; StatVars
        # disagreeing on range give the earliest start to the latest end.
        (
            IdentityWindowRange(start_year=1960, end_year=2020),
            [("1973-01", "2020"), ("1980", "2024")],
            "1960 to 2020",
            "1973 to 2020",
        ),
        (IdentityWindowRange(start_year=1950), [("1973", "2024")], "1950 or later", None),
    ],
)
def test_year_unavailable_for_a_named_window_outside_the_served_span(window, served, asked, shown):
    facts = spec_departure_facts(_spec(window=window, served=served), operations=())

    assert facts == [
        DepartureFact(
            code="year_unavailable",
            binding_ref="when:",
            asked=asked,
            shown=shown or "1973 to 2024",
            requested_years=(window.start_year, window.end_year),
            data_years=(1973, 2024),
        )
    ]


@pytest.mark.parametrize(
    ("years", "first", "last", "asked", "data_years"),
    [
        pytest.param((2020, 2026), "2010", "2030", "2020 to 2026", (2010, 2030), id="inside"),
        pytest.param((1875, 1875), "1960", "2023", "1875", (1960, 2023), id="before"),
        pytest.param((2035, None), "2000-06", "2004-06", "2035 or later", (2000, 2004), id="open"),
        pytest.param((2010, 2012), "2020", "2020", "2010 to 2012", (2020, 2020), id="one-date"),
        pytest.param((2020, 2025), None, "2019", "2020 to 2025", None, id="no-stated-span"),
    ],
)
def test_a_window_gap_is_year_unavailable_showing_nothing(years, first, last, asked, data_years):
    """A no-data answer for years its series holds no point in departs as `year_unavailable`
    with the years asked, the years the series runs, and nothing shown."""
    start, end = years
    gap = WindowGap(years=TimeWindow(start_year=start, end_year=end), first=first, last=last)

    assert window_gap_fact(gap) == DepartureFact(
        code="year_unavailable",
        binding_ref="when:",
        asked=asked,
        requested_years=years,
        data_years=data_years,
    )
    assert window_gap_fact(None) is None


@pytest.mark.parametrize(
    "window",
    [
        IdentityWindowAll(),
        IdentityWindowLatest(),
        IdentityWindowRange(start_year=2015, end_year=2020),
        IdentityWindowRange(end_year=2024),
    ],
)
def test_no_year_fact_for_a_window_inside_the_served_span(window):
    assert spec_departure_facts(_spec(window=window), operations=()) == []


_FORECAST_NOT_RUN = DepartureFact(
    code="operation_not_applied", binding_ref=None, asked="in 2030", verb="forecast"
)


@pytest.mark.parametrize(
    ("served", "expected"),
    [
        # The data ends in 2024: the year fact states the forecast's year.
        (
            (("1973", "2024"),),
            [
                DepartureFact(
                    code="year_unavailable",
                    binding_ref="when:",
                    asked="2030",
                    shown="1973 to 2024",
                    requested_years=(2030, 2030),
                    data_years=(1973, 2024),
                )
            ],
        ),
        # A candidates or no-data part reads no Spec, so no year fact states it.
        (None, [_FORECAST_NOT_RUN]),
        # A target year inside the served span raises no year fact.
        ((("1973", "2035"),), [_FORECAST_NOT_RUN]),
        # A Spec stating no served span raises no year fact.
        ((), [_FORECAST_NOT_RUN]),
    ],
)
def test_a_forecast_departs_as_its_year_only_when_a_year_fact_states_it(served, expected):
    """`served` is the span of the part's Spec, `None` for a part with no Spec."""
    spec = _spec(served=served) if served is not None else None
    forecast = Operation(verb="forecast", phrase="in 2030", target_year=2030)

    assert spec_departure_facts(spec, operations=[forecast]) == expected


@pytest.mark.parametrize(
    ("verb", "applied"),
    [("share_of", False), ("rank", False), ("growth", False), ("report", True)],
)
def test_operation_not_applied_for_an_operation_search_does_not_run(verb, applied):
    operation = Operation(verb=verb, phrase="share of ODA going to Kenya out of")

    facts = spec_departure_facts(_spec(), operations=[operation])

    expected = DepartureFact(
        code="operation_not_applied",
        binding_ref=None,
        asked="share of ODA going to Kenya out of",
        verb=verb,
    )
    assert facts == ([] if applied else [expected])


@pytest.mark.parametrize(("method", "modelled"), [("Modelled", True), ("GrantEquivalent", False)])
def test_modelled_values_from_the_served_measurement_method(method, modelled):
    facts = spec_departure_facts(_spec(method=method), operations=())

    expected = DepartureFact(code="modelled_values", binding_ref=None, shown="modelled")
    assert facts == ([expected] if modelled else [])


@pytest.mark.parametrize(
    ("methods", "projected", "modelled"),
    [
        (("Modelled", "Reported"), 0, True),
        (("Reported", None), 0, False),
        # A conversion on a projected rate or deflator makes its points estimates.
        (("Reported", "Reported"), 2, True),
    ],
)
def test_modelled_values_when_some_series_are_modelled(methods, projected, modelled):
    """Series that select different methods fire the departure as partial when any of
    them is modelled, and so does a series holding points converted on projected rates."""
    conversion = ObservationSeriesConversion(
        recipe_id="recipe/Test",
        table_provenance=SeriesFacetRef(ref="provenance/TestRateTables"),
        exchange_rate_table_version="2026-01-01T00:00:00Z",
        fallback_count=0,
        projected_count=projected,
    )
    series = [
        ObservationSeries(
            stat_var=SeriesStatVarRef(ref="Count_Person"),
            entities=[SeriesEntityRef(role="observationAbout", ref="country/ETH")],
            points=[],
            total_obs=0,
            facet={"measurementMethod": SeriesFacetRef(ref=method)} if method else {},
            conversion=conversion if index == 0 else None,
        )
        for index, method in enumerate(methods)
    ]
    spec = _spec()
    spec = spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={
                    "facets": {
                        "measurementMethod": ResolvedFacet(basis="fallback", selection="per_series")
                    },
                    "observations": SpecObservations(series=series, total_obs=0, truncated=False),
                }
            )
        }
    )

    facts = spec_departure_facts(spec, operations=())

    expected = DepartureFact(code="modelled_values", binding_ref=None, partial=True)
    assert facts == ([expected] if modelled else [])


@pytest.mark.parametrize(("end_year", "modelled"), [(2027, True), (2023, False)])
def test_modelled_values_when_estimates_continue_the_reported_series(end_year, modelled):
    """Kenya's reported series runs to 2024 and an `Estimated` facet of its source to 2032
    (`_recorded_and_estimated_obs`). A window from 2020 to 2027 serves the estimates for 2025
    to 2027 under the reported facet, so the departure fires as partial. A window inside the
    reported years serves no estimate and departs nothing."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_recorded_and_estimated_obs())
    observations, _warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=TimeWindow(start_year=2020, end_year=end_year),
        coverage=empty_coverage_snapshot(),
    )
    assert observations is not None and observations.series
    spec = _spec()
    spec = spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"observations": observations})}
    )

    facts = spec_departure_facts(spec, operations=())

    expected = DepartureFact(code="modelled_values", binding_ref=None, partial=True)
    assert facts == ([expected] if modelled else [])


@pytest.mark.parametrize(
    ("role", "binding_ref"), [("destination", "where:"), ("origin", "origin:")]
)
@pytest.mark.parametrize(
    ("entity_words", "asked"),
    [
        # crs-006 "aid from EU institutions to Ethiopia": the request's own words.
        ({"European Union": "EU institutions"}, "EU institutions"),
        # A request that named the group by its label.
        ({}, "European Union"),
        # A door holding no words for any entity, as a resubmit, takes the label as asked.
        # The part then leaves it out (`without_request_words`).
        (None, "European Union"),
    ],
)
def test_group_read_as_members_asks_in_the_request_s_words(role, binding_ref, entity_words, asked):
    groups = {
        role: EntityGroup(
            ref="EuropeanUnion", label="European Union", entity_type="ContinentalUnion"
        )
    }

    facts = spec_departure_facts(_spec(groups=groups), operations=(), entity_words=entity_words)

    assert [
        (fact.code, fact.binding_ref, fact.asked, fact.group, fact.group_dcid) for fact in facts
    ] == [("group_read_as_members", binding_ref, asked, "European Union", "EuropeanUnion")]


@pytest.mark.parametrize(
    ("fact", "stated"),
    [
        pytest.param(
            DepartureFact(code="group_read_as_members", binding_ref="where:", asked="Africa"),
            DepartureFact(code="group_read_as_members", binding_ref="where:"),
            id="request-words-dropped",
        ),
        pytest.param(
            DepartureFact(code="measure_not_applied", binding_ref="how:x", asked="loans"),
            DepartureFact(code="measure_not_applied", binding_ref="how:x", values=("loans",)),
            id="measure-value-moves-to-values",
        ),
        pytest.param(
            DepartureFact(
                code="member_unavailable",
                binding_ref="source:",
                asked="Aid flow B",
                members=("Aid flow B",),
            ),
            DepartureFact(
                code="member_unavailable",
                binding_ref="source:",
                asked="Aid flow B",
                members=("Aid flow B",),
            ),
            id="spec-members-stand",
        ),
    ],
)
def test_a_door_without_request_words_asks_only_for_what_the_spec_names(
    fact: DepartureFact, stated: DepartureFact
):
    """A resubmit holds no words of the request, so its part states no `asked` the words
    gave. `member_unavailable` asks for the Spec's own members, which no request words name,
    so it keeps them."""
    assert without_request_words(fact) == stated


_AFRICA_GROUP = EntityGroup(ref="africa", label="Africa", entity_type="Continent")


def _served_from(spec: Spec, provenance: str) -> Spec:
    facets = {"provenance": ResolvedFacet(basis="fallback", ref=provenance, label=provenance)}
    return spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"facets": facets})})


def _stated_left_out(spec: Spec, role: str) -> dict[str, object]:
    """`left_out` and `sum_differs` as the wire states the group `spec` reads on `role` once
    they are derived (`spec_build.with_members_left_out`), each absent when it is."""
    groups = with_members_left_out(spec).resolved.entity_groups
    assert groups is not None
    stated = groups[role].model_dump(mode="json")
    return {key: stated[key] for key in ("left_out", "sum_differs") if key in stated}


@pytest.mark.parametrize(
    ("role", "provenance", "excluded"),
    [
        pytest.param(
            "destination",
            "provenance/OECDCreditorReportingSystem",
            (
                "Africa, unallocated by country (DAC)",
                "Eastern Africa, unallocated by country (DAC)",
                "Middle Africa, unallocated by country (DAC)",
                "North of Sahara, unallocated by country (DAC)",
                "South of Sahara, unallocated by country (DAC)",
                "Southern Africa, unallocated by country (DAC)",
                "Western Africa, unallocated by country (DAC)",
            ),
            id="a-flow-dataset-publishing-the-residuals",
        ),
        pytest.param("destination", "provenance/WorldBankIDS", (), id="a-dataset-publishing-none"),
        # "aid from African countries to Kenya": DAC publishes the residuals as recipients.
        pytest.param(
            "origin", "provenance/OECDCreditorReportingSystem", (), id="a-role-publishing-none"
        ),
        pytest.param(
            "observationAbout",
            "provenance/OECDCreditorReportingSystem",
            (),
            id="a-subject-role",
        ),
    ],
)
def test_a_group_read_as_members_names_the_residuals_its_total_counts(role, provenance, excluded):
    """Africa read as its members leaves out the regional amounts DAC's Africa total counts,
    and names those the served dataset publishes on the role the group fills. The Spec's
    group states the same labels (`EntityGroup.left_out`), absent where it leaves none out."""
    spec = _served_from(_spec(groups={role: _AFRICA_GROUP}), provenance)

    (fact,) = spec_departure_facts(spec, operations=())

    assert (fact.code, fact.excluded, fact.sum_differs) == (
        "group_read_as_members",
        excluded,
        False,
    )
    assert _stated_left_out(spec, role) == ({"left_out": list(excluded)} if excluded else {})


def test_a_summed_group_counts_its_members_alone_and_names_the_residuals_it_binds():
    """Africa summed binds two of its residual buckets beside Ghana and Kenya. The group's
    size counts the two members, the bound residuals are named as counted, the rest of its
    total's residuals as left out, and an unserved residual departs as no member."""
    entities = [
        ResolvedEntity(ref="country/GHA", label="Ghana", role="destination"),
        ResolvedEntity(ref="country/KEN", label="Kenya", role="destination"),
        ResolvedEntity(
            ref="DACAfricaUnallocated",
            label="Africa, unallocated by country (DAC)",
            role="destination",
        ),
        ResolvedEntity(
            ref="DACEasternAfricaUnallocated",
            label="Eastern Africa, unallocated by country (DAC)",
            role="destination",
        ),
    ]
    spec = _served_from(
        _spec(groups={"destination": _AFRICA_GROUP}), "provenance/OECDCreditorReportingSystem"
    )
    resolved = spec.resolved.model_copy(update={"entities": entities}).stating_unserved(
        (("destination", "DACAfricaUnallocated"), ("destination", "country/KEN"))
    )

    unavailable, members = spec_departure_facts(
        spec.model_copy(update={"resolved": resolved}), operations=()
    )

    assert (unavailable.code, unavailable.entities, unavailable.group_size) == (
        "entity_unavailable",
        ("Kenya",),
        2,
    )
    assert (members.code, members.group_size, members.included, members.excluded) == (
        "group_read_as_members",
        2,
        (
            "Africa, unallocated by country (DAC)",
            "Eastern Africa, unallocated by country (DAC)",
        ),
        (
            "Middle Africa, unallocated by country (DAC)",
            "North of Sahara, unallocated by country (DAC)",
            "South of Sahara, unallocated by country (DAC)",
            "Southern Africa, unallocated by country (DAC)",
            "Western Africa, unallocated by country (DAC)",
        ),
    )


def test_a_group_with_no_node_names_the_residuals_of_the_total_its_role_keeps():
    """A resolver group ("Asia (DAC)") has no node to find its total by, so the role reading
    it as members keeps the total aside (`ResolvedSpec.group_totals`). CRS publishes every
    residual DAC's Asia total counts but the Mekong Delta Project."""
    spec = _served_from(
        _spec(groups={"destination": EntityGroup(label="Asia (DAC)")}),
        "provenance/OECDCreditorReportingSystem",
    )
    spec = spec.model_copy(
        update={"resolved": spec.resolved.stating_group_totals((("destination", "DACAsia"),))}
    )

    (fact,) = spec_departure_facts(spec, operations=())

    assert (fact.code, fact.excluded) == (
        "group_read_as_members",
        (
            "Asia, unallocated by country (DAC)",
            "Central Asia, unallocated by country (DAC)",
            "Far East Asia, unallocated by country (DAC)",
            "Middle East, unallocated by country (DAC)",
            "South and Central Asia, unallocated by country (DAC)",
            "South Asia, unallocated by country (DAC)",
        ),
    )


@pytest.mark.parametrize(
    ("role", "provenance", "sum_differs"),
    [
        pytest.param("origin", "provenance/OECDDAC1", True, id="a-dataset-publishing-the-total"),
        pytest.param(
            "origin",
            "provenance/OECDCreditorReportingSystem",
            False,
            id="a-dataset-publishing-none",
        ),
        pytest.param("destination", "provenance/OECDDAC1", False, id="a-role-publishing-none"),
    ],
)
def test_a_group_whose_published_total_claims_no_sum_says_the_members_can_differ(
    role, provenance, sum_differs
):
    """DAC1 publishes "DAC countries, total" as a donor, a figure the sum over the DAC's
    members can differ from (`PartyRow.sums_members` false). The DAC read as its members on a
    role a served dataset publishes that total on says so, and counts no residual."""
    spec = _served_from(
        _spec(groups={role: EntityGroup(ref="groups/OECD.DAC", label="DAC")}), provenance
    )
    spec = spec.model_copy(
        update={"resolved": spec.resolved.stating_group_totals(((role, "DACCountriesTotal"),))}
    )

    (fact,) = spec_departure_facts(spec, operations=())

    assert (fact.code, fact.excluded, fact.sum_differs) == (
        "group_read_as_members",
        (),
        sum_differs,
    )
    assert _stated_left_out(spec, role) == ({"sum_differs": True} if sum_differs else {})


def test_a_group_read_as_its_total_departs_nothing():
    group = _AFRICA_GROUP.model_copy(update={"total": "DACAfrica"})
    spec = _served_from(_spec(groups={"destination": group}), "provenance/OECDDAC2A")

    assert spec_departure_facts(spec, operations=()) == []
    assert with_members_left_out(spec) is spec


_SOURCE_SLOT = Slot(
    key=SlotKey(axis="source", label="source"),
    binding=BindingValue(
        value=SlotValue(ref=GraphRef(ref="dc/base/SrcA", label="Source A"), value_kind="source")
    ),
)
_SET_SLOT = Slot(
    key=SlotKey(
        axis="how",
        property=GraphRef(ref="financingSource", label="financing source"),
        label="source",
    ),
    binding=BindingSet(
        values=[
            SlotValue(ref=GraphRef(ref=ref, label=ref), value_kind="enum_value")
            for ref in ("SourceA", "SourceB")
        ]
    ),
)
_FLOW_B = GraphRef(ref="Flow_B", label="Aid flow B")


_USD = ResolvedFacet(basis="fallback", ref="CurrencyUSD", label="US Dollar", currency="USD")
_PERCENT = ResolvedFacet(basis="fallback", ref="Percent", label="Percent")
_SOURCE_A = ResolvedFacet(basis="fallback", ref="dc/base/SrcA", label="Source A")


@pytest.mark.parametrize(
    ("slots", "unserved", "facets", "fact"),
    [
        pytest.param(
            [_SOURCE_SLOT, _SET_SLOT],
            (_FLOW_B,),
            {"provenance": _SOURCE_A, "unit": _USD},
            DepartureFact(
                code="member_unavailable",
                binding_ref="source:",
                asked="Aid flow B",
                shown="Source A",
                members=("Aid flow B",),
                unit="US dollars",
            ),
            id="bound-source-currency-unit",
        ),
        pytest.param(
            [_SET_SLOT],
            (_FLOW_B, GraphRef(ref="Flow_C", label="")),
            {"unit": _PERCENT},
            DepartureFact(
                code="member_unavailable",
                binding_ref="how:financingSource",
                asked="Aid flow B, Flow_C",
                members=("Aid flow B", "Flow_C"),
                unit="Percent",
            ),
            id="set-binding-unit-by-label",
        ),
        pytest.param(
            [],
            (_FLOW_B,),
            {},
            DepartureFact(
                code="member_unavailable",
                binding_ref=None,
                asked="Aid flow B",
                members=("Aid flow B",),
            ),
            id="no-slot-no-facets",
        ),
        pytest.param([_SET_SLOT], (), {"unit": _USD}, None, id="every-member-served"),
    ],
)
def test_member_unavailable_names_the_members_the_group_leaves_out(
    slots: list[Slot],
    unserved: tuple[GraphRef, ...],
    facets: dict[str, ResolvedFacet],
    fact: DepartureFact | None,
):
    """The members holding data that the answer's one group leaves out, by label, else by
    dcid, on the source slot where the answer binds one, else on the set binding naming
    them, with the source and the unit the answer's served series carry: a currency by its
    plural noun, another unit by label. A group serving every member departs nothing."""
    spec = _spec()
    spec = spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"slots": slots, "facets": facets}
            ).stating_unserved_members(unserved),
        }
    )

    facts = [f for f in spec_departure_facts(spec, operations=()) if f.code == "member_unavailable"]

    assert facts == ([fact] if fact is not None else [])


def test_entity_unavailable_names_each_role_s_unserved_entities_on_its_token():
    """Unserved entities group by the token naming their role, in entity order; one with
    no label is named by its dcid. A served entity is not named."""
    entities = [
        ResolvedEntity(ref="country/KEN", label="Kenya", role="destination"),
        ResolvedEntity(ref="country/SOM", label="Somaliland", role="destination"),
        ResolvedEntity(ref="org/gates", label="Gates Foundation", role="origin"),
        ResolvedEntity(ref="org/unnamed", role="origin"),
    ]
    spec = _spec()
    resolved = spec.resolved.model_copy(update={"entities": entities}).stating_unserved(
        (("origin", "org/unnamed"), ("destination", "country/SOM"), ("origin", "org/gates"))
    )

    facts = spec_departure_facts(spec.model_copy(update={"resolved": resolved}), operations=())

    assert facts == [
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="Somaliland",
            entities=("Somaliland",),
        ),
        DepartureFact(
            code="entity_unavailable",
            binding_ref="origin:",
            asked="Gates Foundation, org/unnamed",
            entities=("Gates Foundation", "org/unnamed"),
        ),
    ]


@pytest.mark.parametrize(
    ("window", "coverage_window", "latest_years"),
    [
        # Ghana's series ends before 2024, the latest period Kenya's holds.
        (IdentityWindowLatest(), TimeWindow(start_year=2024, end_year=2024), (2024, 2024)),
        (IdentityWindowLatest(), TimeWindow(start_year=2019, end_year=2024), (2019, 2024)),
        # Under a range the years asked are the requested window's, stated elsewhere.
        (
            IdentityWindowRange(start_year=2015, end_year=2020),
            TimeWindow(start_year=2015, end_year=2020),
            None,
        ),
    ],
)
def test_entity_unavailable_on_a_latest_request_names_the_latest_period(
    window, coverage_window, latest_years
):
    entities = [
        ResolvedEntity(ref="country/KEN", label="Kenya", role="observationAbout"),
        ResolvedEntity(ref="country/GHA", label="Ghana", role="observationAbout"),
    ]
    spec = _spec(window=window)
    resolved = (
        spec.resolved.model_copy(
            update={
                "entities": entities,
                "coverage": spec.resolved.coverage.model_copy(update={"window": coverage_window}),
            }
        )
    ).stating_unserved((("observationAbout", "country/GHA"),))

    (fact,) = [
        fact
        for fact in spec_departure_facts(
            spec.model_copy(update={"resolved": resolved}), operations=()
        )
        if fact.code == "entity_unavailable"
    ]

    assert (fact.entities, fact.latest_years) == (("Ghana",), latest_years)


def test_entity_unavailable_on_a_group_asks_for_the_group_and_names_no_member():
    """crs-006 "aid from EU institutions to Ethiopia": the request named the group, never
    Bulgaria, Cyprus, or Latvia, so `asked` is the group and the fact counts the 27."""
    members = [
        ResolvedEntity(ref=f"country/{code}", label=label, role="origin")
        for code, label in (("AUT", "Austria"), ("BGR", "Bulgaria"), ("CYP", "Cyprus"))
    ]
    eth = ResolvedEntity(ref="country/ETH", label="Ethiopia", role="destination")
    group = EntityGroup(label="European Union", entity_type="ContinentalUnion")
    spec = _spec(groups={"origin": group})
    resolved = spec.resolved.model_copy(update={"entities": [*members, eth]}).stating_unserved(
        (("origin", "country/BGR"), ("origin", "country/CYP"))
    )

    (fact,) = [
        fact
        for fact in spec_departure_facts(
            spec.model_copy(update={"resolved": resolved}), operations=()
        )
        if fact.code == "entity_unavailable"
    ]

    assert (fact.binding_ref, fact.asked, fact.group_size) == ("origin:", "European Union", 3)
    assert fact.group_dcid is None
    assert fact.entities == ("Bulgaria", "Cyprus")


def test_an_entity_unserved_inside_a_group_counts_toward_the_group_alone():
    """mls-026 "How much does Norway give to the UN?": Norway gives to no one as a member
    of the 193 United Nations members, and is served as the donor. The group's count holds
    it; Norway the donor departs not at all."""
    members = [
        ResolvedEntity(ref=f"country/{code}", label=label, role="destination")
        for code, label in (("KEN", "Kenya"), ("NOR", "Norway"), ("SOM", "Somalia"))
    ]
    donor = ResolvedEntity(ref="country/NOR", label="Norway", role="origin")
    spec = _spec(groups={"destination": EntityGroup(label="United Nations")})
    resolved = spec.resolved.model_copy(update={"entities": [donor, *members]}).stating_unserved(
        (("destination", "country/NOR"), ("destination", "country/SOM"))
    )

    facts = [
        fact
        for fact in spec_departure_facts(
            spec.model_copy(update={"resolved": resolved}), operations=()
        )
        if fact.code == "entity_unavailable"
    ]

    assert [(f.binding_ref, f.entities, f.group_size) for f in facts] == [
        ("where:", ("Norway", "Somalia"), 3)
    ]


_AFRICA = EntityGroup(ref="africa", label="Africa", entity_type="Continent")
_ASIA = EntityGroup(ref="asia", label="Asia", entity_type="Continent")
_BALTIC = EntityGroup(label="Baltic states")
_PLACES = {
    "country/GHA": "Ghana",
    "country/TZA": "Tanzania",
    "country/KEN": "Kenya",
    "country/NPL": "Nepal",
    "country/JPN": "Japan",
    "country/NOR": "Norway",
}
_AFRICAN = frozenset({"country/GHA", "country/TZA", "country/KEN"})
# Two sets of one size that share Kenya.
_EAST = EntityGroup(ref="east", label="East")
_WEST = EntityGroup(ref="west", label="West")
# A set with no node, of the same size, that sorts first by label.
_NODELESS = EntityGroup(label="Coast")
_EAST_MEMBERS = frozenset({"country/KEN", "country/TZA"})
_WEST_MEMBERS = frozenset({"country/KEN", "country/GHA"})
_ASIAN = frozenset({"country/NPL", "country/JPN"})


@pytest.mark.parametrize(
    ("named_sets", "bound", "unserved", "asked", "entities", "sets"),
    [
        pytest.param(
            [(_AFRICA, _AFRICAN)],
            [*sorted(_AFRICAN), "country/NOR"],
            ["country/GHA", "country/TZA"],
            "Africa",
            (),
            (UnservedMembers("Africa", "africa", 3, "Africa", ("Ghana", "Tanzania")),),
            id="a region beside a place",
        ),
        pytest.param(
            [(_AFRICA, _AFRICAN)],
            [*sorted(_AFRICAN), "country/NOR"],
            ["country/GHA", "country/NOR"],
            "Norway, Africa",
            ("Norway",),
            (UnservedMembers("Africa", "africa", 3, "Africa", ("Ghana",)),),
            id="the place named beside a region is unserved too",
        ),
        pytest.param(
            [(_BALTIC, _AFRICAN)],
            [*sorted(_AFRICAN), "country/NOR"],
            ["country/KEN"],
            "the Baltics",
            (),
            (UnservedMembers("Baltic states", None, 3, "the Baltics", ("Kenya",)),),
            id="a resolver group beside a place",
        ),
        pytest.param(
            [(_AFRICA, _AFRICAN), (_ASIA, _ASIAN)],
            [*sorted(_AFRICAN), *sorted(_ASIAN)],
            ["country/GHA", "country/JPN", "country/TZA"],
            "Africa, Asia",
            (),
            (
                UnservedMembers("Africa", "africa", 3, "Africa", ("Ghana", "Tanzania")),
                UnservedMembers("Asia", "asia", 2, "Asia", ("Japan",)),
            ),
            id="two regions on one role",
        ),
        pytest.param(
            [(_ASIA, _ASIAN), (_AFRICA, _AFRICAN | _ASIAN)],
            [*sorted(_AFRICAN), *sorted(_ASIAN), "country/NOR"],
            ["country/JPN"],
            "Africa",
            (),
            (UnservedMembers("Africa", "africa", 5, "Africa", ("Japan",)),),
            id="a member of two sets counts toward the larger",
        ),
        pytest.param(
            [],
            [*sorted(_AFRICAN), "country/NOR"],
            ["country/GHA", "country/TZA"],
            "Ghana, Tanzania",
            ("Ghana", "Tanzania"),
            (),
            id="places named one by one",
        ),
        *(
            pytest.param(
                order,
                [*sorted(_AFRICAN), "country/NPL"],
                ["country/KEN"],
                "East",
                (),
                (UnservedMembers("East", "east", 2, "East", ("Kenya",)),),
                id=f"a member of two sets of one size counts toward the first by dcid ({label})",
            )
            for label, order in (
                ("as listed", [(_EAST, _EAST_MEMBERS), (_WEST, _WEST_MEMBERS)]),
                ("reversed", [(_WEST, _WEST_MEMBERS), (_EAST, _EAST_MEMBERS)]),
                ("a set with no node", [(_NODELESS, _WEST_MEMBERS), (_EAST, _EAST_MEMBERS)]),
            )
        ),
    ],
)
def test_entity_unavailable_counts_a_named_set_s_members_against_the_set(
    named_sets, bound, unserved, asked, entities, sets
):
    """A role binding every member of a set the request named beside other places carries
    no entity group. The set's members without values are counted under it, and only a
    place the request named one by one is named as the ask."""
    spec = _spec()
    resolved = (
        spec.resolved.model_copy(
            update={
                "entities": [
                    ResolvedEntity(ref=dcid, label=_PLACES[dcid], role="observationAbout")
                    for dcid in bound
                ]
            }
        )
        .stating_unserved(tuple(("observationAbout", dcid) for dcid in unserved))
        .stating_named_sets(
            tuple(("observationAbout", group, members) for group, members in named_sets),
            frozenset(),
        )
    )

    facts = spec_departure_facts(
        spec.model_copy(update={"resolved": resolved}),
        operations=(),
        entity_words={"Baltic states": "the Baltics"},
    )

    assert facts == [
        DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked=asked,
            entities=entities,
            sets=sets,
        )
    ]


def test_breakdown_over_a_flow_axis_is_cleared_by_that_role():
    assert breakdown_not_applied("origin", asked="by creditor") == DepartureFact(
        code="breakdown_not_applied", binding_ref="origin:", asked="by creditor", over="origin"
    )
    assert breakdown_not_applied("destination", asked=None).binding_ref == "where:"


def test_ordered_facts_follow_precedence_and_drop_repeats():
    direction = DepartureFact(code="direction_discarded", binding_ref="where:", asked="Kenya")
    year = DepartureFact(code="year_unavailable", binding_ref="when:", asked="2030")
    measure = DepartureFact(code="measure_not_applied", binding_ref="how:nettingBasis")

    assert ordered_facts([direction, year, measure, direction]) == (measure, year, direction)


def test_a_raised_fact_survives_only_while_the_part_holds_its_warning():
    kept = Warning(code="direction_discarded", severity="warn", message="Kenya was named.")
    dropped = Warning(code="query_qualifier_dropped", severity="degraded", message="net")
    direction = DepartureFact(code="direction_discarded", binding_ref="where:", asked="Kenya")
    measure = DepartureFact(code="measure_not_applied", binding_ref="how:nettingBasis")
    raised = {warning_key(kept): direction, warning_key(dropped): measure}

    # A copy stamped with a spec id keeps its key.
    stamped = kept.model_copy(update={"spec_id": "spec_a"})
    assert surviving_facts(raised, [kept, stamped]) == (direction,)


def test_part_facts_join_raised_and_read_facts_in_precedence_order():
    direction = DepartureFact(code="direction_discarded", binding_ref="where:", asked="Kenya")
    share = Operation(verb="share_of", phrase="share of")

    facts = part_departure_facts(_spec(method="Modelled"), raised=[direction], operations=[share])

    assert [fact.code for fact in facts] == [
        "operation_not_applied",
        "modelled_values",
        "direction_discarded",
    ]
    # A part with no Spec reads only its operations.
    assert [fact.code for fact in part_departure_facts(None, raised=(), operations=[share])] == [
        "operation_not_applied"
    ]


@pytest.mark.parametrize(
    ("named", "property_dcid"),
    [
        (["CRSTypeOfFlow10"], "crsTypeOfFlow"),
        (["CRSTypeOfFlow10", "CRSPurposeCode510"], None),
        (["NotACode"], None),
    ],
)
def test_qualifier_property_is_the_one_code_system_the_rows_share(named, property_dcid):
    assert qualifier_property([GraphRef(ref=dcid, label=dcid) for dcid in named]) == property_dcid


# The flagged match of crs-010 "Gates Foundation grants to Nigeria", as the panel recorded
# it: its how slot filters and the axes its measure noun ("gross disbursements") states.
_CRS_010_SLOT_FILTERS = [
    ("crsPurpose", "binder", []),
    ("crsTypeOfFinance", "binder", [("CRSTypeOfFinance100", "grant")]),
    ("crsTypeOfFlow", "declared_default", [("CRSTypeOfFlow10", "official development assistance")]),
    ("financialFlowType", "measure_elected", [("Disbursement", "disbursement")]),
    ("nettingBasis", "measure_elected", [("Gross", "gross")]),
    ("crsCooperationModality", "binder", []),
]
_CRS_010_MEASURE_LABEL_AXES = [
    "populationType",
    "measuredProperty",
    "measurementQualifier",
    "financialFlowType",
    "nettingBasis",
]


def _crs_010_flagged() -> Spec:
    """crs-010's flagged match: its how slot filters and its measure noun's axes."""
    spec = _spec()
    trace = spec.resolved.trace.model_copy(
        update={
            "slot_filters": [
                ResolvedFilter(
                    key=SlotKey(axis="how", property=GraphRef(ref=prop, label=prop), label=prop),
                    binding_kind="value" if refs else "unbound",
                    refs=[GraphRef(ref=dcid, label=label) for dcid, label in refs],
                    provenance=provenance,
                )
                for prop, provenance, refs in _CRS_010_SLOT_FILTERS
            ]
        }
    )
    resolved = spec.resolved.model_copy(
        update={"trace": trace, "measure_label_axes": _CRS_010_MEASURE_LABEL_AXES}
    )
    return spec.model_copy(update={"resolved": resolved})


def test_the_scope_limit_is_the_values_the_engine_supplied():
    """crs-010: the declared ODA default is the one limit. "grant" is the bind call's
    reading of "grants", and gross disbursements are the measure noun. The education
    sector sits on the StatVar, with the purpose slot open, so it is no limit."""
    assert scope_limited_fact(_crs_010_flagged()) == DepartureFact(
        code="scope_limited", binding_ref="how:crsTypeOfFlow", limits=("ODA",)
    )


def test_a_no_data_part_states_its_scope_limit_as_its_last_departure():
    """crs-010: the limit is a `scope_limited` departure after the part's other ones, and
    the plan's departure tokens are the `departures` texts."""
    part = build_no_data_part(
        "no_observations",
        variable_text="grants",
        warnings=[],
        flagged_match=_crs_010_flagged(),
        gate="no_member_has_data_at_place",
        operations=[Operation(verb="share_of", phrase="share of grants")],
    )

    assert [(d.code, d.binding_ref) for d in part.departures] == [
        ("operation_not_applied", None),
        ("scope_limited", "how:crsTypeOfFlow"),
    ]
    assert part.departures[-1].text == ("This check covered only ODA.")
    assert part.sentence_plan is not None
    assert [t.text for t in part.sentence_plan if t.segment == "departure" and t.text.strip()] == [
        d.text for d in part.departures
    ]


def test_a_limit_with_no_name_is_left_out():
    """A declared default the code index does not hold and the graph gave no name, so its
    label is its dcid, is no limit; a code-index short name still names one."""
    spec = _spec()
    trace = spec.resolved.trace.model_copy(
        update={
            "slot_filters": [
                ResolvedFilter(
                    key=SlotKey(axis="how", property=GraphRef(ref=prop, label=prop), label=prop),
                    binding_kind="value",
                    refs=[GraphRef(ref=dcid, label=label)],
                    provenance="declared_default",
                )
                for prop, dcid, label in [
                    ("crsPurpose", "DAC/Health", "DAC/Health"),
                    ("crsTypeOfFlow", "CRSTypeOfFlow10", "CRSTypeOfFlow10"),
                ]
            ]
        }
    )

    fact = scope_limited_fact(
        spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"trace": trace})})
    )

    assert fact is not None
    assert (fact.binding_ref, fact.limits) == ("how:crsTypeOfFlow", ("ODA",))


def _asks(phrase: str) -> DepartureFact:
    return measure_not_applied(asked=phrase, property_dcid=None)


@pytest.mark.parametrize(
    ("phrase", "fact"),
    [
        # Operation wording: mls-016, weo-021, ids-026.
        ("Trend in", None),
        ("compare the", None),
        ("countries have the highest external debt stocks", None),
        # Requests the answer could apply: mls-030, mls-033, weo-045, and "IMF" (weo-013),
        # which the part drops when its byline names the IMF.
        ("through regional development banks", _asks("through regional development banks")),
        (
            "not counting what goes through multilaterals",
            _asks("not counting what goes through multilaterals"),
        ),
        (
            "from the April 2025 World Economic Outlook",
            _asks("from the April 2025 World Economic Outlook"),
        ),
        ("IMF", _asks("IMF")),
        # mls-022: a price basis, cleared by the unit token.
        (
            "in constant 2022 dollars",
            price_basis_not_applied(asked="in constant 2022 dollars", shown=None),
        ),
    ],
)
def test_a_dropped_phrase_departs_unless_it_is_operation_wording(phrase, fact):
    assert _dropped_phrase_fact(phrase) == fact


_GROSS = measure_not_applied(asked="gross", property_dcid="nettingBasis")
_NET = measure_not_applied(asked="net", property_dcid="nettingBasis")
_CURRENT_PRICES = price_basis_not_applied(asked="current prices", shown=None)
# The recorded sentences of weo-013 and mls-030.
_WEO_013 = (
    "Sub-Saharan Africa's GDP, 1980 to 2025. Current US dollars · IMF World Economic Outlook (WEO)."
)
_MLS_030 = (
    "Norway's imputed multilateral ODA to 148 recipients for rural development, 2011 to "
    "2024. Current US dollars · ONE Imputed Multilateral Sectors."
)


@pytest.mark.parametrize(
    ("facts", "sentences", "stated"),
    [
        # weo-009: "gross" beside "Gross capital formation".
        ([_GROSS], ["Gross capital formation as a share of GDP in Ghana, 2000 to 2024."], []),
        # weo-013: residue, the byline names the IMF.
        ([_asks("IMF")], [_WEO_013], []),
        # mls-030: a request the sentence does not state.
        (
            [_asks("through regional development banks")],
            [_MLS_030],
            [_asks("through regional development banks")],
        ),
        # crs-037: one reading of a candidates part states "net", so the part does.
        ([_NET], ["Net ODA disbursements.", "Gross ODA disbursements."], []),
        ([_NET], ["Gross ODA disbursements.", "ODA flows."], [_NET]),
        # weo-004: one reading's byline names the IMF.
        (
            [_asks("IMF")],
            [
                "Kenya's GDP in Kenyan Shilling, 1980 to 2024. IMF World Economic Outlook (WEO).",
                "Kenya's GDP growth, 1961 to 2024. World Development Indicators.",
            ],
            [],
        ),
        # A part with no sentence keeps the qualifier.
        ([_GROSS], [], [_GROSS]),
        # weo-002, weo-045: a price basis the part's byline does not state.
        ([_CURRENT_PRICES], ["GDP."], []),
        (
            [dataclasses.replace(_CURRENT_PRICES, shown="current US dollars")],
            ["GDP."],
            [dataclasses.replace(_CURRENT_PRICES, shown="current US dollars")],
        ),
    ],
)
def test_a_part_states_only_the_facts_its_heading_and_byline_do_not(facts, sentences, stated):
    words = [content_words(sentence) for sentence in sentences]
    assert list(stated_facts(facts, stated=words)) == stated


def _unavailable(*entities: str, **fields) -> DepartureFact:
    return DepartureFact(
        code="entity_unavailable",
        binding_ref=fields.pop("binding_ref", "where:"),
        asked=fields.pop("asked", ", ".join(entities)),
        entities=entities,
        **fields,
    )


@pytest.mark.parametrize(
    ("facts", "merged"),
    [
        pytest.param(
            [_unavailable("Ghana", latest_years=(2024, 2024)), _unavailable("Togo", "Ghana")],
            [_unavailable("Ghana", "Togo", latest_years=(2024, 2024))],
            id="the_first_keeps_its_period_and_asks_for_every_entity",
        ),
        pytest.param(
            [
                _unavailable(
                    "Bulgaria", asked="EU institutions", group="European Union", group_size=27
                ),
                _unavailable("Cyprus", sources=("Eurostat",)),
            ],
            [
                _unavailable(
                    "Bulgaria",
                    "Cyprus",
                    asked="EU institutions",
                    group="European Union",
                    group_size=27,
                    sources=("Eurostat",),
                )
            ],
            id="a_group_keeps_the_request_s_words",
        ),
        pytest.param(
            [_unavailable("Ghana"), _unavailable("France", binding_ref="origin:")],
            [_unavailable("Ghana"), _unavailable("France", binding_ref="origin:")],
            id="each_role_departs_on_its_own",
        ),
        pytest.param(
            [
                _unavailable(
                    asked="Africa",
                    sets=(UnservedMembers("Africa", "africa", 59, "Africa", ("Ghana",)),),
                ),
                _unavailable(
                    "Norway",
                    asked="Norway, Africa",
                    sets=(UnservedMembers("Africa", "africa", 59, "Africa", ("Ghana", "Togo")),),
                ),
            ],
            [
                _unavailable(
                    "Norway",
                    asked="Norway, Africa",
                    sets=(UnservedMembers("Africa", "africa", 59, "Africa", ("Ghana", "Togo")),),
                )
            ],
            id="a_set_s_members_merge_under_the_set",
        ),
    ],
)
def test_a_role_s_unavailable_entities_merge_into_one_fact(facts, merged):
    assert merged_entity_facts(facts) == merged
