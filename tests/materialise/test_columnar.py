"""Differential tests compare columnar rules with row-level reference functions.
A seeded corpus covers null roles, irregular periods, tied latest rows, and
edge-case observation values.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pyarrow as pa
import pytest

from dc_sdmx import SdmxObservation
from dc_sdmx.errors import SdmxResponseError
from materialise import columnar
from materialise.facets import CoverageAdmission, FacetSelection, PlaceAdmission
from materialise.window import AppliedWindow, Coverage, RequestedWindow, _leading_year
from series_coverage.schema import RoleNames

from . import _row_oracle
from ._table_support import table_from_cells, table_from_observations

_ROLES = ("recipient",)
_STAT_VAR = "dcid:MalariaGrantDisbursements"

# This corpus covers the values where `pc.cast` and `float()` disagree,
# plus ordinary decimals, which `observed_values` must also parse the same
# way both functions do.
_VALUE_CORPUS = (
    "1",
    "1.5",
    "-2",
    "1e3",
    "1E3",
    " 5 ",
    "5 ",
    "1_000",
    "nan",
    "NaN",
    "-nan",
    "inf",
    "-inf",
    "Infinity",
    "",
    "NotApplicable",
    "1,000",
    "0x10",
    "+3",
    ".5",
    "5.",
    "1e400",
    "-0",
    "123.456",
    "-987.65",
    "42",
)


def _row(
    *,
    role_values: dict[str, str],
    time_period: str | None,
    value: str,
    unit: str | None = None,
    measurement_method: str | None = None,
    provenance: str | None = None,
    facet_id: str | None = None,
) -> SdmxObservation:
    return SdmxObservation(
        variable=_STAT_VAR,
        role_values=role_values,
        unit=unit,
        measurement_method=measurement_method,
        observation_period=None,
        provenance=provenance,
        time_period=time_period,
        value=value,
        scaling_factor=None,
        facet_id=facet_id,
    )


def _table(rows: tuple[SdmxObservation, ...]) -> pa.Table:
    return table_from_observations(rows, roles=_ROLES)


# ---------------------------------------------------------------------------
# Pure Python row-level reference implementations of the rules the columnar
# functions below are compared against, written independently of the
# production implementation.
# ---------------------------------------------------------------------------


def _reference_has_observed_value(row: SdmxObservation) -> bool:
    """A row carries a real observation when `OBS_VALUE` is non-empty and
    parses as a float, NaN and infinities included.
    """
    if row.value == "":
        return False
    try:
        float(row.value)
    except ValueError:
        return False
    return True


def _reference_leading_year(time_period: str | None) -> int | None:
    """`int(time_period[:4])`, `None` on a `None` input or an unparseable one."""
    if time_period is None:
        return None
    try:
        return int(time_period[:4])
    except ValueError:
        return None


def _reference_row_in_window(row: SdmxObservation, window: RequestedWindow) -> bool:
    """A row whose leading year is unparseable or missing passes (fails
    open). A bounded year fails below `window.start_year` or above
    `window.end_year`.
    """
    year = _reference_leading_year(row.time_period)
    if year is None:
        return True
    if window.start_year is not None and year < window.start_year:
        return False
    return not (window.end_year is not None and year > window.end_year)


_AXIS_FIELD = {
    "provenance": "provenance",
    "measurementMethod": "measurement_method",
    "unit": "unit",
}


def _reference_row_matches_selections(
    row: SdmxObservation, selections: Mapping[str, FacetSelection]
) -> bool:
    """Every axis in `selections` agrees with `row`'s own field on that axis,
    a null cell equal to a `None` ref.
    """
    return all(
        getattr(row, _AXIS_FIELD[axis]) == selection.ref for axis, selection in selections.items()
    )


def _reference_coverage(
    rows: Sequence[SdmxObservation],
    *,
    declared_roles: Sequence[str],
    applied_window: AppliedWindow | None,
) -> Coverage:
    """Row count, one `{label, count}` per declared role with the count of
    distinct non-null dcids, then `years` with the count of distinct
    parseable leading years, then the applied window entry, plus the same
    per-label distinct-value sets `entity_refs` carries.
    """
    dimensions: list[dict[str, object]] = []
    entity_refs: dict[str, frozenset[str | int]] = {}
    for role in declared_roles:
        distinct_entities = {row.role_values[role] for row in rows if role in row.role_values}
        dimensions.append({"label": role, "count": len(distinct_entities)})
        entity_refs[role] = frozenset(distinct_entities)
    distinct_years = {
        year for row in rows if (year := _reference_leading_year(row.time_period)) is not None
    }
    dimensions.append({"label": "years", "count": len(distinct_years)})
    entity_refs["years"] = frozenset(distinct_years)
    window_entry = applied_window.as_trace_entry() if applied_window is not None else None
    return Coverage(
        rows=len(rows),
        dimensions=tuple(dimensions),
        window=window_entry,
        entity_refs=entity_refs,
    )


def _reference_served_entities_by_role(rows: Sequence[SdmxObservation]) -> dict[str, set[str]]:
    """The distinct entity dcid served in each role across `rows`."""
    served: dict[str, set[str]] = {}
    for row in rows:
        for role, value in row.role_values.items():
            served.setdefault(role, set()).add(value)
    return served


def _reference_served_year_bounds(rows: Sequence[SdmxObservation]) -> tuple[int, int] | None:
    """The min and max of `_leading_year(row.time_period)` over `rows`,
    nulls skipped, `None` when no row carries a parseable leading year.
    """
    years = [year for row in rows if (year := _leading_year(row.time_period)) is not None]
    if not years:
        return None
    return min(years), max(years)


# ---------------------------------------------------------------------------
# observed_values against per-value float(), over the full edge corpus.
# ---------------------------------------------------------------------------


def _float_or_none(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _assert_float_equal(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    if math.isnan(expected):
        assert math.isnan(actual)
        return
    assert actual == expected


def test_observed_values_matches_float_over_the_full_corpus():
    values = pa.array(_VALUE_CORPUS, type=pa.string())
    result = columnar.observed_values(values).to_pylist()

    for raw, actual in zip(_VALUE_CORPUS, result, strict=True):
        _assert_float_equal(actual, _float_or_none(raw))


def test_observed_values_specific_edge_cases():
    cases = {
        " 5 ": 5.0,
        "1_000": 1000.0,
        "nan": math.nan,
        "inf": math.inf,
        "": None,
        "1,000": None,
    }
    values = pa.array(list(cases.keys()), type=pa.string())
    result = columnar.observed_values(values).to_pylist()

    for expected, actual in zip(cases.values(), result, strict=True):
        _assert_float_equal(actual, expected)


def test_observed_mask_matches_the_observed_value_rule():
    rows = tuple(
        _row(role_values={"recipient": "country/TGO"}, time_period="2015", value=raw)
        for raw in _VALUE_CORPUS
    )
    table = _table(rows)

    mask = columnar.observed_mask(table).to_pylist()
    expected = [_reference_has_observed_value(row) for row in rows]

    assert mask == expected


# ---------------------------------------------------------------------------
# leading_years against window._leading_year, over odd strings.
# ---------------------------------------------------------------------------


def test_leading_years_matches_leading_year_on_odd_strings():
    periods = ("2015", "2015-12", "2015-12-25", "20a5", "", "0001", "9999", "abcd", "2015x")
    table = _table(
        tuple(
            _row(role_values={"recipient": "country/TGO"}, time_period=p, value="1")
            for p in periods
        )
    )

    years = columnar.leading_years(table).to_pylist()
    expected = [_leading_year(p) for p in periods]

    assert years == expected


# ---------------------------------------------------------------------------
# latest_rows: ties and an all-unparseable input.
# ---------------------------------------------------------------------------


def test_latest_rows_keeps_every_row_tied_at_the_latest_period():
    tied_a = _row(role_values={"recipient": "country/TGO"}, time_period="2020", value="1")
    tied_b = _row(role_values={"recipient": "country/USA"}, time_period="2020", value="2")
    earlier = _row(role_values={"recipient": "country/KEN"}, time_period="2019", value="3")
    rows = (earlier, tied_a, tied_b)

    result = columnar.latest_rows(_table(rows))
    oracle = _row_oracle._latest_rows(rows)

    assert result.equals(_table(oracle))
    assert result.num_rows == 2


@pytest.mark.parametrize(
    ("year", "kept", "served"),
    [
        pytest.param(None, ["2020", "2020"], ["2020", "2020"], id="no-year-elected"),
        pytest.param(2019, ["2019", "2020", "2020"], ["2019"], id="an-earlier-year"),
        pytest.param(2020, ["2020", "2020"], ["2020", "2020"], id="the-newest-year"),
        pytest.param(2018, ["2020", "2020"], ["2020", "2020"], id="a-year-no-row-holds"),
    ],
)
def test_latest_rows_keep_the_elected_year_beside_the_latest_period(
    year: int | None, kept: list[str], served: list[str]
):
    """Two series end in 2020 and one in 2019. A StatVar that elected 2019 keeps that row
    beside the two at the newest period, and serves it alone. One that elected a year no row
    holds, or no year, serves the two at the newest period."""
    rows = (
        _row(role_values={"recipient": "country/KEN"}, time_period="2019", value="3"),
        _row(role_values={"recipient": "country/TGO"}, time_period="2020", value="1"),
        _row(role_values={"recipient": "country/USA"}, time_period="2020", value="2"),
    )

    beside = columnar.latest_rows(_table(rows), year=year)

    assert beside["TIME_PERIOD"].to_pylist() == kept
    assert columnar.elected_rows(beside, year=year)["TIME_PERIOD"].to_pylist() == served


def test_year_mask_keeps_the_rows_of_one_year():
    rows = tuple(
        _row(role_values={"recipient": "country/TGO"}, time_period=period, value="1")
        for period in ("2019", "2020-06", None, "20a5", "2020")
    )

    assert columnar.year_mask(_table(rows), year=2020).to_pylist() == [
        False,
        True,
        False,
        False,
        True,
    ]


def test_latest_rows_all_unparseable_input_is_empty():
    rows = (
        _row(role_values={"recipient": "country/TGO"}, time_period="20a5", value="1"),
        _row(role_values={"recipient": "country/USA"}, time_period=None, value="2"),
        _row(role_values={"recipient": "country/KEN"}, time_period="", value="3"),
    )

    result = columnar.latest_rows(_table(rows))
    oracle = _row_oracle._latest_rows(rows)

    assert oracle == ()
    assert result.num_rows == 0
    assert result.schema == _table(rows).schema


def test_latest_rows_empty_table_is_empty():
    empty_table = _table(())
    result = columnar.latest_rows(empty_table)
    assert result.num_rows == 0
    assert result.schema == empty_table.schema


# ---------------------------------------------------------------------------
# selection_mask with a None ref.
# ---------------------------------------------------------------------------


def _selection(*, axis: str, ref: str | None) -> FacetSelection:
    return FacetSelection(
        axis=axis,
        ref=ref,
        basis="bound",
        default_policy=None,
        default_policy_version=None,
        selection_reason=None,
        dispatch_facet_id="facet1",
    )


def test_selection_mask_none_ref_matches_a_null_cell_only():
    row_with_provenance = _row(
        role_values={"recipient": "country/TGO"},
        time_period="2015",
        value="1",
        provenance="dc/base",
    )
    row_without_provenance = _row(
        role_values={"recipient": "country/USA"},
        time_period="2015",
        value="1",
        provenance=None,
    )
    rows = (row_with_provenance, row_without_provenance)
    selections = {"provenance": _selection(axis="provenance", ref=None)}

    mask = columnar.selection_mask(_table(rows), selections=selections).to_pylist()
    expected = [_reference_row_matches_selections(row, selections) for row in rows]

    assert mask == [False, True]
    assert mask == expected


def test_selection_mask_no_axes_matches_every_row():
    rows = (
        _row(role_values={"recipient": "country/TGO"}, time_period="2015", value="1"),
        _row(role_values={"recipient": "country/USA"}, time_period="2016", value="2"),
    )

    mask = columnar.selection_mask(_table(rows), selections={}).to_pylist()

    assert mask == [True, True]


# ---------------------------------------------------------------------------
# A continued method: selection_mask admits it, continuation_mask keeps it past the
# served series.
# ---------------------------------------------------------------------------


def _continued(**overrides: object) -> FacetSelection:
    return replace(
        _selection(axis="measurementMethod", ref=None), continued_by="Estimated", **overrides
    )


def _gdp(place: str, year: str, method: str | None) -> SdmxObservation:
    return _row(
        role_values={"recipient": place},
        time_period=year,
        value="1",
        measurement_method=method,
        provenance="provenance/IMFWorldEconomicOutlook",
    )


def test_selection_mask_admits_the_continuing_method_beside_the_ref():
    rows = (
        _gdp("country/KEN", "2024", None),
        _gdp("country/KEN", "2025", "Estimated"),
        _gdp("country/KEN", "2025", "Modelled"),
    )

    mask = columnar.selection_mask(_table(rows), selections={"measurementMethod": _continued()})

    assert mask.to_pylist() == [True, True, False]


@pytest.mark.parametrize(
    ("rows", "kept"),
    [
        # Each place's estimates serve only the years after its own reported series ends.
        pytest.param(
            (
                _gdp("country/KEN", "2024", None),
                _gdp("country/KEN", "2025", "Estimated"),
                _gdp("country/FRA", "2025", None),
                _gdp("country/FRA", "2025", "Estimated"),
                _gdp("country/FRA", "2026", "Estimated"),
            ),
            [True, True, True, False, True],
            id="reported-wins-a-year-both-hold",
        ),
        # An estimate before the reported series' last year fills no gap in it.
        pytest.param(
            (
                _gdp("country/KEN", "2020", None),
                _gdp("country/KEN", "2021", "Estimated"),
                _gdp("country/KEN", "2022", None),
                _gdp("country/KEN", "2023", "Estimated"),
            ),
            [True, False, True, True],
            id="no-estimate-inside-the-reported-span",
        ),
        # A place with no reported row in the read keeps every estimate.
        pytest.param(
            (_gdp("country/KEN", "2025", "Estimated"), _gdp("country/KEN", "2026", "Estimated")),
            [True, True],
            id="estimates-only",
        ),
    ],
)
def test_continuation_mask_keeps_estimates_past_each_series_reported_end(rows, kept):
    table = _table(rows)

    assert (
        columnar.continuation_mask(
            table, selections={"measurementMethod": _continued()}
        ).to_pylist()
        == kept
    )
    # Without a continuation every row passes.
    plain = {"measurementMethod": _selection(axis="measurementMethod", ref=None)}
    assert columnar.continuation_mask(table, selections=plain).to_pylist() == [True] * len(rows)


# ---------------------------------------------------------------------------
# split_by_variable: prefixed and bare variables, and an unknown variable.
# ---------------------------------------------------------------------------


def test_split_by_variable_routes_prefixed_and_bare_forms_and_drops_unknown():
    batch = pa.RecordBatch.from_pydict(
        {"variableMeasured": ["dcid:Count_Person", "Count_Person", "Other_Var", "GDP"]}
    )

    result = columnar.split_by_variable(batch, index_by_bare_stat_var={"Count_Person": 0, "GDP": 1})

    assert result[0].to_pydict() == {"variableMeasured": ["dcid:Count_Person", "Count_Person"]}
    assert result[1].to_pydict() == {"variableMeasured": ["GDP"]}


def test_split_by_variable_empty_mapping_yields_no_entries():
    batch = pa.RecordBatch.from_pydict({"variableMeasured": ["Count_Person"]})

    assert columnar.split_by_variable(batch, index_by_bare_stat_var={}) == {}


# ---------------------------------------------------------------------------
# served_year_bounds: the min/max of leading_years, nulls skipped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param((), id="empty-table"),
        pytest.param(
            (
                _row(role_values={"recipient": "country/TGO"}, time_period=None, value="1"),
                _row(role_values={"recipient": "country/USA"}, time_period="", value="2"),
                _row(role_values={"recipient": "country/KEN"}, time_period="n/a", value="3"),
            ),
            id="null-or-unparseable-periods",
        ),
    ],
)
def test_served_year_bounds_is_none_without_parseable_periods(
    rows: tuple[SdmxObservation, ...],
):
    assert columnar.served_year_bounds(_table(rows)) is None


def test_served_year_bounds_mixed_table_skips_nulls_and_unparseable():
    rows = (
        _row(role_values={"recipient": "country/TGO"}, time_period="2015", value="1"),
        _row(role_values={"recipient": "country/USA"}, time_period="2020-Q1", value="2"),
        _row(role_values={"recipient": "country/KEN"}, time_period="", value="3"),
        _row(role_values={"recipient": "country/TGO"}, time_period=None, value="4"),
        _row(role_values={"recipient": "country/USA"}, time_period="2018", value="5"),
    )

    assert columnar.served_year_bounds(_table(rows)) == (2015, 2020)


def test_coverage_carries_entity_ref_sets_alongside_the_counts():
    """`entity_refs` holds the same distinct values each `dimensions`
    count is drawn from: dcid strings for a declared role, leading-year
    ints for `"years"`. A collection's own coverage unions these sets
    across members.
    """
    rows = (
        _row(role_values={"recipient": "country/TGO"}, time_period="2015", value="1"),
        _row(role_values={"recipient": "country/USA"}, time_period="2015", value="2"),
        _row(role_values={"recipient": "country/TGO"}, time_period="2016", value="3"),
    )

    result = columnar.coverage(_table(rows), declared_roles=_ROLES, applied_window=None)

    assert result.entity_refs["recipient"] == frozenset({"country/TGO", "country/USA"})
    assert result.entity_refs["years"] == frozenset({2015, 2016})
    dims_by_label = {dim["label"]: dim["count"] for dim in result.dimensions}
    assert dims_by_label["recipient"] == len(result.entity_refs["recipient"])
    assert dims_by_label["years"] == len(result.entity_refs["years"])


def test_coverage_uses_arrow_deduplication_at_the_row_ceiling():
    row_count = 250_000
    string_type = pa.string()
    table = pa.table(
        {
            "recipient": pa.concat_arrays(
                [
                    pa.repeat(pa.scalar("country/TGO", type=string_type), 125_000),
                    pa.repeat(pa.scalar("country/USA", type=string_type), 100_000),
                    pa.repeat(pa.scalar(None, type=string_type), 25_000),
                ]
            ),
            "TIME_PERIOD": pa.concat_arrays(
                [
                    pa.repeat(pa.scalar("2015", type=string_type), 100_000),
                    pa.repeat(pa.scalar("2016-06", type=string_type), 100_000),
                    pa.repeat(pa.scalar("unparseable", type=string_type), 25_000),
                    pa.repeat(pa.scalar(None, type=string_type), 25_000),
                ]
            ),
        }
    )

    result = columnar.coverage(table, declared_roles=_ROLES, applied_window=None)

    assert result == Coverage(
        rows=row_count,
        dimensions=(
            {"label": "recipient", "count": 2},
            {"label": "years", "count": 2},
        ),
        window=None,
        entity_refs={
            "recipient": frozenset({"country/TGO", "country/USA"}),
            "years": frozenset({2015, 2016}),
        },
    )


# ---------------------------------------------------------------------------
# Seeded differential test: the table and row paths agree over many
# randomly generated inputs, across every function this module mirrors.
# ---------------------------------------------------------------------------

_ENTITIES = ("country/TGO", "country/USA", "country/KEN")
_PERIODS = (None, "", "20a5", "2015", "2015-12", "2015-12-25", "2016", "2016-06")
_UNITS: tuple[str | None, ...] = (None, "USD", "KES")
_METHODS: tuple[str | None, ...] = (None, "GrantEquivalent", "Census")
_PROVENANCES: tuple[str | None, ...] = (None, "OECD_DAC2A", "dc/base")


def _random_row(rng: random.Random) -> SdmxObservation:
    role_values = {}
    # A role cell is either bound, or omitted entirely (the NA-fallback /
    # absent-column case `_parse_row` produces).
    if rng.random() < 0.85:
        role_values["recipient"] = rng.choice(_ENTITIES)
    if rng.random() < 0.3:
        # An undeclared role: present in `role_values`, but not in `_ROLES`,
        # so `table_from_observations(roles=_ROLES)` never builds a column
        # for it. Both paths must ignore it identically.
        role_values["donor"] = rng.choice(_ENTITIES)
    return _row(
        role_values=role_values,
        time_period=rng.choice(_PERIODS),
        value=rng.choice(_VALUE_CORPUS),
        unit=rng.choice(_UNITS),
        measurement_method=rng.choice(_METHODS),
        provenance=rng.choice(_PROVENANCES),
        facet_id=rng.choice((None, "f1", "f2")),
    )


def _random_selections(rng: random.Random, rows: tuple[SdmxObservation, ...]) -> dict:
    axis_by_field = {
        "provenance": "provenance",
        "measurementMethod": "measurement_method",
        "unit": "unit",
    }
    selections = {}
    for axis, field in axis_by_field.items():
        if rng.random() < 0.5:
            continue
        if rows and rng.random() < 0.5:
            ref = getattr(rng.choice(rows), field)
        else:
            ref = rng.choice((None, "no-such-value"))
        selections[axis] = _selection(axis=axis, ref=ref)
    return selections


def test_differential_against_reference_implementations():
    rng = random.Random(20260904)
    for _ in range(200):
        rows = tuple(_random_row(rng) for _ in range(rng.randint(0, 12)))
        table = _table(rows)

        # observed_mask
        observed = columnar.observed_mask(table).to_pylist()
        assert observed == [_reference_has_observed_value(row) for row in rows]

        # range_window_mask
        start_year = rng.choice((None, 2014, 2015, 2016))
        end_year = rng.choice((None, 2015, 2016, 2017))
        window = RequestedWindow(kind="range", start_year=start_year, end_year=end_year)
        range_mask = columnar.range_window_mask(table, window=window).to_pylist()
        assert range_mask == [_reference_row_in_window(row, window) for row in rows]

        # served_year_bounds
        assert columnar.served_year_bounds(table) == _reference_served_year_bounds(rows)

        # latest_rows
        latest_table = columnar.latest_rows(table)
        latest_oracle = _row_oracle._latest_rows(rows)
        assert latest_table.equals(_table(latest_oracle))

        # latest_rows and elected_rows, for a StatVar that elected a year
        year = rng.choice((2014, 2015, 2016, 2017))
        beside = columnar.latest_rows(table, year=year)
        elected_oracle = _row_oracle._latest_rows(rows, year=year)
        assert columnar.elected_rows(beside, year=year).equals(_table(elected_oracle))
        assert columnar.elected_rows(beside, year=None).equals(beside)
        assert {*latest_table["TIME_PERIOD"].to_pylist()} <= {*beside["TIME_PERIOD"].to_pylist()}

        # selection_mask
        selections = _random_selections(rng, rows)
        selection_mask = columnar.selection_mask(table, selections=selections).to_pylist()
        assert selection_mask == [
            _reference_row_matches_selections(row, selections) for row in rows
        ]

        # coverage
        applied_window = AppliedWindow(
            kind="range", start_year=start_year, end_year=end_year, origin="query"
        )
        coverage = columnar.coverage(table, declared_roles=_ROLES, applied_window=applied_window)
        coverage_reference = _reference_coverage(
            rows, declared_roles=_ROLES, applied_window=applied_window
        )
        assert coverage == coverage_reference

        # served_entities_by_role: the table only ever carries declared-role
        # columns, so the reference's output is narrowed to `_ROLES` before
        # comparing. An undeclared role key in a hand-built row has no
        # column to land in on the table side.
        served = columnar.served_entities_by_role(table, roles=_ROLES)
        served_reference = _reference_served_entities_by_role(rows)
        served_reference = {
            role: served_reference[role] for role in _ROLES if role in served_reference
        }
        assert served == served_reference


def test_admission_mask_keeps_a_row_only_where_its_facet_was_admitted_at_its_place():
    cells = [
        {"observationAbout": "country/P1", "facetId": "fB"},
        {"observationAbout": "country/P2", "facetId": "fB"},
        {"observationAbout": "country/P2", "facetId": "fD"},
    ]
    table = table_from_cells(cells, roles=("observationAbout",))
    admission = PlaceAdmission(places_by_facet={"fB": frozenset({"country/P2"})})

    mask = columnar.admission_mask(table, admission=admission)

    assert mask.to_pylist() == [False, True, False]


class _CountedPairs(dict[str, frozenset[tuple[str, str]]]):
    """`pairs_by_facet` counting its lookups."""

    lookups = 0

    def get(self, key, default=None):
        self.lookups += 1
        return super().get(key, default)


@pytest.mark.parametrize(
    ("role_names", "admitted_pair"),
    [
        pytest.param(
            RoleNames(donor="donor", recipient="recipient"),
            ("country/NOR", "country/KEN"),
            id="both",
        ),
        pytest.param(
            RoleNames(donor=None, recipient="recipient"), ("", "country/KEN"), id="recipient"
        ),
        pytest.param(RoleNames(donor="donor", recipient=None), ("country/NOR", ""), id="donor"),
    ],
)
def test_scoped_coverage_admission_keeps_only_its_facets_admitted_role_pairs(
    role_names, admitted_pair
):
    """A shared facet id cannot let a source-scoped flow serve another source's pair. A role
    the variable lacks reads as "", and each distinct facet and pair is looked up once."""
    cells = [
        {"facetId": "f-source-b", "donor": "country/NOR", "recipient": "country/KEN"},
        {"facetId": "f-source-b", "donor": "country/USA", "recipient": "country/UGA"},
        {"facetId": "f-source-a", "donor": "country/NOR", "recipient": "country/KEN"},
    ]
    table = table_from_cells(
        [{role: cell[role] for role in ("facetId", *role_names.declared)} for cell in cells * 3],
        roles=role_names.declared,
    )
    pairs = _CountedPairs({"f-source-b": frozenset({admitted_pair})})
    admission = CoverageAdmission(pairs_by_facet=pairs, role_names=role_names)

    mask = columnar.admission_mask(table, admission=admission)

    assert mask.to_pylist() == [True, False, False] * 3
    assert pairs.lookups == 3


def test_scoped_two_role_coverage_admission_refuses_a_missing_declared_role_column():
    """A stale two-role admission cannot treat a recipient-only SDMX response as admitted."""
    table = table_from_cells(
        [{"facetId": "f-source-b", "recipient": "country/KEN"}],
        roles=("recipient",),
    )
    admission = CoverageAdmission(
        pairs_by_facet={"f-source-b": frozenset({("country/NOR", "country/KEN")})},
        role_names=RoleNames(donor="donor", recipient="recipient"),
    )

    with pytest.raises(SdmxResponseError, match="without donor"):
        columnar.admission_mask(table, admission=admission)


@pytest.mark.parametrize(
    ("missing", "cell"),
    [
        # The parser fills an absent `facetId` column with nulls, so this is also the
        # absent-column case.
        ("facetId", {"observationAbout": "country/P2", "facetId": None}),
        # The parser maps a `NotApplicable` place to null.
        ("observationAbout", {"observationAbout": None, "facetId": "fB"}),
    ],
)
def test_admission_mask_refuses_a_row_it_cannot_check(missing: str, cell: Mapping[str, str | None]):
    cells = [{"observationAbout": "country/P2", "facetId": "fB"}, cell]
    table = table_from_cells(cells, roles=("observationAbout",))
    admission = PlaceAdmission(places_by_facet={"fB": frozenset({"country/P2"})})

    with pytest.raises(SdmxResponseError, match=f"1 of 2 SDMX rows carry no {missing}"):
        columnar.admission_mask(table, admission=admission)


@pytest.mark.parametrize(
    ("admission", "expected"),
    [
        # Place admission refuses a row naming no facet or no place, so both reach its mask.
        pytest.param(
            PlaceAdmission(places_by_facet={"fA": frozenset({"country/KEN"})}),
            ["2019", "2020", "2021", "2022"],
            id="place",
        ),
        # Coverage admission keeps every row, so a row naming no facet stays and a row
        # naming another facet, with or without a place, goes.
        pytest.param(CoverageAdmission(), ["2019", "2020", "2021"], id="coverage"),
    ],
)
def test_dispatched_rows_keeps_a_members_own_facets_and_rows_its_mask_refuses(admission, expected):
    # A group chunk fetches the union of its members' facet ids, so a facet only another
    # member dispatched comes back for this member's variable and must not serve.
    batch = pa.RecordBatch.from_pydict(
        {
            "facetId": ["fA", "fQ", None, "fA", "fQ"],
            "observationAbout": ["country/KEN", "country/KEN", "country/KEN", "country/KEN", None],
            "TIME_PERIOD": ["2019", "2019-01", "2020", "2021", "2022"],
        }
    )

    kept = columnar.dispatched_rows(batch, facet_ids=("fA",), admission=admission)

    assert kept.column("TIME_PERIOD").to_pylist() == expected
