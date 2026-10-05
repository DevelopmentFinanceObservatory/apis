"""Checks streamed groupby and pivot results against the whole-table implementations."""

from __future__ import annotations

import pytest

from materialise.errors import MaterialisationUsageError
from materialise.reshape import (
    ColumnClassification,
    PivotNotUniqueError,
    RollupNotSummableError,
    Table,
    groupby_sum,
    groupby_sum_chunks,
    pivot,
    pivot_chunks,
)

_GROUPBY_COLUMNS = (
    "stat_var",
    "recipient",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)


def _groupby_row(
    stat_var: str, recipient: str, year: str, value: float, *, unit: str = "USD"
) -> tuple[object, ...]:
    return (stat_var, recipient, year, value, unit, "GrantEquivalent", "OECD_DAC2A", None)


_GROUPBY_CLASSIFICATION = ColumnClassification(
    facet_axis_columns=frozenset({"unit", "measurementMethod", "provenance"}),
    non_droppable_columns=frozenset({"recipient", "time_period", "unit", "provenance"}),
)


def _chunks(
    columns: tuple[str, ...], rows: list[tuple[object, ...]], bounds: list[int]
) -> list[Table]:
    return [
        Table(columns=columns, rows=tuple(rows[bounds[i] : bounds[i + 1]]))
        for i in range(len(bounds) - 1)
    ]


class TestGroupbySumChunksDigest:
    """Checks streamed `stat_var` collapse against whole-table output across chunk boundaries."""

    def test_a_group_spanning_a_chunk_boundary_matches_the_whole_table_result(self):
        # KEN's SV_A and SV_B rows land in different chunks and must merge.
        rows = sorted(
            [
                _groupby_row("dcid:SV_A", "country/KEN", "2015", 10.0),
                _groupby_row("dcid:SV_A", "country/KEN", "2016", 20.0),
                _groupby_row("dcid:SV_A", "country/TZA", "2015", 30.0),
                _groupby_row("dcid:SV_B", "country/KEN", "2015", 5.0),
                _groupby_row("dcid:SV_B", "country/UGA", "2015", 7.0),
                _groupby_row("dcid:SV_B", "country/UGA", "2016", 8.0),
            ],
            key=lambda r: (r[0], r[1], r[2]),
        )
        whole = Table(columns=_GROUPBY_COLUMNS, rows=tuple(rows))
        expected = groupby_sum(
            whole,
            axis="stat_var",
            classification=_GROUPBY_CLASSIFICATION,
            summable_axes=frozenset({"stat_var"}),
            missing_declared_as_zero=True,
        )

        streamed = groupby_sum_chunks(
            iter(_chunks(_GROUPBY_COLUMNS, rows, [0, 2, 3, 5, len(rows)])),
            axis="stat_var",
            classification=_GROUPBY_CLASSIFICATION,
            summable_axes=frozenset({"stat_var"}),
            missing_declared_as_zero=True,
        )

        assert streamed == expected
        # Both contributors sum to 15.0 for KEN in 2015.
        assert ("country/KEN", "2015", 15.0, "USD", "GrantEquivalent", "OECD_DAC2A", None) in (
            streamed.rows
        )

    def test_a_single_chunk_matches_too(self):
        rows = sorted(
            [
                _groupby_row("dcid:SV_A", "country/KEN", "2015", 10.0),
                _groupby_row("dcid:SV_A", "country/KEN", "2016", 20.0),
            ],
            key=lambda r: (r[0], r[1], r[2]),
        )
        whole = Table(columns=_GROUPBY_COLUMNS, rows=tuple(rows))
        expected = groupby_sum(
            whole,
            axis="stat_var",
            classification=_GROUPBY_CLASSIFICATION,
            summable_axes=frozenset({"stat_var"}),
            missing_declared_as_zero=True,
        )
        streamed = groupby_sum_chunks(
            iter([whole]),
            axis="stat_var",
            classification=_GROUPBY_CLASSIFICATION,
            summable_axes=frozenset({"stat_var"}),
            missing_declared_as_zero=True,
        )
        assert streamed == expected

    def test_facet_homogeneity_still_refuses(self):
        # The conflicting units occur in separate chunks.
        rows = [
            _groupby_row("dcid:SV_A", "country/KEN", "2015", 10.0, unit="USD"),
            _groupby_row("dcid:SV_B", "country/KEN", "2015", 5.0, unit="EUR"),
        ]
        with pytest.raises(RollupNotSummableError) as exc_info:
            groupby_sum_chunks(
                iter(_chunks(_GROUPBY_COLUMNS, rows, [0, 1, 2])),
                axis="stat_var",
                classification=_GROUPBY_CLASSIFICATION,
                summable_axes=frozenset({"stat_var"}),
                missing_declared_as_zero=True,
            )
        assert exc_info.value.precondition == "facet_homogeneity"

    def test_requires_at_least_one_chunk(self):
        with pytest.raises(MaterialisationUsageError):
            groupby_sum_chunks(
                iter([]),
                axis="stat_var",
                classification=_GROUPBY_CLASSIFICATION,
                summable_axes=frozenset({"stat_var"}),
            )


_PRECEDENCE_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit")
_PRECEDENCE_CLASSIFICATION = ColumnClassification(
    facet_axis_columns=frozenset({"unit"}),
    non_droppable_columns=frozenset({"recipient", "time_period", "unit"}),
)

# Each fixture triggers two refusal conditions and is sorted by `row_sort_key`.
_PRECEDENCE_FIXTURES = {
    "unique_input_grain_vs_facet_homogeneity": (
        (
            ("dcid:SV_A", "country/KEN", "2015", 10.0, "USD"),
            ("dcid:SV_B", "country/KEN", "2015", 5.0, "EUR"),
            ("dcid:SV_C", "country/TZA", "2016", 7.0, "USD"),
            ("dcid:SV_C", "country/TZA", "2016", 7.0, "USD"),
        ),
        None,
    ),
    "unique_input_grain_vs_parent_child_overlap": (
        (
            ("dcid:SV_X", "country/KEN", "2015", 10.0, "USD"),
            ("dcid:SV_Y", "country/KEN", "2015", 5.0, "USD"),
            ("dcid:SV_Z", "country/TZA", "2016", 7.0, "USD"),
            ("dcid:SV_Z", "country/TZA", "2016", 7.0, "USD"),
        ),
        {"dcid:SV_Y": "dcid:SV_X"},
    ),
    "unique_input_grain_vs_missing_members": (
        (
            (None, "country/UGA", "2016", 5.0, "USD"),
            ("dcid:SV_A", "country/KEN", "2015", 10.0, "USD"),
            ("dcid:SV_C", "country/TZA", "2017", 7.0, "USD"),
            ("dcid:SV_C", "country/TZA", "2017", 7.0, "USD"),
        ),
        None,
    ),
    "facet_homogeneity_vs_parent_child_overlap": (
        # KEN's first-seen group has the hierarchy violation; TZA's group has the facet conflict.
        # Interleaved rows keep KEN first in whole-table scan order.
        (
            ("dcid:SV_A", "country/KEN", "2015", 10.0, "USD"),
            ("dcid:SV_B", "country/TZA", "2016", 10.0, "USD"),
            ("dcid:SV_X", "country/KEN", "2015", 20.0, "USD"),
            ("dcid:SV_Y", "country/TZA", "2016", 20.0, "EUR"),
        ),
        {"dcid:SV_X": "dcid:SV_A"},
    ),
    "facet_homogeneity_vs_missing_members": (
        (
            (None, "country/UGA", "2016", 5.0, "USD"),
            ("dcid:SV_A", "country/KEN", "2015", 10.0, "USD"),
            ("dcid:SV_B", "country/KEN", "2015", 5.0, "EUR"),
        ),
        None,
    ),
    "parent_child_overlap_vs_missing_members": (
        (
            (None, "country/UGA", "2016", 5.0, "USD"),
            ("dcid:SV_X", "country/KEN", "2015", 10.0, "USD"),
            ("dcid:SV_Y", "country/KEN", "2015", 5.0, "USD"),
        ),
        {"dcid:SV_Y": "dcid:SV_X"},
    ),
}


def _even_bounds(row_count: int, chunk_count: int) -> list[int]:
    """Returns increasing bounds for evenly sized, non-empty chunks."""
    chunk_count = min(chunk_count, row_count)
    base, remainder = divmod(row_count, chunk_count)
    bounds = [0]
    for i in range(chunk_count):
        bounds.append(bounds[-1] + base + (1 if i < remainder else 0))
    return bounds


class TestGroupbySumChunksPrecedenceParity:
    """The streamed path must match whole-table refusal precedence across 1, 2, and 4 chunks."""

    @pytest.mark.parametrize("fixture_name", sorted(_PRECEDENCE_FIXTURES))
    @pytest.mark.parametrize("chunk_count", [1, 2, 4])
    def test_streamed_precedence_matches_whole_table(self, fixture_name, chunk_count):
        rows, hierarchy = _PRECEDENCE_FIXTURES[fixture_name]
        whole = Table(columns=_PRECEDENCE_COLUMNS, rows=rows)

        with pytest.raises(RollupNotSummableError) as whole_exc:
            groupby_sum(
                whole,
                axis="stat_var",
                classification=_PRECEDENCE_CLASSIFICATION,
                summable_axes=frozenset({"stat_var"}),
                hierarchy=hierarchy,
                missing_declared_as_zero=False,
            )

        bounds = _even_bounds(len(rows), chunk_count)
        with pytest.raises(RollupNotSummableError) as streamed_exc:
            groupby_sum_chunks(
                iter(_chunks(_PRECEDENCE_COLUMNS, list(rows), bounds)),
                axis="stat_var",
                classification=_PRECEDENCE_CLASSIFICATION,
                summable_axes=frozenset({"stat_var"}),
                hierarchy=hierarchy,
                missing_declared_as_zero=False,
            )

        assert streamed_exc.value.precondition == whole_exc.value.precondition
        assert streamed_exc.value.args == whole_exc.value.args


_PIVOT_CLASSIFICATION = ColumnClassification(
    facet_axis_columns=frozenset({"unit", "measurementMethod", "provenance"}),
    non_droppable_columns=frozenset({"stat_var", "recipient", "time_period", "unit", "provenance"}),
)


class TestPivotChunksDigest:
    """Checks streamed pivots against whole-table pivots."""

    def test_a_wide_pivot_split_across_three_chunks_matches_the_whole_table_result(self):
        rows = [
            _groupby_row("dcid:SV_A", "country/KEN", "2015", 10.0),
            _groupby_row("dcid:SV_A", "country/KEN", "2016", 20.0),
            _groupby_row("dcid:SV_A", "country/TZA", "2015", 5.0),
            _groupby_row("dcid:SV_A", "country/TZA", "2016", 7.0),
        ]
        whole = Table(columns=_GROUPBY_COLUMNS, rows=tuple(rows))
        expected = pivot(
            whole,
            index="recipient",
            columns="time_period",
            column_cap=100,
            cell_cap=1000,
            classification=_PIVOT_CLASSIFICATION,
        )

        streamed = pivot_chunks(
            iter(_chunks(_GROUPBY_COLUMNS, rows, [0, 1, 3, len(rows)])),
            index="recipient",
            columns="time_period",
            column_cap=100,
            cell_cap=1000,
            classification=_PIVOT_CLASSIFICATION,
        )

        assert streamed == expected
        assert streamed.columns == ("recipient", "2015", "2016")

    def test_a_duplicate_key_across_chunks_still_refuses_pivot_not_unique(self):
        rows = [
            _groupby_row("dcid:SV_A", "country/KEN", "2015", 10.0),
            _groupby_row("dcid:SV_A", "country/KEN", "2015", 999.0),
        ]
        with pytest.raises(PivotNotUniqueError):
            pivot_chunks(
                iter(_chunks(_GROUPBY_COLUMNS, rows, [0, 1, 2])),
                index="recipient",
                columns="time_period",
                column_cap=100,
                cell_cap=1000,
                classification=_PIVOT_CLASSIFICATION,
            )

    def test_requires_at_least_one_chunk(self):
        with pytest.raises(MaterialisationUsageError):
            pivot_chunks(
                iter([]),
                index="recipient",
                columns="time_period",
                column_cap=100,
                cell_cap=1000,
                classification=_PIVOT_CLASSIFICATION,
            )

    @pytest.mark.parametrize("bounds", [[0, 3], [0, 1, 3], [0, 1, 2, 3]])
    def test_duplicate_key_detail_matches_pivot_across_chunk_splits(self, bounds):
        # Three duplicates verify detail is stable across chunk boundaries.
        rows = [
            _groupby_row("dcid:SV_A", "country/KEN", "2015", 10.0),
            _groupby_row("dcid:SV_A", "country/KEN", "2015", 20.0),
            _groupby_row("dcid:SV_A", "country/KEN", "2015", 30.0),
        ]
        whole = Table(columns=_GROUPBY_COLUMNS, rows=tuple(rows))
        with pytest.raises(PivotNotUniqueError) as whole_exc:
            pivot(
                whole,
                index="recipient",
                columns="time_period",
                column_cap=100,
                cell_cap=1000,
                classification=_PIVOT_CLASSIFICATION,
            )
        with pytest.raises(PivotNotUniqueError) as streamed_exc:
            pivot_chunks(
                iter(_chunks(_GROUPBY_COLUMNS, rows, bounds)),
                index="recipient",
                columns="time_period",
                column_cap=100,
                cell_cap=1000,
                classification=_PIVOT_CLASSIFICATION,
            )
        assert streamed_exc.value.duplicate_keys == whole_exc.value.duplicate_keys
        assert streamed_exc.value.truncated == whole_exc.value.truncated
