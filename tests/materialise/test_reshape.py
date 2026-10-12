"""Tests reshape classification, operations, execution order, and seeded value-preservation cases."""

from __future__ import annotations

import random
from typing import cast

import pytest

from materialise.errors import MaterialisationUsageError
from materialise.reshape import (
    _MAX_HOMOGENEITY_VALUES_PER_GROUP,
    _MAX_NAMED_DUPLICATE_KEYS,
    SUPPORTED_ENRICHMENTS,
    ColumnNotDroppableError,
    ColumnUnknownError,
    DecodePlan,
    EnrichmentPlan,
    EnrichmentUnavailableWarning,
    ExpectedPopulation,
    GroupbyPlan,
    ParsedColumns,
    PivotAnalysis,
    PivotNotHomogeneousError,
    PivotNotUniqueError,
    PivotOverCellCapError,
    PivotOverColumnCapError,
    PivotPlan,
    ReshapePlan,
    RollupNotSummableError,
    Table,
    analyse_pivot,
    apply_reshapes,
    classify_columns,
    decode_columns,
    enrich_columns,
    groupby_sum,
    parse_column_tokens,
    pivot,
    select_columns,
)
from materialise.trace import ManifestTrace, ResolvedStatVar

_STAT_VAR = "dcid:MalariaGrantDisbursements"
_BASE_COLUMNS = (
    "stat_var",
    "entity",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)


def _row(
    entity: str,
    period: str,
    value: float,
    *,
    unit: str = "USD",
    method: str = "Standard",
    provenance: str = "OECD_DAC",
) -> tuple:
    return (_STAT_VAR, entity, period, value, unit, method, provenance, None)


class TestClassifyColumns:
    def test_classifies_a_realistic_manifest_trace_shaped_input(self):
        # A realistic ManifestTrace, built the way generation.py's own
        # `_generate_member` would: `resolved_stat_vars[0]["roles"]` is
        # exactly the role-key list `grammar.derive_columns` would place
        # between `stat_var` and `time_period`.
        trace = ManifestTrace(
            applied_window=None,
            graph_revision="rev-1",
            source_revisions=(),
            resolved_stat_vars=(
                ResolvedStatVar(
                    ref=_STAT_VAR, roles=("provider", "recipient"), observation_count=3
                ).as_trace_entry(),
            ),
            facets=(),
            normalisations=(),
            conversion=None,
            frame_grammar_version="1.0.0",
            materialiser_build="test-build",
            capability_version="v1",
        )
        roles = tuple(trace.resolved_stat_vars[0]["roles"])
        columns = (
            "stat_var",
            *roles,
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        )

        classification = classify_columns(columns)

        assert classification.facet_axis_columns == {"unit", "measurementMethod", "provenance"}
        assert classification.non_droppable_columns == {
            "stat_var",
            "provider",
            "recipient",
            "time_period",
            "unit",
            "provenance",
        }

    def test_measurement_method_is_a_facet_axis_but_not_non_droppable(self):
        classification = classify_columns(_BASE_COLUMNS)

        assert "measurementMethod" in classification.facet_axis_columns
        assert "measurementMethod" not in classification.non_droppable_columns


class TestGroupbySum:
    def test_happy_path_collapses_the_axis_and_sums_value(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row("country/KEN", "2020", 10.0), _row("country/TGO", "2020", 20.0)),
        )
        classification = classify_columns(table.columns)

        result = groupby_sum(
            table,
            axis="entity",
            classification=classification,
            summable_axes=frozenset({"entity"}),
        )

        assert result.columns == (
            "stat_var",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        )
        assert result.rows == ((_STAT_VAR, "2020", 30.0, "USD", "Standard", "OECD_DAC", None),)

    def test_missing_declared_as_zero_admits_a_null_axis_member(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row(None, "2020", 5.0), _row("country/TGO", "2020", 20.0)),
        )
        classification = classify_columns(table.columns)

        result = groupby_sum(
            table,
            axis="entity",
            classification=classification,
            summable_axes=frozenset({"entity"}),
            missing_declared_as_zero=True,
        )

        assert result.rows == ((_STAT_VAR, "2020", 25.0, "USD", "Standard", "OECD_DAC", None),)

    def test_refuses_a_facet_axis_as_rollup_not_summable(self):
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="unit",
                classification=classification,
                summable_axes=frozenset({"unit"}),
            )

        assert excinfo.value.code == "rollup_not_summable"
        assert excinfo.value.axis == "unit"
        assert excinfo.value.precondition == "facet_axis"

    def test_refuses_an_axis_the_graph_has_not_declared_summable(self):
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset(),
            )

        assert excinfo.value.precondition == "not_declared_summable"

    def test_refuses_a_duplicate_full_grain_as_unique_input_grain(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row("country/KEN", "2020", 10.0), _row("country/KEN", "2020", 10.0)),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
            )

        assert excinfo.value.precondition == "unique_input_grain"

    def test_refuses_heterogeneous_facets_within_a_collapsed_group(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(
                _row("country/KEN", "2020", 10.0, unit="USD"),
                _row("country/TGO", "2020", 20.0, unit="EUR"),
            ),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
            )

        assert excinfo.value.precondition == "facet_homogeneity"

    def test_refuses_a_parent_and_child_present_in_the_same_slice(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row("region/parent", "2020", 100.0), _row("region/child", "2020", 10.0)),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                hierarchy={"region/child": "region/parent"},
            )

        assert excinfo.value.precondition == "parent_child_overlap"

    def test_refuses_a_null_member_without_a_missing_as_zero_declaration(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row(None, "2020", 5.0), _row("country/TGO", "2020", 20.0)),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
            )

        assert excinfo.value.precondition == "missing_members"

    def test_refuses_a_value_column_absent_from_the_table_as_column_unknown(self):
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)

        with pytest.raises(ColumnUnknownError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                value_column="bogus_value",
            )

        assert excinfo.value.code == "column_unknown"
        assert excinfo.value.column == "bogus_value"

    def test_empty_hierarchy_does_not_walk_pairs_or_raise_on_unrelated_names(self):
        # Without a hierarchy mapping, ancestor-shaped refs are summed normally.
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row("region/parent", "2020", 100.0), _row("region/child", "2020", 10.0)),
        )
        classification = classify_columns(table.columns)

        result = groupby_sum(
            table,
            axis="entity",
            classification=classification,
            summable_axes=frozenset({"entity"}),
        )

        assert result.rows == ((_STAT_VAR, "2020", 110.0, "USD", "Standard", "OECD_DAC", None),)

    def test_a_non_empty_hierarchy_still_refuses_a_genuine_pair_among_unrelated_members(self):
        # The declared parent-child pair is among unrelated members in the same slice.
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(
                _row("region/parent", "2020", 100.0),
                _row("region/child", "2020", 10.0),
                _row("region/unrelated_a", "2020", 1.0),
                _row("region/unrelated_b", "2020", 2.0),
            ),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                hierarchy={"region/child": "region/parent"},
            )

        assert excinfo.value.precondition == "parent_child_overlap"
        assert "region/parent" in str(excinfo.value)
        assert "region/child" in str(excinfo.value)

    def test_a_non_empty_hierarchy_with_no_related_members_in_slice_passes(self):
        # No slice member appears in the supplied hierarchy.
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(
                _row("region/unrelated_a", "2020", 1.0),
                _row("region/unrelated_b", "2020", 2.0),
                _row("region/unrelated_c", "2020", 3.0),
            ),
        )
        classification = classify_columns(table.columns)

        result = groupby_sum(
            table,
            axis="entity",
            classification=classification,
            summable_axes=frozenset({"entity"}),
            hierarchy={"region/child": "region/parent"},
        )

        assert result.rows == ((_STAT_VAR, "2020", 6.0, "USD", "Standard", "OECD_DAC", None),)

    def test_expected_population_one_column_missing_one_refuses_naming_it(self):
        # One-column populations apply the same expected set to every group.
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)
        population = ExpectedPopulation(
            columns=("entity",), members=frozenset({("country/KEN",), ("country/TGO",)})
        )

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                expected_population=population,
            )

        assert excinfo.value.precondition == "missing_members"
        assert "country/TGO" in str(excinfo.value)

    def test_expected_population_one_column_full_coverage_passes(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row("country/KEN", "2020", 10.0), _row("country/TGO", "2020", 20.0)),
        )
        classification = classify_columns(table.columns)
        population = ExpectedPopulation(
            columns=("entity",), members=frozenset({("country/KEN",), ("country/TGO",)})
        )

        result = groupby_sum(
            table,
            axis="entity",
            classification=classification,
            summable_axes=frozenset({"entity"}),
            expected_population=population,
        )

        assert result.rows == ((_STAT_VAR, "2020", 30.0, "USD", "Standard", "OECD_DAC", None),)

    def test_expected_population_none_checks_missingness_only_at_cell_level(self):
        # Population checks run only when `expected_population` is supplied.
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)

        result = groupby_sum(
            table,
            axis="entity",
            classification=classification,
            summable_axes=frozenset({"entity"}),
        )

        assert result.rows == ((_STAT_VAR, "2020", 10.0, "USD", "Standard", "OECD_DAC", None),)

    def test_expected_population_zero_row_member_refuses_then_sums_with_declaration(self):
        # A zero-row population member refuses unless missing values count as zero.
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)
        population = ExpectedPopulation(
            columns=("entity",), members=frozenset({("country/KEN",), ("country/TGO",)})
        )

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                expected_population=population,
            )
        assert excinfo.value.precondition == "missing_members"

        result = groupby_sum(
            table,
            axis="entity",
            classification=classification,
            summable_axes=frozenset({"entity"}),
            expected_population=population,
            missing_declared_as_zero=True,
        )

        assert result.rows == ((_STAT_VAR, "2020", 10.0, "USD", "Standard", "OECD_DAC", None),)

    def test_apply_reshapes_plumbs_expected_population_through_the_groupby_plan(self):
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        plan = ReshapePlan(
            groupby=GroupbyPlan(
                axis="entity",
                summable_axes=frozenset({"entity"}),
                expected_population=ExpectedPopulation(
                    columns=("entity",), members=frozenset({("country/KEN",), ("country/TGO",)})
                ),
            )
        )

        with pytest.raises(RollupNotSummableError) as excinfo:
            apply_reshapes(table, plan=plan)

        assert excinfo.value.precondition == "missing_members"

    def test_expected_population_heterogeneous_kept_groups_complete_frame_passes(self):
        # Health expects purposes a and b. Education expects c. A flat set would falsely
        # require c in the health group.
        columns = (*_BASE_COLUMNS[:2], "sector", "purpose", *_BASE_COLUMNS[2:])
        rows = (
            (
                _STAT_VAR,
                "country/KEN",
                "health",
                "purpose/a",
                "2020",
                10.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                _STAT_VAR,
                "country/KEN",
                "health",
                "purpose/b",
                "2020",
                20.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                _STAT_VAR,
                "country/KEN",
                "education",
                "purpose/c",
                "2020",
                5.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
        )
        table = Table(columns=columns, rows=rows)
        classification = classify_columns(columns)
        population = ExpectedPopulation(
            columns=("purpose", "sector"),
            members=frozenset(
                {
                    ("purpose/a", "health"),
                    ("purpose/b", "health"),
                    ("purpose/c", "education"),
                }
            ),
        )

        result = groupby_sum(
            table,
            axis="purpose",
            classification=classification,
            summable_axes=frozenset({"purpose"}),
            expected_population=population,
        )

        assert set(result.rows) == {
            (_STAT_VAR, "country/KEN", "health", "2020", 30.0, "USD", "Standard", "OECD_DAC", None),
            (
                _STAT_VAR,
                "country/KEN",
                "education",
                "2020",
                5.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
        }

    def test_expected_population_member_missing_within_one_group_refuses_naming_group(self):
        columns = (*_BASE_COLUMNS[:2], "sector", "purpose", *_BASE_COLUMNS[2:])
        rows = (
            (
                _STAT_VAR,
                "country/KEN",
                "health",
                "purpose/a",
                "2020",
                10.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                _STAT_VAR,
                "country/KEN",
                "education",
                "purpose/c",
                "2020",
                5.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
        )
        table = Table(columns=columns, rows=rows)
        classification = classify_columns(columns)
        population = ExpectedPopulation(
            columns=("purpose", "sector"),
            members=frozenset(
                {
                    ("purpose/a", "health"),
                    ("purpose/b", "health"),
                    ("purpose/c", "education"),
                }
            ),
        )

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="purpose",
                classification=classification,
                summable_axes=frozenset({"purpose"}),
                expected_population=population,
            )

        assert excinfo.value.precondition == "missing_members"
        message = str(excinfo.value)
        assert "purpose/b" in message
        assert "health" in message

    def test_expected_population_composes_across_two_sequential_rollups(self):
        # The first rollup skips TGO's absent education cell; the second catches it.
        columns = (*_BASE_COLUMNS[:2], "sector", "purpose", *_BASE_COLUMNS[2:])
        rows = (
            (
                _STAT_VAR,
                "country/KEN",
                "health",
                "purpose/a",
                "2020",
                10.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                _STAT_VAR,
                "country/KEN",
                "health",
                "purpose/b",
                "2020",
                20.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                _STAT_VAR,
                "country/KEN",
                "education",
                "purpose/c",
                "2020",
                5.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                _STAT_VAR,
                "country/TGO",
                "health",
                "purpose/a",
                "2020",
                7.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                _STAT_VAR,
                "country/TGO",
                "health",
                "purpose/b",
                "2020",
                3.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
        )
        table = Table(columns=columns, rows=rows)
        classification = classify_columns(columns)
        purpose_population = ExpectedPopulation(
            columns=("purpose", "sector"),
            members=frozenset(
                {
                    ("purpose/a", "health"),
                    ("purpose/b", "health"),
                    ("purpose/c", "education"),
                }
            ),
        )

        first = groupby_sum(
            table,
            axis="purpose",
            classification=classification,
            summable_axes=frozenset({"purpose"}),
            expected_population=purpose_population,
        )

        second_classification = classify_columns(first.columns)
        sector_population = ExpectedPopulation(
            columns=("sector",), members=frozenset({("health",), ("education",)})
        )

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                first,
                axis="sector",
                classification=second_classification,
                summable_axes=frozenset({"sector"}),
                expected_population=sector_population,
            )

        assert excinfo.value.precondition == "missing_members"
        assert "education" in str(excinfo.value)

    def test_expected_population_axis_absent_from_columns_raises_value_error(self):
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)
        population = ExpectedPopulation(columns=("sector",), members=frozenset({("health",)}))

        with pytest.raises(ValueError, match="entity"):
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                expected_population=population,
            )

    def test_expected_population_kept_column_not_a_table_column_raises_value_error(self):
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)
        population = ExpectedPopulation(
            columns=("entity", "sector"), members=frozenset({("country/KEN", "health")})
        )

        with pytest.raises(ValueError, match="sector"):
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                expected_population=population,
            )

    def test_expected_population_kept_column_that_is_a_facet_axis_raises_value_error(self):
        # `unit` is a facet column. Kept population columns are grouping columns.
        table = Table(columns=_BASE_COLUMNS, rows=(_row("country/KEN", "2020", 10.0),))
        classification = classify_columns(table.columns)
        population = ExpectedPopulation(
            columns=("entity", "unit"), members=frozenset({("country/KEN", "USD")})
        )

        with pytest.raises(ValueError, match="unit"):
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
                expected_population=population,
            )

    @pytest.mark.parametrize("missing_declared_as_zero", [False, True])
    def test_expected_population_unmatched_group_raises_value_error(
        self, missing_declared_as_zero: bool
    ):
        # The population describes TGO, while the table contains KEN. This shape error raises
        # regardless of `missing_declared_as_zero`.
        columns = (*_BASE_COLUMNS[:2], "sector", *_BASE_COLUMNS[2:])
        table = Table(
            columns=columns,
            rows=(
                (
                    _STAT_VAR,
                    "country/KEN",
                    "health",
                    "2020",
                    10.0,
                    "USD",
                    "Standard",
                    "OECD_DAC",
                    None,
                ),
            ),
        )
        classification = classify_columns(columns)
        population = ExpectedPopulation(
            columns=("sector", "entity"), members=frozenset({("health", "country/TGO")})
        )

        with pytest.raises(ValueError, match="country/KEN"):
            groupby_sum(
                table,
                axis="sector",
                classification=classification,
                summable_axes=frozenset({"sector"}),
                expected_population=population,
                missing_declared_as_zero=missing_declared_as_zero,
            )

    def test_expected_population_stat_var_rollup_collapses_collision_free(self):
        # SV_A and SV_B share purpose 110 and merge; SV_C's purpose 120 stays separate.
        # The expected population pairs each StatVar with its decoded purpose.
        sv_a, sv_b, sv_c = "dcid:StatVarA", "dcid:StatVarB", "dcid:StatVarC"
        columns = (*_BASE_COLUMNS[:1], "stat_var_crsPurpose", *_BASE_COLUMNS[1:])

        def row(stat_var: str, purpose: str, value: float) -> tuple:
            return (
                stat_var,
                purpose,
                "country/KEN",
                "2020",
                value,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            )

        table = Table(
            columns=columns,
            rows=(
                row(sv_a, "purpose/110", 5.0),
                row(sv_b, "purpose/110", 3.0),
                row(sv_c, "purpose/120", 7.0),
            ),
        )
        classification = classify_columns(columns)
        population = ExpectedPopulation(
            columns=("stat_var", "stat_var_crsPurpose"),
            members=frozenset(
                {
                    (sv_a, "purpose/110"),
                    (sv_b, "purpose/110"),
                    (sv_c, "purpose/120"),
                }
            ),
        )

        result = groupby_sum(
            table,
            axis="stat_var",
            classification=classification,
            summable_axes=frozenset({"stat_var"}),
            expected_population=population,
        )

        assert set(result.rows) == {
            (
                "purpose/110",
                "country/KEN",
                "2020",
                8.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
            (
                "purpose/120",
                "country/KEN",
                "2020",
                7.0,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            ),
        }

    def test_expected_population_stat_var_rollup_missing_member_refuses_naming_it(self):
        # Removing SV_B leaves purpose 110 short of its declared population.
        sv_a, sv_b, sv_c = "dcid:StatVarA", "dcid:StatVarB", "dcid:StatVarC"
        columns = (*_BASE_COLUMNS[:1], "stat_var_crsPurpose", *_BASE_COLUMNS[1:])

        def row(stat_var: str, purpose: str, value: float) -> tuple:
            return (
                stat_var,
                purpose,
                "country/KEN",
                "2020",
                value,
                "USD",
                "Standard",
                "OECD_DAC",
                None,
            )

        table = Table(
            columns=columns,
            rows=(
                row(sv_a, "purpose/110", 5.0),
                row(sv_c, "purpose/120", 7.0),
            ),
        )
        classification = classify_columns(columns)
        population = ExpectedPopulation(
            columns=("stat_var", "stat_var_crsPurpose"),
            members=frozenset(
                {
                    (sv_a, "purpose/110"),
                    (sv_b, "purpose/110"),
                    (sv_c, "purpose/120"),
                }
            ),
        )

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="stat_var",
                classification=classification,
                summable_axes=frozenset({"stat_var"}),
                expected_population=population,
            )

        assert excinfo.value.precondition == "missing_members"
        assert sv_b in str(excinfo.value)
        assert "purpose/110" in str(excinfo.value)

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param(
                (
                    _row("country/KEN", "2020", 1e308),
                    _row("country/TGO", "2020", 1e308),
                ),
                id="overflow",
            ),
            pytest.param(
                (
                    _row("country/KEN", "2020", float("nan")),
                    _row("country/TGO", "2020", 20.0),
                ),
                id="nan-cell",
            ),
        ],
    )
    def test_refuses_a_non_finite_total(self, rows: tuple):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=rows,
        )
        classification = classify_columns(table.columns)

        with pytest.raises(RollupNotSummableError) as excinfo:
            groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
            )

        assert excinfo.value.precondition == "finite_total"


class TestExpectedPopulation:
    def test_construction_refuses_empty_columns(self):
        with pytest.raises(ValueError):
            ExpectedPopulation(columns=(), members=frozenset())

    def test_construction_refuses_a_repeated_column_name(self):
        with pytest.raises(ValueError):
            ExpectedPopulation(columns=("entity", "entity"), members=frozenset())

    def test_construction_refuses_a_member_tuple_with_the_wrong_width(self):
        with pytest.raises(ValueError):
            ExpectedPopulation(columns=("entity", "sector"), members=frozenset({("country/KEN",)}))


class TestSelectColumns:
    _POST_GROUPBY_COLUMNS = (
        "stat_var",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )

    def test_happy_path_preserves_the_callers_order(self):
        table = Table(
            columns=self._POST_GROUPBY_COLUMNS,
            rows=((_STAT_VAR, "2020", 30.0, "USD", "Standard", "OECD_DAC", None),),
        )
        classification = classify_columns(table.columns)
        keep = ("time_period", "stat_var", "unit", "provenance", "value")

        result = select_columns(table, keep=keep, classification=classification)

        assert result.columns == keep
        assert result.rows == (("2020", _STAT_VAR, "USD", "OECD_DAC", 30.0),)

    def test_refuses_an_unknown_column_name(self):
        table = Table(
            columns=self._POST_GROUPBY_COLUMNS,
            rows=((_STAT_VAR, "2020", 30.0, "USD", "Standard", "OECD_DAC", None),),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(ColumnUnknownError) as excinfo:
            select_columns(table, keep=("bogus_column",), classification=classification)

        assert excinfo.value.code == "column_unknown"
        assert excinfo.value.column == "bogus_column"

    def test_an_axis_a_prior_groupby_collapsed_is_unknown_not_not_droppable(self):
        # The table lacks `entity`, so selection raises `ColumnUnknownError`.
        table = Table(
            columns=self._POST_GROUPBY_COLUMNS,
            rows=((_STAT_VAR, "2020", 30.0, "USD", "Standard", "OECD_DAC", None),),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(ColumnUnknownError):
            select_columns(table, keep=("entity", "value"), classification=classification)

    def test_refuses_dropping_a_non_droppable_column(self):
        table = Table(
            columns=self._POST_GROUPBY_COLUMNS,
            rows=((_STAT_VAR, "2020", 30.0, "USD", "Standard", "OECD_DAC", None),),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(ColumnNotDroppableError) as excinfo:
            select_columns(
                table,
                keep=("time_period", "value", "unit", "provenance"),
                classification=classification,
            )

        assert excinfo.value.code == "column_not_droppable"
        assert excinfo.value.columns == ("stat_var",)

    def test_refuses_dropping_two_non_droppable_columns_naming_both_in_order(self):
        table = Table(
            columns=self._POST_GROUPBY_COLUMNS,
            rows=((_STAT_VAR, "2020", 30.0, "USD", "Standard", "OECD_DAC", None),),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(ColumnNotDroppableError) as excinfo:
            select_columns(
                table, keep=("time_period", "value", "provenance"), classification=classification
            )

        assert excinfo.value.code == "column_not_droppable"
        assert excinfo.value.columns == ("stat_var", "unit")
        assert "stat_var" in str(excinfo.value)
        assert "unit" in str(excinfo.value)


class TestEnrichColumns:
    _COLUMNS = ("stat_var", "entity", "donor", "time_period", "value")

    def test_happy_path_fills_every_requested_field(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                (_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0),
                (_STAT_VAR, "country/TGO", "donor/A", "2020", 2.0),
            ),
        )
        lookup = {
            "country/KEN": {"iso2": "KE", "iso3": "KEN", "label": "Kenya"},
            "country/TGO": {"iso2": "TG", "iso3": "TGO", "label": "Togo"},
        }

        result, warnings = enrich_columns(
            table,
            requests=(("entity", "iso2"), ("entity", "iso3"), ("entity", "label")),
            lookup=lookup,
        )

        assert result.columns == (*self._COLUMNS, "entity_iso2", "entity_iso3", "entity_label")
        assert result.rows == (
            (_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0, "KE", "KEN", "Kenya"),
            (_STAT_VAR, "country/TGO", "donor/A", "2020", 2.0, "TG", "TGO", "Togo"),
        )
        assert warnings == ()

    def test_two_entity_columns_in_one_call_each_get_their_own_columns(self):
        table = Table(
            columns=self._COLUMNS,
            rows=((_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0),),
        )
        lookup = {
            "country/KEN": {"iso3": "KEN"},
            "donor/A": {"label": "Donor A"},
        }

        result, warnings = enrich_columns(
            table,
            requests=(("entity", "iso3"), ("donor", "label")),
            lookup=lookup,
        )

        assert result.columns == (*self._COLUMNS, "entity_iso3", "donor_label")
        assert result.rows == (
            (_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0, "KEN", "Donor A"),
        )
        assert warnings == ()

    def test_a_missing_lookup_entry_fills_empty_and_warns_once_per_affected_request(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                (_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0),
                (_STAT_VAR, "country/TGO", "donor/A", "2020", 2.0),
            ),
        )
        lookup = {"country/KEN": {"iso3": "KEN", "label": "Kenya"}}

        result, warnings = enrich_columns(
            table,
            requests=(("entity", "iso3"), ("entity", "label")),
            lookup=lookup,
        )

        assert result.rows[1] == (_STAT_VAR, "country/TGO", "donor/A", "2020", 2.0, "", "")
        # Each missing field has its own warning and output column.
        assert len(warnings) == 2
        iso3_warning = cast(EnrichmentUnavailableWarning, warnings[0])
        label_warning = cast(EnrichmentUnavailableWarning, warnings[1])
        assert iso3_warning["code"] == "enrichment_unavailable"
        assert iso3_warning["severity"] == "warn"
        assert iso3_warning["column"] == "entity_iso3"
        assert iso3_warning["count"] == 1
        assert label_warning["column"] == "entity_label"
        assert label_warning["count"] == 1

    def test_only_the_affected_request_warns_when_another_request_has_no_gaps(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                (_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0),
                (_STAT_VAR, "country/TGO", "donor/A", "2020", 2.0),
            ),
        )
        lookup = {
            "country/KEN": {"iso3": "KEN", "label": "Kenya"},
            "country/TGO": {"iso3": "TGO"},
        }

        result, warnings = enrich_columns(
            table,
            requests=(("entity", "iso3"), ("entity", "label")),
            lookup=lookup,
        )

        assert result.rows[1] == (_STAT_VAR, "country/TGO", "donor/A", "2020", 2.0, "TGO", "")
        assert len(warnings) == 1
        warning = cast(EnrichmentUnavailableWarning, warnings[0])
        assert warning["column"] == "entity_label"
        assert warning["count"] == 1

    def test_refuses_an_unsupported_enrichment_field(self):
        table = Table(
            columns=self._COLUMNS, rows=((_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0),)
        )

        with pytest.raises(MaterialisationUsageError):
            enrich_columns(table, requests=(("entity", "gdp"),), lookup={})

    def test_refuses_an_entity_column_absent_from_the_table(self):
        table = Table(
            columns=self._COLUMNS, rows=((_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0),)
        )

        with pytest.raises(MaterialisationUsageError):
            enrich_columns(table, requests=(("bogus", "iso3"),), lookup={})

    def test_refuses_a_duplicate_request(self):
        table = Table(
            columns=self._COLUMNS, rows=((_STAT_VAR, "country/KEN", "donor/A", "2020", 1.0),)
        )

        with pytest.raises(MaterialisationUsageError):
            enrich_columns(table, requests=(("entity", "iso3"), ("entity", "iso3")), lookup={})


class TestDecodeColumns:
    """Tests decoding with open field names and declared-absent values."""

    _COLUMNS = ("stat_var", "entity", "time_period", "value")

    def test_happy_path_appends_one_column_per_code_system(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("dcid:SV_A", "country/KEN", "2020", 1.0),
                ("dcid:SV_B", "country/KEN", "2020", 2.0),
            ),
        )
        lookup = {
            "dcid:SV_A": {"crsPurpose": "Basic health care", "crsTypeOfFinance": "Standard grant"},
            "dcid:SV_B": {"crsPurpose": "Malaria control", "crsTypeOfFinance": "Standard grant"},
        }

        result, warnings = decode_columns(
            table,
            requests=(("stat_var", "crsPurpose"), ("stat_var", "crsTypeOfFinance")),
            lookup=lookup,
        )

        assert result.columns == (
            *self._COLUMNS,
            "stat_var_crsPurpose",
            "stat_var_crsTypeOfFinance",
        )
        assert result.rows == (
            ("dcid:SV_A", "country/KEN", "2020", 1.0, "Basic health care", "Standard grant"),
            ("dcid:SV_B", "country/KEN", "2020", 2.0, "Malaria control", "Standard grant"),
        )
        assert warnings == ()

    def test_a_declared_absent_reading_is_a_real_value_with_no_warning(self):
        # The lookup contains `not_declared` as an explicit value.
        table = Table(columns=self._COLUMNS, rows=(("dcid:SV_A", "country/KEN", "2020", 1.0),))
        lookup = {"dcid:SV_A": {"crsCooperationModality": "not_declared"}}

        result, warnings = decode_columns(
            table, requests=(("stat_var", "crsCooperationModality"),), lookup=lookup
        )

        assert result.rows == (("dcid:SV_A", "country/KEN", "2020", 1.0, "not_declared"),)
        assert warnings == ()

    def test_a_genuine_index_miss_fills_empty_and_warns_distinguishably_from_declared_absent(
        self,
    ):
        # SV_B is absent from the lookup; SV_A carries the explicit sentinel.
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("dcid:SV_A", "country/KEN", "2020", 1.0),
                ("dcid:SV_B", "country/KEN", "2020", 2.0),
            ),
        )
        lookup = {"dcid:SV_A": {"crsPurpose": "not_declared"}}

        result, warnings = decode_columns(
            table, requests=(("stat_var", "crsPurpose"),), lookup=lookup
        )

        assert result.rows[0][-1] == "not_declared"
        assert result.rows[1][-1] == ""
        assert len(warnings) == 1
        warning = cast(EnrichmentUnavailableWarning, warnings[0])
        assert warning["code"] == "enrichment_unavailable"
        assert warning["column"] == "stat_var_crsPurpose"
        assert warning["count"] == 1

    def test_no_closed_field_vocabulary_any_code_system_name_is_accepted(self):
        # Decode accepts code-system names outside the enrichment vocabulary.
        table = Table(columns=self._COLUMNS, rows=(("dcid:SV_A", "country/KEN", "2020", 1.0),))

        result, warnings = decode_columns(
            table,
            requests=(("stat_var", "someFutureCodeSystem"),),
            lookup={"dcid:SV_A": {"someFutureCodeSystem": "A Label"}},
        )

        assert result.columns[-1] == "stat_var_someFutureCodeSystem"
        assert result.rows[0][-1] == "A Label"
        assert warnings == ()

    def test_refuses_an_entity_column_absent_from_the_table(self):
        table = Table(columns=self._COLUMNS, rows=(("dcid:SV_A", "country/KEN", "2020", 1.0),))

        with pytest.raises(MaterialisationUsageError):
            decode_columns(table, requests=(("bogus", "crsPurpose"),), lookup={})

    def test_refuses_a_duplicate_request(self):
        table = Table(columns=self._COLUMNS, rows=(("dcid:SV_A", "country/KEN", "2020", 1.0),))

        with pytest.raises(MaterialisationUsageError):
            decode_columns(
                table,
                requests=(("stat_var", "crsPurpose"), ("stat_var", "crsPurpose")),
                lookup={},
            )


class TestParseColumnTokens:
    _TABLE_COLUMNS = ("stat_var", "entity", "time_period", "value")
    _ENTITY_COLUMNS = ("entity",)

    def test_a_plain_column_name_selects(self):
        result = parse_column_tokens(
            ("entity", "value"),
            table_columns=self._TABLE_COLUMNS,
            entity_columns=self._ENTITY_COLUMNS,
            supported_fields=SUPPORTED_ENRICHMENTS,
        )

        assert result == ParsedColumns(select=("entity", "value"), enrichments=())

    def test_a_qualified_token_is_an_enrichment_request(self):
        result = parse_column_tokens(
            ("entity_iso3",),
            table_columns=self._TABLE_COLUMNS,
            entity_columns=self._ENTITY_COLUMNS,
            supported_fields=SUPPORTED_ENRICHMENTS,
        )

        assert result == ParsedColumns(select=(), enrichments=(("entity", "iso3"),))

    def test_selection_and_enrichment_tokens_mix_with_order_preserved(self):
        result = parse_column_tokens(
            ("value", "entity_iso3", "entity", "entity_label"),
            table_columns=self._TABLE_COLUMNS,
            entity_columns=self._ENTITY_COLUMNS,
            supported_fields=SUPPORTED_ENRICHMENTS,
        )

        assert result == ParsedColumns(
            select=("value", "entity"),
            enrichments=(("entity", "iso3"), ("entity", "label")),
        )

    def test_a_real_column_named_like_a_token_shadows_the_enrichment_reading(self):
        table_columns = (*self._TABLE_COLUMNS, "entity_iso3")

        result = parse_column_tokens(
            ("entity_iso3",),
            table_columns=table_columns,
            entity_columns=self._ENTITY_COLUMNS,
            supported_fields=SUPPORTED_ENRICHMENTS,
        )

        assert result == ParsedColumns(select=("entity_iso3",), enrichments=())

    def test_an_unknown_token_names_every_qualified_alternative(self):
        with pytest.raises(ColumnUnknownError) as excinfo:
            parse_column_tokens(
                ("bogus",),
                table_columns=self._TABLE_COLUMNS,
                entity_columns=("entity", "donor"),
                supported_fields=SUPPORTED_ENRICHMENTS,
            )

        assert excinfo.value.column == "bogus"
        assert excinfo.value.alternatives is not None
        assert [(a.ref, a.label) for a in excinfo.value.alternatives] == [
            ("entity_iso2", "entity_iso2"),
            ("entity_iso3", "entity_iso3"),
            ("entity_kind", "entity_kind"),
            ("entity_label", "entity_label"),
            ("entity_type", "entity_type"),
            ("donor_iso2", "donor_iso2"),
            ("donor_iso3", "donor_iso3"),
            ("donor_kind", "donor_kind"),
            ("donor_label", "donor_label"),
            ("donor_type", "donor_type"),
        ]

    def test_a_bare_field_requests_it_for_every_entity_column_once(self):
        # A caller asks for kinds without knowing the table's roles; a qualified token the
        # bare one covers adds no second request.
        result = parse_column_tokens(
            ("donor_kind", "kind", "value", "entity_kind"),
            table_columns=self._TABLE_COLUMNS,
            entity_columns=("donor", "entity"),
            supported_fields=SUPPORTED_ENRICHMENTS,
        )

        assert result == ParsedColumns(
            select=("value",), enrichments=(("donor", "kind"), ("entity", "kind"))
        )

    def test_an_entity_column_containing_an_underscore_resolves_correctly(self):
        table_columns = ("stat_var", "observation_about", "time_period", "value")

        result = parse_column_tokens(
            ("observation_about_iso3",),
            table_columns=table_columns,
            entity_columns=("observation_about",),
            supported_fields=SUPPORTED_ENRICHMENTS,
        )

        assert result == ParsedColumns(select=(), enrichments=(("observation_about", "iso3"),))


class TestPivot:
    _COLUMNS = ("entity", "time_period", "value")
    _LARGE_CELL_CAP = 1_000_000

    def _pivot(self, table: Table, /, **kwargs):
        kwargs.setdefault("cell_cap", self._LARGE_CELL_CAP)
        kwargs.setdefault("classification", classify_columns(table.columns))
        return pivot(table, **kwargs)

    def test_happy_path_widens_columns_first_seen_order(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0),
                ("country/KEN", "2021", 2.0),
                ("country/TGO", "2020", 3.0),
            ),
        )

        result = self._pivot(table, index="entity", columns="time_period", column_cap=10)

        assert result.columns == ("entity", "2020", "2021")
        assert result.rows == (("country/KEN", 1.0, 2.0), ("country/TGO", 3.0, None))

    def test_an_excel_scale_column_cap_admits_this_table(self):
        # Excel's column cap is 16,384.
        table = Table(
            columns=self._COLUMNS,
            rows=(("country/KEN", "2020", 1.0), ("country/TGO", "2021", 2.0)),
        )

        result = self._pivot(table, index="entity", columns="time_period", column_cap=16_384)

        assert result.columns == ("entity", "2020", "2021")

    def test_refuses_a_duplicate_index_columns_key(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(("country/KEN", "2020", 1.0), ("country/KEN", "2020", 2.0)),
        )

        with pytest.raises(PivotNotUniqueError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=10)

        assert excinfo.value.code == "pivot_not_unique"
        assert excinfo.value.duplicate_keys == (("country/KEN", "2020"),)
        assert excinfo.value.truncated is False

    def test_a_key_repeated_many_times_is_named_once_alongside_other_duplicates(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0),
                ("country/KEN", "2020", 2.0),
                ("country/KEN", "2020", 3.0),
                ("country/KEN", "2020", 4.0),
                ("country/TGO", "2021", 5.0),
                ("country/TGO", "2021", 6.0),
            ),
        )

        with pytest.raises(PivotNotUniqueError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=10)

        # KEN/2020 is reported once despite four occurrences.
        assert excinfo.value.duplicate_keys == (
            ("country/KEN", "2020"),
            ("country/TGO", "2021"),
        )
        assert excinfo.value.truncated is False

    def test_a_key_repeated_thousands_of_times_still_names_it_once(self):
        # Five thousand repeats of one key produce one reported duplicate.
        rows = tuple(("country/KEN", "2020", float(i)) for i in range(5_000))
        table = Table(columns=self._COLUMNS, rows=rows)

        with pytest.raises(PivotNotUniqueError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=10)

        assert excinfo.value.duplicate_keys == (("country/KEN", "2020"),)
        assert excinfo.value.truncated is False

    def test_more_distinct_duplicate_keys_than_the_cap_truncates_by_distinct_count(self):
        distinct_key_count = _MAX_NAMED_DUPLICATE_KEYS + 3
        rows = tuple(
            row
            for i in range(distinct_key_count)
            for row in ((f"country/E{i}", "2020", float(i)), (f"country/E{i}", "2020", float(i)))
        )
        table = Table(columns=self._COLUMNS, rows=rows)

        with pytest.raises(PivotNotUniqueError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=1000)

        assert len(excinfo.value.duplicate_keys) == _MAX_NAMED_DUPLICATE_KEYS
        assert excinfo.value.truncated is True

    def test_refuses_over_the_injected_column_cap(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0),
                ("country/KEN", "2021", 2.0),
                ("country/KEN", "2022", 3.0),
            ),
        )

        with pytest.raises(PivotOverColumnCapError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=2)

        assert excinfo.value.code == "pivot_over_column_cap"
        assert excinfo.value.count == 4  # index + 3 distinct periods
        assert excinfo.value.cap == 2

    def test_refuses_over_the_injected_cell_cap_even_under_the_column_cap(self):
        # The table has 4 columns and 8 cells; only the 5-cell cap is exceeded.
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0),
                ("country/KEN", "2021", 2.0),
                ("country/KEN", "2022", 3.0),
                ("country/TGO", "2020", 4.0),
            ),
        )

        with pytest.raises(PivotOverCellCapError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=10, cell_cap=5)

        assert excinfo.value.code == "pivot_over_cell_cap"
        assert excinfo.value.count == 8  # 2 index values x 4 output columns
        assert excinfo.value.cap == 5

    def test_refuses_distinct_int_and_str_columns_values_that_stringify_alike(self):
        # Integer 2020 and string "2020" become the same output column name.
        table = Table(
            columns=self._COLUMNS,
            rows=(("country/KEN", 2020, 1.0), ("country/KEN", "2020", 2.0)),
        )

        with pytest.raises(PivotNotUniqueError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=10)

        assert excinfo.value.code == "pivot_not_unique"
        assert excinfo.value.duplicate_keys == ((2020, "2020"),)
        assert excinfo.value.truncated is False

    def test_refuses_none_and_the_string_none_that_stringify_alike(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(("country/KEN", None, 1.0), ("country/KEN", "None", 2.0)),
        )

        with pytest.raises(PivotNotUniqueError) as excinfo:
            self._pivot(table, index="entity", columns="time_period", column_cap=10)

        assert excinfo.value.code == "pivot_not_unique"
        assert excinfo.value.duplicate_keys == ((None, "None"),)
        assert excinfo.value.truncated is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"index": "bogus", "columns": "time_period", "value_column": "value"},
            {"index": "entity", "columns": "bogus", "value_column": "value"},
            {"index": "entity", "columns": "time_period", "value_column": "bogus"},
        ],
    )
    def test_refuses_a_column_name_absent_from_the_table_as_column_unknown(self, kwargs):
        table = Table(columns=self._COLUMNS, rows=(("country/KEN", "2020", 1.0),))

        with pytest.raises(ColumnUnknownError) as excinfo:
            self._pivot(table, column_cap=10, **kwargs)

        assert excinfo.value.code == "column_unknown"
        assert excinfo.value.column == "bogus"


class TestPivotHomogeneity:
    # `unit` and `provenance` are non-droppable columns omitted by this pivot.
    _COLUMNS = ("entity", "time_period", "value", "unit", "provenance")

    def test_a_dropped_non_droppable_column_that_varies_within_an_index_group_refuses(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0, "USD", "OECD_DAC"),
                ("country/KEN", "2021", 2.0, "EUR", "OECD_DAC"),
            ),
        )
        classification = classify_columns(table.columns)

        with pytest.raises(PivotNotHomogeneousError) as excinfo:
            pivot(
                table,
                index="entity",
                columns="time_period",
                column_cap=10,
                cell_cap=1_000_000,
                classification=classification,
            )

        assert excinfo.value.code == "pivot_not_homogeneous"
        assert excinfo.value.offenders == (("unit", "country/KEN", ("EUR", "USD")),)
        assert excinfo.value.truncated is False

    def test_a_homogeneous_dropped_column_never_trips_the_check(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0, "USD", "OECD_DAC"),
                ("country/KEN", "2021", 2.0, "USD", "OECD_DAC"),
                ("country/TGO", "2020", 3.0, "USD", "OECD_DAC"),
            ),
        )
        classification = classify_columns(table.columns)

        result = pivot(
            table,
            index="entity",
            columns="time_period",
            column_cap=10,
            cell_cap=1_000_000,
            classification=classification,
        )

        assert result.columns == ("entity", "2020", "2021")

    def test_a_column_constant_within_each_index_group_never_trips_the_check(self):
        # Unit varies across entities but stays constant within each entity.
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0, "USD", "OECD_DAC"),
                ("country/KEN", "2021", 2.0, "USD", "OECD_DAC"),
                ("country/TGO", "2020", 3.0, "EUR", "OECD_DAC"),
                ("country/TGO", "2021", 4.0, "EUR", "OECD_DAC"),
            ),
        )
        classification = classify_columns(table.columns)

        result = pivot(
            table,
            index="entity",
            columns="time_period",
            column_cap=10,
            cell_cap=1_000_000,
            classification=classification,
        )

        assert result.columns == ("entity", "2020", "2021")

    def test_a_dropped_column_absent_from_non_droppable_columns_is_exempt(self):
        # `status` is droppable and may vary within an index group.
        columns = (*self._COLUMNS[:3], "status")
        table = Table(
            columns=columns,
            rows=(
                ("country/KEN", "2020", 1.0, "provisional"),
                ("country/KEN", "2021", 2.0, "final"),
            ),
        )
        classification = classify_columns(table.columns)
        assert "status" not in classification.non_droppable_columns

        result = pivot(
            table,
            index="entity",
            columns="time_period",
            column_cap=10,
            cell_cap=1_000_000,
            classification=classification,
        )

        assert result.columns == ("entity", "2020", "2021")

    def test_a_group_with_far_more_distinct_values_than_the_cap_is_bounded_and_truncated(self):
        # One entity has more unit values than the evidence cap; distinct periods keep pivot keys
        # unique, isolating the homogeneity check.
        distinct_value_count = _MAX_HOMOGENEITY_VALUES_PER_GROUP + 50
        rows = tuple(
            ("country/KEN", f"period-{i}", float(i), f"UNIT{i}", "OECD_DAC")
            for i in range(distinct_value_count)
        )
        table = Table(columns=self._COLUMNS, rows=rows)
        classification = classify_columns(table.columns)

        with pytest.raises(PivotNotHomogeneousError) as excinfo:
            pivot(
                table,
                index="entity",
                columns="time_period",
                column_cap=10,
                cell_cap=1_000_000,
                classification=classification,
            )

        assert len(excinfo.value.offenders) == 1
        _column, _index_value, values = excinfo.value.offenders[0]
        assert len(values) == _MAX_HOMOGENEITY_VALUES_PER_GROUP
        assert excinfo.value.truncated is True

    def test_more_offenders_than_the_cap_truncates(self):
        rows = tuple(
            (f"country/E{i}", "2020", float(i), "USD" if i % 2 == 0 else "EUR", "OECD_DAC")
            for i in range(_MAX_NAMED_DUPLICATE_KEYS + 3)
        )
        rows += tuple(
            (f"country/E{i}", "2021", float(i), "EUR" if i % 2 == 0 else "USD", "OECD_DAC")
            for i in range(_MAX_NAMED_DUPLICATE_KEYS + 3)
        )
        table = Table(columns=self._COLUMNS, rows=rows)
        classification = classify_columns(table.columns)

        with pytest.raises(PivotNotHomogeneousError) as excinfo:
            pivot(
                table,
                index="entity",
                columns="time_period",
                column_cap=1000,
                cell_cap=1_000_000,
                classification=classification,
            )

        assert len(excinfo.value.offenders) == _MAX_NAMED_DUPLICATE_KEYS
        assert excinfo.value.truncated is True


class TestAnalysePivot:
    _COLUMNS = ("entity", "time_period", "value")
    # Two-role input shape.
    _ROLE_COLUMNS = ("stat_var", "provider", "recipient", "time_period", "value", "provenance")

    def test_a_unique_pair_reports_width_and_cells(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(
                ("country/KEN", "2020", 1.0),
                ("country/KEN", "2021", 2.0),
                ("country/TGO", "2020", 3.0),
            ),
        )
        classification = classify_columns(table.columns)

        analysis = analyse_pivot(
            table, index="entity", columns="time_period", classification=classification
        )

        assert isinstance(analysis, PivotAnalysis)
        assert analysis.index == "entity"
        assert analysis.columns == "time_period"
        assert analysis.unique is True
        assert analysis.homogeneous is True
        assert analysis.width == 3  # index column + 2 distinct periods
        assert analysis.cells == 6  # 2 distinct entities x 3 output columns
        assert analysis.varying == ()

    def test_a_duplicate_key_names_the_varying_column_and_its_frame_wide_count(self):
        table = Table(
            columns=self._ROLE_COLUMNS,
            rows=(
                (_STAT_VAR, "providerA", "recipientX", "2020", 1.0, "OECD_DAC"),
                (_STAT_VAR, "providerA", "recipientY", "2020", 2.0, "OECD_DAC"),
                (_STAT_VAR, "providerB", "recipientZ", "2021", 3.0, "OECD_DAC"),
            ),
        )
        classification = classify_columns(table.columns)

        analysis = analyse_pivot(
            table, index="provider", columns="time_period", classification=classification
        )

        assert analysis.unique is False
        assert analysis.duplicate_keys == (("providerA", "2020"),)
        # The varying count covers all three recipients in the frame.
        assert analysis.varying == (("recipient", 3),)
        with pytest.raises(PivotNotUniqueError) as excinfo:
            pivot(
                table,
                index="provider",
                columns="time_period",
                column_cap=100,
                cell_cap=1000,
                classification=classification,
            )
        assert excinfo.value.varying == analysis.varying

    def test_the_only_varying_column_can_be_provenance(self):
        table = Table(
            columns=self._ROLE_COLUMNS,
            rows=(
                (_STAT_VAR, "providerA", "recipientX", "2020", 1.0, "OECD_DAC"),
                (_STAT_VAR, "providerA", "recipientX", "2020", 2.0, "OTHER_SRC"),
                (_STAT_VAR, "providerB", "recipientY", "2021", 3.0, "OECD_DAC"),
            ),
        )
        classification = classify_columns(table.columns)

        analysis = analyse_pivot(
            table, index="provider", columns="time_period", classification=classification
        )

        assert analysis.unique is False
        # Only provenance varies within the duplicate group.
        assert analysis.varying == (("provenance", 2),)

    def test_a_stringification_collision_is_not_unique_with_no_varying_column(self):
        table = Table(
            columns=self._COLUMNS,
            rows=(("country/KEN", 2020, 1.0), ("country/KEN", "2020", 2.0)),
        )
        classification = classify_columns(table.columns)

        analysis = analyse_pivot(
            table, index="entity", columns="time_period", classification=classification
        )

        assert analysis.unique is False
        assert analysis.duplicate_keys == ()
        assert analysis.stringification_collisions == ((2020, "2020"),)
        # Stringification makes the pivot non-unique; no duplicate group exists for `varying`.
        assert analysis.varying == ()

    def test_a_non_homogeneous_pair_reports_offenders_and_stays_unique(self):
        columns = ("entity", "time_period", "value", "unit", "provenance")
        table = Table(
            columns=columns,
            rows=(
                ("country/KEN", "2020", 1.0, "USD", "OECD_DAC"),
                ("country/KEN", "2021", 2.0, "EUR", "OECD_DAC"),
            ),
        )
        classification = classify_columns(table.columns)

        analysis = analyse_pivot(
            table, index="entity", columns="time_period", classification=classification
        )

        assert analysis.unique is True
        assert analysis.homogeneous is False
        assert analysis.offenders == (("unit", "country/KEN", ("EUR", "USD")),)
        assert analysis.varying == ()


class TestApplyReshapesOrder:
    def test_a_groupby_collapsed_axis_is_unknown_to_a_later_selection(self):
        table = Table(
            columns=_BASE_COLUMNS,
            rows=(_row("country/KEN", "2020", 10.0), _row("country/TGO", "2020", 20.0)),
        )
        plan = ReshapePlan(
            groupby=GroupbyPlan(axis="entity", summable_axes=frozenset({"entity"})),
            select=("entity", "time_period", "value"),
        )

        with pytest.raises(ColumnUnknownError) as excinfo:
            apply_reshapes(table, plan=plan)

        assert excinfo.value.column == "entity"

    def test_a_pivot_widened_column_is_not_selectable_selection_ran_first(self):
        table = Table(
            columns=("entity", "time_period", "value"),
            rows=(("country/KEN", "2020", 1.0), ("country/KEN", "2021", 2.0)),
        )
        plan = ReshapePlan(
            select=("2020",),
            pivot=PivotPlan(index="entity", columns="time_period", column_cap=10, cell_cap=1_000),
        )

        with pytest.raises(ColumnUnknownError) as excinfo:
            apply_reshapes(table, plan=plan)

        assert excinfo.value.column == "2020"

    def test_the_full_chain_runs_groupby_then_select_then_enrich_then_pivot(self):
        # KEN's two donor values sum to 10.0 before pivoting.
        columns = (
            "stat_var",
            "entity",
            "donor",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        )
        table = Table(
            columns=columns,
            rows=(
                (
                    _STAT_VAR,
                    "country/KEN",
                    "donor/A",
                    "2020",
                    6.0,
                    "USD",
                    "Standard",
                    "OECD_DAC",
                    None,
                ),
                (
                    _STAT_VAR,
                    "country/KEN",
                    "donor/B",
                    "2020",
                    4.0,
                    "USD",
                    "Standard",
                    "OECD_DAC",
                    None,
                ),
                (
                    _STAT_VAR,
                    "country/KEN",
                    "donor/A",
                    "2021",
                    5.0,
                    "USD",
                    "Standard",
                    "OECD_DAC",
                    None,
                ),
                (
                    _STAT_VAR,
                    "country/TGO",
                    "donor/A",
                    "2020",
                    20.0,
                    "USD",
                    "Standard",
                    "OECD_DAC",
                    None,
                ),
                (
                    _STAT_VAR,
                    "country/TGO",
                    "donor/A",
                    "2021",
                    8.0,
                    "USD",
                    "Standard",
                    "OECD_DAC",
                    None,
                ),
            ),
        )
        plan = ReshapePlan(
            groupby=GroupbyPlan(axis="donor", summable_axes=frozenset({"donor"})),
            select=("stat_var", "entity", "time_period", "value", "unit", "provenance"),
            enrichment=EnrichmentPlan(
                requests=(("entity", "iso3"),),
                lookup={"country/KEN": {"iso3": "KEN"}, "country/TGO": {"iso3": "TGO"}},
            ),
            pivot=PivotPlan(index="entity", columns="time_period", column_cap=10, cell_cap=1_000),
        )

        result, warnings = apply_reshapes(table, plan=plan)

        # Pivot output keeps the index and period columns; enrichment columns are dropped.
        assert result.columns == ("entity", "2020", "2021")
        assert set(result.rows) == {
            ("country/KEN", 10.0, 5.0),
            ("country/TGO", 20.0, 8.0),
        }
        assert warnings == ()

    def test_decode_runs_before_groupby_so_the_decoded_column_is_already_present(self):
        # Both decoded rows share a purpose, so the entity rollup sums 6.0 and 4.0.
        table = Table(
            columns=("stat_var", "entity", "time_period", "value"),
            rows=(
                ("dcid:SV_A", "country/KEN", "2020", 6.0),
                ("dcid:SV_A", "country/TGO", "2020", 4.0),
            ),
        )
        plan = ReshapePlan(
            decode=DecodePlan(
                requests=(("stat_var", "crsPurpose"),),
                lookup={"dcid:SV_A": {"crsPurpose": "Basic health care"}},
            ),
            groupby=GroupbyPlan(axis="entity", summable_axes=frozenset({"entity"})),
        )

        result, warnings = apply_reshapes(table, plan=plan)

        assert result.columns == ("stat_var", "time_period", "value", "stat_var_crsPurpose")
        assert result.rows == (("dcid:SV_A", "2020", 10.0, "Basic health care"),)
        assert warnings == ()

    def test_a_decoded_column_is_free_to_drop_via_selection(self):
        # Selection uses the pre-decode classification, so the added column can be dropped.
        table = Table(
            columns=("stat_var", "time_period", "value"),
            rows=(("dcid:SV_A", "2020", 6.0),),
        )
        plan = ReshapePlan(
            decode=DecodePlan(
                requests=(("stat_var", "crsPurpose"),),
                lookup={"dcid:SV_A": {"crsPurpose": "Basic health care"}},
            ),
            select=("stat_var", "time_period", "value"),
        )

        result, _warnings = apply_reshapes(table, plan=plan)

        assert result.columns == ("stat_var", "time_period", "value")

    def test_a_decoded_column_is_selectable_like_any_other(self):
        table = Table(
            columns=("stat_var", "time_period", "value"),
            rows=(("dcid:SV_A", "2020", 6.0),),
        )
        plan = ReshapePlan(
            decode=DecodePlan(
                requests=(("stat_var", "crsPurpose"),),
                lookup={"dcid:SV_A": {"crsPurpose": "Basic health care"}},
            ),
            select=("stat_var", "time_period", "stat_var_crsPurpose"),
        )

        result, _warnings = apply_reshapes(table, plan=plan)

        assert result.columns == ("stat_var", "time_period", "stat_var_crsPurpose")
        assert result.rows == (("dcid:SV_A", "2020", "Basic health care"),)


def _random_table(rng: random.Random) -> Table:
    """Builds a unique-grain table with shared facets and distinct entities."""
    periods = ["2019", "2020", "2021"]
    entity_counter = 0
    rows = []
    for period in rng.sample(periods, k=rng.randint(1, len(periods))):
        for _ in range(rng.randint(1, 4)):
            entity = f"country/E{entity_counter}"
            entity_counter += 1
            value = round(rng.uniform(-50.0, 50.0), 2)
            rows.append(_row(entity, period, value))
    rng.shuffle(rows)
    return Table(columns=_BASE_COLUMNS, rows=tuple(rows))


def _random_pivot_table(rng: random.Random) -> Table:
    """Builds a three-column table with unique entity-period keys."""
    entities = [f"country/E{i}" for i in range(4)]
    periods = ["2018", "2019", "2020"]
    pairs = [(entity, period) for entity in entities for period in periods]
    rng.shuffle(pairs)
    chosen = pairs[: rng.randint(3, len(pairs))]
    rows = tuple((entity, period, round(rng.uniform(-10.0, 10.0), 2)) for entity, period in chosen)
    return Table(columns=("entity", "time_period", "value"), rows=rows)


class TestPropertyReshapesPreserveValues:
    def test_select_columns_is_a_positional_projection(self):
        rng = random.Random(20260802)
        for _ in range(200):
            table = _random_table(rng)
            classification = classify_columns(table.columns)
            optional = [c for c in table.columns if c not in classification.non_droppable_columns]
            rng.shuffle(optional)
            dropped = set(optional[: rng.randint(0, len(optional))])
            keep = [c for c in table.columns if c not in dropped]

            result = select_columns(table, keep=tuple(keep), classification=classification)

            assert result.columns == tuple(keep)
            index_of = {name: i for i, name in enumerate(table.columns)}
            for original_row, result_row in zip(table.rows, result.rows, strict=True):
                for column, value in zip(result.columns, result_row, strict=True):
                    assert value == original_row[index_of[column]]

    def test_groupby_sum_partitions_the_input_values(self):
        rng = random.Random(4200802)
        for _ in range(150):
            table = _random_table(rng)
            classification = classify_columns(table.columns)

            result = groupby_sum(
                table,
                axis="entity",
                classification=classification,
                summable_axes=frozenset({"entity"}),
            )

            value_pos = table.columns.index("value")
            result_value_pos = result.columns.index("value")
            assert sum(row[result_value_pos] for row in result.rows) == pytest.approx(
                sum(row[value_pos] for row in table.rows)
            )
            # Each period total matches the input rows for that period.
            period_pos = table.columns.index("time_period")
            result_period_pos = result.columns.index("time_period")
            for result_row in result.rows:
                period = result_row[result_period_pos]
                expected = sum(row[value_pos] for row in table.rows if row[period_pos] == period)
                assert result_row[result_value_pos] == pytest.approx(expected)

    def test_pivot_preserves_every_present_cell_value(self):
        rng = random.Random(99001)
        for _ in range(150):
            table = _random_pivot_table(rng)
            classification = classify_columns(table.columns)

            result = pivot(
                table,
                index="entity",
                columns="time_period",
                column_cap=1_000,
                cell_cap=1_000_000,
                classification=classification,
            )

            expected = {(entity, period): value for entity, period, value in table.rows}
            period_columns = result.columns[1:]
            for row in result.rows:
                entity = row[0]
                for period, value in zip(period_columns, row[1:], strict=True):
                    assert value == expected.get((entity, period))
