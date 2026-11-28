"""Tests the groupby door's axis checks, StatVar collision refusal, and grammar-frame rules.
A synthetic non-summable code system covers a declaration no live CRS system uses.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import datasets_api.groupby as groupby_module
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.groupby import (
    ReporterCheck,
    collapse_axes,
    contributing_dcids,
    kept_axes,
    refuse_decode_gaps,
    refuse_stat_var_collision,
    refuse_unknown_by_axes,
    yields_grammar_frame,
)
from materialise.reshape import RollupNotSummableError, Table
from materialise.roles import KNOWN_ROLE_NAMES

_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-09T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)

_PURPOSE_HEALTH = CodeRow(
    dcid="dcid:CRSPurposeCode122",
    code_system="crsPurpose",
    code="122",
    label="Basic health care",
    synonyms=(),
    parent=None,
)
_PURPOSE_MALARIA = CodeRow(
    dcid="dcid:CRSPurposeCode12250",
    code_system="crsPurpose",
    code="12250",
    label="Malaria control",
    synonyms=(),
    parent=None,
)
_FINANCE_GRANT = CodeRow(
    dcid="dcid:CRSTypeOfFinanceCode110",
    code_system="crsTypeOfFinance",
    code="110",
    label="Standard grant",
    synonyms=(),
    parent=None,
)

_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="sector",
    ),
    CodeSystemDeclaration(
        code_system="crsTypeOfFinance",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="instrument",
    ),
    # No live CRS declaration is non-summable, so this synthetic system covers that branch.
    CodeSystemDeclaration(
        code_system="futureCodeSystem",
        hierarchical=False,
        disjoint_from_children=False,
        summable=False,
        absent_reading="not_declared",
        plain_name="axis",
    ),
)


def _index(*, statvars: tuple[StatVarRow, ...]) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT),
        statvars=statvars,
    )


def _statvar(
    dcid: str,
    *,
    constraint_properties: dict[str, str],
    measurement_qualifier: str | None = None,
) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=measurement_qualifier,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties=constraint_properties,
    )


_TABLE_COLUMNS = ("stat_var", "financingProvider", "financingRecipient", "time_period", "value")


def _table(rows: tuple[tuple[object, ...], ...]) -> Table:
    return Table(columns=_TABLE_COLUMNS, rows=rows)


class TestCollapseAxes:
    def test_naming_one_axis_to_keep_collapses_the_other_declared_summable_decoded_axes(self):
        collapsed = collapse_axes(
            by=["crsPurpose"],
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance"),
            declarations={d.code_system: d for d in _DECLARATIONS},
        )

        assert collapsed == ("crsTypeOfFinance",)

    def test_naming_every_declared_summable_decoded_axis_collapses_nothing(self):
        collapsed = collapse_axes(
            by=["crsPurpose", "crsTypeOfFinance"],
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance"),
            declarations={d.code_system: d for d in _DECLARATIONS},
        )

        assert collapsed == ()

    def test_a_declared_not_summable_axis_never_collapses_even_when_decoded_and_unkept(self):
        # `summable=False` keeps this decoded, unselected axis out of the collapse set.
        collapsed = collapse_axes(
            by=["crsPurpose"],
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance", "futureCodeSystem"),
            declarations={d.code_system: d for d in _DECLARATIONS},
        )

        assert collapsed == ("crsTypeOfFinance",)

    def test_a_by_entry_naming_an_axis_this_frame_never_decoded_is_not_an_error(self):
        collapsed = collapse_axes(
            by=["crsPurpose", "crsCooperationModality"],
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance"),
            declarations={d.code_system: d for d in _DECLARATIONS},
        )

        assert collapsed == ("crsTypeOfFinance",)


class TestRefuseStatVarCollision:
    def test_no_stat_var_column_is_a_no_op(self):
        table = Table(columns=("entity", "value"), rows=(("country/KEN", 1.0),))
        index = _index(statvars=())

        refuse_stat_var_collision(table, index=index, decoded_code_systems=("crsPurpose",))

    def test_distinct_measures_sharing_every_decoded_axis_value_refuse(self):
        # Gross and net StatVars share decoded axes. `measurement_qualifier` distinguishes them,
        # and the frame omits that field.
        table = _table(
            (
                ("dcid:SV_gross", "country/USA", "country/KEN", "2022", 10.0),
                ("dcid:SV_net", "country/USA", "country/KEN", "2022", 4.0),
            )
        )
        index = _index(
            statvars=(
                _statvar(
                    "SV_gross",
                    constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid},
                    measurement_qualifier="gross",
                ),
                _statvar(
                    "SV_net",
                    constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid},
                    measurement_qualifier="net",
                ),
            )
        )

        try:
            refuse_stat_var_collision(table, index=index, decoded_code_systems=("crsPurpose",))
        except RollupNotSummableError as exc:
            assert exc.axis == "stat_var"
            assert exc.precondition == "stat_var_ambiguous"
        else:
            raise AssertionError("expected RollupNotSummableError")

    def test_distinct_measures_differing_on_a_decoded_axis_do_not_collide(self):
        table = _table(
            (
                ("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0),
                ("dcid:SV_malaria", "country/USA", "country/KEN", "2022", 7.0),
            )
        )
        index = _index(
            statvars=(
                _statvar("SV_health", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),
                _statvar("SV_malaria", constraint_properties={"crsPurpose": _PURPOSE_MALARIA.dcid}),
            )
        )

        refuse_stat_var_collision(table, index=index, decoded_code_systems=("crsPurpose",))

    def test_the_same_dcid_repeated_across_rows_is_not_a_collision(self):
        table = _table(
            (
                ("dcid:SV_health", "country/USA", "country/KEN", "2021", 5.0),
                ("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0),
            )
        )
        index = _index(
            statvars=(
                _statvar("SV_health", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),
            )
        )

        refuse_stat_var_collision(table, index=index, decoded_code_systems=("crsPurpose",))


class TestRefuseUnknownByAxes:
    def test_a_declared_not_summable_axis_refuses_too(self):
        # `futureCodeSystem` declares a non-summable axis, which this function refuses.
        try:
            refuse_unknown_by_axes(
                by=["futureCodeSystem"], declarations={d.code_system: d for d in _DECLARATIONS}
            )
        except RollupNotSummableError as exc:
            assert exc.axis == "futureCodeSystem"
            assert exc.precondition == "not_declared_summable"
        else:
            raise AssertionError("expected RollupNotSummableError")

    def test_a_name_no_deployment_declares_at_all_refuses(self):
        try:
            refuse_unknown_by_axes(
                by=["crsPurpose", "crsTypoedAxis"],
                declarations={d.code_system: d for d in _DECLARATIONS},
            )
        except RollupNotSummableError as exc:
            assert exc.axis == "crsTypoedAxis"
            assert exc.precondition == "not_declared_summable"
        else:
            raise AssertionError("expected RollupNotSummableError")

    @pytest.mark.parametrize(
        "by",
        [
            pytest.param(["crsPurpose"], id="declared-summable-axis"),
            *(pytest.param([role], id=f"role-{role}") for role in sorted(KNOWN_ROLE_NAMES)),
            pytest.param(["stat_var"], id="stat_var"),
            # `value` is a real frame1 column even though grouping by it changes nothing.
            *(
                pytest.param([column], id=f"column-{column}")
                for column in (
                    "time_period",
                    "value",
                    "unit",
                    "measurementMethod",
                    "provenance",
                    "status",
                )
            ),
            # Role names and code systems can appear together in `by`.
            pytest.param(
                ["financingRecipient", "crsPurpose"],
                id="role-alongside-summable-axis",
            ),
        ],
    )
    def test_known_axes_are_allowed(self, by):
        refuse_unknown_by_axes(by=by, declarations={d.code_system: d for d in _DECLARATIONS})

    def test_a_role_shaped_typo_still_refuses(self):
        # This typo matches no known role, frame column, `stat_var`, or summable code system.
        try:
            refuse_unknown_by_axes(
                by=["finiancingRecipient"], declarations={d.code_system: d for d in _DECLARATIONS}
            )
        except RollupNotSummableError as exc:
            assert exc.axis == "finiancingRecipient"
            assert exc.precondition == "not_declared_summable"
        else:
            raise AssertionError("expected RollupNotSummableError")


class TestRefuseDecodeGaps:
    def test_every_ref_fully_decoded_is_a_no_op(self):
        refuse_decode_gaps(
            lookup={"dcid:SV_A": {"crsPurpose": "Basic health care"}},
            refs=("dcid:SV_A",),
            code_systems=("crsPurpose",),
        )

    def test_a_ref_missing_from_the_lookup_entirely_refuses(self):
        try:
            refuse_decode_gaps(
                lookup={"dcid:SV_A": {"crsPurpose": "Basic health care"}},
                refs=("dcid:SV_A", "dcid:SV_ghost"),
                code_systems=("crsPurpose",),
            )
        except RollupNotSummableError as exc:
            assert exc.axis == "stat_var"
            assert exc.precondition == "index_gap"
            assert "dcid:SV_ghost" in str(exc)
        else:
            raise AssertionError("expected RollupNotSummableError")

    def test_a_ref_present_but_missing_one_code_system_cell_refuses(self):
        # The ref has a `crsPurpose` cell but lacks `crsTypeOfFinance`.
        try:
            refuse_decode_gaps(
                lookup={"dcid:SV_A": {"crsPurpose": "Basic health care"}},
                refs=("dcid:SV_A",),
                code_systems=("crsPurpose", "crsTypeOfFinance"),
            )
        except RollupNotSummableError as exc:
            assert exc.precondition == "index_gap"
            assert "crsTypeOfFinance" in str(exc)
        else:
            raise AssertionError("expected RollupNotSummableError")


class TestYieldsGrammarFrame:
    """The sum yields a grammar frame when the frame decodes a summable axis and every decoded
    axis the sum leaves alone holds one value across the resolved members. A summable axis named
    in `by` stays on that frame as a grain column (`kept_axes`)."""

    def test_by_keeping_a_decoded_axis_still_yields_a_frame_that_keeps_it(self):
        declarations = {d.code_system: d for d in _DECLARATIONS}
        assert yields_grammar_frame(
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance"),
            declarations=declarations,
            lookup={},
        )
        assert kept_axes(
            by=["time_period", "crsPurpose", "crsCooperationModality"],
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance"),
            declarations=declarations,
        ) == ("crsPurpose",)

    def test_nothing_decoded_at_all_never_yields_a_frame(self):
        # `stat_var` joins a collapse only when at least one CRS axis is decoded.
        assert not yields_grammar_frame(
            decoded_code_systems=(),
            declarations={d.code_system: d for d in _DECLARATIONS},
            lookup={},
        )

    def test_an_unsummed_axis_holding_one_value_yields_a_frame(self):
        # Every member carries the same `futureCodeSystem` value, so the column the sum
        # leaves alone is constant and the rows are one series per entity tuple.
        assert yields_grammar_frame(
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance", "futureCodeSystem"),
            declarations={d.code_system: d for d in _DECLARATIONS},
            lookup={
                "dcid:SV_A": {"crsPurpose": "Health", "futureCodeSystem": "One"},
                "dcid:SV_B": {"crsPurpose": "Education", "futureCodeSystem": "One"},
            },
        )

    def test_an_unsummed_axis_holding_two_values_yields_no_frame(self):
        # The sum groups by `futureCodeSystem`, so two values leave two series per entity
        # tuple and the rows are no grammar frame.
        assert not yields_grammar_frame(
            decoded_code_systems=("crsPurpose", "crsTypeOfFinance", "futureCodeSystem"),
            declarations={d.code_system: d for d in _DECLARATIONS},
            lookup={
                "dcid:SV_A": {"crsPurpose": "Health", "futureCodeSystem": "One"},
                "dcid:SV_B": {"crsPurpose": "Health", "futureCodeSystem": "Two"},
            },
        )


class TestContributingDcids:
    def test_zero_row_members_are_absent_even_when_the_column_has_other_rows(self):
        table = _table(
            (
                ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0),
                ("dcid:SV_A", "country/USA", "country/TZA", "2022", 3.0),
                ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0),
            )
        )
        assert contributing_dcids(table) == frozenset({"dcid:SV_A", "dcid:SV_B"})

    def test_no_stat_var_column_at_all_is_empty(self):
        table = Table(columns=("value",), rows=((10.0,),))
        assert contributing_dcids(table) == frozenset()

    def test_no_rows_at_all_is_empty(self):
        assert contributing_dcids(_table(())) == frozenset()


_NORTH = "provenance/LedgerNorth"
_SOUTH = "provenance/LedgerSouth"
_EAST = "provenance/LedgerEast"


def _reporter_check(
    rows: list[tuple[str, str, str, str | None]],
    *,
    members: tuple[str, ...],
    reads_zero: bool = True,
) -> ReporterCheck:
    """A check over `rows` of `(StatVar, recipient, year, reporting organisation)`, expecting
    every one of `members` in each total, and reading a missing member as zero in every year
    when `reads_zero`, else in none."""
    check = ReporterCheck(
        cell_columns=("financingRecipient", "time_period"),
        members=members,
        lookup={},
        kept_code_systems=(),
        reads_zero=lambda provenance, year: reads_zero,
    )
    stat_vars, recipients, years, reporters = (list(column) for column in zip(*rows, strict=True))
    check.note(
        pa.array(stat_vars, pa.string()),
        pa.array(reporters, pa.string()),
        pa.table({"financingRecipient": recipients, "time_period": years}),
    )
    return check


class TestReporterCheck:
    @pytest.mark.parametrize(
        ("rows", "summed"),
        [
            pytest.param(
                [
                    ("dcid:A", "country/KEN", "2022", _NORTH),
                    ("dcid:B", "country/KEN", "2022", None),
                ],
                0,
                id="null-reporter-mixes-with-none",
            ),
            pytest.param(
                [
                    ("dcid:A", "country/KEN", "2022", _NORTH),
                    ("dcid:B", "country/KEN", "2022", _SOUTH),
                    ("dcid:A", "country/UGA", "2022", _EAST),
                ],
                2,
                id="non-mixing-reporter-uncounted",
            ),
        ],
    )
    def test_summed_reporters_counts_the_reporters_of_mixing_totals(self, rows, summed):
        check = _reporter_check(rows, members=("dcid:A", "dcid:B"))

        assert check.summed_reporters() == summed

    @pytest.mark.parametrize("reads_zero", [True, False], ids=["reads-zero", "reads-unknown"])
    def test_only_totals_missing_a_member_are_read_into_python(self, monkeypatch, reads_zero):
        """Of 20,000 recipients' totals, Kenya's 2022 total alone lacks member B. Python reads
        that total's one row and the one row of Kenya's own B series, and no other, and the
        check refuses that total where its absence reads unknown."""
        rows = [
            (stat_var, f"country/R{i}", year, reporter)
            for i in range(20_000)
            for year in ("2021", "2022")
            for stat_var, reporter in (("dcid:A", _NORTH), ("dcid:B", _SOUTH))
        ]
        rows += [
            ("dcid:A", "country/KEN", "2021", _NORTH),
            ("dcid:B", "country/KEN", "2021", _SOUTH),
            ("dcid:A", "country/KEN", "2022", _NORTH),
        ]
        check = _reporter_check(rows, members=("dcid:A", "dcid:B"), reads_zero=reads_zero)
        read: list[int] = []
        python_rows = groupby_module._python_rows

        def counting(table: pa.Table) -> list[dict[str, object]]:
            read.append(table.num_rows)
            return python_rows(table)

        monkeypatch.setattr(groupby_module, "_python_rows", counting)

        if reads_zero:
            check.refuse_unknown_absence()
        else:
            with pytest.raises(RollupNotSummableError) as refused:
                check.refuse_unknown_absence()
            assert refused.value.precondition == "missing_members"
            assert "'dcid:B'" in str(refused.value)
            assert "country/KEN" in str(refused.value)

        assert read == [1, 1]
