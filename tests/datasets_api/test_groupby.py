"""Tests the groupby door's axis checks, StatVar collision refusal, and grammar-frame rules.
A synthetic non-summable code system covers a declaration no live CRS system uses.
"""

from __future__ import annotations

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.groupby import (
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
