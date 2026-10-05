"""Tests for `build_decode_plan` use a synthetic `CodeIndex` to pin declared and absent
readings, including the distinction between bare index dcids and prefixed frame refs.
"""

from __future__ import annotations

from typing import cast

from code_index import AbsentReading, CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.decode import build_decode_plan
from materialise.reshape import Table

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
_FUTURE_A = CodeRow(
    dcid="dcid:FutureCodeA",
    code_system="futureCodeSystem",
    code="A",
    label="Future Value A",
    synonyms=(),
    parent=None,
)

_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=False,
        absent_reading="not_declared",
        plain_name="sector",
    ),
    # A different reading proves the value comes from the declaration. The cast bypasses
    # the closed production literal for this fixture only.
    CodeSystemDeclaration(
        code_system="futureCodeSystem",
        hierarchical=False,
        disjoint_from_children=False,
        summable=False,
        absent_reading=cast(AbsentReading, "unmeasured"),
        plain_name="axis",
    ),
)


def _index(*, statvars: tuple[StatVarRow, ...]) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _FUTURE_A),
        statvars=statvars,
    )


def _statvar(dcid: str, *, constraint_properties: dict[str, str]) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties=constraint_properties,
    )


_TABLE_COLUMNS = ("stat_var", "financingProvider", "financingRecipient", "time_period", "value")


def _table(rows: tuple[tuple[object, ...], ...]) -> Table:
    return Table(columns=_TABLE_COLUMNS, rows=rows)


def test_no_stat_var_column_at_all_returns_none():
    table = Table(columns=("entity", "value"), rows=(("country/KEN", 1.0),))
    index = _index(statvars=())

    assert build_decode_plan(table, index=index) is None


def test_a_frame_whose_statvars_declare_no_known_code_system_decodes_nothing():
    table = _table((("dcid:SV_unknown", "p1", "r1", "2020", 1.0),))
    index = _index(statvars=())  # The index has no row for this StatVar.

    assert build_decode_plan(table, index=index) is None


def test_happy_path_resolves_a_label_for_a_declared_code():
    table = _table((("dcid:SV_A", "p1", "r1", "2020", 1.0),))
    index = _index(
        statvars=(_statvar("SV_A", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),)
    )

    plan = build_decode_plan(table, index=index)

    assert plan is not None
    assert plan.requests == (("stat_var", "crsPurpose"),)
    assert plan.lookup == {"dcid:SV_A": {"crsPurpose": "Basic health care"}}


def test_a_property_not_declared_on_the_row_reads_as_the_declaration_own_absent_reading():
    # The sibling declares `crsPurpose`. This row reads its declared absence value.
    table_two_rows = _table(
        (
            ("dcid:SV_A", "p1", "r1", "2020", 1.0),
            ("dcid:SV_B", "p1", "r2", "2021", 2.0),
        )
    )
    index_two = _index(
        statvars=(
            _statvar("SV_A", constraint_properties={}),
            _statvar("SV_B", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),
        )
    )

    plan = build_decode_plan(table_two_rows, index=index_two)

    assert plan is not None
    assert plan.requests == (("stat_var", "crsPurpose"),)
    assert plan.lookup["dcid:SV_A"] == {"crsPurpose": "not_declared"}
    assert plan.lookup["dcid:SV_B"] == {"crsPurpose": "Basic health care"}


def test_the_absent_reading_value_placed_in_the_cell_is_read_generically_not_hardcoded():
    # This declaration uses "unmeasured", so a hard-coded CRS sentinel fails.
    table_two_rows = _table(
        (
            ("dcid:SV_A", "p1", "r1", "2020", 1.0),
            ("dcid:SV_B", "p1", "r2", "2021", 2.0),
        )
    )
    index = _index(
        statvars=(
            _statvar("SV_A", constraint_properties={}),
            _statvar("SV_B", constraint_properties={"futureCodeSystem": _FUTURE_A.dcid}),
        )
    )

    plan = build_decode_plan(table_two_rows, index=index)

    assert plan is not None
    assert plan.lookup["dcid:SV_A"] == {"futureCodeSystem": "unmeasured"}
    assert plan.lookup["dcid:SV_B"] == {"futureCodeSystem": "Future Value A"}


def test_a_declared_value_the_index_carries_no_code_row_for_is_a_genuine_miss():
    # The declared code has no `CodeRow`. Omit it so decode handles the miss.
    table = _table((("dcid:SV_A", "p1", "r1", "2020", 1.0),))
    index = _index(
        statvars=(_statvar("SV_A", constraint_properties={"crsPurpose": "dcid:NoSuchCodeRow"}),)
    )

    plan = build_decode_plan(table, index=index)

    assert plan is not None
    assert plan.lookup["dcid:SV_A"] == {}


def test_a_constraint_property_the_index_does_not_declare_is_never_requested():
    # Without a matching `CodeSystemDeclaration`, `crsMysteryAxis` has no code system to decode.
    table = _table((("dcid:SV_A", "p1", "r1", "2020", 1.0),))
    index = _index(
        statvars=(_statvar("SV_A", constraint_properties={"crsMysteryAxis": "dcid:Whatever"}),)
    )

    assert build_decode_plan(table, index=index) is None


def test_a_statvar_absent_from_the_index_entirely_is_a_genuine_miss_for_every_request():
    table = _table(
        (
            ("dcid:SV_known", "p1", "r1", "2020", 1.0),
            ("dcid:SV_ghost", "p1", "r2", "2021", 2.0),
        )
    )
    index = _index(
        statvars=(_statvar("SV_known", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),)
    )

    plan = build_decode_plan(table, index=index)

    assert plan is not None
    assert plan.lookup["dcid:SV_known"] == {"crsPurpose": "Basic health care"}
    assert plan.lookup["dcid:SV_ghost"] == {}


def test_extra_refs_decode_a_zero_row_member_absent_from_the_table():
    # The groupby population includes this resolved StatVar with zero rows.
    table = _table((("dcid:SV_A", "p1", "r1", "2020", 1.0),))
    index = _index(
        statvars=(
            _statvar("SV_A", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),
            _statvar("SV_zero", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),
        )
    )

    plan = build_decode_plan(table, index=index, extra_refs=("dcid:SV_zero",))

    assert plan is not None
    assert plan.lookup["dcid:SV_A"] == {"crsPurpose": "Basic health care"}
    assert plan.lookup["dcid:SV_zero"] == {"crsPurpose": "Basic health care"}


def test_extra_refs_alone_with_no_stat_var_column_still_builds_a_plan():
    table = Table(columns=("entity", "value"), rows=(("country/KEN", 1.0),))
    index = _index(
        statvars=(_statvar("SV_zero", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),)
    )

    plan = build_decode_plan(table, index=index, extra_refs=("dcid:SV_zero",))

    assert plan is not None
    assert plan.lookup == {"dcid:SV_zero": {"crsPurpose": "Basic health care"}}


def test_an_index_gap_extra_ref_decodes_the_same_way_a_table_row_would():
    table = _table((("dcid:SV_A", "p1", "r1", "2020", 1.0),))
    index = _index(
        statvars=(_statvar("SV_A", constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid}),)
    )

    plan = build_decode_plan(table, index=index, extra_refs=("dcid:SV_ghost",))

    assert plan is not None
    assert plan.lookup["dcid:SV_ghost"] == {}


def test_two_code_systems_produce_two_sorted_requests():
    table = _table((("dcid:SV_A", "p1", "r1", "2020", 1.0),))
    index = _index(
        statvars=(
            _statvar(
                "SV_A",
                constraint_properties={
                    "futureCodeSystem": _FUTURE_A.dcid,
                    "crsPurpose": _PURPOSE_HEALTH.dcid,
                },
            ),
        )
    )

    plan = build_decode_plan(table, index=index)

    assert plan is not None
    assert plan.requests == (("stat_var", "crsPurpose"), ("stat_var", "futureCodeSystem"))
