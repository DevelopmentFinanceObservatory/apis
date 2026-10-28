"""Checks the Arrow groupby fold and sum against the row engines they stand in for."""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
import pytest

from materialise import groupby_columnar, reshape
from materialise.errors import MaterialisationUsageError
from materialise.frame_build import build_frame
from materialise.groupby_columnar import (
    decode_table,
    fold_groupby_sum_table,
    groupby_sum_table,
    rows_table,
)
from materialise.reshape import (
    ExpectedPopulation,
    GroupbyPlan,
    ReshapePlan,
    apply_reshapes,
    classify_columns,
    decode_columns,
    fold_groupby_sum_chunks,
)

_LOOKUP = {
    "dcid:SV_A": {"sector": "health"},
    "dcid:SV_B": {"sector": "health"},
    "dcid:SV_C": {"sector": "water"},
    "dcid:SV_D": {"sector": "health"},
}
_SUMMABLE = frozenset({"stat_var", "stat_var_sector"})
_USD = {"currency": "USD", "prices": "constant", "base_year": 2024}

# One group sums 1e16, 1.0 and -1e16. Left to right that is 0.0 and Python's compensated `sum`
# gives 1.0, so the fold and the whole-table sum must each keep their own order.
_CANCELLING = {"dcid:SV_A": 1e16, "dcid:SV_B": 1.0, "dcid:SV_D": -1e16}


def _row(stat_var: str, recipient: str, year: str, value: float, **cells: object) -> dict:
    return {
        "stat_var": stat_var,
        "recipient": recipient,
        "time_period": year,
        "value": value,
        "unit": _USD,
        "measurementMethod": "Reported",
        **cells,
    }


def _plain_rows() -> list[dict]:
    rows = [
        _row(stat_var, recipient, year, 0.1 * (index + 1) + 1 / 3)
        for index, (stat_var, recipient, year) in enumerate(
            (s, r, y)
            for s in ("dcid:SV_A", "dcid:SV_B", "dcid:SV_C")
            for r in ("country/KEN", "country/TZA", "country/UGA")
            for y in ("2015", "2016")
        )
    ]
    rows += [_row(s, "country/ZMB", "2015", v) for s, v in _CANCELLING.items()]
    return rows


def _table(rows: list[dict]) -> pa.Table:
    table = build_frame(role_keys=("recipient",), rows=rows).table
    decoded, _ = decode_table(table, requests=(("stat_var", "sector"),), lookup=_LOOKUP)
    return decoded


def _population(members: list[tuple[str, str]]) -> ExpectedPopulation:
    return ExpectedPopulation(columns=("stat_var", "stat_var_sector"), members=frozenset(members))


# Every member the plain rows hold. ZMB's group also holds SV_D, which no group expects.
_FULL_POPULATION = _population(
    [("dcid:SV_A", "health"), ("dcid:SV_B", "health"), ("dcid:SV_C", "water")]
)
_POPULATION_WITH_ABSENT = _population([*_FULL_POPULATION.members, ("dcid:SV_E", "water")])


def _twice(table: pa.Table) -> pa.Table:
    return pa.concat_tables([table.slice(0, 1), table.slice(0, 1), table.slice(1)])


def _reversed(table: pa.Table) -> pa.Table:
    return table.take(pa.array(range(table.num_rows - 1, -1, -1)))


_CASES = [
    pytest.param(_plain_rows(), None, {}, id="plain"),
    pytest.param(
        _plain_rows(), None, {"expected_population": _FULL_POPULATION}, id="population-whole"
    ),
    pytest.param(
        _plain_rows(),
        None,
        {"expected_population": _POPULATION_WITH_ABSENT},
        id="absent-member",
    ),
    pytest.param(
        _plain_rows(),
        None,
        {
            "expected_population": _POPULATION_WITH_ABSENT,
            "missing_declared_as_zero": True,
        },
        id="absent-member-reads-zero",
    ),
    pytest.param(
        _plain_rows(),
        None,
        {"expected_population": _population([("dcid:SV_A", "health")])},
        id="undescribed-group",
    ),
    pytest.param(
        [
            *_plain_rows(),
            _row("dcid:SV_A", "country/RWA", "2015", 1.0),
            _row("dcid:SV_B", "country/RWA", "2015", 2.0, measurementMethod="Modelled"),
        ],
        None,
        {},
        id="estimated-total",
    ),
    pytest.param(
        [
            *_plain_rows(),
            _row("dcid:SV_A", "country/RWA", "2015", 1.0),
            _row("dcid:SV_B", "country/RWA", "2015", 2.0, unit="unit/EUR"),
        ],
        None,
        {},
        id="unit-mix",
    ),
    pytest.param(
        [
            *_plain_rows(),
            _row("dcid:SV_A", "country/RWA", "2015", 1.7e308),
            _row("dcid:SV_B", "country/RWA", "2015", 1.7e308),
        ],
        None,
        {},
        id="not-finite",
    ),
    pytest.param(_plain_rows(), _twice, {}, id="repeated-grain"),
    pytest.param(_plain_rows(), _reversed, {}, id="out-of-order"),
]


def _outcome(call: Callable[[], object]) -> tuple:
    """What a call gives, in comparable form: its rows, or its refusal's type, precondition
    and arguments."""
    try:
        result = call()
    except Exception as error:
        return ("raised", type(error), getattr(error, "precondition", None), error.args)
    table = getattr(result, "table", result)
    violation = getattr(result, "violation", None)
    if violation is not None:
        return (
            "violation",
            type(violation),
            getattr(violation, "precondition", None),
            violation.args,
        )
    return ("table", rows_table(table))


@pytest.mark.parametrize(("rows", "rearrange", "options"), _CASES)
def test_the_arrow_fold_gives_the_row_fold_s_outcome(monkeypatch, rows, rearrange, options):
    """The Arrow fold gives the row fold's outcome; when it hands the rows to the row fold,
    it builds them one slice at a time."""
    table = _table(rows)
    if rearrange is not None:
        table = rearrange(table)
    arguments = {
        "axis": "stat_var",
        "classification": classify_columns(tuple(table.column_names)),
        "summable_axes": _SUMMABLE,
        **options,
    }
    monkeypatch.setattr(groupby_columnar, "_FALLBACK_SLICE_ROWS", 4)
    built: list[int] = []
    real_rows = groupby_columnar.canonical_frame_rows
    monkeypatch.setattr(
        groupby_columnar,
        "canonical_frame_rows",
        lambda arrow_table: built.append(arrow_table.num_rows) or real_rows(arrow_table),
    )

    built_by_arrow: list[int] = []

    def arrow_fold():
        try:
            return fold_groupby_sum_table(table, **arguments)
        finally:
            built_by_arrow.extend(built)

    arrow = _outcome(arrow_fold)
    rows_fold = _outcome(lambda: fold_groupby_sum_chunks(iter((rows_table(table),)), **arguments))

    assert arrow == rows_fold
    assert all(rows <= 4 for rows in built_by_arrow)
    if rearrange is _reversed:
        assert arrow[1] is MaterialisationUsageError


@pytest.mark.parametrize(("rows", "rearrange", "options"), _CASES)
def test_the_arrow_sum_gives_groupby_sum_s_outcome(rows, rearrange, options):
    """The second axis of a collapse, `stat_var_sector`, summed over the first fold's groups."""
    table = _table(rows)
    if rearrange is not None:
        table = rearrange(table)
    population = options.get("expected_population")
    arguments = {
        "axis": "stat_var_sector",
        "summable_axes": _SUMMABLE,
        "missing_declared_as_zero": options.get("missing_declared_as_zero", False),
        "expected_population": (
            None
            if population is None
            else ExpectedPopulation(
                columns=("stat_var_sector",),
                members=frozenset((sector,) for _sv, sector in population.members),
            )
        ),
    }
    without_stat_var = table.drop_columns(["stat_var"])

    arrow = _outcome(lambda: groupby_sum_table(without_stat_var, **arguments))
    rows_sum = _outcome(
        lambda: apply_reshapes(
            rows_table(without_stat_var), plan=ReshapePlan(groupby=GroupbyPlan(**arguments))
        )[0]
    )

    assert arrow == rows_sum


def test_each_engine_keeps_its_own_summation_order():
    """The fold adds left to right and the whole-table sum compensates, as the row engines do.
    The cancelling group tells them apart."""
    table = _table([_row(s, "country/ZMB", "2015", v) for s, v in _CANCELLING.items()])

    folded = fold_groupby_sum_table(
        table,
        axis="stat_var",
        classification=classify_columns(tuple(table.column_names)),
        summable_axes=_SUMMABLE,
    ).table
    summed = groupby_sum_table(
        table.drop_columns(["stat_var", "stat_var_sector"]).append_column(
            "stat_var_sector", pa.array(["health", "water", "food"])
        ),
        axis="stat_var_sector",
        summable_axes=_SUMMABLE,
    )

    assert isinstance(folded, pa.Table)
    assert isinstance(summed, pa.Table)
    assert folded.column("value").to_pylist() == [0.0]
    assert summed.column("value").to_pylist() == [1.0]


def test_a_plain_fold_reads_no_row_as_python(monkeypatch):
    """A fold with no refusal folds no row in Python and converts no row to Python objects."""
    table = _table(_plain_rows())
    folded_rows: list[tuple] = []
    converted: list[pa.Table] = []
    monkeypatch.setattr(
        reshape._GroupbySumFold, "process_row", lambda _self, row: folded_rows.append(row)
    )
    real_rows = groupby_columnar.canonical_frame_rows
    monkeypatch.setattr(
        groupby_columnar,
        "canonical_frame_rows",
        lambda arrow_table: converted.append(arrow_table) or real_rows(arrow_table),
    )

    outcome = fold_groupby_sum_table(
        table,
        axis="stat_var",
        classification=classify_columns(tuple(table.column_names)),
        summable_axes=_SUMMABLE,
    )

    assert isinstance(outcome.table, pa.Table)
    # (recipient, year, sector) groups: 3 x 2 x 2, and ZMB's one.
    assert outcome.table.num_rows == 3 * 2 * 2 + 1
    assert folded_rows == []
    assert converted == []


def test_decode_table_gives_decode_columns_cells_and_warnings():
    rows = [*_plain_rows(), _row("dcid:SV_Z", "country/KEN", "2015", 4.0)]
    table = build_frame(role_keys=("recipient",), rows=rows).table
    requests = (("stat_var", "sector"),)

    decoded, warnings = decode_table(table, requests=requests, lookup=_LOOKUP)

    assert (rows_table(decoded), warnings) == decode_columns(
        rows_table(table), requests=requests, lookup=_LOOKUP
    )
    assert [warning["code"] for warning in warnings] == ["enrichment_unavailable"]


def test_the_row_fold_folds_each_slice_before_building_the_next(monkeypatch):
    """The fallback folds a slice's rows before it builds the next slice's, so it holds one
    slice of rows at a time."""
    table = _twice(_table(_plain_rows()))
    events: list[str] = []
    monkeypatch.setattr(groupby_columnar, "_FALLBACK_SLICE_ROWS", 2)
    real_rows = groupby_columnar.canonical_frame_rows
    monkeypatch.setattr(
        groupby_columnar,
        "canonical_frame_rows",
        lambda arrow_table: events.append("build") or real_rows(arrow_table),
    )
    real_fold = reshape._GroupbySumFold.process_row
    monkeypatch.setattr(
        reshape._GroupbySumFold,
        "process_row",
        lambda fold, row: events.append("fold") or real_fold(fold, row),
    )

    fold_groupby_sum_table(
        table,
        axis="stat_var",
        classification=classify_columns(tuple(table.column_names)),
        summable_axes=_SUMMABLE,
    )

    assert events.count("build") > 1
    assert events.index("fold") < events.index("build", 1)
