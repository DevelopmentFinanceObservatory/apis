"""`TableScan` reads Arrow frames and row tables alike, and leaves a column out of the codes
table when one of its cells could not key it. `bundle_codes` reports whether its label read
completed, and names the headings of a pivot that spread a coded column."""

from __future__ import annotations

import pytest

from dc_graph import GraphCapExceededError, GraphDeadlineExceededError, GraphReadError
from materialise.bundle import CodeEntry
from materialise.bundle_columns import TableScan, bundle_codes, column_docs
from materialise.cell_safety import spreadsheet_safe
from materialise.frame_build import build_frame
from materialise.reshape import Table

_ROWS = [
    {
        "stat_var": "dcid:Amount_A",
        "recipient": "country/KEN",
        "time_period": "2015",
        "value": 1.0,
        "unit": "CurrencyUSD",
        "provenance": "OECD_DAC2A",
    },
    {
        "stat_var": "dcid:Amount_B",
        "recipient": "country/TGO",
        "time_period": "2019-06",
        "value": 2.0,
        "unit": "CurrencyUSD",
        "provenance": "OECD_DAC2A",
    },
]


def _scan(table) -> TableScan:
    scan = TableScan(role_columns=("recipient",))
    assert list(scan.observe((table,))) == [table]
    return scan


def _facts(scan: TableScan) -> tuple[object, ...]:
    return (
        scan.rows,
        {column: scan.codes(column) for column in scan.columns},
        {column: scan.empty(column) for column in scan.columns},
        scan.year_window,
    )


_CONVERTED = {"currency": "EUR", "prices": "constant", "base_year": 2024}


@pytest.mark.parametrize(("unit", "unit_codes"), [(None, {"CurrencyUSD"}), (_CONVERTED, set())])
def test_a_frame_and_its_rows_scan_to_the_same_facts(unit, unit_codes):
    """A converted unit is a struct `request` in the frame and a `UnitRequest` in its rows; on
    both paths it keeps the unit column out of the codes table."""
    rows_in = [_ROWS[0], {**_ROWS[1], "unit": unit}] if unit is not None else _ROWS
    frame = build_frame(role_keys=("recipient",), rows=rows_in)
    rows = Table(columns=frame.columns, rows=frame.rows)

    assert _facts(_scan(frame)) == _facts(_scan(rows))
    scan = _scan(frame)
    assert scan.codes("unit") == unit_codes
    assert scan.codes("recipient") == {"country/KEN", "country/TGO"}
    assert scan.codes("value") == frozenset()
    assert scan.year_window == (2015, 2019)
    assert scan.empty("status")
    assert not scan.coded("status")


@pytest.mark.parametrize("cell", ["-negative", 7])
def test_a_column_holding_a_cell_that_cannot_key_the_codes_table_is_not_coded(cell):
    table = Table(
        columns=("stat_var", "unit"),
        rows=(("dcid:Amount_A", "CurrencyUSD"), ("dcid:Amount_A", cell)),
    )
    scan = _scan(table)

    assert not scan.coded("unit")
    assert scan.coded("stat_var")
    codes = bundle_codes((scan,), known_labels={}, lookup=lambda refs: {})
    assert [entry.code for entry in codes.entries] == ["dcid:Amount_A"]


def _fails(error: Exception):
    def lookup(refs):
        raise error

    return lookup


_NAMED = {"Amount_A": {"label": "Amount A"}, "country/KEN": {"label": "Kenya", "type": "Country"}}


@pytest.mark.parametrize(
    ("lookup", "complete", "kenya"),
    [
        (lambda refs: _NAMED, True, CodeEntry(code="country/KEN", label="Kenya", kind="place")),
        # A code the graph holds no name for is a complete read.
        (lambda refs: {}, True, CodeEntry(code="country/KEN", label="country/KEN", kind="place")),
        (None, True, CodeEntry(code="country/KEN", label="country/KEN", kind="entity")),
        (
            _fails(GraphReadError("503")),
            False,
            CodeEntry(code="country/KEN", label="country/KEN", kind="entity"),
        ),
        (
            _fails(GraphDeadlineExceededError("deadline")),
            False,
            CodeEntry(code="country/KEN", label="country/KEN", kind="entity"),
        ),
    ],
    ids=["named", "unnamed", "no-read", "read-failed", "deadline"],
)
def test_bundle_codes_reports_whether_the_label_read_completed(lookup, complete, kenya):
    table = Table(columns=("stat_var", "recipient"), rows=(("dcid:Amount_A", "country/KEN"),))
    codes = bundle_codes((_scan(table),), known_labels={}, lookup=lookup)

    assert codes.label_read_complete is complete
    assert kenya in codes.entries


def test_a_label_read_over_the_byte_cap_propagates():
    table = Table(columns=("recipient",), rows=(("country/KEN",),))
    error = GraphCapExceededError("too big", cap_kind="bytes", cap=1, observed=2)

    with pytest.raises(GraphCapExceededError):
        bundle_codes((_scan(table),), known_labels={}, lookup=_fails(error))


@pytest.mark.parametrize(
    ("pivot_columns", "headings", "kind"),
    [
        ("recipient", ("country/KEN", "country/TGO"), "place"),
        ("stat_var", ("dcid:Amount_A", "dcid:Amount_B"), "variable"),
        ("time_period", ("2015", "2019"), None),
        ("recipient", ("country/KEN", "=formula"), None),
    ],
    ids=["role", "variable", "uncoded-column", "formula-heading"],
)
def test_a_pivot_on_a_coded_column_names_its_headings_in_the_codes_table(
    pivot_columns, headings, kind
):
    table = Table(columns=("provider", *headings), rows=(("country/DEU", 1.0, 2.0),))
    scan = TableScan(
        role_columns=("provider", "recipient"),
        pivot_index="provider",
        pivot_columns=pivot_columns,
    )
    list(scan.observe((table,)))

    codes = bundle_codes((scan,), known_labels={}, lookup=lambda refs: {})
    docs = {doc.name: doc for doc in column_docs(scan, code_system_names={})}

    heading_kinds = {entry.code: entry.kind for entry in codes.entries if entry.code in headings}
    assert heading_kinds == (dict.fromkeys(headings, kind) if kind else {})
    for heading in headings:
        # A heading is a column name, so it takes no foreign key. A doc names it as written.
        doc = docs[spreadsheet_safe(heading)]
        assert not doc.coded
        assert ("Names are in `data/codes.csv`." in doc.description) is (kind is not None)
