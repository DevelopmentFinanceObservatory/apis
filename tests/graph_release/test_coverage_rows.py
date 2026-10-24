"""Pins `role_map` and `coverage_rows` against the published coverage build.

`coverage_rows/published.json` holds the published dev build's rows and role names for the
variables of the recorded `RawInputs`; its `source` field names the build. Seventeen of the
twenty variables bear roles: two-role ones, lone destination-axis ones and lone origin-axis ones
(DAC1).
"""

from __future__ import annotations

import dataclasses
import json
from typing import cast

import pyarrow as pa
import pytest

from dfo_platform.roles import ROLE_AXES
from graph_release.coverage import HORIZON, coverage_rows, role_map
from graph_release.format import conform_part
from graph_release.platform import PlatformRefusal
from series_coverage.schema import RoleNames
from tests.graph_release.recorded import FIXTURES, platform_parts

_TWO_ROLES = "Amount_FinancialFlow_Disbursement_DAC2aAidType299_Nominal"


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads((FIXTURES / "coverage_rows/published.json").read_text())


def test_coverage_rows_equal_the_published_build(published: dict) -> None:
    parts = platform_parts()
    roles = role_map(parts.edges, ROLE_AXES)

    rows = coverage_rows(parts.series, roles)

    assert {variable: dataclasses.asdict(names) for variable, names in roles.items()} == (
        published["role_names"]
    )
    assert rows.column_names == published["columns"] == HORIZON.schema.names
    assert rows.cast(HORIZON.schema).to_pylist() == [
        dict(zip(published["columns"], row, strict=True)) for row in published["rows"]
    ]
    assert {(names.donor is None, names.recipient is None) for names in roles.values()} == {
        (False, False),
        (False, True),
        (True, False),
    }


def test_coverage_rows_read_series_split_into_files_with_their_own_dictionaries() -> None:
    parts = platform_parts()
    roles = role_map(parts.edges, ROLE_AXES)
    half = parts.series.num_rows // 2
    files = pa.concat_tables(
        [
            conform_part(parts.series.slice(0, half), name="series"),
            conform_part(parts.series.slice(half), name="series"),
        ]
    )
    assert files["variable_measured"].num_chunks == 2
    first, second = (
        cast("pa.DictionaryArray", chunk).dictionary for chunk in files["variable_measured"].chunks
    )
    assert first != second

    assert coverage_rows(files, roles).equals(coverage_rows(parts.series, roles))


@pytest.mark.parametrize(
    "role_axes",
    [
        pytest.param({}, id="lone-role-on-no-axis"),
        pytest.param(
            {**ROLE_AXES, "financingProvider": "destination", "financingRecipient": "origin"},
            id="pair-sorted-destination-first",
        ),
    ],
)
def test_role_map_refuses_roles_off_the_rule(role_axes: dict) -> None:
    with pytest.raises(PlatformRefusal) as refused:
        role_map(platform_parts().edges, role_axes)

    assert refused.value.code == "missing_role_declaration"


def test_coverage_rows_refuse_series_whose_entities_disagree_with_the_roles() -> None:
    parts = platform_parts()
    roles = dict(role_map(parts.edges, ROLE_AXES))
    roles[_TWO_ROLES] = RoleNames(donor="financingProvider", recipient=None)

    with pytest.raises(PlatformRefusal) as refused:
        coverage_rows(parts.series, roles)

    assert refused.value.code == "missing_role_declaration"
    assert _TWO_ROLES in str(refused.value)
