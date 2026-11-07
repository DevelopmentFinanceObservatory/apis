"""Tests `select` normalization and validation of arrayed dimensions."""

from __future__ import annotations

import pytest

from assembler.errors import BindingInvalid
from assembler.ports import DimensionView, StructuralView
from assembler.select import (
    check_select_nonempty,
    check_single_arrayed_dimension,
    has_arrayed_dimension,
    normalize_select,
)


def test_scalar_values_become_single_value_bindings():
    normalized = normalize_select(select={"statvar": "health_oda", "recipient": "country/ETH"})
    assert normalized == [
        {
            "bindings": [
                {"dimension": "statvar", "values": ["health_oda"]},
                {"dimension": "recipient", "values": ["country/ETH"]},
            ]
        }
    ]


def test_array_value_passes_through_as_the_bindings_values():
    normalized = normalize_select(select={"recipient": ["country/ETH", "country/GHA"]})
    assert normalized == [
        {"bindings": [{"dimension": "recipient", "values": ["country/ETH", "country/GHA"]}]}
    ]


def test_normalize_produces_exactly_one_selector():
    normalized = normalize_select(
        select={"statvar": "x", "recipient": "country/ETH", "prices": "constant"}
    )
    assert len(normalized) == 1
    assert len(normalized[0]["bindings"]) == 3


_VIEW = StructuralView(
    dimensions=(
        DimensionView(
            name="stat_var", values=("dcid:X",), total_count=1, truncated=False, plural=False
        ),
        DimensionView(
            name="recipient", values=("country/KEN",), total_count=26, truncated=True, plural=True
        ),
    ),
    period_grain="year",
    coverage=("2020", "2021"),
    span=None,
    series_count=26,
)


def test_check_select_nonempty_passes_a_populated_map():
    check_select_nonempty(select={"statvar": "x"}, view=_VIEW)  # must not raise


def test_check_select_nonempty_refuses_an_empty_map_naming_the_frames_dimensions():
    with pytest.raises(BindingInvalid) as excinfo:
        check_select_nonempty(select={}, view=_VIEW)
    assert excinfo.value.pointer == "/select"
    assert "stat_var (1 value), recipient (26 values)" in excinfo.value.detail


def test_check_single_arrayed_dimension_passes_zero_or_one_array():
    check_single_arrayed_dimension(select={"statvar": "x", "recipient": "country/ETH"})
    check_single_arrayed_dimension(select={"recipient": ["country/ETH", "country/GHA"]})


def test_check_single_arrayed_dimension_refuses_two_arrays():
    with pytest.raises(BindingInvalid) as excinfo:
        check_single_arrayed_dimension(
            select={"recipient": ["country/ETH", "country/GHA"], "prices": ["current", "constant"]}
        )
    exc = excinfo.value
    # "prices" sorts before "recipient" -- the second offending dimension,
    # the one the pointer names, is "recipient".
    assert exc.pointer == "/select/recipient"
    assert exc.dimension == "recipient"
    assert exc.legal_values == ("prices", "recipient")


def test_has_arrayed_dimension_false_for_an_all_scalar_select():
    assert has_arrayed_dimension(select={"statvar": "x", "recipient": "country/ETH"}) is False


def test_has_arrayed_dimension_true_for_one_arrayed_dimension():
    assert has_arrayed_dimension(select={"recipient": ["country/ETH", "country/GHA"]}) is True


def test_has_arrayed_dimension_false_for_an_empty_select():
    assert has_arrayed_dimension(select={}) is False
