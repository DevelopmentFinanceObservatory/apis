"""Check constraint fixtures and their component schemas."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from assembler.catalogue.constraints import CONSTRAINT_REGISTRY
from assembler.mint.validate import component_registry

_REGISTRY_BY_ID = {spec.constraint_id: spec for spec in CONSTRAINT_REGISTRY}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


@pytest.fixture
def fixtures_dir(catalogue_dir):
    return catalogue_dir / "fixtures" / "constraints"


@pytest.mark.parametrize(
    "constraint_id",
    [
        "log_scale_requires_positive_values",
        "range_contains_visible_data",
        "comparable_units_required",
    ],
)
def test_axis_constraints_key_on_position_encoding(constraint_id):
    """The three axis constraints bind through the shared position encoding only."""
    spec = _REGISTRY_BY_ID[constraint_id]
    assert spec.encoding_classes == ("position",)
    assert spec.component_types == ()


def test_stat_change_basis_observed_keys_on_stat_card_type():
    spec = _REGISTRY_BY_ID["stat_change_basis_observed"]
    assert spec.encoding_classes == ()
    assert spec.component_types == ("stat_card",)


def test_bar_zero_in_domain_keys_on_bar_chart_type():
    spec = _REGISTRY_BY_ID["bar_zero_in_domain"]
    assert spec.encoding_classes == ()
    assert spec.component_types == ("bar_chart",)


@pytest.mark.parametrize(
    "constraint_id,fixture_name",
    [
        ("log_scale_requires_positive_values", "positive.json"),
        ("log_scale_requires_positive_values", "negative.json"),
        ("log_scale_requires_positive_values", "negative_authored_range_min.json"),
        ("range_contains_visible_data", "positive.json"),
        ("range_contains_visible_data", "negative.json"),
        ("range_contains_visible_data", "negative_min_bound.json"),
        ("comparable_units_required", "positive.json"),
        ("comparable_units_required", "negative_null_vs_string.json"),
        ("comparable_units_required", "positive_composed.json"),
        ("comparable_units_required", "negative_composed_denominator_differs.json"),
        ("stat_change_basis_observed", "positive.json"),
        ("stat_change_basis_observed", "negative.json"),
        ("bar_zero_in_domain", "positive.json"),
        ("bar_zero_in_domain", "negative.json"),
    ],
)
def test_fixture_component_is_schema_valid(release, fixtures_dir, constraint_id, fixture_name):
    context = _read_json(fixtures_dir / constraint_id / fixture_name)
    component = context["component"]
    schema = release.component_schemas[component["type"]]
    jsonschema.Draft202012Validator(schema, registry=component_registry(release)).validate(
        component
    )


def test_log_scale_requires_positive_values_positive(fixtures_dir):
    context = _read_json(fixtures_dir / "log_scale_requires_positive_values" / "positive.json")
    assert _REGISTRY_BY_ID["log_scale_requires_positive_values"].check(context) is None


def test_log_scale_requires_positive_values_negative(fixtures_dir):
    context = _read_json(fixtures_dir / "log_scale_requires_positive_values" / "negative.json")
    failure = _REGISTRY_BY_ID["log_scale_requires_positive_values"].check(context)
    assert failure is not None
    assert "eth_oda" in failure.detail
    assert "value 0" in failure.detail


def test_log_scale_requires_positive_values_negative_authored_range_min(fixtures_dir):
    context = _read_json(
        fixtures_dir / "log_scale_requires_positive_values" / "negative_authored_range_min.json"
    )
    failure = _REGISTRY_BY_ID["log_scale_requires_positive_values"].check(context)
    assert failure is not None
    assert failure.pointer == "/component/y_axis/range/min"


def test_range_contains_visible_data_positive(fixtures_dir):
    context = _read_json(fixtures_dir / "range_contains_visible_data" / "positive.json")
    assert _REGISTRY_BY_ID["range_contains_visible_data"].check(context) is None


def test_range_contains_visible_data_negative(fixtures_dir):
    context = _read_json(fixtures_dir / "range_contains_visible_data" / "negative.json")
    failure = _REGISTRY_BY_ID["range_contains_visible_data"].check(context)
    assert failure is not None
    assert failure.pointer == "/component/y_axis/range/max"


def test_range_contains_visible_data_negative_min_bound(fixtures_dir):
    context = _read_json(fixtures_dir / "range_contains_visible_data" / "negative_min_bound.json")
    failure = _REGISTRY_BY_ID["range_contains_visible_data"].check(context)
    assert failure is not None
    assert failure.pointer == "/component/y_axis/range/min"


@pytest.mark.parametrize("fixture_name", ["positive.json", "positive_composed.json"])
def test_comparable_units_required_positive(fixtures_dir, fixture_name):
    context = _read_json(fixtures_dir / "comparable_units_required" / fixture_name)
    assert _REGISTRY_BY_ID["comparable_units_required"].check(context) is None


@pytest.mark.parametrize(
    "fixture_name",
    ["negative_null_vs_string.json", "negative_composed_denominator_differs.json"],
)
def test_comparable_units_required_negative(fixtures_dir, fixture_name):
    context = _read_json(fixtures_dir / "comparable_units_required" / fixture_name)
    failure = _REGISTRY_BY_ID["comparable_units_required"].check(context)
    assert failure is not None
    assert "eth_oda" in failure.detail and "ken_oda" in failure.detail


def test_stat_change_basis_observed_positive(fixtures_dir):
    context = _read_json(fixtures_dir / "stat_change_basis_observed" / "positive.json")
    assert _REGISTRY_BY_ID["stat_change_basis_observed"].check(context) is None


def test_stat_change_basis_observed_negative(fixtures_dir):
    context = _read_json(fixtures_dir / "stat_change_basis_observed" / "negative.json")
    failure = _REGISTRY_BY_ID["stat_change_basis_observed"].check(context)
    assert failure is not None
    assert "2018" in failure.detail


def test_bar_zero_in_domain_positive(fixtures_dir):
    """The positive fixture is realistic all-positive data with the
    unbounded range mint actually produces, proving the constraint allows
    ordinary single-sign bar_chart data."""
    context = _read_json(fixtures_dir / "bar_zero_in_domain" / "positive.json")
    assert _REGISTRY_BY_ID["bar_zero_in_domain"].check(context) is None


def test_bar_zero_in_domain_negative(fixtures_dir):
    """The negative fixture carries an authored, bounded range that
    excludes zero, the actual hazard this constraint exists for."""
    context = _read_json(fixtures_dir / "bar_zero_in_domain" / "negative.json")
    failure = _REGISTRY_BY_ID["bar_zero_in_domain"].check(context)
    assert failure is not None
    assert failure.pointer == "/component/value_axis/range"
    assert "excludes zero" in failure.detail


@pytest.mark.parametrize(
    "context",
    [
        pytest.param(
            {
                "component": {
                    "value_axis": {"scale": "linear", "range": {"min": None, "max": None}},
                    "series": [{"id": "s", "points": [{"period": "2020", "value": 1_000_000}]}],
                }
            },
            id="unbounded-range",
        ),
        pytest.param(
            {
                "component": {
                    "value_axis": {"scale": "linear", "range": {"min": -10, "max": 10}},
                    "series": [{"id": "s", "points": [{"period": "2020", "value": 5}]}],
                }
            },
            id="bounded-range-including-zero",
        ),
        pytest.param(
            {
                "component": {
                    "value_axis": {"scale": "linear", "range": {"min": 200, "max": None}},
                    "series": [{"id": "s", "points": [{"period": "2020", "value": 250}]}],
                }
            },
            id="one-open-bound",
        ),
    ],
)
def test_bar_zero_in_domain_passes_legal_linear_range(context):
    """Open and zero-inclusive linear ranges allow ordinary bar-chart data."""
    assert _REGISTRY_BY_ID["bar_zero_in_domain"].check(context) is None


def test_log_scale_requires_positive_values_bar_chart_negative():
    """Every fixture for this constraint is a line_chart. `_value_axis_key`
    reads `value_axis` for a bar_chart component, proving that branch
    fires."""
    context = {
        "component": {
            "type": "bar_chart",
            "value_axis": {"scale": "log", "range": {"min": None, "max": None}},
            "series": [{"id": "s", "points": [{"period": "2020", "value": 0}]}],
        }
    }
    failure = _REGISTRY_BY_ID["log_scale_requires_positive_values"].check(context)
    assert failure is not None
    assert failure.pointer.startswith("/component/value_axis/")


def test_log_scale_requires_positive_values_bar_chart_positive():
    context = {
        "component": {
            "type": "bar_chart",
            "value_axis": {"scale": "log", "range": {"min": None, "max": None}},
            "series": [{"id": "s", "points": [{"period": "2020", "value": 5}]}],
        }
    }
    assert _REGISTRY_BY_ID["log_scale_requires_positive_values"].check(context) is None


def test_range_contains_visible_data_bar_chart_positive():
    context = {
        "component": {
            "type": "bar_chart",
            "value_axis": {"scale": "linear", "range": {"min": 0, "max": 25}},
            "series": [{"id": "s", "points": [{"period": "2020", "value": 20}]}],
        }
    }
    assert _REGISTRY_BY_ID["range_contains_visible_data"].check(context) is None


def test_range_contains_visible_data_line_chart_control():
    """A line_chart component keys on `y_axis`, proving `_value_axis_key`'s
    other branch alongside the bar_chart tests above."""
    context = {
        "component": {
            "type": "line_chart",
            "y_axis": {"scale": "linear", "range": {"min": 0, "max": 10}},
            "series": [{"id": "s", "points": [{"period": "2020", "value": 20}]}],
        }
    }
    failure = _REGISTRY_BY_ID["range_contains_visible_data"].check(context)
    assert failure is not None
    assert failure.pointer.startswith("/component/y_axis/")
