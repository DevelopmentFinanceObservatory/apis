"""Tests `search.observations`: the preview caps, the shared pure helpers, and the engine-free import."""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from materialise.roles import _ROLE_AXES, axis_for_role
from search.observations import (
    ObservationSettings,
    as_entity_role_key,
    currency_unit_ref,
    fixed_selection,
    member_row_label,
    member_year_order,
    observation_settings,
    role_bulk_order,
    series_rank,
)

_ORIGIN_ROLES = [role for role in _ROLE_AXES if axis_for_role(role) == "origin"]
_DESTINATION_ROLES = [role for role in _ROLE_AXES if axis_for_role(role) == "destination"]


def test_importing_the_module_loads_no_engine_module() -> None:
    probe = (
        "import sys, search.observations\n"
        "engine = sorted(m for m in sys.modules if m.startswith('search.engine'))\n"
        "assert not engine, engine"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class TestObservationSettings:
    @pytest.mark.parametrize(
        "env_var",
        [
            "SEARCH_OBS_PREVIEW_MAX_PAIRS",
            "SEARCH_OBS_PREVIEW_POINTS",
            "SEARCH_OBS_INLINE_MAX",
            "SEARCH_ROLE_BULK_SERIES_MAX",
        ],
    )
    def test_a_cap_below_one_raises_naming_its_env_var(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str
    ) -> None:
        monkeypatch.setenv(env_var, "0")
        with pytest.raises(ValueError, match=env_var):
            ObservationSettings()

    def test_env_values_override_the_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_OBS_INLINE_MAX", "7")
        monkeypatch.setenv("SEARCH_ROLE_BULK_SERIES_MAX", "3")
        built = ObservationSettings()
        assert (built.obs_inline_max, built.role_bulk_series_max) == (7, 3)

    def test_the_accessor_returns_one_object(self) -> None:
        assert observation_settings() is observation_settings()


# ---------------------------------------------------------------------------
# member_row_label: pure composition, no graph/materialise involved.
# ---------------------------------------------------------------------------


def test_member_row_label_composes_sorted_constraint_values():
    # Sorting property names puts crsPurpose before gender.
    constraints = {"gender": "gender-value/Female", "crsPurpose": "crsPurpose-value/Malaria"}
    labels = {"crsPurpose-value/Malaria": "Malaria", "gender-value/Female": "Female"}
    assert member_row_label("sv1", constraints, labels) == "Malaria and Female"


def test_member_row_label_falls_back_to_statvar_label_when_no_constraints():
    assert member_row_label("sv1", {}, {"sv1": "Population, total"}) == "Population, total"


def test_member_row_label_fallback_stays_none_when_unresolved():
    """Missing constraints and StatVar labels produce `None`."""
    assert member_row_label("sv1", {}, {}) is None


def test_member_row_label_identical_empty_constraints_render_whatever_each_svs_own_label_is():
    """Members with no constraints can share identical resolved StatVar labels."""
    labels = {"sv1": "same name", "sv2": "same name"}
    assert member_row_label("sv1", {}, labels) == "same name"
    assert member_row_label("sv2", {}, labels) == "same name"


def test_member_row_label_identical_nonempty_constraints_render_the_same_text():
    """Members bound to the same constraint value produce the same label."""
    constraints = {"crsPurpose": "crsPurpose-value/Malaria"}
    labels = {"crsPurpose-value/Malaria": "Malaria"}
    assert (
        member_row_label("sv1", constraints, labels)
        == member_row_label("sv2", constraints, labels)
        == "Malaria"
    )


def test_member_row_label_synthetic_property_no_crs_system_declares():
    """An unknown constraint property composes from its resolved label."""
    constraints = {"widgetColor": "widgetColor-value/Blue"}
    labels = {"widgetColor-value/Blue": "Blue"}
    assert member_row_label("sv1", constraints, labels) == "Blue"


def test_series_rank_orders_by_latest_year_then_value_then_identity():
    latest: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[str, float | None] | None] = {
        ("sv/B", ()): ("2023", 10.0),
        ("sv/A", ()): ("2023", 10.0),  # identity breaks the value tie
        ("sv/C", ()): ("2023-06", 50.0),  # the year, not the full date, ranks
        ("sv/D", ()): ("2022", 999.0),  # an older year loses to any value in a newer one
        ("sv/E", ()): ("2023", None),  # no value follows the valued points of its year
        ("sv/F", ()): None,  # no point comes last
    }
    ranked = sorted(latest, key=lambda key: series_rank(key, latest[key]))
    assert [sv for sv, _roles in ranked] == ["sv/C", "sv/A", "sv/B", "sv/E", "sv/D", "sv/F"]


def test_fixed_selection_takes_columns_round_robin_in_column_order():
    columns = [["a1", "a2", "a3"], ["b1"], [], ["c1", "c2"]]
    assert fixed_selection(columns, 5) == ["a1", "b1", "c1", "a2", "c2"]
    assert fixed_selection(columns, 2) == ["a1", "b1"]
    assert fixed_selection(columns, 10) == ["a1", "b1", "c1", "a2", "c2", "a3"]


def test_member_year_order_puts_the_newest_year_first_and_undated_last():
    assert member_year_order({"sv/b": 2022, "sv/a": 2022, "sv/c": 2024, "sv/d": None}) == [
        "sv/c",
        "sv/a",
        "sv/b",
        "sv/d",
    ]


@pytest.mark.parametrize(
    ("rows", "role_currency", "order"),
    [
        (0, False, "ranked"),
        (10_000, False, "ranked"),
        (10_001, False, "fixed"),
        (1, True, "fixed"),
    ],
)
def test_role_bulk_order_ranks_at_or_under_the_row_gate_and_never_a_role_currency(
    search_settings, rows: int, role_currency: bool, order: str
):
    search_settings(role_bulk_rank_max_rows=10_000)
    assert role_bulk_order(rows=rows, converts_to_role_currency=role_currency) == order


class TestAsEntityRoleKey:
    """Every declared name _ROLE_AXES places gets a non-None key on its own
    side, read off the table itself so this test tracks the table if it grows."""

    @pytest.mark.parametrize("role", _ORIGIN_ROLES)
    def test_origin_axis_role_maps_to_origin_key(self, role: str) -> None:
        assert as_entity_role_key(role) == "origin"

    @pytest.mark.parametrize("role", _DESTINATION_ROLES)
    def test_destination_axis_role_maps_to_destination_key(self, role: str) -> None:
        assert as_entity_role_key(role) == "destination"

    def test_observation_about_maps_to_itself(self) -> None:
        assert as_entity_role_key("observationAbout") == "observationAbout"

    def test_unrecognised_role_returns_none_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = as_entity_role_key("guarantor")
        assert result is None
        assert "guarantor" in caplog.text


@pytest.mark.parametrize(
    ("currency", "expected"),
    [("USD", "CurrencyUSD"), ("observationAbout", None), ("usd", None), (None, None)],
)
def test_currency_unit_ref_names_only_iso_codes(currency: str | None, expected: str | None) -> None:
    assert currency_unit_ref(currency) == expected
