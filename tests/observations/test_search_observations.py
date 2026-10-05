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
    latest_value_rank,
    member_row_label,
    observation_settings,
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


def test_latest_value_rank_orders_descending_with_no_data_last():
    ranked = sorted([5.0, None, 10.0, 0.0], key=latest_value_rank)
    assert ranked == [10.0, 5.0, 0.0, None]


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
