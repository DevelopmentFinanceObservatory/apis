"""Tests for `materialise.cadence.cadence_label` and `known_cadence_label`."""

from __future__ import annotations

import pytest

from materialise.cadence import cadence_label, known_cadence_label


@pytest.mark.parametrize(
    ("token", "label"),
    [
        ("P1D", "Daily"),
        ("P1W", "Weekly"),
        ("P1M", "Monthly"),
        ("P3M", "Quarterly"),
        ("P6M", "Semiannual"),
        ("P1Y", "Annual"),
        ("P2Y", "Biennial"),
        ("P5Y", "Quinquennial"),
    ],
)
def test_recognised_tokens_map_to_their_display_label(token: str, label: str):
    assert cadence_label(token) == label


def test_unrecognised_token_is_returned_unchanged():
    assert cadence_label("P4M") == "P4M"


def test_none_maps_to_none():
    assert cadence_label(None) is None


@pytest.mark.parametrize("token", ["P1Y", "P3M"])
def test_known_cadence_label_agrees_with_cadence_label_for_recognised_tokens(token: str):
    assert known_cadence_label(token) == cadence_label(token)


@pytest.mark.parametrize("token", ["P4M", "p1y", "", "Annual"])
def test_known_cadence_label_is_none_for_an_unrecognised_token(token: str):
    assert known_cadence_label(token) is None


def test_known_cadence_label_is_none_for_none():
    assert known_cadence_label(None) is None
