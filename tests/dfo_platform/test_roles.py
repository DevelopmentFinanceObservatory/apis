"""Tests the role wording each flow family declares and the noun each column reads under."""

from __future__ import annotations

from typing import get_args

import pytest
from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
)

from dfo_platform.columns import RESERVED_COLUMN_NOUNS, ROLE_NOUNS
from dfo_platform.roles import (
    GENERIC_ROLE_WORDING,
    ROLE_AXES,
    ROLE_WORDING,
    FlowFamily,
    role_wording,
)


def test_every_flow_family_declares_its_wording() -> None:
    assert set(ROLE_WORDING) == set(get_args(FlowFamily.__value__))


@pytest.mark.parametrize(
    ("family", "possessor", "origin", "destination"),
    [
        ("aid", "origin", ("provider", "providers", "from"), ("recipient", "recipients", "to")),
        (
            "debt_service",
            "destination",
            ("creditor", "creditors", "owed to"),
            ("debtor", "debtors", "of"),
        ),
        (
            "debt_stock",
            "destination",
            ("creditor", "creditors", "owed to"),
            ("debtor", "debtors", "of"),
        ),
        (
            "debt_lending",
            "destination",
            ("creditor", "creditors", "from"),
            ("debtor", "debtors", "to"),
        ),
        (None, "origin", ("provider", "providers", "from"), ("recipient", "recipients", "to")),
    ],
)
def test_role_wording_per_family(
    family: FlowFamily | None,
    possessor: str,
    origin: tuple[str, str, str],
    destination: tuple[str, str, str],
) -> None:
    wording = role_wording(family)
    assert wording.possessor == possessor
    for axis, expected in (("origin", origin), ("destination", destination)):
        noun = wording.noun(axis)
        assert (noun.singular, noun.plural, noun.preposition) == expected


def test_no_family_reads_the_generic_wording() -> None:
    assert role_wording(None) is GENERIC_ROLE_WORDING


def test_every_role_on_a_flow_axis_and_every_reserved_column_has_a_noun() -> None:
    assert set(ROLE_AXES) <= set(ROLE_NOUNS)
    assert set(RESERVED_COLUMN_NOUNS) == {
        STAT_VAR_COLUMN,
        TIME_PERIOD_COLUMN,
        VALUE_COLUMN,
        UNIT_COLUMN,
        STATUS_COLUMN,
        *FACET_IDENTITY_COLUMNS,
    }
