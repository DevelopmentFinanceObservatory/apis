"""Tests the role wording each flow family declares."""

from __future__ import annotations

from typing import get_args

import pytest

from dfo_platform.roles import (
    GENERIC_ROLE_WORDING,
    ROLE_WORDING,
    FlowFamily,
    role_wording,
)


def test_every_flow_family_declares_its_wording() -> None:
    assert set(ROLE_WORDING) == set(get_args(FlowFamily.__value__))


@pytest.mark.parametrize(
    ("family", "possessor", "origin", "destination"),
    [
        ("aid", "origin", ("donor", "donors", "from"), ("recipient", "recipients", "to")),
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
