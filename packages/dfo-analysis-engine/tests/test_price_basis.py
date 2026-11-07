"""`price_basis_reading`: the price basis and currency name of a unit, and the
name `unit_name` gives a converted unit from it."""

from __future__ import annotations

import pytest
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart, UnitRequest

from dfo_analysis_engine.price_basis import price_basis_reading
from dfo_analysis_engine.units import unit_name


@pytest.mark.parametrize(
    ("currency", "prices", "base_year", "deflation_basis", "reading"),
    [
        ("USD", "current", None, None, "current prices, US dollars"),
        ("EUR", "constant", 2020, None, "constant 2020 prices, euros"),
        ("GBP", "constant", "1999", None, "constant 1999 prices, pounds sterling"),
        ("JPY", "constant", None, None, "constant prices, JPY"),
        ("NGN", "constant", 2024, "donor", "constant 2024 prices, NGN, donor prices"),
        ("NGN", "constant", 2024, "recipient", "constant 2024 prices, NGN, recipient prices"),
    ],
)
def test_price_basis_reading(
    currency: str,
    prices: str,
    base_year: int | str | None,
    deflation_basis: str | None,
    reading: str,
) -> None:
    assert (
        price_basis_reading(
            currency=currency,
            prices=prices,
            base_year=base_year,
            deflation_basis=deflation_basis,
        )
        == reading
    )


def test_a_unit_named_alone_or_as_a_composed_side_names_its_deflation_basis() -> None:
    donor = UnitRequest("NGN", "constant", 2024, deflation_basis="donor")

    assert unit_name(donor) == "converted to constant 2024 prices, NGN, donor prices"
    assert unit_name(ComposedUnit(donor, StatVarPart("dcid:Count_Person"))) == (
        "constant 2024 prices, NGN, donor prices per Count_Person"
    )
