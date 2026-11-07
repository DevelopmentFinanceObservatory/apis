"""Checks the deployed recipe bodies and `recipe_body_digest`."""

from __future__ import annotations

import dataclasses

import pytest

from materialise.recipe_bodies import (
    RECIPE_BODIES,
    RECIPE_CLASSES,
    _check_deflation_bases,
    recipe_body_digest,
)
from materialise.recipes import deflation_basis

_RECIPE = RECIPE_BODIES["ConversionRecipeDAC"]


def test_recipe_bodies_are_keyed_by_their_conversion_recipe_member():
    assert sorted(RECIPE_BODIES) == [
        "ConversionRecipeDAC",
        "ConversionRecipeIDS",
        "ConversionRecipeIMF",
    ]
    for recipe_id, recipe in RECIPE_BODIES.items():
        assert recipe.recipe_id == recipe_id


def test_recipe_classes_publish_the_sorted_recipe_members():
    assert tuple(sorted(RECIPE_BODIES)) == RECIPE_CLASSES


@pytest.mark.parametrize("recipe_id", sorted(RECIPE_BODIES))
def test_every_recipe_converts_some_population_type_through_some_role(recipe_id: str):
    """An empty set would refuse every StatVar, so a body may not ship with one."""
    recipe = RECIPE_BODIES[recipe_id]
    assert recipe.flow_population_types
    assert recipe.deflator_entity_roles


def test_table_provenances_are_the_two_one_conversion_tables():
    assert {recipe.table_provenance for recipe in RECIPE_BODIES.values()} == {
        "provenance/ONEConversionTablesDAC",
        "provenance/ONEConversionTablesIMF",
    }


def test_recipe_body_digest_is_stable_across_calls():
    assert recipe_body_digest(_RECIPE) == recipe_body_digest(_RECIPE)


def test_recipe_body_digest_differs_between_the_imf_and_ids_bodies():
    """The IDS recipe reads the IMF tables, so only its role and population
    sets tell the two digests apart."""
    assert recipe_body_digest(RECIPE_BODIES["ConversionRecipeIMF"]) != recipe_body_digest(
        RECIPE_BODIES["ConversionRecipeIDS"]
    )


def test_recipe_body_digest_changes_when_any_field_replaced():
    baseline = recipe_body_digest(_RECIPE)
    replacements: dict[str, object] = {
        "recipe_id": "ConversionRecipeOther",
        "exchange_rate_variable": "OtherExchangeRate",
        "deflator_variable": "OtherDeflator",
        "deflator_entity_roles": frozenset({"financingRecipient"}),
        "flow_population_types": frozenset({"FinancialFlow"}),
        "forecast_policy": "OtherPolicy",
        "table_provenance": "provenance/Other",
        "fallback_entity": "OtherFallbackEntity",
    }
    assert set(replacements) == {field.name for field in dataclasses.fields(_RECIPE)}
    for field, new_value in replacements.items():
        changed = dataclasses.replace(_RECIPE, **{field: new_value})
        assert recipe_body_digest(changed) != baseline, field


_FLOW = ("financingProvider", "financingRecipient")


@pytest.mark.parametrize(
    ("recipe_id", "role_sets", "expected"),
    [
        pytest.param(
            "ConversionRecipeDAC", (_FLOW,), "donor", id="dac2a-aid-deflates-by-the-donor"
        ),
        pytest.param(
            "ConversionRecipeIDS", (_FLOW,), "recipient", id="ids-debt-deflates-by-the-debtor"
        ),
        pytest.param(
            "ConversionRecipeDAC",
            (("observationAbout",),),
            "donor",
            id="dac-subject-deflates-by-the-dac-table",
        ),
        pytest.param("ConversionRecipeIMF", (("observationAbout",),), "national", id="imf-gdp"),
        pytest.param(
            "ConversionRecipeDAC",
            (_FLOW, ("observationAbout",)),
            "donor",
            id="dac-statvars-on-two-roles-of-one-basis",
        ),
        pytest.param(
            "ConversionRecipeIMF",
            (("financingProvider",), ("observationAbout",)),
            None,
            id="imf-statvars-on-two-bases",
        ),
        pytest.param(
            "ConversionRecipeIDS", (("financingProvider",),), None, id="no-deflator-role-declared"
        ),
    ],
)
def test_the_deflation_basis_follows_the_recipe_and_the_role_it_deflates_by(
    recipe_id: str, role_sets: tuple[tuple[str, ...], ...], expected: str | None
):
    assert deflation_basis(RECIPE_BODIES[recipe_id], role_sets=role_sets) == expected


def test_a_recipe_role_with_no_deflation_basis_fails_the_check(monkeypatch: pytest.MonkeyPatch):
    unmapped = dataclasses.replace(_RECIPE, deflator_entity_roles=frozenset({"lender"}))
    monkeypatch.setitem(RECIPE_BODIES, "ConversionRecipeLender", unmapped)
    with pytest.raises(RuntimeError, match="lender"):
        _check_deflation_bases()
