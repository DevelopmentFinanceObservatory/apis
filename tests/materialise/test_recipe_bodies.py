"""Checks the deployed recipe bodies and `recipe_body_digest`."""

from __future__ import annotations

import dataclasses

import pytest

from materialise.recipe_bodies import RECIPE_BODIES, RECIPE_CLASSES, recipe_body_digest

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
