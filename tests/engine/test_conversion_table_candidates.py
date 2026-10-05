"""Conversion-table StatVars are conversion inputs, so recall never offers them as answers."""

from __future__ import annotations

from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.retrieval.derive import CONVERSION_TABLE_STAT_VARS, derive_shapes
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.fixtures import FakeGraph

_ANSWER_SV = "Recalled_Person_Count"
_RATE_SV = RECIPE_BODIES["ConversionRecipeDAC"].exchange_rate_variable
_DEFLATOR_SV = RECIPE_BODIES["ConversionRecipeIMF"].deflator_variable


def test_every_recipe_s_rate_and_deflator_variable_is_a_table_stat_var():
    assert {
        sv for r in RECIPE_BODIES.values() for sv in (r.exchange_rate_variable, r.deflator_variable)
    } == CONVERSION_TABLE_STAT_VARS
    assert len(CONVERSION_TABLE_STAT_VARS) == 4


def test_recalled_table_stat_vars_never_reach_confirmation():
    graph = FakeGraph(
        nodes={
            _ANSWER_SV: {"label": "answer", "arcs": _std_sv_arcs("Person", "count")},
            _RATE_SV: {"label": "rate", "arcs": _std_sv_arcs("CurrencyUnit", "exchangeRate")},
            _DEFLATOR_SV: {
                "label": "deflator",
                "arcs": _std_sv_arcs("EconomicActivity", "deflator"),
            },
        },
        obs={},
        detect={},
        resolve={},
    )
    shapes = derive_shapes(
        confirmed_svs=[_RATE_SV, _DEFLATOR_SV, _ANSWER_SV],
        graph=graph,
        sv_scores={_RATE_SV: 0.95, _DEFLATOR_SV: 0.9, _ANSWER_SV: 0.5},
        sv_types=dict.fromkeys((_RATE_SV, _DEFLATOR_SV, _ANSWER_SV), "StatisticalVariable"),
    )
    members = {sv for shape in shapes for sv in (shape.sv_arc_facts or {})}
    assert members == {_ANSWER_SV}
