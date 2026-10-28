"""Refusal codes the analysis door owns: every engine refusal (`dfo_analysis_engine.refusal`)
and `several_members`.

Every one answers 422. `ANALYSIS_STATUSES` holds them, since the code registry maps only
`rolling_populations_differ` and `several_members` on `datasets.analysis` so far.
"""

from __future__ import annotations

from collections.abc import Mapping

from dfo_analysis_engine.refusal import RefusalCode

_ENGINE_DEFINITIONS: Mapping[RefusalCode, str] = {
    RefusalCode.OP_SPEC_INVALID: "An analysis step's arguments do not form a valid operation.",
    RefusalCode.STAT_VAR_UNRESOLVED: "A step needs one variable and its input holds several.",
    RefusalCode.ROLE_AMBIGUOUS: "A step names a column its input has no grain for, or a role "
    "several of its columns fill.",
    RefusalCode.OPERATION_UNCOVERED: "A step names an aggregation outside the supported set.",
    RefusalCode.FRAME_TOO_LARGE: "A step would write more rows than a frame may hold.",
    RefusalCode.UNIT_HETEROGENEOUS: "A step's input holds several units and the step pins none.",
    RefusalCode.FACET_HETEROGENEOUS: "A step's input holds several facet selections of one "
    "variable.",
    RefusalCode.COVERAGE_INSUFFICIENT: "A step's result covers less of its expected grid than "
    "the coverage floor.",
    RefusalCode.NON_FINITE_RESULT: "A step's arithmetic overflows to a non-finite value.",
    RefusalCode.MIXED_TIME_GRAIN: "A step's input mixes periods of different grains or spans.",
    RefusalCode.COMPARISON_PERIODS_DISJOINT: "A comparison's two sides share no period.",
    RefusalCode.GROWTH_GAP_YEAR: "A growth window's endpoint has no value.",
    RefusalCode.CROSS_FRAME_UNIT_MISMATCH: "A two-sided step's sides hold different units.",
    RefusalCode.CROSS_FRAME_UNIT_UNBOUND: "One side of a two-sided step declares no unit.",
    RefusalCode.CROSS_FRAME_FACET_MISMATCH: "A two-sided step's sides read one variable under "
    "different facet selections.",
    RefusalCode.SHARE_OF_GRAIN_MISMATCH: "A share's numerator and denominator hold different "
    "grains.",
    RefusalCode.FILTER_PREDICATE_TYPE_MISMATCH: "A filter predicate tests a column of another "
    "type.",
    RefusalCode.FILTER_UNIT_CONVERSION_UNSUPPORTED: "A filter threshold needs a unit conversion "
    "the engine does not make.",
    RefusalCode.SHARE_OF_ZERO_DENOMINATOR: "A share's denominator is zero in a period.",
    RefusalCode.SHARE_OF_KEY_UNKNOWN: "A share pairs a column one side does not hold.",
    RefusalCode.SHARE_OF_KEY_UNMATCHED: "A share's numerator keys match no denominator key.",
    RefusalCode.QUOTIENT_OF_ZERO_DENOMINATOR: "Every matched denominator of a quotient is zero.",
    RefusalCode.QUOTIENT_OF_NO_KEY_MATCHED: "A quotient's numerator keys match no denominator key.",
    RefusalCode.COMPARE_NO_KEY_MATCHED: "A comparison's paired members match no key.",
    RefusalCode.SUM_OF_NO_KEY_MATCHED: "A sum's paired members match no key.",
    RefusalCode.QUOTIENT_OF_SIDE_UNIT_UNSUPPORTED: "A quotient side holds a unit that is a "
    "ratio already.",
    RefusalCode.COMPOSED_UNIT_SUM_UNSUPPORTED: "A step would add values in a composed unit.",
    RefusalCode.JOIN_OUTPUT_COLLISION: "A join's two sides hold the same variable, so its rows "
    "cannot be told apart.",
    RefusalCode.JOIN_NO_COMMON_KEY_COLUMNS: "A join's sides share no key column.",
    RefusalCode.DERIVED_INPUT_UNUSABLE: "A step reads an input whose periods are one span.",
    RefusalCode.DERIVED_INPUT_ORDINAL: "A step reads rank positions or a ratio as amounts.",
    RefusalCode.PLACE_LEVELS_MIXED: "A step would pool places or parties that hold one "
    "another's figures.",
    RefusalCode.ROLLING_ANCHOR_UNRESOLVED: "No latest period could be established for a "
    "rolling step, or a rolling literal is not the latest period at its offset.",
    RefusalCode.ROLLING_GRAIN_UNSUPPORTED: "A rolling step's roots hold periods other than "
    "calendar years.",
    RefusalCode.ROLLING_POPULATIONS_DIFFER: "The rolling steps of one analysis recipe count "
    "the latest period over different populations of series.",
}


def _title(code: str) -> str:
    return code.replace("_", " ").title()


ANALYSIS_DOOR_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    **{code.value: (_title(code.value), _ENGINE_DEFINITIONS[code]) for code in RefusalCode},
    "several_members": (
        "Several Members",
        "An analysis step takes one reading for each input, and an input names a dataset "
        "that holds several.",
    ),
}

ANALYSIS_STATUSES: Mapping[str, int] = dict.fromkeys(ANALYSIS_DOOR_CODE_CATALOGUE, 422)

if set(RefusalCode) - set(_ENGINE_DEFINITIONS):
    raise AssertionError(
        f"engine codes without a definition: {sorted(set(RefusalCode) - set(_ENGINE_DEFINITIONS))}"
    )
