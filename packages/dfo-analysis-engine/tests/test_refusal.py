"""The engine-exception mapping for the verbs the differential cases do not cover,
and the composite rewrite of a `stat_var_unresolved` refusal."""

from __future__ import annotations

import pytest
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart

from dfo_analysis_engine.engine.aggregate import ComposedUnitSumError
from dfo_analysis_engine.engine.common import StatVarUnresolvedError
from dfo_analysis_engine.engine.two_sided import NonFiniteCellError
from dfo_analysis_engine.refusal import (
    CompositeComponent,
    Refusal,
    RefusalCode,
    composite_refusal,
    refusal_for_engine_error,
)

_PER_HEAD = ComposedUnit(numerator="dcid:USDollar", denominator=StatVarPart("dcid:Count_Person"))


@pytest.mark.parametrize(
    ("verb", "exc", "code", "fix"),
    [
        pytest.param(
            "sum_of",
            ComposedUnitSumError(_PER_HEAD, side="left"),
            "composed_unit_sum_unsupported",
            {"verb": "sum_of", "unit": _PER_HEAD.as_json(), "side": "left"},
            id="sum_of-names-the-side",
        ),
        pytest.param(
            "aggregate",
            ComposedUnitSumError(_PER_HEAD, side="left"),
            "composed_unit_sum_unsupported",
            {"verb": "aggregate", "unit": _PER_HEAD.as_json()},
            id="aggregate-names-no-side",
        ),
        pytest.param(
            "compare", ComposedUnitSumError(_PER_HEAD), None, None, id="compare-adds-nothing"
        ),
        pytest.param(
            "quotient_of",
            NonFiniteCellError(time_period="2022", reason="overflow"),
            "non_finite_result",
            {"group_key": {"time_period": "2022"}, "aggregation": "quotient_of"},
            id="quotient_of-overflow",
        ),
        pytest.param("join", ValueError("not an engine refusal"), None, None, id="other-exception"),
    ],
)
def test_each_verb_refuses_its_own_exceptions(
    verb: str, exc: Exception, code: str | None, fix: dict[str, object] | None
) -> None:
    refusal = refusal_for_engine_error(verb, exc)

    if code is None:
        assert refusal is None
        return
    assert refusal is not None
    assert (refusal.as_json()["code"], refusal.fix) == (code, fix)


_FAMILY = StatVarUnresolvedError(reason="several", distinct_stat_vars=["dcid:A", "dcid:B"])
_COMPOSITE = (
    CompositeComponent(role="bilateral", stat_vars=("dcid:A",)),
    CompositeComponent(role="imputed", stat_vars=("dcid:B", "dcid:C")),
)


@pytest.mark.parametrize(
    ("components", "rewritten"),
    [
        pytest.param([(), _COMPOSITE], True, id="composite-side"),
        pytest.param([()], False, id="plain-frame"),
    ],
)
def test_a_composite_refusal_names_its_components(
    components: list[tuple[CompositeComponent, ...]], rewritten: bool
) -> None:
    plain = refusal_for_engine_error("growth", _FAMILY)
    assert isinstance(plain, Refusal)

    refusal = composite_refusal(plain, components)

    assert refusal.code is RefusalCode.STAT_VAR_UNRESOLVED
    assert ("components" in refusal.fix) is rewritten
    if rewritten:
        assert refusal.fix["components"] == [
            {"role": "bilateral", "stat_vars": 1},
            {"role": "imputed", "stat_vars": 2},
        ]
