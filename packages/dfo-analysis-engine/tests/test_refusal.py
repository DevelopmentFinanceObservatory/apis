"""The engine-exception mapping for the verbs the differential cases do not cover,
and the composite rewrite of a `stat_var_unresolved` refusal."""

from __future__ import annotations

import pytest
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart, UnitRequest

from dfo_analysis_engine.engine.aggregate import ComposedUnitSumError
from dfo_analysis_engine.engine.common import StatVarUnresolvedError
from dfo_analysis_engine.engine.compare import CompareAmbiguousSliceError, CompareUnitMismatchError
from dfo_analysis_engine.engine.paired import PairedNoKeyMatchedError, PairedUnknownColumnError
from dfo_analysis_engine.engine.two_sided import NonFiniteCellError
from dfo_analysis_engine.refusal import (
    CompositeComponent,
    Refusal,
    RefusalCode,
    _deflated_apart,
    composite_refusal,
    refusal_for_engine_error,
)

_PER_HEAD = ComposedUnit(numerator="dcid:USDollar", denominator=StatVarPart("dcid:Count_Person"))
_NGN_DONOR = UnitRequest("NGN", "constant", 2024, deflation_basis="donor")
_NGN_RECIPIENT = UnitRequest("NGN", "constant", 2024, deflation_basis="recipient")


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
        *(
            pytest.param(
                verb,
                PairedNoKeyMatchedError(
                    left_keys=[(("destination", "country/KEN"),)],
                    right_keys=[(("observationAbout", "country/NGA"),)],
                    left_key_count=1,
                    right_key_count=3,
                ),
                f"{verb}_no_key_matched",
                {
                    "left_keys": [{"destination": "country/KEN"}],
                    "right_keys": [{"observationAbout": "country/NGA"}],
                    "left_key_count": 1,
                    "right_key_count": 3,
                },
                id=f"{verb}-no-member-pairs",
            )
            for verb in ("compare", "sum_of")
        ),
        pytest.param(
            "sum_of",
            PairedUnknownColumnError(side="left", columns=["recipient"], available=["donor"]),
            "role_ambiguous",
            {"parameter": "on", "columns": ["recipient"], "available": ["donor"], "side": "left"},
            id="sum_of-on-names-an-unknown-column",
        ),
        pytest.param(
            "compare",
            CompareAmbiguousSliceError(
                side="right", entity={}, available=["destination", "origin"], by=["destination"]
            ),
            "role_ambiguous",
            {
                "parameter": "right_entity",
                "side": "right",
                "columns": [],
                "members": ["destination"],
                "available": ["destination", "origin"],
            },
            id="compare-paired-member-ambiguous",
        ),
        pytest.param(
            "compare",
            CompareUnitMismatchError(left_unit=_NGN_DONOR, right_unit=_NGN_RECIPIENT),
            "cross_frame_unit_mismatch",
            {"left": _NGN_DONOR.as_json(), "right": _NGN_RECIPIENT.as_json()},
            id="compare-deflation-bases-differ",
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


@pytest.mark.parametrize(
    ("left", "right", "apart"),
    [
        pytest.param(_NGN_DONOR, _NGN_RECIPIENT, True, id="one-recipe-two-bases"),
        pytest.param(
            UnitRequest("NGN", "constant", 2024, "ConversionRecipeDAC", "donor"),
            UnitRequest("NGN", "constant", 2024, "ConversionRecipeIDS", "recipient"),
            True,
            id="dac2a-donor-beside-ids-recipient",
        ),
        pytest.param(
            UnitRequest("NGN", "constant", 2024, "ConversionRecipeDAC", "donor"),
            UnitRequest("NGN", "constant", 2024, "ConversionRecipeIMF", "donor"),
            False,
            id="two-recipes-one-basis",
        ),
        pytest.param(
            _NGN_DONOR,
            UnitRequest("NGN", "constant", 2020, deflation_basis="recipient"),
            False,
            id="two-base-years",
        ),
        pytest.param(
            _NGN_DONOR, UnitRequest("NGN", "constant", 2024), False, id="one-basis-stated"
        ),
    ],
)
def test_two_units_are_deflated_apart_when_only_their_stated_bases_differ(
    left: UnitRequest, right: UnitRequest, apart: bool
) -> None:
    """The recipe a cell names plays no part, as in dfo-ai's analysis-mcp, so a DAC2a
    donor-priced side and an IDS recipient-priced side get the current-prices retry."""
    assert _deflated_apart(left, right) is apart


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
