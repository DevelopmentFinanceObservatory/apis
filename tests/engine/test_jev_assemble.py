"""Jev assembly: each threshold's cut-off, the behaviours a rewrite loses silently, and the
house policies on answer sets no dev item reaches."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import pytest

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.binders.base import ApplyPolicies, BindRequest
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.binders.jev_assemble import (
    ASK_TEXT,
    Confidence,
    JevAnswers,
    assemble,
    assemble_quantity,
    bind_result,
    frame_stated_form,
)
from search.engine.binders.jev_menu import jev_menu
from search.engine.binders.jev_request import bind_request, quantity_request
from search.engine.frame import Frame, FrameNormalisation, Stated, StatedBy, door_frame
from search.engine.qualifier_match import QualifierMatch, QualifierTie, names_plain_name
from search.engine.region.phase_rank_bind import _HousePolicyStep
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import Axis, GraphRef

_SECTOR = "how:crsPurpose"
_CODES = ["CRSPurposeCode111", "CRSPurposeCode112"]


def _shape(
    index: int,
    *,
    population: str = "Person",
    measured: str = "count",
    qualifier: str | None = None,
    denominator: str | None = None,
    label: str | None = None,
    slots: Sequence[tuple[Axis, str, list[str]]] = (),
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=f"shape{index}",
        label=label or f"shape {index}",
        pop_type_dcid=population,
        meas_prop_dcid=measured,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=qualifier,
        meas_denom_dcid=denominator,
        slot_keys=tuple(
            SlotKeyDraft(axis=axis, property_dcid=prop, label=prop) for axis, prop, _ in slots
        ),
        slot_taxonomy={f"{axis}:{prop}": values for axis, prop, values in slots},
    )


def _frame() -> Frame:
    return door_frame(
        door="parsed",
        variable="aid",
        entities=[],
        entity_roles={},
        member_entities=frozenset(),
        pins={},
        date_request=None,
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset(),
    )


def _request(
    shapes: Sequence[ShapeDraft],
    *,
    dropped: tuple[str, ...] = (),
    ties: Sequence[tuple[QualifierTie, ...]] = (),
    apply_policies: ApplyPolicies | None = None,
) -> BindRequest:
    frame = _frame()
    step = _HousePolicyStep(
        shapes=shapes, frame=frame, admitted_qualifiers=(), availability=(), code_index=get_index()
    )
    return BindRequest(
        variable="aid",
        shapes=list(shapes),
        admitted_ties=ties,
        dropped_qualifier_phrases=dropped,
        frame=frame,
        query_text="aid",
        apply_policies=apply_policies or step,
        policy_chooses_on=step.chooses_on,
    )


def _answers(
    nouls: Mapping[str, float] | None = None,
    *,
    elect: Mapping[int, float] | None = None,
    choices: Mapping[str, Mapping[str, float]] | None = None,
) -> JevAnswers:
    picked = dict(choices or {})
    if elect is not None:
        picked["elect"] = {f"Shape {i}": p for i, p in elect.items()}
    return JevAnswers(nouls=dict(nouls or {}), choices=picked)


def _assemble(
    request: BindRequest, answers: JevAnswers, *, form: str | None = None
) -> tuple[_UniversalBindOutput, Confidence]:
    return assemble(jev_menu(request), answers, stated_form=form)


def _binding(output: _UniversalBindOutput, prop: str) -> SlotBindingDraft:
    return next(b for b in output.bindings if f"{b.axis}:{b.property_dcid}" == prop)


# One menu of two shapes on the sector slot, for the value thresholds.
_SECTORS = _shape(0, slots=[("how", "crsPurpose", _CODES)])


def _sector_values(nouls: Mapping[str, float]) -> tuple[str, str, tuple[str, ...], str]:
    output, _conf = _assemble(
        _request([_SECTORS], dropped=("across all sectors",)),
        _answers({"m|0": 0.9, **nouls}, elect={0: 1.0}),
    )
    draft = _binding(output, _SECTOR)
    return draft.kind, draft.evidence, tuple(draft.value_dcids), draft.requested_all_phrase or ""


@dataclass(frozen=True)
class _Cut:
    """One threshold: a scenario reading one decision, and the answer that crosses it."""

    scenario: Callable[[Mapping[str, float]], object]
    qid: str
    below: float
    above: float
    decided_below: object
    decided_above: object


def _ask(nouls: Mapping[str, float]) -> object:
    output, _conf = _assemble(_request([_shape(0)]), _answers(nouls, elect={0: 1.0}))
    return output.ask is not None


def _pool(answers: Callable[[Mapping[str, float]], JevAnswers], shapes: list[ShapeDraft]):
    def read(nouls: Mapping[str, float]) -> object:
        _output, conf = _assemble(_request(shapes), answers(nouls))
        assert conf.pool is not None
        return sorted(conf.pool.measuring)

    return read


def _near_elect(nouls: Mapping[str, float]) -> object:
    answers = _answers({"m|0": 1.0, "m|1": 1.0}, elect={0: 0.5, 1: nouls["elect|1"]})
    _output, conf = _assemble(_request([_shape(0), _shape(1, population="Place")]), answers)
    assert conf.pool is not None
    return sorted(conf.pool.measuring)


def _alternates(shapes: list[ShapeDraft], elect: Mapping[int, float], best: float):
    """The alternates of a menu whose elected shape 0 scores `best`, near enough to the
    threshold under test that the near-best bar on alternates passes both sides of it."""

    def read(nouls: Mapping[str, float]) -> object:
        output, _conf = _assemble(_request(shapes), _answers({"m|0": best, **nouls}, elect=elect))
        return output.also_plausible or None

    return read


def _near_alt(nouls: Mapping[str, float]) -> object:
    p = nouls["elect|1"]
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 1.0 - p, 1: p})
    output, _conf = _assemble(_request([_shape(0), _shape(1, population="Place")]), answers)
    return output.also_plausible


def _elected(shapes: list[ShapeDraft], elect: Mapping[int, float], base: Mapping[str, float]):
    def read(nouls: Mapping[str, float]) -> object:
        output, conf = _assemble(_request(shapes), _answers({**base, **nouls}, elect=elect))
        return output.chosen_shape_index, conf.rules

    return read


def _declined(nouls: Mapping[str, float]) -> object:
    answers = _answers(nouls, elect={0: 1.0}, choices={"electq": {"Shape 0": 0.3, "none": 0.7}})
    output, _conf = _assemble(_request([_shape(0)]), answers)
    return output.ask is not None


def _gate(nouls: Mapping[str, float]) -> object:
    answers = _answers(nouls, choices={"q|form": {"per_person": 0.9, "none": 0.1}})
    return assemble_quantity(answers, []).stated_form


_PLAIN_AND_DIVIDED = [_shape(0), _shape(1, denominator="Count_Person")]
_VALUE_ADDED = _shape(0, population="EconomicProduction", measured="valueAdded")
_OUTPUT = _shape(1, population="EconomicActivity", measured="amount", qualifier="Nominal")
_PERSON_AND_PLACE = [_shape(0), _shape(1, population="Place")]
_AMOUNT_AND_GE = [
    _shape(0, population="FinancialFlow", measured="grantEquivalent", qualifier="Nominal"),
    _shape(1, population="FinancialFlow", measured="amount", qualifier="Nominal"),
]

_CUTS = {
    "ask": _Cut(_ask, "m|0", 0.15, 0.16, True, False),
    "decline": _Cut(_declined, "m|0", 0.4, 0.41, True, False),
    "measures": _Cut(
        _pool(lambda n: _answers({"m|0": 0.6, **n}, elect={0: 1.0}), [_VALUE_ADDED, _OUTPUT]),
        "m|1",
        0.5,
        0.51,
        [0],
        [0, 1],
    ),
    "near": _Cut(
        _pool(lambda n: _answers({"m|0": 0.9, **n}), [_shape(0), _shape(1, population="Place")]),
        "m|1",
        0.74,
        0.76,
        [0],
        [0, 1],
    ),
    "near_elect": _Cut(_near_elect, "elect|1", 0.09, 0.11, [0], [0, 1]),
    "widen": _Cut(
        _pool(lambda n: _answers({"m|0": 0.9, **n}, elect={0: 1.0}), [_VALUE_ADDED, _OUTPUT]),
        "m|1",
        0.69,
        0.71,
        [0],
        [0, 1],
    ),
    "alt_floor": _Cut(
        _alternates([_shape(0), _shape(1, population="Place")], {0: 0.5, 1: 0.5}, 0.32),
        "m|1",
        0.3,
        0.31,
        None,
        [1],
    ),
    "near_alt": _Cut(_near_alt, "elect|1", 0.33, 0.34, None, [1]),
    "measure_value": _Cut(
        _elected(_PERSON_AND_PLACE, {0: 0.6, 1: 0.4}, {"m|0": 0.9, "m|1": 0.9}),
        "mv|Place",
        0.4,
        0.41,
        (0, []),
        (1, ["named: populationType"]),
    ),
    "policy_value": _Cut(
        _elected(_AMOUNT_AND_GE, {0: 0.6, 1: 0.4}, {"m|0": 0.9, "m|1": 0.9}),
        "mv|grantEquivalent",
        0.8,
        0.81,
        (1, ["house: amount_vs_ge"]),
        (0, ["named: measuredProperty"]),
    ),
    "named": _Cut(
        _elected(
            [_shape(0), _shape(1, population="Place")], {0: 0.6, 1: 0.4}, {"m|0": 0.9, "m|1": 0.9}
        ),
        "named|1",
        0.5,
        0.51,
        (0, []),
        (1, ["named: dataset"]),
    ),
    "value": _Cut(
        _sector_values,
        f"n|{_SECTOR}|0",
        0.6,
        0.61,
        ("unbound", "mentioned", (), ""),
        ("value", "mentioned", ("CRSPurposeCode111",), ""),
    ),
    "part": _Cut(
        lambda n: _sector_values({f"n|{_SECTOR}|0": 0.7, **n}),
        f"part|{_SECTOR}",
        0.15,
        0.16,
        ("unbound", "mentioned", (), ""),
        ("value", "mentioned", ("CRSPurposeCode111",), ""),
    ),
    "part_n": _Cut(
        lambda n: _sector_values({f"part|{_SECTOR}": 0.1, **n}),
        f"n|{_SECTOR}|0",
        0.8,
        0.81,
        ("unbound", "mentioned", (), ""),
        ("value", "mentioned", ("CRSPurposeCode111",), ""),
    ),
    "several": _Cut(
        lambda n: _sector_values({f"n|{_SECTOR}|0": 0.9, f"n|{_SECTOR}|1": 0.9, **n}),
        f"several|{_SECTOR}",
        0.5,
        0.51,
        ("value", "mentioned", ("CRSPurposeCode111",), ""),
        ("set", "mentioned", ("CRSPurposeCode111", "CRSPurposeCode112"), ""),
    ),
    "requested_all": _Cut(
        _sector_values,
        f"all|{_SECTOR}|0",
        0.5,
        0.51,
        ("unbound", "mentioned", (), ""),
        ("unbound", "requested_all", (), "across all sectors"),
    ),
    "gate": _Cut(_gate, "q|gate", 0.49, 0.5, None, "per_person"),
}


@pytest.mark.parametrize("cut", _CUTS.values(), ids=_CUTS.keys())
def test_one_answer_across_a_threshold_moves_its_decision(cut: _Cut):
    assert cut.scenario({cut.qid: cut.below}) == cut.decided_below
    assert cut.scenario({cut.qid: cut.above}) == cut.decided_above


def test_a_missing_answer_reads_as_zero():
    answers = _answers()

    output, conf = _assemble(_request([_shape(0)]), answers)

    assert answers.noul("m|0") == 0.0
    assert answers.choice("elect") is None
    assert output.ask == ASK_TEXT
    assert conf.rules == ["ask: no shape measures the phrase"]


def test_none_in_electq_enters_no_shape_s_election():
    answers = _answers(
        {"m|0": 0.9},
        elect={0: 1.0},
        choices={"electq": {"Shape 0": 0.2, "none": 0.8}},
    )

    _output, conf = _assemble(_request([_shape(0)]), answers)

    assert conf.election == {0: pytest.approx(0.6)}


@pytest.mark.parametrize(("named", "evidence"), [(0.6, "sole_option"), (0.61, "mentioned")])
def test_a_one_member_slot_binds_its_member(named: float, evidence: str):
    shape = _shape(0, slots=[("how", "crsPurpose", ["CRSPurposeCode111"])])
    answers = _answers({"m|0": 0.9, f"n|{_SECTOR}|0": named}, elect={0: 1.0})

    output, _conf = _assemble(_request([shape]), answers)

    draft = _binding(output, _SECTOR)
    assert (draft.kind, draft.value_dcids, draft.evidence) == (
        "value",
        ["CRSPurposeCode111"],
        evidence,
    )


@pytest.mark.parametrize(
    ("chosen", "bound"),
    [
        ("CRSPurposeCode510", "how:crsPurpose"),
        ("CRSCooperationModalityA", "how:crsCooperationModality"),
    ],
)
def test_a_tie_binds_the_chosen_row_on_its_own_slot(chosen: str, bound: str):
    """ "budget support" ties a cooperation modality with a CRS purpose, and the row Jev
    chooses binds on its own code system's slot whichever row the tie lists first."""
    index = get_index()
    rows = tuple(
        QualifierMatch(
            code_system=row.code_system,
            code_dcid=row.dcid,
            label=row.label,
            source="label",
            matched_text="budget support",
        )
        for row in (
            index.code_by_dcid("CRSCooperationModalityA"),
            index.code_by_dcid("CRSPurposeCode510"),
        )
        if row is not None
    )
    shape = _shape(
        0,
        slots=[
            (
                "how",
                "crsCooperationModality",
                ["CRSCooperationModalityA", "CRSCooperationModalityB"],
            ),
            ("how", "crsPurpose", ["CRSPurposeCode510", "CRSPurposeCode111"]),
        ],
    )
    other = next(r.code_dcid for r in rows if r.code_dcid != chosen)
    answers = _answers({"m|0": 0.9}, elect={0: 1.0}, choices={"tie|0|0": {chosen: 0.8, other: 0.2}})

    output, _conf = _assemble(
        _request([shape], ties=[(QualifierTie(matched_text="budget support", rows=rows),)]),
        answers,
    )

    assert len(rows) == 2
    assert {f"{b.axis}:{b.property_dcid}": (b.kind, b.value_dcids) for b in output.bindings} == {
        "how:crsCooperationModality": ("unbound", []),
        "how:crsPurpose": ("unbound", []),
    } | {bound: ("value", [chosen])}


def test_a_source_slot_is_unbound_and_asked_about():
    shape = _shape(0, slots=[("source", "provenance", ["provenance/A", "provenance/B"])])
    request = _request([shape])

    output, _conf = _assemble(request, _answers({"m|0": 0.9}, elect={0: 1.0}))

    assert _binding(output, "source:provenance").kind == "unbound"
    _state, questions = bind_request(jev_menu(request))
    assert not [q for q in questions if "source:" in q]


def test_a_value_under_another_named_value_is_dropped():
    shape = _shape(0, slots=[("how", "crsPurpose", ["CRSPurposeCode110", "CRSPurposeCode111"])])
    answers = _answers({"m|0": 0.9, f"n|{_SECTOR}|0": 0.9, f"n|{_SECTOR}|1": 0.95}, elect={0: 1.0})

    output, _conf = _assemble(_request([shape]), answers)

    assert _binding(output, _SECTOR).value_dcids == ["CRSPurposeCode110"]


def test_the_declared_default_is_left_to_the_engine():
    prop = "how:dac1AidType"
    shape = _shape(0, slots=[("how", "dac1AidType", ["DAC1AidType1010", "DAC1AidType1015"])])
    answers = _answers({"m|0": 0.9, f"n|{prop}|0": 0.9}, elect={0: 1.0})

    output, _conf = _assemble(_request([shape]), answers)

    assert _binding(output, prop).kind == "unbound"


def test_a_shape_a_house_policy_sets_aside_is_never_listed():
    shapes = [
        _shape(0, population="FinancialFlow", measured="grantEquivalent", qualifier="Nominal"),
        _shape(1, population="FinancialFlow", measured="amount", qualifier="Nominal"),
    ]
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 0.5, 1: 0.5})

    output, conf = _assemble(_request(shapes), answers)

    assert conf.rules == ["house: amount_vs_ge"]
    assert conf.policy_dropped == {0}
    assert output.also_plausible is None


def test_equal_elections_go_to_the_lower_menu_index():
    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={1: 0.5, 0: 0.5})

    output, conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == 0
    assert conf.pool is not None and conf.pool.measuring == (0, 1)


def test_the_house_policies_see_the_pool_with_the_election():
    seen: list[PolicyPool] = []

    def record(pool: PolicyPool, **_kwargs: object) -> PolicyChoice:
        seen.append(pool)
        return PolicyChoice(survivor=1, fired=(), dropped=frozenset({0}), facts=())

    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 0.6, 1: 0.4})

    output, _conf = _assemble(_request(shapes, apply_policies=record), answers)

    assert seen == [PolicyPool(measuring=(0, 1), election={0: 0.6, 1: 0.4})]
    assert output.chosen_shape_index == 1


def test_an_ask_runs_no_house_policy():
    calls: list[PolicyPool] = []

    def record(pool: PolicyPool, **_kwargs: object) -> PolicyChoice:
        calls.append(pool)
        raise AssertionError("no policy on an ask")

    output, conf = _assemble(_request([], apply_policies=record), _answers())

    assert output.ask == ASK_TEXT
    assert (calls, conf.pool) == ([], None)


@pytest.mark.parametrize(
    ("shapes", "fired"),
    [
        pytest.param(
            [
                _shape(0, population="FinancialFlow", slots=[("how", "nettingBasis", ["Net"])]),
                _shape(
                    1, population="FinancialFlow", slots=[("how", "nettingBasis", ["Gross", "Net"])]
                ),
            ],
            "net_vs_gross",
            id="gross-over-net",
        ),
        pytest.param(
            [
                _shape(0, population="FinancialObligation"),
                _shape(
                    1,
                    population="FinancialFlow",
                    slots=[("how", "financialFlowType", ["Disbursement"])],
                ),
            ],
            "commit_vs_disb",
            id="disbursement-over-commitment",
        ),
    ],
)
def test_a_house_policy_elects_over_the_election(shapes: list[ShapeDraft], fired: str):
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 0.6, 1: 0.4})

    output, conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == 1
    assert conf.rules == [f"house: {fired}"]
    assert conf.policy_dropped == {0}


def test_a_named_dataset_keeps_the_shapes_it_names():
    shapes = [_shape(0, label="CRS series"), _shape(1, label="DAC2a series"), _shape(2)]
    answers = _answers(
        {"m|0": 0.9, "m|1": 0.9, "m|2": 0.9, "named|1": 0.9, "named|2": 0.8},
        elect={0: 0.5, 1: 0.2, 2: 0.3},
    )

    output, conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == 2
    assert conf.rules == ["named: dataset"]
    assert conf.pool is not None and sorted(conf.pool.measuring) == [1, 2]


def test_the_form_sets_aside_divided_shapes_only_when_the_phrase_states_none():
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 0.4, 1: 0.6})

    plain, conf = _assemble(_request(_PLAIN_AND_DIVIDED), answers)
    framed, _ = _assemble(_request(_PLAIN_AND_DIVIDED), answers, form="per_person")

    assert plain.chosen_shape_index == 0
    assert conf.rules == ["form: phrase states none, shapes with a denominator set aside"]
    assert framed.chosen_shape_index == 1


@pytest.mark.parametrize(("measure", "also"), [(0.8, []), (0.81, [1])])
def test_an_alternate_stands_only_near_the_best_measure_score(measure: float, also: list[int]):
    """An alternate scoring under 0.9 of the best measure score is a weaker reading, the
    bar a Gemini nomination clears on its rating (`gemini.nominated_near_best`)."""
    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers({"m|0": 0.9, "m|1": measure}, elect={0: 0.6, 1: 0.4})

    result, _conf = bind_result(jev_menu(_request(shapes)), answers, stated_form=None)

    assert result.binder == "jev"
    assert result.output.also_plausible == also
    assert result.policy_pool is not None
    assert result.policy_choice is not None and result.policy_choice.survivor == 0


def test_another_dataset_s_copy_of_the_elected_reading_is_no_alternate():
    shapes = [_shape(0, label="UNICEF series"), _shape(1, label="WHO series")]
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 1.0})

    output, _conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == 0
    assert not output.also_plausible


_GDP = {"population": "EconomicActivity", "measured": "amount"}
_NOMINAL_GDP = _shape(0, **_GDP, qualifier="Nominal")
_REAL_GDP = _shape(1, **_GDP, qualifier="RealValue")
_GDP_VALUE_ADDED = _shape(2, population="EconomicProduction", measured="valueAdded")


@pytest.mark.parametrize(
    ("shapes", "real", "elected"),
    [
        pytest.param([_NOMINAL_GDP, _REAL_GDP], 0.7, 1, id="constant-prices-no-value-added"),
        pytest.param([_shape(0, **_GDP), _REAL_GDP], 0.7, 1, id="no-qualifier-no-value-added"),
        pytest.param(
            [_NOMINAL_GDP, _REAL_GDP, _GDP_VALUE_ADDED], 0.7, 0, id="value-added-below-the-bar"
        ),
        pytest.param(
            [_NOMINAL_GDP, _REAL_GDP, _GDP_VALUE_ADDED], 0.81, 1, id="value-added-above-the-bar"
        ),
    ],
)
def test_jev_names_real_terms_at_the_policy_bar_only_where_a_policy_chooses(
    shapes: list[ShapeDraft], real: float, elected: int
):
    """`nominal_vs_va` chooses a price basis among output shapes only beside a value-added
    shape, so "GDP in constant prices" over nominal and real GDP alone narrows to real
    terms at the ordinary bar, and beside value added only above `policy_value`."""
    nouls = {f"m|{i}": 0.9 for i in range(len(shapes))} | {"mv|RealValue": real}
    answers = _answers(nouls, elect={0: 0.4, 1: 0.3, 2: 0.3})

    output, _conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == elected


def test_only_a_shape_measuring_the_phrase_above_ask_is_elected():
    """The one shape over `ask` wins though another, under it, holds nearly all the
    election."""
    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers({"m|0": 0.12, "m|1": 0.2}, elect={0: 0.97, 1: 0.03})

    output, conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == 1
    assert conf.pool is not None and conf.pool.measuring == (1,)


# A CRS shape with a sector and a type of finance, and a DAC2a shape stating the type of
# finance in its own taxonomy, with no sector; `net_vs_gross` prefers the DAC2a shape.
_CRS_LOANS = _shape(
    0,
    population="FinancialFlow",
    slots=[
        ("how", "crsPurpose", ["CRSPurposeCode121", "CRSPurposeCode111"]),
        ("how", "crsTypeOfFinance", ["CRSTypeOfFinance420", "CRSTypeOfFinance110"]),
        ("how", "nettingBasis", ["Net"]),
    ],
)
_DAC2A_LOANS = _shape(
    1,
    population="FinancialFlow",
    slots=[
        ("how", "dacTypeOfFinance", ["DACTypeOfFinanceGrants", "DACTypeOfFinanceNonGrants"]),
        ("how", "nettingBasis", ["Gross", "Net"]),
    ],
)


@pytest.mark.parametrize(
    ("health", "elected"),
    [pytest.param(0.0, 1, id="loans-alone-moves"), pytest.param(0.9, 0, id="health-loans-stays")],
)
def test_another_taxonomy_holds_only_as_many_named_values_as_it_states(health: float, elected: int):
    """DAC2a reads "loans" on its own type of finance, which stands for the CRS shape's
    loan, so a policy moves "loans" there. "Health loans" names a sector too, which DAC2a
    states nowhere, so the CRS pick keeps both."""
    answers = _answers(
        {
            "m|0": 0.9,
            "m|1": 0.9,
            "n|how:crsPurpose|0": health,
            "n|how:crsTypeOfFinance|0": 0.9,
            "n|how:dacTypeOfFinance|1": 0.9,
        },
        elect={0: 0.6, 1: 0.4},
    )

    output, _conf = _assemble(_request([_CRS_LOANS, _DAC2A_LOANS]), answers)

    assert output.chosen_shape_index == elected


def test_a_request_without_a_query_asks_no_query_question():
    request = _request([_shape(0)])
    bare = BindRequest(
        variable=request.variable,
        shapes=request.shapes,
        frame=request.frame,
        query_text=None,
        apply_policies=request.apply_policies,
    )

    state, questions = bind_request(jev_menu(bare))

    assert state == {"variable_phrase": "aid"}
    assert sorted(questions) == ["elect", "m|0", "named|0"]


def test_the_all_question_s_examples_each_name_a_slot_by_its_plain_name():
    """A ``requested_all`` claim stands only on a phrase naming the slot's plain name
    (`_validate_requested_all_bindings`), so each example the question gives names one."""
    plain_names = {d.plain_name for d in get_index().declarations.values()}
    _state, questions = bind_request(jev_menu(_request([_SECTORS], dropped=("in total",))))
    instructions = str(questions[f"all|{_SECTOR}|0"].get("instructions"))
    examples = re.findall(r"'([^']+)'", instructions.split(" as ", 1)[1])

    assert examples
    assert all(any(names_plain_name(e, n) for n in plain_names) for e in examples)


def test_the_quantity_request_reads_its_own_phrase_when_it_differs():
    denominators = [GraphRef(ref="Count_Person", label="Population")]

    state, questions = quantity_request(
        "debt to GDP", denominators, bind_variable="debt to GDP in Kenya", query="q"
    )

    assert state["variable_phrase"] == "debt to GDP in Kenya"
    assert state["quantity_phrase"] == "debt to GDP"
    assert state["denominators"] == ["0: Population"]
    assert sorted(questions) == ["q|base", "q|denominator", "q|form", "q|gate"]
    assert all("`quantity_phrase`" in str(q.get("instructions")) for q in questions.values())


@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        pytest.param({"q|form": {"none": 0.9}}, (None, None, None), id="no-form"),
        pytest.param(
            {"q|form": {"share": 0.9}, "q|base": {"none": 0.9}}, (None, None, "share"), id="no-base"
        ),
        pytest.param(
            {"q|form": {"rate": 0.9}, "q|base": {"GDP": 0.9}, "q|denominator": {"0: GDP": 0.9}},
            ("GDP", 0, "rate"),
            id="listed-base",
        ),
        pytest.param(
            {
                "q|form": {"rate": 0.9},
                "q|base": {"GDP": 0.9},
                "q|denominator": {"none of these": 1},
            },
            ("GDP", 1, "rate"),
            id="unlisted-base",
        ),
    ],
)
def test_the_quantity_read(choices: dict, expected: tuple):
    answers = _answers({"q|gate": 0.9}, choices=choices)

    out = assemble_quantity(answers, [GraphRef(ref="GDP", label="GDP")])

    assert (out.base_words, out.denominator_index, out.stated_form) == expected


def test_the_quantity_read_refuses_answers_without_its_gate():
    with pytest.raises(ValueError, match=r"no q\|gate answer"):
        assemble_quantity(_answers(), [])


def test_the_frame_states_a_form_only_when_the_question_names_one():
    def framed(value: str, stated_by: StatedBy) -> Frame:
        form = Stated[tuple[str, ...]](value=(value,), stated_by=stated_by)
        return _frame().model_copy(update={"normalisation": FrameNormalisation(form=form)})

    assert frame_stated_form(None) is None
    assert frame_stated_form(_frame()) is None
    assert frame_stated_form(framed("level", "default")) is None
    assert frame_stated_form(framed("per_person", "reader")) == "per_person"


_GE = _shape(0, population="FinancialFlow", measured="grantEquivalent", qualifier="Nominal")
_AMOUNT = _shape(1, population="FinancialFlow", measured="amount", qualifier="Nominal")


@pytest.mark.parametrize(
    ("shapes", "nouls", "elected", "pool", "rules"),
    [
        pytest.param(
            [_GE, _AMOUNT],
            {"mv|grantEquivalent": 0.58},
            1,
            [0, 1],
            ["house: amount_vs_ge"],
            id="convention-variant-to-the-policy",
        ),
        pytest.param(
            [_GE, _AMOUNT],
            {"mv|grantEquivalent": 0.95},
            0,
            [0],
            ["named: measuredProperty"],
            id="named-value-keeps-it",
        ),
        pytest.param(
            [_GE, replace(_AMOUNT, datasets=("provenance/OtherDataset",))],
            {"mv|grantEquivalent": 0.58},
            0,
            [0],
            [],
            id="another-dataset-s-variant-stays-out",
        ),
        pytest.param(
            [_shape(0), _shape(1, population="Place")], {}, 0, [0], [], id="other-reading-stays-out"
        ),
    ],
)
def test_the_pool_holds_a_convention_variant_the_election_sets_aside(
    shapes: list[ShapeDraft], nouls: dict[str, float], elected: int, pool: list[int], rules: list
):
    """ "education grants" with no start year: Jev elects grant equivalent 0.92 to 0.075, and
    the amount shape, measuring the phrase within `near`, still reaches `amount_vs_ge`,
    which keeps the amount. A shape of another reading, or of the same reading in another
    dataset, that the election sets aside stays out."""
    answers = _answers({"m|0": 0.86, "m|1": 0.81, **nouls}, elect={0: 0.92, 1: 0.075})

    output, conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == elected
    assert conf.pool is not None and sorted(conf.pool.measuring) == pool
    assert conf.rules == rules


@pytest.mark.parametrize(("none", "asks"), [(0.35, False), (0.36, True)])
def test_the_part_asks_when_electq_puts_none_above_every_shape(none: float, asks: bool):
    shapes = [_shape(0), _shape(1, population="Place")]
    electq = {"Shape 0": 0.3, "Shape 1": 0.7 - none, "none": none}
    answers = _answers({"m|0": 0.3, "m|1": 0.3}, elect={0: 0.5, 1: 0.5}, choices={"electq": electq})

    output, conf = _assemble(_request(shapes), answers)

    assert (output.ask is not None) is asks
    assert ("ask: none of the shapes gives what the query asks for" in conf.rules) is asks


@pytest.mark.parametrize(("form", "elected", "also"), [(None, 0, [1]), ("per_person", 1, None)])
def test_a_reading_of_another_base_is_an_alternate_only_when_the_phrase_states_no_form(
    form: str | None, elected: int, also: list[int] | None
):
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 0.45, 1: 0.55})

    output, _conf = _assemble(_request(_PLAIN_AND_DIVIDED), answers, form=form)

    assert output.chosen_shape_index == elected
    assert output.also_plausible == also


def test_the_bind_result_carries_each_shape_s_scores():
    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers(
        {"m|0": 0.9, "m|1": 0.8},
        elect={0: 0.6, 1: 0.4},
        choices={"electq": {"Shape 0": 0.5, "Shape 1": 0.1, "none": 0.4}},
    )

    result, _conf = bind_result(jev_menu(_request(shapes)), answers, stated_form=None)

    assert result.scores == {"measure": [0.9, 0.8], "election": [0.55, 0.25], "none": 0.4}
