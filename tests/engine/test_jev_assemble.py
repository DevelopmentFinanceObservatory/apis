"""Jev assembly: each threshold's cut-off, the behaviours a rewrite loses silently, and the
house policies on answer sets no dev item reaches."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import cast

import pytest

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.binders.base import ApplyPolicies, BindRequest
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.binders.jev_assemble import (
    ASK_TEXT,
    SPAN_SEARCH_BUDGET,
    Confidence,
    JevAnswers,
    SpanRound,
    _Claim,
    _owners,
    _Search,
    assemble,
    assemble_quantity,
    bind_result,
    frame_stated_form,
    span_round,
)
from search.engine.binders.jev_menu import asked_values, jev_menu
from search.engine.binders.jev_request import (
    SPAN_MEASURE,
    bind_request,
    quantity_request,
    span_id,
)
from search.engine.door_names import own_texts
from search.engine.frame import Frame, FrameNormalisation, Stated, StatedBy, door_frame
from search.engine.qualifier_match import QualifierMatch, QualifierTie, names_plain_name
from search.engine.region.phase_rank_bind import _HousePolicyStep
from search.engine.role_text import DoorRole
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


def _frame(*, donors: Sequence[str] = ()) -> Frame:
    return door_frame(
        door="parsed",
        variable="aid",
        entities=list(donors),
        entity_texts=own_texts(list(donors)),
        entity_roles={name: [DoorRole(axis="origin", span=name)] for name in donors},
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
    variable: str = "aid",
    dropped: tuple[str, ...] = (),
    ties: Sequence[tuple[QualifierTie, ...]] = (),
    availability: Sequence[bool | None] = (),
    donors: Sequence[str] = (),
    apply_policies: ApplyPolicies | None = None,
) -> BindRequest:
    frame = _frame(donors=donors)
    step = _HousePolicyStep(
        shapes=shapes,
        frame=frame,
        admitted_qualifiers=(),
        availability=availability,
        code_index=get_index(),
    )
    return BindRequest(
        variable=variable,
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
    _output, conf = _assemble(_request([_shape(0), _shape(1, population="Place")]), answers)
    return sorted(conf.plausible)


def _nominated(nouls: Mapping[str, float]) -> object:
    """The plausible shapes and the alternates of two shapes measuring the phrase equally
    well, shape 0 elected at 0.5."""
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 0.5, 1: nouls["elect|1"]})
    output, conf = _assemble(_request([_shape(0), _shape(1, population="Place")]), answers)
    return sorted(conf.plausible), output.also_plausible


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
    "decline": _Cut(_declined, "m|0", 0.3, 0.31, True, False),
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
    "near_alt": _Cut(_near_alt, "elect|1", 0.33, 0.34, [0], [0, 1]),
    # A plausible shape is nominated at 0.9 of the elected shape's election, Gemini's
    # rated-pool bar (`gemini.RATED_POOL_FRACTION`).
    "nominated": _Cut(_nominated, "elect|1", 0.425, 0.45, ([0, 1], None), ([0, 1], [1])),
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
    answers = _answers({"m|0": 0.9, "m|1": measure}, elect={0: 0.52, 1: 0.48})

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
    (`bind.validate_requested_all_bindings`), so each example the question gives names one."""
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


# -- span ownership ----------------------------------------------------------------------

_POSITION: tuple[Axis, str, list[str]] = (
    "how",
    "financialPositionType",
    ["AmountOutstanding", "AmountUndisbursed"],
)
_GUARANTEE: tuple[Axis, str, list[str]] = (
    "how",
    "guaranteeStatus",
    ["PublicOrPubliclyGuaranteed", "PubliclyGuaranteedClaim", "NotPubliclyGuaranteed"],
)
_DEBTOR: tuple[Axis, str, list[str]] = (
    "how",
    "idsDebtorType",
    [
        "IDSDebtorTypeCB",
        "IDSDebtorTypeGG",
        "IDSDebtorTypeOPS",
        "IDSDebtorTypePS",
        "IDSDebtorTypePrivateSector",
    ],
)
_CREDITOR: tuple[Axis, str, list[str]] = (
    "how",
    "idsCreditorType",
    ["IDSCreditorTypePCBK", "IDSCreditorTypePRVT"],
)
_RESOURCE: tuple[Axis, str, list[str]] = ("how", "financingResource", ["Loan", "DebtSecurity"])
_IDS = _shape(
    0,
    population="FinancialPosition",
    measured="amount",
    label="External debt, by creditor",
    slots=[_POSITION, _GUARANTEE, _DEBTOR, _CREDITOR, _RESOURCE],
)
_IDS_WITHOUT_DEBTOR = _shape(
    1,
    population="FinancialPosition",
    measured="amount",
    label="External debt",
    slots=[_POSITION, _GUARANTEE],
)


def _ids_member(**constraints: str) -> dict[str, str]:
    return {"financialPositionType": "AmountOutstanding", **constraints}


# The IDS debt-stock members as the recorded Kenya, Zambia and Bangladesh binds read them.
# No member holds the guarantee-status aggregate beside a debtor type, nor a private
# non-guaranteed debtor beside private creditors, nor a loan beside commercial banks.
_IDS_MEMBERS = (
    _ids_member(guaranteeStatus="PublicOrPubliclyGuaranteed"),
    _ids_member(idsDebtorType="IDSDebtorTypePS"),
    _ids_member(idsDebtorType="IDSDebtorTypeGG"),
    _ids_member(
        guaranteeStatus="PubliclyGuaranteedClaim", idsDebtorType="IDSDebtorTypePrivateSector"
    ),
    _ids_member(
        guaranteeStatus="NotPubliclyGuaranteed", idsDebtorType="IDSDebtorTypePrivateSector"
    ),
    _ids_member(
        guaranteeStatus="NotPubliclyGuaranteed",
        idsDebtorType="IDSDebtorTypePrivateSector",
        idsCreditorType="IDSCreditorTypePBND",
    ),
    _ids_member(idsCreditorType="IDSCreditorTypePRVT"),
    _ids_member(idsCreditorType="IDSCreditorTypePCBK"),
    _ids_member(financingResource="Loan"),
)


def _span_answers(
    request: BindRequest,
    named: Mapping[tuple[str, str], float],
    spans: Mapping[str, str | tuple[str, float]],
    **nouls: float,
) -> JevAnswers:
    """Both rounds' answers over `request`: each slot value of `named`, keyed by its slot
    (``axis:property``) and dcid, at its probability, and each span answer of `spans`,
    keyed by value dcid or ``measure``, at 1.0 or at the probability given beside it."""
    menu = jev_menu(request)
    props: dict[str, tuple[Axis, str]] = {}
    asked: dict[str, float] = {}
    for (pid, value), p in named.items():
        axis, prop = pid.split(":")
        props[value] = (cast(Axis, axis), prop)
        asked[f"n|{pid}|{asked_values(menu, props[value]).index(value)}"] = p
    choices: dict[str, Mapping[str, float]] = {}
    for key, answer in spans.items():
        words, p = (answer, 1.0) if isinstance(answer, str) else answer
        choices[SPAN_MEASURE if key == "measure" else span_id(props[key], key)] = {words: p}
    return _answers({"m|0": 0.95, **asked, **nouls}, elect={0: 1.0}, choices=choices)


def _bound(output: _UniversalBindOutput) -> dict[str, list[str]]:
    return {
        b.property_dcid or "": list(b.value_dcids)
        for b in output.bindings
        if b.kind in ("value", "set")
    }


_KENYA = "public and publicly guaranteed external debt stocks"


@pytest.mark.parametrize(
    ("debtor", "creditor", "kept"),
    [
        pytest.param((0.7, 1.0), (0.8, 1.0), "idsCreditorType", id="likelier-claim-unheld"),
        pytest.param((0.7, 1.0), (0.9, 0.5), "idsDebtorType", id="weaker-words-lose"),
    ],
)
def test_two_values_on_one_stretch_keep_the_likelier_claim_held_or_not(debtor, creditor, kept):
    """ "private sector" names the private debtor and the private creditors. The claim with
    the larger log-odds of naming its value in those words, logit(p_n x p_span), stays,
    whether or not a member holds it beside the guarantee status, so a typed value stays
    against a weaker one that happens to fit the data."""
    request = _request(
        [_IDS], variable="not publicly guaranteed external debt owed by the private sector"
    )
    answers = _span_answers(
        request,
        {
            ("how:guaranteeStatus", "NotPubliclyGuaranteed"): 0.95,
            ("how:idsDebtorType", "IDSDebtorTypePrivateSector"): debtor[0],
            ("how:idsCreditorType", "IDSCreditorTypePRVT"): creditor[0],
        },
        {
            "NotPubliclyGuaranteed": "not publicly guaranteed",
            "IDSDebtorTypePrivateSector": ("private sector", debtor[1]),
            "IDSCreditorTypePRVT": ("private sector", creditor[1]),
            "measure": "external debt",
        },
        **{"part|how:idsCreditorType": 0.27},
    )

    output, _conf = assemble(
        jev_menu(request), answers, stated_form=None, populations={0: _IDS_MEMBERS}
    )

    assert set(_bound(output)) == {"guaranteeStatus", kept}


def test_two_weak_claims_on_one_stretch_keep_one():
    """Both claims on "private sector" weigh below even odds (logit(0.8 x 0.55) and
    logit(0.75 x 0.55)). A value is dropped only for one that keeps the words, so the
    likelier stays and the user's words still narrow the answer."""
    request = _request([_IDS], variable="external debt owed by the private sector")
    answers = _span_answers(
        request,
        {
            ("how:idsDebtorType", "IDSDebtorTypePrivateSector"): 0.8,
            ("how:idsCreditorType", "IDSCreditorTypePRVT"): 0.75,
        },
        {
            "IDSDebtorTypePrivateSector": ("private sector", 0.55),
            "IDSCreditorTypePRVT": ("private sector", 0.55),
            "measure": "external debt",
        },
        **{"part|how:idsCreditorType": 0.27, "part|how:idsDebtorType": 0.3},
    )

    output, conf = assemble(jev_menu(request), answers, stated_form=None)

    assert _bound(output) == {"idsDebtorType": ["IDSDebtorTypePrivateSector"]}
    assert conf.inferred == {"how:idsCreditorType": ("IDSCreditorTypePRVT",)}


@pytest.mark.parametrize(
    ("population", "bound"),
    [
        pytest.param(
            _IDS_MEMBERS,
            {"idsCreditorType": ["IDSCreditorTypePCBK"]},
            id="complete-population-drops-the-unheld-loan",
        ),
        pytest.param(
            None,
            {"idsCreditorType": ["IDSCreditorTypePCBK"], "financingResource": ["Loan"]},
            id="no-population-keeps-it",
        ),
    ],
)
def test_a_value_in_the_measure_s_words_stays_only_where_members_hold_it(population, bound):
    """A loan in the measure's "debt" stays only beside a value a member holds it with.
    With no complete population nothing is known, so it stays and the engine refuses
    loudly."""
    request = _request([_IDS], variable="external debt to commercial banks")
    answers = _span_answers(
        request,
        {
            ("how:idsCreditorType", "IDSCreditorTypePCBK"): 0.93,
            ("how:financingResource", "Loan"): 0.73,
        },
        {"IDSCreditorTypePCBK": "commercial banks", "Loan": "debt", "measure": "external debt"},
        **{"part|how:financingResource": 0.35},
    )
    populations = {0: population} if population is not None else None

    output, _conf = assemble(jev_menu(request), answers, stated_form=None, populations=populations)

    assert _bound(output) == bound


@pytest.mark.parametrize(
    ("population", "bound"),
    [
        pytest.param(_IDS_MEMBERS, {"idsCreditorType": ["IDSCreditorTypePCBK"]}, id="narrows"),
        pytest.param(
            (_ids_member(idsCreditorType="IDSCreditorTypePCBK", financingResource="Loan"),),
            {"idsCreditorType": ["IDSCreditorTypePCBK"], "financingResource": ["Loan"]},
            id="narrows-nothing",
        ),
        pytest.param(None, {"idsCreditorType": ["IDSCreditorTypePCBK"]}, id="no-population"),
    ],
)
def test_a_value_without_words_of_its_own_stays_only_where_it_narrows_nothing(population, bound):
    """Jev infers a loan (0.73) beside "commercial banks" and finds no words for it. It
    stays only when every member holding what the bind keeps holds a loan too."""
    request = _request([_IDS], variable="external debt to commercial banks")
    answers = _span_answers(
        request,
        {
            ("how:idsCreditorType", "IDSCreditorTypePCBK"): 0.93,
            ("how:financingResource", "Loan"): 0.73,
        },
        {"IDSCreditorTypePCBK": "commercial banks", "Loan": "none", "measure": "external debt"},
        **{"part|how:financingResource": 0.35},
    )
    populations = {0: population} if population is not None else None

    output, _conf = assemble(jev_menu(request), answers, stated_form=None, populations=populations)

    assert _bound(output) == bound


_NET = _shape(
    0,
    population="FinancialFlow",
    measured="amount",
    slots=[
        ("how", "nettingBasis", ["Gross", "Net"]),
        ("how", "netOf", ["PrincipalRepayment", "InterestPayment"]),
    ],
)


@pytest.mark.parametrize(
    ("population", "bound"),
    [
        pytest.param(
            (
                {"nettingBasis": "Gross"},
                {"nettingBasis": "Net", "netOf": "PrincipalRepayment"},
            ),
            {"nettingBasis": ["Net"], "netOf": ["PrincipalRepayment"]},
            id="complete-population-keeps-both",
        ),
        pytest.param(None, {"nettingBasis": ["Net"]}, id="no-population-keeps-one"),
    ],
)
def test_one_concept_on_two_properties_keeps_both_values(population, bound):
    """ "Net flows" names one concept stored as a netting basis and what the flow is net of
    (`test_source_code._CONCESSIONAL_NET`). Every member that is net is net of principal
    repayments, so binding the second narrows nothing and both stay. Without a complete
    population no value is exempt."""
    request = _request([_NET], variable="net ODA flows")
    answers = _span_answers(
        request,
        {("how:nettingBasis", "Net"): 0.95, ("how:netOf", "PrincipalRepayment"): 0.8},
        {"Net": "net", "PrincipalRepayment": "net", "measure": "ODA flows"},
        **{"part|how:netOf": 0.5, "part|how:nettingBasis": 0.5},
    )
    populations = {0: population} if population is not None else None

    output, _conf = assemble(jev_menu(request), answers, stated_form=None, populations=populations)

    assert _bound(output) == bound


def test_the_span_round_asks_only_for_a_shape_naming_values_on_two_slots():
    """One named slot asks nothing more. Two ask which words name each value and the
    measure, keyed by slot and value."""
    request = _request([_IDS], variable=_KENYA)
    one = _span_answers(request, {("how:guaranteeStatus", "PublicOrPubliclyGuaranteed"): 0.97}, {})
    two = _span_answers(
        request,
        {
            ("how:guaranteeStatus", "PublicOrPubliclyGuaranteed"): 0.97,
            ("how:idsDebtorType", "IDSDebtorTypePS"): 0.75,
        },
        {},
        **{"part|how:idsDebtorType": 0.3},
    )
    menu = jev_menu(request)

    assert span_round(menu, one, stated_form=None) == SpanRound({}, ())
    asked = span_round(menu, two, stated_form=None)
    assert sorted(asked.questions) == [
        "span|how:guaranteeStatus|PublicOrPubliclyGuaranteed",
        "span|how:idsDebtorType|IDSDebtorTypePS",
        SPAN_MEASURE,
    ]
    assert asked.shapes == (0,)
    options = asked.questions[SPAN_MEASURE].get("criteria") or {}
    assert {"public", "public and publicly guaranteed", "none"} <= set(options)


def test_the_house_policies_weigh_the_drafts_the_span_rule_leaves():
    """The pick's public-sector debtor shares its words with the guarantee status, so a
    shape without a debtor slot holds every value the bind keeps."""
    seen: list[Mapping[int, bool] | None] = []

    def recording(pool, *, ratings=None, holds=None):
        seen.append(holds)
        return PolicyChoice(survivor=pool.measuring[0], fired=(), dropped=frozenset(), facts=())

    request = _request([_IDS, _IDS_WITHOUT_DEBTOR], variable=_KENYA, apply_policies=recording)
    answers = _span_answers(
        request,
        {
            ("how:guaranteeStatus", "PublicOrPubliclyGuaranteed"): 0.97,
            ("how:idsDebtorType", "IDSDebtorTypePS"): 0.75,
        },
        {
            "PublicOrPubliclyGuaranteed": "public and publicly guaranteed",
            "IDSDebtorTypePS": "public",
        },
        **{"m|1": 0.95, "part|how:idsDebtorType": 0.3},
    )
    answers = JevAnswers(
        nouls=answers.nouls, choices={**answers.choices, "elect": {"Shape 0": 0.6, "Shape 1": 0.4}}
    )

    _assemble(request, answers)

    assert seen == [{0: True, 1: True}]


def test_a_word_written_twice_pays_for_one_value_each_time():
    """ "private" names the debtor at its first occurrence and the creditors at its second,
    so neither claim takes the other's words."""
    request = _request([_IDS], variable="private sector debt to private creditors")
    answers = _span_answers(
        request,
        {
            ("how:idsDebtorType", "IDSDebtorTypePrivateSector"): 0.95,
            ("how:idsCreditorType", "IDSCreditorTypePRVT"): 0.9,
        },
        {
            "IDSDebtorTypePrivateSector": "private sector",
            "IDSCreditorTypePRVT": "private",
            "measure": "debt",
        },
        **{"part|how:idsDebtorType": 0.9, "part|how:idsCreditorType": 0.9},
    )

    output, conf = assemble(jev_menu(request), answers, stated_form=None)

    assert set(_bound(output)) == {"idsDebtorType", "idsCreditorType"}
    assert conf.inferred == {}


def test_three_claims_on_a_word_written_twice_keep_two():
    """The debtor takes the first "private", so the creditors and the non-guaranteed status
    contest the second. Each pair fits, but the three together do not: the lightest drops."""
    request = _request([_IDS], variable="private sector debt to private creditors")
    answers = _span_answers(
        request,
        {
            ("how:idsDebtorType", "IDSDebtorTypePrivateSector"): 0.95,
            ("how:idsCreditorType", "IDSCreditorTypePRVT"): 0.9,
            ("how:guaranteeStatus", "NotPubliclyGuaranteed"): 0.8,
        },
        {
            "IDSDebtorTypePrivateSector": "private sector",
            "IDSCreditorTypePRVT": "private",
            "NotPubliclyGuaranteed": "private",
            "measure": "debt",
        },
        **{"part|how:idsDebtorType": 0.9, "part|how:idsCreditorType": 0.9},
    )

    output, conf = assemble(jev_menu(request), answers, stated_form=None)

    assert set(_bound(output)) == {"idsDebtorType", "idsCreditorType"}
    assert conf.inferred == {"how:guaranteeStatus": ("NotPubliclyGuaranteed",)}


_HEALTH_AND_LOANS = _shape(
    0,
    population="FinancialFlow",
    measured="amount",
    slots=[
        ("how", "crsPurpose", ["CRSPurposeCode120", "CRSPurposeCode110"]),
        ("how", "crsTypeOfFinance", ["CRSTypeOfFinance420", "CRSTypeOfFinance110"]),
    ],
)


def test_a_value_named_by_the_measure_s_own_words_stays_where_no_member_holds_it():
    """ "loans" is the measure's whole span and the finance type's: the user chose it, so it
    stays though only health grants and education loans exist, and the engine refuses."""
    request = _request([_HEALTH_AND_LOANS], variable="health loans")
    answers = _span_answers(
        request,
        {
            ("how:crsPurpose", "CRSPurposeCode120"): 0.95,
            ("how:crsTypeOfFinance", "CRSTypeOfFinance420"): 0.95,
        },
        {"CRSPurposeCode120": "health", "CRSTypeOfFinance420": "loans", "measure": "loans"},
    )
    population = (
        {"crsPurpose": "CRSPurposeCode120", "crsTypeOfFinance": "CRSTypeOfFinance110"},
        {"crsPurpose": "CRSPurposeCode110", "crsTypeOfFinance": "CRSTypeOfFinance420"},
    )

    output, _conf = assemble(
        jev_menu(request), answers, stated_form=None, populations={0: population}
    )

    assert _bound(output) == {
        "crsPurpose": ["CRSPurposeCode120"],
        "crsTypeOfFinance": ["CRSTypeOfFinance420"],
    }


def test_each_value_of_a_set_pays_for_its_own_words():
    """A set's value Jev finds no words for is judged alone and leaves the set; the value
    the phrase names stays."""
    request = _request([_HEALTH_AND_LOANS], variable="health loans")
    answers = _span_answers(
        request,
        {
            ("how:crsPurpose", "CRSPurposeCode120"): 0.95,
            ("how:crsPurpose", "CRSPurposeCode110"): 0.85,
            ("how:crsTypeOfFinance", "CRSTypeOfFinance420"): 0.95,
        },
        {
            "CRSPurposeCode120": "health",
            "CRSPurposeCode110": "none",
            "CRSTypeOfFinance420": "loans",
            "measure": "none",
        },
        **{"several|how:crsPurpose": 0.9},
    )

    output, conf = assemble(jev_menu(request), answers, stated_form=None)

    assert _bound(output)["crsPurpose"] == ["CRSPurposeCode120"]
    assert conf.inferred == {"how:crsPurpose": ("CRSPurposeCode110",)}


def test_a_set_keeps_the_value_no_other_claim_contests():
    """B's words are its own; A and C claim the same word, so one of them stays."""
    shape = _shape(0, slots=[("how", "p", ["A", "B"]), ("how", "q", ["C", "D"])])
    request = _request([shape], variable="x y")
    answers = _span_answers(
        request,
        {("how:p", "A"): 0.95, ("how:p", "B"): 0.95, ("how:q", "C"): 0.95},
        {"A": "x", "B": "y", "C": "x", "measure": "none"},
        **{"several|how:p": 0.7},
    )

    output, _conf = assemble(jev_menu(request), answers, stated_form=None)

    bound = _bound(output)
    assert "B" in bound.get("p", [])
    assert ("A" in bound.get("p", [])) != ("C" in bound.get("q", []))


def _owners_by_enumeration(worded: Sequence[_Claim], reserved: frozenset[int]) -> set[_Claim]:
    """Step 1 of the span rule over every subset of `worded`: the maximal word-consistent
    subset with the largest sum of w, ties to the first sorted (property, value) list.
    Occurrences of two slots clash on a shared word, of one slot when one strictly
    contains the other."""

    def clash(a: _Claim, o: frozenset[int], b: _Claim, q: frozenset[int]) -> bool:
        return bool(o & q) if a.slot != b.slot else o < q or q < o

    def fits(claims: Sequence[_Claim]) -> bool:
        placing = [c for c in claims if c.occurrences]
        choices = [[o for o in c.occurrences or () if not o & reserved] for c in placing]
        return any(
            not any(
                clash(a, o, b, q)
                for (a, o), (b, q) in combinations(zip(placing, chosen, strict=True), 2)
            )
            for chosen in product(*choices)
        )

    fitting = [list(c) for r in range(len(worded) + 1) for c in combinations(worded, r)]
    fitting = [c for c in fitting if fits(c)]
    maximal = [c for c in fitting if not any(fits([*c, x]) for x in worded if x not in c)]
    return set(
        min(
            maximal,
            key=lambda c: (
                -sum(x.weight for x in c),
                sorted((x.property_dcid, x.value) for x in c),
            ),
        )
    )


def _random_claims(rng: random.Random) -> tuple[list[_Claim], frozenset[int]]:
    """Up to eight claims on two or three slots over four words, each written once, twice or
    not read, some spanning two words, and an anchor on one word half the time."""
    slots = rng.randint(2, 3)
    claims = []
    for k in range(rng.randint(1, 8)):
        slot = rng.randrange(slots)
        occurrences = []
        for _ in range(rng.choice([0, 1, 1, 2])):
            start = rng.randrange(4)
            occurrences.append(frozenset(range(start, min(start + rng.randint(1, 2), 4))))
        claims.append(_Claim(slot, f"p{slot}", f"v{k}", rng.uniform(-3, 3), tuple(occurrences)))
    reserved = frozenset({rng.randrange(4)}) if rng.random() < 0.5 else frozenset()
    return claims, reserved


@pytest.mark.parametrize("seed", range(200))
def test_the_word_owners_are_the_heaviest_maximal_subset_of_every_subset(seed: int):
    """The search in groups (`_owners`) keeps what enumerating every subset keeps."""
    claims, reserved = _random_claims(random.Random(seed))

    owners = _owners(claims, reserved, _Search(SPAN_SEARCH_BUDGET))

    assert set(owners) == _owners_by_enumeration(claims, reserved)


@pytest.mark.parametrize(
    ("spans", "kept"),
    [
        ({"A": "public or publicly guaranteed", "B": "publicly guaranteed"}, ["A"]),
        ({"A": "public", "B": "public"}, ["A", "B"]),
    ],
    ids=["contained", "equal"],
)
def test_a_value_whose_words_contain_another_s_of_its_slot_takes_them(
    spans: dict[str, str], kept: list[str]
):
    """Two values of one slot on the same words coexist as a set, but a value whose words
    lie strictly within another's names no second value: the heavier keeps the words."""
    shape = _shape(0, slots=[("how", "p", ["A", "B"]), ("how", "q", ["C"])])
    request = _request([shape], variable="public or publicly guaranteed debt")
    answers = _span_answers(
        request,
        {("how:p", "A"): 0.9, ("how:p", "B"): 0.85, ("how:q", "C"): 0.9},
        {**spans, "C": "debt", "measure": "none"},
        **{"several|how:p": 0.8},
    )

    output, _conf = assemble(jev_menu(request), answers, stated_form=None)

    assert _bound(output) == {"p": kept, "q": ["C"]}


def test_tied_word_owners_fall_to_the_first_sorted_list_across_groups():
    """Two groups of two claims contending for one word each, every claim weighing 1: each
    group keeps its claim on the first property, as the whole subset's sorted list does."""
    claims = [
        _Claim(0, "p0", "a", 1.0, (frozenset({0}),)),
        _Claim(1, "p1", "b", 1.0, (frozenset({0}),)),
        _Claim(1, "p1", "d", 1.0, (frozenset({1}),)),
        _Claim(0, "p0", "c", 1.0, (frozenset({1}),)),
    ]

    owners = _owners(claims, frozenset(), _Search(SPAN_SEARCH_BUDGET))

    assert {c.value for c in owners} == {"a", "c"}
    assert set(owners) == _owners_by_enumeration(claims, frozenset())


def test_twenty_values_claiming_one_word_settle_within_the_search_budget():
    """Ten values of each of two slots all claim "x": the heavier slot keeps every one,
    and the span rule settles it without passing its budget."""
    ps = [f"P{i}" for i in range(10)]
    qs = [f"Q{i}" for i in range(10)]
    shape = _shape(0, slots=[("how", "p", ps), ("how", "q", qs)])
    request = _request([shape], variable="x y")
    named = {("how:p", v): 0.9 + i * 0.005 for i, v in enumerate(ps)}
    named |= {("how:q", v): 0.9 + i * 0.004 for i, v in enumerate(qs)}
    answers = _span_answers(
        request,
        named,
        {**dict.fromkeys(ps + qs, "x"), "measure": "none"},
        **{"several|how:p": 0.9, "several|how:q": 0.9},
    )

    output, conf = assemble(jev_menu(request), answers, stated_form=None)

    assert _bound(output) == {"p": ps}
    assert conf.inferred == {"how:q": tuple(qs)}


_ONE_GUARANTEE = replace(
    _IDS,
    slot_taxonomy={
        **(_IDS.slot_taxonomy or {}),
        "how:guaranteeStatus": ["PublicOrPubliclyGuaranteed"],
    },
)


def test_a_named_one_member_value_contests_its_words_by_weight():
    """The one-member guarantee status and the public-sector debtor both read "public". The
    guarantee status weighs more and keeps the word, so the debtor is left unbound."""
    request = _request([_ONE_GUARANTEE], variable="public debt")
    answers = _span_answers(
        request,
        {
            ("how:guaranteeStatus", "PublicOrPubliclyGuaranteed"): 0.95,
            ("how:idsDebtorType", "IDSDebtorTypePS"): 0.85,
        },
        {"PublicOrPubliclyGuaranteed": "public", "IDSDebtorTypePS": "public", "measure": "debt"},
        **{"part|how:idsDebtorType": 0.9},
    )
    population = (
        {"guaranteeStatus": "PublicOrPubliclyGuaranteed"},
        {"idsDebtorType": "IDSDebtorTypePS"},
    )

    output, conf = assemble(
        jev_menu(request), answers, stated_form=None, populations={0: population}
    )

    assert _bound(output) == {"guaranteeStatus": ["PublicOrPubliclyGuaranteed"]}
    assert conf.inferred == {"how:idsDebtorType": ("IDSDebtorTypePS",)}


def test_a_named_one_member_value_that_loses_its_words_stays_as_the_slot_s_only_option():
    request = _request([_ONE_GUARANTEE], variable="public debt")
    answers = _span_answers(
        request,
        {
            ("how:guaranteeStatus", "PublicOrPubliclyGuaranteed"): 0.7,
            ("how:idsDebtorType", "IDSDebtorTypePS"): 0.95,
        },
        {"PublicOrPubliclyGuaranteed": "public", "IDSDebtorTypePS": "public", "measure": "debt"},
        **{"part|how:idsDebtorType": 0.9},
    )

    output, conf = assemble(jev_menu(request), answers, stated_form=None)

    assert _binding(output, "how:guaranteeStatus").evidence == "sole_option"
    assert _bound(output)["idsDebtorType"] == ["IDSDebtorTypePS"]
    assert conf.inferred == {"how:guaranteeStatus": ("PublicOrPubliclyGuaranteed",)}


@pytest.mark.parametrize(
    ("shape", "start", "bound"),
    [
        pytest.param(
            _IDS, 0, {"guaranteeStatus": ["PublicOrPubliclyGuaranteed"]}, id="matched-words"
        ),
        pytest.param(
            _ONE_GUARANTEE,
            0,
            {"guaranteeStatus": ["PublicOrPubliclyGuaranteed"]},
            id="matched-words-one-member",
        ),
        pytest.param(
            _IDS,
            None,
            {
                "guaranteeStatus": ["PublicOrPubliclyGuaranteed"],
                "idsDebtorType": ["IDSDebtorTypePS"],
            },
            id="no-offsets",
        ),
    ],
)
def test_a_value_the_matcher_fixed_keeps_its_words_whatever_jev_weighs(shape, start, bound):
    """The matcher fixed the guarantee status from "public"; the debtor reading the same word
    loses however likely Jev rates it, and so when the status is the slot's only value and
    Jev does not name it. A match read from other text reserves no words."""
    match = QualifierMatch(
        code_system="guaranteeStatus",
        code_dcid="PublicOrPubliclyGuaranteed",
        label="public or publicly guaranteed",
        source="label",
        matched_text="public",
        start=start,
        end=None if start is None else start + len("public"),
    )
    request = replace(_request([shape], variable="public debt"), admitted_qualifiers=[(match,)])
    answers = _span_answers(
        request,
        {("how:idsDebtorType", "IDSDebtorTypePS"): 0.99},
        {"IDSDebtorTypePS": "public", "measure": "debt"},
        **{"part|how:idsDebtorType": 0.9},
    )

    output, _conf = assemble(jev_menu(request), answers, stated_form=None)

    assert _bound(output) == bound


_GE = _shape(0, population="FinancialFlow", measured="grantEquivalent", qualifier="Nominal")
_AMOUNT = _shape(1, population="FinancialFlow", measured="amount", qualifier="Nominal")
# The grant-equivalent family lists a modality the amount family lacks.
_MODALITY = "crsCooperationModality"
_GE_MODALITIES = replace(
    _GE,
    slot_keys=(SlotKeyDraft(axis="how", property_dcid=_MODALITY, label=_MODALITY),),
    slot_taxonomy={f"how:{_MODALITY}": ["CRSCooperationModalityB021", "CRSCooperationModalityB05"]},
)
_AMOUNT_MODALITIES = replace(
    _AMOUNT,
    slot_keys=_GE_MODALITIES.slot_keys,
    slot_taxonomy={f"how:{_MODALITY}": ["CRSCooperationModalityB021"]},
)
_EVERY_MODALITY = {"mv|grantEquivalent": 0.58, f"all|how:{_MODALITY}|0": 0.9}
# An imputed multilateral flow, its own dataset, and the amount of the donor house default.
_IMPUTED = replace(
    _shape(0, population="FinancialFlow", measured="attributedAmount", qualifier="Nominal"),
    datasets=("provenance/ONEImputedMultilateralSectors",),
)
_DONOR_DEFAULT_AMOUNT = replace(_AMOUNT, datasets=("provenance/OECDDAC1",))
# Two measures, the second with data where the first has none.
_LIFE = replace(_shape(0, measured="lifeExpectancy"), datasets=("provenance/WHO",))
_HEALTHY_LIFE = replace(_shape(1, measured="healthyLifeExpectancy"), datasets=("provenance/WHO",))


@pytest.mark.parametrize(
    ("shapes", "nouls", "dropped", "availability", "donors", "elected", "pool", "rules"),
    [
        pytest.param(
            [_GE, _AMOUNT],
            {"mv|grantEquivalent": 0.58},
            (),
            (),
            (),
            1,
            [0, 1],
            ["house: amount_vs_ge"],
            id="convention-variant-to-the-policy",
        ),
        pytest.param(
            [_GE_MODALITIES, _AMOUNT_MODALITIES],
            _EVERY_MODALITY,
            ("total",),
            (),
            (),
            1,
            [0, 1],
            ["house: amount_vs_ge"],
            id="total-names-no-slot-so-the-policy-still-reaches-it",
        ),
        pytest.param(
            [_GE_MODALITIES, _AMOUNT_MODALITIES],
            _EVERY_MODALITY,
            ("across all modalities",),
            (),
            (),
            0,
            [0, 1],
            [],
            id="every-modality-sets-the-variant-aside",
        ),
        pytest.param(
            [_GE, _AMOUNT],
            {"mv|grantEquivalent": 0.95},
            (),
            (),
            (),
            0,
            [0],
            ["named: measuredProperty"],
            id="named-value-keeps-it",
        ),
        pytest.param(
            [_shape(0), _shape(1, population="Place")],
            {},
            (),
            (),
            (),
            0,
            [0],
            [],
            id="other-reading-stays-out",
        ),
        pytest.param(
            [_LIFE, _HEALTHY_LIFE],
            {},
            (),
            (False, True),
            (),
            0,
            [0],
            [],
            id="a-measure-the-election-sets-aside-stays-out-for-having-data",
        ),
        pytest.param(
            [_IMPUTED, _DONOR_DEFAULT_AMOUNT],
            {},
            (),
            (),
            ("France",),
            0,
            [0],
            [],
            id="the-house-default-s-amount-stays-out-beside-an-imputed-lead",
        ),
    ],
)
def test_the_pool_holds_a_convention_variant_the_election_sets_aside(
    shapes: list[ShapeDraft],
    nouls: dict[str, float],
    dropped: tuple[str, ...],
    availability: tuple[bool, ...],
    donors: tuple[str, ...],
    elected: int,
    pool: list[int],
    rules: list,
):
    """ "education grants" with no start year: Jev elects grant equivalent 0.92 to 0.075, and
    the amount shape, measuring the phrase within `near`, still reaches `amount_vs_ge`,
    which keeps the amount. "Total" names no slot, so Jev's claim that it asks for every
    modality is demoted before the holds check, and the amount shape still reaches the
    policy. "Across all modalities" asks for the one the amount shape lacks, so the policy
    keeps grant equivalent. A shape of another reading that the election sets aside stays
    out, and so does another measure that has data where the lead has none, or that the
    house default dataset of a donor question publishes beside an imputed lead, since data
    at the places and the house default are no convention between measures."""
    answers = _answers({"m|0": 0.86, "m|1": 0.81, **nouls}, elect={0: 0.92, 1: 0.075})

    output, conf = _assemble(
        _request(shapes, dropped=dropped, availability=availability, donors=donors), answers
    )

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


@pytest.mark.parametrize(("form", "elected", "also"), [(None, 0, None), ("per_person", 1, None)])
def test_a_reading_of_another_base_is_no_alternate(
    form: str | None, elected: int, also: list[int] | None
):
    """A phrase stating a form elects the shape dividing by its base and offers no other
    base. A phrase stating none elects the plain measure and offers no ratio."""
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


@pytest.mark.parametrize(
    ("measure", "none", "none_first"),
    [
        pytest.param(0.9, 0.5, True, id="elected-under-none"),
        pytest.param(0.9, 0.2, False, id="elected-over-none"),
        pytest.param(0.3, 0.5, False, id="asks"),
    ],
)
def test_the_bind_result_is_none_first_when_it_elects_under_none(
    measure: float, none: float, none_first: bool
):
    """A shape measuring the phrase above `decline` is elected though ``electq`` puts
    ``none`` above it, and the result says so; a part that asks elects nothing to doubt."""
    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers(
        {"m|0": measure, "m|1": measure - 0.1},
        elect={0: 0.6, 1: 0.4},
        choices={"electq": {"Shape 0": 0.6 - none, "Shape 1": 0.4, "none": none}},
    )

    result, _conf = bind_result(jev_menu(_request(shapes)), answers, stated_form=None)

    assert (result.output.ask is not None) is (measure < 0.5)
    assert result.none_first is none_first
