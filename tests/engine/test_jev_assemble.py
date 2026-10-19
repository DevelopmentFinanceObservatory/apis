"""Jev assembly: each threshold's cut-off, the behaviours a rewrite loses silently, and the
house policies on answer sets no dev item reaches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pytest

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.binders.base import BindRequest
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
    apply_policies: Callable[[PolicyPool], PolicyChoice] | None = None,
) -> BindRequest:
    frame = _frame()
    step = apply_policies or _HousePolicyStep(
        shapes=shapes, frame=frame, admitted_qualifiers=(), availability=(), code_index=get_index()
    )
    return BindRequest(
        variable="aid",
        shapes=list(shapes),
        dropped_qualifier_phrases=dropped,
        frame=frame,
        query_text="aid",
        apply_policies=step,
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


def _alternates(shapes: list[ShapeDraft], elect: Mapping[int, float]):
    def read(nouls: Mapping[str, float]) -> object:
        output, _conf = _assemble(_request(shapes), _answers({"m|0": 0.9, **nouls}, elect=elect))
        return output.also_plausible

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


def _gate(nouls: Mapping[str, float]) -> object:
    answers = _answers(nouls, choices={"q|form": {"per_person": 0.9, "none": 0.1}})
    return assemble_quantity(answers, []).stated_form


_PLAIN_AND_DIVIDED = [_shape(0), _shape(1, denominator="Count_Person")]
_VALUE_ADDED = _shape(0, population="EconomicProduction", measured="valueAdded")
_OUTPUT = _shape(1, population="EconomicActivity", measured="amount", qualifier="Nominal")
_NOMINAL_AND_REAL = [_shape(0), _shape(1, qualifier="RealValue")]

_CUTS = {
    "ask": _Cut(_ask, "m|0", 0.1, 0.11, True, False),
    "measures": _Cut(
        _alternates(_PLAIN_AND_DIVIDED, {0: 1.0}),
        "m|1",
        0.5,
        0.51,
        None,
        [1],
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
        _alternates([_shape(0), _shape(1, population="Place")], {0: 0.9, 1: 0.1}),
        "m|1",
        0.3,
        0.31,
        None,
        [1],
    ),
    "near_alt": _Cut(_near_alt, "elect|1", 0.04, 0.05, None, [1]),
    "measure_value": _Cut(
        _elected(_NOMINAL_AND_REAL, {0: 0.6, 1: 0.4}, {"m|0": 0.9, "m|1": 0.9}),
        "mv|RealValue",
        0.4,
        0.41,
        (0, []),
        (1, ["named: measurementQualifier"]),
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
    assert conf.policy_dropped == {1}
    assert output.also_plausible is None


def test_equal_elections_go_to_the_lower_menu_index():
    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={1: 0.5, 0: 0.5})

    output, conf = _assemble(_request(shapes), answers)

    assert output.chosen_shape_index == 0
    assert conf.pool is not None and conf.pool.measuring == (0, 1)


def test_the_house_policies_see_the_pool_with_the_election():
    seen: list[PolicyPool] = []

    def record(pool: PolicyPool) -> PolicyChoice:
        seen.append(pool)
        return PolicyChoice(survivor=1, fired=(), dropped=frozenset({0}), facts=())

    shapes = [_shape(0), _shape(1, population="Place")]
    answers = _answers({"m|0": 0.9, "m|1": 0.9}, elect={0: 0.6, 1: 0.4})

    output, _conf = _assemble(_request(shapes, apply_policies=record), answers)

    assert seen == [PolicyPool(measuring=(0, 1), election={0: 0.6, 1: 0.4}, rated=True)]
    assert output.chosen_shape_index == 1


def test_an_ask_runs_no_house_policy():
    calls: list[PolicyPool] = []

    def record(pool: PolicyPool) -> PolicyChoice:
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


def test_the_bind_result_carries_each_alternate_s_measure_score():
    shapes = [_shape(0), _shape(1, population="Place"), _shape(2, population="Region")]
    answers = _answers({"m|0": 0.9, "m|1": 0.7, "m|2": 0.4}, elect={0: 0.4, 1: 0.3, 2: 0.3})

    result, _conf = bind_result(jev_menu(_request(shapes)), answers, stated_form=None)

    assert result.binder == "jev"
    assert result.output.also_plausible == [1, 2]
    assert result.also_plausible_support == {1: pytest.approx(0.7), 2: pytest.approx(0.4)}
    assert result.policy_pool is not None
    assert result.policy_choice is not None and result.policy_choice.survivor == 0


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
