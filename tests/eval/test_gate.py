"""The paired regression rule (C3) as a table: one row per rule line, then the edge rows."""

from __future__ import annotations

import pytest

from search.eval.checks import Verdict
from search.eval.gate import Observation, Outcome, exit_code, is_new_case, judge

P, F, U, NA = Verdict.PASS, Verdict.FAIL, Verdict.UNVERIFIABLE, Verdict.NA

# One letter per sample: P pass, F fail, U unverifiable, N n/a, I a check whose graph or
# coverage read failed, T a run that got no answer after its retries, R a pass on a sample
# the app refused as a malformed request, X a head run that failed, scored n/a on a check
# other than `task_outcome`, ~ a check the expectation asserts that the sample's status
# left unscored, on the whole answer. On a case with two expected parts: 0 and 1 a sample that failed only that
# part, a a sample that reached and passed part 0 and never reached part 1. A base pass on
# `disposition` or `answer` whose sample's `measure`: C passes, W fails. The same on a
# case with two expected parts: w fails `measure` on part 1 only, v fails `measure` on part
# 0 and `entities` on part 1. D a pass and E a failure on a sample that stayed degraded
# after its retries.
_SAMPLE = {
    "P": Observation(P),
    "F": Observation(F),
    "U": Observation(U),
    "N": Observation(NA),
    "I": Observation(U, infra="read"),
    "T": Observation(U, infra="run"),
    "R": Observation(P, request_rejected=True),
    "X": Observation(NA, run_failed=True),
    "~": Observation(Verdict.UNREACHED),
    "D": Observation(P, degraded=True),
    "E": Observation(F, degraded=True),
    "0": Observation(F, parts={0: F}),
    "1": Observation(F, parts={1: F}),
    "a": Observation(Verdict.UNREACHED, parts={1: Verdict.UNREACHED}),
    "C": Observation(P, content={"measure": Observation(P)}),
    "W": Observation(P, content={"measure": Observation(F)}),
    "w": Observation(P, part_count=2, content={"measure": Observation(F, parts={1: F})}),
    "v": Observation(
        P,
        part_count=2,
        content={"measure": Observation(F, parts={0: F}), "entities": Observation(F, parts={1: F})},
    ),
}


def obs(letters: str) -> list[Observation]:
    return [_SAMPLE[letter] for letter in letters]


def row(
    name: str,
    outcome: Outcome,
    *,
    head: str,
    base: str | None = None,
    confirm: str | None = None,
    check: str = "measure",
    known: bool = False,
    task: bool = False,
    new: bool = False,
    blocks: bool = False,
):
    """``confirm`` is two letters: the head sample, then the base sample. ``task`` says
    the case's known failures hold ``task_outcome``."""
    return pytest.param(
        check,
        obs(head),
        None if base is None else obs(base),
        None if confirm is None else tuple(obs(confirm)),
        known,
        task,
        new,
        outcome,
        blocks,
        id=name,
    )


ROWS = [
    row("any head pass", Outcome.PASS, head="PFF"),
    row("known failure", Outcome.KNOWN_FAILURE, head="FFF", known=True),
    row("new case", Outcome.NEW_CASE_FAILURE, head="FUF", new=True, blocks=True),
    row("regression", Outcome.REGRESSION, head="FFF", base="PPP", confirm="FP", blocks=True),
    row("confirm head U", Outcome.REGRESSION, head="FFF", base="PPP", confirm="UP", blocks=True),
    row("confirm head passes", Outcome.UNSTABLE, head="FFF", base="PPP", confirm="PP"),
    row("confirm base fails", Outcome.UNSTABLE, head="FFF", base="PPP", confirm="FF"),
    row("pre-existing", Outcome.PRE_EXISTING, head="FFF", base="PFU"),
    row("unverifiable at base", Outcome.UNVERIFIABLE_AT_BASE, head="FFF", base="PUP", blocks=True),
    row("structural", Outcome.STRUCTURAL_FAILURE, head="FFF", check="structural", blocks=True),
    row("infra at head", Outcome.INFRA, head="IPP", check="groundedness"),
    row("infra at base", Outcome.INFRA, head="FFF", base="PIP"),
    row("infra in the confirm round", Outcome.INFRA, head="FFF", base="PPP", confirm="FI"),
    row("no head pass goes to base", Outcome.NEEDS_BASE, head="FFU"),
    row("base all pass awaits confirm", Outcome.NEEDS_CONFIRM, head="FFF", base="PNP"),
    # Edge rows.
    row("FFU at head", Outcome.PRE_EXISTING, head="FFU", base="FFF"),
    row("n/a counts as pass", Outcome.PASS, head="NFF"),
    row("n/a at base counts as pass", Outcome.NEEDS_CONFIRM, head="FFF", base="NNN"),
    # An unreached sample is no evidence: each side votes over the samples that reached
    # the check.
    row("unreached on every head sample", Outcome.PASS, head="~~~"),
    row("head votes over reached samples", Outcome.NEEDS_CONFIRM, head="~FF", base="PPP"),
    row(
        "reached head samples all fail",
        Outcome.REGRESSION,
        head="~FF",
        base="PPP",
        confirm="FP",
        blocks=True,
    ),
    row("unreached on every base sample", Outcome.PRE_EXISTING, head="FFF", base="~~~"),
    row("base votes over reached samples", Outcome.NEEDS_CONFIRM, head="FFF", base="PP~"),
    row(
        "reached base samples all pass",
        Outcome.REGRESSION,
        head="FFF",
        base="PP~",
        confirm="FP",
        blocks=True,
    ),
    row("unreached in the confirm base", Outcome.UNSTABLE, head="FFF", base="PPP", confirm="F~"),
    row("base passed a part a head sample failed", Outcome.NEEDS_CONFIRM, head="011", base="aaa"),
    # A head failure on the whole answer: a base sample that missed a part is unreached.
    row(
        "whole-answer head failure, base missed a part",
        Outcome.PRE_EXISTING,
        head="FFF",
        base="aaa",
    ),
    row(
        "whole-answer confirm head failure, confirm base missed a part",
        Outcome.UNSTABLE,
        head="FFF",
        base="PPP",
        confirm="Fa",
    ),
    row(
        "confirm base passed the part the confirm head failed",
        Outcome.REGRESSION,
        head="000",
        base="aaa",
        confirm="0a",
        blocks=True,
    ),
    row(
        "confirm base never reached the part the confirm head failed",
        Outcome.UNSTABLE,
        head="000",
        base="aaa",
        confirm="1a",
    ),
    # A base sample that failed another part passed the part the head failed.
    row("base failed only another part", Outcome.NEEDS_CONFIRM, head="000", base="111"),
    row(
        "confirm base failed only another part",
        Outcome.REGRESSION,
        head="000",
        base="111",
        confirm="01",
        blocks=True,
    ),
    # A degraded base sample is no baseline: each base round votes over the samples that
    # were not degraded, and one with none is base inconclusive.
    row("base degraded on every sample", Outcome.BASE_INCONCLUSIVE, head="FFF", base="DDD"),
    row("degraded base failures", Outcome.BASE_INCONCLUSIVE, head="FFF", base="EDE"),
    row("base judged on undegraded samples", Outcome.NEEDS_CONFIRM, head="FFF", base="PPE"),
    row("undegraded base failure", Outcome.PRE_EXISTING, head="FFF", base="DFD"),
    row(
        "regression on undegraded samples",
        Outcome.REGRESSION,
        head="FFF",
        base="PDE",
        confirm="FP",
        blocks=True,
    ),
    row("degraded confirm base", Outcome.BASE_INCONCLUSIVE, head="FFF", base="PPP", confirm="FD"),
    row("rejected at base is U", Outcome.UNVERIFIABLE_AT_BASE, head="FFF", base="PPR", blocks=True),
    row("rejected at head is scored", Outcome.PASS, head="RRR"),
    row("infra outranks a known failure", Outcome.INFRA, head="I", known=True),
    row("failed run, known task failure", Outcome.KNOWN_FAILURE, head="TTF", task=True),
    row("failed read, known task failure", Outcome.INFRA, head="TIF", task=True),
    row("base infra, known task failure", Outcome.INFRA, head="FFF", base="PIP", task=True),
    # A failed run is infra at base. At head the paired driver scores it as a failed
    # `task_outcome` sample, so it reaches the head rows as F.
    row("known failure, head run failed: not fixed", Outcome.KNOWN_FAILURE, head="PPX", known=True),
    row("head run failed is n/a elsewhere", Outcome.PASS, head="XXX"),
    row("base run failure", Outcome.INFRA, head="FFF", base="PTP", check="task_outcome"),
    row("confirm base run failure", Outcome.INFRA, head="FFF", base="PPP", confirm="FT"),
    row(
        "head run failure as F",
        Outcome.REGRESSION,
        head="FFF",
        base="PPP",
        confirm="FP",
        check="task_outcome",
        blocks=True,
    ),
    # The head side decides before base and confirm samples are read.
    row("head pass, base infra", Outcome.PASS, head="PPP", base="I"),
    row("known failure, base infra", Outcome.KNOWN_FAILURE, head="FFF", base="I", known=True),
    row("head pass, confirm infra", Outcome.PASS, head="PFF", base="PPP", confirm="II"),
    row(
        "confirm head F, base U",
        Outcome.UNVERIFIABLE_AT_BASE,
        head="FFF",
        base="PPP",
        confirm="FU",
        blocks=True,
    ),
    row(
        "confirm head U, base U",
        Outcome.UNVERIFIABLE_AT_BASE,
        head="FFF",
        base="PPP",
        confirm="UU",
        blocks=True,
    ),
    row(
        "answer",
        Outcome.REGRESSION,
        head="FFF",
        base="PPP",
        confirm="FP",
        check="answer",
        blocks=True,
    ),
    # A base sample that passes `disposition` or `answer` on the wrong series has no right
    # answer for the head to regress from.
    row(
        "wrong answer at base",
        Outcome.WRONG_ANSWER_AT_BASE,
        head="FFF",
        base="WWW",
        check="disposition",
    ),
    row(
        "right answer at base",
        Outcome.REGRESSION,
        head="FFF",
        base="CCC",
        confirm="FC",
        check="disposition",
        blocks=True,
    ),
    row(
        "a right base sample goes to confirm",
        Outcome.NEEDS_CONFIRM,
        head="FFF",
        base="CWW",
        check="disposition",
    ),
    # One right base sample is a right answer the head lost, whatever the confirm draws.
    row(
        "confirm base wrong",
        Outcome.REGRESSION,
        head="FFF",
        base="CWW",
        confirm="FW",
        check="disposition",
        blocks=True,
    ),
    row(
        "confirm base right",
        Outcome.REGRESSION,
        head="FFF",
        base="WWC",
        confirm="FC",
        check="disposition",
        blocks=True,
    ),
    row(
        "wrong answer, some base unreached",
        Outcome.WRONG_ANSWER_AT_BASE,
        head="FFF",
        base="W~W",
        check="disposition",
    ),
    # `answer` leaves a known-failure `measure` out, so base passes it on a series the case
    # knows is wrong; the sample's own `measure` verdict still counts.
    row(
        "wrong answer at base, answer",
        Outcome.WRONG_ANSWER_AT_BASE,
        head="FFF",
        base="WWW",
        check="answer",
    ),
    # A `disposition` failure concerns the whole answer, so base is wrong only when it is
    # wrong on every expected part.
    row(
        "disposition, base wrong on one of two parts",
        Outcome.REGRESSION,
        head="FFF",
        base="www",
        confirm="Fw",
        check="disposition",
        blocks=True,
    ),
    row(
        "disposition, base wrong on every part",
        Outcome.WRONG_ANSWER_AT_BASE,
        head="FFF",
        base="vvv",
        check="disposition",
    ),
    # `answer` names the parts the head failed; base must be wrong on each of them.
    row(
        "answer, wrong only on a part the head passed",
        Outcome.NEEDS_CONFIRM,
        head="000",
        base="www",
        check="answer",
    ),
    row(
        "answer, wrong on the part the head failed",
        Outcome.WRONG_ANSWER_AT_BASE,
        head="111",
        base="www",
        check="answer",
    ),
    row(
        "answer, right on one failed part",
        Outcome.NEEDS_CONFIRM,
        head="011",
        base="www",
        check="answer",
    ),
    row(
        "answer, wrong on every failed part",
        Outcome.WRONG_ANSWER_AT_BASE,
        head="011",
        base="vvv",
        check="answer",
    ),
    row(
        "other checks ignore content",
        Outcome.NEEDS_CONFIRM,
        head="FFF",
        base="WWW",
        check="warnings",
    ),
]


@pytest.mark.parametrize(
    ("check", "head", "base", "confirm", "known", "task", "new_case", "outcome", "blocks"), ROWS
)
def test_rule_table(check, head, base, confirm, known, task, new_case, outcome, blocks):
    judgement = judge(
        check=check,
        head=head,
        base=base,
        confirm=confirm,
        known_failure=known,
        known_task_failure=task,
        new_case=new_case,
    )
    assert judgement.outcome == outcome
    assert judgement.blocks is blocks


def test_an_observation_holds_its_parts_read_only_and_hashes():
    held = {0: F}
    observation = Observation(F, parts=held)
    held[1] = Verdict.UNREACHED
    assert observation.parts == {0: F}
    with pytest.raises(TypeError):
        observation.parts[1] = Verdict.UNREACHED
    assert hash(observation) == hash(Observation(F, parts={0: F}))
    assert Observation(P, parts={}) == Observation(P)
    assert Observation(NA, parts={0: Verdict.UNREACHED}) == Observation(NA)


def test_unstable_pass_is_reported():
    assert judge(check="measure", head=obs("PFP")).unstable is True
    assert judge(check="measure", head=obs("PUP")).unstable is False


@pytest.mark.parametrize(("head", "reached"), [("~~~", False), ("~P~", True), ("N", True)])
def test_a_pass_says_whether_a_head_sample_reached_the_check(head, reached):
    judgement = judge(check="measure", head=obs(head))
    assert (judgement.outcome, judgement.reached) == (Outcome.PASS, reached)


@pytest.mark.parametrize(
    ("head", "fixed"),
    [
        (obs("PNP"), True),
        (obs("PFP"), False),
        (obs("FFF"), False),
        (obs("PPX"), False),
        (obs("~~~"), False),
    ],
)
def test_known_failure_passing_on_every_head_sample_is_fixed(head, fixed):
    judgement = judge(check="measure", head=head, known_failure=True)
    assert judgement.outcome == Outcome.KNOWN_FAILURE
    assert judgement.fixed is fixed


def test_degraded_sample_is_reported():
    judgement = judge(check="measure", head=[Observation(P, degraded=True), *obs("PP")])
    assert judgement.outcome == Outcome.PASS
    assert judgement.degraded is True


@pytest.mark.parametrize(
    ("base_hashes", "new"),
    [
        ({"c": "h"}, False),
        ({"c": "other"}, True),
        ({}, True),
        (None, True),
    ],
    ids=["unchanged", "changed", "absent at base", "base corpus does not load"],
)
def test_new_case(base_hashes, new):
    assert is_new_case("c", "h", base_hashes) is new


@pytest.mark.parametrize(
    ("outcomes", "code"),
    [
        ([Outcome.PASS, Outcome.KNOWN_FAILURE, Outcome.PRE_EXISTING, Outcome.UNSTABLE], 0),
        ([Outcome.PASS, Outcome.REGRESSION], 1),
        ([Outcome.UNVERIFIABLE_AT_BASE], 1),
        ([Outcome.PASS, Outcome.WRONG_ANSWER_AT_BASE], 0),
        ([Outcome.PASS, Outcome.INFRA], 3),
        ([Outcome.INFRA, Outcome.NEW_CASE_FAILURE], 1),
        ([Outcome.PASS, Outcome.BASE_INCONCLUSIVE], 3),
        ([Outcome.BASE_INCONCLUSIVE, Outcome.REGRESSION], 1),
    ],
)
def test_exit_code(outcomes, code):
    assert exit_code(outcomes) == code


def test_exit_code_refuses_a_verdict_still_waiting_for_a_run():
    with pytest.raises(ValueError, match="needs_base"):
        exit_code([Outcome.NEEDS_BASE])
