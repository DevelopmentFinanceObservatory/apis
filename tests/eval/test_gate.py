"""The paired regression rule (C3) as a table: one row per rule line, then the edge rows."""

from __future__ import annotations

import pytest

from search.eval.checks import Verdict
from search.eval.gate import Observation, Outcome, exit_code, is_new_case, judge

P, F, U, NA = Verdict.PASS, Verdict.FAIL, Verdict.UNVERIFIABLE, Verdict.NA

# One letter per sample: P pass, F fail, U unverifiable, N n/a, I a check whose graph or
# coverage read failed, T a run that got no answer after its retries, R a pass on a sample
# the app refused as a malformed request, X a head run that failed, scored n/a on a check
# other than `task_outcome`.
_SAMPLE = {
    "P": Observation(P),
    "F": Observation(F),
    "U": Observation(U),
    "N": Observation(NA),
    "I": Observation(U, infra="read"),
    "T": Observation(U, infra="run"),
    "R": Observation(P, request_rejected=True),
    "X": Observation(NA, run_failed=True),
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


def test_unstable_pass_is_reported():
    assert judge(check="measure", head=obs("PFP")).unstable is True
    assert judge(check="measure", head=obs("PUP")).unstable is False


@pytest.mark.parametrize(
    ("head", "fixed"),
    [(obs("PNP"), True), (obs("PFP"), False), (obs("FFF"), False), (obs("PPX"), False)],
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
        ([Outcome.PASS, Outcome.INFRA], 3),
        ([Outcome.INFRA, Outcome.NEW_CASE_FAILURE], 1),
    ],
)
def test_exit_code(outcomes, code):
    assert exit_code(outcomes) == code


def test_exit_code_refuses_a_verdict_still_waiting_for_a_run():
    with pytest.raises(ValueError, match="needs_base"):
        exit_code([Outcome.NEEDS_BASE])
