"""The paired regression rule: one gate verdict per (case, check) from per-sample verdicts.

``judge`` reads a check's verdicts on the head samples and, when the rule needs them, the
base samples and one confirm round (one head sample, then one base sample). It returns
``NEEDS_BASE`` or ``NEEDS_CONFIRM`` while the verdict depends on a run that has not
happened, and the driver calls it again once that run is scored. The head samples decide
first, so a base or confirm run taken for one check cannot change another check's verdict.
The rows, in the order they apply:

- a head sample whose run or graph or coverage read failed (``Observation.infra``): infra,
  except that a failed run (``"run"``) on a case whose ``known_failure.checks`` holds
  ``task_outcome`` is a known failure, so a case that times out on every run does not stop
  every change; a failed graph or coverage read (``"read"``) stays infra;
- the check is in the case's ``known_failure.checks``: known failure, reported fixed when
  every head sample passes and none is a failed run (``Observation.run_failed``) or
  unreached;
- every head sample is unreached, or a reached head sample passes: pass, reported unstable
  when another reached head sample fails. A pass no reached head sample supports
  (``Judgement.reached`` false) can turn into a failure on more samples;
- no head sample passes and the case is new or changed: new-case failure, blocking;
- no head pass on ``structural``: structural failure, blocking;
- a base sample is infra: infra;
- every base sample is unreached, or a reached base sample fails: pre-existing;
- a reached base sample is unverifiable and none fails: unverifiable at base, blocking
  until an override releases it;
- every reached base sample passes ``disposition`` or ``answer`` on a wrong answer (see
  below): wrong answer at base, which does not block;
- every reached base sample passes: the confirm round decides. Infra in it is infra as
  above. A confirm head sample that fails or is unverifiable is a regression when the
  confirm base sample passes and unverifiable at base when it is unverifiable; anything
  else, an unreached sample on either side included, is unstable. A confirm base sample
  that passes on a wrong answer makes it wrong answer at base.

``scripts/eval_paired.py`` hands in a head sample whose run failed as a failed
``task_outcome`` sample with no infra flag, so a head-side ``"run"`` failure in the first row
comes only from the nightly (``search.eval.sync``), which has no base to compare.

n/a counts as a pass. An unreached sample (``Verdict.UNREACHED``: the expectation asserts
the check, but the sample's status or kind is one the case does not accept, or an expected
part is missing, so the check never ran) is no evidence about the check on either side, and
each side votes over the samples that reached it. A check unreached on every head sample is
``disposition``'s failure, which that check reports, so it passes here. A check unreached on
every base sample has no base pass to regress from, so a head that fixes the status and
then fails the check scores pre-existing.

On a case with ``expect.parts`` a base sample that did not pass is read against the parts
the head failed on. It fails, or is unverifiable, when it does so on one of them; it is
unreached when it is unreached on every one of them; otherwise it passed the parts the head
failed, and counts as a pass. When a head failure concerns the whole answer (its ``parts``
is None), a base sample that did not pass counts as its verdict on the whole answer. The
initial base samples are read against every part any reached initial head sample failed on,
so a base that passed a part some head sample failed goes to the confirm round. The
confirm base sample is read against the confirm head sample's failed parts alone, so the
confirm round blocks with a regression only when the confirm base passes a part the confirm
head failed.

A base sample's pass on ``disposition`` or ``answer`` (``WRONG_ANSWER_CHECKS``) is a pass on a
wrong answer when the same sample fails a check that says which series or places answer
(``CONTENT_CHECKS``: ``measure``, ``entities``, ``provenance``), read over the parts the
head failed as above. The driver hands those observations in as ``Observation.content``.
A head that refuses, or answers in another status, where base answered the wrong series
then has no right answer to regress from. The sample's own content verdicts decide, so a
content check in the case's ``known_failure.checks`` counts as it scored on that sample:
``answer`` leaves known failures out, and a base sample passes it on a series the case
already knows is wrong.

A base sample the app refused as a malformed request (``request_rejected``) is
unverifiable, since the head runner's request may carry fields the base app does not know.
At head the checks score the refusal as the problem document it is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from search.eval.checks import ANSWER, Verdict

type Infra = Literal["run", "read"]


class Outcome(StrEnum):
    PASS = "pass"
    KNOWN_FAILURE = "known_failure"
    NEW_CASE_FAILURE = "new_case_failure"
    STRUCTURAL_FAILURE = "structural_failure"
    REGRESSION = "regression"
    UNSTABLE = "unstable"
    PRE_EXISTING = "pre_existing"
    UNVERIFIABLE_AT_BASE = "unverifiable_at_base"
    WRONG_ANSWER_AT_BASE = "wrong_answer_at_base"
    INFRA = "infra"
    NEEDS_BASE = "needs_base"
    NEEDS_CONFIRM = "needs_confirm"


BLOCKING = frozenset(
    {
        Outcome.NEW_CASE_FAILURE,
        Outcome.STRUCTURAL_FAILURE,
        Outcome.REGRESSION,
        Outcome.UNVERIFIABLE_AT_BASE,
    }
)
"""Outcomes that fail the gate with exit 1. The ``eval-override`` label releases them."""

WRONG_ANSWER_CHECKS = frozenset({"disposition", ANSWER})
"""Checks a base sample can pass on a wrong answer: the right status on the wrong series."""

CONTENT_CHECKS = frozenset({"measure", "entities", "provenance"})
"""Checks that say which series and places answer: the five-tuple and StatVars, the bound
entities, and the dataset the facets name."""

PENDING = frozenset({Outcome.NEEDS_BASE, Outcome.NEEDS_CONFIRM})
"""Outcomes waiting on a base run or a confirm round."""


@dataclass(frozen=True)
class Observation:
    """One sample's result for one check."""

    verdict: Verdict
    infra: Infra | None = None
    """``"run"`` when the runner got no answer after its retries, ``"read"`` when the
    check could not read the graph or coverage it scores against."""
    request_rejected: bool = False
    degraded: bool = False
    run_failed: bool = False
    """A head sample whose run failed after its retries, scored rather than infra: it fails
    ``task_outcome`` and is n/a elsewhere, and it never counts towards fixed."""
    parts: Mapping[int, Verdict] | None = field(default=None, hash=False)
    """The check's ``Result.parts``: each expected part that did not pass, by position, as a
    read-only copy, or None when the verdict concerns the whole answer, and on a pass or n/a.
    Hashing leaves it out, so equal observations still hash alike."""
    content: Mapping[str, Observation] = field(default_factory=dict, hash=False)
    """The same sample's observations on ``CONTENT_CHECKS``, by check, on a base sample of a
    check in ``WRONG_ANSWER_CHECKS``, as a read-only copy. Hashing leaves it out."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", MappingProxyType(dict(self.content)))
        if self.parts is None:
            return
        held = None if self.verdict in (Verdict.PASS, Verdict.NA) else dict(self.parts)
        object.__setattr__(self, "parts", None if held is None else MappingProxyType(held))


@dataclass(frozen=True)
class Judgement:
    outcome: Outcome
    unstable: bool = False
    """A pass that failed on some head sample."""
    fixed: bool = False
    """A known failure that passed on every head sample."""
    degraded: bool = False
    """Some sample behind the verdict stayed degraded after its retries."""
    reached: bool = True
    """False on a pass when no head sample reached the check, so more head samples could
    still fail it."""

    @property
    def blocks(self) -> bool:
        return self.outcome in BLOCKING


def judge(
    *,
    check: str,
    head: Sequence[Observation],
    base: Sequence[Observation] | None = None,
    confirm: tuple[Observation, Observation] | None = None,
    known_failure: bool = False,
    known_task_failure: bool = False,
    new_case: bool = False,
) -> Judgement:
    """The gate verdict for one check of one case.

    ``base`` is None until the base side ran, and ``confirm`` is the (head, base) pair of
    the confirm round, None until it ran. ``known_failure`` says the check is in the case's
    ``known_failure.checks``, ``known_task_failure`` that those checks hold
    ``task_outcome``, and ``new_case`` that the case is new or changed against base.
    """
    read: list[Observation] = list(head)

    def verdict(
        outcome: Outcome, *, unstable: bool = False, fixed: bool = False, reached: bool = True
    ) -> Judgement:
        degraded = any(o.degraded for o in read)
        return Judgement(
            outcome, unstable=unstable, fixed=fixed, degraded=degraded, reached=reached
        )

    if (head_infra := _head_infra(head, known_task_failure=known_task_failure)) is not None:
        return verdict(head_infra)
    head_v = [_effective(o, "head") for o in head]
    if known_failure:
        fixed = all(v == Verdict.PASS for v in head_v) and not any(o.run_failed for o in head)
        return verdict(Outcome.KNOWN_FAILURE, fixed=fixed)
    reached = [v for v in head_v if v != Verdict.UNREACHED]
    if not reached or Verdict.PASS in reached:
        return verdict(Outcome.PASS, unstable=Verdict.FAIL in reached, reached=bool(reached))
    if new_case:
        return verdict(Outcome.NEW_CASE_FAILURE)
    if check == "structural":
        return verdict(Outcome.STRUCTURAL_FAILURE)
    if base is None:
        return verdict(Outcome.NEEDS_BASE)
    read += base
    if any(o.infra for o in base):
        return verdict(Outcome.INFRA)
    failed = _failed_parts(head)
    reached_base = [o for o in base if _effective(o, "base", failed=failed) != Verdict.UNREACHED]
    base_v = [_effective(o, "base", failed=failed) for o in reached_base]
    if not base_v or Verdict.FAIL in base_v:
        return verdict(Outcome.PRE_EXISTING)
    if Verdict.UNVERIFIABLE in base_v:
        return verdict(Outcome.UNVERIFIABLE_AT_BASE)
    if all(_wrong_answer(o, check=check, failed=failed) for o in reached_base):
        return verdict(Outcome.WRONG_ANSWER_AT_BASE)
    if confirm is None:
        return verdict(Outcome.NEEDS_CONFIRM)
    read += confirm
    if (head_infra := _head_infra(confirm[:1], known_task_failure=known_task_failure)) is not None:
        return verdict(head_infra)
    if confirm[1].infra:
        return verdict(Outcome.INFRA)
    if _effective(confirm[0], "head") in (Verdict.PASS, Verdict.UNREACHED):
        return verdict(Outcome.UNSTABLE)
    confirm_failed = _failed_parts(confirm[:1])
    match _effective(confirm[1], "base", failed=confirm_failed):
        case Verdict.PASS if _wrong_answer(confirm[1], check=check, failed=confirm_failed):
            return verdict(Outcome.WRONG_ANSWER_AT_BASE)
        case Verdict.PASS:
            return verdict(Outcome.REGRESSION)
        case Verdict.UNVERIFIABLE:
            return verdict(Outcome.UNVERIFIABLE_AT_BASE)
    return verdict(Outcome.UNSTABLE)


def _head_infra(head: Sequence[Observation], *, known_task_failure: bool) -> Outcome | None:
    """Infra, a known failure for a known task failure's failed runs, or None."""
    infra = {o.infra for o in head} - {None}
    if not infra:
        return None
    return Outcome.KNOWN_FAILURE if known_task_failure and infra == {"run"} else Outcome.INFRA


def _effective(
    observation: Observation,
    side: Literal["head", "base"],
    *,
    failed: AbstractSet[int] | None = None,
) -> Verdict:
    """The sample's verdict on the pass, fail, unverifiable, unreached scale the rule reads.

    n/a is a pass, and a base sample the app refused as a malformed request is
    unverifiable.
    ``failed`` holds the parts the head failed on (``_failed_parts``), None when a head
    failure concerns the whole answer. A sample with parts is read over ``failed`` alone:
    it fails, or is unverifiable, when it does so on one of them, is unreached when it is
    unreached on every one of them, and passed them otherwise. With ``failed`` None it
    reads as its verdict on the whole answer.
    """
    if observation.request_rejected and side == "base":
        return Verdict.UNVERIFIABLE
    if observation.verdict == Verdict.NA:
        return Verdict.PASS
    if observation.parts is None or not failed:
        return observation.verdict
    on_failed = {observation.parts.get(i, Verdict.PASS) for i in failed}
    for verdict in (Verdict.FAIL, Verdict.UNVERIFIABLE):
        if verdict in on_failed:
            return verdict
    return Verdict.UNREACHED if on_failed == {Verdict.UNREACHED} else Verdict.PASS


def _wrong_answer(observation: Observation, *, check: str, failed: AbstractSet[int] | None) -> bool:
    """Whether a base sample passes ``check`` on a wrong answer: the check is in
    ``WRONG_ANSWER_CHECKS`` and the sample fails a content check over ``failed``."""
    return check in WRONG_ANSWER_CHECKS and any(
        _effective(o, "base", failed=failed) == Verdict.FAIL for o in observation.content.values()
    )


def _failed_parts(head: Iterable[Observation]) -> frozenset[int] | None:
    """The parts the reached head samples that did not pass failed on, or None when one of
    them concerns the whole answer."""
    failed: set[int] = set()
    for o in head:
        if _effective(o, "head") in (Verdict.PASS, Verdict.UNREACHED):
            continue
        if o.parts is None:
            return None
        failed |= {i for i, v in o.parts.items() if v in (Verdict.FAIL, Verdict.UNVERIFIABLE)}
    return frozenset(failed)


def is_new_case(case_id: str, head_hash: str, base_hashes: Mapping[str, str] | None) -> bool:
    """Whether a case is new or changed against the base corpus.

    ``base_hashes`` maps each base case id to its expectation hash. It is None when the
    base corpus does not load under the head case model, which makes every case new.
    """
    return base_hashes is None or base_hashes.get(case_id) != head_hash


def exit_code(outcomes: Iterable[Outcome]) -> int:
    """1 when any verdict blocks, else 3 when any is infra, else 0.

    A blocking verdict is a finding whatever else could not be judged, so it outranks
    infra. Raises ``ValueError`` on a verdict still waiting for a run.
    """
    seen = set(outcomes)
    if waiting := seen & PENDING:
        raise ValueError(f"unresolved verdicts: {sorted(waiting)}")
    if seen & BLOCKING:
        return 1
    return 3 if Outcome.INFRA in seen else 0
