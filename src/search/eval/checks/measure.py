"""``measure``: the answer's five-tuple, slot bindings and, where pinned, StatVars.

A measure matches when one of its forms (one per family, or the inline shape) matches on
every shape key it gives, every bound slot (a value, or a list for a set binding), every
open slot, and the exact StatVar set when it pins one. On a candidates part, one candidate
must match the shape and slots, and each pinned StatVar must be served by its own
candidate; extra candidates pass and their count is not asserted. Slots read
``definition.identity.slots``, whose bindings are dense: a slot the identity omits is not
bound. Slots the form does not name are not asserted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import BoundValue, Gl, MeasureForm, StatusExpect
from search.eval.checks.answers import Answer, answers
from search.eval.checks.base import (
    NA,
    PASS,
    READ_ERRORS,
    PartView,
    Result,
    ScoreContext,
    all_of,
    any_of,
    fail,
    infra_error,
    pairs_one_to_one,
    read_path,
    register_part,
    served_stat_vars,
    unconfirmed,
    unverifiable,
)


def _shape_value(answer: Answer, key: str, ctx: ScoreContext) -> object:
    """The identity's shape member: a dcid, or None when absent.

    A denominator whose target is a ProvisionalNode (``PerCapita``) is written into the
    identity as ``{unresolved: true}`` with no ref; its dcid, when the target carries one,
    is ``resolved.shape.measurement_denominator.ref``. Without that dcid, as on an
    unsupported part, the marker itself is returned and equals no expected value.
    """
    value = answer.identity(f"shape.{key}", ctx)
    if key != "measurement_denominator" or not isinstance(value, Mapping):
        return value
    if "ref" in value:
        return value["ref"]
    if answer.is_spec:
        dcid = answer.read("resolved.shape.measurement_denominator.ref", ctx)
        if dcid is not None:
            return dcid
    return value


def _shape(form: MeasureForm, answer: Answer, ctx: ScoreContext) -> Result:
    if isinstance(form.shape, Gl):
        return unverifiable("shape is unconfirmed")
    for key, expected in form.shape.items():
        if unconfirmed(expected):
            return unverifiable(f"shape {key} is unconfirmed")
        got = _shape_value(answer, key, ctx)
        if got != expected:
            return fail(f"shape {key}: got {got!r}, expected {expected!r}")
    return PASS


def _bindings(answer: Answer, ctx: ScoreContext) -> dict[str, frozenset[str] | None]:
    """The dcids each identity slot binds by property: one for a value or exact binding,
    several for a set, None for an unbound slot."""
    at = f"{answer.at}.definition.identity.slots[]"
    bound: dict[str, frozenset[str] | None] = {}
    for slot in answer.identity("slots", ctx) or []:
        refs = None
        match read_path(slot, "binding.state", ctx, at=at):
            case "value" | "exact":
                refs = frozenset([read_path(slot, "binding.ref", ctx, at=at)])
            case "set":
                refs = frozenset(read_path(slot, "binding.refs", ctx, at=at) or ())
        bound[read_path(slot, "property", ctx, at=at)] = refs
    return bound


def _bound(
    slot: str, expected: BoundValue, bindings: Mapping[str, frozenset[str] | None]
) -> Result:
    if unconfirmed(expected):
        return unverifiable(f"slot {slot} is unconfirmed")
    want = frozenset(map(str, expected if isinstance(expected, list) else [expected]))
    got = bindings.get(slot)
    if got != want:
        shown = sorted(got) if got is not None else "unbound"
        return fail(f"slot {slot}: got {shown}, expected {sorted(want)}")
    return PASS


def _stat_vars(form: MeasureForm, answer: Answer, ctx: ScoreContext) -> Result:
    expected = form.stat_vars
    if expected is None:
        return NA
    if unconfirmed(expected):
        return unverifiable("stat_vars are unconfirmed")
    if not answer.is_spec:
        return fail("stat_vars: an unsupported part resolves no StatVar")
    try:
        served = served_stat_vars(answer, ctx)
    except READ_ERRORS as exc:
        return infra_error(f"stat_vars: the members read failed: {exc}")
    if served is None:
        return unverifiable("stat_vars: no members door to name the served StatVars")
    got = sorted(served)
    want = sorted(map(str, expected))
    if got != want:
        return fail(f"stat_vars: got {got}, expected {want}")
    return PASS


def _stat_vars_paired(form: MeasureForm, found: list[Answer], ctx: ScoreContext) -> Result:
    """Each expected StatVar is served by its own candidate; extra candidates pass."""
    expected = form.stat_vars
    if expected is None:
        return NA
    if unconfirmed(expected):
        return unverifiable("stat_vars are unconfirmed")
    want = list(map(str, expected))
    try:
        served = [served_stat_vars(answer, ctx) for answer in found]
    except READ_ERRORS as exc:
        return infra_error(f"stat_vars: the members read failed: {exc}")
    if any(stat_vars is None for stat_vars in served):
        return unverifiable("stat_vars: no members door to name the served StatVars")
    held = [set(stat_vars or ()) for stat_vars in served]
    if pairs_one_to_one(len(want), len(held), lambda sv, spec: want[sv] in held[spec]):
        return PASS
    unserved = [sv for sv in want if not any(sv in refs for refs in held)]
    detail = f"no candidate serves {unserved}" if unserved else "two StatVars share one candidate"
    return fail(f"stat_vars: {detail} among {len(found)} candidates")


def _open(slot: object, bindings: Mapping[str, frozenset[str] | None]) -> Result:
    """An open slot is in the identity, unbound. An identity that leaves the slot out
    serves no member asserting it, which excludes the slot."""
    if unconfirmed(slot):
        return unverifiable(f"open slot {slot} is unconfirmed")
    if str(slot) not in bindings:
        return fail(f"slot {slot} absent, expected open")
    refs = bindings[str(slot)]
    return PASS if refs is None else fail(f"slot {slot}: got {sorted(refs)}, expected unbound")


def match_form(
    form: MeasureForm, answer: Answer, ctx: ScoreContext, *, stat_vars: bool = True
) -> Result:
    """Whether ``answer`` is the measure ``form`` describes; ``stat_vars=False`` leaves
    the StatVar pin to the caller."""
    if form.unconfirmed:
        return unverifiable("bound slots are unconfirmed")
    bindings = _bindings(answer, ctx)
    return all_of(
        [
            _shape(form, answer, ctx),
            *(_bound(slot, value, bindings) for slot, value in form.bound.items()),
            *(_open(slot, bindings) for slot in form.open),
            _stat_vars(form, answer, ctx) if stat_vars else NA,
        ]
    )


def _match_candidates(form: MeasureForm, found: list[Answer], ctx: ScoreContext) -> Result:
    """One candidate matches the form's shape and slots, and its StatVars pair one to one
    with distinct candidates."""
    reading = any_of(
        match_form(form, answer, ctx, stat_vars=False).at(f"candidate {i}")
        for i, answer in enumerate(found)
    )
    return all_of([reading, _stat_vars_paired(form, found, ctx)])


def match_measure(
    form: MeasureForm, part: PartView, found: list[Answer], ctx: ScoreContext
) -> Result:
    """``form``'s verdict on ``found``, the answers ``part`` carries, as ``measure`` scores it."""
    if part.status == "candidates":
        return _match_candidates(form, found, ctx)
    return any_of(match_form(form, answer, ctx) for answer in found)


@register_part("measure")
def measure(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """The answer matches one of the expected measure's forms."""
    expected = block.measure
    if expected is None:
        return NA
    found = answers(part, ctx)
    return any_of(match_measure(form, part, found, ctx) for form in expected.forms)
