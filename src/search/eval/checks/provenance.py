"""``provenance``: the Data Commons and provenance dcid the answer's facets name.

Reads ``resolved.facets.provenance.data_commons.id`` and ``resolved.facets.provenance.ref``,
and for ``bound`` the state of ``definition.identity.facets.provenance``: a value binding is
bound, and an omitted facet is unbound. Each family names its provenance, which a status
block's own ``provenance`` replaces when it names an origin or dataset; a block naming only
``bound`` adds it to the family's. A family passes when its measure, scored as the ``measure`` check scores
it (``measure.match_measure``), holds on the answers that carry its provenance. Measure and
provenance therefore pass on one family and, on a candidates part, on the same candidates.
A family whose slots or dataset are unconfirmed reads unverifiable. When the measure check
fails every family, each answer is held to any family's provenance. A part with no spec,
such as an unsupported part, has no resolved facets, so the check is n/a there. A block
with a ``provenance`` and no ``measure`` needs one answer carrying it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from search.eval.cases import MeasureForm, Provenance, StatusExpect
from search.eval.checks.answers import Answer, answers, over_answers
from search.eval.checks.base import (
    NA,
    PASS,
    PartView,
    Result,
    ScoreContext,
    Verdict,
    all_of,
    any_of,
    fail,
    register_part,
    unconfirmed,
    unverifiable,
)
from search.eval.checks.measure import match_measure


def match_provenance(expected: Provenance, answer: Answer, ctx: ScoreContext) -> Result:
    """Whether ``answer``'s provenance facet is ``expected``'s origin and dataset."""
    if not answer.is_spec:
        return NA
    if expected.origin is not None:
        origin = answer.read("resolved.facets.provenance.data_commons.id", ctx)
        if origin != expected.origin:
            return fail(f"provenance origin: got {origin!r}, expected {expected.origin!r}")
    if expected.dataset is not None:
        if unconfirmed(expected.dataset):
            return unverifiable("provenance dataset is unconfirmed")
        ref = answer.read("resolved.facets.provenance.ref", ctx)
        if ref != expected.dataset:
            return fail(f"provenance dataset: got {ref!r}, expected {expected.dataset!r}")
    if expected.bound is not None:
        bound = answer.identity("facets.provenance.state", ctx) == "value"
        if bound != expected.bound:
            return fail(f"provenance bound: got {bound}, expected {expected.bound}")
    return PASS


def _expected(block: Provenance | None, family: Provenance | None) -> Provenance | None:
    """What a form's answers must carry: the block's provenance when it names an origin or
    dataset, else the family's, with the block's ``bound`` either way."""
    if block is None:
        return family
    if block.names_source or family is None:
        return block
    return family.model_copy(update={"bound": block.bound})


def _family(
    form: MeasureForm, expected: Provenance, part: PartView, found: list[Answer], ctx: ScoreContext
) -> Result:
    """The family's measure, scored as ``measure`` scores it, on the answers that carry the
    family's provenance."""
    carried = [(answer, match_provenance(expected, answer, ctx)) for answer in found]
    carriers = [(answer, r) for answer, r in carried if r.verdict != Verdict.FAIL]
    if not carriers:
        return any_of(r for _, r in carried)
    # One family's provenance reads the same on every carrier: pass, or unverifiable when
    # its dataset is unconfirmed.
    measured = match_measure(form, part, [answer for answer, _ in carriers], ctx)
    where = expected.dataset if isinstance(expected.dataset, str) else expected.origin
    return all_of([carriers[0][1], measured.at(f"on the answers from {where}")])


@register_part("provenance")
def provenance(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    """The answer carries the provenance its expectation names."""
    if block.measure is None:
        if (expected := block.provenance) is None:
            return NA
        return over_answers(
            part, ctx, lambda answer: match_provenance(expected, answer, ctx), reading=True
        )
    families = [
        (form, p)
        for form in block.measure.forms
        if (p := _expected(block.provenance, form.provenance)) is not None
    ]
    found = answers(part, ctx)
    if not families or not any(answer.is_spec for answer in found):
        return NA
    matched = [
        (form, p)
        for form, p in families
        if match_measure(form, part, found, ctx).verdict != Verdict.FAIL
    ]
    if matched:
        return any_of(_family(form, p, part, found, ctx) for form, p in matched)
    return over_answers(
        part,
        ctx,
        lambda answer: any_of(match_provenance(p, answer, ctx) for _, p in families),
        reading=True,
    )
