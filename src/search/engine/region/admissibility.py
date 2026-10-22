"""Admissibility: which readings of a part code lets the bind model elect, and why each
other one is refused.

Each rule judges one shape under the part's frame and names the refusal a part answers
with when that shape is the best it has. The rules run in a fixed order and a shape stops
at the first that refuses it, so a shape refused on a pinned part passed every other rule:

1. measure class: a shape whose population type names another measure class than the
   question (prevalence, incidence, mortality) measures something else
   (``variable_not_resolved``);
2. roles: a subject-only shape cannot hold both flow ends a question names, and a
   role-bearing shape that declares no role on an axis the question needs cannot place it
   (``role_incompatible``);
3. form: every normalisation form the question names must be honoured, a form a wording
   pattern names or, where none does, the one the quantity read's model states
   (`phase_rank_bind._read_stated_quantity`; `quantity.serves_every_form`;
   ``denominator_not_available``, pinned part `form`);
4. pinned codes: a code the question names on a property the shape's slots hold must be
   one the shape's family holds (``qualifier_not_bound``, pinned part `pinned_code`). A
   part naming a source binds first, and its code check runs once the bindings are final
   (`phase_rank_bind._unavailable_qualifiers`), so the refusal flags the elected identity.

The steps around the rules run in a fixed order too: measure and roles; the quantity read;
the form rule; pinned codes; the refusals written to the part's interpretations;
open-axis preference, family election and collapse; seating; the availability read; bind.
The form rule runs before seating, so a shape it refuses holds no seat, and a menu it
empties ends in a refusal because the refusals are already recorded.

Dataset candidacy under the part's source scope runs before these, and the score floor
bounds the menu before them; five-tuple family election, the axes rules and the
measure-family collapse narrow the admissible shapes afterwards
(`phase_rank_bind._phase_prune_and_rank`). The denominator the question names is read by the
quantity read with the form, and the elected shape is held to it at materialise
(`quantity.check_denominator`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from code_index.shared import get_measure_index
from search.engine.interpretation import Interpretation
from search.engine.qualifier_match import QualifierMatch
from search.engine.quantity import serves_every_form
from search.engine.region.alternate_guard import measure_class, shape_measure_class
from search.engine.region.place_roles import lacks_query_axis, subject_shapes_refused
from search.engine.region.state import ResolveState, for_shape, shape_vocabulary
from search.engine.shape import ShapeDraft, is_role_bearing
from search.models import GraphRef, NoDataReason, ResolvedQualifier


def judge_measure_and_roles(
    state: ResolveState, shapes: Sequence[ShapeDraft]
) -> list[Interpretation]:
    """Each of `shapes` judged on its measure class and its roles (rules 1 and 2)."""
    named_class = measure_class(state.variable)
    both_ends = subject_shapes_refused(state)

    def reason(shape: ShapeDraft) -> NoDataReason | None:
        if named_class is not None and shape_measure_class(shape) not in (None, named_class):
            return "variable_not_resolved"
        if (both_ends and not is_role_bearing(shape)) or lacks_query_axis(state, shape):
            return "role_incompatible"
        return None

    return [Interpretation(shape=shape, reason=reason(shape)) for shape in shapes]


def judge_pinned_parts(
    state: ResolveState,
    interpretations: Iterable[Interpretation],
    matches: Mapping[frozenset[str], tuple[QualifierMatch, ...]],
) -> list[Interpretation]:
    """`interpretations` with each admissible one judged on the form the question names and
    on the codes it names (rules 3 and 4). `matches` holds the question's codes read in
    each shape's vocabulary. A part naming a source is judged on the form alone."""
    measure_axes = frozenset(axis.axis for axis in get_measure_index().axes)
    names_source = state.scope is not None and state.scope.named

    def judged(interpretation: Interpretation) -> Interpretation:
        shape = interpretation.shape
        if not interpretation.admissible:
            return interpretation
        if not serves_every_form(state.quantity_requirement, shape, state.denominator_kinds):
            return Interpretation(
                shape=shape, reason="denominator_not_available", pinned_failure="form"
            )
        if names_source:
            return interpretation
        unheld = _unheld_pinned_codes(
            shape, for_shape(matches, shape), state.held_codes_of(shape), measure_axes
        )
        if unheld:
            return Interpretation(
                shape=shape,
                reason="qualifier_not_bound",
                pinned_failure="pinned_code",
                pinned_codes=unheld,
            )
        return interpretation

    return [judged(interpretation) for interpretation in interpretations]


def _unheld_pinned_codes(
    shape: ShapeDraft,
    matches: Sequence[QualifierMatch],
    holds: Callable[[QualifierMatch], bool],
    measure_axes: frozenset[str],
) -> tuple[QualifierMatch, ...]:
    """The codes `matches` name on a property `shape`'s slots hold that its family holds
    neither itself nor through a descendant (`holds`, `bound_values.held_codes`): a code the
    question names is required of every shape whose slots hold its property, as a
    `CodeRule` requirement judges an answer after the fact (`region.requirements.judge`).

    A shape the index carries holds the codes of the rows its part may answer from, and any
    other shape the codes of its recalled members, the members bind and materialise read.
    A measure axis's value is the measure election's (`bound_values.elect_measure_family`)
    and no pinned code."""
    vocabulary = shape_vocabulary(shape) - measure_axes
    return tuple(
        dict.fromkeys(
            match for match in matches if match.code_system in vocabulary and not holds(match)
        )
    )


def pinned_code_records(interpretation: Interpretation) -> tuple[ResolvedQualifier, ...]:
    """The codes a reading refused on a pinned code cannot bind, each as the unbound term a
    ``qualifier_not_bound`` refusal names, one per phrase."""
    by_phrase: dict[str, list[QualifierMatch]] = {}
    for match in interpretation.pinned_codes:
        by_phrase.setdefault(match.phrase, []).append(match)
    return tuple(
        ResolvedQualifier(
            text=phrase_matches[0].matched_text,
            named=[GraphRef(ref=m.code_dcid, label=m.label) for m in phrase_matches],
            slot=phrase_matches[0].code_system,
            disposition="unbound",
        )
        for phrase_matches in by_phrase.values()
    )
