"""Admissibility: which readings of a part code lets the bind model elect, and why each
other one is refused.

Each rule judges one shape under the part's frame and names the refusal a part answers
with when that shape is the best it has. The rules run in a fixed order and a shape stops
at the first that refuses it, so a shape refused on a pinned part passed every other rule:

1. measure class: a shape whose population type names another measure class than the
   question (prevalence, incidence, mortality) measures something else
   (``variable_not_resolved``);
2. roles: a subject-only shape cannot hold both flow ends a question names, a
   role-bearing shape that declares no role on an axis the question needs cannot place it,
   a one-way series whose declared side is not an axis the question needs reads the other
   end of the flow, and a ratio whose declared base axis is not an axis the question needs
   divides by the other party's total, as DAC2a's ODA per capita of the recipient for
   "ODA per capita from Norway" (``role_incompatible``; `place_roles.lacks_query_axis`);
3. form: every normalisation form the question names must be honoured, a form a wording
   pattern names or, where none does, the one the quantity read's model states
   (`phase_rank_bind._read_stated_quantity`; `quantity.serves_every_form`;
   ``denominator_not_available``, pinned part `form`);
4. named measure: a name of measure nouns the question holds, by a label, an
   `also_called` name or a declared acronym of one (`CodeIndex.variable_aliases`, so "GNI
   per capita" names gross national income) (`phase_rank_bind._named_measure_nouns`),
   refuses a shape whose verdict on it is ``contradicts`` (`rule_four_verdicts`) with
   ``variable_not_resolved``: a noun of the shape's population type
   keys on a constraint property every member carries another value on, one neither an
   ancestor nor a descendant of the noun's in the code index, and every other noun that
   shares the name and the population type is contradicted too. "health expenditure"
   refuses the share of GDP spent on education and on the military. A shape the name
   holds, or whose denominator holds it as the base, is not contradicted, and "GDP", the
   name of three nouns, refuses no WEO series at market exchange rates though one of them
   keys on purchasing power parity. `also_called` names a measure for judgement, here and
   in the closest-match test, and never binds;
5. pinned codes: a code the question names on a property the shape's slots hold must be
   one the shape's family holds (``qualifier_not_bound``, pinned part `pinned_code`). A
   part naming a source binds first, and its code check runs once the bindings are final
   (`phase_rank_bind._unavailable_qualifiers`), so the refusal flags the elected identity.

The steps around the rules run in a fixed order too: measure and roles; the quantity read;
the form rule; named measures; pinned codes; the refusals written to the part's interpretations;
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
from search.engine.measure_verdict import ReadingEvidence, Verdict, measure_verdicts, shape_pairs
from search.engine.qualifier_match import NamedMeasure, QualifierMatch
from search.engine.quantity import DenominatorRead, serves_every_form
from search.engine.region.alternate_guard import measure_class, shape_measure_class
from search.engine.region.place_roles import lacks_query_axis, subject_shapes_refused
from search.engine.region.state import ResolveState, for_shape, shape_vocabulary
from search.engine.retrieval.bound_values import shape_constraint_values, shape_population_arcs
from search.engine.retrieval.facts import read_constraints
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
    spans: Sequence[NamedMeasure],
    own: Mapping[str, Sequence[Verdict]] | None = None,
) -> list[Interpretation]:
    """`interpretations` with each admissible one judged on the form the question names, on
    the names of measure nouns it holds, `spans`, and on the codes it names (rules 3 to 5).
    `matches` holds the question's codes read in each shape's vocabulary, and `own` rule 4's
    verdicts on each shape's own members where the phase already read them
    (`own_member_verdicts`). A part naming a source is judged on the form and the nouns
    alone."""
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
        own_verdicts = own.get(shape.shape_id) if own is not None else None
        if "contradicts" in rule_four_verdicts(shape, spans, state.denominator_reads, own_verdicts):
            return Interpretation(shape=shape, reason="variable_not_resolved")
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


def own_member_verdicts(
    shapes: Iterable[ShapeDraft],
    spans: Sequence[NamedMeasure],
    denominators: Mapping[str, DenominatorRead],
) -> dict[str, list[Verdict]]:
    """Rule 4's verdicts on each name in `spans` for each of `shapes`, by shape id, judged
    on its own members (`ShapeDraft.sv_arc_facts`). The phase reads the family of each held
    shape they contradict (`bound_values.read_held_families`) before `rule_four_verdicts`
    judges it again on every member."""
    if not spans:
        return {}
    return {
        shape.shape_id: measure_verdicts(
            spans, draft_evidence(shape, shape.sv_arc_facts or {}, denominators)
        )
        for shape in shapes
    }


def rule_four_verdicts(
    shape: ShapeDraft,
    spans: Sequence[NamedMeasure],
    denominators: Mapping[str, DenominatorRead],
    own: Sequence[Verdict] | None = None,
) -> list[Verdict]:
    """The verdict rule 4 reads on each name in `spans` for `shape`: `own`, its verdicts on
    its own members (`own_member_verdicts`, computed here when `None`), or, when they
    contradict a name and the shape stands for a held family, its verdicts on every member
    of that family (`shape_population_arcs`)."""
    if not spans:
        return []
    if own is None:
        own = own_member_verdicts([shape], spans, denominators)[shape.shape_id]
    if "contradicts" not in own or shape.held_family is None:
        return list(own)
    return measure_verdicts(
        spans, draft_evidence(shape, shape_population_arcs(shape), denominators)
    )


def draft_evidence(
    shape: ShapeDraft, arcs: Mapping[str, dict], denominators: Mapping[str, DenominatorRead]
) -> ReadingEvidence:
    """What `shape` states of its measure before bind: its five-tuple pairs
    (`measure_verdict.shape_pairs`), the constraint values its family holds
    (`shape_constraint_values`), the constraint properties of each member whose arcs `arcs`
    holds, its label as its noun, and the pairs and name the graph gives the StatVar it
    divides by, as `denominators` read them."""
    read = denominators.get(shape.meas_denom_dcid) if shape.meas_denom_dcid else None
    return ReadingEvidence(
        pairs=frozenset(
            [
                *shape_pairs(shape.pop_type_dcid, shape.meas_prop_dcid, shape.stat_type_dcid),
                *shape_constraint_values(shape),
            ]
        ),
        member_properties=tuple(frozenset(read_constraints(member)) for member in arcs.values()),
        noun=shape.label,
        denominator_pairs=frozenset(read.pairs) if read is not None else frozenset(),
        denominator_name=read.name if read is not None else None,
    )


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
