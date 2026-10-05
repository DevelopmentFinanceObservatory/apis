"""Repair draft bindings when taxonomy shape or availability hides a requested value.

``repair_draft_bindings`` removes only an unasked ``kind="value"`` binding whose
``sole_option`` evidence matches a verified singleton in ``slot_taxonomy``. This repairs
one axis recorded under multiple properties, such as ``ageGroupClassification=Infant`` and
``age=Years0``. Mentioned values stay bound. Shape-level co-occurrence looks the same for a
duplicate spelling and independent axes, so it risks dropping a real constraint.

The normal fail-open resolver check must reject the original draft. After eligible drops,
at least one bound value must remain and the remainder must pass strict realizability. This
prevents an empty constraint set from selecting an arbitrary member and prevents a repaired
answer from contradicting a retained value. Otherwise the original draft stays unchanged.

``restore_starved_axis_binding`` handles empty taxonomies caused by the place-availability
floor. It restores a value only for a place-available shape, an open empty slot, a property
with disagreeing unscoped member values, and exactly one whole-word label match outside a
negation or exclusion. The full binding set must remain realizable. On success, the binding
and taxonomy use the matched value and the complete unscoped value list.

Both repairs follow binding order and arc-fact insertion order.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from code_index.query import CodeIndex
from search.engine.bind import SlotBindingDraft
from search.engine.qualifier_match import QualifierMatch, QualifierTie, names_plain_name
from search.engine.retrieval.bound_values import (
    bound_realizable_members,
    bound_values_from_bindings,
)
from search.engine.retrieval.facts import read_constraints
from search.engine.retrieval.representative import _shape_disagreement_props
from search.engine.retrieval.standard import _is_degenerate_binding
from search.engine.shape import ShapeDraft


def repair_draft_bindings(
    bindings: Sequence[SlotBindingDraft],
    members: Sequence[str],
    sv_arc_facts: dict[str, dict],
    slot_taxonomy: dict[str, list[str]],
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Drop verified sole-option bindings when they block resolution.

    The repair applies only when the original fails the normal resolver check. Eligible
    drops must leave a bound value and pass strict realizability.

    Returns the listified input and ``[]`` when unchanged. Otherwise the repaired list keeps
    dropped slots as ``unbound`` and returns their original drafts separately. Iteration
    follows binding order.
    """
    bindings = list(bindings)
    bound_values = bound_values_from_bindings(bindings)
    if bound_realizable_members(members, sv_arc_facts, bound_values):
        return bindings, []

    sole_option_indices = []
    for i, b in enumerate(bindings):
        if not (b.property_dcid and b.kind == "value" and b.evidence == "sole_option"):
            continue
        if len(b.value_dcids) != 1:
            # A multi-value draft cannot itself be a verified singleton.
            continue
        taxonomy_values = slot_taxonomy.get(f"{b.axis}:{b.property_dcid}")
        if taxonomy_values is None or len(taxonomy_values) != 1:
            continue
        if taxonomy_values[0] != b.value_dcids[0]:
            continue
        sole_option_indices.append(i)
    if not sole_option_indices:
        return bindings, []

    repaired = list(bindings)
    dropped: list[SlotBindingDraft] = []
    for i in sole_option_indices:
        dropped.append(repaired[i])
        repaired[i] = repaired[i].model_copy(update={"kind": "unbound", "value_dcids": []})

    repaired_bound_values = bound_values_from_bindings(repaired)
    if not repaired_bound_values:
        # An empty remainder would realise against any member with data.
        return bindings, []

    if bound_realizable_members(members, sv_arc_facts, repaired_bound_values, strict=True):
        return repaired, dropped
    return bindings, []


def detect_degenerate_bindings(
    bindings: Sequence[SlotBindingDraft],
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Find ``value`` or ``set`` bindings with no values.

    These schema-valid drafts would become open probes if downgraded to ``unbound``. The
    pipeline and resolver reject them before materialisation. Non-empty ``value``/``set``
    mismatches and all ``unbound``/``absent`` bindings pass through.

    Returns the unchanged, listified bindings and degenerate entries in input order.
    """
    bindings = list(bindings)
    degenerate = [b for b in bindings if _is_degenerate_binding(b)]
    return bindings, degenerate


def _label_named_in_text(label: str, text: str) -> bool:
    """Match case-insensitive whole words, keeping ``Male`` separate from ``female``."""
    pattern = r"\b" + re.escape(label) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


_NEGATION_WINDOW_WORDS = 3
_NEGATORS = ("not", "no", "never")
_EXCLUSION_PHRASES = (
    "excluding",
    "exclude",
    "excludes",
    "except",
    "without",
    "other than",
    "rather than",
    "excl",
)


def _label_negated_in_text(label: str, text: str) -> bool:
    """Treat the first whole-word match as evidence against a value in a negation or exclusion.

    Check negators and exclusion phrases within the preceding three words, plus a
    fused or spaced ``non`` prefix. Later matches require separate calls.
    """
    pattern = r"\b" + re.escape(label) + r"\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return False
    prefix = text[: match.start()]
    if re.search(r"\bnon[\s-]*$", prefix, re.IGNORECASE):
        return True
    prefix_words = re.findall(r"[\w']+", prefix)
    window_text = " ".join(prefix_words[-_NEGATION_WINDOW_WORDS:]).lower()
    for negator in _NEGATORS:
        if re.search(r"\b" + re.escape(negator) + r"\b", window_text):
            return True
    for phrase in _EXCLUSION_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", window_text):
            return True
    return False


def _unscoped_property_values(shape_draft: ShapeDraft, property_dcid: str) -> list[str]:
    """Return distinct values across all confirmed members, including those without place data,
    in first-occurrence order.
    """
    sv_arc_facts = shape_draft.sv_arc_facts or {}
    values: list[str] = []
    seen: set[str] = set()
    for arcs in sv_arc_facts.values():
        value = read_constraints(arcs).get(property_dcid)
        if value is not None and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def restore_starved_axis_binding(
    bindings: Sequence[SlotBindingDraft],
    shape_draft: ShapeDraft,
    variable: str,
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Restore a place-starved binding when the query names its value.

    A restore requires a place-available shape, an open binding with an empty or missing
    place-scoped taxonomy, and multiple observed values for the property across all members.
    Exactly one value label must match ``variable`` as an unnegated, unexcluded whole word.
    The complete binding set must remain realizable.

    On success, set the binding to ``value``/``mentioned`` and restore the slot taxonomy in
    place to all unscoped values. This keeps grounding options aligned with the binding.
    The draft stays open when matches are ambiguous, excluded, or unrealizable. Under-firing
    is acceptable; over-firing is not.

    Returns the listified input and ``[]`` when unchanged. Otherwise returns a repaired copy
    and the original drafts for restored slots. Iteration follows binding and arc-fact
    insertion order.
    """
    bindings = list(bindings)
    if shape_draft.place_available is not True:
        return bindings, []
    if shape_draft.slot_taxonomy is None:
        return bindings, []

    disagreement_props = _shape_disagreement_props(shape_draft)
    value_labels = shape_draft.value_labels or {}

    result = list(bindings)
    restored: list[SlotBindingDraft] = []
    for i, b in enumerate(result):
        if b.kind not in ("unbound", "absent") or b.value_dcids:
            continue
        if b.property_dcid is None or b.property_dcid not in disagreement_props:
            continue
        taxonomy_key = f"{b.axis}:{b.property_dcid}"
        if shape_draft.slot_taxonomy.get(taxonomy_key):
            continue

        values = _unscoped_property_values(shape_draft, b.property_dcid)
        matches = [
            v
            for v in values
            if (label := value_labels.get(v))
            and _label_named_in_text(label, variable)
            and not _label_negated_in_text(label, variable)
        ]
        if len(matches) != 1:
            continue  # Zero matches leave the slot open. Multiple matches are ambiguous.

        fired = b.model_copy(
            update={"kind": "value", "evidence": "mentioned", "value_dcids": [matches[0]]}
        )
        trial = list(result)
        trial[i] = fired
        trial_bound_values = bound_values_from_bindings(trial)
        sv_arc_facts = shape_draft.sv_arc_facts or {}
        if not bound_realizable_members(list(sv_arc_facts), sv_arc_facts, trial_bound_values):
            continue

        result[i] = fired
        shape_draft.slot_taxonomy[taxonomy_key] = values
        restored.append(b)

    if not restored:
        return bindings, []
    return result, restored


def _slot_open_to_default(
    binding: SlotBindingDraft, shape_draft: ShapeDraft, code_system: str, default_code: str
) -> bool:
    """True when the slot is open and contains the declared code.

    A ``requested_all`` slot keeps its full requested scope.
    """
    if binding.kind not in ("unbound", "absent") or binding.value_dcids:
        return False
    if binding.evidence == "requested_all":
        return False
    slot = next((s for s in shape_draft.slot_keys if s.property_dcid == code_system), None)
    if slot is None:
        return False
    taxonomy_key = f"{slot.axis}:{code_system}"
    taxonomy_values = (shape_draft.slot_taxonomy or {}).get(taxonomy_key, [])
    return default_code in taxonomy_values


def _some_binding_triggers_default(
    bindings: Sequence[SlotBindingDraft],
    scoping_dcids: frozenset[str],
    engine_authored_properties: frozenset[str],
) -> bool:
    """Find a query-derived ``value``/``mentioned`` binding under the default's scope.

    Ignore engine-authored properties because binding evidence defaults to ``mentioned``.
    """
    return any(
        b.property_dcid is not None
        and b.property_dcid not in engine_authored_properties
        and b.kind == "value"
        and b.evidence == "mentioned"
        and bool(b.value_dcids)
        and b.value_dcids[0] in scoping_dcids
        for b in bindings
    )


def _names_trigger_word(default_when_named: tuple[str, ...], text: str) -> bool:
    """Match a whole-word trigger outside negation or exclusion contexts.

    An empty list returns ``False``.

    The ``non-aid`` prefix excludes ``aid`` as a trigger.
    """
    return any(
        _label_named_in_text(word, text) and not _label_negated_in_text(word, text)
        for word in default_when_named
    )


def declared_default_words_named(text: str, *, index: CodeIndex) -> tuple[tuple[str, str], ...]:
    """Return named default triggers in declaration order, then ``when_named`` order.

    The response cache key includes these matches because equal extraction can produce
    different defaults from different query text.
    """
    return tuple(
        (code_system, word)
        for code_system, declaration in index.declarations.items()
        if declaration.default is not None
        for word in declaration.default.when_named
        if _names_trigger_word((word,), text)
    )


def apply_declared_defaults(
    bindings: Sequence[SlotBindingDraft],
    shape_draft: ShapeDraft,
    *,
    index: CodeIndex,
    variable: str = "",
    query_text: str | None = None,
    query_qualifier_matches: tuple[QualifierMatch, ...] = (),
    query_qualifier_ties: tuple[QualifierTie, ...] = (),
    engine_authored_properties: frozenset[str] = frozenset(),
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Apply declared defaults to open slots when query evidence triggers them.

    The packaged code index defines each default and its ``when_bound_under`` scopes or
    ``when_named`` words. ``query_text`` is the raw query only for a single-variable
    extraction, when trigger words can be attributed to this variable. Both it and the
    extracted ``variable`` can trigger a default.

    A slot must be open, have no ``requested_all`` evidence, and contain the default code in
    its taxonomy. Any raw qualifier match or tie for that code system blocks the default,
    including unresolved ambiguity. Then either another query-derived ``value``/
    ``mentioned`` binding enters the declared scope on a non-engine-authored property, or
    the variable/query text names a ``when_named`` word outside negation or exclusion.
    Exclude engine-authored properties from the binding trigger because their default
    evidence is ``mentioned``. Text triggers remain independent.

    Only ``crsTypeOfFlow`` gets the pipeline's separate ``qualifier_not_bound`` refusal
    for no-vocabulary qualifiers. Every declaration reads the same original bindings, so
    trigger decisions are order-independent. The wire records fired defaults as ``declared_default``
    provenance. A fired default warns only where the query names its slot
    (``defaulted_slots_named``).

    Returns the listified input and ``[]`` when unchanged. Otherwise returns the repaired
    bindings and the pre-default entries. Iteration follows binding and declaration order.
    """
    bindings = list(bindings)

    fire: dict[int, str] = {}
    for code_system, declaration in index.declarations.items():
        default = declaration.default
        if default is None:
            continue
        target_index = next(
            (i for i, b in enumerate(bindings) if b.property_dcid == code_system), None
        )
        if target_index is None:
            continue
        if not _slot_open_to_default(
            bindings[target_index], shape_draft, code_system, default.code
        ):
            continue
        if any(m.code_system == code_system for m in query_qualifier_matches):
            continue
        if any(row.code_system == code_system for tie in query_qualifier_ties for row in tie.rows):
            continue
        scoping_dcids = frozenset(
            dcid for scope in default.when_bound_under for dcid in index.descendants(scope)
        )
        triggered = _some_binding_triggers_default(
            bindings, scoping_dcids, engine_authored_properties
        ) or any(
            _names_trigger_word(default.when_named, text)
            for text in (variable, query_text)
            if text is not None
        )
        if not triggered:
            continue
        fire[target_index] = default.code

    if not fire:
        return bindings, []

    result = list(bindings)
    defaulted: list[SlotBindingDraft] = []
    for i, code in fire.items():
        defaulted.append(result[i])
        result[i] = result[i].model_copy(update={"kind": "value", "value_dcids": [code]})
    return result, defaulted


@dataclass(frozen=True, slots=True)
class ShapeSlotDefault:
    """The value an open constraint slot takes on every shape of one ``populationType``.

    The packaged code index declares defaults for the code systems it carries
    (``CodeSystemDeclaration.default``). This declares one for a graph property outside it.
    """

    population_type: str
    property_dcid: str
    value_dcid: str


# A birth in demographic statistics is a live birth, and the crude birth rate is live births
# per person. A births question that names no status reads live births.
SHAPE_SLOT_DEFAULTS: tuple[ShapeSlotDefault, ...] = (
    ShapeSlotDefault(
        population_type="BirthEvent", property_dcid="medicalStatus", value_dcid="LiveBirth"
    ),
)


def apply_shape_slot_defaults(
    bindings: Sequence[SlotBindingDraft],
    shape_draft: ShapeDraft,
    *,
    defaults: Sequence[ShapeSlotDefault] = SHAPE_SLOT_DEFAULTS,
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Bind each open slot that ``defaults`` declares for ``shape_draft``'s ``populationType``.

    A slot takes its default under the conditions ``apply_declared_defaults`` puts on a slot:
    it is unbound or absent, carries no ``requested_all`` evidence, and its taxonomy holds
    the default value. The availability read at the queried place must also have run
    (``ShapeDraft.place_available`` is true or false). Where it succeeded, the taxonomy
    holds the values of the members with data there, so where the family's total has data
    and no member carrying the default has, the slot stays open and the total answers.
    Where the read failed, was skipped, or had no place to read (``place_available`` is
    ``None``), the taxonomy lists every member whether or not it has data, so the slot
    stays open and the total answers. Where no member has data, the default binds and the
    answer is no data. A slot the binder gave a value keeps it.

    Returns the listified input and ``[]`` when unchanged. Otherwise returns the repaired
    bindings and the pre-default entries. Iteration follows declaration order.
    """
    result = list(bindings)
    defaulted: list[SlotBindingDraft] = []
    if shape_draft.place_available is None:
        return result, defaulted
    for default in defaults:
        if default.population_type != shape_draft.pop_type_dcid:
            continue
        target_index = next(
            (i for i, b in enumerate(result) if b.property_dcid == default.property_dcid), None
        )
        if target_index is None:
            continue
        if not _slot_open_to_default(
            result[target_index], shape_draft, default.property_dcid, default.value_dcid
        ):
            continue
        defaulted.append(result[target_index])
        result[target_index] = result[target_index].model_copy(
            update={"kind": "value", "value_dcids": [default.value_dcid]}
        )
    return result, defaulted


def defaulted_slots_named(
    defaulted: Sequence[SlotBindingDraft], *, index: CodeIndex, texts: Sequence[str]
) -> list[tuple[str, str]]:
    """Return ``(plain_name, default label)`` for each defaulted slot one of ``texts``
    names.

    ``defaulted`` holds the pre-default entries ``apply_declared_defaults`` returns. A text
    names a slot by its declared plain name, singular or plural (``names_plain_name``), so
    "top 5 aid flows" names the flow slot. A default fires only on a slot no operation and
    no ``requested_all`` claim opened, so each pair is a slot the query asked about and
    the answer fixed to one value.
    """
    named: list[tuple[str, str]] = []
    for binding in defaulted:
        declaration = index.declarations.get(binding.property_dcid or "")
        if declaration is None or declaration.default is None:
            continue
        if not any(names_plain_name(text, declaration.plain_name) for text in texts):
            continue
        row = index.code_by_dcid(declaration.default.code)
        named.append(
            (declaration.plain_name, row.label if row is not None else declaration.default.code)
        )
    return named


def declared_default_slots_named(text: str, *, index: CodeIndex) -> tuple[str, ...]:
    """Return, in declaration order, each code system with a declared default whose plain
    name ``text`` names.

    The response cache key includes these because ``defaulted_slots_named`` reads the raw
    query, so equal extractions of "aid to Kenya" and "aid flows to Kenya" warn
    differently.
    """
    return tuple(
        code_system
        for code_system, declaration in index.declarations.items()
        if declaration.default is not None and names_plain_name(text, declaration.plain_name)
    )
