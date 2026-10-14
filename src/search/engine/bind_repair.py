"""Apply declared defaults to draft bindings, and read the slots they bind.

``apply_declared_defaults`` binds an open slot to its declared default code when the
variable, the raw query or another binding triggers it, and ``apply_shape_slot_defaults``
binds the code index's default for a slot of a population type. The helpers that match a
value's label as a whole word outside a negation or exclusion serve the bind phase too, and
so does the test of whether a part states a word a dataset declares (`word_stated`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from code_index.query import CodeIndex
from code_index.tokens import tokenize
from code_index.types import ShapeSlotDefault
from search.engine.bind import SlotBindingDraft
from search.engine.function_words import FUNCTION_WORDS
from search.engine.qualifier_match import (
    CONNECTIVES,
    QualifierMatch,
    QualifierTie,
    names_plain_name,
)
from search.engine.retrieval.facts import read_constraints
from search.engine.shape import ShapeDraft


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


def _label_has_negated_occurrence(label: str, text: str) -> bool:
    """Whether any whole-word occurrence is negated, including mixed wording."""
    pattern = r"\b" + re.escape(label) + r"\b"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        prefix = text[: match.start()]
        prefix_words = list(re.finditer(r"[\w']+", prefix))
        context_start = prefix_words[-3].start() if len(prefix_words) >= 3 else 0
        context = prefix[context_start:]
        context = re.sub(pattern, "", context, flags=re.IGNORECASE)
        context = f"{context}{text[match.start() : match.end()]}"
        if _label_negated_in_text(label, context):
            return True
    return False


def stated_words_negated(text: str, *, index: CodeIndex) -> tuple[str, ...]:
    """The words datasets state of their own rows (`DatasetDeclaration.stated_words`) or name
    a party for (`DatasetDeclaration.party_words`) that an occurrence in `text` negates or
    excludes (`_label_has_negated_occurrence`), sorted.

    The response cache key includes them because the rules reading those words read them
    from the verbatim query, which an equal extraction can leave behind: "not bilateral ODA
    to Kenya" and "bilateral ODA to Kenya" can extract alike, and so can "not multilateral
    ODA to Kenya" and "multilateral ODA to Kenya".
    """
    words = {
        declared.word
        for declaration in index.datasets.values()
        for declared in (*declaration.stated_words, *declaration.party_words)
    }
    return tuple(sorted(word for word in words if _label_has_negated_occurrence(word, text)))


def word_stated(word: str, texts: Sequence[str], *, negated: AbstractSet[str]) -> bool:
    """Whether `texts`, a part's variable and typed qualifier phrases, state the casefolded
    `word` as a modifier of the measure (`_modifies_the_measure`), with no occurrence in them
    negated or excluded (`_label_has_negated_occurrence`), and `word` is not among
    `negated`, the declared words the verbatim query negates (`stated_words_negated`, which
    reaches a negation the extraction left out of the variable, as "not bilateral ODA")."""
    if word in negated or not any(_modifies_the_measure(word, text) for text in texts):
        return False
    return not any(_label_has_negated_occurrence(word, text) for text in texts)


def _modifies_the_measure(word: str, text: str) -> bool:
    """Whether the casefolded `word` is a whole word of `text` with nothing before it but
    other words of the measure phrase: no function word (`FUNCTION_WORDS`), connective
    (`CONNECTIVES`) or separator. "Bilateral" modifies the measure in "gross bilateral ODA
    disbursements" and in "bilateral aid for health". In "ODA net of bilateral flows" and
    "ODA, bilateral excluded" it belongs to a clause that restricts the measure."""
    for token in tokenize(text):
        folded = token.text.casefold()
        if token.kind != "word" or folded in FUNCTION_WORDS or folded in CONNECTIVES:
            return False
        if folded == word:
            return True
    return False


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
    pinned_properties: frozenset[str] = frozenset(),
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Apply declared defaults to open slots when query evidence triggers them.

    The packaged code index defines each default and its ``when_bound_under`` scopes or
    ``when_named`` words. ``query_text`` is the raw query only for a single-variable
    extraction, when trigger words can be attributed to this variable. Both it and the
    extracted ``variable`` can trigger a default.

    A slot must be open, have no ``requested_all`` evidence, contain the default code in
    its taxonomy, and hold no pin of the elected dataset. ``pinned_properties`` names the
    pins, and every row of that dataset holds the pinned value, so the slot takes no
    default. Any raw qualifier match or tie for that code system blocks the default,
    including unresolved ambiguity. Then either another query-derived ``value``/
    ``mentioned`` binding enters the declared scope on a non-engine-authored property, or
    the variable/query text names a ``when_named`` word outside negation or exclusion, or
    the default fires on every open slot (``when_open``).
    Exclude engine-authored properties from the binding trigger because their default
    evidence is ``mentioned``. Text triggers remain independent.

    A word a code system declares it has no vocabulary for
    (``CodeSystemDeclaration.no_vocabulary_terms``) gets the pipeline's separate
    ``qualifier_not_bound`` refusal. Every declaration reads the same original bindings, so
    trigger decisions are order-independent. The wire records fired defaults as ``declared_default``
    provenance. A fired default with a declared note states it (``defaulted_slot_notes``);
    one without warns only where the query names its slot (``defaulted_slots_named``).

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
        if target_index is None or code_system in pinned_properties:
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
        triggered = (
            default.when_open
            or _some_binding_triggers_default(bindings, scoping_dcids, engine_authored_properties)
            or any(
                _names_trigger_word(default.when_named, text)
                for text in (variable, query_text)
                if text is not None
            )
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


def apply_shape_slot_defaults(
    bindings: Sequence[SlotBindingDraft],
    shape_draft: ShapeDraft,
    *,
    index: CodeIndex,
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft], list[ShapeSlotDefault]]:
    """Bind each open slot the code index declares a default for on ``shape_draft``'s
    ``populationType`` (``CodeIndex.shape_slot_defaults``).

    A slot takes its default under the conditions ``apply_declared_defaults`` puts on a slot:
    it is unbound or absent, carries no ``requested_all`` evidence, and its taxonomy holds
    the default value. The availability read at the queried place must also have run
    (``ShapeDraft.place_available`` is true or false); where it failed, was skipped, or had
    no place to read (``None``), the slot stays open and the family's total answers. Where
    the read found data on some member but on none that holds the default value
    (`_default_has_no_data`), the default is not applied either: the slot stays open, the
    total answers, and the default is returned as relaxed. A slot the binder gave a value
    keeps it.

    Returns the repaired bindings, the pre-default entries of the slots that took a default,
    and the defaults not applied for want of data, each in declaration order.
    """
    result = list(bindings)
    defaulted: list[SlotBindingDraft] = []
    relaxed: list[ShapeSlotDefault] = []
    if shape_draft.place_available is None:
        return result, defaulted, relaxed
    for default in index.shape_slot_defaults:
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
        if _default_has_no_data(shape_draft, default):
            relaxed.append(default)
            continue
        defaulted.append(result[target_index])
        result[target_index] = result[target_index].model_copy(
            update={"kind": "value", "value_dcids": [default.value_dcid]}
        )
    return result, defaulted, relaxed


def _default_has_no_data(shape_draft: ShapeDraft, default: ShapeSlotDefault) -> bool:
    """Whether the availability read found data at the queried place on some member of
    ``shape_draft`` and on none that reads ``default``'s value. False where the read holds
    no per-member coverage (``ShapeDraft.member_coverage`` is None), as for a role-bearing
    shape, where no member has data, since the default reading then has as much data as the
    total, and where a member with data has no arcs read, since it may hold the value."""
    coverage = shape_draft.member_coverage
    if coverage is None:
        return False
    with_data = [sv for sv, places in coverage.items() if places]
    if not with_data:
        return False
    arcs = shape_draft.sv_arc_facts or {}
    # A member whose arcs were not read may hold the value.
    return all(
        arcs.get(sv) and read_constraints(arcs[sv]).get(default.property_dcid) != default.value_dcid
        for sv in with_data
    )


def defaulted_slots_named(
    defaulted: Sequence[SlotBindingDraft], *, index: CodeIndex, texts: Sequence[str]
) -> list[tuple[str, str]]:
    """Return ``(plain_name, default label)`` for each defaulted slot one of ``texts``
    names.

    ``defaulted`` holds the pre-default entries ``apply_declared_defaults`` returns. A text
    names a slot by its declared plain name, singular or plural (``names_plain_name``), so
    "top 5 aid flows" names the flow slot. A default fires only on a slot no operation and
    no ``requested_all`` claim opened, so each pair is a slot the query asked about and
    the answer fixed to one value. A default with a declared note speaks through it
    (``defaulted_slot_notes``) and is left out here.
    """
    named: list[tuple[str, str]] = []
    for binding in defaulted:
        declaration = index.declarations.get(binding.property_dcid or "")
        if declaration is None or declaration.default is None or declaration.default.note:
            continue
        if not any(names_plain_name(text, declaration.plain_name) for text in texts):
            continue
        row = index.code_by_dcid(declaration.default.code)
        named.append(
            (declaration.plain_name, row.label if row is not None else declaration.default.code)
        )
    return named


def defaulted_slot_notes(defaulted: Sequence[SlotBindingDraft], *, index: CodeIndex) -> list[str]:
    """Return the declared note of each fired default that carries one
    (``DeclaredDefault.note``), in the order of ``defaulted``, the pre-default entries
    ``apply_declared_defaults`` returns. The note is stated whatever the query's words."""
    notes: list[str] = []
    for binding in defaulted:
        declaration = index.declarations.get(binding.property_dcid or "")
        if declaration is not None and declaration.default is not None and declaration.default.note:
            notes.append(declaration.default.note)
    return notes


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
