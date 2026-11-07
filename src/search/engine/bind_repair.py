"""Apply declared defaults to draft bindings, and read the slots they bind.

``apply_declared_defaults`` binds an open slot to its declared default code when the
variable, the raw query or another binding triggers it, ``apply_shape_slot_defaults``
binds the code index's default for a slot of a population type, ``apply_broad_readings``
reads a bound value as the broader value the code index declares for it, and
``apply_named_exclusions`` binds a value whose label a stretch of the query that leaves
something out writes word for word. The helpers that
match a value's label as a whole word outside a negation or exclusion serve the bind phase
too, and so does the test of whether a part states a word a dataset declares
(`word_stated`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from code_index.query import CodeIndex
from code_index.tokens import tokenize, word_tokens
from code_index.types import BroadReading, ShapeSlotDefault
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
from search.render.labels import join_labels

if TYPE_CHECKING:
    from search.engine.exclusion_read import LeftOut


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
    return any(
        occurrence_negated(text, match.start(), match.end())
        for match in re.finditer(pattern, text, re.IGNORECASE)
    )


def occurrence_negated(text: str, start: int, end: int) -> bool:
    """Whether the occurrence of a word at `text[start:end]` is negated or excluded: a
    negator or exclusion phrase among the three words before it, or a fused or spaced
    ``non`` prefix. Other occurrences of the word in those three words are ignored."""
    label = text[start:end]
    pattern = r"\b" + re.escape(label) + r"\b"
    prefix = text[:start]
    prefix_words = list(re.finditer(r"[\w']+", prefix))
    context_start = prefix_words[-3].start() if len(prefix_words) >= 3 else 0
    context = re.sub(pattern, "", prefix[context_start:], flags=re.IGNORECASE)
    return _label_negated_in_text(label, f"{context}{label}")


def word_stated(word: str, texts: Sequence[str]) -> bool:
    """Whether `texts`, a part's variable and typed qualifier phrases, state the casefolded
    `word` as a modifier of the measure (`_modifies_the_measure`), with no occurrence in them
    negated or excluded (`_label_has_negated_occurrence`)."""
    if not any(_modifies_the_measure(word, text) for text in texts):
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
    return _taxonomy_holds(shape_draft, code_system, default_code)


def _above_default_unnamed(
    binding: SlotBindingDraft,
    shape_draft: ShapeDraft,
    code_system: str,
    default_code: str,
    *,
    index: CodeIndex,
    named_codes: frozenset[str],
) -> tuple[str, ...]:
    """The values of the reply's `binding` broader than the declared default that the
    question does not name, in binding order, on a slot whose taxonomy holds the default;
    empty when there are none.

    A code system that declares a default states what a question naming none of its codes
    asks about: a DAC1 question that names no aid type asks about total ODA,
    ``DAC1AidType1010``. A proper ancestor of that default, such as ``DAC1AidType5``
    ("official and private flow"), is a broader and larger quantity, so a reply binding it
    for "total aid" answers a different question. Such a value is the question's only when
    it is one of `named_codes` or lies under one of them. A value at or under the default,
    or beside it, is the reply's to bind. A ``requested_all`` slot keeps its full
    requested scope.
    """
    if binding.kind not in ("value", "set") or binding.evidence == "requested_all":
        return ()
    if not _taxonomy_holds(shape_draft, code_system, default_code):
        return ()
    return tuple(
        value
        for value in binding.value_dcids
        if value != default_code
        and default_code in index.descendants(value)
        and not any(value in index.descendants(code) for code in named_codes)
    )


def _taxonomy_holds(shape_draft: ShapeDraft, property_dcid: str, value_dcid: str) -> bool:
    """Whether ``shape_draft`` has a slot on ``property_dcid`` whose taxonomy holds
    ``value_dcid``."""
    slot = next((s for s in shape_draft.slot_keys if s.property_dcid == property_dcid), None)
    if slot is None:
        return False
    taxonomy_values = (shape_draft.slot_taxonomy or {}).get(f"{slot.axis}:{property_dcid}", [])
    return value_dcid in taxonomy_values


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
    named_codes: frozenset[str] = frozenset(),
) -> tuple[list[SlotBindingDraft], list[SlotBindingDraft]]:
    """Apply declared defaults to open slots when query evidence triggers them.

    The release's code index defines each default and its ``when_bound_under`` scopes or
    ``when_named`` words. ``query_text`` is the raw query only for a single-variable
    extraction, when trigger words can be attributed to this variable. Both it and the
    extracted ``variable`` can trigger a default.

    A slot must be open, have no ``requested_all`` evidence, contain the default code in
    its taxonomy, and hold no pin of the elected dataset. A slot the reply bound to values
    above the default that the question does not name (`_above_default_unnamed`, read
    against `named_codes`, the codes the question names) takes the default as an open one
    does: the default replaces exactly those values and the slot keeps the reply's others,
    so a set beside them stays bound. Wherever the default does not fire the slot keeps the
    reply's values. ``pinned_properties`` names the
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
    bindings and the pre-default entries, each an open slot's entry or, where the default
    replaced reply values, the reply's entry holding only those values. Iteration follows
    binding and declaration order.
    """
    bindings = list(bindings)

    fire: dict[int, tuple[str, tuple[str, ...]]] = {}
    for code_system, declaration in index.declarations.items():
        default = declaration.default
        if default is None:
            continue
        target_index = next(
            (i for i, b in enumerate(bindings) if b.property_dcid == code_system), None
        )
        if target_index is None or code_system in pinned_properties:
            continue
        binding = bindings[target_index]
        replaces = _above_default_unnamed(
            binding,
            shape_draft,
            code_system,
            default.code,
            index=index,
            named_codes=named_codes,
        )
        if not replaces and not _slot_open_to_default(
            binding, shape_draft, code_system, default.code
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
        fire[target_index] = default.code, replaces

    if not fire:
        return bindings, []

    result = list(bindings)
    defaulted: list[SlotBindingDraft] = []
    for i, (code, replaces) in fire.items():
        binding = result[i]
        if not replaces:
            defaulted.append(binding)
            result[i] = binding.model_copy(update={"kind": "value", "value_dcids": [code]})
            continue
        defaulted.append(binding.model_copy(update={"value_dcids": list(replaces)}))
        values = list(
            dict.fromkeys(code if value in replaces else value for value in binding.value_dcids)
        )
        result[i] = binding.model_copy(
            update={
                "kind": "value" if len(values) == 1 else "set",
                "value_dcids": values,
                "evidence": "mentioned",
                "drafted_by": "engine",
            }
        )
    return result, defaulted


def apply_named_exclusions(
    bindings: Sequence[SlotBindingDraft],
    shape_draft: ShapeDraft,
    *,
    exclusions: Sequence[LeftOut],
    pinned_properties: frozenset[str] = frozenset(),
) -> tuple[list[SlotBindingDraft], tuple[tuple[str, str, str], ...]]:
    """Bind each open slot of `shape_draft` holding a value whose graph label is, word for word,
    one of the `exclusions`: "without IMF credit" binds `WithoutIMFCredit` on
    `includesIMFCredit`. An open slot is as `_slot_open_to_default` reads it, and no pin of the
    elected dataset (`pinned_properties`) holds it. The label is the value's own, so the
    answer leaves the phrase out by construction. A label whose every word lies in what the
    stretch leaves out (`LeftOut.things`, taken together) names the value it excludes, so it
    binds nothing.

    Returns the listified bindings, repaired, and each phrase they now bind with the property
    and value it binds.
    """
    result = list(bindings)
    if not exclusions:
        return result, ()
    labels = shape_draft.value_labels or {}
    phrases = {
        words: exclusion.phrase
        for exclusion in exclusions
        if not set(words := tuple(_folded_tokens(exclusion.phrase)))
        <= {word for thing in exclusion.things for word in _folded_tokens(thing)}
    }
    bound: list[tuple[str, str, str]] = []
    for i, binding in enumerate(result):
        prop = binding.property_dcid
        if prop is None or prop in pinned_properties:
            continue
        for value in (shape_draft.slot_taxonomy or {}).get(f"{binding.axis}:{prop}", ()):
            phrase = phrases.get(tuple(_folded_tokens(labels.get(value, ""))))
            if phrase is not None and _slot_open_to_default(binding, shape_draft, prop, value):
                result[i] = binding.model_copy(update={"kind": "value", "value_dcids": [value]})
                bound.append((phrase, prop, value))
                break
    return result, tuple(bound)


def _folded_tokens(text: str) -> list[str]:
    return [token.text.casefold() for token in word_tokens(text)]


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


def apply_broad_readings(
    bindings: Sequence[SlotBindingDraft], shape_draft: ShapeDraft, *, index: CodeIndex
) -> tuple[list[SlotBindingDraft], list[BroadReading]]:
    """Read each bound value the code index declares a broad reading for
    (``CodeIndex.broad_readings``) as the broader value it declares.

    A ``value`` or ``set`` binding holding ``BroadReading.value_dcid`` holds
    ``BroadReading.reads_as`` in its place unless a ``value`` or ``set`` binding on
    ``BroadReading.unless_bound`` the query mentions holds ``BroadReading.held_with``, the
    value every series of the narrow value holds, and when the slot's taxonomy holds the
    broader value. Another value on ``unless_bound`` ("publicly guaranteed government debt")
    names no narrow series, so the reading fires. A set that collapses to one value becomes a
    ``value`` binding.

    Returns the repaired bindings and the readings that fired, in declaration order.
    """
    result = list(bindings)
    fired: list[BroadReading] = []
    for reading in index.broad_readings:
        target_index = next(
            (
                i
                for i, b in enumerate(result)
                if b.property_dcid == reading.property_dcid
                and b.kind in ("value", "set")
                and reading.value_dcid in b.value_dcids
            ),
            None,
        )
        if target_index is None:
            continue
        if any(
            b.property_dcid == reading.unless_bound
            and b.kind in ("value", "set")
            and b.evidence == "mentioned"
            and reading.held_with in b.value_dcids
            for b in result
        ):
            continue
        if not _taxonomy_holds(shape_draft, reading.property_dcid, reading.reads_as):
            continue
        values = list(
            dict.fromkeys(
                reading.reads_as if value == reading.value_dcid else value
                for value in result[target_index].value_dcids
            )
        )
        result[target_index] = result[target_index].model_copy(
            update={"kind": "value" if len(values) == 1 else "set", "value_dcids": values}
        )
        fired.append(reading)
    return result, fired


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
    the answer fixed to one value. A default with a declared note, or one that replaced a
    reply's value (``_replaced``), speaks through ``defaulted_slot_notes`` and is left out
    here.
    """
    named: list[tuple[str, str]] = []
    for binding in defaulted:
        declaration = index.declarations.get(binding.property_dcid or "")
        if declaration is None or declaration.default is None or declaration.default.note:
            continue
        if _replaced(binding):
            continue
        if not any(names_plain_name(text, declaration.plain_name) for text in texts):
            continue
        named.append((declaration.plain_name, _code_label(declaration.default.code, index=index)))
    return named


def defaulted_slot_notes(defaulted: Sequence[SlotBindingDraft], *, index: CodeIndex) -> list[str]:
    """Return a note for each fired default, in the order of ``defaulted``, the pre-default
    entries ``apply_declared_defaults`` returns, whatever the query's words: for a default
    that replaced a reply's broader value (``_replaced``), a sentence naming the default and
    the value it replaced, so an override the question meant to avoid shows; for any other,
    its declared note (``DeclaredDefault.note``) where it carries one."""
    notes: list[str] = []
    for binding in defaulted:
        declaration = index.declarations.get(binding.property_dcid or "")
        if declaration is None or declaration.default is None:
            continue
        if _replaced(binding):
            replaced = join_labels([_code_label(dcid, index=index) for dcid in binding.value_dcids])
            notes.append(
                f"The answer covers {_code_label(declaration.default.code, index=index)}, the "
                f"default {declaration.plain_name}, in place of {replaced}, a broader "
                f"{declaration.plain_name} no word of the question names."
            )
        elif declaration.default.note:
            notes.append(declaration.default.note)
    return notes


def _replaced(binding: SlotBindingDraft) -> bool:
    """Whether a pre-default entry bound a value its default replaced
    (`_above_default_unnamed`); an open slot's entry binds none."""
    return binding.kind in ("value", "set")


def _code_label(dcid: str, *, index: CodeIndex) -> str:
    """The code index's full label for `dcid`, or the dcid where it holds no row."""
    row = index.code_by_dcid(dcid)
    return row.label if row is not None else dcid


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
