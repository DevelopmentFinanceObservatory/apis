"""The two Jev requests of a part: the quantity read's questions and the bind questions.

Each question reads the variable phrase and one thing beside it: a shape, a measure-axis
value, a slot value or a dropped phrase. Jev answers each over the request's state and that
question alone, and `jev_assemble` turns the answers into the bind output and the quantity
read. The state holds the phrases, the query and, for the quantity request, its rules and
denominators; shapes appear only inside question text.

Question ids are positional: ``m|i`` names menu shape i, ``n|P|j`` value j of slot P's
union (`jev_menu.asked_values`), and ``unread|i|k`` word k of shape i's nominated words
(`BindRequest.nominated_words`), so a word's spelling never enters an id. The quantity
read's ids start ``q|``.

A bind whose answers draft values the phrase names on two or more slots of one shape asks a
second request over the same state (`span_questions`): which words name each such value,
``span|P|dcid``, and which name the measure, ``span|measure``.

Once the bind has bound values on the elected shape, the ``unread|`` questions on its words
the bind rated as unread are asked again over the shape as those values restrict it
(`restated_request`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from typesafe_sdk import ChoiceModel, JSONContent, NoulCriteria, NoulModel

from search.engine.bind import SlotBindingDraft, stated_reading
from search.engine.binders.jev_client import JevQuestion
from search.engine.binders.jev_menu import (
    JevMenu,
    Prop,
    asked_values,
    describe,
    measure_values,
    prop_id,
)
from search.engine.shape import ShapeDraft
from search.models import GraphRef

type JevRequest = tuple[dict[str, JSONContent], dict[str, JevQuestion]]
"""A request's state and its questions by id."""

_FORM_WORDS = "a per-person, per-area, percentage, share or ratio form, or the base it divides by"

QUANTITY_PREFIX = "q|"
UNREAD_PREFIX = "unread|"
SPAN_MEASURE = "span|measure"
NONE = "none"
NONE_OF_THESE = "none of these"
_MAX_SPAN_WORDS = 5
_CHOICE_CAP = 255
_WORD_RE = re.compile(r"[^\s\-/]+")

FORMS: dict[str, str] = {
    "per_person": "per capita, per person",
    "per_head_base": "per a number of people, as 'per 100,000 people'",
    "per_area": "per square km, density",
    "rate": "one quantity to another: 'debt to GDP', 'a ratio'",
    "share": "a part of a whole: '% of GDP', 'share of spending'",
    NONE: "a plain amount or count: the phrase's words name no form",
}

QUANTITY_RULES: list[JSONContent] = [
    "Read the quantity from the phrase's own words alone. Set a field only when words in "
    "the phrase name it ('to GDP', 'share of', 'ratio', 'per capita', 'as a percentage "
    "of', 'density').",
    "A measure noun alone ('spending', 'expenditure', 'debt', 'aid', 'GDP') names a plain "
    "amount: no form and no base.",
    "The base is what a ratio or share divides by: 'GDP' in 'debt to GDP', 'GNI' in 'aid as "
    "a share of GNI', 'exports' in 'debt service as a percentage of exports'. A phrase "
    "naming only people or area ('per capita', 'per square km') names no base, and a form "
    "word alone ('ratio', 'rate', 'share') names no base.",
    "'government education spending' names neither a base nor a form.",
]


def bind_request(menu: JevMenu) -> JevRequest:
    """The bind request over `menu`: the variable phrase and, when the request has it, the
    query in the state, and every bind question. Without a query the request asks no
    ``electq`` and no ``mq|i``."""
    request = menu.request
    state: dict[str, JSONContent] = {"variable_phrase": request.variable}
    if request.query_text is not None:
        state["query"] = request.query_text
    return state, _bind_questions(menu, query=request.query_text is not None)


# The question whether a shape's series states what a nominated word names: yes when the
# series measures only what the word names, so the word is read. A calibration run swaps
# these two to choose the wording on labelled pairs.
STATED_WORD_INSTRUCTIONS = (
    "Does the measure {measure} measure only what '{word}' names in `variable_phrase`, "
    "under any wording, value or synonym? Judge the word alone, whatever the rest of the "
    "phrase asks."
)
STATED_WORD_CRITERIA: NoulCriteria = {
    "true": "The measure measures what the word names, in the same or other words or "
    "through a value it is restricted to; or the word names the measure's form (a rate, "
    "share, count, index or average) or the people or things the measure is defined over; "
    "or the word only describes, restates or links what the measure counts.",
    "false": "The word names a narrower or a different thing than the measure counts, so "
    "the measure counts more than the word names, or something else.",
}


def restated_request(
    variable: str,
    shape: ShapeDraft,
    index: int,
    words: Sequence[str],
    *,
    bindings: Sequence[SlotBindingDraft],
    served: Sequence[str] = (),
) -> JevRequest:
    """The stated-word questions asked again once the bind has bound values: the variable
    phrase in the state, and one ``unread|index|k`` noul per word k of `words`, quoting menu
    shape `index` as the values `bindings` restrict it to, those its measure's definition
    fixes, and the `served` series when they differ from its label (`describe`)."""
    desc = describe(shape, bindings=bindings, definition=True, served=served)
    questions: dict[str, JevQuestion] = {
        unread_id(index, k): NoulModel(
            type="noul",
            instructions=STATED_WORD_INSTRUCTIONS.format(measure=desc, word=word),
            criteria=STATED_WORD_CRITERIA,
        )
        for k, word in enumerate(words)
    }
    return {"variable_phrase": variable}, questions


def unread_id(shape: int, word: int) -> str:
    """The ``unread|`` question id of word `word` of menu shape `shape`'s nominated
    words."""
    return f"{UNREAD_PREFIX}{shape}|{word}"


def _measures(desc: str) -> NoulModel:
    return {
        "type": "noul",
        "instructions": (
            f"Does the measure {desc}, or a part of it its slots select, measure the quantity "
            f"`variable_phrase` names? Ignore {_FORM_WORDS}, and ignore which dataset or "
            "source it comes from."
        ),
        "criteria": {
            "true": "It, or a part of it, measures the quantity the phrase names.",
            "false": "It measures a different quantity or a differently named indicator.",
        },
    }


def _bind_questions(menu: JevMenu, *, query: bool) -> dict[str, JevQuestion]:
    request = menu.request
    descs = [describe(s, m.fixed) for s, m in zip(request.shapes, menu.menus, strict=True)]
    q: dict[str, JevQuestion] = {}
    if descs:
        shapes = {f"Shape {i}": desc for i, desc in enumerate(descs)}
        q["elect"] = ChoiceModel(
            type="choice",
            instructions=(
                "Which measure best measures the quantity `variable_phrase` names, ignoring "
                "which dataset or source it comes from?"
            ),
            criteria=shapes,
        )
        if query:
            q["electq"] = ChoiceModel(
                type="choice",
                instructions=(
                    "`variable_phrase` is the measure the user asks for in `query`. Which "
                    "measure gives the user what they ask for?"
                ),
                criteria=shapes | {"none": "None of these measures what the user asks for."},
            )
    for i, desc in enumerate(descs):
        q[f"m|{i}"] = _measures(desc)
        if query:
            q[f"mq|{i}"] = NoulModel(
                type="noul",
                instructions=(
                    f"`variable_phrase` is the measure the user asks for in `query`. Would "
                    f"the measure {desc}, or a part of it its slots select, give the user "
                    f"what they ask for? Ignore {_FORM_WORDS}."
                ),
            )
        q[f"named|{i}"] = NoulModel(
            type="noul",
            instructions=(
                f"Does `variable_phrase` name the dataset or source the measure {desc} comes "
                "from, by its name or acronym (as 'CRS', 'DAC2a' or 'IDS')?"
            ),
        )
        words = request.nominated_words[i] if request.nominated_words else ()
        # The stated-word questions also quote what the measure's definition fixes, which the
        # menu's own description leaves out.
        stated = describe(request.shapes[i], menu.menus[i].fixed, definition=True) if words else ""
        for k, word in enumerate(words):
            q[unread_id(i, k)] = NoulModel(
                type="noul",
                instructions=STATED_WORD_INSTRUCTIONS.format(measure=stated, word=word),
                criteria=STATED_WORD_CRITERIA,
            )
    for row in measure_values(request.shapes):
        also = f" (also called {', '.join(row.synonyms)})" if row.synonyms else ""
        meaning = f" It means: {row.definition}" if row.definition else ""
        q[f"mv|{row.dcid}"] = NoulModel(
            type="noul", instructions=f"Does `variable_phrase` name {row.label}{also}?{meaning}"
        )
    for r in menu.refused:
        q[f"mr|{r}"] = _measures(describe(request.refused[r].shape))

    for prop, (label, values) in menu.properties.items():
        if prop[0] == "source" or not values:
            continue
        pid = prop_id(prop)
        q[f"part|{pid}"] = NoulModel(
            type="noul",
            instructions=(
                f"Does `variable_phrase` ask for one particular {label}, narrower than the "
                "measure it names as a whole?"
            ),
            criteria={
                "true": f"It names a part, subgroup or category of {label} the measure could "
                "be split by, as 'agriculture GDP' names a sector or 'female literacy' a sex.",
                "false": f"Any {label} its words imply is what the measure itself covers in "
                "full, as 'GDP' implies all economic activities and 'life expectancy' implies "
                "at birth.",
            },
        )
        for j, value in enumerate(asked_values(menu, prop)):
            shown = menu.labels.get(value, value)
            q[f"n|{pid}|{j}"] = NoulModel(
                type="noul",
                instructions=(
                    f"Does `variable_phrase` narrow the measure to the {label} '{shown}'?"
                ),
                criteria={
                    "true": (
                        "The phrase names this value, a synonym of it, or a narrower term "
                        "that implies it (as 'neonatal' implies under one month old)."
                    ),
                    "false": (
                        "The phrase says nothing about it, or shares a word with it only "
                        "inside a term that means something else (as 'grant' in 'grant "
                        "equivalent')."
                    ),
                },
            )
        if len(values) >= 2:
            q[f"several|{pid}"] = NoulModel(
                type="noul",
                instructions=(
                    f"Does `variable_phrase` name two or more distinct kinds of {label} "
                    "together, as 'health and education' names two sectors?"
                ),
            )
        for k, phrase in enumerate(request.dropped_qualifier_phrases):
            q[f"all|{pid}|{k}"] = NoulModel(
                type="noul",
                instructions=(
                    f"Does the phrase '{phrase}' ask for every value of {label}, as 'across "
                    "all flows' or 'regardless of flow type' do?"
                ),
            )
    for s, qmenu in enumerate(menu.menus):
        for g, rows in enumerate(qmenu.groups):
            q[f"tie|{s}|{g}"] = ChoiceModel(
                type="choice",
                instructions="Which of these values does `variable_phrase` name?",
                criteria={r.code_dcid: r.label for r in rows},
            )
    return q


def span_id(prop: Prop, value: str) -> str:
    """The ``span|`` question id of `value` on slot `prop`."""
    return f"span|{prop_id(prop)}|{value}"


def span_questions(
    menu: JevMenu, values: Sequence[tuple[Prop, str]], *, measure: int
) -> dict[str, JevQuestion]:
    """The span questions on `values`, each a slot and a value of it, and on the measure of
    menu shape `measure`: which words of the variable phrase name each. The options are the
    phrase's spans (`spans`) and `NONE`."""
    request = menu.request
    options: dict[str, str | None] = {
        **dict.fromkeys(spans(request.variable)),
        NONE: "No words of the phrase name it.",
    }
    shape = request.shapes[measure]
    reading = stated_reading(shape)
    q: dict[str, JevQuestion] = {
        SPAN_MEASURE: ChoiceModel(
            type="choice",
            instructions=(
                "Which words of `variable_phrase` name the quantity the measure "
                f"{describe(shape, menu.menus[measure].fixed)} counts, "
                f"{reading.measured_property} of {reading.population_type}, leaving out "
                "words that narrow it to one part? Choose the shortest span that names it."
            ),
            criteria=options,
        )
    }
    for prop, value in values:
        label = menu.properties[prop][0]
        shown = menu.labels.get(value, value)
        q[span_id(prop, value)] = ChoiceModel(
            type="choice",
            instructions=(
                f"Which words of `variable_phrase` name the {label} '{shown}'? Choose the "
                "shortest span that names it."
            ),
            criteria=options,
        )
    return q


def spans(variable: str) -> list[str]:
    """Every run of one to five words of `variable`, as it spells them, words split at
    whitespace, hyphens and slashes so "GDP" is a span of "debt-to-GDP", at most one fewer
    than a choice holds."""
    words = list(_WORD_RE.finditer(variable))
    found: dict[str, None] = {}
    for i in range(len(words)):
        for j in range(i, min(i + _MAX_SPAN_WORDS, len(words))):
            found[variable[words[i].start() : words[j].end()]] = None
    return list(found)[: _CHOICE_CAP - 1]


def span_occurrences(variable: str, span: str) -> tuple[frozenset[int], ...]:
    """Each occurrence of `span` in `variable`, as the positions of the words it covers,
    words split as `spans` splits them, in phrase order. Empty when `span` is no span of
    `variable`."""
    words = list(_WORD_RE.finditer(variable))
    return tuple(
        frozenset(range(i, j + 1))
        for i in range(len(words))
        for j in range(i, min(i + _MAX_SPAN_WORDS, len(words)))
        if variable[words[i].start() : words[j].end()] == span
    )


def word_positions(variable: str, start: int, end: int) -> frozenset[int]:
    """The positions of the words of `variable`, split as `spans` splits them, that the
    characters `start` to `end` touch."""
    return frozenset(
        k
        for k, word in enumerate(_WORD_RE.finditer(variable))
        if word.start() < end and start < word.end()
    )


def denominator_key(index: int, ref: GraphRef) -> str:
    """The ``denominators`` entry and ``q|denominator`` option of denominator `index`."""
    return f"{index}: {ref.label}"


def quantity_request(
    variable: str,
    denominators: Sequence[GraphRef],
    *,
    bind_variable: str,
    query: str | None,
) -> JevRequest:
    """The quantity request: the quantity read's questions on `variable`, the bind's
    variable phrase less its place words, over the numbered `denominators`.

    The state holds `bind_variable` as ``variable_phrase``, the query when given, the rules
    as ``quantity_rules``, the denominators when there are any, and `variable` as
    ``quantity_phrase`` when it differs from `bind_variable`; the questions read the
    phrase under that key."""
    state: dict[str, JSONContent] = {"variable_phrase": bind_variable}
    if query is not None:
        state["query"] = query
    state["quantity_rules"] = QUANTITY_RULES
    keys = [denominator_key(i, d) for i, d in enumerate(denominators)]
    if keys:
        state["denominators"] = list(keys)
    phrase = "variable_phrase"
    if variable != bind_variable:
        phrase = "quantity_phrase"
        state[phrase] = variable
    questions: dict[str, JevQuestion] = {
        "q|gate": NoulModel(
            type="noul",
            instructions=(
                f"Following `quantity_rules`, do the words of `{phrase}` themselves state a "
                "ratio, rate, share, percentage or per-unit form?"
            ),
        ),
        "q|form": ChoiceModel(
            type="choice",
            instructions=f"Following `quantity_rules`, which form do the words of `{phrase}` ask for?",
            criteria=dict(FORMS),
        ),
        "q|base": ChoiceModel(
            type="choice",
            instructions=(
                f"Following `quantity_rules`, which words of `{phrase}` name the base a ratio "
                "or share divides by? Choose the shortest span that names it."
            ),
            criteria={
                **dict.fromkeys(spans(variable)),
                NONE: "The phrase names no base, or only people or area.",
            },
        ),
    }
    if keys:
        questions["q|denominator"] = ChoiceModel(
            type="choice",
            instructions=f"Which entry of `denominators` is the base `{phrase}` divides by?",
            criteria={
                **dict.fromkeys(keys),
                NONE_OF_THESE: "The phrase names a base that no entry is.",
            },
        )
    return state, questions
