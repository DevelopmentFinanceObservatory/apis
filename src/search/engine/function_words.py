"""The closed lists of words that name nothing a query restricts.

Extraction reads them when it restores a variable's word order and when it judges a
dropped qualifier, and the qualifier coverage rule (`phase_rank_bind._coverage_records`)
reads `FUNCTION_WORDS`, `AMOUNT_WORDS`, `DATA_WORDS` and `FLOW_VERBS` as words no
reader needs to cover, `RATIO_WORDS` on a shape that divides by a denominator,
`COUNT_WORDS` on a shape that measures a count, and `PERSON_WORDS` on one that counts
persons. `NEGATION_CUE` finds the words that negate or exclude what follows them.
"""

from __future__ import annotations

import re

# Words that only join a measure's words ("rate of extreme poverty"). The flow cues
# "to", "from" and "by" stay out, so restoring a variable's order carries no cue into
# its text. "and", "or" and "in" stay out too, since they join two measures or a measure
# and its scope ("unemployment rate and youth unemployment").
JOINING_WORDS = frozenset(
    {"a", "an", "the", "of", "who", "which", "that", "is", "are", "was", "were"}
)

# Prepositions that name nothing a query restricts.
JOINING_PREPOSITIONS = frozenset(
    {
        "of", "in", "to", "from", "by", "for", "on", "at", "with", "within", "as", "per",
        "than", "into", "over", "across", "since", "between",
    }
)  # fmt: skip

# Prepositions that restrict what they attach ("aid excluding debt relief").
RESTRICTING_PREPOSITIONS = frozenset(
    {
        "excluding", "including", "except", "without", "under", "about", "among", "through",
        "towards", "toward", "via", "against", "during", "outside", "beyond",
    }
)  # fmt: skip

# Words a phrase can hold without naming anything a query restricts: the joining words,
# joining prepositions, pronouns, auxiliaries and the possessive "s".
FUNCTION_WORDS = (
    JOINING_WORDS
    | JOINING_PREPOSITIONS
    | frozenset(
        {
            "and", "or", "do", "does", "did", "be", "been", "has", "have", "had", "what",
            "how", "i", "me", "my", "you", "your", "we", "our", "it", "its", "this", "these",
            "those", "there", "s",
        }
    )
)  # fmt: skip

# Nouns that name the quantity a measure counts and nothing about which flows it counts:
# "health spending", "imputed multilateral flows", "CRS figures". A word that restricts
# the flows ("private", "grant", "domestic") stays out.
AMOUNT_WORDS = frozenset(
    {
        "spending", "expenditure", "funding", "financing", "total", "figures", "amount",
        "amounts", "money", "flows",
    }
)  # fmt: skip

# Nouns that name data itself and nothing about what it measures: "health data",
# "education statistics", "aid figures by indicator". A subject question names its subject
# beside them.
DATA_WORDS = frozenset({"data", "statistics", "numbers", "information", "indicators", "series"})

# Verbs that say a flow moved or was reported and nothing about which flows: "net ODA
# received by Kenya", "ODA going to Kenya", "total ODA reported to the DAC", "external
# debt Kenya owes China". The place's role states the direction. "committed" stays out:
# commitments are a different measure.
FLOW_VERBS = frozenset(
    {
        "received", "receive", "receives", "receiving", "given", "provided", "disbursed",
        "spent", "going", "goes", "went", "reported", "reports", "owe", "owes", "owed", "lent",
    }
)  # fmt: skip

# Nouns that say a measure is divided by another: "ODA as a share of GNI", "ODA GNI ratio".
# A shape with a denominator states them; on an amount shape they restrict.
RATIO_WORDS = frozenset({"share", "ratio", "percent", "percentage", "proportion"})

# Nouns that say a measure counts what it names: "number of refugees" is their count. A
# shape measuring a count states them.
COUNT_WORDS = frozenset({"number", "numbers"})

# Nouns that say a measure counts people: "refugee population" is the number of refugees.
# Only a count of persons states them. On a count of households, "people in urban
# households" leaves them to the binder.
PERSON_WORDS = frozenset({"population", "populations", "people", "persons"})

# A "no" that opens the text or a sentence and is closed by a clause mark, a dash or ")"
# answers what came before ("No, count it including imputed multilateral", "No. Show gross
# ODA", "No - show gross ODA"), so it negates nothing after the mark; so does a "never"
# closed by a sentence-final mark. A discourse opener may precede it ("Hmm, no, ..."). A
# "not" is never a reply: "not, as before, multilateral" still negates, as does a negator
# inside a sentence and a "no-" compound. `INTERJECTION` finds such a reply, and
# `NEGATION_CUE` and `bind_repair._label_negated_in_text` pass over it.
NEGATORS = ("not", "no", "never")
_DISCOURSE_OPENERS = ("ok", "okay", "hmm", "oh", "ah", "well", "sorry", "actually")
_CLAUSE_CLOSE = r"(?:\s*[,.;:!?)\u2013\u2014]|\s*-(?=\s)|\s*--)"
_SENTENCE_CLOSE = r"\s*[.!?]"
_SENTENCE_STARTS = (r"\A", r"\A\s", r"[.!?]", r"[.!?]\s", r"[.!?]\s\s")
# Fixed-width lookbehinds, one per way a sentence can start before the reply word.
_OPENING_LOOKBEHINDS = (
    *_SENTENCE_STARTS,
    *(
        f"{start}{word}{gap}"
        for start in _SENTENCE_STARTS[:2] + _SENTENCE_STARTS[3:4]
        for word in _DISCOURSE_OPENERS
        for gap in (r"\s", r",\s")
    ),
)
_OPENS = "(?:^|" + "|".join(f"(?<={look})" for look in _OPENING_LOOKBEHINDS) + ")"
_NOT_OPENS = "".join(f"(?<!{look})" for look in _OPENING_LOOKBEHINDS)
INTERJECTION = re.compile(
    rf"{_OPENS}(?:no\b(?={_CLAUSE_CLOSE})|never\b(?={_SENTENCE_CLOSE}))", re.IGNORECASE
)

# The words that negate or exclude what follows: an exclusion read starts each stretch it
# leaves out at one (`exclusion_read._pieces`), a scope phrase's origin starts at the one
# before it (`stated_scope._cue_start`), and a measure value's name after one in its stretch
# names nothing (`qualifier_match.measure_values_named`). An interjection (`INTERJECTION`)
# is no cue.
NEGATION_CUE = re.compile(
    rf"\b(?:not\b|{_NOT_OPENS}(?:no|never)\b|no\b(?!{_CLAUSE_CLOSE})|never\b(?!{_SENTENCE_CLOSE})|"
    r"non|excluding|exclude|excludes|except|without|other\s+than|rather\s+than|excl)\b",
    re.IGNORECASE,
)
