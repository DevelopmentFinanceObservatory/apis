"""The closed lists of words that name nothing a query restricts.

Extraction reads them when it restores a variable's word order and when it judges a
dropped qualifier, and the qualifier coverage rule (`phase_rank_bind._coverage_records`)
reads `FUNCTION_WORDS`, `AMOUNT_WORDS`, `DATA_WORDS` and `FLOW_VERBS` as words no
reader needs to cover, `RATIO_WORDS` on a shape that divides by a denominator,
and `COUNT_WORDS` on a shape that measures a count.
"""

from __future__ import annotations

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

# Nouns that say a measure counts what it names: "refugee population" is the number of
# refugees, "number of refugees" their population. A shape measuring a count states them.
COUNT_WORDS = frozenset({"number", "numbers", "population", "populations", "people", "persons"})
