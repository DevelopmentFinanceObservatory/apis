"""The content words of a text, those that carry its meaning, for comparing two phrasings."""

from __future__ import annotations

import string

FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "per",
        "by",
        "with",
        "as",
        "at",
        "from",
    }
)
"""The closed list of connectives that carry no meaning of their own. Compared
case-folded."""


def content_words(text: str) -> frozenset[str]:
    """The case-folded words of `text`, edge punctuation stripped, without function words.

    Words split on whitespace only, so "Core/general" stays one word and "(ODA)" reads
    as "oda".
    """
    words: set[str] = set()
    for word in text.lower().split():
        core = word.strip(string.punctuation)
        if core and core not in FUNCTION_WORDS:
            words.add(core)
    return frozenset(words)
