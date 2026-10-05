"""Detect a likely entity name in parsed variable text.

The parsed door uses this advisory heuristic when no entities array is given.
It flags Title-Case words after the first. Acronyms and sentence-initial words
are ignored. Without a place-name lexicon, capitalised common nouns can trigger
false positives. The original text stays unchanged for recall scoring.
"""

from __future__ import annotations

import re

# Warning for Title-Case words after the first when parsed input lacks entities.
ENTITY_LIKE_PHRASING = "variable_text_entity_like"

_TITLE_CASE_RE = re.compile(r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'\-]*$")
_STRIP_CHARS = ".,;:!?()\"'’"


def detect_entity_like_phrasing(text: str) -> str | None:
    """Return the first Title-Case word after the first, or `None`.

    Ignoring the first word avoids flagging ordinary sentence capitalisation.
    """
    words = text.split()
    for word in words[1:]:
        stripped = word.strip(_STRIP_CHARS)
        if stripped and _TITLE_CASE_RE.fullmatch(stripped):
            return stripped
    return None
