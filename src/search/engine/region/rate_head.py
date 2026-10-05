"""Match labels to the concept named by an explicit rate request.

`pipeline.py` uses matches to prune raw counts. `availability.py` pairs shapes
with a shared rate head. This leaf module depends only on `ShapeDraft` to avoid a
cycle between those callers.
"""

from __future__ import annotations

import re

from search.engine.shape import ShapeDraft

_RATE_HEAD_RE = re.compile(
    r"\b(?P<head>[a-z0-9]+)\s+(?:rates?|proportions?|percentages?|percent)\b"
    r"|\b(?P<percent_head>[a-z0-9]+)\s+%",
    re.IGNORECASE,
)
_SEMANTIC_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _rate_head(variable: str) -> str | None:
    """Return the word before an explicit rate term, or `None` for other forms."""
    match = _RATE_HEAD_RE.search(variable)
    if match is None:
        return None
    return (match.group("head") or match.group("percent_head")).lower()


def _labels_share_rate_head(head: str, label: str) -> bool:
    """Match exact tokens or six-character prefixes on words of six or more.

    This captures common inflections without matching short generic terms.
    """
    for token in _SEMANTIC_TOKEN_RE.findall(label.lower()):
        if token == head:
            return True
        if len(head) >= 6 and len(token) >= 6 and token[:6] == head[:6]:
            return True
    return False


def _rate_head_compatible(shape: ShapeDraft, head: str) -> bool:
    """Return whether a normalized candidate measures the raw-count anchor."""
    return _labels_share_rate_head(head, shape.label)
