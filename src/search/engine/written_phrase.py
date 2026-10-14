"""The stretch of a request that writes a variable phrase extraction read from it."""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+(?:-\w+)*")


def request_words(text: str) -> list[str]:
    """The lowercased words of `text`, in order. A hyphenated compound is one word with its
    hyphens removed, so "non-concessional" reads as "nonconcessional" and holds no
    "concessional"."""
    return [word.replace("-", "") for word in _WORD.findall(text.lower())]


def written_phrase(variable: str, query_text: str | None) -> str | None:
    """The words of `query_text` from the first to the last that `variable` holds, joined by
    spaces: the request's own spelling of a phrase extraction may reorder. "net flows on
    external debt for Kenya", read as "external debt net flows", writes "net flows on
    external debt". `None` without a request text, or where it holds no word of the phrase.
    """
    if query_text is None:
        return None
    asked = set(request_words(variable))
    words = request_words(query_text)
    held = [position for position, word in enumerate(words) if word in asked]
    if not held:
        return None
    return " ".join(words[held[0] : held[-1] + 1])
