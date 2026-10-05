# Frozen lexical-scanner baseline used by the direction-corpus evaluation.
# Evaluated through `scripts/eval_role_direction.py`.
#
# Defines its own role-axis type and uses only standard-library imports.
# imported RoleAxis from search.models, which this module inlines below so
# it depends on nothing outside the standard library and can be loaded from
# a clone with no extras installed.

"""Maps query text to lexical flow-direction evidence.

This pure leaf supplies parsed `role_text` and directional-preposition logic
for both `entry/parsed.py` and `region/pipeline.py`, which can import it
without depending on each other."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

type RoleAxis = Literal["origin", "destination"]

# Warning code emitted when a caller-supplied ParsedEntity.role_text does not
# match any entry in _ROLE_TEXT_TO_PREPOSITION below (minted by entry/parsed.py, the
# only caller that attaches it to that entity's own part's warnings list).
ROLE_TEXT_NOT_RECOGNISED = "role_text_not_recognised"


# Single-entity directional-preposition evidence
#
# A directional preposition on a resolved entity makes the pipeline refuse
# when the recalled shape has no matching role.

_ADJACENCY_WORDS = 3
_DONOR_PREPS: frozenset[str] = frozenset({"from"})
_RECIPIENT_PREPS: frozenset[str] = frozenset({"to"})
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]")

# Coordination separators that tokenize between entity anchors. Commas produce no token.
_SEPARATOR_WORDS: frozenset[str] = frozenset({"and", "or"})

# Maps English, French, and Spanish role labels to `from` or `to`.
# Unknown labels, including `None`, follow the ambiguous path. This is an
# explicit word list, with no inferred synonyms.
_ROLE_TEXT_TO_PREPOSITION: dict[str, str] = {
    # English
    "from": "from",
    "donor": "from",
    "provider": "from",
    "funder": "from",
    "source": "from",
    "to": "to",
    "recipient": "to",
    "beneficiary": "to",
    "destination": "to",
    # French
    "de": "from",
    "donateur": "from",
    "bailleur": "from",
    "fournisseur": "from",
    "à": "to",
    "destinataire": "to",
    "bénéficiaire": "to",
    # Spanish entries. `de` is shared with French.
    "donante": "from",
    "proveedor": "from",
    "a": "to",
    "destinatario": "to",
    "beneficiario": "to",
    "destino": "to",
}


def normalise_role_text(role_text: str | None) -> str | None:
    """Maps a role label to `from` or `to`. Unknown text maps to `None`.

    The caller inserts the preposition into synthesized query text for lexical scanning."""
    if role_text is None:
        return None
    return _ROLE_TEXT_TO_PREPOSITION.get(role_text.casefold())


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def detect_direction(query: str, surface: str | None, dcid: str) -> RoleAxis | None:
    """Returns `origin`, `destination`, or `None` for one resolved entity.

    Searches the surface text first, then the dcid's last path segment. At its
    first anchor occurrence, scans the preceding three tokens for the nearest
    `from` or `to`. `None` means no lexical evidence. The caller decides how
    to handle it.
    """
    last_segment = dcid.rsplit("/", 1)[-1]
    raw_anchors: list[str | None] = [surface, last_segment]

    anchors: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in raw_anchors:
        if not raw:
            continue
        tokens = _tokenize(raw)
        key = tuple(tokens)
        if tokens and key not in seen:
            anchors.append(tokens)
            seen.add(key)

    if not anchors:
        return None

    query_tokens = _tokenize(query)
    n = len(query_tokens)

    for anchor_tokens in anchors:
        anchor_len = len(anchor_tokens)
        for i in range(n - anchor_len + 1):
            if query_tokens[i : i + anchor_len] != anchor_tokens:
                continue
            window_start = max(0, i - _ADJACENCY_WORDS)
            window = query_tokens[window_start:i]
            for prep in reversed(window):  # nearest preposition wins
                if prep in _DONOR_PREPS:
                    return "origin"
                if prep in _RECIPIENT_PREPS:
                    return "destination"
    return None


# Coordination-aware direction evidence across a named-entity list
#
# Resolves coordinated entity mentions when later names fall outside
# `detect_direction`'s three-token window. Direction inherits from the
# nearest preceding entity across separator-only gaps.


def _entity_anchor_tokens(surface: str | None, dcid: str) -> list[list[str]]:
    """Tokenised surface and dcid anchors, in matching priority order."""
    last_segment = dcid.rsplit("/", 1)[-1]
    raw_anchors: list[str | None] = [surface, last_segment]

    anchors: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in raw_anchors:
        if not raw:
            continue
        tokens = _tokenize(raw)
        key = tuple(tokens)
        if tokens and key not in seen:
            anchors.append(tokens)
            seen.add(key)
    return anchors


def _first_anchor_span(query_tokens: list[str], anchors: list[list[str]]) -> tuple[int, int] | None:
    """First anchor span in query-token order, or `None` if no anchor appears."""
    n = len(query_tokens)
    for anchor_tokens in anchors:
        anchor_len = len(anchor_tokens)
        for i in range(n - anchor_len + 1):
            if query_tokens[i : i + anchor_len] == anchor_tokens:
                return i, i + anchor_len
    return None


def _adjacent_preposition(query_tokens: list[str], span_start: int) -> RoleAxis | None:
    """Returns the nearest direction in the three tokens before an anchor."""
    window_start = max(0, span_start - _ADJACENCY_WORDS)
    window = query_tokens[window_start:span_start]
    for prep in reversed(window):  # nearest preposition wins
        if prep in _DONOR_PREPS:
            return "origin"
        if prep in _RECIPIENT_PREPS:
            return "destination"
    return None


def _direct_direction(query_tokens: list[str], anchors: list[list[str]]) -> RoleAxis | None:
    """Scans every anchor occurrence in order, preferring surface to dcid.

    Returns the first direction found, including evidence from a later mention when
    the first mention has none."""
    n = len(query_tokens)
    for anchor_tokens in anchors:
        anchor_len = len(anchor_tokens)
        for i in range(n - anchor_len + 1):
            if query_tokens[i : i + anchor_len] != anchor_tokens:
                continue
            direction = _adjacent_preposition(query_tokens, i)
            if direction is not None:
                return direction
    return None


def detect_directions(query: str, entities: Sequence[tuple[str, str]]) -> list[RoleAxis | None]:
    """Returns one direction per `(surface, dcid)` pair, in input order.

    Direct evidence scans every anchor occurrence, surface before dcid, and
    uses the first occurrence with a nearby preposition. Otherwise an entity
    inherits from the nearest query-preceding resolved entity when the gap
    contains only `and`/`or` tokens or no tokens (commas tokenize to none).
    Inheritance follows query order and can chain. Only query-anchored entities
    can source inheritance. Others stay `None`.
    """
    query_tokens = _tokenize(query)
    # A -1 span marks an entity without a query anchor; it stays in `results`.
    positioned: list[tuple[int, int, int, RoleAxis | None]] = []
    for idx, (surface, dcid) in enumerate(entities):
        anchors = _entity_anchor_tokens(surface, dcid)
        span = _first_anchor_span(query_tokens, anchors) if anchors else None
        if span is None:
            positioned.append((-1, -1, idx, None))
            continue
        positioned.append((span[0], span[1], idx, _direct_direction(query_tokens, anchors)))

    results: list[RoleAxis | None] = [None] * len(entities)
    last_direction: RoleAxis | None = None
    last_span_end: int | None = None
    for span_start, span_end, idx, direct in sorted(
        (p for p in positioned if p[0] >= 0), key=lambda p: p[0]
    ):
        if direct is not None:
            resolved = direct
        elif (
            last_direction is not None
            and last_span_end is not None
            and all(token in _SEPARATOR_WORDS for token in query_tokens[last_span_end:span_start])
        ):
            resolved = last_direction
        else:
            resolved = None
        results[idx] = resolved
        if resolved is not None:
            last_direction = resolved
            last_span_end = span_end
    return results
