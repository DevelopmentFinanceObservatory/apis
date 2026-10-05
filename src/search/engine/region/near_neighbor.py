"""Compute the retrieval margin and label evidence for a near-neighbour check.

`resolve_variable` applies the thresholds. This leaf module shares the weak-score
predicate with its retrieval warning so both use the same configured band.
"""

from __future__ import annotations

import re

from search.engine.config import settings
from search.engine.shape import ShapeDraft


def _representative_score_is_weak(score: float | None) -> bool:
    """True when the score is above the relevance floor and below the weak-score threshold.

    A missing score carries no evidence of weakness. The retrieval warning and near-neighbour gate
    share this predicate and configured band.
    """
    return (
        score is not None
        and settings().relevance_threshold < score < settings().weak_score_threshold
    )


# Word-character runs only -- strips surrounding punctuation (parens, commas, '%', ...)
# so a label like "... (per person)" tokenizes to {"per", "person"}, not
# {"(per", "person)"}, and compares correctly against a sibling's plain-text
# "... per person". Shared by _label_token_similarity and _discriminating_token so
# both signals use one tokenizer.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _shapes_semantically_distinct(a: ShapeDraft, b: ShapeDraft) -> bool:
    """True when measuredProperty or populationType differs between the shapes."""
    return a.meas_prop_dcid != b.meas_prop_dcid or a.pop_type_dcid != b.pop_type_dcid


def _best_near_neighbor_index(
    chosen: ShapeDraft,
    chosen_index: int,
    capped_shapes: list[ShapeDraft],
) -> int | None:
    """Index of the highest-scoring distinct candidate with a retrieval score, or None.

    `capped_shapes` is score-descending, so the first match is the best neighbour. Shapes
    without a score provide no measured rival evidence and are skipped.
    """
    for i, cand in enumerate(capped_shapes):
        if i == chosen_index:
            continue
        if cand.representative_score is None:
            continue
        if not _shapes_semantically_distinct(chosen, cand):
            continue
        return i
    return None


def _label_token_similarity(a: str, b: str) -> float:
    """Case-insensitive Jaccard similarity of punctuation-stripped token sets.

    Return zero if either set is empty.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _discriminating_token(query: str, chosen_label: str, neighbor_label: str) -> str | None:
    """Return the first query token unique to `chosen_label`, or None if the query also names
    a neighbour-only token.

    This symmetric test distinguishes a real choice from wording that invokes both concepts.
    Tokenisation matches `_label_token_similarity`, including punctuation stripping, and the
    lexicographically first match makes the result deterministic.
    """
    query_tokens = _tokenize(query)
    chosen_tokens = _tokenize(chosen_label)
    neighbor_tokens = _tokenize(neighbor_label)
    if query_tokens & (neighbor_tokens - chosen_tokens):
        return None
    matches = query_tokens & (chosen_tokens - neighbor_tokens)
    return min(matches) if matches else None


def _near_neighbor_check(
    variable: str,
    chosen: ShapeDraft,
    chosen_index: int,
    capped_shapes: list[ShapeDraft],
) -> tuple[int, float, float] | None:
    """Return `(neighbour_index, margin, label_similarity)` for the caller to threshold.

    Return None when the chosen shape lacks a score, has a weak score, or has no scored,
    semantically distinct neighbour. A weak chosen score makes a small margin uninformative.
    The gate applies the same margin rule to cross-domain and same-population neighbours.
    """
    if chosen.representative_score is None:
        return None
    if _representative_score_is_weak(chosen.representative_score):
        return None
    neighbor_index = _best_near_neighbor_index(chosen, chosen_index, capped_shapes)
    if neighbor_index is None:
        return None
    neighbor = capped_shapes[neighbor_index]
    assert neighbor.representative_score is not None  # _best_near_neighbor_index skips None
    margin = chosen.representative_score - neighbor.representative_score
    # Compare against the displayed concept label, which may differ from the scored member.
    similarity = _label_token_similarity(variable, chosen.label)
    return neighbor_index, margin, similarity


NEAR_NEIGHBOR_UNRESOLVED_MESSAGE = (
    "the chosen reading and a semantically distinct alternative scored closely enough to be "
    "ambiguous; the ambiguity stands because another reading could not be checked."
)
"""The `near_neighbor_ambiguous` message of a tie that stays stated while an alternate is
unevaluated (`ResolveState.tie_unresolved`) and fewer than two tied readings are served."""
