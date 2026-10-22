"""Splitting of one Jev request that is too large to send whole.

Jev judges each question from the state and that question alone, so a request split into
several that share its state asks the same questions of the same state, and only Jev's
run-to-run noise can move an answer. `split_questions` keeps a request whole when its
estimate is within the budget, else splits its questions into the fewest chunks that each
fit with the state, balanced by size. Jev refuses a request over 64k tokens and a question
over 32k; the 24k budget keeps every chunk well inside both.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from typesafe_sdk import JSONContent

# Estimated input tokens per character of the JSON body. On the dev first requests Jev's
# input tokens ran 0.235 to 0.378 per character (p50 0.26); 0.3 errs towards more chunks.
TOKENS_PER_CHAR = 0.3
SPLIT_BUDGET = 24_000


def estimate_tokens(obj: object) -> int:
    """Estimated input tokens of `obj` as JSON."""
    return math.ceil(len(json.dumps(obj)) * TOKENS_PER_CHAR)


class RequestTooLarge(ValueError):
    """A Jev state, or a state with one of its questions, over the request budget, which no
    split can send."""


def split_questions[Q](
    state: JSONContent, questions: Mapping[str, Q], *, budget: int = SPLIT_BUDGET
) -> list[dict[str, Q]]:
    """`questions` in the fewest chunks whose estimate with `state` is within `budget`.

    Questions go to the least-loaded chunk, largest first, so the chunks come out of nearly
    equal size; each chunk keeps the original question order.

    Raises:
        RequestTooLarge: The state alone, or the state with one question, exceeds
            `budget`.
    """
    base = estimate_tokens({"state": state, "questions": {}})
    sizes = {qid: estimate_tokens({qid: q}) for qid, q in questions.items()}
    total = sum(sizes.values())
    if base + total <= budget:
        return [dict(questions)]
    room = budget - base
    if room <= 0 or max(sizes.values()) > room:
        raise RequestTooLarge(
            f"a Jev state of {base} estimated tokens or one of its questions exceeds the "
            f"{budget}-token request budget"
        )
    largest_first = sorted(sizes, key=sizes.__getitem__, reverse=True)
    k = math.ceil(total / room)
    while True:
        bins: list[list[str]] = [[] for _ in range(k)]
        load = [0] * k
        for qid in largest_first:
            i = load.index(min(load))
            bins[i].append(qid)
            load[i] += sizes[qid]
        if max(load) <= room:
            break
        k += 1
    order = {qid: n for n, qid in enumerate(questions)}
    return [{qid: questions[qid] for qid in sorted(b, key=order.__getitem__)} for b in bins]
