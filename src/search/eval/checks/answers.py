"""The answers a response part carries, and reads of each answer's identity.

An answer is a body holding ``definition.identity``: a definite part's spec, each spec a
candidates part offers, a no-data part's ``flagged_match`` (the elected measure, present on
some reasons), or an unsupported part's ``unsupported`` (its definition, with no spec). The
measure, entity, window and unit checks read the identity, so one path serves every status.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from search.eval.checks.base import (
    NA,
    PartView,
    Result,
    ScoreContext,
    all_of,
    any_of,
    join,
    read_path,
)


@dataclass(frozen=True)
class Answer:
    at: str
    """The schema path of ``body``."""
    body: Mapping[str, Any]
    is_spec: bool
    """False for an unsupported part's definition, which has no ``resolved`` half."""

    def read(self, path: str, ctx: ScoreContext) -> Any:
        return read_path(self.body, path, ctx, at=self.at)

    def identity(self, path: str, ctx: ScoreContext) -> Any:
        return self.read(f"definition.identity.{path}", ctx)


def answers(part: PartView, ctx: ScoreContext) -> list[Answer]:
    """The answers ``part`` carries. A no-data part without a flagged match has none."""
    at = PartView.path
    match part.status:
        case "definite":
            return [Answer(join(at, "spec"), read_path(part.body, "spec", ctx, at=at), True)]
        case "candidates":
            specs = read_path(part.body, "candidates.specs", ctx, at=at) or []
            return [Answer(join(at, "candidates.specs[]"), spec, True) for spec in specs]
        case "no_data":
            flagged = read_path(part.body, "no_data.flagged_match", ctx, at=at)
            return [Answer(join(at, "no_data.flagged_match"), flagged, True)] if flagged else []
        case "unsupported":
            body = read_path(part.body, "unsupported", ctx, at=at)
            return [Answer(join(at, "unsupported"), body, False)]
    return []


def over_answers(
    part: PartView,
    ctx: ScoreContext,
    assertion: Callable[[Answer], Result],
    *,
    reading: bool,
) -> Result:
    """``assertion`` over the answers ``part`` carries, n/a when it carries none.

    A candidates part offers several readings of one request: an assertion about the
    reading (``reading=True``: measure, provenance) holds when one candidate satisfies it,
    and an assertion about the request (entities, window, unit) must hold on every one.
    """
    found = answers(part, ctx)
    if part.status == "candidates":
        results = [assertion(answer).at(f"candidate {i}") for i, answer in enumerate(found)]
    else:
        results = [assertion(answer) for answer in found]
    if not results:
        return NA
    return any_of(results) if reading else all_of(results)
