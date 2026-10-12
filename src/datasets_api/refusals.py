"""Render a Datasets refusal, `classify_datasets_refusal`.

A refusal renders from its own attributes (`dfo_platform.problem.classify_refusal`), with two
materialisation exceptions apart. Conversion refusals render through `conversion_errors`, since
their `cause` is a bare code. A `MixedGrainError` is a fault on this side, which
`datasets_api.failures` maps. Every door classifies refusals through this one function, so the
answer does not depend on which door's classifier runs first.
"""

from __future__ import annotations

from datasets_api.doors.conversion_errors import classify_conversion_exception
from dfo_platform import ProblemSpec
from dfo_platform.problem import classify_refusal
from materialise.errors import MixedGrainError


def classify_datasets_refusal(exc: BaseException) -> ProblemSpec | None:
    """Render `exc` if it is a refusal, else return `None`."""
    if isinstance(exc, MixedGrainError):
        return None
    return classify_conversion_exception(exc) or classify_refusal(exc)
