"""The mint door's exception classification, `classify_mint_exception`. It checks failures
that are not refusals (`datasets_api.failures`) first, then refusals (`datasets_api.refusals`)."""

from __future__ import annotations

from datasets_api.backpressure import Backpressure
from datasets_api.failures import classify_failure
from datasets_api.refusals import classify_datasets_refusal
from dfo_platform import ProblemSpec


class MintBackpressure(Backpressure):
    """Mint or shared-materialisation capacity is full, including the definition door's
    shared gate."""

    code = "service_unavailable"

    detail = "the mint door is at its concurrent-materialisation capacity right now; retry"


def classify_mint_exception(exc: BaseException) -> ProblemSpec | None:
    return classify_failure(exc) or classify_datasets_refusal(exc)
