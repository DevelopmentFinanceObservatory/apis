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


class MintBackgroundDeclined(MintBackpressure):
    """A background mint (`datasets_api.background_admission`) met a limit a foreground mint
    would wait on or be refused by, or one it could take only by leaving no place for
    foreground callers, and was refused with nothing queued. It answers 503
    `service_unavailable` with `constraint_id` `mint_background_declined`."""

    constraint_id = "mint_background_declined"

    detail = (
        "a background mint is admitted only without waiting and only while it leaves capacity "
        "for other callers, and it could not be right now; nothing was queued"
    )


def classify_mint_exception(exc: BaseException) -> ProblemSpec | None:
    return classify_failure(exc) or classify_datasets_refusal(exc)
