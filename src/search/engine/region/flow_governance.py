"""Whether a recalled flow family speaks for the query over the subject-only ones beside it.

Recall keeps families down to its relevance floor, so a flow family recalled for "GDP of
Ethiopia" at 0.61 enters the pool beside the GDP families above it. Such a family governs
the query only when it leads the ranking or its own recall score clears
``SEARCH_UNKNOWN_SCORE_THRESHOLD`` (`governs`). The pass reads it to decide whether a
role-bearing shape outranks the subject-only ones (`phase_rank_bind._role_bearing_outranks`),
and the raw-text door to decide whether an organisation can be a party to the measure's
flow (`source_choice.choose_reading`).
"""

from __future__ import annotations

from search.engine.config import settings


def governs(*, top: bool, score: float | None) -> bool:
    """Whether a role-bearing family governs the query: it is the top-ranked one (`top`), or
    its measured recall `score` is at least ``SEARCH_UNKNOWN_SCORE_THRESHOLD``. A family with
    no measured score governs only from the top."""
    return top or (score is not None and score >= settings().unknown_score_threshold)
