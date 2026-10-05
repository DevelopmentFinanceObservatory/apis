"""EntryParams: the prologue-computed values that more than one
ResolveInput.kind entry module reads. Built once by
search.engine.core._resolve_pipeline_events before it dispatches on
inp.kind, and threaded unchanged into entry.raw_text, entry.parsed and
entry.dataset_resubmit.
"""

from __future__ import annotations

from dataclasses import dataclass

from series_coverage.shared import CoverageSnapshot
from skills_index import SkillsIndex


@dataclass(frozen=True)
class EntryParams:
    # Shared monotonic deadline for extraction, resubmit, fan-out, and combining.
    # It uses the same clock as now_ms(), so every stage compares it directly.
    deadline: float

    # From request.response_options.include_sentence. Defaults to True when unset.
    include_sentence: bool

    # Shared candidate cap for raw_text and dataset_resubmit.
    max_candidates: int

    # Place-as-constraint flag uses the config default across parsed and raw-text inputs.
    pac: bool

    # SSE requests with coverage enabled stream trailing slot-options events,
    # keeping `response` independent of the probe. Buffered responses embed coverage.
    stream_option_coverage: bool
    include_option_coverage: bool

    # Unset request options resolve to false. `include_observations` feeds warnings.
    # `stream_observations` selects trailing SSE events or inline buffered data.
    stream_observations: bool
    include_observations: bool

    # The request option defaults to false. `skills_index` is the request's shared
    # snapshot, or `None` when skills are off or the index has no initial build.
    include_skills: bool
    skills_index: SkillsIndex | None

    # Single coverage snapshot shared by consumers and the cache key, keeping its
    # index and stamp from the same artifact generation. Required at construction.
    coverage: CoverageSnapshot
