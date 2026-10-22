"""Scores the 39-query direction corpus against recorded extraction fixtures.

`raw_extraction` uses recorded `Extraction.entities`; `guarded_extraction`
passes the same entities and their variables through `_validate_role_spans`,
as `extract()` does. Both implement `eval_role_direction.py`'s scanner
interface, so their accuracy difference measures the guard's effect on the
same model output.

Fixtures in `tests/fixtures/llm_responses.json` use the key
`fixture_key(Extraction, system, query)`, where `system` is the
extraction prompt template, its year left as `[[YEAR]]`. A missing fixture exits as a setup
error and, when `HARVEST_MISS_PATH_ENV` is set, is logged as a harvest miss so
`scripts/harvest_llm_fixtures.py` can record it. Scoring runs offline, without
model or graph calls.

Labels match entities by `span`, then `text`; unmatched labels return
`flow_role="unspecified"`. The scanner's `dcid` input is unused because
recorded extractions carry no dcid.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Literal

import search.engine.extract as extract
from search.engine.extract import ExtractedEntity, Extraction, _validate_role_spans
from tests.fixtures import HARVEST_MISS_PATH_ENV, _append_harvest_line
from tests.fixtures.keys import fixture_key

REPO = Path(__file__).parent.parent.parent
FIXTURES_PATH = REPO / "tests" / "fixtures" / "llm_responses.json"

Axis = Literal["origin", "destination"]


@cache
def _bank() -> dict:
    return json.loads(FIXTURES_PATH.read_text())


def _fixture_key(query: str) -> str:
    return fixture_key(Extraction, extract._EXTRACTION_SYSTEM_PROMPT, query)


def _recorded_extraction(query: str) -> Extraction:
    raw = _bank().get(_fixture_key(query))
    if raw is None:
        harvest_path = os.environ.get(HARVEST_MISS_PATH_ENV)
        if harvest_path:
            _append_harvest_line(
                harvest_path,
                outcome="miss",
                key=_fixture_key(query),
                schema_name=Extraction.__name__,
                system=extract._EXTRACTION_SYSTEM_PROMPT,
                prompt=query,
            )
        sys.exit(
            f"corpus/direction/recorded_extraction.py: no recorded Extraction "
            f"fixture for query {query!r} (key {_fixture_key(query)!r}); re-run "
            f"scripts/harvest_llm_fixtures.py's direction-corpus collector"
        )
    return Extraction.model_validate(raw)


def _axis_for(entities: Sequence[ExtractedEntity], surface: str) -> Axis | None:
    for entity in entities:
        if (entity.span or "") == surface or entity.text == surface:
            return entity.flow_role if entity.flow_role != "unspecified" else None
    return None


def raw_extraction(query: str, entities: Sequence[tuple[str | None, str]]) -> list[Axis | None]:
    """Scans the recorded `Extraction.entities` directly."""
    recorded = _recorded_extraction(query).entities
    return [
        _axis_for(recorded, surface) if surface is not None else None for surface, _dcid in entities
    ]


def guarded_extraction(query: str, entities: Sequence[tuple[str | None, str]]) -> list[Axis | None]:
    """Scans recorded entities after `_validate_role_spans`, passing their variables
    as `extract()` does."""
    recorded_extraction = _recorded_extraction(query)
    recorded = _validate_role_spans(
        query, recorded_extraction.entities, recorded_extraction.variables
    )
    return [
        _axis_for(recorded, surface) if surface is not None else None for surface, _dcid in entities
    ]
