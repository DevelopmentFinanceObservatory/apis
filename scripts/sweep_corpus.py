"""Run every corpus case offline through FakeGraph and FakeLLM.

The sweep scores nothing. It exists so `scripts/harvest_llm_fixtures.py` can collect the
LLM fixture keys the corpus reaches: each case posts its request to the engine, and a
fixture miss lands in the harvest file. A refusal (`EngineInputError`) counts as completed.

The script always exits 0. When `SEARCH_SWEEP_CORPUS_OUTCOME_PATH` is set, it writes
attempted/completed counts and the ids of cases that raised.

Usage:
    uv run python scripts/sweep_corpus.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ENGINE_ROOT / "src"))
sys.path.insert(0, str(_ENGINE_ROOT))

from entity_resolution import (  # noqa: E402
    EntityResolver,
    SharedResolver,
    acquire_shared,
    release_shared,
)
from search.engine.core import resolve_async  # noqa: E402
from search.engine.errors import EngineInputError  # noqa: E402
from search.eval.cases import Case, load_cases  # noqa: E402
from search.eval.live_clients import bind_run_catalogue  # noqa: E402
from search.eval.runner import build_request  # noqa: E402
from search.models import ResolveRequest  # noqa: E402
from tests.engine._fake_sdmx import FakeSdmxClient  # noqa: E402
from tests.engine._harness import _FAKE_GRAPH, _FAKE_LLM, PINNED_DATE  # noqa: E402

OUTCOME_PATH_ENV = "SEARCH_SWEEP_CORPUS_OUTCOME_PATH"
"""Names the file main() writes its own machine-readable outcome summary to,
when set -- see the module docstring. Unset by default, which makes the
write a no-op for an ordinary direct run of this script."""

OUTCOME_SCHEMA_VERSION = 1
"""The outcome summary's layout version. A reader refuses a summary at a
version it does not know, so a layout change here is a loud failure there
rather than a silently misread field."""

# One fixture-backed client serves every case without live SDMX calls.
_FAKE_SDMX = FakeSdmxClient()


def _acquire_resolver() -> tuple[SharedResolver, EntityResolver]:
    """Acquire one offline resolver for the sweep, caching its pack under `runs/`."""
    os.environ["RESOLVEKIT_OFFLINE"] = "1"
    if not os.environ.get("RESOLVEKIT_CACHE_DIR"):
        cache_dir = _ENGINE_ROOT / "runs" / "sweep_corpus_resolvekit"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RESOLVEKIT_CACHE_DIR"] = str(cache_dir)
    shared = acquire_shared()
    return shared, EntityResolver(graph=_FAKE_GRAPH, shared=shared)


def _run_one(case: Case, resolver: EntityResolver):
    request = ResolveRequest.model_validate(build_request(case))
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = None
        return asyncio.run(
            resolve_async(
                request, graph=_FAKE_GRAPH, resolver=resolver, llm=_FAKE_LLM, sdmx=_FAKE_SDMX
            )
        )


def _write_outcome_summary(*, attempted: int, completed: int, failures: list[dict]) -> None:
    """Write attempted/completed counts and failed case ids when the output path is set."""
    outcome_path = os.environ.get(OUTCOME_PATH_ENV)
    if not outcome_path:
        return
    summary = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "attempted": attempted,
        "completed": completed,
        "failures": failures,
    }
    Path(outcome_path).write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    cases = load_cases()
    errors: list[tuple[str, str, str]] = []

    bind_run_catalogue()
    shared, resolver = _acquire_resolver()
    try:
        for case in cases:
            try:
                _run_one(case, resolver)
            except EngineInputError:
                # A refusal is an answer; it reaches no LLM call the harvest is missing.
                continue
            except Exception as exc:
                errors.append((case.id, type(exc).__name__, str(exc)[:150]))
    finally:
        release_shared(shared)

    total = len(cases)
    _write_outcome_summary(
        attempted=total,
        completed=total - len(errors),
        failures=[
            {"gid": case_id, "error_class": error_class, "message": msg}
            for case_id, error_class, msg in errors
        ],
    )
    print(f"{total} cases, errors={len(errors)}")
    for case_id, error_class, msg in errors:
        print(f"  [ERROR] {case_id}: {error_class}: {msg}")


if __name__ == "__main__":
    main()
