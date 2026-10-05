"""Every corpus case resolves offline without raising.

Each case in `corpus/` runs once through `offline_resolve` with the suite's fake graph, the
recorded FakeLLM bank, fixture-backed SDMX and empty materialise doubles. A refusal
(`EngineInputError`) is an answer and passes. Any other exception fails, and the suite's miss
guard fails a case that missed a recorded LLM fixture. The test scores nothing. The eval gate
scores answers.

The module keeps every corpus case runnable offline, and it is how
`scripts/harvest_llm_fixtures.py` reaches the corpus's LLM fixture keys. The harvest's pytest
pass collects it with the rest of the suite and prunes the bank to the keys that pass reached.
It therefore carries no skip or deselecting marker, so a run of the suite always runs every case.
"""

from __future__ import annotations

from contextlib import suppress

import pytest

from materialise.rate_tables import RateTableCache
from search.engine.errors import EngineInputError
from search.eval.cases import Case, load_cases
from search.eval.runner import build_request
from search.models import ResolveRequest
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import build_materialise_double, offline_resolve

_CASES = load_cases()


@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
def test_corpus_case_resolves_offline(case: Case) -> None:
    request = ResolveRequest.model_validate(build_request(case))
    materialise_graph, materialise_sdmx, _requests = build_materialise_double({})
    # A refusal is an answer.
    with suppress(EngineInputError):
        offline_resolve(
            request,
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        )
