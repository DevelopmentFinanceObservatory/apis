"""Pins the sweep's request construction: it posts the same body the eval runner builds."""

from __future__ import annotations

from unittest.mock import create_autospec

import scripts.sweep_corpus as m
from entity_resolution import EntityResolver
from search.eval.cases import validate_case


def _case(**fields):
    return validate_case(
        {
            "id": "sweep-case",
            "query": "aid disbursed to Kenya",
            "why": "A raw-text query that reaches the engine.",
            "expect": {"declined": {"reason": "out_of_scope"}, "kind": "declined"},
            **fields,
        },
        families={},
    )


def test_run_one_asks_for_observations_only_when_the_case_opts_in(monkeypatch):
    requests = []

    async def fake_resolve_async(request, **_kwargs):
        requests.append(request)

    monkeypatch.setattr(m, "resolve_async", fake_resolve_async)
    resolver = create_autospec(EntityResolver, instance=True)

    m._run_one(_case(), resolver=resolver)
    m._run_one(_case(include_observations=True), resolver=resolver)

    assert requests[0].response_options.include_observations is None
    assert requests[1].response_options.include_observations is True
