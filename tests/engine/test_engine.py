"""An empty materialisation cannot produce a definite response."""

from __future__ import annotations

from search.engine.retrieval.retrieve import Materialised
from search.models import CoverageBare
from tests.engine._harness import make_request, offline_resolve


def test_standard_empty_materialised_returns_no_data(monkeypatch):
    """An empty materialisation falls back to ``no_data``."""

    def _fake_materialise(*_args, **_kwargs):
        return Materialised(sv_dcids=[], has_data=False, coverage=CoverageBare(has_data=False))

    monkeypatch.setattr("search.engine.region.materialise_step.materialise", _fake_materialise)
    # std-06b: a standard shape that resolves definite via the dominance rule.
    result = offline_resolve(make_request("fertility rate in Kenya"))
    assert result.status == "no_data"
    part = result.parts[0]
    assert part.status == "no_data"
    assert part.no_data.reason == "no_observations"
