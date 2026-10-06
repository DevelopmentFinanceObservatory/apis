"""Tests the candidates path's bound-unit gate.

The elected SET group (SV_A/SV_B) precedes alternate SV_C. Each candidate passes the gate on
its own read plan (`bound_unit_gate`), so a failed or oversized preview leaves it admitted.
"""

from __future__ import annotations

import pytest

from materialise.errors import UpstreamVersionChangedError
from search.engine.region import candidate_collapse as candidate_collapse_module
from search.engine.region import candidate_unit_election as candidate_unit_election_module
from search.engine.region.observations import fetch as fetch_module
from tests.engine.test_facet_unit_gate_multi_statvar import (
    _SV_A,
    _SV_B,
    _SV_C,
    _obs_facet,
    _run,
    _SetPlusAltBindLLM,
)


def _three_way_obs() -> dict:
    return {
        f"{_SV_A}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=10.0)],
        f"{_SV_B}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=20.0)],
        f"{_SV_C}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=20.0)],
    }


def _three_way(**kwargs):
    return _run(
        obs=_three_way_obs(),
        unit_text="US dollars",
        llm=_SetPlusAltBindLLM(),
        also_plausible_svs=[_SV_C],
        **kwargs,
    )


def _spec_svs(result) -> set[frozenset[str]]:
    return {frozenset(sv.ref.dcid for sv in s.resolved.members) for s in result.specs}


def test_the_candidates_gate_reads_no_preview(monkeypatch):
    """Each candidate passes the gate on its own read plan, so the gate makes no
    observations read of its own; both candidates are admitted."""

    def _no_read(*args, **kwargs):
        raise AssertionError("the bound-unit gate must not read observations")

    monkeypatch.setattr(candidate_collapse_module, "_build_observations", _no_read)
    result = _three_way()

    assert result.status == "candidates", result
    assert _spec_svs(result) == {frozenset({_SV_A, _SV_B}), frozenset({_SV_C})}


def test_a_preview_read_timeout_leaves_both_candidates_admitted(monkeypatch):
    """A preview read that would cross the shared deadline decides nothing: the gate reads no
    preview, so both candidates are admitted."""

    def _times_out(*args, **kwargs):
        raise TimeoutError("simulated shared-deadline crossing")

    monkeypatch.setattr(fetch_module, "materialise_preview", _times_out)
    result = _three_way()

    assert result.status == "candidates", result
    assert _spec_svs(result) == {frozenset({_SV_A, _SV_B}), frozenset({_SV_C})}


def test_an_oversized_preview_leaves_both_candidates_admitted(search_settings):
    """The elected SET group has two pairs, over the cap of 1, and alternate SV_C one. The
    gate decides from each read plan, so neither is refused `scope_too_broad`."""
    search_settings(obs_preview_max_pairs=1)
    result = _three_way()

    assert result.status == "candidates", result
    assert result.unsupported is None
    assert _spec_svs(result) == {frozenset({_SV_A, _SV_B}), frozenset({_SV_C})}


def test_unestablished_candidate_election_fails_the_request(monkeypatch):
    """An upstream version change during a candidate's unit election raises through the
    candidates door rather than dropping that candidate as unavailable."""

    def _raise(*args, **kwargs):
        raise UpstreamVersionChangedError("rate table changed version during its load")

    monkeypatch.setattr(candidate_unit_election_module, "decide_part_unit", _raise)
    with pytest.raises(UpstreamVersionChangedError):
        _run(
            obs=_three_way_obs(),
            unit_text="US dollars",
            llm=_SetPlusAltBindLLM(),
            also_plausible_svs=[_SV_C],
        )


def test_an_empty_read_leaves_the_candidate_admitted():
    """SV_B's elected USD read would return no rows, but the coverage index holds its facet,
    so the SET candidate's plan serves both StatVars and the candidate is admitted beside the
    SV_C alternate: the gate reads no preview."""
    obs = _three_way_obs()
    result = _run(
        obs=obs,
        unit_text="US dollars",
        llm=_SetPlusAltBindLLM(),
        also_plausible_svs=[_SV_C],
        sdmx_obs={key: facets for key, facets in obs.items() if not key.startswith(_SV_B)},
    )

    assert result.status == "candidates", result
    assert _spec_svs(result) == {frozenset({_SV_A, _SV_B}), frozenset({_SV_C})}
