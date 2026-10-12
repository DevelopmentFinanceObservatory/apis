"""The bind-menu cap preserves up to `min_recall_bind_slots` query-recalled shapes even when
higher-scoring topic expansions crowd the menu. Synthetic shapes scored above the unknown-score
floor isolate the reserve and top-K ranking in `_phase_prune_and_rank`.
"""

from __future__ import annotations

import asyncio
from typing import cast

from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.region.phase_rank_bind import _phase_prune_and_rank
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot

_ROLES = ("financingProvider", "financingRecipient")


def _shape(shape_id: str, *, score: float, recalled: bool) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=f"FinancialFlow{shape_id}",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=_ROLES,
        representative_score=score,
        recalled=recalled,
    )


def _state(shapes_found: list[ShapeDraft]) -> ResolveState:
    """Build minimal state. `entities` skips graph probing."""
    return ResolveState(
        variable="aid to Kenya",
        entities=["Kenya"],
        date_request=None,
        detect_query="aid to Kenya",
        role_query="aid to Kenya",
        dropped_qualifier_phrases=[],
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        shapes_found=shapes_found,
        coverage=empty_coverage_snapshot(),
    )


def _run_prune(shapes_found: list[ShapeDraft]) -> list[ShapeDraft]:
    state = _state(shapes_found)
    result = asyncio.run(_phase_prune_and_rank(state))
    assert result is None, "expected _phase_prune_and_rank to fall through"
    return state.capped_shapes


def _assert_score_descending(capped: list[ShapeDraft]) -> None:
    scores = [s.representative_score for s in capped]
    assert scores == sorted(scores, reverse=True), (
        f"capped_shapes must stay score-descending (near_neighbor.py and the "
        f"cascade-widening fan-out in candidates.py both depend on it); got {scores}"
    )


def test_crowding_both_recalled_shapes_survive_the_cap():
    # Ten higher-ranked topics would crowd out both recalled shapes without the reserve (#743).
    topic_only = [_shape(f"topic{i}", score=0.90 - i * 0.01, recalled=False) for i in range(10)]
    recalled = [
        _shape("recalled_hi", score=0.70, recalled=True),
        _shape("recalled_lo", score=0.66, recalled=True),
    ]
    capped = _run_prune(topic_only + recalled)

    assert len(capped) == settings().bind_top_k
    capped_ids = [s.shape_id for s in capped]
    assert "recalled_hi" in capped_ids
    assert "recalled_lo" in capped_ids
    # The two reserved slots leave six places for the highest-ranked topics.
    for i in range(6, 10):
        assert f"topic{i}" not in capped_ids
    for i in range(6):
        assert f"topic{i}" in capped_ids
    _assert_score_descending(capped)


def test_fewer_than_reserve_recalled_reserves_only_what_exists():
    # With one recalled shape, the reserve claims one slot.
    assert settings().min_recall_bind_slots == 2
    non_recalled = [_shape(f"s{i}", score=0.90 - i * 0.01, recalled=False) for i in range(8)]
    recalled = [_shape("recalled_only", score=0.66, recalled=True)]
    capped = _run_prune(non_recalled + recalled)

    assert len(capped) == settings().bind_top_k
    capped_ids = [s.shape_id for s in capped]
    assert "recalled_only" in capped_ids
    # The reserve displaces the lowest-ranked shape, s7.
    assert "s7" not in capped_ids
    for i in range(7):
        assert f"s{i}" in capped_ids
    _assert_score_descending(capped)


def test_none_recalled_menu_is_the_plain_cap():
    shapes = [_shape(f"s{i}", score=0.90 - i * 0.01, recalled=False) for i in range(10)]
    capped = _run_prune(shapes)

    assert [s.shape_id for s in capped] == [f"s{i}" for i in range(settings().bind_top_k)]
    _assert_score_descending(capped)


def test_recalled_already_inside_top_k_menu_is_unchanged():
    # Both recalled shapes already fit within the top K.
    shapes = [_shape(f"s{i}", score=0.90 - i * 0.01, recalled=(i in (1, 4))) for i in range(10)]
    capped = _run_prune(shapes)

    assert [s.shape_id for s in capped] == [f"s{i}" for i in range(settings().bind_top_k)]
    _assert_score_descending(capped)


def test_reserve_zero_menu_identical_to_plain_cap(search_settings):
    # A zero reserve leaves the plain top-K cap unchanged.
    non_recalled = [_shape(f"s{i}", score=0.90 - i * 0.01, recalled=False) for i in range(8)]
    recalled = [_shape("recalled_only", score=0.66, recalled=True)]
    search_settings(min_recall_bind_slots=0)
    capped = _run_prune(non_recalled + recalled)

    assert [s.shape_id for s in capped] == [f"s{i}" for i in range(settings().bind_top_k)]
    _assert_score_descending(capped)
