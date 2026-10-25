"""The bind menu is the top `bind_top_k` admissible shapes in rank order, recalled or topic
expanded alike. Synthetic shapes scored above the unknown-score floor isolate the top-K
ranking in `_phase_prune_and_rank`.
"""

from __future__ import annotations

import asyncio
from typing import cast

from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.door_names import own_texts
from search.engine.region.phase_rank_bind import _phase_prune_and_rank
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._quantity import QuantityReadOnlyLLM

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
        entity_texts=own_texts(["Kenya"]),
        date_request=None,
        detect_query="aid to Kenya",
        role_query="aid to Kenya",
        dropped_qualifier_phrases=[],
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=QuantityReadOnlyLLM(),
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
        f"capped_shapes must stay score-descending (the status rule offers rivals in "
        f"menu order); got {scores}"
    )


def test_topic_expansions_ranked_above_recalled_shapes_fill_the_menu():
    # Ten higher-ranked topics fill the menu ahead of both recalled shapes.
    topic_only = [_shape(f"topic{i}", score=0.90 - i * 0.01, recalled=False) for i in range(10)]
    recalled = [
        _shape("recalled_hi", score=0.70, recalled=True),
        _shape("recalled_lo", score=0.66, recalled=True),
    ]
    capped = _run_prune(topic_only + recalled)

    assert [s.shape_id for s in capped] == [f"topic{i}" for i in range(settings().bind_top_k)]
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
