"""The bind menu never holds part of a topic root (`region.seating.seat_menu`): a root
seats every leaf or none. Each row gives the admissible shapes in rank order, as
`_rank_key` leaves them, and the seated menu of eight."""

from __future__ import annotations

import dataclasses

import pytest

from search.engine.region.seating import seat_menu
from search.engine.shape import ShapeDraft

_SEATS = 8
_FLOW_ROLES = ("financingProvider", "financingRecipient")
_AID_ROOT = "dc/topic/Aid"
_HEALTH_ROOT = "dc/topic/Health"


def _scored(shape_id: str, score: float, *, flow: bool = False) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=shape_id,
        meas_prop_dcid="amount" if flow else "count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=_FLOW_ROLES if flow else None,
        representative_score=score,
    )


def _leaves(prefix: str, count: int, root: str, root_score: float | None, *, rank: float):
    """`count` leaves of `root`. `rank` is the rank derive gave them, just under the best
    scored StatVar of the pool."""
    return [
        ShapeDraft(
            shape_id=f"{prefix}{i}",
            label=f"{prefix}{i}",
            pop_type_dcid=f"{prefix}{i}",
            meas_prop_dcid="value",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            expansion_rank_score=rank,
            expansion_root_score=root_score,
            expansion_root=root,
        )
        for i in range(count)
    ]


_DAC2A = _scored("dac2a", 0.6224, flow=True)
_PER_GNI = _scored("dac2a_per_gni", 0.5962, flow=True)
_PER_CAPITA = _scored("dac2a_per_capita", 0.5857, flow=True)
_NINE_SCORED = [_scored(f"scored{i}", 0.90 - i / 100) for i in range(9)]


@pytest.mark.parametrize(
    ("ranked", "menu"),
    [
        pytest.param(
            # "under 5 mortality rate": the root's four leaves stay together behind the
            # best scored shape, and three more scored shapes take the seats left.
            [
                _NINE_SCORED[0],
                *_leaves("leaf", 4, _HEALTH_ROOT, 0.9708, rank=0.8999),
                *_NINE_SCORED[1:],
            ],
            ["scored0", "leaf0", "leaf1", "leaf2", "leaf3", "scored1", "scored2", "scored3"],
            id="a-four-leaf-root-above-eight-scored-shapes-seats-whole",
        ),
        pytest.param(
            # "aid to Bangladesh": 26 leaves rank 0.6254, under a DAC1 family at 0.6255 the
            # role rule refused, and ahead of every admissible flow.
            [*_leaves("sdg", 26, _AID_ROOT, 0.6528, rank=0.6254), _DAC2A, _PER_GNI, _PER_CAPITA],
            ["dac2a", "dac2a_per_gni", "dac2a_per_capita"],
            id="a-root-larger-than-the-seats-left-seats-no-leaf-and-the-shapes-behind-it-sit",
        ),
        pytest.param(
            [
                _scored("first", 0.80),
                *_leaves("leaf", 7, _AID_ROOT, 0.75, rank=0.7999),
                _scored("second", 0.70),
            ],
            ["first", *(f"leaf{i}" for i in range(7))],
            id="a-root-that-fits-the-seats-left-exactly-fills-them",
        ),
        pytest.param(
            [
                _scored("flow", 0.74, flow=True),
                *_leaves("health", 9, _HEALTH_ROOT, 0.72, rank=0.7399),
                *_leaves("aid", 3, _AID_ROOT, 0.70, rank=0.7398),
                _scored("weak_flow", 0.66, flow=True),
            ],
            ["flow", "aid0", "aid1", "aid2", "weak_flow"],
            id="a-root-that-does-not-fit-is-passed-over-for-one-that-does",
        ),
        pytest.param(
            _leaves("leaf", 9, _AID_ROOT, 0.8, rank=0.8),
            [],
            id="a-list-of-one-oversized-root-seats-nothing",
        ),
        pytest.param(
            [
                *_leaves("leaf", 3, _AID_ROOT, 0.70, rank=0.7399),
                _scored("first", 0.74),
                _scored("tied", 0.70),
                _scored("below", 0.69),
            ],
            ["first", "tied", "leaf0", "leaf1", "leaf2", "below"],
            id="a-root-tying-a-scored-shape-stands-behind-it",
        ),
        pytest.param(
            # "refugees from Somalia": the root matched at 1.0 and the best scored shape at
            # 0.6356, so the leaf takes the seat behind that shape and ahead of the rest.
            [
                _scored("dac1", 0.6356, flow=True),
                _scored("dac1_grant_equivalent", 0.6355, flow=True),
                *_leaves("sdg_refugees", 1, _AID_ROOT, 1.0, rank=0.6355),
                _scored("crs", 0.6143, flow=True),
            ],
            ["dac1", "sdg_refugees0", "dac1_grant_equivalent", "crs"],
            id="a-root-at-1.0-stands-behind-the-best-scored-shape-alone",
        ),
        pytest.param(
            # A collapse stands a family's best score where its earliest member stood, so
            # a weaker scored shape can come ahead of the leaves.
            [
                _scored("collapsed", 0.59),
                *_leaves("leaf", 2, _AID_ROOT, 0.65, rank=0.6223),
                _DAC2A,
            ],
            ["collapsed", "dac2a", "leaf0", "leaf1"],
            id="scored-shapes-out-of-score-order-still-stand-ahead-of-a-root-they-outrank",
        ),
        pytest.param(
            [
                _scored("best", 0.70),
                *_leaves("health", 2, _HEALTH_ROOT, 0.90, rank=0.6999),
                *_leaves("aid", 1, _AID_ROOT, 0.80, rank=0.6998),
                _scored("weak", 0.60),
            ],
            ["best", "health0", "health1", "aid0", "weak"],
            id="two-roots-above-the-ceiling-stand-behind-the-best-scored-shape-in-root-order",
        ),
        pytest.param(
            _NINE_SCORED,
            [f"scored{i}" for i in range(8)],
            id="a-list-with-no-expansion-keeps-its-first-seats",
        ),
    ],
)
def test_a_root_seats_every_leaf_or_none(ranked, menu):
    assert [shape.shape_id for shape in seat_menu(ranked, _SEATS)] == menu


@pytest.mark.parametrize(
    ("ranked", "seats", "menu"),
    [
        pytest.param(
            [_scored("first", 0.80), *_leaves("leaf", 1, _AID_ROOT, 0.75, rank=0.7999)],
            0,
            [],
            id="no-seat-seats-nothing",
        ),
        pytest.param(
            [
                *_leaves("pair", 2, _AID_ROOT, 0.95, rank=0.7999),
                *_leaves("single", 1, _HEALTH_ROOT, 0.90, rank=0.7998),
                _scored("first", 0.80),
            ],
            1,
            ["first"],
            id="one-seat-goes-to-the-best-scored-shape",
        ),
        pytest.param(
            [
                *_leaves("pair", 2, _AID_ROOT, 0.95, rank=0.95),
                *_leaves("one", 1, _HEALTH_ROOT, 0.9, rank=0.9),
            ],
            1,
            ["one0"],
            id="one-seat-passes-over-a-pair-for-a-single-leaf",
        ),
        pytest.param(
            [
                _scored("first", 0.80),
                *_leaves("unscored_root", 2, _AID_ROOT, None, rank=0.7999),
                _scored("second", 0.70),
            ],
            _SEATS,
            ["first", "second", "unscored_root0", "unscored_root1"],
            id="a-root-with-no-score-stands-behind-every-scored-shape",
        ),
        pytest.param(
            [
                _scored("first", 0.80),
                dataclasses.replace(_scored("scored_leaf", 0.75), expansion_root=_AID_ROOT),
                *_leaves("leaf", 1, _AID_ROOT, 0.79, rank=0.7999),
            ],
            _SEATS,
            ["first", "leaf0", "scored_leaf"],
            id="a-shape-with-a-score-and-a-root-is-seated-as-a-scored-shape",
        ),
        pytest.param(
            # The UN resolver scores on its own scale, so its shapes take the seats ours
            # leave, and its scores cap no root of ours.
            [
                dataclasses.replace(_scored("un_first", 0.95), pool="un_system"),
                dataclasses.replace(_scored("un_second", 0.90), pool="un_system"),
                _scored("ours", 0.68),
                *_leaves("leaf", 1, _AID_ROOT, 0.80, rank=0.6799),
            ],
            3,
            ["ours", "leaf0", "un_first"],
            id="the-un-pool-fills-the-seats-ours-leaves",
        ),
    ],
)
def test_the_seat_count_bounds_the_menu(ranked, seats, menu):
    assert [shape.shape_id for shape in seat_menu(ranked, seats)] == menu


def test_scored_shapes_are_seated_in_the_order_they_come_in():
    """Scored shapes are never reordered, so a collapse that reorders them is kept."""
    ranked = [_scored("a", 0.70), _scored("b", 0.80), _scored("c", 0.75)]

    assert seat_menu(ranked, _SEATS) == ranked
