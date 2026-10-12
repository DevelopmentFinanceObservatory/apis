"""Checks topic-expansion rank order, score clamping, and bind-menu admission."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import cast

import pytest

from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.interpret import Recall
from search.engine.region.phase_rank_bind import _phase_prune_and_rank
from search.engine.region.state import ResolveState
from search.engine.retrieval.derive import _EXPANSION_RANK_EPSILON, derive_shapes
from search.engine.shape import ShapeDraft
from tests.engine._builders import place_sets
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.fixtures import FakeGraph

# derive_shapes: expansion rank stamping and pool order

_STRONG_ROOT = "dc/topic/Strong"
_WEAK_ROOT = "dc/topic/Weak"


@pytest.fixture(autouse=True)
def _every_topic_root_qualifies(search_settings: pytest.MonkeyPatch) -> None:
    """Ensures both fixture Topics expand so tests can compare their ranks."""
    search_settings(topic_expand_margin=1.0)


def _two_root_graph() -> FakeGraph:
    """Builds two topic groups with distinct populations and two recalled candidates."""
    nodes: dict = {
        _STRONG_ROOT: {
            "arcs": {
                "relevantVariable": {
                    "nodes": [
                        {"dcid": "StrongExpanded_1", "types": ["StatisticalVariable"]},
                        {"dcid": "StrongExpanded_2", "types": ["StatisticalVariable"]},
                    ]
                }
            }
        },
        _WEAK_ROOT: {
            "arcs": {
                "relevantVariable": {
                    "nodes": [
                        {"dcid": "WeakExpanded_1", "types": ["StatisticalVariable"]},
                        {"dcid": "WeakExpanded_2", "types": ["StatisticalVariable"]},
                    ]
                }
            }
        },
        "StrongExpanded_1": {"label": "s1", "arcs": _std_sv_arcs("Country", "count")},
        "StrongExpanded_2": {"label": "s2", "arcs": _std_sv_arcs("Country", "count")},
        "WeakExpanded_1": {"label": "w1", "arcs": _std_sv_arcs("Household", "count")},
        "WeakExpanded_2": {"label": "w2", "arcs": _std_sv_arcs("Household", "count")},
        "RecallHigh_SV": {"label": "high", "arcs": _std_sv_arcs("Person", "count")},
        "RecallLow_SV": {"label": "low", "arcs": _std_sv_arcs("Institution", "count")},
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _derive_two_roots(strong_root_score: float = 0.8) -> list[ShapeDraft]:
    return derive_shapes(
        confirmed_svs=[
            _STRONG_ROOT,
            _WEAK_ROOT,
            "RecallHigh_SV",
            "RecallLow_SV",
        ],
        graph=_two_root_graph(),
        sv_scores={
            _STRONG_ROOT: strong_root_score,
            _WEAK_ROOT: 0.5,
            "RecallHigh_SV": 0.9,
            "RecallLow_SV": 0.7,
        },
        sv_types={
            _STRONG_ROOT: "Topic",
            _WEAK_ROOT: "Topic",
            "RecallHigh_SV": "StatisticalVariable",
            "RecallLow_SV": "StatisticalVariable",
        },
    )


def _shape_with(shapes: list[ShapeDraft], sv_dcid: str) -> ShapeDraft:
    return next(s for s in shapes if sv_dcid in (s.sv_arc_facts or {}))


def _members(shapes: list[ShapeDraft]) -> list[set[str]]:
    """Return derived groups in pool order."""
    return [set(s.sv_arc_facts or {}) for s in shapes]


class TestExpansionRankStamp:
    """Expansion-only groups rank by their root score, capped below the best recall."""

    def test_expansion_only_groups_carry_no_representative_score(self):
        shapes = _derive_two_roots()
        strong = _shape_with(shapes, "StrongExpanded_1")
        weak = _shape_with(shapes, "WeakExpanded_1")
        assert strong.representative_score is None
        assert weak.representative_score is None
        # Every member of each group reached it through its own topic only.
        assert set(strong.sv_arc_facts or {}) == {"StrongExpanded_1", "StrongExpanded_2"}
        assert set(weak.sv_arc_facts or {}) == {"WeakExpanded_1", "WeakExpanded_2"}

    def test_expansion_rank_score_is_the_root_score_clamped_under_the_best_recall(self):
        shapes = _derive_two_roots()
        assert _shape_with(shapes, "StrongExpanded_1").expansion_rank_score == (
            min(0.8, 0.9 - _EXPANSION_RANK_EPSILON)
        )
        assert _shape_with(shapes, "WeakExpanded_1").expansion_rank_score == 0.5

    def test_recalled_shapes_carry_no_expansion_rank(self):
        shapes = _derive_two_roots()
        recalled = _shape_with(shapes, "RecallHigh_SV")
        assert recalled.representative_score == 0.9
        assert recalled.expansion_rank_score is None

    def test_rank_score_prefers_the_measured_score(self):
        shapes = _derive_two_roots()
        assert _shape_with(shapes, "RecallHigh_SV").rank_score == 0.9
        assert _shape_with(shapes, "StrongExpanded_1").rank_score == min(
            0.8, 0.9 - _EXPANSION_RANK_EPSILON
        )
        assert _shape_with(shapes, "WeakExpanded_1").rank_score == 0.5

    def test_groups_hold_their_expansion_rank_across_their_own_members(self):
        """Every group member carries its root's expansion rank."""
        shapes = _derive_two_roots()
        assert _shape_with(shapes, "StrongExpanded_2").expansion_rank_score == (
            _shape_with(shapes, "StrongExpanded_1").expansion_rank_score
        )

    def test_pool_order_interleaves_expansion_groups_by_root_rank(self):
        """Groups interleave with recalled candidates by root score: 0.9, 0.8, 0.7, then 0.5."""
        shapes = _derive_two_roots()
        assert _members(shapes) == [
            {"RecallHigh_SV"},
            {"StrongExpanded_1", "StrongExpanded_2"},
            {"RecallLow_SV"},
            {"WeakExpanded_1", "WeakExpanded_2"},
        ]

    def test_inflated_root_still_ranks_below_the_best_recalled_candidate(self):
        """A 1.0 topic self-match stays below the best recalled candidate."""
        shapes = _derive_two_roots(strong_root_score=1.0)
        strong = _shape_with(shapes, "StrongExpanded_1")
        assert strong.expansion_rank_score == 0.9 - _EXPANSION_RANK_EPSILON
        assert strong.expansion_rank_score < 0.9
        assert _members(shapes) == [
            {"RecallHigh_SV"},
            {"StrongExpanded_1", "StrongExpanded_2"},
            {"RecallLow_SV"},
            {"WeakExpanded_1", "WeakExpanded_2"},
        ]


# _phase_prune_and_rank: the capped bind menu


def _shape(
    shape_id: str,
    *,
    score: float | None,
    rank: float | None = None,
    pop_type: str | None = None,
    meas_prop: str = "count",
    declared_roles: tuple[str, ...] = ("observationAbout",),
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=pop_type if pop_type is not None else f"Person{shape_id}",
        meas_prop_dcid=meas_prop,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=declared_roles,
        representative_score=score,
        expansion_rank_score=rank,
    )


def _state(
    shapes_found: list[ShapeDraft], *, variable: str = "total population of Kenya"
) -> ResolveState:
    """A minimal ResolveState carrying only what _phase_prune_and_rank reads,
    following test_bind_slot_reserve.py's `_state` convention. entities
    non-empty and no place set that expands together skip the
    sibling-availability probe, leaving the graph field unused."""
    return ResolveState(
        variable=variable,
        entities=["Kenya"],
        date_request=None,
        detect_query=variable,
        role_query=variable,
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


def _run_prune(
    shapes_found: list[ShapeDraft], *, variable: str = "total population of Kenya"
) -> list[ShapeDraft]:
    state = _state(shapes_found, variable=variable)
    result = asyncio.run(_phase_prune_and_rank(state))
    assert result is None, "expected _phase_prune_and_rank to fall through"
    return state.capped_shapes


class TestCappedMenuHoldsTopRankedExpansionGroups:
    """Eight scored shapes and four expansion members compete for `bind_top_k`.

    The 0.85 group ranks between scored shapes. All scores stay above the unknown-score threshold.
    """

    def _menu(self) -> list[ShapeDraft]:
        scored = [
            _shape(f"scored{i}", score=score)
            for i, score in enumerate((0.90, 0.84, 0.78, 0.74, 0.70, 0.68, 0.67, 0.66))
        ]
        expansion = [
            _shape("strongA", score=None, rank=0.85),
            _shape("strongB", score=None, rank=0.85),
            _shape("weakA", score=None, rank=0.15),
            _shape("weakB", score=None, rank=0.15),
        ]
        return _run_prune(scored + expansion)

    def test_strongest_expansion_group_reaches_the_menu(self):
        capped_ids = [s.shape_id for s in self._menu()]
        assert "strongA" in capped_ids
        assert "strongB" in capped_ids

    def test_weaker_expansion_group_is_capped_away(self):
        capped_ids = [s.shape_id for s in self._menu()]
        assert "weakA" not in capped_ids
        assert "weakB" not in capped_ids

    def test_expansion_group_outranks_the_lowest_scored_shapes(self):
        """The 0.85 group displaces the two lowest-scored shapes."""
        capped_ids = [s.shape_id for s in self._menu()]
        for shape_id in ("scored6", "scored7"):
            assert shape_id not in capped_ids

    def test_strongest_expansion_group_sits_between_the_scored_shapes(self):
        """The 0.85 group sits between scores 0.90 and 0.84."""
        menu = self._menu()
        assert len(menu) == settings().bind_top_k
        assert [s.shape_id for s in menu][:4] == ["scored0", "strongA", "strongB", "scored1"]
        assert [s.rank_score for s in menu] == sorted([s.rank_score for s in menu], reverse=True)

    def test_no_expansion_shape_carries_a_numeric_score(self):
        menu = self._menu()
        expansion = [s for s in menu if s.shape_id.startswith(("strong", "weak"))]
        assert expansion
        for shape in expansion:
            assert shape.representative_score is None
            assert shape.expansion_rank_score is not None


class TestScoreFloorReadsTopicRootScore:
    """An expansion-only shape meets `unknown_score_threshold` on its topic root's recall score.

    Its expansion rank is capped below the best admitted recall and only orders the menu. Live
    "tuberculosis incidence rate in Ethiopia" recalls `dc/topic/sdg_3.3.2` at 0.8405, and with
    the withheld CDC count gone the cap puts `sdg/SH_TBS_INCD` at 0.6483 under a 0.6484 recall.
    """

    def test_strong_root_passes_the_floor_under_a_low_cap(self):
        state = _state(
            [
                _shape("tularemia", score=0.6484),
                _shape("tb_incidence", score=None, rank=0.6483, meas_prop="value"),
            ],
            variable="tuberculosis incidence rate",
        )
        state.shapes_found[1] = dataclasses.replace(
            state.shapes_found[1], expansion_root_score=0.8405
        )
        assert asyncio.run(_phase_prune_and_rank(state)) is None
        assert [s.shape_id for s in state.capped_shapes] == ["tb_incidence"]
        assert state.n_below_score_floor == 1

    def test_root_under_the_floor_is_refused(self):
        expansion = dataclasses.replace(
            _shape("weak_topic", score=None, rank=0.62, meas_prop="value"),
            expansion_root_score=0.62,
        )
        state = _state([_shape("scored", score=0.70), expansion])
        assert asyncio.run(_phase_prune_and_rank(state)) is None
        assert [s.shape_id for s in state.capped_shapes] == ["scored"]
        assert state.n_below_score_floor == 1

    def test_only_shapes_under_the_floor_resolve_no_variable(self):
        expansion = dataclasses.replace(
            _shape("weak_topic", score=None, rank=0.6318, meas_prop="value"),
            expansion_root_score=0.62,
        )
        state = _state([_shape("syphilis", score=0.6319), expansion])
        state.rcl = Recall(candidates=[], resolutions={})
        state.place_sets = place_sets({"Kenya": "country/KEN"})
        result = asyncio.run(_phase_prune_and_rank(state))
        assert result is not None
        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"

    def test_derive_stamps_the_uncapped_root_score(self):
        shapes = _derive_two_roots(strong_root_score=1.0)
        strong = _shape_with(shapes, "StrongExpanded_1")
        assert strong.expansion_rank_score == 0.9 - _EXPANSION_RANK_EPSILON
        assert strong.expansion_root_score == 1.0
        assert strong.measured_score == 1.0
        assert _shape_with(shapes, "WeakExpanded_1").expansion_root_score == 0.5
        recalled = _shape_with(shapes, "RecallHigh_SV")
        assert recalled.expansion_root_score is None
        assert recalled.measured_score == 0.9


# _phase_prune_and_rank: measure-class check


class TestMeasureClassConflictLeavesTheMenu:
    """A shape whose population type's name names another measure class than the query never
    reaches the menu.

    Recorded "HIV prevalence" recalls `dc/topic/HIV`, which reaches the SDG 3.3.1 incidence
    series above the floor, beside a CDC count whose population type's name names no class.
    """

    def _shapes(self) -> list[ShapeDraft]:
        return [
            dataclasses.replace(
                _shape("hiv_count", score=0.8029),
                label="Count of HIV/Acquired Immune Deficiency Syndrome incidents",
                pop_type_name="MedicalConditionIncident",
            ),
            dataclasses.replace(
                _shape("sdg_incidence", score=None, rank=0.7867, meas_prop="value"),
                label="Number of new HIV infections per 1,000 uninfected population",
                pop_type_name=(
                    "Number of new HIV infections per 1,000 uninfected population, by sex and age"
                ),
                expansion_root_score=0.7867,
            ),
        ]

    @pytest.mark.parametrize(
        ("variable", "capped", "refused"),
        [
            ("HIV prevalence", ["hiv_count"], 1),
            ("new HIV infections", ["hiv_count", "sdg_incidence"], 0),
            ("HIV", ["hiv_count", "sdg_incidence"], 0),
        ],
    )
    def test_a_conflicting_shape_never_reaches_capped_shapes(self, variable, capped, refused):
        state = _state(self._shapes(), variable=variable)
        assert asyncio.run(_phase_prune_and_rank(state)) is None
        assert sorted(s.shape_id for s in state.capped_shapes) == capped
        assert state.n_measure_class_refused == refused
        assert state.n_below_score_floor == 0

    def test_a_menu_the_check_empties_resolves_no_variable(self):
        state = _state(self._shapes()[1:], variable="HIV prevalence")
        state.rcl = Recall(candidates=[], resolutions={})
        state.place_sets = place_sets({"Kenya": "country/KEN"})
        result = asyncio.run(_phase_prune_and_rank(state))
        assert result is not None
        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"


# _phase_prune_and_rank: cross-population menu admission

_ROLE_BEARING = ("financingProvider", "financingRecipient")


class TestCrossPopulationExpansionDropped:
    """Flow wording lets a role-bearing anchor prune an unscored shape from another population.

    Indicator wording keeps the expansion because a topic match alone gives no evidence that the
    query names a different population.
    """

    def _anchor(self) -> ShapeDraft:
        return _shape(
            "crs_flow",
            score=0.74,
            pop_type="FinancialObligation",
            meas_prop="amount",
            declared_roles=_ROLE_BEARING,
        )

    def test_role_bearing_anchor_drops_a_cross_population_standard_shape(self):
        standard_sdg = _shape(
            "sdg_tof_hlthl",
            score=None,
            rank=0.70,
            pop_type="Thing",
            meas_prop="value",
        )
        menu = _run_prune(
            [self._anchor(), standard_sdg], variable="ODA commitments to health in Kenya"
        )
        assert [s.shape_id for s in menu] == ["crs_flow"]

    def test_standard_anchor_keeps_a_different_population_unscored_shape(self):
        """Matching role structure preserves the SDG expansion despite its different population."""
        anchor = _shape("cm_01", score=0.86, pop_type="Person", meas_prop="who/CM_01")
        sdg_mort = _shape(
            "sh_dyn_mort", score=None, rank=0.86, pop_type="SDG_SH_DYN_MORT", meas_prop="value"
        )
        menu = _run_prune([anchor, sdg_mort])
        assert {s.shape_id for s in menu} == {"cm_01", "sh_dyn_mort"}

    def test_same_population_unscored_shape_is_kept(self):
        """An equal population type keeps an unscored shape."""
        anchor = _shape(
            "crs_flow",
            score=0.74,
            pop_type="FinancialObligation",
            meas_prop="amount",
            declared_roles=_ROLE_BEARING,
        )
        sibling = _shape(
            "crs_other_prop",
            score=None,
            rank=0.70,
            pop_type="FinancialObligation",
            meas_prop="otherAmount",
        )
        menu = _run_prune([anchor, sibling])
        assert {s.shape_id for s in menu} == {"crs_flow", "crs_other_prop"}

    def test_nothing_is_pruned_without_a_scored_shape(self):
        unscored = _shape("expansion", score=None, rank=0.70, pop_type="Thing", meas_prop="value")
        menu = _run_prune([unscored])
        assert [s.shape_id for s in menu] == ["expansion"]

    def test_anchor_is_the_highest_scored_shape_not_the_first_listed(self):
        """The highest representative score sets the anchor, regardless of list order."""
        low_standard = _shape("low_standard", score=0.30, pop_type="Thing", meas_prop="value")
        anchor = _shape(
            "crs_flow",
            score=0.74,
            pop_type="FinancialObligation",
            meas_prop="amount",
            declared_roles=_ROLE_BEARING,
        )
        cross_pop = _shape("sdg_std", score=None, rank=0.70, pop_type="Thing", meas_prop="value")
        menu = _run_prune([low_standard, anchor, cross_pop], variable="ODA commitments to Kenya")
        assert "sdg_std" not in [s.shape_id for s in menu]

    def test_pruned_shape_frees_its_menu_slot(self):
        """Pruning the expansion before capping frees a menu slot for a scored shape."""
        anchor = _shape(
            "crs_flow",
            score=0.90,
            pop_type="FinancialObligation",
            meas_prop="amount",
            declared_roles=_ROLE_BEARING,
        )
        scored = [_shape(f"scored{i}", score=0.80 - i * 0.02) for i in range(7)]
        cross_pop = _shape("sdg_std", score=None, rank=0.95, pop_type="Thing", meas_prop="value")
        menu = _run_prune([anchor, *scored, cross_pop], variable="ODA commitments to Kenya")
        ids = [s.shape_id for s in menu]
        assert "sdg_std" not in ids
        assert len(ids) == settings().bind_top_k
        assert "scored6" in ids


class TestIndicatorFramedQueryKeepsTheExpansionShape:
    """Indicator wording keeps the SDG expansion when role-bearing recall shares only a disease noun.

    Pruning requires flow or aid wording.
    """

    def test_indicator_framed_query_keeps_a_cross_population_expansion_shape(self):
        anchor = _shape(
            "crs_tb_flow",
            score=0.74,
            pop_type="FinancialObligation",
            meas_prop="amount",
            declared_roles=_ROLE_BEARING,
        )
        sdg_incidence = _shape(
            "sdg_sh_tbs_incd",
            score=None,
            rank=0.70,
            pop_type="SDG_SH_TBS_INCD",
            meas_prop="value",
        )
        menu = _run_prune(
            [anchor, sdg_incidence], variable="tuberculosis incidence rate in Ethiopia"
        )
        assert {s.shape_id for s in menu} == {"crs_tb_flow", "sdg_sh_tbs_incd"}


class TestAdmissionAnchorMatchesTheSortKey:
    """The prune anchor follows the ranked-shape sort key, including breadth ties."""

    def test_anchor_is_the_sorts_first_scored_shape_on_a_breadth_tie(self):
        from search.engine.region.phase_rank_bind import _rank_key

        broad = _shape("broad", score=0.74, pop_type="FinancialObligation", meas_prop="amount")
        narrow = _shape("narrow", score=0.74, pop_type="FinancialObligation", meas_prop="amount")
        broad = dataclasses.replace(broad, sv_arc_facts={key: {} for key in ["a", "b", "c"]})
        narrow = dataclasses.replace(narrow, sv_arc_facts={key: {} for key in ["a"]})

        # Breadth breaks the rank tie. `_rank_key` must match the anchor's `min()` order.
        assert min([narrow, broad], key=_rank_key) is broad

    def test_anchor_is_the_highest_ranked_scored_shape_by_rank_key(self):
        from search.engine.region.phase_rank_bind import _rank_key

        # The anchor is chosen among scored shapes, regardless of expansion rank.
        scored = _shape("scored", score=0.60, pop_type="Thing", meas_prop="value")
        unscored = _shape("expanded", score=None, rank=0.95, pop_type="Thing", meas_prop="value")
        chosen = min([unscored, scored], key=_rank_key)
        assert chosen is unscored  # the sort's own first place
        scored_only = [s for s in [unscored, scored] if s.representative_score is not None]
        assert min(scored_only, key=_rank_key) is scored


# Two roots clamped to the SAME ceiling must keep their own relative order

_ROOT_A = "dc/topic/RootA"
_ROOT_B = "dc/topic/RootB"


def _two_clamped_roots_graph() -> FakeGraph:
    """Both topic roots, scored 0.90 and 0.85, exceed the 0.80 recalled candidate."""
    nodes: dict = {
        _ROOT_A: {
            "arcs": {
                "relevantVariable": {
                    "nodes": [{"dcid": "RootAExpanded_1", "types": ["StatisticalVariable"]}]
                }
            }
        },
        _ROOT_B: {
            "arcs": {
                "relevantVariable": {
                    "nodes": [{"dcid": "RootBExpanded_1", "types": ["StatisticalVariable"]}]
                }
            }
        },
        "RootAExpanded_1": {"label": "a1", "arcs": _std_sv_arcs("PopA", "count")},
        "RootBExpanded_1": {"label": "b1", "arcs": _std_sv_arcs("PopB", "count")},
        "RecallReal_SV": {"label": "real", "arcs": _std_sv_arcs("Person", "count")},
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


class TestMultipleClampedRootsKeepTheirOwnOrder:
    """Roots at 0.90 and 0.85 both exceed the 0.80 recall ceiling.

    The clamp must preserve their order or `shape_id` can rank the weaker group first.
    """

    def _derive(self) -> list[ShapeDraft]:
        return derive_shapes(
            confirmed_svs=[_ROOT_A, _ROOT_B, "RecallReal_SV"],
            graph=_two_clamped_roots_graph(),
            sv_scores={_ROOT_A: 0.90, _ROOT_B: 0.85, "RecallReal_SV": 0.80},
            sv_types={
                _ROOT_A: "Topic",
                _ROOT_B: "Topic",
                "RecallReal_SV": "StatisticalVariable",
            },
        )

    def test_stronger_root_ranks_strictly_above_the_weaker_one(self):
        shapes = self._derive()
        strong = _shape_with(shapes, "RootAExpanded_1")
        weak = _shape_with(shapes, "RootBExpanded_1")
        assert strong.expansion_rank_score is not None
        assert weak.expansion_rank_score is not None
        assert strong.expansion_rank_score > weak.expansion_rank_score

    def test_neither_root_outranks_the_real_candidate_that_set_the_ceiling(self):
        """The clamp's own invariant survives: clamped or not, no expansion
        member outranks the real recall candidate."""
        shapes = self._derive()
        strong = _shape_with(shapes, "RootAExpanded_1")
        weak = _shape_with(shapes, "RootBExpanded_1")
        real = _shape_with(shapes, "RecallReal_SV")
        assert real.representative_score == 0.80
        assert strong.expansion_rank_score < 0.80
        assert weak.expansion_rank_score < 0.80


# Uses recorded detection and node fixtures for tuberculosis expansion families.


def _derive_real_tb_pool() -> list[ShapeDraft]:
    graph = FakeGraph()
    candidates = graph.detect_svs("tuberculosis incidence rate").candidates
    assert candidates, (
        "tests/fixtures/graph_detect.json lost its 'tuberculosis incidence rate' entry"
    )
    confirmed_svs = [c.dcid for c in candidates]
    sv_scores = {c.dcid: c.score for c in candidates}
    sv_types = {c.dcid: c.type_of for c in candidates}
    return derive_shapes(
        confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores, sv_types=sv_types
    )


def _run_prune_over(shapes_found: list[ShapeDraft], *, variable: str) -> list[ShapeDraft]:
    state = ResolveState(
        variable=variable,
        entities=["Ethiopia"],
        date_request=None,
        detect_query=variable,
        role_query=variable,
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
    result = asyncio.run(_phase_prune_and_rank(state))
    assert result is None, "expected _phase_prune_and_rank to fall through"
    return state.capped_shapes


class TestTuberculosisIncidenceRateRegression:
    """The expected SDG shape and a medical-condition false friend have topic-root scores of
    0.8405 and 0.8168. Both exceed the 0.80 recall ceiling, so clamping must preserve their order.

    The fixture tests both query wordings, with and without "rate".
    """

    def test_expected_sdg_shape_outranks_the_false_friend_sibling_without_rate_wording(self):
        shapes = _derive_real_tb_pool()
        menu = _run_prune_over(shapes, variable="tuberculosis incidence")
        ids = [s.shape_id for s in menu]
        assert ids.index("sdg_sh_tbs_incd_value_measuredvalue") < ids.index(
            "medicalconditionincident_count_measuredvalue_per_count_person"
        )

    def test_expected_sdg_shape_outranks_the_false_friend_sibling_with_rate_wording(self):
        """Adding "rate" removes the scored count, leaving the two clamped groups to rank."""
        shapes = _derive_real_tb_pool()
        menu = _run_prune_over(shapes, variable="tuberculosis incidence rate")
        ids = [s.shape_id for s in menu]
        assert ids.index("sdg_sh_tbs_incd_value_measuredvalue") < ids.index(
            "medicalconditionincident_count_measuredvalue_per_count_person"
        )
