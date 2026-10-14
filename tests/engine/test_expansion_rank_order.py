"""Checks topic-expansion rank order, score clamping, and bind-menu admission."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import cast

import pytest

from code_index.shared import get_index
from entity_resolution import Admission, EntityResolver
from origin_catalogue import BASE, UN_SYSTEM
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.events import Event, FrameEvent
from search.engine.interpret import Recall
from search.engine.interpretation import Interpretation
from search.engine.region import phase_rank_bind
from search.engine.region.phase_rank_bind import (
    _declined_terminal,
    _phase_prune_and_rank,
    _refused_on_pinned,
)
from search.engine.region.pipeline import _un_pass_skip
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState
from search.engine.retrieval import derive
from search.engine.retrieval.derive import _EXPANSION_RANK_EPSILON, derive_shapes
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._builders import place_sets
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._quantity import QuantityReadOnlyLLM
from tests.engine._source_scopes import part_scope
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
)

# derive_shapes: expansion rank stamping and pool order

_STRONG_ROOT = "dc/topic/Strong"
_WDI = "dc/base/WorldDevelopmentIndicators"
_EUROSTAT = "dc/base/EurostatData_GDP"
_WEAK_ROOT = "dc/topic/Weak"


@pytest.fixture(autouse=True)
def _every_topic_root_qualifies(search_settings: pytest.MonkeyPatch) -> None:
    """Ensures both fixture Topics expand so tests can compare their ranks."""
    search_settings(topic_expand_margin=1.0)


def _two_root_graph(weak_population: str = "Household") -> FakeGraph:
    """Builds two topic groups with distinct populations and two recalled candidates. With
    `weak_population` "Country" both roots reach one five-tuple, so one group holds their
    leaves together."""
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
        "WeakExpanded_1": {"label": "w1", "arcs": _std_sv_arcs(weak_population, "count")},
        "WeakExpanded_2": {"label": "w2", "arcs": _std_sv_arcs(weak_population, "count")},
        "RecallHigh_SV": {"label": "high", "arcs": _std_sv_arcs("Person", "count")},
        "RecallLow_SV": {"label": "low", "arcs": _std_sv_arcs("Institution", "count")},
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _derive_two_roots(
    strong_root_score: float = 0.8, *, weak_population: str = "Household"
) -> list[ShapeDraft]:
    return derive_shapes(
        confirmed_svs=[
            _STRONG_ROOT,
            _WEAK_ROOT,
            "RecallHigh_SV",
            "RecallLow_SV",
        ],
        graph=_two_root_graph(weak_population),
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
    root: str | None = None,
) -> ShapeDraft:
    """A shape. `root` is the topic root an unscored shape was reached through; its root
    score is `rank`."""
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
        expansion_root_score=rank if root is not None else None,
        expansion_root=root,
    )


def _scored(count: int, *, start: float = 0.9) -> list[ShapeDraft]:
    """`count` scored shapes `s0` onwards, each 0.01 under the one before, from `start`."""
    return [_shape(f"s{i}", score=start - i / 100) for i in range(count)]


def _leaves(count: int, root_score: float) -> list[ShapeDraft]:
    """`count` leaves `leaf0` onwards of one root recalled at `root_score`."""
    return [
        _shape(f"leaf{i}", score=None, rank=root_score, root=_STRONG_ROOT) for i in range(count)
    ]


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


def _run_prune(
    shapes_found: list[ShapeDraft], *, variable: str = "total population of Kenya"
) -> list[ShapeDraft]:
    state = _state(shapes_found, variable=variable)
    result = asyncio.run(_phase_prune_and_rank(state))
    assert result is None, "expected _phase_prune_and_rank to fall through"
    return state.capped_shapes


class TestMenuSeatsWholeRoots:
    """Eight scored shapes and two topic roots of two leaves each compete for `bind_top_k`.

    The 0.85 root stands behind the 0.90 shape and seats both leaves, five more scored
    shapes fill the menu, and the 0.15 root is never reached. All scores stay above the
    unknown-score threshold.
    """

    def _menu(self) -> list[ShapeDraft]:
        scored = [
            _shape(f"scored{i}", score=score)
            for i, score in enumerate((0.90, 0.84, 0.78, 0.74, 0.70, 0.68, 0.67, 0.66))
        ]
        expansion = [
            _shape("strongA", score=None, rank=0.85, root=_STRONG_ROOT),
            _shape("strongB", score=None, rank=0.85, root=_STRONG_ROOT),
            _shape("weakA", score=None, rank=0.15, root=_WEAK_ROOT),
            _shape("weakB", score=None, rank=0.15, root=_WEAK_ROOT),
        ]
        return _run_prune(scored + expansion)

    def test_the_menu_seats_a_root_whole_at_its_rank(self):
        menu = self._menu()
        assert len(menu) == settings().bind_top_k
        assert [s.shape_id for s in menu] == [
            "scored0",
            "strongA",
            "strongB",
            "scored1",
            "scored2",
            "scored3",
            "scored4",
            "scored5",
        ]

    def test_no_expansion_shape_carries_a_numeric_score(self):
        leaves = [s for s in self._menu() if s.shape_id.startswith(("strong", "weak"))]
        assert len(leaves) == 2
        for leaf in leaves:
            assert leaf.representative_score is None
            assert leaf.expansion_rank_score is not None

    def test_a_refused_scored_shape_caps_no_root(self):
        """The ceiling on a root's rank is the best admissible scored shape. "HIV
        prevalence" refuses the incidence count at 0.80 on its measure class, so the root at
        0.75, ranked 0.7999 by derive, sits behind the prevalence series at 0.70. The
        ranked list the menu is seated from stays in rank order."""
        shapes = [
            dataclasses.replace(
                _shape("incidence", score=0.80), pop_type_name="Number of new HIV infections"
            ),
            _shape("topic_leaf", score=None, rank=0.75, root=_STRONG_ROOT),
            _shape("prevalence", score=0.70),
        ]
        shapes[1] = dataclasses.replace(shapes[1], expansion_rank_score=0.7999)
        state = _state(shapes, variable="HIV prevalence")
        assert asyncio.run(_phase_prune_and_rank(state)) is None
        assert [s.shape_id for s in state.capped_shapes] == ["prevalence", "topic_leaf"]
        assert [s.shape_id for s in state.ranked_shapes] == ["topic_leaf", "prevalence"]

    def test_the_not_electable_list_lists_a_root_whole_or_not_at_all(self):
        """Every reading refused on a form: the flow and its ratio are listed with the
        small root between them, and the nine leaves of the large root, which rank order
        alone lists ahead of the flow, are left out."""
        large = [_shape(f"large{i}", score=None, rank=0.6254, root=_STRONG_ROOT) for i in range(9)]
        small = [_shape(f"small{i}", score=None, rank=0.60, root=_WEAK_ROOT) for i in range(2)]
        refused = [
            Interpretation(shape=shape, reason="denominator_not_available", pinned_failure="form")
            for shape in (
                *large,
                _shape("flow", score=0.6224),
                *small,
                _shape("ratio", score=0.5962),
            )
        ]
        measure_class_refusal = Interpretation(
            shape=_shape("other_class", score=0.9), reason="variable_not_resolved"
        )
        seated = _refused_on_pinned([*refused, measure_class_refusal], None)
        assert [i.shape.shape_id for i in seated] == ["flow", "small0", "small1", "ratio"]
        assert all(i.pinned_failure == "form" for i in seated)

    def _oversized_root(self, root_score: float = 0.80, leaves: int = 9) -> list[ShapeDraft]:
        return [
            _shape(f"leaf{i}", score=None, rank=root_score, root=_STRONG_ROOT)
            for i in range(leaves)
        ]

    @pytest.mark.parametrize(
        ("scenario", "outcome"),
        [
            pytest.param("refused", None, id="a-reading-refused-on-a-code"),
            pytest.param("nothing-refused", "variable_not_resolved", id="nothing-refused"),
            pytest.param("scoped", "qualifier_not_bound", id="a-scope-removed-a-reading"),
            pytest.param("excluded", "qualifier_not_bound", id="an-exclusion-removed-a-reading"),
            pytest.param(
                "excluded-and-refused",
                "qualifier_not_bound",
                id="an-exclusion-removed-a-reading-and-another-was-refused",
            ),
            pytest.param(
                "excluded-and-outranked",
                "qualifier_not_bound",
                id="an-exclusion-removed-the-reading-an-in-scope-one-lost-to",
            ),
            pytest.param(
                "excluded-leaf",
                "variable_not_resolved",
                id="an-exclusion-removed-one-leaf-of-the-root-that-seats-nothing",
            ),
        ],
    )
    def test_a_part_seating_nothing_ends_as_a_part_its_prunes_left_no_reading(
        self, monkeypatch, scenario, outcome
    ):
        """A part whose admissible readings are only the leaves of a root larger than the
        menu offers none, so it holds nothing they read and ends as the same part with
        those leaves below the score floor: field for field, frame event for frame event.

        "aid to Kenya, not imputed" refuses its flow on a code, so the menu is empty, the
        flow is the not-electable list and bind runs. With nothing refused the part
        refuses `variable_not_resolved`. A part whose named source removed a reading
        refuses `qualifier_not_bound` naming the source (`_declined_terminal`). A part
        whose exclusion removed a reading refuses on the exclusion before bind, with one
        frame event and the scope marked as settling the answer, whether or not another
        reading was refused. It holds what its eligible readings read, an in-scope reading
        that lost the family election to the excluded one included. An exclusion that
        removes only a leaf of the root that seats nothing took no reading the menu would
        have seated, so the part refuses `variable_not_resolved` and the scope settles
        nothing, as with no exclusion."""
        refuses = scenario in ("refused", "excluded-and-refused")
        excluded_shape = "leaf0" if scenario == "excluded-leaf" else "outside"
        excludes = scenario.startswith("excluded") and excluded_shape == "outside"
        flow = [_shape("flow", score=0.85)] if refuses else []
        if refuses:
            judge = phase_rank_bind.judge_pinned_parts

            def refusing_the_flow(state, interpretations, matches):
                return [
                    Interpretation(
                        shape=i.shape, reason="qualifier_not_bound", pinned_failure="pinned_code"
                    )
                    if i.shape.shape_id == "flow"
                    else i
                    for i in judge(state, interpretations, matches)
                ]

            monkeypatch.setattr(phase_rank_bind, "judge_pinned_parts", refusing_the_flow)
        outside = [_shape("outside", score=0.9)] if scenario == "scoped" or excludes else []
        if scenario == "excluded-and-outranked":
            # An in-scope reading with a vocabulary of its own, which the excluded reading
            # beats in the five-tuple election.
            sector = SlotKeyDraft(axis="how", property_dcid="economicSector", label="sector")
            outside.append(dataclasses.replace(_shape("inside", score=0.7), slot_keys=(sector,)))
            elect = phase_rank_bind.elect_five_tuple_family
            monkeypatch.setattr(
                phase_rank_bind,
                "elect_five_tuple_family",
                lambda shapes, phrases_read, **kwargs: [
                    s for s in elect(shapes, phrases_read, **kwargs) if s.shape_id != "inside"
                ],
            )
        dataset = next(iter(get_index().datasets))
        scope = None
        if outside or scenario == "excluded-leaf":
            scope = part_scope(named=dataset) if scenario == "scoped" else None
            scope = scope or part_scope(excluded=[dataset])
            # The scope holds every shape but `outside`, and its exclusion is not idle.
            monkeypatch.setattr(
                phase_rank_bind,
                "_dataset_candidates",
                lambda shapes, given: [
                    s for s in shapes if given is None or s.shape_id != excluded_shape
                ],
            )
            monkeypatch.setattr(phase_rank_bind, "with_idle_exclusions", lambda given, _: given)

        # Ten leaves where the exclusion takes one, so nine are left and still seat nothing.
        leaves = 10 if scenario == "excluded-leaf" else 9
        seating_nothing, result, sent = self._pruned(
            [*outside, *flow, *self._oversized_root(leaves=leaves)], scope
        )
        floored, floored_result, sent_floored = self._pruned(
            [*outside, *flow, *self._oversized_root(root_score=0.60, leaves=leaves)], scope
        )

        assert seating_nothing.capped_shapes == []
        assert seating_nothing.interpretations.admissible == ()
        if outcome is None:
            assert result is None
            refused = seating_nothing.interpretations.refused_on_pinned
            assert [i.shape.shape_id for i in refused] == ["flow"]
            assert len(sent) == 1
        else:
            assert result is not None
            assert (result.status, result.no_data_reason) == ("no_data", outcome)
            assert len(sent) == (1 if excludes else 0)
            assert seating_nothing.scope_settled is excludes
        assert sent == sent_floored
        if scenario != "scoped":
            assert (result is None) == (floored_result is None)
            if result is not None and floored_result is not None:
                assert result.no_data_reason == floored_result.no_data_reason
        # Left out: the shapes, recall and clients the parts are given; the judgement of
        # each shape, which the leaves pass here and miss on the floor there, with the codes
        # read in the vocabularies the pinned parts are judged in and the memo of each
        # judged shape's codes; and the clock.
        given = {"shapes_found", "rcl", "llm", "coverage", "emit"}
        per_shape = {
            "interpretations",
            "pinned_part_matches",
            "shape_held_codes",
            "timing",
            "shape_t0",
        }
        differing = {
            spec.name
            for spec in dataclasses.fields(seating_nothing)
            if spec.name not in given | per_shape
            and getattr(seating_nothing, spec.name) != getattr(floored, spec.name)
        }
        assert differing == set()

    @pytest.mark.parametrize(
        ("shapes", "excluded", "removed", "binds"),
        [
            pytest.param(_scored(9), {"s8"}, False, True, id="a-shape-past-the-last-seat"),
            pytest.param(_scored(10), {"s8"}, False, True, id="the-first-shape-past-the-last-seat"),
            pytest.param(_scored(9), {"s0"}, True, True, id="a-seated-shape-the-next-replaces"),
            pytest.param(_scored(9), {"s7"}, True, True, id="the-shape-in-the-last-seat"),
            pytest.param(_scored(8), {"s7"}, True, True, id="the-last-of-exactly-a-menu"),
            pytest.param(_scored(1), {"s0"}, True, False, id="the-only-shape"),
            pytest.param(_scored(3), {"s0", "s1", "s2"}, True, False, id="every-scored-shape"),
            pytest.param(
                [*_scored(3), *_leaves(3, 0.895)],
                {"leaf0"},
                True,
                True,
                id="a-leaf-of-a-root-seated-whole",
            ),
            pytest.param(
                [*_scored(8), *_leaves(5, 0.895)],
                {"leaf0"},
                True,
                True,
                id="a-leaf-of-a-seated-root-whose-seat-the-next-shape-takes",
            ),
            pytest.param(
                [*_scored(8), *_leaves(9, 0.895)],
                {"leaf0", "leaf1"},
                True,
                True,
                id="leaves-whose-loss-seats-a-root-that-displaces-seated-shapes",
            ),
            pytest.param(
                _leaves(9, 0.80), {"leaf0"}, False, True, id="a-leaf-whose-loss-seats-its-root"
            ),
            pytest.param(
                _leaves(9, 0.80),
                {f"leaf{i}" for i in range(9)},
                False,
                False,
                id="every-leaf-of-a-root-larger-than-the-menu",
            ),
            pytest.param(
                [*_scored(4), *_leaves(8, 0.70)],
                {"s0", "s1", "s2", "s3"},
                True,
                True,
                id="every-seated-shape-with-a-root-taking-the-freed-seats",
            ),
            pytest.param(
                [*_scored(4), *_leaves(8, 0.50)],
                {"s0", "s1", "s2", "s3"},
                True,
                False,
                id="every-seated-shape-beside-a-root-under-the-floor",
            ),
            pytest.param(
                [*_scored(1), *_leaves(9, 0.50)],
                {"s0"},
                True,
                False,
                id="the-seated-shape-beside-a-root-larger-than-the-menu",
            ),
            pytest.param(
                [*_scored(8, start=0.7), _shape("top", score=0.95), *_leaves(2, 0.9)],
                {"top"},
                True,
                True,
                id="the-first-shape-above-a-seated-root",
            ),
        ],
    )
    def test_a_declining_bind_settles_only_an_exclusion_that_removed_an_offered_reading(
        self, monkeypatch, bind_catalogue, shapes, excluded, removed, binds
    ):
        """A menu of eight over scored readings and the leaves of one root. The exclusion
        removed a reading when the menu would have offered it without the exclusion and
        does not offer it with it (`ResolveState.scope_removed_readings`). The part then
        refuses on the exclusion, before bind where what is left seats nothing and else
        when bind declines what is left: settled, with the UN pass skipped under a scope
        allowing a UN System Data Commons source. Where the exclusion removed nothing the
        part refuses `variable_not_resolved`, unsettled, and the UN pass runs.

        Of nine scored readings the ninth is offered neither way, so excluding it removed
        nothing. The first and the eighth are offered, as is a leaf of a root seated
        whole. A nine-leaf root seats nothing beside eight scored readings; losing two
        leaves it seats, and displaces seven readings the part was offered, so the
        exclusion removed those. A nine-leaf root alone seats nothing: losing one leaf
        gives the part a menu and costs it nothing, and losing every leaf leaves it as it
        was with no exclusion."""
        un_source = "undata/p/Agency"
        bind_catalogue(FakeCatalogue(provenances={un_source: provenance_row(un_source, UN_SYSTEM)}))
        dataset = next(iter(get_index().datasets))
        scope = part_scope(excluded=[dataset])
        assert scope is not None
        scope = dataclasses.replace(scope, allowed=scope.allowed | {un_source})
        monkeypatch.setattr(
            phase_rank_bind,
            "_dataset_candidates",
            lambda found, given: [s for s in found if given is None or s.shape_id not in excluded],
        )
        monkeypatch.setattr(phase_rank_bind, "with_idle_exclusions", lambda given, _: given)

        state, result, _sent = self._pruned(list(shapes), scope)

        assert state.scope_removed_readings is removed
        assert excluded.isdisjoint(s.shape_id for s in state.capped_shapes)
        assert (result is None) is binds
        assert bool(state.capped_shapes) is binds
        declined = result or _declined_terminal(state)
        assert declined.no_data_reason == (
            "qualifier_not_bound" if removed else "variable_not_resolved"
        )
        assert state.scope_settled is removed
        assert _un_pass_skip(declined, state) == ("source_scope" if removed else None)

    def _pruned(
        self, shapes: list[ShapeDraft], scope=None
    ) -> tuple[ResolveState, RegionResult | None, list[FrameEvent]]:
        """The state `_phase_prune_and_rank` leaves for `shapes`, its result, and the frame
        events it sent."""
        events: list[Event] = []

        async def collect(event: Event) -> None:
            events.append(event)

        state = _state(shapes)
        state.rcl = Recall(candidates=[], resolutions={})
        state.scope = scope
        state.emit = collect
        result = asyncio.run(_phase_prune_and_rank(state))
        return state, result, [event for event in events if isinstance(event, FrameEvent)]


def _judged(state) -> list[str]:
    """The shapes past the score floor, which the admissibility rules judged."""
    return [i.shape.shape_id for i in state.interpretations.interpretations]


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
        assert _judged(state) == ["tb_incidence"]

    def test_root_under_the_floor_is_refused(self):
        expansion = dataclasses.replace(
            _shape("weak_topic", score=None, rank=0.62, meas_prop="value"),
            expansion_root_score=0.62,
        )
        state = _state([_shape("scored", score=0.70), expansion])
        assert asyncio.run(_phase_prune_and_rank(state)) is None
        assert [s.shape_id for s in state.capped_shapes] == ["scored"]
        assert _judged(state) == ["scored"]

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
        assert strong.expansion_root == _STRONG_ROOT
        assert strong.measured_score == 1.0
        weak = _shape_with(shapes, "WeakExpanded_1")
        assert (weak.expansion_root_score, weak.expansion_root) == (0.5, _WEAK_ROOT)
        recalled = _shape_with(shapes, "RecallHigh_SV")
        assert recalled.expansion_root_score is None
        assert recalled.expansion_root is None
        assert recalled.measured_score == 0.9

    def test_a_group_of_two_roots_leaves_takes_the_stronger_root(self):
        shapes = _derive_two_roots(weak_population="Country")
        group = _shape_with(shapes, "WeakExpanded_1")
        assert set(group.sv_arc_facts or {}) >= {"StrongExpanded_1", "WeakExpanded_1"}
        assert (group.expansion_root, group.expansion_root_score) == (_STRONG_ROOT, 0.8)

    def test_a_superseding_variable_takes_the_root_of_the_leaf_it_stands_for(self, monkeypatch):
        monkeypatch.setattr(derive, "current_catalogue", lambda: None)
        monkeypatch.setattr(
            derive, "stand_in", lambda sv, *, catalogue: {"Old_Leaf": "New_Leaf"}.get(sv, sv)
        )
        roots = {"Old_Leaf": _STRONG_ROOT, "Weak_Leaf": _WEAK_ROOT}
        ranks = {"Old_Leaf": 0.8, "Weak_Leaf": 0.5}
        pool = derive._stand_pool(
            ["Old_Leaf", "Weak_Leaf"],
            sv_scores={},
            recall_svs=set(),
            expansion_rank=ranks,
            expansion_root_score={"Old_Leaf": 0.8, "Weak_Leaf": 0.5},
            expansion_root=roots,
        )
        assert pool == ["New_Leaf", "Weak_Leaf"]
        assert (roots["New_Leaf"], ranks["New_Leaf"]) == (_STRONG_ROOT, 0.8)


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
        interpretations = state.interpretations.interpretations
        assert sum(i.reason == "variable_not_resolved" for i in interpretations) == refused
        assert len(interpretations) == len(self._shapes())

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


class TestCrossPopulationExpansionStays:
    """An unscored shape from another population than the top-ranked scored shape stays on
    the menu whatever the query's wording: no prune reads flow or aid words, and the model
    elects among the readings admissibility leaves."""

    def _anchor(self) -> ShapeDraft:
        return _shape(
            "crs_flow",
            score=0.74,
            pop_type="FinancialObligation",
            meas_prop="amount",
            declared_roles=_ROLE_BEARING,
        )

    def test_flow_wording_keeps_a_cross_population_standard_shape(self):
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
        assert [s.shape_id for s in menu] == ["crs_flow", "sdg_tof_hlthl"]

    def test_standard_anchor_keeps_a_different_population_unscored_shape(self):
        anchor = _shape("cm_01", score=0.86, pop_type="Person", meas_prop="who/CM_01")
        sdg_mort = _shape(
            "sh_dyn_mort", score=None, rank=0.86, pop_type="SDG_SH_DYN_MORT", meas_prop="value"
        )
        menu = _run_prune([anchor, sdg_mort])
        assert {s.shape_id for s in menu} == {"cm_01", "sh_dyn_mort"}

    def test_same_population_unscored_shape_is_kept(self):
        sibling = _shape(
            "crs_other_prop",
            score=None,
            rank=0.70,
            pop_type="FinancialObligation",
            meas_prop="otherAmount",
        )
        menu = _run_prune([self._anchor(), sibling])
        assert {s.shape_id for s in menu} == {"crs_flow", "crs_other_prop"}


class TestRankKeyOrder:
    """The ranked-shape sort key breaks score ties by breadth."""

    def test_the_broader_shape_sorts_first_on_a_score_tie(self):
        from search.engine.region.phase_rank_bind import _rank_key

        broad = _shape("broad", score=0.74, pop_type="FinancialObligation", meas_prop="amount")
        narrow = _shape("narrow", score=0.74, pop_type="FinancialObligation", meas_prop="amount")
        broad = dataclasses.replace(broad, sv_arc_facts={key: {} for key in ["a", "b", "c"]})
        narrow = dataclasses.replace(narrow, sv_arc_facts={key: {} for key in ["a"]})

        # Breadth breaks the rank tie.
        assert min([narrow, broad], key=_rank_key) is broad

    @pytest.mark.parametrize(
        ("regional_score", "expected"),
        [
            (0.8895, ["per_unresolved", "per_count_person"]),
            (0.89, ["per_count_person", "per_unresolved"]),
        ],
        ids=["tie_wider_first", "higher_score_first"],
    )
    def test_country_breadth_breaks_only_an_exact_score_tie(
        self, bind_catalogue, regional_score, expected
    ):
        """Two readings of one sentence: the one with data in more countries leads on an
        exact tie, and a higher score still leads whatever its breadth."""
        from search.engine.region.phase_rank_bind import _rank_key

        bind_catalogue(
            restrictive_catalogue(
                provenances=[provenance_row(_WDI, BASE), provenance_row(_EUROSTAT, BASE)],
                base_variables={
                    "GdpPerCapita": ([_WDI], ["Earth", "country/KEN", "country/USA"]),
                    "GdpPerCountPerson": (
                        [_EUROSTAT],
                        ["country/DEU", "nuts/DE11", "nuts/DE12", "nuts/DE13"],
                    ),
                },
                scan_places=[
                    "Earth",
                    "country/DEU",
                    "country/KEN",
                    "country/USA",
                    "nuts/DE11",
                    "nuts/DE12",
                    "nuts/DE13",
                ],
            )
        )
        wide = dataclasses.replace(
            _shape("per_unresolved", score=0.8895), sv_arc_facts={"GdpPerCapita": {}}
        )
        regional = dataclasses.replace(
            _shape("per_count_person", score=regional_score),
            sv_arc_facts={"GdpPerCountPerson": {}},
        )

        ranked = sorted([regional, wide], key=_rank_key)

        assert [s.shape_id for s in ranked] == expected


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
