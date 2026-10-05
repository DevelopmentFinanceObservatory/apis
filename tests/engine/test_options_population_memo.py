"""Tests caching, expiry, and single-flight behaviour in `OptionsPopulationMemo`."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from graph_generation import FakeGeneration, GenerationStamp
from search.engine.errors import EngineInputError
from search.engine.options.population import OptionsPopulation
from search.engine.options.population_memo import (
    OptionsPopulationMemo,
    PopulationInFlight,
    _population_key,
)
from search.engine.options.resolve import resolve_options_target
from search.models import (
    BaseIdentity,
    IdentityBindingUnbound,
    IdentityBindingValue,
    IdentityShapeSchema,
    IdentitySlot,
    IdentityWindowAll,
    OptionsFor,
    OptionsRequest,
)
from tests.code_index.aid_identity import (
    AID_TYPE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_PROVENANCE,
    imputed_education_aid_identity,
)
from tests.fixtures import FakeGraph
from tests.statvar_members.conftest import FakeMemberArcReader, kenya_reader

# Mirrors test_options_population.py's own _NON_INDEX_SHAPE: a five-tuple the
# packaged index carries no row for, so options_population sources it via
# statvar_members.enumerate_members -- the one branch that reads `graph`.
_NON_INDEX_SHAPE = IdentityShapeSchema(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)


def _identity(*, entities: dict[str, list[str]] | None = None) -> BaseIdentity:
    return BaseIdentity(
        shape=_NON_INDEX_SHAPE,
        slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())],
        entities=entities or {},
        requested_window=IdentityWindowAll(),
    )


def _graph_calls(graph: FakeMemberArcReader) -> int:
    return graph.calls["node_subjects"] + graph.calls["node_arcs_many"]


class _CountingReader:
    """Counts `node_subjects` calls and can block the first call for single-flight tests."""

    def __init__(self, inner: FakeMemberArcReader, *, block_first: bool = False) -> None:
        self._inner = inner
        self.node_subjects_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._block_first = block_first

    def node_subjects(self, *args, **kwargs):
        self.node_subjects_calls += 1
        if self._block_first and self.node_subjects_calls == 1:
            self.started.set()
            assert self.release.wait(timeout=5), "test deadlocked waiting for release"
        return self._inner.node_subjects(*args, **kwargs)

    def node_arcs_many(self, *args, **kwargs):
        return self._inner.node_arcs_many(*args, **kwargs)


class _FailFirstReader:
    """Raises on the first `node_subjects` call, then delegates."""

    def __init__(self, inner: FakeMemberArcReader) -> None:
        self._inner = inner
        self.node_subjects_calls = 0

    def node_subjects(self, *args, **kwargs):
        self.node_subjects_calls += 1
        if self.node_subjects_calls == 1:
            raise EngineInputError(
                "simulated transport failure", code="request_invalid", status=400
            )
        return self._inner.node_subjects(*args, **kwargs)

    def node_arcs_many(self, *args, **kwargs):
        return self._inner.node_arcs_many(*args, **kwargs)


def test_miss_derives_and_returns_the_real_population():
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    graph = kenya_reader()

    result = memo.get_or_derive(_identity(), graph)

    assert isinstance(result, OptionsPopulation)
    assert result.source == "graph"
    assert result.stat_var_dcids == ("LifeExpectancy_Person",)
    assert _graph_calls(graph) > 0


def test_hit_within_ttl_makes_no_second_derivation():
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    graph = kenya_reader()
    identity = _identity()

    first = memo.get_or_derive(identity, graph)
    calls_after_first = _graph_calls(graph)
    second = memo.get_or_derive(identity, graph)

    assert second == first
    assert _graph_calls(graph) == calls_after_first, "a hit must never touch the graph again"


def test_entry_lifetime_with_injected_clock_forces_a_fresh_derivation():
    """Without a generation provider, entries use the unknown stamp's `max_age_s`."""
    clock = [0.0]
    memo = OptionsPopulationMemo(enabled=True, max_entries=10, clock=lambda: clock[0])
    graph = kenya_reader()
    identity = _identity()
    max_age_s = GenerationStamp(token=None).max_age_s

    memo.get_or_derive(identity, graph)
    calls_after_first = _graph_calls(graph)

    clock[0] = max_age_s - 1  # still within the unknown stamp's own lifetime
    memo.get_or_derive(identity, graph)
    assert _graph_calls(graph) == calls_after_first, "still within lifetime: must stay a hit"

    clock[0] = max_age_s + 1  # past the unknown stamp's own lifetime
    memo.get_or_derive(identity, graph)
    assert _graph_calls(graph) > calls_after_first, "past the lifetime: must re-derive"


def test_concurrent_callers_for_the_same_key_single_flight_to_one_derivation():
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    reader = _CountingReader(kenya_reader(), block_first=True)
    identity = _identity()
    results: list[OptionsPopulation] = []
    results_lock = threading.Lock()

    def worker():
        result = memo.get_or_derive(identity, reader)
        with results_lock:
            results.append(result)

    t1 = threading.Thread(target=worker)
    t1.start()
    assert reader.started.wait(timeout=5), "leader never reached the fake reader"
    t2 = threading.Thread(target=worker)
    t2.start()
    time.sleep(0.05)  # give t2 a chance to park on the in-flight slot
    reader.release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert reader.node_subjects_calls == 1, "two concurrent callers must share one derivation"
    assert len(results) == 2
    assert results[0] == results[1]


def test_a_follower_past_its_deadline_raises_promptly_while_the_owner_keeps_deriving():
    """A follower timeout leaves the owner's derivation running for later callers."""
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    reader = _CountingReader(kenya_reader(), block_first=True)
    identity = _identity()

    def owner():
        memo.get_or_derive(identity, reader)

    t = threading.Thread(target=owner)
    t.start()
    assert reader.started.wait(timeout=5), "owner never reached the fake reader"

    before = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        memo.get_or_derive(identity, reader, deadline=time.monotonic() + 0.05)
    assert time.monotonic() - before < 2, "the follower must not have waited for the owner"

    reader.release.set()
    t.join(timeout=5)

    # The owner's own result landed regardless of the timed-out follower: a
    # later call is a hit, not a fresh derivation.
    calls_after_owner = reader.node_subjects_calls
    result = memo.get_or_derive(identity, reader)
    assert result.stat_var_dcids == ("LifeExpectancy_Person",)
    assert reader.node_subjects_calls == calls_after_owner, "a later call must hit, not re-derive"


def test_join_inflight_false_raises_at_once_without_affecting_the_owner():
    """`join_inflight=False` raises `PopulationInFlight` immediately and leaves the owner running."""
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    reader = _CountingReader(kenya_reader(), block_first=True)
    identity = _identity()

    def owner():
        memo.get_or_derive(identity, reader)

    t = threading.Thread(target=owner)
    t.start()
    assert reader.started.wait(timeout=5), "owner never reached the fake reader"

    before = time.monotonic()
    with pytest.raises(PopulationInFlight):
        memo.get_or_derive(identity, reader, join_inflight=False)
    assert time.monotonic() - before < 1, "a non-joining caller must never wait on the owner"

    reader.release.set()
    t.join(timeout=5)

    # The owner's own result landed regardless of the non-joining caller: a
    # later (joining) call is a hit, not a fresh derivation.
    calls_after_owner = reader.node_subjects_calls
    result = memo.get_or_derive(identity, reader)
    assert result.stat_var_dcids == ("LifeExpectancy_Person",)
    assert reader.node_subjects_calls == calls_after_owner, "a later call must hit, not re-derive"


def test_a_failed_derivation_is_not_cached_and_retries_on_the_next_call():
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    reader = _FailFirstReader(kenya_reader())
    identity = _identity()

    with pytest.raises(EngineInputError):
        memo.get_or_derive(identity, reader)

    result = memo.get_or_derive(identity, reader)

    assert result.stat_var_dcids == ("LifeExpectancy_Person",)
    assert reader.node_subjects_calls == 2, "the failed attempt must not have been cached"


def test_a_waiter_on_a_failing_inflight_derivation_gets_the_same_failure():
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    identity = _identity()
    started = threading.Event()
    release = threading.Event()

    class _FailingSlowReader:
        def node_subjects(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5), "test deadlocked waiting for release"
            raise EngineInputError("boom", code="request_invalid", status=400)

        def node_arcs_many(self, *args, **kwargs):
            raise AssertionError("node_arcs_many must not be reached after node_subjects raises")

    errors: list[BaseException] = []

    def worker():
        try:
            memo.get_or_derive(identity, _FailingSlowReader())
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t1.start()
    assert started.wait(timeout=5)
    t2 = threading.Thread(target=worker)
    t2.start()
    time.sleep(0.05)
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(errors) == 2
    assert all(isinstance(exc, EngineInputError) for exc in errors)


def test_same_shape_and_slots_share_a_key_across_different_anchors():
    """Anchors do not affect the key because population derivation ignores `identity.entities`."""
    kenya = _identity(entities={"observationAbout": ["country/KEN"]})
    ghana = _identity(entities={"observationAbout": ["country/GHA"]})

    assert _population_key(kenya) == _population_key(ghana)

    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    graph = kenya_reader()

    memo.get_or_derive(kenya, graph)
    calls_after_first = _graph_calls(graph)
    memo.get_or_derive(ghana, graph)

    assert _graph_calls(graph) == calls_after_first, "a different anchor must still hit"


def test_a_different_slot_binding_is_a_different_key():
    unbound = _identity()
    bound = BaseIdentity(
        shape=_NON_INDEX_SHAPE,
        slots=[IdentitySlot(property="gender", binding=IdentityBindingValue(ref="Male"))],
        entities={},
        requested_window=IdentityWindowAll(),
    )

    assert _population_key(unbound) != _population_key(bound)


def test_disabled_memo_derives_every_time_with_no_bookkeeping():
    memo = OptionsPopulationMemo(enabled=False)
    graph = kenya_reader()
    identity = _identity()

    memo.get_or_derive(identity, graph)
    calls_after_first = _graph_calls(graph)
    memo.get_or_derive(identity, graph)

    assert _graph_calls(graph) > calls_after_first, "a disabled memo must never cache a hit"


def test_max_entries_below_one_raises():
    with pytest.raises(ValueError, match="max_entries"):
        OptionsPopulationMemo(enabled=True, max_entries=0)


def test_max_entries_evicts_the_least_recently_used_key():
    memo = OptionsPopulationMemo(enabled=True, max_entries=1)
    graph = kenya_reader()
    slot_a = IdentitySlot(property="gender", binding=IdentityBindingUnbound())
    identity_a = BaseIdentity(
        shape=_NON_INDEX_SHAPE, slots=[slot_a], entities={}, requested_window=IdentityWindowAll()
    )
    identity_b = BaseIdentity(
        shape=_NON_INDEX_SHAPE,
        slots=[IdentitySlot(property="gender", binding=IdentityBindingValue(ref="Male"))],
        entities={},
        requested_window=IdentityWindowAll(),
    )

    memo.get_or_derive(identity_a, graph)
    memo.get_or_derive(identity_b, graph)  # evicts identity_a's entry (max_entries=1)
    calls_before_repeat = _graph_calls(graph)
    memo.get_or_derive(identity_a, graph)

    assert _graph_calls(graph) > calls_before_repeat, "identity_a's entry must have been evicted"


def test_two_memos_over_different_readers_never_share_a_population():
    """Separate memo instances keep results from different readers isolated, even for one identity."""
    identity = _identity(entities={"observationAbout": ["country/KEN"]})
    req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})
    graph = FakeGraph(nodes={"country/KEN": {"label": "Kenya", "type": "Country"}})

    reader_a = kenya_reader()
    memo_a = OptionsPopulationMemo(enabled=True, max_entries=10)
    reader_b = FakeMemberArcReader(
        arcs={
            "LifeExpectancy_Person_Alt": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("Person",),
                "measuredProperty": ("lifeExpectancy",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": (),
                "observationProperties": (),
            }
        }
    )
    memo_b = OptionsPopulationMemo(enabled=True, max_entries=10)

    target_a = resolve_options_target(
        req, graph=graph, materialise_graph=reader_a, population_memo=memo_a
    )
    target_b = resolve_options_target(
        req, graph=graph, materialise_graph=reader_b, population_memo=memo_b
    )

    assert target_a.population.stat_var_dcids == ("LifeExpectancy_Person",)
    assert target_b.population.stat_var_dcids == ("LifeExpectancy_Person_Alt",)

    # Each memo's own cache now holds only its own reader's result: a repeat
    # call against a reader that raises on any further call still hits,
    # proving memo_a never absorbed memo_b's entry (or vice versa).
    class _PoisonedReader:
        def node_subjects(self, *args, **kwargs):
            raise AssertionError("must not be called again: this key is already cached")

        def node_arcs_many(self, *args, **kwargs):
            raise AssertionError("must not be called again: this key is already cached")

    repeat_a = resolve_options_target(
        req, graph=graph, materialise_graph=_PoisonedReader(), population_memo=memo_a
    )
    repeat_b = resolve_options_target(
        req, graph=graph, materialise_graph=_PoisonedReader(), population_memo=memo_b
    )
    assert repeat_a.population.stat_var_dcids == target_a.population.stat_var_dcids
    assert repeat_b.population.stat_var_dcids == target_b.population.stat_var_dcids


# ---------------------------------------------------------------------------
# Generation keying: every entry's own key and lifetime come from the
# `graph_generation.GenerationProvider` stamp captured for that call.
# ---------------------------------------------------------------------------


def test_a_repeat_call_under_the_same_token_is_a_hit():
    generation = FakeGeneration(token="gen-a")
    memo = OptionsPopulationMemo(enabled=True, max_entries=10, generation=generation)
    graph = kenya_reader()
    identity = _identity()

    memo.get_or_derive(identity, graph)
    calls_after_first = _graph_calls(graph)
    memo.get_or_derive(identity, graph)

    assert _graph_calls(graph) == calls_after_first, "the same token must serve a hit"


def test_advancing_the_token_is_a_miss_for_an_identical_key():
    generation = FakeGeneration(token="gen-a")
    memo = OptionsPopulationMemo(enabled=True, max_entries=10, generation=generation)
    graph = kenya_reader()
    identity = _identity()

    memo.get_or_derive(identity, graph)
    calls_after_first = _graph_calls(graph)

    generation.advance("gen-b")
    memo.get_or_derive(identity, graph)
    assert _graph_calls(graph) > calls_after_first, "a token change must re-derive"


def test_unknown_and_known_never_share_an_entry():
    generation = FakeGeneration()  # starts unknown
    memo = OptionsPopulationMemo(enabled=True, max_entries=10, generation=generation)
    graph = kenya_reader()
    identity = _identity()

    memo.get_or_derive(identity, graph)
    calls_after_first = _graph_calls(graph)

    generation.advance("gen-a")  # unknown -> known, not itself a token "change"
    memo.get_or_derive(identity, graph)
    assert _graph_calls(graph) > calls_after_first, (
        "unknown and a known token must never share the identical key"
    )


def test_a_post_flip_caller_never_joins_a_pre_flip_in_flight_derivation():
    """Calls under different generation stamps derive independently."""
    generation = FakeGeneration(token="gen-a")
    memo = OptionsPopulationMemo(enabled=True, max_entries=10, generation=generation)
    identity = _identity()
    owner_reader = _CountingReader(kenya_reader(), block_first=True)

    owner_thread = threading.Thread(target=lambda: memo.get_or_derive(identity, owner_reader))
    owner_thread.start()
    assert owner_reader.started.wait(timeout=5), "owner never reached the fake reader"

    generation.advance("gen-b")
    follower_reader = kenya_reader()
    # join_inflight=False, so this caller must not park behind the pre-flip
    # owner at all. Its own (post-flip) key has no owner of its own yet.
    result = memo.get_or_derive(identity, follower_reader, join_inflight=False)
    assert result.stat_var_dcids == ("LifeExpectancy_Person",)

    owner_reader.release.set()
    owner_thread.join(timeout=5)


def test_stats_counts_hits_misses_evictions_and_entries():
    memo = OptionsPopulationMemo(enabled=True, max_entries=1)
    identity_a = BaseIdentity(
        shape=_NON_INDEX_SHAPE,
        slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())],
        entities={},
        requested_window=IdentityWindowAll(),
    )
    identity_b = BaseIdentity(
        shape=_NON_INDEX_SHAPE,
        slots=[IdentitySlot(property="gender", binding=IdentityBindingValue(ref="Male"))],
        entities={},
        requested_window=IdentityWindowAll(),
    )

    memo.get_or_derive(identity_a, kenya_reader())  # miss
    memo.get_or_derive(identity_a, kenya_reader())  # hit
    memo.get_or_derive(identity_b, kenya_reader())  # miss, evicts identity_a's entry

    assert memo.stats() == {"hits": 1, "misses": 2, "evictions": 1, "entries": 1}


def test_clear_drops_every_stored_entry():
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)
    graph = kenya_reader()
    identity = _identity()

    memo.get_or_derive(identity, graph)
    calls_after_first = _graph_calls(graph)

    memo.clear()
    memo.get_or_derive(identity, graph)
    assert _graph_calls(graph) > calls_after_first, "a cleared memo must re-derive on the next call"


@pytest.mark.parametrize("first_bound", [True, False], ids=["bound-first", "unbound-first"])
def test_an_identity_posted_with_or_without_its_pins_derives_one_population(first_bound: bool):
    """The options door reads a posted identity in canonical form, so the imputed identity
    naming its table and binding its invariants and the one leaving them unbound share one
    memo entry and one population."""
    bound = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    unbound = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    for slot in unbound["slots"]:
        if slot["property"] in ("crsTypeOfFlow", AID_TYPE):
            slot["binding"] = {"state": "unbound"}
    unbound["entities"] = bound["entities"] = {}
    posted = [bound, unbound] if first_bound else [unbound, bound]
    memo = OptionsPopulationMemo(enabled=True, max_entries=10)

    targets = [
        resolve_options_target(
            OptionsRequest(
                identity=BaseIdentity.model_validate(document),
                **{"for": OptionsFor(role="destination")},
            ),
            graph=FakeGraph(),
            materialise_graph=FakeMemberArcReader(arcs={}),
            population_memo=memo,
        )
        for document in posted
    ]

    first, second = (target.population for target in targets)
    assert first is second
    assert len(first.stat_var_dcids) == IMPUTED_MEMBER_COUNT
    assert memo.stats()["misses"] == 1
