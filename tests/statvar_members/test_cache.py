"""`MemberEnumerator`'s shared member-group cache: hits, generation keying and lifetime,
query keying, the one member cap, single flight and the group it hands its waiters, uncached
deadline refusals and unsettled groups, cached settled groups beside dismissed inconsistent
candidates, and the weight bound."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from graph_generation.config import (
    GRAPH_CACHE_MAX_AGE_SECONDS,
    GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
)
from graph_generation.stamp import UNKNOWN_KEY
from statvar_members.cache import MemberEnumerator
from statvar_members.enumerate import (
    _REPROBE_BUDGET,
    DEFAULT_MEMBER_CAP,
    MemberEnumerationTooLarge,
    MemberGroup,
    MemberReadDeadline,
)
from statvar_members.types import UNRESOLVED_NODE, FiveTuple

from .conftest import (
    _KENYA_LIFE_EXPECTANCY_ARCS,
    FakeMemberArcReader,
    FakeSubjectPage,
    FlakyMemberArcReader,
    kenya_reader,
)

_KENYA_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)
_UNBOUND = {"gender": ("unbound",)}
_COUNT_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)


def _thirty_bucket_reader() -> FakeMemberArcReader:
    arcs = {
        f"Count_Person_Bucket{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        }
        for i in range(30)
    }
    return FakeMemberArcReader(arcs=arcs)


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _kenya(
    enumerator: MemberEnumerator,
    reader: FakeMemberArcReader,
    *,
    stamp_key: str = "rev1",
    **query: Any,
) -> MemberGroup:
    query.setdefault("slots", _UNBOUND)
    return enumerator.enumerate(
        graph=reader,
        stamp_key=stamp_key,
        five_tuple=_KENYA_FIVE_TUPLE,
        **query,
    )


def test_member_group_cache_hit_skips_every_graph_call():
    reader = kenya_reader()
    enumerator = MemberEnumerator()
    _kenya(enumerator, reader)
    calls_after_first = dict(reader.calls)

    _kenya(enumerator, reader)

    assert reader.calls == calls_after_first


def test_a_new_stamp_key_walks_again_and_the_old_key_still_hits():
    reader = kenya_reader()
    enumerator = MemberEnumerator()
    _kenya(enumerator, reader, stamp_key="rev1")

    _kenya(enumerator, reader, stamp_key="rev2")
    assert reader.calls["node_subjects"] == 2

    _kenya(enumerator, reader, stamp_key="rev1")
    assert reader.calls["node_subjects"] == 2


@pytest.mark.parametrize(
    ("stamp_key", "lifetime"),
    [
        ("rev1", GRAPH_CACHE_MAX_AGE_SECONDS),
        (UNKNOWN_KEY, GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS),
        (f"{UNKNOWN_KEY}@build7", GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS),
    ],
)
def test_an_entry_lives_for_its_stamp_keys_lifetime(stamp_key: str, lifetime: int):
    reader = kenya_reader()
    clock = _FakeClock()
    enumerator = MemberEnumerator(timer=clock)
    _kenya(enumerator, reader, stamp_key=stamp_key)

    clock.now = lifetime - 1
    _kenya(enumerator, reader, stamp_key=stamp_key)
    assert reader.calls["node_subjects"] == 1

    clock.now = lifetime + 1
    _kenya(enumerator, reader, stamp_key=stamp_key)
    assert reader.calls["node_subjects"] == 2


@pytest.mark.parametrize(
    "second",
    [
        {"slots": {"gender": ("value", "Male")}},
        {"readings": {"gender": "not_declared"}},
        {"extra_candidates": ("LifeExpectancy_Person_Male",)},
        {"role_keys": ("observationAbout",)},
        {"open_role_keys": ("origin",)},
        {"subject_ceiling": 20_000},
    ],
    ids=[
        "slots",
        "readings",
        "extra_candidates",
        "role_keys",
        "open_role_keys",
        "subject_ceiling",
    ],
)
def test_each_query_field_keys_its_own_entry(second: dict[str, Any]):
    reader = kenya_reader()
    enumerator = MemberEnumerator()
    _kenya(enumerator, reader)

    _kenya(enumerator, reader, **second)

    assert reader.calls["node_subjects"] == 2


def test_the_full_read_deadline_stays_out_of_the_key():
    reader = kenya_reader()
    enumerator = MemberEnumerator()
    _kenya(enumerator, reader, deadline=time.monotonic() + 60)

    _kenya(enumerator, reader, deadline=time.monotonic() + 60, full_read_deadline=1e12)

    assert reader.calls["node_subjects"] == 1


def test_the_enumerator_walks_under_its_one_member_cap():
    assert MemberEnumerator().member_cap == DEFAULT_MEMBER_CAP
    enumerator = MemberEnumerator(member_cap=25)

    with pytest.raises(MemberEnumerationTooLarge) as exc_info:
        enumerator.enumerate(
            graph=_thirty_bucket_reader(), stamp_key="rev1", five_tuple=_COUNT_FIVE_TUPLE, slots={}
        )

    assert (exc_info.value.count, exc_info.value.member_cap) == (30, 25)


def test_a_deadline_refusal_is_not_cached():
    reader = kenya_reader()
    enumerator = MemberEnumerator()

    with pytest.raises(MemberReadDeadline):
        _kenya(enumerator, reader, deadline=time.monotonic() - 1)
    group = _kenya(enumerator, reader)

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person"]
    assert reader.calls["node_subjects"] == 1


def test_a_group_whose_reads_left_a_candidate_undismissed_is_not_cached():
    """A forward read that drops a member's arcs leaves it inconsistent, and without a root
    nothing dismisses it. The group is a fact of that read, not of the generation, so the
    next call reads the graph again."""
    flaky = FlakyMemberArcReader(
        arcs=_KENYA_LIFE_EXPECTANCY_ARCS, empty_always=["LifeExpectancy_Person_Male"]
    )
    healthy = kenya_reader()
    enumerator = MemberEnumerator()
    male = {"gender": ("value", "Male")}

    unsettled = _kenya(enumerator, flaky, slots=male)
    assert unsettled.undismissed_candidates == ("LifeExpectancy_Person_Male",)
    settled = _kenya(enumerator, healthy, slots=male)

    assert [member.stat_var for member in settled.members] == ["LifeExpectancy_Person_Male"]
    assert healthy.calls["node_subjects"] == 1


def _sector_reader() -> FakeMemberArcReader:
    """A GDP root under an unresolved denominator beside one more sector series than
    `_REPROBE_BUDGET` re-probes. Each sector series reads no denominator, so it is an
    absence suspect, and the ones past the budget stay inconsistent."""
    shared = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("EconomicActivity",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "observationProperties": (),
        "activitySource": ("GrossDomesticProduction",),
    }
    arcs = {
        "Amount_Gdp_Nominal_PerCapita": {
            **shared,
            "measurementDenominator": (UNRESOLVED_NODE,),
            "constraintProperties": ("activitySource",),
        },
        **{
            f"Amount_Gdp_Nominal_Sector{i:02d}": {
                **shared,
                "measurementDenominator": (),
                "constraintProperties": ("activitySource", "economicSector"),
                "economicSector": (f"Sector{i:02d}",),
            }
            for i in range(_REPROBE_BUDGET + 1)
        },
    }
    return FakeMemberArcReader(arcs=arcs)


def test_a_settled_group_with_inconsistent_candidates_its_root_dismisses_is_cached():
    """Candidates the walk could not re-probe stay inconsistent, and a confirmed root whose
    keys they exceed dismisses each one, so the group is settled and the next call reads it
    from the cache. Unstored, every concurrent walk of the key would run in turn behind the
    single-flight lock."""
    reader = _sector_reader()
    enumerator = MemberEnumerator()

    def walk() -> MemberGroup:
        return enumerator.enumerate(
            graph=reader,
            stamp_key="rev1",
            five_tuple=FiveTuple(
                population_type="EconomicActivity",
                measured_property="amount",
                stat_type="measuredValue",
                measurement_qualifier="Nominal",
                denominator_unresolved=True,
            ),
            slots={"activitySource": ("value", "GrossDomesticProduction")},
        )

    first = walk()
    calls_after_first = dict(reader.calls)
    assert first.inconsistent_candidates
    assert first.undismissed_candidates == ()
    assert [member.stat_var for member in first.members] == ["Amount_Gdp_Nominal_PerCapita"]

    assert walk() is first
    assert reader.calls == calls_after_first


def test_the_weight_bound_evicts_the_least_recently_used_group():
    reader = _thirty_bucket_reader()
    # Each group weighs 31, one plus its 30-dcid family, so the bound holds one.
    enumerator = MemberEnumerator(max_weight=40)

    def walk(stamp_key: str) -> None:
        enumerator.enumerate(
            graph=reader, stamp_key=stamp_key, five_tuple=_COUNT_FIVE_TUPLE, slots={}
        )

    walk("rev1")
    walk("rev2")
    walk("rev2")
    assert reader.calls["node_subjects"] == 2

    walk("rev1")
    assert reader.calls["node_subjects"] == 3


class _GatedReader(FlakyMemberArcReader):
    """A reader of the Kenya arcs whose reverse reads wait for `release`, so a walk can be
    held in flight. `empty_always` empties those dcids' forward reads."""

    def __init__(self, *, empty_always: Sequence[str] = ()) -> None:
        super().__init__(arcs=_KENYA_LIFE_EXPECTANCY_ARCS, empty_always=empty_always)
        self.entered = threading.Event()
        self.release = threading.Event()

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> FakeSubjectPage:
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        self.entered.set()
        assert self.release.wait(timeout=5)
        return page


def test_two_concurrent_misses_on_one_key_run_one_enumeration():
    """The second call starts while the first is inside its walk. With single flight it waits
    for the first and reads its group; without, it reaches the reader within the 0.2 s the
    test holds the first walk open."""
    reader = _GatedReader()
    enumerator = MemberEnumerator()
    results: list[MemberGroup] = []

    def call() -> None:
        results.append(_kenya(enumerator, reader))

    first = threading.Thread(target=call)
    first.start()
    assert reader.entered.wait(timeout=5)
    second = threading.Thread(target=call)
    second.start()
    time.sleep(0.2)
    reader.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert reader.calls["node_subjects"] == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_a_call_waiting_on_an_uncached_walk_takes_its_group():
    """A group with an undismissed candidate is not stored, yet the call that waited on its
    walk takes it rather than walking in turn behind the flight lock."""
    reader = _GatedReader(empty_always=["LifeExpectancy_Person_Male"])
    enumerator = MemberEnumerator()
    male = {"gender": ("value", "Male")}
    results: list[MemberGroup] = []

    def call() -> None:
        results.append(_kenya(enumerator, reader, slots=male))

    first = threading.Thread(target=call)
    first.start()
    assert reader.entered.wait(timeout=5)
    second = threading.Thread(target=call)
    second.start()
    time.sleep(0.2)
    reader.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert reader.calls["node_subjects"] == 1
    assert len(results) == 2
    assert results[0] is results[1]
    assert results[0].undismissed_candidates == ("LifeExpectancy_Person_Male",)


def test_a_waiting_call_refuses_at_its_own_deadline():
    reader = _GatedReader()
    enumerator = MemberEnumerator()
    first = threading.Thread(target=lambda: _kenya(enumerator, reader))
    first.start()
    assert reader.entered.wait(timeout=5)

    with pytest.raises(MemberReadDeadline) as exc_info:
        _kenya(enumerator, reader, deadline=time.monotonic() + 0.05)

    reader.release.set()
    first.join(timeout=5)
    assert (exc_info.value.candidates_considered, exc_info.value.candidate_cap) == (0, 4_000)
    assert reader.calls["node_subjects"] == 1


def test_each_walk_logs_one_info_line(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="statvar_members.cache")
    enumerator = MemberEnumerator()
    reader = kenya_reader()

    _kenya(enumerator, reader, spec_id="sp_kenya")
    _kenya(enumerator, reader, spec_id="sp_kenya")

    [record] = [r for r in caplog.records if r.name == "statvar_members.cache"]
    assert record.levelno == logging.INFO
    assert isinstance(record.args, tuple)
    spec_id, five_tuple, truncated, proven, *_, full_read, _, _, outcome, _ = record.args
    assert (spec_id, five_tuple, truncated, proven, full_read, outcome) == (
        "sp_kenya",
        "Person/lifeExpectancy/measuredValue/-/-",
        False,
        True,
        False,
        "settled",
    )
