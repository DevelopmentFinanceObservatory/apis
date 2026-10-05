"""`MemberEnumerator`'s member-group TTL cache: hit, expiry, and independent
keying across distinct slot bindings."""

from __future__ import annotations

import inspect

import pytest

from statvar_members.cache import MemberEnumerator
from statvar_members.enumerate import MemberEnumerationTooLarge
from statvar_members.types import FiveTuple

from .conftest import FakeMemberArcReader, kenya_reader

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


def test_member_group_cache_hit_skips_every_graph_call():
    reader = kenya_reader()
    enumerator = MemberEnumerator(graph=reader)
    enumerator.enumerate(five_tuple=_KENYA_FIVE_TUPLE, slots=_UNBOUND)
    calls_after_first = dict(reader.calls)

    enumerator.enumerate(five_tuple=_KENYA_FIVE_TUPLE, slots=_UNBOUND)

    assert reader.calls == calls_after_first


def test_member_group_cache_expires_after_its_ttl():
    reader = kenya_reader()
    clock = _FakeClock()
    enumerator = MemberEnumerator(graph=reader, clock=clock, member_group_ttl=60.0)
    enumerator.enumerate(five_tuple=_KENYA_FIVE_TUPLE, slots=_UNBOUND)
    calls_after_first = dict(reader.calls)

    clock.now += 61.0
    enumerator.enumerate(five_tuple=_KENYA_FIVE_TUPLE, slots=_UNBOUND)

    # One batch re-fetch plus one individual re-verify of the sole
    # antichain winner's constraint set.
    assert reader.calls["node_arcs_many"] == calls_after_first["node_arcs_many"] + 2


def test_different_slot_bindings_are_cached_independently():
    reader = kenya_reader()
    enumerator = MemberEnumerator(graph=reader)

    male_only = enumerator.enumerate(
        five_tuple=_KENYA_FIVE_TUPLE, slots={"gender": ("value", "Male")}
    )
    unbound = enumerator.enumerate(five_tuple=_KENYA_FIVE_TUPLE, slots=_UNBOUND)

    assert [member.stat_var for member in male_only.members] == ["LifeExpectancy_Person_Male"]
    assert [member.stat_var for member in unbound.members] == ["LifeExpectancy_Person"]


def test_stricter_cap_reenumerates_and_refuses_rather_than_reusing_looser_group():
    reader = _thirty_bucket_reader()
    enumerator = MemberEnumerator(graph=reader)

    group = enumerator.enumerate(five_tuple=_COUNT_FIVE_TUPLE, slots={}, member_cap=40)
    assert len(group.members) == 30

    with pytest.raises(MemberEnumerationTooLarge) as exc_info:
        enumerator.enumerate(five_tuple=_COUNT_FIVE_TUPLE, slots={}, member_cap=25)

    assert exc_info.value.count == 30
    assert exc_info.value.member_cap == 25


def test_enumerate_member_cap_default_is_700():
    # Mirrors statvar_members.enumerate.enumerate_members' own default --
    # see that function's test for the benchmark this value is sized against.
    assert inspect.signature(MemberEnumerator.enumerate).parameters["member_cap"].default == 700


def test_different_extra_candidates_are_cached_independently():
    # Two different posted seeds over the same five_tuple/slots are two
    # different asks -- `extra_candidates` rides in the cache key, so the
    # second call re-reads the graph rather than reusing the first seed's
    # answer.
    reader = kenya_reader()
    enumerator = MemberEnumerator(graph=reader)
    enumerator.enumerate(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots=_UNBOUND,
        extra_candidates=("LifeExpectancy_Person_Male",),
    )
    calls_after_first = dict(reader.calls)

    enumerator.enumerate(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots=_UNBOUND,
        extra_candidates=("LifeExpectancy_Person_Female",),
    )

    assert reader.calls["node_arcs_many"] > calls_after_first["node_arcs_many"]


def test_extra_candidates_cache_hit_skips_every_graph_call():
    reader = kenya_reader()
    enumerator = MemberEnumerator(graph=reader)
    enumerator.enumerate(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots=_UNBOUND,
        extra_candidates=("LifeExpectancy_Person_Male",),
    )
    calls_after_first = dict(reader.calls)

    enumerator.enumerate(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots=_UNBOUND,
        extra_candidates=("LifeExpectancy_Person_Male",),
    )

    assert reader.calls == calls_after_first
