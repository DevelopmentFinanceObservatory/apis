"""`MemberEnumerator`'s shared member-group cache: hits, generation keying and lifetime,
query keying, the one member cap, single flight and the group it hands its waiters, uncached
deadline refusals and unsettled groups, cached settled groups beside dismissed inconsistent
candidates, and the weight bound.

With `GRAPH_RELEASE_ORIGIN` on, every DFO family of the recorded release's slice walks the
release's graph reader to the group a walk of the graph finds, a family with a base member walks
the graph, a walk refuses with no release bound, and a swap keys afresh."""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from dc_graph import parse_arcs
from graph_generation.config import (
    GRAPH_CACHE_MAX_AGE_SECONDS,
    GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
)
from graph_generation.stamp import UNKNOWN_KEY
from graph_release.context import ReleaseNotBound, bound_release
from graph_release.service import Release, ReleaseService
from origin_catalogue import bound_catalogue
from origin_catalogue.arc_tables import ArcTablesBuilder
from origin_catalogue.catalogue import CatalogueSnapshot
from origin_catalogue.families import FamilyKey
from origin_catalogue.graph import _wire
from statvar_members.cache import MemberEnumerator
from statvar_members.enumerate import (
    _REPROBE_BUDGET,
    DEFAULT_MEMBER_CAP,
    DegenerateObservationProperties,
    MemberEnumerationTooLarge,
    MemberGroup,
    MemberReadDeadline,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    SlotBinding,
)
from statvar_members.types import UNRESOLVED_NODE, FiveTuple
from tests.graph_release.recorded import platform_parts, recorded_dfo_arcs
from tests.graph_release.support import build_test_release
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue
from tests.origin_catalogue.support import build_labelled

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

    # Each thread runs in a copy of the test's context, as a request pool's do, so a walk
    # under `GRAPH_RELEASE_ORIGIN` reads the bound release.
    first = threading.Thread(target=contextvars.copy_context().run, args=(call,))
    first.start()
    assert reader.entered.wait(timeout=5)
    second = threading.Thread(target=contextvars.copy_context().run, args=(call,))
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

    first = threading.Thread(target=contextvars.copy_context().run, args=(call,))
    first.start()
    assert reader.entered.wait(timeout=5)
    second = threading.Thread(target=contextvars.copy_context().run, args=(call,))
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
    first = threading.Thread(
        target=contextvars.copy_context().run, args=(lambda: _kenya(enumerator, reader),)
    )
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


_POSITIONS = FiveTuple(
    population_type="FinancialPosition",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
    measurement_denominator="Count_Person",
)
_REFUSALS = (
    DegenerateObservationProperties,
    MemberEnumerationTooLarge,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
)
"""The refusals a walk raises for its family, which the release's walk must raise alike."""
_BASE_POSITION = "Amount_FinancialPosition_Base_Nominal_AsAFractionOf_Count_Person"
"""A base variable of `_POSITIONS`, a family the recorded release otherwise holds whole."""


@pytest.fixture(scope="module")
def recorded_slice() -> tuple[dict[str, dict[str, tuple[str, ...]]], CatalogueSnapshot]:
    """The recorded slice's DFO variables and `_BASE_POSITION` as our graph answers them, and a
    catalogue holding the DFO variables and that base variable as the catalogue job reads them."""
    builder = ArcTablesBuilder()
    read = recorded_dfo_arcs(builder)
    # The nodes the catalogue job holds for the DFO variables' denominators.
    for denominator in sorted(
        {
            arc.dcid
            for rows in builder.table("dfo").values()
            for arc in rows
            if arc.property == "measurementDenominator"
            and arc.dcid is not None
            and arc.dcid not in read.variables
        }
        - {"Count_Person"}
    ):
        builder.add("schema", denominator, None)
    builder.add("base", "Count_Person", {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}})
    builder.add(
        "base",
        _BASE_POSITION,
        {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "populationType": {"nodes": [{"dcid": "FinancialPosition"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
            "measurementDenominator": {"nodes": [{"dcid": "Count_Person"}]},
            "observationProperties": {"nodes": [{"dcid": "financingRecipient"}]},
        },
    )
    variables = {
        dcid: rows
        for table in ("dfo", "base")
        for dcid, rows in builder.table(table).items()
        if dcid != "Count_Person"
    }
    properties = sorted({arc.property for rows in variables.values() for arc in rows})
    graph = {
        dcid: parse_arcs(_wire(rows), properties, mark_unresolved=True)
        for dcid, rows in variables.items()
    }
    return graph, restrictive_catalogue(dfo_variables=read.variables, arcs=build_labelled(builder))


def _origin_release(root: Path, *, revision: str = "rev-a") -> Release:
    """The recorded slice's release, opened with its origin view."""
    uri = (root / "releases").as_uri()
    build_test_release(uri, tables=platform_parts().tables(), dc_serving_revision=revision)
    service = ReleaseService(
        uri=uri, local_dir=root / "local", recheck_seconds=0, build_origin=True
    )
    return service.snapshot()


def _five_tuple(key: FamilyKey) -> FiveTuple:
    return FiveTuple(
        population_type=key.population_type or None,
        measured_property=key.measured_property or None,
        stat_type=key.stat_type or None,
        measurement_qualifier=key.measurement_qualifier,
        measurement_denominator=key.measurement_denominator,
        denominator_unresolved=key.denominator_unresolved,
    )


def test_every_dfo_family_walks_the_release_to_the_group_the_graph_gives(
    recorded_slice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each family's walk with every slot unbound and with each property bound to each of its
    values, members, family and order included, reads the graph for none of them."""
    variables, catalogue = recorded_slice
    release = _origin_release(tmp_path)
    assert release.origin is not None
    queries = []
    for key, summary in release.origin.families.items():
        if key.population_type == "FinancialPosition":
            continue
        unbound: dict[str, SlotBinding] = dict.fromkeys(summary.values, ("unbound",))
        queries.append((_five_tuple(key), unbound))
        queries += [
            (_five_tuple(key), {**unbound, prop: ("value", value)})
            for prop, values in summary.values.items()
            for value in values
        ]
    assert len({five_tuple for five_tuple, _ in queries}) == 5
    graph = FakeMemberArcReader(arcs=variables)

    def walk(enumerator: MemberEnumerator, five_tuple: FiveTuple, slots) -> object:
        try:
            return enumerator.enumerate(
                graph=graph, stamp_key="rev1", five_tuple=five_tuple, slots=slots
            )
        except _REFUSALS as exc:
            return type(exc), str(exc)

    def walks() -> list[object]:
        enumerator = MemberEnumerator()
        with bound_catalogue(catalogue):
            return [walk(enumerator, five_tuple, slots) for five_tuple, slots in queries]

    monkeypatch.delenv("GRAPH_RELEASE_ORIGIN", raising=False)
    live = walks()
    walked = dict(graph.calls)
    monkeypatch.setenv("GRAPH_RELEASE_ORIGIN", "1")
    with bound_release(release):
        served = walks()

    assert served == live
    assert graph.calls == walked
    groups = [group for group in live if isinstance(group, MemberGroup)]
    assert len(groups) > len(queries) / 2
    assert sum(len(group.members) for group in groups) > len(groups)


def test_a_family_with_a_base_member_walks_the_graph(
    recorded_slice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variables, catalogue = recorded_slice
    release = _origin_release(tmp_path)
    graph = FakeMemberArcReader(arcs=variables)
    monkeypatch.setenv("GRAPH_RELEASE_ORIGIN", "1")

    with bound_release(release), bound_catalogue(catalogue):
        group = MemberEnumerator().enumerate(
            graph=graph, stamp_key="rev1", five_tuple=_POSITIONS, slots={}
        )

    assert graph.calls["node_subjects"] == 1
    assert _BASE_POSITION in group.family


@pytest.mark.unbound_release
def test_a_walk_under_the_origin_flag_refuses_with_no_release_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPH_RELEASE_ORIGIN", "1")
    reader = kenya_reader()

    with pytest.raises(ReleaseNotBound):
        _kenya(MemberEnumerator(), reader)

    assert reader.calls["node_subjects"] == 0


def test_a_release_swap_keys_the_walk_afresh(
    recorded_slice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family the release does not hold, read from the graph under one stamp key, walks
    again under the next release and hits again under the first."""
    variables, catalogue = recorded_slice
    first = _origin_release(tmp_path / "a", revision="rev-a")
    second = _origin_release(tmp_path / "b", revision="rev-b")
    assert first.release_id != second.release_id
    graph = FakeMemberArcReader(arcs=variables)
    enumerator = MemberEnumerator()
    monkeypatch.setenv("GRAPH_RELEASE_ORIGIN", "1")

    def walk(release: Release) -> None:
        with bound_release(release), bound_catalogue(catalogue):
            enumerator.enumerate(graph=graph, stamp_key="rev1", five_tuple=_POSITIONS, slots={})

    walk(first)
    walk(second)
    assert graph.calls["node_subjects"] == 2
    walk(first)
    assert graph.calls["node_subjects"] == 2
