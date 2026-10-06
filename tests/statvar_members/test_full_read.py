"""`enumerate_members(subject_ceiling=...)`: a truncated first page without proof settles by
reading the whole filtered set, a set past the ceiling refuses, and a read past the deadline
raises with the window it was reading."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

import statvar_members.enumerate as enumerate_module
from statvar_members.enumerate import (
    CANDIDATE_CAP,
    MemberReadDeadline,
    SlotBinding,
    enumerate_members,
)
from statvar_members.types import FiveTuple

from .conftest import FakeMemberArcReader, FakeSubjectPage

_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)
_FEMALE: dict[str, SlotBinding] = {"gender": ("value", "Female"), "age": ("unbound",)}
_ROOT = "Count_Person_Female"
_PARTS = CANDIDATE_CAP + 100


def _arcs(**constraints: str) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": tuple(sorted(constraints)),
        "observationProperties": (),
        **{name: (value,) for name, value in constraints.items()},
    }


def _family(part_prefix: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """The female population and `_PARTS` female age bands. The fake pages subjects in dcid
    order, so a `part_prefix` sorting before `_ROOT` puts the root past the first page."""
    arcs = {f"{part_prefix}{i:05d}": _arcs(gender="Female", age=f"Years{i}") for i in range(_PARTS)}
    arcs[_ROOT] = _arcs(gender="Female")
    return arcs


class _CountingReader(FakeMemberArcReader):
    """Records each reverse read's `max_results` and each arc read's dcid count."""

    def __init__(self, arcs: Mapping[str, Mapping[str, tuple[str, ...]]]) -> None:
        super().__init__(arcs=arcs)
        self.subject_windows: list[int] = []
        self.arcs_read = 0

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> FakeSubjectPage:
        self.subject_windows.append(max_results)
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self.arcs_read += len(dcids)
        return super().node_arcs_many(dcids, properties=properties)


def test_a_root_past_the_first_page_settles_by_the_full_read():
    reader = _CountingReader(_family("Count_Person_Age"))

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots=_FEMALE, graph=reader, subject_ceiling=20_000
    )

    assert [member.stat_var for member in group.members] == [_ROOT]
    assert (group.truncated, group.proven_complete, group.full_read) == (False, True, True)
    assert (group.candidates_considered, group.candidate_cap) == (_PARTS + 1, 20_000)
    assert reader.subject_windows == [CANDIDATE_CAP, 20_000]


def test_without_a_ceiling_the_same_page_stays_unproven():
    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_FEMALE,
        graph=_CountingReader(_family("Count_Person_Age")),
        member_cap=_PARTS,
    )

    assert (group.truncated, group.proven_complete, group.full_read) == (True, False, False)


def test_the_full_read_reuses_the_first_pages_arcs():
    """Every page holds age bands only, which name a constraint outside the bound `gender`,
    so no root sits on it: the walk goes straight to the full read, and that read's batch
    covers only the subjects past the page."""
    reader = _CountingReader(_family("Count_Person_Age"))

    enumerate_members(five_tuple=_FIVE_TUPLE, slots=_FEMALE, graph=reader, subject_ceiling=20_000)

    # Each subject's batch read once, plus the trusted re-read of the one antichain winner.
    assert reader.arcs_read == _PARTS + 1 + 1


def test_a_root_on_the_first_page_settles_without_the_full_read():
    reader = _CountingReader(_family("Count_Person_Female_Age"))

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots=_FEMALE, graph=reader, subject_ceiling=20_000
    )

    assert [member.stat_var for member in group.members] == [_ROOT]
    assert (group.truncated, group.proven_complete, group.full_read) == (True, True, False)
    assert reader.subject_windows == [CANDIDATE_CAP]


def test_a_cap_refusal_of_the_page_alone_defers_to_the_full_read():
    """A candidate reading no constraint may be a root whose arcs a batch read dropped, so
    the page is confirmed first. Its 4,000 age bands exceed the member cap, and the root
    past the page then dominates them all."""
    arcs = _family("Count_Person_Age")
    arcs["Count_Person_A_Unconstrained"] = {**_arcs(), "gender": ("Female",)}
    reader = _CountingReader(arcs)

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots=_FEMALE, graph=reader, subject_ceiling=20_000
    )

    assert [member.stat_var for member in group.members] == [_ROOT]
    assert (group.proven_complete, group.full_read) == (True, True)


def test_a_set_past_the_ceiling_refuses_with_the_ceiling_it_crossed():
    reader = _CountingReader(_family("Count_Person_Age"))
    ceiling = CANDIDATE_CAP + 50

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots=_FEMALE, graph=reader, subject_ceiling=ceiling
    )

    assert (group.truncated, group.proven_complete, group.full_read) == (True, False, True)
    assert (group.candidates_considered, group.candidate_cap) == (ceiling, ceiling)
    assert group.members == ()


class _StallingFullRead(_CountingReader):
    """Advances `clock` past the deadline and fails, as a live client's deadline error does,
    when the full read starts."""

    def __init__(self, arcs, clock: list[float]) -> None:
        super().__init__(arcs)
        self._clock = clock

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        if max_results > CANDIDATE_CAP:
            self._clock[0] = 100.0
            raise TimeoutError("the read ran past its deadline")
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )


@pytest.mark.parametrize(
    ("deadline", "considered", "cap"),
    [(0.0, 0, CANDIDATE_CAP), (50.0, 0, 20_000)],
    ids=["before_the_first_page", "during_the_full_read"],
)
def test_a_read_past_the_deadline_refuses_with_the_window_it_was_reading(
    monkeypatch: pytest.MonkeyPatch, deadline: float, considered: int, cap: int
):
    clock = [10.0]
    monkeypatch.setattr(enumerate_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    reader = _StallingFullRead(_family("Count_Person_Age"), clock)

    with pytest.raises(MemberReadDeadline) as exc_info:
        enumerate_members(
            five_tuple=_FIVE_TUPLE,
            slots=_FEMALE,
            graph=reader,
            subject_ceiling=20_000,
            deadline=deadline,
        )

    assert (exc_info.value.candidates_considered, exc_info.value.candidate_cap) == (
        considered,
        cap,
    )


class _SlowFullRead(_CountingReader):
    """Moves `clock` to 60 when the full read starts and records each reverse read's
    deadline."""

    def __init__(self, arcs, clock: list[float]) -> None:
        super().__init__(arcs)
        self._clock = clock
        self.deadlines: list[float | None] = []

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        self.deadlines.append(deadline)
        if max_results > CANDIDATE_CAP:
            self._clock[0] = 60.0
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )


def test_the_full_read_runs_under_its_own_deadline(monkeypatch: pytest.MonkeyPatch):
    """The page reads under `deadline`, 50; the full read runs to 60 under
    `full_read_deadline`, 90, and settles."""
    clock = [10.0]
    monkeypatch.setattr(enumerate_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    reader = _SlowFullRead(_family("Count_Person_Age"), clock)

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_FEMALE,
        graph=reader,
        subject_ceiling=20_000,
        deadline=50.0,
        full_read_deadline=90.0,
    )

    assert [member.stat_var for member in group.members] == [_ROOT]
    assert group.full_read
    assert reader.deadlines == [50.0, 90.0]


def test_the_full_read_deadline_defaults_to_the_deadline(monkeypatch: pytest.MonkeyPatch):
    clock = [10.0]
    monkeypatch.setattr(enumerate_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    reader = _SlowFullRead(_family("Count_Person_Age"), clock)

    with pytest.raises(MemberReadDeadline) as exc_info:
        enumerate_members(
            five_tuple=_FIVE_TUPLE,
            slots=_FEMALE,
            graph=reader,
            subject_ceiling=20_000,
            deadline=50.0,
        )

    assert exc_info.value.candidate_cap == 20_000
    assert reader.deadlines == [50.0, 50.0]


def test_an_open_role_twin_past_the_first_page_is_elected_by_the_full_read():
    """The page holds the subject-only root and age bands, and the root's by-provider twin
    sorts past it. With origin asked open, no member on the page serves it, so the root
    does not prove the page complete and the full read elects the twin."""
    arcs = _family("Count_Person_Female_Age")
    twin = "Count_Person_Female_ZByProvider"
    arcs[twin] = {**_arcs(gender="Female"), "observationProperties": ("financingProvider",)}
    reader = _CountingReader(arcs)

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_FEMALE,
        graph=reader,
        open_role_keys=("origin",),
        subject_ceiling=20_000,
    )

    assert [member.stat_var for member in group.members] == [twin]
    assert group.elected_open_role_keys == {"origin"}
    assert (group.truncated, group.full_read) == (False, True)
    assert reader.subject_windows == [CANDIDATE_CAP, 20_000]


def test_without_a_ceiling_a_page_serving_no_asked_open_role_stays_unproven():
    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_FEMALE,
        graph=_CountingReader(_family("Count_Person_Female_Age")),
        open_role_keys=("origin",),
    )

    assert [member.stat_var for member in group.members] == [_ROOT]
    assert group.elected_open_role_keys == frozenset()
    assert (group.truncated, group.proven_complete) == (True, False)
