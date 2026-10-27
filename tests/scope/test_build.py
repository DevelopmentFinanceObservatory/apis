"""`build_part_scope`, the per-part scope state, the option vocabulary and the wire entry."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from code_index import scope_conventions
from code_index.shared import get_index
from search.engine.spec_id import identity_spec_id
from search.models import SCOPE_OPTION_ID_PATTERN, Entry, PartScope, TextOrigin
from search.scope import rules as scope_rules
from search.scope.build import build_part_scope
from search.scope.compile import Refusal
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import ScopeFacts
from search.scope.vocabulary import (
    MEASURE_BASIS_OPTIONS,
    PROVIDER_OPTIONS,
    REGION_OPTIONS,
    option_axis,
    sector_option_id,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.scope._axes import (
    BASE,
    CONFLICTS,
    GRANT_EQUIVALENT,
    REGION_ALL,
    SECTOR_ONE,
    SECTOR_SET,
    carried,
    explicit,
    switch,
    tables,
)

_FACTS = ScopeFacts(index=get_index(), coverage=empty_coverage_snapshot())


def _rules(seen: list):
    """Rules returning the synthetic tables, each recording what it read."""
    by_axis = {table.axis: table for table in tables()}

    def rule_for(axis):
        def rule(base, record, facts):
            seen.append((axis, base, record, facts))
            return by_axis[axis]

        return rule

    return {axis: rule_for(axis) for axis in by_axis}


def test_a_part_scope_names_its_choices_by_the_base_spec_id():
    seen: list = []
    record = AxisRecord(axis="sector", rule="test", active=SECTOR_SET, provenance="default")

    scope = build_part_scope(
        base=BASE,
        entries=[switch(SECTOR_ONE)],
        records={"sector": record},
        facts=_FACTS,
        rules=_rules(seen),
        conflicts=CONFLICTS,
    )

    assert isinstance(scope, PartScope)
    base_spec_id = identity_spec_id(BASE)
    assert scope.id == scope.base.spec_id == base_spec_id
    assert [c.id for c in scope.choices] == [
        f"{base_spec_id}:sector",
        f"{base_spec_id}:region",
        f"{base_spec_id}:measure_basis",
    ]
    assert [e.option_id for e in scope.entries] == [SECTOR_ONE]
    assert scope.compiled.spec_id == identity_spec_id(scope.compiled.identity) != base_spec_id
    assert [(c.role, c.identity) for c in scope.compiled.components] == [
        ("whole", scope.compiled.identity)
    ]
    assert scope.ledger == []
    # Each rule reads the base, its axis's record and the facts, in axis order.
    assert [(axis, record_read) for axis, _, record_read, _ in seen] == [
        ("sector", record),
        ("region", None),
        ("measure_basis", None),
    ]
    assert all(base is BASE and facts is _FACTS for _, base, _, facts in seen)


def test_a_part_scope_round_trips_through_its_wire_form():
    scope = build_part_scope(
        base=BASE,
        entries=[explicit(GRANT_EQUIVALENT), carried(REGION_ALL)],
        records={},
        facts=_FACTS,
        rules=_rules([]),
        conflicts=CONFLICTS,
    )

    assert isinstance(scope, PartScope)
    assert [d.cause for d in scope.dropped_entries] == ["conflicts_with_entry"]
    assert PartScope.model_validate_json(scope.model_dump_json()) == scope


def test_a_refused_selection_passes_the_refusal_through():
    refusal = build_part_scope(
        base=BASE,
        entries=[explicit(REGION_ALL), explicit(GRANT_EQUIVALENT)],
        records={},
        facts=_FACTS,
        rules=_rules([]),
        conflicts=CONFLICTS,
    )

    assert isinstance(refusal, Refusal)
    assert (refusal.code, refusal.entry, refusal.blocker) == ("conflicting_requested_options", 1, 0)


def test_a_base_settling_no_live_axis_holds_no_choices_and_refuses_an_entry_on_one():
    unscoped = BASE.model_copy(
        update={
            "slots": [slot for slot in BASE.slots if slot.property != "crsPurpose"],
            "entities": {"origin": ["country/USA"], "destination": ["country/KEN"]},
        }
    )
    empty = build_part_scope(base=unscoped, entries=[], records={}, facts=_FACTS)
    refused = build_part_scope(
        base=unscoped, entries=[explicit(SECTOR_SET)], records={}, facts=_FACTS
    )

    assert isinstance(empty, PartScope)
    assert (empty.choices, empty.compiled.identity) == ([], unscoped)
    assert refused == Refusal(code="requested_option_unavailable", entry=0, cause="axis_absent")


def test_the_rule_version_moves_with_the_declared_conventions(monkeypatch):
    before = scope_rules.rule_version({}, ())

    monkeypatch.setattr(scope_conventions, "SECTOR_GROUPINGS", ())

    assert scope_rules.rule_version({}, ()) != before
    assert scope_rules.rule_version({}, CONFLICTS) != scope_rules.rule_version({}, ())


def test_binders_read_entries_on_their_axis_explicit_first_for_their_concept():
    health = carried(SECTOR_ONE).model_copy(update={"concept": "health"})
    education = carried(SECTOR_SET).model_copy(update={"concept": "education"})
    words = explicit(SECTOR_SET)
    state = ScopeState(entries=(health, education, words, explicit(GRANT_EQUIVALENT)))

    assert state.entries_on("sector", concept="health") == (words, health)
    assert state.entries_on("sector") == (words,)


def test_a_part_settles_each_axis_once():
    state = ScopeState()
    record = AxisRecord(axis="sector", rule="test", active=SECTOR_SET, provenance="default")
    state.write(record)

    with pytest.raises(ValueError, match="already settled"):
        state.write(record)
    assert state.records == {"sector": record}


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"active": REGION_ALL, "provenance": "default"}, id="option-of-another-axis"),
        pytest.param({"active": SECTOR_SET, "provenance": "explicit"}, id="entry-missing"),
    ],
)
def test_an_axis_record_refuses_an_inconsistent_settlement(fields):
    with pytest.raises(ValueError):
        AxisRecord(axis="sector", rule="test", **fields)


def test_option_ids_name_their_axis_and_sector_codes_by_number():
    assert sector_option_id(["CRSPurposeCode130", "CRSPurposeCode120"]) == "sector:120+130"
    assert (
        sector_option_id(
            [
                code
                for g in scope_conventions.SECTOR_GROUPINGS
                if g.concept == "humanitarian"
                for code in g.codes
            ]
        )
        == "sector:720+730+740"
    )
    for option in (*MEASURE_BASIS_OPTIONS, *PROVIDER_OPTIONS, *REGION_OPTIONS):
        assert re.fullmatch(SCOPE_OPTION_ID_PATTERN, option)
        assert option_axis(option) == option.partition(":")[0]
    with pytest.raises(ValueError):
        option_axis("basis:gross")


def test_an_entry_refuses_an_option_of_another_axis():
    with pytest.raises(ValidationError):
        Entry(
            axis="sector",
            option_id=REGION_ALL,
            provenance="explicit",
            origin=TextOrigin(text="all of Africa", start=0, end=13),
        )
