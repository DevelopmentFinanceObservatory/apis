"""The scope rules `compile` holds to, checked over every selection of a synthetic table.

Each axis of `tests.scope._axes` takes no entry, or one explicit, carried or switch entry
on each of its options, under both availabilities of grant equivalents: 980 selections.
Every compiled selection is then switched to each option of each axis.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import get_args

import pytest

from search.models import (
    BaseIdentity,
    EntitiesBinding,
    Entry,
    ScopeReasonCode,
)
from search.scope.compile import Compiled, Refusal, compile
from search.scope.edits import ScopeTableError, apply_binding
from search.scope.tables import AxisTable, CompileFacts
from tests.scope._axes import (
    BASE,
    CONFLICTS,
    GRANT_EQUIVALENT,
    GROSS,
    MEMBERS,
    REGION_ALL,
    REGION_MEMBERS,
    RESIDUAL,
    SECTOR_ONE,
    SECTOR_SET,
    carried,
    explicit,
    facts,
    switch,
    tables,
)

_REASON_CODES = frozenset(get_args(ScopeReasonCode))
_KINDS = (explicit, carried, switch)


def _selections() -> Iterator[tuple[bool, tuple[Entry, ...]]]:
    for published in (True, False):
        per_axis = [
            [None, *(kind(option.option_id) for option in table.options for kind in _KINDS)]
            for table in tables(grant_equivalent_published=published)
        ]
        for combo in itertools.product(*per_axis):
            yield published, tuple(entry for entry in combo if entry is not None)


_SELECTIONS = list(_selections())


def _compiled(published: bool, entries: tuple[Entry, ...]) -> Compiled | Refusal:
    return compile(BASE, entries, facts(grant_equivalent_published=published))


def _active(result: Compiled) -> dict[str, tuple[str, str]]:
    return {choice.axis: (choice.active, choice.provenance) for choice in result.choices}


def _conflict(first: str, second: str) -> bool:
    return any({c.first, c.second} == {first, second} for c in CONFLICTS)


def _with_switch(entries: tuple[Entry, ...], option_id: str) -> tuple[Entry, ...]:
    """`entries` with the switch to `option_id` in place of any entry on its axis, as a
    client posts a switch."""
    axis = option_id.partition(":")[0]
    return (*(e for e in entries if e.axis != axis), switch(option_id))


def test_the_enumeration_covers_refusals_drops_and_moved_defaults():
    """A guard on the property tests below: the selections reach every outcome they check."""
    results = [_compiled(p, e) for p, e in _SELECTIONS]
    compiled = [r for r in results if isinstance(r, Compiled)]

    assert len(_SELECTIONS) == 980
    assert {r.code for r in results if isinstance(r, Refusal)} == {
        "requested_option_unavailable",
        "conflicting_requested_options",
    }
    assert any(r.dropped for r in compiled)
    assert any(
        choice.provenance == "default" and choice.active != choice.options[0].option_id
        for r in compiled
        for choice in r.choices
    )


def test_compile_holds_its_rules_on_every_selection():
    for published, entries in _SELECTIONS:
        try:
            _check_rules(published, entries)
        except AssertionError as exc:
            raise AssertionError(f"{published=} {entries=}") from exc


def _check_rules(published: bool, entries: tuple[Entry, ...]) -> None:
    result = _compiled(published, entries)
    applied_explicit = [e for e in entries if e.provenance == "explicit"]

    if isinstance(result, Refusal):
        refused = entries[result.entry]
        assert result.code in ("requested_option_unavailable", "conflicting_requested_options")
        # A refusal names an explicit entry, or the applied entry that blocks a whole axis.
        assert refused.provenance == "explicit" or result.blocker is None
        return

    active = _active(result)
    # Explicit entries are never moved, and every carried entry is applied or reported.
    for entry in applied_explicit:
        assert active[entry.axis] == (entry.option_id, "explicit")
    dropped = [d.entry for d in result.dropped]
    for entry in (e for e in entries if e.provenance == "carried"):
        assert active[entry.axis] == (entry.option_id, "carried") or entry in dropped
    assert all(d.code == "carried_choice_not_applied" for d in result.dropped)
    assert {e.option_id for e in result.entries} == {
        option for option, provenance in active.values() if provenance != "default"
    }

    # Defaults are derived after the entries: none conflicts with an applied entry.
    settled = [opt for opt, prov in active.values() if prov != "default"]
    for option, provenance in active.values():
        if provenance == "default":
            assert not any(_conflict(option, other) for other in settled)

    # Every unavailable option carries a reason code, and the active option is servable.
    for choice in result.choices:
        for option in choice.options:
            if option.availability.status != "available":
                assert option.availability.code in _REASON_CODES
        (held,) = [o for o in choice.options if o.option_id == choice.active]
        assert held.availability.code != "conflicts_with"

    # The order entries are posted in never matters.
    reordered = _compiled(published, tuple(reversed(entries)))
    assert isinstance(reordered, Compiled)
    assert (reordered.choices, reordered.spec_id, reordered.dropped) == (
        result.choices,
        result.spec_id,
        result.dropped,
    )


def test_every_switch_is_applied_refused_or_a_no_op():
    """A switch to the active option compiles to the same spec_id. A switch to an option
    shown unavailable is refused. A switch to any other option applies it explicitly and
    moves no other explicit or carried entry."""
    for published, entries in _SELECTIONS:
        try:
            _check_switches(published, entries)
        except AssertionError as exc:
            raise AssertionError(f"{published=} {entries=}") from exc


def _check_switches(published: bool, entries: tuple[Entry, ...]) -> None:
    result = _compiled(published, entries)
    if isinstance(result, Refusal):
        return
    active = _active(result)

    for choice in result.choices:
        for option in choice.options:
            switched = _compiled(published, _with_switch(result.entries, option.option_id))
            if option.option_id == choice.active:
                assert isinstance(switched, Compiled)
                assert switched.spec_id == result.spec_id
            elif option.availability.status == "unavailable":
                assert isinstance(switched, Refusal)
            else:
                assert isinstance(switched, Compiled)
                moved = _active(switched)
                assert moved[choice.axis] == (option.option_id, "explicit")
                for axis, (opt, prov) in active.items():
                    if axis != choice.axis and prov != "default":
                        assert moved[axis] == (opt, prov)


def test_the_identity_is_a_function_of_the_selection():
    by_selection: dict[tuple, str] = {}
    by_spec_id: dict[str, tuple] = {}
    for published, entries in _SELECTIONS:
        result = _compiled(published, entries)
        if isinstance(result, Refusal):
            continue
        selection = tuple(sorted((c.axis, c.active) for c in result.choices))
        assert by_selection.setdefault(selection, result.spec_id) == result.spec_id
        assert by_spec_id.setdefault(result.spec_id, selection) == selection
    assert len(by_spec_id) > 1


def test_a_selection_the_base_holds_compiles_to_the_base_itself():
    result = compile(BASE, (explicit(SECTOR_SET), carried(GROSS)), facts())

    assert isinstance(result, Compiled)
    assert result.identity is BASE
    assert result.spec_id == result.base_spec_id


def test_a_default_moves_when_an_explicit_entry_blocks_it():
    """Grant equivalents exclude the residual bucket, so the region default yields to
    members only, keeps its default provenance, and the residual leaves the identity."""
    result = compile(BASE, (explicit(GRANT_EQUIVALENT),), facts())

    assert isinstance(result, Compiled)
    assert _active(result)["region"] == (REGION_MEMBERS, "default")
    assert result.identity.entities["destination"] == list(MEMBERS)
    assert result.identity.shape.measured_property == "grantEquivalent"
    assert "nettingBasis" not in {slot.property for slot in result.identity.slots}
    (residual_option,) = [
        o
        for c in result.choices
        if c.axis == "region"
        for o in c.options
        if o.option_id == REGION_ALL
    ]
    assert residual_option.availability.code == "conflicts_with"
    assert residual_option.availability.detail == {
        "entry": GRANT_EQUIVALENT,
        "cause": "grant_equivalents_lack_residuals",
    }


@pytest.mark.parametrize(
    ("entries", "code", "entry", "blocker", "cause"),
    [
        pytest.param(
            (explicit("provider:inclusive"),),
            "requested_option_unavailable",
            0,
            None,
            "axis_absent",
            id="absent-axis",
        ),
        pytest.param(
            (explicit("sector:720+730+740"),),
            "requested_option_unavailable",
            0,
            None,
            "option_unknown",
            id="unknown-option",
        ),
        pytest.param(
            (explicit(REGION_ALL), explicit(GRANT_EQUIVALENT)),
            "conflicting_requested_options",
            1,
            0,
            "grant_equivalents_lack_residuals",
            id="cross-axis-conflict",
        ),
        pytest.param(
            (explicit(SECTOR_SET), switch(SECTOR_ONE)),
            "conflicting_requested_options",
            1,
            0,
            "both_on_sector",
            id="two-options-on-one-axis",
        ),
        pytest.param(
            (carried(REGION_ALL), switch(GRANT_EQUIVALENT)),
            "conflicting_requested_options",
            1,
            0,
            "grant_equivalents_lack_residuals",
            id="switch-blocked-by-a-carried-entry",
        ),
    ],
)
def test_refusals_name_the_entry_its_blocker_and_cause(entries, code, entry, blocker, cause):
    assert compile(BASE, entries, facts()) == Refusal(
        code=code, entry=entry, blocker=blocker, cause=cause
    )


def test_an_explicit_entry_on_an_unpublished_basis_refuses_with_its_reason():
    refusal = compile(BASE, (explicit(GRANT_EQUIVALENT),), facts(grant_equivalent_published=False))

    assert refusal == Refusal(
        code="requested_option_unavailable", entry=0, cause="basis_not_published"
    )


@pytest.mark.parametrize(
    ("entries", "cause"),
    [
        pytest.param((carried("provider:inclusive"),), "axis_absent", id="absent-axis"),
        pytest.param((carried("sector:720+730+740"),), "option_unknown", id="unknown-option"),
        pytest.param((explicit(SECTOR_SET), carried(SECTOR_ONE)), "entry_on_axis", id="same-axis"),
        pytest.param(
            (explicit(REGION_ALL), carried(GRANT_EQUIVALENT)),
            "conflicts_with_entry",
            id="conflicts-with-the-words",
        ),
    ],
)
def test_a_carried_entry_the_part_cannot_take_is_dropped_with_its_cause(entries, cause):
    result = compile(BASE, entries, facts())

    assert isinstance(result, Compiled)
    assert [(d.entry, d.code, d.cause) for d in result.dropped] == [
        (entries[-1], "carried_choice_not_applied", cause)
    ]


def test_a_carried_entry_on_an_unpublished_basis_is_dropped_as_unavailable():
    result = compile(BASE, (carried(GRANT_EQUIVALENT),), facts(grant_equivalent_published=False))

    assert isinstance(result, Compiled)
    assert [d.cause for d in result.dropped] == ["option_unavailable"]
    assert _active(result)["measure_basis"] == (GROSS, "default")


def test_entries_over_no_axis_tables_compile_to_the_base_with_no_choices():
    result = compile(BASE, (), CompileFacts(axes=()))

    assert isinstance(result, Compiled)
    assert (result.choices, result.identity, result.entries) == ((), BASE, ())


def test_a_switch_edits_only_the_members_its_axis_binds():
    result = compile(BASE, (switch(SECTOR_ONE), switch(REGION_MEMBERS)), facts())

    assert isinstance(result, Compiled)
    slots = {slot.property: slot.binding for slot in result.identity.slots}
    assert slots["crsPurpose"].model_dump() == {"state": "value", "ref": "CRSPurposeCode120"}
    assert slots["nettingBasis"].model_dump() == {"state": "value", "ref": "Gross"}
    assert result.identity.entities == {"origin": ["country/USA"], "destination": list(MEMBERS)}
    assert RESIDUAL not in result.identity.entities["destination"]
    assert _active(result)["measure_basis"] == (GROSS, "default")


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"options": ()}, id="no-options"),
        pytest.param({"preference": ("measure_basis:commitments",)}, id="unknown-preference"),
        pytest.param({"held": "measure_basis:commitments"}, id="unknown-held"),
        pytest.param({"axis": "region"}, id="options-of-another-axis"),
    ],
)
def test_an_axis_table_refuses_what_compile_cannot_read(change):
    basis = tables()[2]
    fields = {
        "axis": basis.axis,
        "rule": basis.rule,
        "options": basis.options,
        "preference": basis.preference,
        "held": basis.held,
    }

    with pytest.raises(ScopeTableError):
        AxisTable(**{**fields, **change})


def test_an_entities_edit_refuses_to_empty_a_role():
    only_residual: BaseIdentity = BASE.model_copy(
        update={"entities": {**BASE.entities, "destination": [RESIDUAL]}}
    )
    with pytest.raises(ScopeTableError):
        apply_binding(
            only_residual,
            EntitiesBinding(role="destination", add=[]),
            [EntitiesBinding(role="destination", add=[RESIDUAL])],
        )
