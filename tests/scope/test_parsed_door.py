"""A parsed-door answer states its scope choices when its binders record an axis, and none
when they record nothing. The request's scope words and carried entries reach the binders,
and those that attach to the part compile."""

from __future__ import annotations

import pytest

from search.engine.region import phase_answer
from search.models import (
    ChoiceOrigin,
    DefinitePart,
    DroppedEntry,
    Entry,
    Option,
    OptionAvailability,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveRequest,
    ShapeBinding,
    SlotBinding,
    TextOrigin,
)
from search.scope import rules as scope_rules
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import AxisTable
from tests.engine._harness import offline_resolve

_HELD = "sector:1"
_NET = "measure_basis:net"
_GROSS = "measure_basis:gross"
_GRANT_EQUIVALENT = "measure_basis:grant_equivalent"


def _rule(base, record, facts) -> AxisTable | None:
    """A synthetic axis, present when its binder recorded it, holding the recorded option."""
    if record is None:
        return None
    return AxisTable(
        axis="sector",
        rule="test_rule",
        options=tuple(
            Option(
                option_id=option_id,
                label=option_id,
                binding=SlotBinding(slot="testSlot", state="value", refs=[option_id]),
                availability=OptionAvailability(status="available"),
            )
            for option_id in (_HELD, "sector:2")
        ),
        preference=(record.active,),
        held=record.active,
    )


def _basis_rule(base, record, facts) -> AxisTable | None:
    """A synthetic measure basis, present when its binder recorded it, holding the recorded
    option; grant equivalents are unpublished."""
    if record is None:
        return None
    unpublished = OptionAvailability(status="unavailable", code="basis_not_published")
    return AxisTable(
        axis="measure_basis",
        rule="test_basis",
        options=tuple(
            Option(
                option_id=option_id,
                label=option_id,
                binding=ShapeBinding(measured_property="amount", netting_basis=basis),
                availability=(
                    unpublished
                    if option_id == _GRANT_EQUIVALENT
                    else OptionAvailability(status="available")
                ),
            )
            for option_id, basis in ((_NET, "Net"), (_GROSS, "Gross"), (_GRANT_EQUIVALENT, None))
        ),
        preference=(record.active,),
        held=record.active,
    )


@pytest.fixture(autouse=True)
def _registered_rule(monkeypatch):
    monkeypatch.setattr(scope_rules, "AXIS_RULES", {"sector": _rule, "measure_basis": _basis_rule})


def _write_a_record(monkeypatch, *, concept: str | None = None) -> None:
    """Have the definite answer's binders record the synthetic sector axis, asked for
    `concept`."""
    _write_records(monkeypatch, sector_concept=concept)


def _write_records(
    monkeypatch, *, sector_concept: str | None = None, basis: bool = False
) -> list[ScopeState]:
    """Have the definite answer's binders record the synthetic sector axis, applying the
    first entry on it for `sector_concept` when it names the held option, and, with
    `basis`, the measure basis as a binder applying precedence would: the first entry on
    the axis, else net by default. Returns the states the binders saw."""
    assemble = phase_answer._assemble_definite_answer
    seen: list[ScopeState] = []

    async def recording(state):
        selection = state.scope_selection
        seen.append(selection)
        sector = next(
            (
                e
                for e in selection.entries_on("sector", concept=sector_concept)
                if e.option_id == _HELD
            ),
            None,
        )
        selection.write(
            AxisRecord(
                axis="sector",
                rule="test_rule",
                active=_HELD,
                provenance="default" if sector is None else sector.provenance,
                entry=sector,
                concept=sector_concept,
            )
        )
        if basis:
            entries = selection.entries_on("measure_basis")
            selection.write(
                AxisRecord(
                    axis="measure_basis",
                    rule="test_basis",
                    active=entries[0].option_id if entries else _NET,
                    provenance=entries[0].provenance if entries else "default",
                    entry=entries[0] if entries else None,
                )
            )
        return await assemble(state)

    monkeypatch.setattr(phase_answer, "_assemble_definite_answer", recording)
    return seen


def _answer(*, user_message: str | None = None, carried: list[Entry] | None = None) -> DefinitePart:
    response = offline_resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")]
                    )
                ],
                user_message=user_message,
                carried_entries=carried,
            )
        )
    )
    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    return part


def _carried(axis, option_id: str, *, concept: str | None = None) -> Entry:
    return Entry(
        axis=axis,
        option_id=option_id,
        provenance="carried",
        concept=concept,
        origin=ChoiceOrigin(choice_id=f"earlier:{axis}"),
    )


def test_a_recorded_axis_puts_scope_choices_on_the_parsed_answer(monkeypatch):
    _write_a_record(monkeypatch)

    spec = _answer().spec

    scope = spec.resolved.scope_choices
    assert scope is not None
    assert (scope.id, scope.base.spec_id, scope.compiled.spec_id) == (spec.spec_id,) * 3
    assert [(c.axis, c.active, c.provenance) for c in scope.choices] == [
        ("sector", _HELD, "default")
    ]
    assert scope.entries == []


def test_no_record_leaves_scope_choices_none():
    assert _answer().spec.resolved.scope_choices is None


def test_the_user_message_states_an_explicit_entry_the_binders_read(monkeypatch):
    seen = _write_records(monkeypatch, basis=True)

    scope = _answer(user_message="total net fertility in Kenya").spec.resolved.scope_choices

    assert scope is not None
    (entry,) = scope.entries
    assert (entry.option_id, entry.provenance) == (_NET, "explicit")
    assert entry.origin.model_dump() == {"kind": "text", "text": "net", "start": 6, "end": 9}
    choice = next(c for c in scope.choices if c.axis == "measure_basis")
    assert (choice.active, choice.provenance) == (_NET, "explicit")
    assert seen[0].total


def test_an_explicit_entry_wins_and_the_carried_one_on_its_axis_is_dropped(monkeypatch):
    _write_records(monkeypatch, basis=True)
    carried = _carried("measure_basis", _GROSS)

    scope = _answer(user_message="net fertility in Kenya", carried=[carried]).spec
    scope = scope.resolved.scope_choices

    assert scope is not None
    assert [e.option_id for e in scope.entries] == [_NET]
    assert scope.dropped_entries == [DroppedEntry(entry=carried, cause="entry_on_axis")]


def test_a_carried_entry_attaches_to_the_part_asking_for_its_concept(monkeypatch):
    _write_a_record(monkeypatch, concept="education")
    health = _carried("sector", _HELD, concept="health")
    education = _carried("sector", _HELD, concept="education")
    basis = _carried("measure_basis", _GROSS)

    scope = _answer(carried=[health, education, basis]).spec.resolved.scope_choices

    # Health is another part's concept and the part settles no basis: neither entry is
    # the part's, so neither applies nor is reported dropped.
    assert scope is not None
    assert scope.entries == [education]
    assert scope.dropped_entries == []
    assert [(c.axis, c.provenance) for c in scope.choices] == [("sector", "carried")]


def test_a_stated_word_on_an_axis_the_part_lacks_is_reported_not_applied(monkeypatch):
    """The part has no provider axis, so "bilateral" applies nothing; the part reports it
    rather than dropping it unseen. A carried entry there stays another part's."""
    _write_a_record(monkeypatch)
    carried = _carried("provider", "provider:inclusive")

    scope = _answer(user_message="bilateral fertility in Kenya", carried=[carried]).spec
    scope = scope.resolved.scope_choices

    assert scope is not None
    (dropped,) = scope.dropped_entries
    assert (dropped.entry.option_id, dropped.entry.provenance, dropped.code, dropped.cause) == (
        "provider:bilateral",
        "explicit",
        "stated_choice_not_applied",
        "axis_absent",
    )
    assert dropped.entry.origin.model_dump() == {
        "kind": "text",
        "text": "bilateral",
        "start": 0,
        "end": 9,
    }
    assert scope.entries == []


def test_a_stated_entry_made_for_a_concept_attaches_only_where_its_axis_holds_the_concept():
    """A "health" sector entry on a part with no sector axis (a recipient's total, which
    settles its basis alone) is another part's, so it is not reported dropped there. A stated
    word with no concept on the absent axis is the part's own, and is."""
    health = Entry(
        axis="sector",
        option_id=_HELD,
        provenance="explicit",
        concept="health",
        origin=TextOrigin(text="health", start=0, end=6),
    )
    bilateral = Entry(
        axis="provider",
        option_id="provider:bilateral",
        provenance="explicit",
        origin=TextOrigin(text="bilateral", start=0, end=9),
    )
    basis = AxisRecord(axis="measure_basis", rule="test_basis", active=_NET, provenance="default")

    total = ScopeState(entries=(health, bilateral), records={"measure_basis": basis})
    sector = AxisRecord(
        axis="sector", rule="test_rule", active=_HELD, provenance="default", concept="health"
    )
    health_part = ScopeState(entries=(health,), records={"sector": sector})

    assert total.attached() == (bilateral,)
    assert health_part.attached() == (health,)


def test_an_unavailable_stated_option_is_dropped_and_the_default_derives(monkeypatch):
    """A fresh search never refuses its scope: the stated option the part cannot serve is
    reported with the reason it cannot apply, and the axis takes its default."""
    _write_records(monkeypatch, basis=True)

    scope = _answer(user_message="fertility in Kenya in grant equivalents").spec
    scope = scope.resolved.scope_choices

    assert scope is not None
    assert scope.entries == []
    (dropped,) = scope.dropped_entries
    assert (dropped.entry.option_id, dropped.code, dropped.cause) == (
        _GRANT_EQUIVALENT,
        "stated_choice_not_applied",
        "basis_not_published",
    )
    choice = next(c for c in scope.choices if c.axis == "measure_basis")
    assert choice.provenance == "default"
