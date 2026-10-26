"""A parsed-door answer states its scope choices when its binders record an axis, and none
when they record nothing."""

from __future__ import annotations

import pytest

from search.engine.region import phase_answer
from search.models import (
    DefinitePart,
    Option,
    OptionAvailability,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveRequest,
    SlotBinding,
)
from search.scope import rules as scope_rules
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable
from tests.engine._harness import offline_resolve

_HELD = "sector:1"


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


@pytest.fixture(autouse=True)
def _registered_rule(monkeypatch):
    monkeypatch.setattr(scope_rules, "AXIS_RULES", {"sector": _rule})


def _write_a_record(monkeypatch) -> None:
    """Have the definite answer's binders record the synthetic axis."""
    assemble = phase_answer._assemble_definite_answer

    async def recording(state):
        state.scope_selection.write(
            AxisRecord(axis="sector", rule="test_rule", active=_HELD, provenance="default")
        )
        return await assemble(state)

    monkeypatch.setattr(phase_answer, "_assemble_definite_answer", recording)


def _answer() -> DefinitePart:
    response = offline_resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")]
                    )
                ]
            )
        )
    )
    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    return part


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
