"""What a corpus case expects of an LLM leg, read from its status-keyed `expect`.

Bind and extract cells score one LLM call, so they need only a few facts from a case: the
statuses it accepts, the five-tuples that would be the right menu choice, the StatVars an
alternate nomination should reach, and the entities extraction should find. A `gl`
(unconfirmed) value is left out, so a case with unconfirmed values scores on what is confirmed.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from search.eval.cases import (
    CandidatesExpect,
    Case,
    Expect,
    Gl,
    MeasureForm,
    PartExpect,
    ShapeKey,
    StatusExpect,
    StatusKey,
)

# A partial five-tuple keyed by shape key; `None` asserts the dimension is absent.
type ShapeAssertion = Mapping[ShapeKey, str | None]

_SHAPE_STATUSES: tuple[StatusKey, ...] = ("definite", "candidates")


@dataclass(frozen=True, slots=True)
class LegExpectation:
    """One leg's accepted statuses, acceptable menu shapes, and expected StatVars."""

    statuses: tuple[StatusKey, ...]
    shapes: tuple[ShapeAssertion, ...]
    stat_vars: tuple[str, ...]

    @property
    def sole_status(self) -> StatusKey | None:
        """The one accepted status, or None when the case accepts several."""
        return self.statuses[0] if len(self.statuses) == 1 else None


def _scopes(case: Case) -> list[Expect | PartExpect]:
    """The expectation per leg in query order: each part, or the whole expectation."""
    return list(case.expect.parts) if case.expect.parts is not None else [case.expect]


def _blocks(scope: Expect | PartExpect) -> Iterator[tuple[StatusKey, StatusExpect]]:
    for status in scope.accepted_statuses:
        block = scope.for_status(status)
        if block is not None:
            yield status, block


def _forms(block: StatusExpect) -> list[MeasureForm]:
    forms = list(block.measure.forms) if block.measure is not None else []
    if isinstance(block, CandidatesExpect):
        forms.extend(block.family_forms.values())
    return forms


def _confirmed_shape(form: MeasureForm) -> ShapeAssertion | None:
    """The form's shape when every asserted dimension is confirmed and one is non-null."""
    if isinstance(form.shape, Gl) or any(isinstance(v, Gl) for v in form.shape.values()):
        return None
    shape = {key: value for key, value in form.shape.items() if not isinstance(value, Gl)}
    return shape if any(value is not None for value in shape.values()) else None


def _strings(values: list | None) -> list[str]:
    return [v for v in values or [] if isinstance(v, str)]


def leg_expectation(case: Case, leg: int) -> LegExpectation | None:
    """The expectation for one leg, or None when the case has no such leg."""
    scopes = _scopes(case)
    if not 0 <= leg < len(scopes):
        return None
    scope = scopes[leg]
    shapes: list[ShapeAssertion] = []
    stat_vars: list[str] = []
    for status, block in _blocks(scope):
        for form in _forms(block):
            stat_vars.extend(_strings(form.stat_vars))
            if status in _SHAPE_STATUSES and (shape := _confirmed_shape(form)):
                shapes.append(shape)
    return LegExpectation(
        statuses=scope.accepted_statuses,
        shapes=tuple(shapes),
        stat_vars=tuple(dict.fromkeys(stat_vars)),
    )


def case_entities(case: Case) -> set[str]:
    """Every confirmed entity dcid the case expects, across legs and accepted statuses."""
    return {
        entity.dcid
        for scope in _scopes(case)
        for _status, block in _blocks(scope)
        for entity in block.entities or []
        if isinstance(entity.dcid, str)
    }


def case_stat_vars(case: Case) -> set[str]:
    """Every confirmed StatVar the case expects, across legs and accepted statuses."""
    return {
        sv
        for scope in _scopes(case)
        for _status, block in _blocks(scope)
        for form in _forms(block)
        for sv in _strings(form.stat_vars)
    }
