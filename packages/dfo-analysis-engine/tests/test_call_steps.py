"""`shed_coverage`: which coverage members a step list over its byte bound sheds, and in what
order."""

from __future__ import annotations

import pytest
import rfc8785

from dfo_analysis_engine.call_steps import shed_coverage


def _entities(count: int) -> list[dict[str, object]]:
    return [{"entity": {"ref": f"country/C{n:02d}"}, "periods": 3, "of": 3} for n in range(count)]


def _periods(count: int) -> list[dict[str, object]]:
    return [{"period": str(2000 + n), "entities": 5} for n in range(count)]


_STEPS: list[dict[str, object]] = [
    {"id": "s1", "op": "rank", "entity_coverage": _entities(4)},
    {"id": "s2", "op": "rank", "entity_coverage": _entities(8)},
    {"id": "s3", "op": "share", "period_coverage": _periods(6)},
]


def _without(*members: tuple[int, str]) -> list[dict[str, object]]:
    steps = [dict(step) for step in _STEPS]
    for index, member in members:
        del steps[index][member]
    return steps


@pytest.mark.parametrize(
    ("max_bytes", "expected", "shed"),
    [
        (len(rfc8785.dumps(_STEPS)), _STEPS, 0),
        (len(rfc8785.dumps(_STEPS)) - 1, _without((1, "entity_coverage")), 1),
        (
            len(rfc8785.dumps(_without((1, "entity_coverage"), (0, "entity_coverage")))) - 1,
            _without((1, "entity_coverage"), (0, "entity_coverage"), (2, "period_coverage")),
            3,
        ),
        (10, _without((1, "entity_coverage"), (0, "entity_coverage"), (2, "period_coverage")), 3),
    ],
    ids=["fits", "largest-entity-first", "entities-then-periods", "cannot-fit"],
)
def test_a_list_over_its_bound_sheds_coverage_largest_first_until_it_fits(
    max_bytes: int, expected: list[dict[str, object]], shed: int
):
    fitted, count = shed_coverage(_STEPS, size=len(rfc8785.dumps(_STEPS)), max_bytes=max_bytes)

    assert (fitted, count) == (expected, shed)
    assert len(rfc8785.dumps(fitted)) <= max_bytes or shed == 3
    assert "entity_coverage" in _STEPS[1]
