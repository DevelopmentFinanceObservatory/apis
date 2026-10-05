"""Replays cross-shape conjunction cases and checks each part's status and StatVars."""

from __future__ import annotations

import pytest

from search.eval.cases import load_cases
from tests.engine._harness import make_request, offline_resolve

_CROSS_SHAPE = [
    case
    for case in load_cases()
    if case.tags.get("conjunction") == "cross_shape" and case.expect.parts
]

assert _CROSS_SHAPE, (
    "no cases tagged {conjunction: cross_shape} with parts found in corpus/ -- "
    "corpus tagging regressed silently"
)

_PARAMS = [pytest.param(case, id=case.id) for case in _CROSS_SHAPE]


@pytest.mark.parametrize("case", _PARAMS)
def test_conjunction_cross_shape(case):
    """Each part of a cross_shape case has the status its expectation names, and every
    definite part carries its own variable_text back-pointer and the expected StatVars."""
    result = offline_resolve(make_request(case.query))
    expected_parts = case.expect.parts

    assert [part.status for part in result.parts] == [
        expected.accepted_statuses[0] for expected in expected_parts
    ]

    for i, (part, expected) in enumerate(zip(result.parts, expected_parts, strict=True)):
        if part.status != "definite":
            continue
        assert part.variable_text is not None, f"{case.id}: parts[{i}].variable_text is None"
        [form] = expected.definite.measure.forms
        survivors = [sv.ref for sv in part.spec.resolved.stat_vars.items]
        assert survivors == form.stat_vars, (
            f"{case.id}: expected parts[{i}] StatVars to be {form.stat_vars!r}, got {survivors!r}"
        )
