"""Wire tests for `ResolvedSpec.equivalent_readings`: default omission,
serialization, recursion limits, and resubmit-ready entries.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.models import ResolvedSpec, ResubmitIdentityPayload
from tests.engine._builders import conjoin_spec, shape_five_tuple

_SHAPE = shape_five_tuple("Person", "count")


def _twin_spec(constraint_val: str = "TwinVal", sv_dcid: str = "twin_sv"):
    return conjoin_spec(_SHAPE, constraint_val, sv_dcid=sv_dcid)


def test_defaults_to_none() -> None:
    spec = conjoin_spec(_SHAPE, "Val")
    assert spec.resolved.equivalent_readings is None


def test_omitted_from_serialised_output_when_none() -> None:
    spec = conjoin_spec(_SHAPE, "Val")
    dumped = spec.model_dump(mode="json", exclude_none=True)
    assert "equivalent_readings" not in dumped["resolved"]


def test_populated_field_round_trips_through_build_spec() -> None:
    twin = _twin_spec()
    spec = conjoin_spec(_SHAPE, "Val", equivalent_readings=[twin])

    assert spec.resolved.equivalent_readings == [twin]

    dumped = spec.model_dump(mode="json", exclude_none=True)
    assert dumped["resolved"]["equivalent_readings"][0]["spec_id"] == twin.spec_id


def test_recursion_guard_rejects_entry_with_its_own_equivalent_readings() -> None:
    twin_of_twin = _twin_spec(constraint_val="TwinOfTwin", sv_dcid="twin_of_twin_sv")
    twin = _twin_spec()
    # `model_copy` bypasses validation, allowing the nested state tested below.
    nested_twin = twin.model_copy(
        update={
            "resolved": twin.resolved.model_copy(update={"equivalent_readings": [twin_of_twin]})
        }
    )

    with pytest.raises(ValidationError, match="equivalent_readings"):
        conjoin_spec(_SHAPE, "Val", equivalent_readings=[nested_twin])


def test_recursion_guard_allows_a_twin_with_no_equivalent_readings_of_its_own() -> None:
    twin = _twin_spec()
    spec = conjoin_spec(_SHAPE, "Val", equivalent_readings=[twin])

    # No error: a single level of nesting is the supported shape.
    ResolvedSpec.model_validate(spec.resolved.model_dump())


def test_populated_entry_carries_everything_a_resubmit_identity_requires() -> None:
    twin = _twin_spec()
    spec = conjoin_spec(_SHAPE, "Val", equivalent_readings=[twin])
    entry = spec.resolved.equivalent_readings[0]

    assert entry.spec_id
    assert entry.definition.identity is not None
    assert entry.resolved.slots
    assert entry.resolved.members

    resubmit_input = ResubmitIdentityPayload(
        identity=entry.definition.identity,
    )

    assert resubmit_input.identity == entry.definition.identity
