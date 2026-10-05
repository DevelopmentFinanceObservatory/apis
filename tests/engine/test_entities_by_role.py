"""Tests conversion from flow axes to declared role names."""

from __future__ import annotations

import pytest

from search.engine.errors import RoleEntityUnplaced
from search.engine.shape import ShapeDraft, entities_by_role


def _draft(declared_roles: tuple[str, ...] | None) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=declared_roles,
    )


def test_declaration_free_shape_binds_the_subject_key() -> None:
    shape = _draft(None)
    assert entities_by_role(shape=shape, subject=["country/KEN"]) == {
        "observationAbout": ["country/KEN"]
    }


def test_declared_vocabulary_binds_its_own_names() -> None:
    shape = _draft(("creditor", "debtor"))
    result = entities_by_role(
        shape=shape,
        by_axis={"origin": ["country/FRA"], "destination": ["country/KEN"]},
    )
    assert result == {"creditor": ["country/FRA"], "debtor": ["country/KEN"]}


def test_empty_axis_list_omits_its_role_key() -> None:
    shape = _draft(("creditor", "debtor"))
    result = entities_by_role(shape=shape, by_axis={"destination": ["country/KEN"]})
    assert result == {"debtor": ["country/KEN"]}
    assert "creditor" not in result


def test_nothing_bound_is_an_empty_map() -> None:
    shape = _draft(None)
    assert entities_by_role(shape=shape) == {}


def test_subject_entity_on_a_declaring_shape_refuses() -> None:
    shape = _draft(("creditor", "debtor"))
    with pytest.raises(RoleEntityUnplaced):
        entities_by_role(shape=shape, subject=["country/KEN"])


def test_entity_on_an_unnamed_axis_refuses() -> None:
    shape = _draft(("creditor",))
    with pytest.raises(RoleEntityUnplaced):
        entities_by_role(shape=shape, by_axis={"destination": ["country/KEN"]})


def test_unplaceable_declaration_never_fabricates_a_subject() -> None:
    shape = _draft(("beneficiaryOrganisation",))
    with pytest.raises(RoleEntityUnplaced):
        entities_by_role(shape=shape, subject=["country/KEN"])
