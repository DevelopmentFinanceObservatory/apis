"""Tests for shape.py: is_role_bearing, is_un_shape and role_names_by_axis.

Verifies the declared-roles gate is_role_bearing reads and the flow-axis
placement role_names_by_axis performs.
"""

import pytest

from search.engine.shape import ShapeDraft, is_role_bearing, is_un_shape, role_names_by_axis


def _shape_with_roles(declared_roles: tuple[str, ...] | None) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=declared_roles,
    )


@pytest.mark.parametrize(
    ("declared_roles", "expected"),
    [
        pytest.param(None, False, id="no-declared-roles"),
        pytest.param(("observationAbout",), False, id="subject-only"),
        pytest.param(("provider", "recipient"), True, id="provider-recipient"),
        pytest.param(("creditor", "debtor"), True, id="creditor-debtor"),
    ],
)
def test_is_role_bearing_declared_roles_truth_table(
    declared_roles: tuple[str, ...] | None, expected: bool
) -> None:
    shape = _shape_with_roles(declared_roles=declared_roles)
    assert is_role_bearing(shape) is expected


class TestRoleNamesByAxis:
    def test_declared_names_are_placed_on_their_axes(self) -> None:
        shape = _shape_with_roles(("creditor", "debtor"))
        assert role_names_by_axis(shape) == {"origin": "creditor", "destination": "debtor"}

    def test_subject_only_declaration_has_no_flow_axes(self) -> None:
        shape = _shape_with_roles(("observationAbout",))
        assert role_names_by_axis(shape) == {}

    def test_unstamped_draft_has_no_flow_axes(self) -> None:
        shape = _shape_with_roles(None)
        assert role_names_by_axis(shape) == {}

    def test_unplaceable_declaration_is_role_bearing_with_no_axes(self) -> None:
        shape = _shape_with_roles(("beneficiaryOrganisation",))
        assert is_role_bearing(shape) is True
        assert role_names_by_axis(shape) == {}


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        pytest.param(None, False, id="no-member-facts"),
        pytest.param({}, False, id="empty"),
        pytest.param({"undata/unicef/SERIES-ED_LR": {}}, True, id="un-member"),
        pytest.param({"Count_Person": {}}, False, id="base-member"),
        pytest.param({"undata/unicef/SERIES-ED_LR": {}, "Count_Person": {}}, False, id="mixed"),
    ],
)
def test_is_un_shape_requires_every_member_to_be_a_un_variable(
    members: dict[str, dict] | None, expected: bool
) -> None:
    shape = ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts=members,
    )
    assert is_un_shape(shape) is expected
