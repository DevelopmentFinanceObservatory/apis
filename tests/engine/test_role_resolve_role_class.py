"""The members a role-bearing answer lists are the ones the mint serves.

`role_resolve.role_members_served` keeps only members the mint's walk can admit
(`_mint_admits`): a member asserting a constraint whose value node does not resolve fails
the walk's own test (`statvar_members.constraints_resolve`), so it is dropped before the
role-class election. A member whose constraints are unknown takes no part in the election
and is kept. The last test runs the mint's walk over the same members.
"""

from __future__ import annotations

import pytest

from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.role_resolve import role_members_served
from statvar_members import UNRESOLVED_NODE, enumerate_members
from statvar_members.types import FiveTuple
from tests.statvar_members.conftest import FakeMemberArcReader

_RECIPIENT = ("financingRecipient",)
_PAIR = ("financingProvider", "financingRecipient")


def _arcs(roles: tuple[str, ...], **constraints: str | None) -> dict:
    """Graph arcs of a member; a constraint given `None` has a value node with no dcid."""
    return {
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        "observationProperties": {"nodes": [{"dcid": r} for r in roles]},
        **{
            p: {"nodes": [{"dcid": v}] if v is not None else [{"name": "no dcid"}]}
            for p, v in constraints.items()
        },
    }


@pytest.mark.parametrize(
    ("members", "served"),
    [
        pytest.param(
            {
                "Total": _arcs(_RECIPIENT, flowType="X"),
                "Part": _arcs(_RECIPIENT, flowType="X", guarantee=None),
            },
            ["Total"],
            id="a-part-under-its-total",
        ),
        pytest.param(
            {"Total": _arcs(_RECIPIENT, flowType="X"), "Other": _arcs(_RECIPIENT, indicator=None)},
            ["Total"],
            id="an-undominated-member-of-the-same-roles",
        ),
        pytest.param(
            {"Total": _arcs(_RECIPIENT, flowType="X"), "Other": _arcs(_PAIR, indicator=None)},
            ["Total"],
            id="an-undominated-member-of-another-role-set",
        ),
        pytest.param(
            {"Total": _arcs(_RECIPIENT, flowType="X"), "Unread": {}},
            ["Total", "Unread"],
            id="a-member-whose-constraints-are-unknown",
        ),
    ],
)
def test_a_member_with_an_unresolved_constraint_is_not_listed(
    members: dict[str, dict], served: list[str]
):
    facts = {sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in members.items()}

    assert role_members_served(facts, {}, ["destination"], ()) == served


def test_the_mint_serves_the_total_beside_an_unresolved_member_of_another_role_set():
    five_tuple = FiveTuple(
        population_type="FinancialPosition",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
    )

    def _walk_arcs(roles: tuple[str, ...], **constraints: str) -> dict[str, tuple[str, ...]]:
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("FinancialPosition",),
            "measuredProperty": ("amount",),
            "statType": ("measuredValue",),
            "measurementQualifier": ("Nominal",),
            "measurementDenominator": (),
            "constraintProperties": tuple(constraints),
            "observationProperties": roles,
            **{p: (v,) for p, v in constraints.items()},
        }

    group = enumerate_members(
        five_tuple=five_tuple,
        slots={"flowType": ("unbound",), "indicator": ("unbound",)},
        graph=FakeMemberArcReader(
            arcs={
                "Total": _walk_arcs(_RECIPIENT, flowType="X"),
                "Other": _walk_arcs(_PAIR, indicator=UNRESOLVED_NODE),
            }
        ),
        role_keys=("destination",),
    )

    assert [member.stat_var for member in group.members] == ["Total"]


def test_a_member_declaring_a_constraint_with_no_value_arc_is_not_listed():
    """`Other` lists `indicator` in constraintProperties and carries no `indicator` arc. The
    mint's walk rejects it for the absent value, so it does not read as the total of
    `Total`, and both sides serve `Total`."""
    other = {
        "constraintProperties": {"nodes": [{"dcid": "indicator"}]},
        "observationProperties": {"nodes": [{"dcid": r} for r in _RECIPIENT]},
    }
    facts = {
        "Total": ArcsMemberFacts(arcs=_arcs(_RECIPIENT, flowType="X")),
        "Other": ArcsMemberFacts(arcs=other),
    }

    assert role_members_served(facts, {}, ["destination"], ()) == ["Total"]

    walk_arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("FinancialPosition",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "measurementDenominator": (),
        "observationProperties": _RECIPIENT,
    }
    group = enumerate_members(
        five_tuple=FiveTuple(
            population_type="FinancialPosition",
            measured_property="amount",
            stat_type="measuredValue",
            measurement_qualifier="Nominal",
        ),
        slots={"flowType": ("unbound",), "indicator": ("unbound",)},
        graph=FakeMemberArcReader(
            arcs={
                "Total": {**walk_arcs, "constraintProperties": ("flowType",), "flowType": ("X",)},
                "Other": {**walk_arcs, "constraintProperties": ("indicator",)},
            }
        ),
        role_keys=("destination",),
    )
    assert [member.stat_var for member in group.members] == ["Total"]
