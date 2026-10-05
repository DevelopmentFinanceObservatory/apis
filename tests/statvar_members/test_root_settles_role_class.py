"""A walk reports a family complete only when the role class it elects is the full family's.

Each case compares the walk over every member, the true election, with the walk over what a
truncated page or a flaky read shows. A root on a truncated page must elect the role class
(`statvar_members.enumerate._root_elects_role_class`), and a root outside the covering class
dismisses no unread candidate.
"""

from __future__ import annotations

import pytest

from statvar_members import MemberGroup, MemberRoleDisagreement, enumerate_members
from statvar_members.types import FiveTuple

from .conftest import FakeMemberArcReader, FlakyMemberArcReader

_RECIPIENT = ("financingRecipient",)
_PROVIDER = ("financingProvider",)
_PAIR = ("financingProvider", "financingRecipient")
_SUBJECT = ("observationAbout",)

_FIVE_TUPLE = FiveTuple(
    population_type="FinancialPosition",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_CONSTRAINTS = {"financialPositionType": "AmountOutstanding", "guaranteeStatus": "PPG"}
_VALUE_SLOTS = {prop: ("value", value) for prop, value in _CONSTRAINTS.items()}
_WITH_EXTRA = {**_VALUE_SLOTS, "extra": ("unbound",)}


def _arcs(constraints: dict[str, str], roles: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("FinancialPosition",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "measurementDenominator": (),
        "constraintProperties": tuple(constraints),
        "observationProperties": () if roles == _SUBJECT else roles,
        **{prop: (value,) for prop, value in constraints.items()},
    }


def _true_election(
    arcs: dict[str, dict[str, tuple[str, ...]]], slots: dict, role_keys: tuple[str, ...]
) -> tuple[str, ...] | str:
    try:
        group = enumerate_members(
            five_tuple=_FIVE_TUPLE,
            slots=slots,
            graph=FakeMemberArcReader(arcs=arcs),
            role_keys=role_keys,
        )
    except MemberRoleDisagreement:
        return "tie"
    return tuple(member.stat_var for member in group.members)


def _claims_complete(group: MemberGroup) -> bool:
    return (not group.truncated or group.proven_complete) and not group.undismissed_candidates


# (case, root roles, unseen member roles, role keys)
_TRUNCATED = [
    ("no-key one-role root, unseen subject", _RECIPIENT, _SUBJECT, ()),
    ("no-key one-role root, unseen provider-only", _RECIPIENT, _PROVIDER, ()),
    ("no-key two-role root, unseen subject", _PAIR, _SUBJECT, ()),
    ("no-key two-role root, unseen total", _PAIR, _RECIPIENT, ()),
    ("destination, one-role root, unseen by-creditor", _RECIPIENT, _PAIR, ("destination",)),
    ("destination, two-role root, unseen total", _PAIR, _RECIPIENT, ("destination",)),
    (
        "both keys, root lacks origin, unseen by-creditor",
        _RECIPIENT,
        _PAIR,
        ("origin", "destination"),
    ),
    ("origin, root lacks origin, unseen provider-only", _RECIPIENT, _PROVIDER, ("origin",)),
]


@pytest.mark.parametrize(
    ("root", "unseen", "role_keys"),
    [case[1:] for case in _TRUNCATED],
    ids=[case[0] for case in _TRUNCATED],
)
def test_a_truncated_root_never_claims_a_family_whose_election_differs(root, unseen, role_keys):
    full = {"Root": _arcs(_CONSTRAINTS, root), "Unseen": _arcs(_CONSTRAINTS, unseen)}
    truth = _true_election(full, _VALUE_SLOTS, role_keys)

    seen = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_VALUE_SLOTS,
        graph=FakeMemberArcReader(arcs={"Root": full["Root"]}, force_truncated=True),
        role_keys=role_keys,
    )

    if seen.proven_complete:
        assert truth == tuple(member.stat_var for member in seen.members)


# (case, root roles, role keys, the arc a read always drops from the other candidate)
_DISMISSED = [
    ("no-key one-role root, subject candidate", _RECIPIENT, (), "guaranteeStatus"),
    ("no-key two-role root, subject candidate", _PAIR, (), "guaranteeStatus"),
    ("destination, two-role root, total candidate", _PAIR, ("destination",), "guaranteeStatus"),
]


@pytest.mark.parametrize(
    ("root", "role_keys", "dropped"),
    [case[1:] for case in _DISMISSED],
    ids=[case[0] for case in _DISMISSED],
)
def test_a_dismissed_candidate_never_hides_a_different_election(root, role_keys, dropped):
    other = _RECIPIENT if role_keys else _SUBJECT
    full = {
        "Root": _arcs(_CONSTRAINTS, root),
        "Other": _arcs({**_CONSTRAINTS, "extra": "E"}, other),
    }
    truth = _true_election(full, _WITH_EXTRA, role_keys)

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_WITH_EXTRA,
        graph=FlakyMemberArcReader(arcs=full, drop_always={"Other": dropped}),
        role_keys=role_keys,
    )

    if _claims_complete(group):
        assert truth == tuple(member.stat_var for member in group.members)
