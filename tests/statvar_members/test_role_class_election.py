"""`elect_role_class` picks the role class that answers the bound role keys.

A member lacking a role its sibling carries is the aggregate over that role: a
recipient-only debt total beside its by-creditor variant, a provider-only line beside
provider-and-recipient rows. Role coverage runs first, then the antichain over constraint
keys, then the role-twin collapse: of two members with equal constraint maps whose role
sets nest, the one declaring fewer roles survives. Of the members left, those of the
smallest role set answer when every other role set left contains it and their constraint
keys are among those of every other member left. An open role key
joins the bound ones only where a member the bound keys elect, or a role twin of one,
places it. `enumerate_members` runs the same election on the graph walk.

The IDS fixtures mirror the dev graph: a total declaring `financingRecipient` and a
by-creditor twin declaring `financingProvider, financingRecipient` with identical
constraints, plus a lender-named two-role variable in the same five-tuple.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

import pytest

import statvar_members.enumerate as enumerate_module
from statvar_members import (
    CANDIDATE_CAP,
    AbsentReading,
    MemberRoleDisagreement,
    antichain_minimal_ids,
    covers_role_keys,
    elect_role_class,
    elect_role_class_of,
    electing_role_keys,
    enumerate_members,
    role_class_facts,
)
from statvar_members.enumerate import _ROLE_PROBE_BUDGET
from statvar_members.types import FiveTuple

from .conftest import FakeMemberArcReader, FlakyMemberArcReader

_RECIPIENT = ("financingRecipient",)
_PAIR = ("financingProvider", "financingRecipient")
_PROVIDER = ("financingProvider",)
_SUBJECT = ("observationAbout",)

_FLOW = {"flowType": "ServicingPayment"}
_FLOW_AND_GUARANTEE = {"flowType": "ServicingPayment", "guarantee": "Public"}


def _elect(
    members: dict[str, tuple[dict[str, str], tuple[str, ...]]],
    *role_keys: str,
    open_role_keys: tuple[str, ...] = (),
):
    return elect_role_class(
        {dcid: constraints for dcid, (constraints, _roles) in members.items()},
        {dcid: roles for dcid, (_keys, roles) in members.items()},
        list(members),
        role_keys=role_keys,
        open_role_keys=open_role_keys,
        readings={},
    )


_DEBT = {
    "total": (_FLOW, _RECIPIENT),
    "total_guaranteed": (_FLOW_AND_GUARANTEE, _RECIPIENT),
    "by_creditor": (_FLOW, _PAIR),
    "by_creditor_guaranteed": (_FLOW_AND_GUARANTEE, _PAIR),
}


def test_a_recipient_alone_elects_the_recipient_only_total():
    election = _elect(_DEBT, "destination")

    assert election.members == ("total",)
    assert election.family == ("total", "total_guaranteed")
    assert not election.tied


def test_no_bound_role_elects_the_recipient_only_total():
    election = _elect(_DEBT)

    assert election.members == ("total",)
    assert election.family == ("total", "total_guaranteed")


def test_a_named_creditor_elects_the_by_creditor_rows():
    election = _elect(_DEBT, "origin", "destination")

    assert election.members == ("by_creditor",)
    assert election.family == ("by_creditor", "by_creditor_guaranteed")
    assert not election.tied


def test_an_open_creditor_elects_the_by_creditor_rows():
    election = _elect(_DEBT, "destination", open_role_keys=("origin",))

    assert election.members == ("by_creditor",)
    assert election.family == ("by_creditor", "by_creditor_guaranteed")


def test_an_open_origin_no_member_declares_elects_under_the_bound_keys_alone():
    """A recipient-only total beside a subject-only member of the same constraints, with
    an open origin. No member covers it, so coverage reads the recipient alone and the
    subject-only member drops, where keeping every member would tie the two role sets."""
    family = {"total": (_FLOW, _RECIPIENT), "subject": (_FLOW, _SUBJECT)}

    election = _elect(family, "destination", open_role_keys=("origin",))

    assert election == _elect(family, "destination")
    assert election.members == ("total",)
    assert not election.tied


_DISBURSEMENT = {"flowType": "Disbursement"}
_UNTWINNED = {"total": (_FLOW, _RECIPIENT), "lender": (_DISBURSEMENT, _PAIR)}


def _same_map(*role_sets: tuple[str, ...]) -> dict[str, tuple[dict[str, str], tuple[str, ...]]]:
    """One member per role set, all of one constraint map, so each is a twin of the rest."""
    return {f"m{position}": (_FLOW, roles) for position, roles in enumerate(role_sets)}


@pytest.mark.parametrize(
    ("members", "role_keys", "open_role_keys", "expected"),
    [
        (_same_map(_PAIR), (), ("origin", "destination"), {"origin", "destination"}),
        (_same_map(_PAIR, _RECIPIENT), (), ("origin", "destination"), {"origin", "destination"}),
        (_same_map(_RECIPIENT), (), ("origin", "destination"), {"destination"}),
        (_same_map(_RECIPIENT, _SUBJECT), (), ("origin", "destination"), {"destination"}),
        (_same_map(_RECIPIENT, _PROVIDER), (), ("origin", "destination"), set()),
        (_same_map(_SUBJECT), (), ("origin", "destination"), set()),
        (_same_map(_PAIR), ("destination",), ("origin",), {"destination", "origin"}),
        (_same_map(_RECIPIENT), ("destination",), ("origin",), {"destination"}),
        (_same_map(_PAIR), ("destination",), (), {"destination"}),
        (_same_map(_PAIR), (), (), set()),
        (_same_map(_RECIPIENT, _PAIR), ("destination",), ("origin",), {"destination", "origin"}),
        (_UNTWINNED, ("destination",), ("origin",), {"destination"}),
        (_UNTWINNED, (), ("origin",), set()),
        (_UNTWINNED, (), ("origin", "destination"), {"destination"}),
        (
            {"subject": (_DISBURSEMENT, _SUBJECT), "total": (_FLOW, _RECIPIENT)},
            (),
            ("destination",),
            set(),
        ),
    ],
    ids=[
        "both open keys covered",
        "the pair wins over a covered single",
        "only one open key covered keeps it",
        "one covered beside a subject-only twin",
        "two single keys tie with no pair covered",
        "no open key covered",
        "open key joins a bound key",
        "open key no member covers leaves the bound key",
        "no open keys returns the bound keys",
        "no keys at all",
        "a twin of the elected total places the open key",
        "a two-role member of other values places nothing beside the total",
        "the same with no key bound",
        "the total places its own key and the other member's is set aside",
        "a role-bearing member of other values is no twin of the subject-only one",
    ],
)
def test_electing_role_keys_keeps_the_largest_open_subset_the_elected_members_place(
    members, role_keys, open_role_keys, expected
):
    facts = role_class_facts(
        {dcid: constraints for dcid, (constraints, _roles) in members.items()},
        {dcid: roles for dcid, (_constraints, roles) in members.items()},
    )

    keys = electing_role_keys(
        facts, list(members), role_keys=role_keys, open_role_keys=open_role_keys, readings={}
    )

    assert keys == frozenset(expected)
    assert _elect(members, *role_keys, open_role_keys=open_role_keys).role_keys == keys


def test_a_creditor_alone_elects_the_by_creditor_rows():
    election = _elect(_DEBT, "origin")

    assert election.members == ("by_creditor",)


def test_a_declared_role_name_is_a_role_key():
    election = _elect(_DEBT, "financingProvider")

    assert election.members == ("by_creditor",)


def test_a_subject_key_elects_the_subject_only_class():
    members = {
        "gdp": ({"activity": "GDP"}, _SUBJECT),
        "gni_received": ({"activity": "GDP"}, _RECIPIENT),
    }

    assert _elect(members, "observationAbout").family == ("gdp",)
    assert _elect(members, "destination").family == ("gni_received",)


def test_no_role_key_elects_the_subject_only_class():
    """An identity that binds no entity is about its subject. A role-bearing member that
    asserts fewer constraints, or as many, does not displace the subject-only class."""
    equal_keys = {
        "gdp": ({"activity": "GDP"}, _SUBJECT),
        "gni_received": ({"activity": "GDP"}, _RECIPIENT),
    }
    fewer_keys = {
        "gdp_by_sector": ({"activity": "GDP", "sector": "Health"}, _SUBJECT),
        "gni_received": ({"activity": "GDP"}, _RECIPIENT),
        "gni_by_creditor": ({"activity": "GDP"}, _PAIR),
    }

    assert _elect(equal_keys).family == ("gdp",)
    assert not _elect(equal_keys).tied
    assert _elect(fewer_keys).members == ("gdp_by_sector",)
    assert _elect(fewer_keys).family == ("gdp_by_sector",)


def test_no_role_key_without_a_subject_only_member_keeps_every_member():
    members = {
        "line": (_FLOW, _PROVIDER),
        "total": (_FLOW, _RECIPIENT),
        "by_creditor": (_FLOW, _PAIR),
    }

    election = _elect(members)

    assert election.tied
    assert election.members == ("line", "total")


def test_provider_only_and_recipient_only_members_tie_without_a_role_key():
    members = {"line": (_FLOW, _PROVIDER), "total": (_FLOW, _RECIPIENT)}

    assert _elect(members).tied
    assert _elect(members, "origin").members == ("line",)
    assert _elect(members, "destination").members == ("total",)


def test_a_provider_only_row_beside_provider_and_recipient_rows():
    """A DAC 1-style line is the total over recipients: it answers a provider alone, and
    a bound recipient leaves the two-role rows."""
    members = {
        "line": (_FLOW, _PROVIDER),
        "flow_a": (_FLOW, _PAIR),
        "flow_b": (_FLOW, _PAIR),
    }

    assert _elect(members, "origin").members == ("line",)
    assert _elect(members).members == ("line",)
    with_recipient = _elect(members, "origin", "destination")
    assert with_recipient.members == ("flow_a", "flow_b")
    assert with_recipient.family == ("flow_a", "flow_b")


def test_role_keys_no_member_covers_keep_every_member():
    """The caller's reproducibility check names the undeclared role."""
    members = {"flow_a": (_FLOW, _PAIR), "flow_b": (_FLOW, _PAIR)}

    election = _elect(members, "observationAbout")

    assert election.family == ("flow_a", "flow_b")
    assert not election.tied


def test_constraint_keys_reduce_before_roles():
    """A by-creditor root answers a recipient when the recipient-only class holds only a
    narrower slice: the slice is dominated on its constraint keys, as it is without roles."""
    members = {
        "by_creditor": (_FLOW, _PAIR),
        "total_guaranteed": (_FLOW_AND_GUARANTEE, _RECIPIENT),
    }

    election = _elect(members, "destination")

    assert election.members == ("by_creditor",)
    assert election.family == ("by_creditor",)
    assert not election.tied


def test_a_total_ties_with_a_two_role_member_of_other_constraint_keys():
    """Neither constraint key set contains the other, so both members survive the
    antichain. The recipient-only member breaks down by a property the by-creditor member
    does not hold, so it is not that member's total and the two tie. The by-creditor
    member answers once the creditor is bound."""
    members = {
        "total_by_flow": (_FLOW, _RECIPIENT),
        "by_creditor_by_guarantee": ({"guarantee": "Public"}, _PAIR),
    }

    election = _elect(members, "destination")

    assert election.tied
    assert election.members == tuple(members)
    assert election.family == tuple(members)
    assert _elect(members, "origin", "destination").members == ("by_creditor_by_guarantee",)


@pytest.mark.parametrize(
    ("by_creditor", "tied"),
    [
        pytest.param(_FLOW_AND_GUARANTEE, False, id="keys inside the by-creditor member's"),
        pytest.param({"guarantee": "Public", "term": "Long"}, True, id="a key it lacks"),
    ],
)
def test_a_total_answers_only_when_its_keys_are_among_a_two_role_members(by_creditor, tied):
    """`guarantee` reads `not_declared`, so the total survives beside a by-creditor member
    holding it. The total answers when the by-creditor member holds every key of the
    total, and ties when the total holds a key that member lacks."""
    members = {"total": _FLOW, "by_creditor": by_creditor}

    election = elect_role_class(
        members,
        {"total": _RECIPIENT, "by_creditor": _PAIR},
        list(members),
        role_keys=("destination",),
        readings={"guarantee": "not_declared"},
    )

    assert election.tied is tied
    assert election.members == (tuple(members) if tied else ("total",))


_DOMINATED_TOTAL = {
    "total_by_flow": (_FLOW, _RECIPIENT),
    "total_by_guarantee_and_term": ({"guarantee": "Public", "term": "Long"}, _RECIPIENT),
    "by_creditor_by_guarantee": ({"guarantee": "Public"}, _PAIR),
}


def test_a_total_only_a_two_role_member_dominates_stays_a_member():
    """`guarantee` reads `not_declared`, so the by-creditor member survives beside the
    total by flow and leaves for it. It then dominates nothing: the members are the
    antichain of the recipient-only family alone."""
    members = {
        "total_by_flow": _FLOW,
        "total_by_guarantee_and_term": {**_FLOW_AND_GUARANTEE, "term": "Long"},
        "by_creditor_by_guarantee": _FLOW_AND_GUARANTEE,
    }

    election = elect_role_class(
        members,
        {
            "total_by_flow": _RECIPIENT,
            "total_by_guarantee_and_term": _RECIPIENT,
            "by_creditor_by_guarantee": _PAIR,
        },
        list(members),
        role_keys=("destination",),
        readings={"guarantee": "not_declared"},
    )

    assert election.members == ("total_by_flow", "total_by_guarantee_and_term")
    assert election.family == election.members
    assert not election.tied


def test_role_sets_that_do_not_nest_tie_on_different_constraint_keys():
    members = {
        "total_by_flow": (_FLOW, _RECIPIENT),
        "line_by_guarantee": ({"guarantee": "Public"}, _PROVIDER),
        "by_creditor_by_term": ({"term": "Long"}, _PAIR),
    }

    election = _elect(members)

    assert election.tied
    assert election.members == tuple(members)
    assert election.family == tuple(members)


def test_a_not_declared_property_blocks_dominance_inside_a_role_class():
    election = elect_role_class(
        {"total": _FLOW, "total_guaranteed": _FLOW_AND_GUARANTEE, "by_creditor": _FLOW},
        {"total": _RECIPIENT, "total_guaranteed": _RECIPIENT, "by_creditor": _PAIR},
        ["total", "total_guaranteed", "by_creditor"],
        role_keys=("destination",),
        readings={"guarantee": "not_declared"},
    )

    assert election.members == ("total", "total_guaranteed")
    assert not election.tied


@pytest.mark.parametrize("open_role_keys", [(), ("origin",)], ids=["no open key", "open creditor"])
def test_a_total_answers_beside_a_by_creditor_member_of_other_constraint_values(open_role_keys):
    """Equal constraint keys with another value are a different population, so the total
    does not displace the by-creditor member as a twin. The total still answers a
    question that binds no creditor, and an open creditor no twin of the total places is
    set aside."""
    election = _elect(_UNTWINNED, "destination", open_role_keys=open_role_keys)

    assert election.members == ("total",)
    assert election.family == ("total",)
    assert not election.tied
    assert election.role_keys == {"destination"}


def test_a_bound_creditor_reads_the_by_creditor_member_of_other_constraint_values():
    election = _elect(_UNTWINNED, "origin", "destination")

    assert election.members == ("lender",)
    assert election.role_keys == {"origin", "destination"}


_DAC1 = {
    "donor_grants": (_FLOW, _PROVIDER),
    "donor_grants_public": (_FLOW_AND_GUARANTEE, _PROVIDER),
    "donor_loans": (_DISBURSEMENT, _PROVIDER),
    "to_west_bank": ({"flowType": "Arrears"}, _PAIR),
    "to_west_bank_public": ({"flowType": "Arrears", "guarantee": "Public"}, _PAIR),
}
_DONOR_ROWS = ("donor_grants", "donor_loans")
_DONOR_FAMILY = ("donor_grants", "donor_grants_public", "donor_loans")


@pytest.mark.parametrize(
    ("role_keys", "open_role_keys", "members", "family", "elected_keys"),
    [
        (("origin",), (), _DONOR_ROWS, _DONOR_FAMILY, {"origin"}),
        (("origin",), ("destination",), _DONOR_ROWS, _DONOR_FAMILY, {"origin"}),
        ((), ("destination",), _DONOR_ROWS, _DONOR_FAMILY, set()),
        ((), (), _DONOR_ROWS, _DONOR_FAMILY, set()),
        (
            ("origin", "destination"),
            (),
            ("to_west_bank",),
            ("to_west_bank", "to_west_bank_public"),
            {"origin", "destination"},
        ),
        (
            ("destination",),
            ("origin",),
            ("to_west_bank",),
            ("to_west_bank", "to_west_bank_public"),
            {"origin", "destination"},
        ),
    ],
    ids=[
        "donor bound",
        "donor bound, recipient open",
        "nothing bound, recipient open",
        "nothing bound",
        "donor and recipient bound",
        "recipient bound, donor open",
    ],
)
def test_donor_totals_beside_donor_and_recipient_rows_of_other_values(
    role_keys, open_role_keys, members, family, elected_keys
):
    """A DAC 1-style table: donor totals, and two rows to one recipient at constraint
    values no total holds. The totals answer until a recipient is bound, and an open
    recipient no total and no twin of one places is set aside."""
    election = _elect(_DAC1, *role_keys, open_role_keys=open_role_keys)

    assert election.members == members
    assert election.family == family
    assert not election.tied
    assert election.role_keys == elected_keys


_TIE = {"line": (_FLOW, _PROVIDER), "total": (_DISBURSEMENT, _RECIPIENT)}
_KEY_CELLS = [
    (role_keys, open_role_keys)
    for role_keys in ((), ("origin",), ("destination",), ("origin", "destination"))
    for open_role_keys in ((), ("origin",), ("destination",), ("origin", "destination"))
]


@pytest.mark.parametrize(
    "members",
    [_DEBT, _UNTWINNED, _DAC1, _TIE, _DOMINATED_TOTAL],
    ids=["twins", "untwinned", "dac1", "tie", "dominated total"],
)
@pytest.mark.parametrize(("role_keys", "open_role_keys"), _KEY_CELLS)
def test_an_election_repeats_over_its_family_and_under_its_elected_keys(
    members, role_keys, open_role_keys
):
    """The mint elects over the family Search stated, under the open keys the identity
    carries, which are the ones the first election read: each repeat returns the same
    family and members. The mint serves the antichain of the family, which is the
    election's members unless it ties."""
    election = _elect(members, *role_keys, open_role_keys=open_role_keys)
    family_antichain = antichain_minimal_ids(
        {dcid: frozenset(members[dcid][0]) for dcid in election.family},
        election.family,
        readings={},
    )
    elected_open = tuple(election.role_keys - set(role_keys))

    over_family = _elect(
        {dcid: members[dcid] for dcid in election.family},
        *role_keys,
        open_role_keys=open_role_keys,
    )
    elected_open_asked = _elect(members, *role_keys, open_role_keys=elected_open)
    elected_bound = _elect(members, *election.role_keys)

    assert over_family.family == election.family
    assert over_family.members == election.members
    assert election.tied or election.members == family_antichain
    assert elected_open_asked == election
    assert elected_bound == election


def test_a_named_creditor_keeps_a_by_creditor_member_of_other_values():
    members = {
        "by_creditor": (_FLOW, _PAIR),
        "lender": ({"flowType": "Disbursement"}, _PAIR),
        "total": (_FLOW, _RECIPIENT),
    }

    election = _elect(members, "origin", "destination")

    assert election.members == ("by_creditor", "lender")
    assert not election.tied


def test_crs_members_declaring_both_roles_never_collapse():
    members = {"GrantsA": ({"crsPurpose": "A"}, _PAIR), "GrantsB": ({"crsPurpose": "A"}, _PAIR)}

    for role_keys in ((), ("destination",), ("origin", "destination")):
        assert _elect(members, *role_keys).members == ("GrantsA", "GrantsB")


def _quadratic_twin_reference(
    roles_by_id: Mapping[str, tuple[str, ...]],
    constraints_by_id: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]:
    """Reference: compare every member with every other member."""
    return tuple(
        item_id
        for item_id, roles in roles_by_id.items()
        if not any(
            frozenset(other_roles) < frozenset(roles)
            and constraints_by_id[other] == constraints_by_id[item_id]
            for other, other_roles in roles_by_id.items()
        )
    )


def _undisplaced(
    constraints_by_id: Mapping[str, Mapping[str, str]],
    roles_by_id: Mapping[str, tuple[str, ...]],
    order: list[str],
) -> tuple[str, ...]:
    """The ids of `order` the election's twin collapse keeps, in their order."""
    displaced = enumerate_module._displaced(role_class_facts(constraints_by_id, roles_by_id), order)
    return tuple(item_id for item_id in order if item_id not in displaced)


_ROLE_POOL = ("a", "b", "c", "d")


@pytest.mark.parametrize("seed", range(200))
def test_the_twin_collapse_matches_the_quadratic_reference(seed: int):
    """The twin collapse keeps the members no member of the same constraint map and
    fewer roles displaces, in any order given."""
    rng = random.Random(seed)
    roles = {
        f"m{i}": tuple(rng.sample(_ROLE_POOL, rng.randint(1, len(_ROLE_POOL))))
        for i in range(rng.randint(1, 40))
    }
    constraints = {item_id: {"k": str(rng.randrange(rng.randint(1, 5)))} for item_id in roles}

    kept = _undisplaced(constraints, roles, list(roles))

    assert kept == _quadratic_twin_reference(roles, constraints)
    shuffled = list(roles)
    rng.shuffle(shuffled)
    assert set(_undisplaced(constraints, roles, shuffled)) == set(kept)


class _CountingRoleSet(frozenset):
    comparisons = 0

    def __lt__(self, other):
        type(self).comparisons += 1
        return super().__lt__(other)


def test_the_twin_collapse_compares_distinct_role_sets_for_a_crs_sized_family(monkeypatch):
    """12,771 members over four constraint maps and three role sets compare at most
    4 x 3 x 3 role sets, where comparing every pair would take millions."""
    rng = random.Random(0)
    roles = {f"m{i}": rng.choice([("a", "b"), ("a",), ("b",)]) for i in range(12_771)}
    constraints = {item_id: {"k": str(rng.randrange(4))} for item_id in roles}
    monkeypatch.setattr(enumerate_module, "frozenset", _CountingRoleSet, raising=False)
    _CountingRoleSet.comparisons = 0

    kept = _undisplaced(constraints, roles, list(roles))

    assert len(kept) < len(roles)
    assert _CountingRoleSet.comparisons <= 4 * 3 * 3


@pytest.mark.parametrize("open_role_keys", [(), ("origin", "destination")])
@pytest.mark.parametrize(
    "role_keys", [(), ("origin",), ("origin", "destination"), ("financingRecipient",)]
)
def test_facts_built_once_elect_any_subset_as_the_subset_alone(role_keys, open_role_keys):
    """`elect_role_class_of` over facts built for every id elects a subset as
    `elect_role_class` elects it from the subset's own maps, twins whose partner is outside
    the subset included. The members of an untied election are the antichain of its
    family alone."""
    rng = random.Random(1)
    roles = {f"m{i}": rng.choice([_PAIR, _PROVIDER, _RECIPIENT, _SUBJECT]) for i in range(400)}
    constraints = {
        item_id: {key: str(rng.randrange(3)) for key in ("a", "b", "c") if rng.random() < 0.7}
        for item_id in roles
    }
    readings: Mapping[str, AbsentReading] = {"b": "not_declared", "c": "not_declared"}
    facts = role_class_facts(constraints, roles)

    for _ in range(50):
        subset = [item_id for item_id in roles if rng.random() < 0.3]
        alone = elect_role_class(
            {item_id: constraints[item_id] for item_id in subset},
            {item_id: roles[item_id] for item_id in subset},
            subset,
            role_keys=role_keys,
            open_role_keys=open_role_keys,
            readings=readings,
        )
        of_facts = elect_role_class_of(
            facts, subset, role_keys=role_keys, open_role_keys=open_role_keys, readings=readings
        )

        assert of_facts == alone
        assert alone.tied or alone.members == antichain_minimal_ids(
            {item_id: frozenset(constraints[item_id]) for item_id in alone.family},
            alone.family,
            readings=readings,
        )


def test_an_axis_two_declared_roles_share_covers_no_axis_key():
    assert not covers_role_keys(("recipient", "debtor"), ("destination",))
    assert covers_role_keys(("recipient", "debtor"), ("debtor",))


# ---------------------------------------------------------------------------
# The graph walk
# ---------------------------------------------------------------------------

_FIVE_TUPLE = FiveTuple(
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_SLOTS = {"flowType": ("value", "ServicingPayment"), "guarantee": ("unbound",)}


def _arcs(roles: tuple[str, ...], *, guarantee: str | None = None) -> dict[str, tuple[str, ...]]:
    constraints = ("flowType",) if guarantee is None else ("flowType", "guarantee")
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("FinancialFlow",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "measurementDenominator": (),
        "constraintProperties": constraints,
        "observationProperties": roles,
        "flowType": ("ServicingPayment",),
        "guarantee": () if guarantee is None else (guarantee,),
    }


_DEBT_ARCS = {
    "Service": _arcs(_RECIPIENT),
    "Service_Guaranteed": _arcs(_RECIPIENT, guarantee="Public"),
    "Service_ByCreditor": _arcs(_PAIR),
    "Service_Guaranteed_ByCreditor": _arcs(_PAIR, guarantee="Public"),
}


def _walk(*role_keys: str, arcs=_DEBT_ARCS, reader=None, open_role_keys: tuple[str, ...] = ()):
    return enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_SLOTS,
        graph=reader or FakeMemberArcReader(arcs=arcs),
        role_keys=role_keys,
        open_role_keys=open_role_keys,
    )


class _SequencedReader(FakeMemberArcReader):
    """Serves `reads[dcid][n]` on the nth forward read of `dcid`, the last one thereafter.
    Every other dcid, and the reverse query, read `arcs`."""

    def __init__(self, *, arcs, reads: Mapping[str, list[dict[str, tuple[str, ...]] | None]]):
        super().__init__(arcs=arcs)
        self._reads = reads
        self._count: dict[str, int] = {}

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        result = super().node_arcs_many(dcids, properties=properties)
        for dcid, served in self._reads.items():
            if dcid in result:
                position = self._count.get(dcid, 0)
                self._count[dcid] = position + 1
                read = served[min(position, len(served) - 1)]
                result[dcid] = (
                    None if read is None else {prop: read.get(prop, ()) for prop in properties}
                )
        return result


def test_the_walk_answers_a_recipient_with_the_plain_total():
    group = _walk("destination")

    assert [member.stat_var for member in group.members] == ["Service"]
    assert group.family == ("Service", "Service_Guaranteed")
    assert group.members[0].declared_roles == _RECIPIENT


def test_the_walk_answers_a_named_creditor_with_the_by_creditor_rows():
    group = _walk("origin", "destination")

    assert [member.stat_var for member in group.members] == ["Service_ByCreditor"]
    assert group.family == ("Service_ByCreditor", "Service_Guaranteed_ByCreditor")


def test_the_walk_without_a_role_key_answers_with_the_plain_total():
    assert [member.stat_var for member in _walk().members] == ["Service"]


def test_the_walk_without_a_role_key_answers_with_the_subject_only_members():
    arcs = {**_DEBT_ARCS, "Service_Subject_Guaranteed": _arcs((), guarantee="Public")}

    group = _walk(arcs=arcs)

    assert [member.stat_var for member in group.members] == ["Service_Subject_Guaranteed"]
    assert group.family == ("Service_Subject_Guaranteed",)
    assert group.members[0].declared_roles == _SUBJECT


def test_the_walk_refuses_role_sets_it_cannot_order():
    arcs = {"Service": _arcs(_RECIPIENT), "Service_Line": _arcs(_PROVIDER)}

    with pytest.raises(MemberRoleDisagreement) as exc_info:
        _walk(arcs=arcs)

    assert exc_info.value.roles_by_stat_var == {
        "Service": _RECIPIENT,
        "Service_Line": _PROVIDER,
    }


def test_a_role_key_breaks_the_tie_between_role_sets_the_walk_cannot_order():
    arcs = {"Service": _arcs(_RECIPIENT), "Service_Line": _arcs(_PROVIDER)}

    assert [member.stat_var for member in _walk("origin", arcs=arcs).members] == ["Service_Line"]


def test_a_dropped_role_arc_does_not_cost_the_walk_a_covering_member():
    """A batch read that drops `observationProperties` makes a member read subject-only.
    The walk re-reads it before role coverage drops it."""
    arcs = {
        "Service_ByCreditor": _arcs(_PAIR),
        "Service_Guaranteed_ByCreditor": _arcs(_PAIR, guarantee="Public"),
    }
    reader = FlakyMemberArcReader(
        arcs=arcs, drop_first_batch={"Service_ByCreditor": "observationProperties"}
    )

    group = _walk("origin", "destination", reader=reader)

    assert [member.stat_var for member in group.members] == ["Service_ByCreditor"]
    assert group.members[0].declared_roles == _PAIR
    assert group.inconsistent_candidates == ()


def test_a_by_creditor_twin_read_only_as_a_root_twin_answers_an_open_creditor():
    """The twin reads empty on the batch and on the filter re-probe, so the election before
    the twin pass sees the recipient-only total alone and reads the destination alone. Read
    as a root twin, the twin covers the open origin and the final election elects it."""
    arcs = {"Service": _arcs(_RECIPIENT), "Service_ByCreditor": _arcs(_PAIR)}
    reader = _SequencedReader(
        arcs=arcs, reads={"Service_ByCreditor": [None, None, arcs["Service_ByCreditor"]]}
    )

    group = _walk("destination", reader=reader, open_role_keys=("origin",))

    assert [member.stat_var for member in group.members] == ["Service_ByCreditor"]
    assert group.undismissed_candidates == ()


def test_a_by_creditor_twin_settled_out_leaves_the_total_answering_an_open_creditor():
    """The batch reads the twin by-creditor, then its trusted re-read shows another flow
    type and it leaves the members. No member left declares the open origin, so the total
    answers under the destination alone, where the creditor key would keep the
    subject-only sibling and tie the two role sets."""
    arcs = {
        "Service": _arcs(_RECIPIENT),
        "Service_Subject": _arcs(()),
        "Service_ByCreditor": _arcs(_PAIR),
    }
    moved = {**arcs["Service_ByCreditor"], "flowType": ("Disbursement",)}
    reader = _SequencedReader(
        arcs=arcs, reads={"Service_ByCreditor": [arcs["Service_ByCreditor"], moved]}
    )

    group = _walk("destination", reader=reader, open_role_keys=("origin",))

    assert [member.stat_var for member in group.members] == ["Service"]


def test_many_subject_only_siblings_cost_one_more_read_and_refuse_nothing():
    """A five-tuple with 95 subject-only members beside a recipient-only one: with the
    recipient bound, the subject-only members are re-read in one batch and dropped. The
    total's constraint keys dominate theirs, so none is read alone."""
    arcs = {f"Subject_{position}": _arcs((), guarantee=f"G{position}") for position in range(95)}
    arcs["Service"] = _arcs(_RECIPIENT)
    reader = FakeMemberArcReader(arcs=arcs)
    alone = FakeMemberArcReader(arcs={"Service": arcs["Service"]})

    group = _walk("destination", reader=reader)
    _walk("destination", reader=alone)

    assert [member.stat_var for member in group.members] == ["Service"]
    assert group.undismissed_candidates == ()
    assert group.inconsistent_candidates == ()
    assert reader.calls["node_arcs_many"] == alone.calls["node_arcs_many"] + 1


# ---------------------------------------------------------------------------
# Completeness of the elected role class
# ---------------------------------------------------------------------------


def _window_family(recipient_root: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """3,999 guaranteed by-creditor members, the by-creditor root, and a recipient-only root
    named `recipient_root`: one candidate more than `CANDIDATE_CAP`. The reader pages in
    dcid order, so the last of them falls past the window."""
    arcs = {
        f"Service_A{position:04d}": _arcs(_PAIR, guarantee=f"G{position}")
        for position in range(CANDIDATE_CAP - 1)
    }
    arcs["Service_ByCreditor"] = _arcs(_PAIR)
    arcs[recipient_root] = _arcs(_RECIPIENT)
    return arcs


def test_a_by_creditor_root_does_not_prove_a_truncated_page_a_recipient_only_root_would_replace():
    group = _walk(
        "destination", reader=FakeMemberArcReader(arcs=_window_family("Service_Recipient"))
    )

    assert group.truncated
    assert group.candidates_considered == CANDIDATE_CAP
    assert [member.stat_var for member in group.members] == ["Service_ByCreditor"]
    assert group.root_confirmed
    assert not group.proven_complete


def test_a_recipient_only_root_proves_a_truncated_page_past_which_a_by_creditor_root_sits():
    group = _walk(
        "destination", reader=FakeMemberArcReader(arcs=_window_family("Service_A_Recipient"))
    )

    assert group.truncated
    assert [member.stat_var for member in group.members] == ["Service_A_Recipient"]
    assert group.proven_complete


def test_role_arcs_two_batch_reads_drop_are_read_alone_before_the_election():
    """Both recipient-only members read subject-only on the first batch and on its re-read.
    Read alone, the total declares its recipient and replaces the by-creditor root."""
    dropped = dict.fromkeys(("Service", "Service_Guaranteed"), "observationProperties")
    reader = FlakyMemberArcReader(
        arcs=_DEBT_ARCS, drop_first_batch=dropped, drop_second_batch=dropped
    )

    group = _walk("destination", reader=reader)

    assert [member.stat_var for member in group.members] == ["Service"]
    assert group.members[0].declared_roles == _RECIPIENT
    assert group.undismissed_candidates == ()


def test_a_subject_only_winner_whose_role_arc_two_batch_reads_drop_is_read_alone():
    """With no role key the subject-only member answers. A recipient-only member that reads
    subject-only twice would join it; read alone, it leaves the class."""
    arcs = {
        "Service": _arcs(()),
        "Service_Recipient": _arcs(_RECIPIENT),
        "Service_ByCreditor": _arcs(_PAIR),
    }
    dropped = {"Service_Recipient": "observationProperties"}
    reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch=dropped, drop_second_batch=dropped)

    group = _walk(reader=reader)

    assert [member.stat_var for member in group.members] == ["Service"]
    assert group.undismissed_candidates == ()


def _flow_arcs(roles: tuple[str, ...], *, flow: str) -> dict[str, tuple[str, ...]]:
    return {**_arcs(roles), "flowType": (flow,)}


def test_role_suspects_past_the_budget_stay_undismissed():
    """Seventeen recipient-only members of distinct flow types read subject-only on both
    batch reads, beside a by-creditor member. The budget reads sixteen alone, and the walk
    names the seventeenth rather than answer without it."""
    suspects = {
        f"Service_R{position:02d}": _flow_arcs(_RECIPIENT, flow=f"F{position:02d}")
        for position in range(_ROLE_PROBE_BUDGET + 1)
    }
    arcs = {**suspects, "Service_Line": _flow_arcs(_PAIR, flow="F00")}
    dropped = dict.fromkeys(suspects, "observationProperties")
    reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch=dropped, drop_second_batch=dropped)

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots={"flowType": ("unbound",), "guarantee": ("unbound",)},
        graph=reader,
        role_keys=("destination",),
    )

    assert [member.stat_var for member in group.members] == sorted(suspects)[:-1]
    assert group.undismissed_candidates == (sorted(suspects)[-1],)
    assert group.inconsistent_candidates == group.undismissed_candidates


def test_a_root_outside_the_covering_class_dismisses_no_unread_candidate():
    """With no role key, a subject-only member covers and the by-creditor root does not.
    Every read of the subject-only member drops the `statType` the reverse query filtered
    on, so it stays unread; read, it would answer alone."""
    arcs = {
        "Service_ByCreditor": _arcs(_PAIR),
        "Service_Subject_Guaranteed": _arcs((), guarantee="Public"),
    }
    reader = FlakyMemberArcReader(arcs=arcs, drop_always={"Service_Subject_Guaranteed": "statType"})

    group = _walk(reader=reader)

    assert not group.root_confirmed
    assert group.undismissed_candidates == ("Service_Subject_Guaranteed",)


# ---------------------------------------------------------------------------
# IDS role twins on the graph walk
# ---------------------------------------------------------------------------

_PROVIDER_ROLE = "financingProvider"
_RECIPIENT_ROLE = "financingRecipient"
_TOTAL = "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_Nominal"
_BY_CREDITOR = (
    "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
)
_LENDER = "Amount_FinancialPosition_AmountOutstanding_DebtInstrument_IDSNonConcessional_Nominal"
_PPG_CONSTRAINTS = {
    "financialPositionType": "AmountOutstanding",
    "guaranteeStatus": "PublicOrPubliclyGuaranteed",
}
# The lender-named variable's constraints as the dev graph serves them.
_LENDER_CONSTRAINTS = {
    "financialPositionType": "AmountOutstanding",
    "financingResource": "DebtInstrument",
    "idsConcessionality": "IDSNonConcessional",
}
_IDS_FIVE_TUPLE = FiveTuple(
    population_type="FinancialPosition",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_UNBOUND_SLOTS = dict.fromkeys({**_PPG_CONSTRAINTS, **_LENDER_CONSTRAINTS}, ("unbound",))
_VALUE_SLOTS = {name: ("value", value) for name, value in _PPG_CONSTRAINTS.items()}


def _ids_arcs(constraints: dict[str, str], roles: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("FinancialPosition",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "measurementDenominator": (),
        "constraintProperties": tuple(constraints),
        "observationProperties": roles,
        **{name: (value,) for name, value in constraints.items()},
    }


def _twin_reader(*, lender: bool = False, force_truncated: bool = False) -> FakeMemberArcReader:
    arcs = {
        _TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,)),
        _BY_CREDITOR: _ids_arcs(_PPG_CONSTRAINTS, (_PROVIDER_ROLE, _RECIPIENT_ROLE)),
    }
    if lender:
        arcs[_LENDER] = _ids_arcs(_LENDER_CONSTRAINTS, (_PROVIDER_ROLE, _RECIPIENT_ROLE))
    return FakeMemberArcReader(arcs=arcs, force_truncated=force_truncated)


def _ids_members(
    *role_keys: str, lender: bool = False, open_role_keys: tuple[str, ...] = ()
) -> tuple[str, ...]:
    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_UNBOUND_SLOTS,
        graph=_twin_reader(lender=lender),
        role_keys=role_keys,
        open_role_keys=open_role_keys,
    )
    return tuple(member.stat_var for member in group.members)


def test_the_walk_resolves_unbound_twins_to_the_total():
    assert _ids_members() == (_TOTAL,)
    assert _ids_members("destination") == (_TOTAL,)


def test_the_walk_resolves_a_bound_creditor_to_the_breakdown():
    assert _ids_members("origin", "destination") == (_BY_CREDITOR,)


def test_the_walk_requires_bound_roles_before_the_antichain():
    """X has fewer constraints but lacks the bound provider role. The antichain would
    drop Y for X and the role requirement would then drop X, leaving nothing."""
    x = _ids_arcs({"financialPositionType": "AmountOutstanding"}, (_RECIPIENT_ROLE,))
    y = _ids_arcs(
        {"financialPositionType": "AmountOutstanding", "guaranteeStatus": "PPG"},
        (_PROVIDER_ROLE, _RECIPIENT_ROLE),
    )

    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots={"financialPositionType": ("unbound",), "guaranteeStatus": ("unbound",)},
        graph=FakeMemberArcReader(arcs={"X": x, "Y": y}),
        role_keys=("origin",),
    )

    assert [member.stat_var for member in group.members] == ["Y"]


def test_the_walk_keeps_every_candidate_when_none_declares_the_bound_role():
    """The group stays non-empty so the caller's own role check refuses, naming the role."""
    assert _ids_members("observationAbout") == (_TOTAL,)


@pytest.mark.parametrize("role_keys", [(), ("destination",)])
def test_the_total_ties_with_a_lender_named_variable_of_other_keys(role_keys):
    """A lender-named two-role variable has its own constraint keys and no twin, so the
    collapse leaves it beside the total. The total breaks down by a property the variable
    does not hold, so the walk reads it as the total of no creditor series and refuses."""
    with pytest.raises(MemberRoleDisagreement) as exc_info:
        _ids_members(*role_keys, lender=True)

    assert set(exc_info.value.roles_by_stat_var) == {_TOTAL, _LENDER}


def test_the_total_answers_beside_a_lender_named_variable_once_its_slots_are_bound():
    """Binding the total's own constraint values leaves the lender-named variable out of
    the candidates, and the total answers."""
    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_VALUE_SLOTS,
        graph=_twin_reader(lender=True),
        role_keys=("destination",),
    )

    assert [member.stat_var for member in group.members] == [_TOTAL]


def test_an_open_creditor_reads_the_breakdown_and_the_lender_named_variable():
    """The by-creditor twin of the total places the open creditor, so the election runs
    with the creditor and the lender-named variable answers beside the twin."""
    members = _ids_members("destination", lender=True, open_role_keys=("origin",))

    assert set(members) == {_BY_CREDITOR, _LENDER}


@pytest.mark.parametrize(
    ("lender_constraints", "members", "elected_open_role_keys"),
    [
        pytest.param(
            {**_PPG_CONSTRAINTS, "guaranteeStatus": "PrivateNonGuaranteed"},
            [_TOTAL],
            frozenset(),
            id="the total's keys under another value",
        ),
        pytest.param(_LENDER_CONSTRAINTS, [_LENDER], {"origin"}, id="other keys"),
    ],
)
def test_an_open_creditor_only_a_lender_named_variable_places(
    lender_constraints, members, elected_open_role_keys
):
    """With no by-creditor twin of the total, a lender-named variable holding the total's
    constraint keys leaves for the total, no member the recipient elects holds a creditor
    series, and the walk reports that it read no open key. One holding other keys ties
    with the total under the recipient alone, so the open creditor joins and it answers."""
    arcs = {
        _TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,)),
        _LENDER: _ids_arcs(lender_constraints, (_PROVIDER_ROLE, _RECIPIENT_ROLE)),
    }

    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_UNBOUND_SLOTS,
        graph=FakeMemberArcReader(arcs=arcs),
        role_keys=("destination",),
        open_role_keys=("origin",),
    )

    assert [member.stat_var for member in group.members] == members
    assert group.family == tuple(members)
    assert group.elected_open_role_keys == elected_open_role_keys


def test_a_lender_named_variable_joins_the_breakdown_when_the_creditor_is_bound():
    assert set(_ids_members("origin", "destination", lender=True)) == {_BY_CREDITOR, _LENDER}


def test_a_truncated_page_with_a_displaceable_root_is_not_proven_complete():
    """A total past the window could displace the by-creditor root."""
    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_VALUE_SLOTS,
        graph=FakeMemberArcReader(
            arcs={_BY_CREDITOR: _ids_arcs(_PPG_CONSTRAINTS, (_PROVIDER_ROLE, _RECIPIENT_ROLE))},
            force_truncated=True,
        ),
        role_keys=("destination",),
    )

    assert [member.stat_var for member in group.members] == [_BY_CREDITOR]
    assert group.proven_complete is False


def test_a_truncated_page_with_the_total_as_root_is_proven_complete():
    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_VALUE_SLOTS,
        graph=_twin_reader(force_truncated=True),
        role_keys=("destination",),
    )

    assert [member.stat_var for member in group.members] == [_TOTAL]
    assert group.proven_complete is True


def test_a_truncated_page_with_both_roles_bound_lets_the_by_creditor_root_prove_complete():
    """With both roles bound, no twin covering them declares fewer roles."""
    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_VALUE_SLOTS,
        graph=FakeMemberArcReader(
            arcs={_BY_CREDITOR: _ids_arcs(_PPG_CONSTRAINTS, (_PROVIDER_ROLE, _RECIPIENT_ROLE))},
            force_truncated=True,
        ),
        role_keys=("origin", "destination"),
    )

    assert group.proven_complete is True


def _truncated_walk(arcs: dict[str, dict[str, tuple[str, ...]]]):
    return enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_VALUE_SLOTS,
        graph=FakeMemberArcReader(arcs=arcs, force_truncated=True),
        role_keys=(),
    )


def test_one_role_root_no_keys_unseen_subject_only_member_changes_the_answer():
    """With no role key bound, a subject-only member of the root's constraints takes the
    role class, so a one-role root on a truncated page proves nothing."""
    full = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_VALUE_SLOTS,
        graph=FakeMemberArcReader(
            arcs={
                _TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,)),
                "Subject": _ids_arcs(_PPG_CONSTRAINTS, ()),
            }
        ),
        role_keys=(),
    )
    seen = _truncated_walk({_TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,))})

    assert [member.stat_var for member in full.members] == ["Subject"]
    assert [member.stat_var for member in seen.members] == [_TOTAL]
    assert seen.proven_complete is False


def test_one_role_root_no_keys_unseen_provider_only_twin_ties():
    """A provider-only member of the root's constraints ties with a recipient-only root
    when no role key is bound, so the root on a truncated page proves nothing."""
    with pytest.raises(MemberRoleDisagreement):
        enumerate_members(
            five_tuple=_IDS_FIVE_TUPLE,
            slots=_VALUE_SLOTS,
            graph=FakeMemberArcReader(
                arcs={
                    _TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,)),
                    "Line": _ids_arcs(_PPG_CONSTRAINTS, (_PROVIDER_ROLE,)),
                }
            ),
            role_keys=(),
        )

    seen = _truncated_walk({_TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,))})

    assert seen.proven_complete is False


def test_a_subject_only_root_with_no_keys_proves_a_truncated_family_complete():
    seen = _truncated_walk({"Subject": _ids_arcs(_PPG_CONSTRAINTS, ())})

    assert [member.stat_var for member in seen.members] == ["Subject"]
    assert seen.proven_complete is True


# ---------------------------------------------------------------------------
# Completeness of the open keys the election reads
# ---------------------------------------------------------------------------

_NARROW = {**_PPG_CONSTRAINTS, "idsConcessionality": "IDSNonConcessional"}
_NARROW_SLOTS = {**_VALUE_SLOTS, "idsConcessionality": ("unbound",)}


def _open_destination_walk(arcs, *, force_truncated: bool = False, reader=None):
    return enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=_NARROW_SLOTS,
        graph=reader or FakeMemberArcReader(arcs=arcs, force_truncated=force_truncated),
        role_keys=(),
        open_role_keys=("destination",),
    )


def test_a_role_bearing_root_proves_no_open_key_a_subject_only_member_would_set_aside():
    """With no key bound, a subject-only member answers and an open recipient joins only
    where its twin places it. A narrower subject-only member past the window has no such
    twin and would answer alone, so the recipient-only root on a truncated page proves
    nothing."""
    total = {_TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,))}
    full = _open_destination_walk({**total, "Subject_Narrow": _ids_arcs(_NARROW, ())})
    seen = _open_destination_walk(total, force_truncated=True)

    assert [member.stat_var for member in full.members] == ["Subject_Narrow"]
    assert full.elected_open_role_keys == frozenset()
    assert [member.stat_var for member in seen.members] == [_TOTAL]
    assert seen.elected_open_role_keys == {"destination"}
    assert not seen.root_confirmed
    assert not seen.proven_complete


def test_a_subject_only_root_beside_its_twin_proves_the_open_key_on_a_truncated_page():
    """The subject-only root dominates every subject-only member past the window, so the
    election with no key bound reads it whatever sits there, and its recipient-only twin
    places the open recipient."""
    seen = _open_destination_walk(
        {
            "Subject": _ids_arcs(_PPG_CONSTRAINTS, ()),
            _TOTAL: _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,)),
        },
        force_truncated=True,
    )

    assert [member.stat_var for member in seen.members] == [_TOTAL]
    assert seen.elected_open_role_keys == {"destination"}
    assert seen.proven_complete


def test_a_role_arc_dropped_from_the_member_that_admits_an_open_key_is_read_alone():
    """`Narrow` declares a recipient, and both batch reads drop the arc. Read subject-only,
    it answers the unbound question, its twin places the open recipient, and the broader
    total answers. Read alone it declares its recipient, the subject-only member left has
    no twin, and that member answers with the open key set aside."""
    slots = {
        "financialPositionType": ("value", "AmountOutstanding"),
        "guaranteeStatus": ("unbound",),
        "idsConcessionality": ("unbound",),
    }
    arcs = {
        _TOTAL: _ids_arcs({"financialPositionType": "AmountOutstanding"}, (_RECIPIENT_ROLE,)),
        "Narrow": _ids_arcs(_PPG_CONSTRAINTS, (_RECIPIENT_ROLE,)),
        "Narrow_Twin": _ids_arcs(_PPG_CONSTRAINTS, (_PROVIDER_ROLE, _RECIPIENT_ROLE)),
        "Subject_Narrowest": _ids_arcs(_NARROW, ()),
    }
    dropped = {"Narrow": "observationProperties"}
    reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch=dropped, drop_second_batch=dropped)

    group = enumerate_members(
        five_tuple=_IDS_FIVE_TUPLE,
        slots=slots,
        graph=reader,
        role_keys=(),
        open_role_keys=("destination",),
    )

    assert [member.stat_var for member in group.members] == ["Subject_Narrowest"]
    assert group.elected_open_role_keys == frozenset()
    assert group.undismissed_candidates == ()
