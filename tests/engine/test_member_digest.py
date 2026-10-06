"""What Search states of a walked family is what the Datasets mint states of the same walk:
the walk's antichain-minimal members as the served count, and the SHA-256 hash of the sorted,
RFC 8785 canonical dcids of its whole family as the digest, none when the walk left
candidates unread. End-to-end agreement with the mint is covered by
tests/datasets_api/test_member_digest_cross_api.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import rfc8785

from code_index.members import FamilyQuery
from search.engine.retrieval.bound_values import (
    StatedMembers,
    WalkedMembers,
    identity_query,
    stated_members,
    walk_key,
)
from search.engine.spec_id import mint_identity
from statvar_members import FiveTuple, Member, MemberGroup
from tests.code_index.aid_identity import education_aid_identity

_FIVE_TUPLE = FiveTuple(population_type="Person", measured_property="count")
_QUERY = FamilyQuery(_FIVE_TUPLE, slots={"gender": ("unbound",)}, source=None)


def _group(family: tuple[str, ...], *, truncated: bool = False) -> MemberGroup:
    """A walked group whose first family member is its one antichain-minimal member."""
    return MemberGroup(
        five_tuple=_FIVE_TUPLE,
        members=(Member(stat_var=family[0], constraints={}, declared_roles=("observationAbout",)),),
        family=family,
        candidates_considered=len(family),
        candidate_cap=4_000,
        truncated=truncated,
        proven_complete=True,
        root_confirmed=True,
    )


def _expected_digest(dcids: list[str]) -> str:
    return hashlib.sha256(rfc8785.dumps(sorted(dcids))).hexdigest()


def test_a_complete_walk_states_its_served_members_and_hashes_its_sorted_family():
    walked = WalkedMembers.of(_QUERY, _group(("Count_Person", "Count_Person_Female")))

    assert walked.key == walk_key(_QUERY)
    assert walked.stated == StatedMembers(
        served_count=1, digest=_expected_digest(["Count_Person", "Count_Person_Female"])
    )


def test_the_digest_is_independent_of_confirmation_order():
    forward = WalkedMembers.of(_QUERY, _group(("Count_Person", "Count_Person_Female")))
    backward = WalkedMembers.of(_QUERY, _group(("Count_Person_Female", "Count_Person")))

    assert forward.stated.digest == backward.stated.digest


@pytest.mark.parametrize("truncated", [True, False])
def test_a_walk_that_left_candidates_unread_states_no_digest(truncated: bool):
    """A truncated page proven complete still serves its members, so the count stands; the
    family past its window is unread, so neither surface digests it."""
    group = _group(("Count_Person",), truncated=truncated)

    walked = WalkedMembers.of(_QUERY, group)

    assert group.stated_family == (None if truncated else frozenset({"Count_Person"}))
    assert walked.stated == StatedMembers(
        served_count=1, digest=None if truncated else _expected_digest(["Count_Person"])
    )


def test_a_walk_asked_an_open_role_its_members_do_not_serve_states_the_answers_members():
    """A part asks every origin of a family the index does not resolve. Its walk elects
    under the bound keys alone, and the identity minted from the answer names no open role,
    so that identity's own query finds the walk. A walk that elected under the open key
    resolved another identity's members and states nothing for this one."""
    document = education_aid_identity()
    document["shape"]["population_type"] = "EconomicActivity"
    _, identity = mint_identity(document)
    asked = replace(identity_query(identity), open_role_keys=frozenset({"origin"}))
    dropped = _group(("Amount_EconomicActivity",))
    served = replace(dropped, elected_open_role_keys=frozenset({"origin"}))

    assert stated_members(identity, WalkedMembers.of(asked, dropped)) == StatedMembers(
        served_count=1, digest=_expected_digest(["Amount_EconomicActivity"])
    )
    assert WalkedMembers.of(asked, served).key == walk_key(asked)
    assert stated_members(identity, WalkedMembers.of(asked, served)) is None
