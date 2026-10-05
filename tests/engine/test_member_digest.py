"""Only exact populations receive a member digest, the SHA-256 hash of sorted,
RFC 8785 canonical dcids. Tests also pin Shape wiring; end-to-end agreement with
`statvar_members.enumerate_members` is covered by test_crs_family_member_digest.py.
"""

from __future__ import annotations

import hashlib

import rfc8785

from search.engine.retrieval.retrieve import MemberPopulation
from search.engine.shape import ShapeDraft
from search.engine.spec_build import build_shape_model, compute_member_digest


def _shape_draft() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
    )


def _expected_digest(dcids: list[str]) -> str:
    return hashlib.sha256(rfc8785.dumps(sorted(dcids))).hexdigest()


def test_no_population_yields_no_digest():
    assert compute_member_digest(None) is None


def test_inexact_population_yields_no_digest():
    population = MemberPopulation(
        admitted=2, admitted_dcids=("sv_b", "sv_a"), confirmed=1, exact=False
    )

    assert compute_member_digest(population) is None


def test_exact_population_hashes_the_sorted_bare_dcids():
    population = MemberPopulation(
        admitted=2, admitted_dcids=("sv_b", "sv_a"), confirmed=2, exact=True
    )

    digest = compute_member_digest(population)

    assert digest == _expected_digest(["sv_a", "sv_b"])


def test_digest_is_independent_of_admitted_dcids_order():
    """Reordering the same member set leaves its sorted digest unchanged."""
    forward = MemberPopulation(
        admitted=3, admitted_dcids=("sv_a", "sv_b", "sv_c"), confirmed=3, exact=True
    )
    reversed_order = MemberPopulation(
        admitted=3, admitted_dcids=("sv_c", "sv_b", "sv_a"), confirmed=3, exact=True
    )

    assert compute_member_digest(forward) == compute_member_digest(reversed_order)


def test_digest_moves_with_membership():
    narrower = MemberPopulation(admitted=1, admitted_dcids=("sv_a",), confirmed=1, exact=True)
    broader = MemberPopulation(admitted=2, admitted_dcids=("sv_a", "sv_b"), confirmed=2, exact=True)

    assert compute_member_digest(narrower) != compute_member_digest(broader)


def test_build_shape_model_carries_the_caller_supplied_digest_onto_the_wire():
    shape_draft = _shape_draft()
    population = MemberPopulation(
        admitted=2, admitted_dcids=("sv_b", "sv_a"), confirmed=2, exact=True
    )

    shape = build_shape_model(
        shape_draft,
        slot_keys=[],
        five_tuple_refs={},
        member_count=population.admitted,
        members=population.admitted_dcids,
        member_digest=compute_member_digest(population),
    )

    assert shape.member_digest == _expected_digest(["sv_a", "sv_b"])


def test_build_shape_model_defaults_member_digest_to_none():
    """A caller without a population leaves the wire field null."""
    shape_draft = _shape_draft()

    shape = build_shape_model(
        shape_draft, slot_keys=[], five_tuple_refs={}, member_count=1, members=None
    )

    assert shape.member_digest is None
