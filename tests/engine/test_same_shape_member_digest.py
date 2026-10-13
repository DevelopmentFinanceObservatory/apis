"""Tests merged `member_count` over admitted populations, and the mint's served count and
family digest a merged Spec states.

Fixtures distinguish admitted members from StatVars confirmed at probe entities. No graph or LLM
calls are made. The fixture members sit outside the code index, so a merged Spec states what its
parts' walk settled (`same_shape._merged_walked`) when the merged identity is the one that walk
resolves, and nothing otherwise.
"""

from __future__ import annotations

import dataclasses

import pytest

from search.engine.retrieval.retrieve import MemberPopulation
from search.engine.same_shape import collapse_same_shape
from search.models import Spec
from statvar_members import member_digest
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._conjoin_fixtures import _PERSON_SHAPE, _region
from tests.engine.test_same_shape import (
    _roster_variant_spec,
    _where_bound_spec_with_member_count,
)


def _population(dcids: list[str], *, exact: bool = True) -> MemberPopulation:
    """Build a `MemberPopulation` with the supplied admitted dcids and exactness."""
    return MemberPopulation(
        admitted=len(dcids), admitted_dcids=tuple(dcids), confirmed=len(dcids), exact=exact
    )


def _with_population(region, population: MemberPopulation | None):
    """Attach the admitted population to a fixture `RegionResult`."""
    return dataclasses.replace(region, member_population=population)


def _with_shape_member_count(spec: Spec, count: int) -> Spec:
    """Set a fixture shape count independently of its population, without changing `spec_id`."""
    return spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"shape": spec.resolved.shape.model_copy(update={"member_count": count})}
            )
        }
    )


def _stating(spec: Spec, served_count: int, digest: str) -> Spec:
    """`spec` stating `served_count` and `digest`, as a part whose walk settled them does."""
    resolved = spec.resolved
    return spec.model_copy(
        update={
            "resolved": resolved.model_copy(
                update={
                    "stat_vars": resolved.stat_vars.model_copy(
                        update={"served_count": served_count}
                    ),
                    "shape": resolved.shape.model_copy(update={"member_digest": digest}),
                }
            )
        }
    )


def test_collapse_same_shape_single_slot_merge_counts_the_admitted_union_and_states_no_walk():
    """The count covers five admitted dcids; only two StatVars are confirmed. The parts'
    walks settled their own values, not the merged set, so the merge states neither."""
    digest = member_digest(["sv_a"])
    spec_a = _stating(_spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a"), 1, digest)
    spec_b = _stating(_spec(_PERSON_SHAPE, "ValB", sv_dcid="sv_b"), 1, digest)
    r_a = _with_population(
        _region(spec_a, "var A", earliest_index=0), _population(["sv_a", "sv_x", "sv_y"])
    )
    r_b = _with_population(
        _region(spec_b, "var B", earliest_index=1), _population(["sv_b", "sv_z"])
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec
    assert {sv.ref.ref for sv in merged.resolved.members} == {"sv_a", "sv_b"}
    assert merged.resolved.shape.member_count == 5
    assert merged.resolved.shape.member_digest is None
    assert merged.resolved.stat_vars.served_count is None


def test_collapse_same_shape_roster_merge_counts_the_admitted_union():
    """A zero-diff roster merge (diff_key None): the two members' own
    admitted populations only partly overlap (sv_a shared, sv_x/sv_y each
    member's own) -- the merged shape publishes the union's own size, not
    either member's alone."""
    spec_a = _roster_variant_spec("country/KEN", "sv_a", member_count=1, carry_open_what_slot=False)
    spec_b = _roster_variant_spec("country/KEN", "sv_a", member_count=1, carry_open_what_slot=True)
    r_a = _with_population(
        _region(spec_a, "population of Kenya", earliest_index=0), _population(["sv_a", "sv_x"])
    )
    r_b = _with_population(
        _region(spec_b, "population of Kenya", earliest_index=1), _population(["sv_a", "sv_y"])
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    assert effective[0].spec.resolved.shape.member_count == 3


def test_collapse_same_shape_no_member_digest_when_a_member_is_inexact():
    """An inexact admitted population withholds the digest and keeps the maximum member count."""
    spec_a = _with_shape_member_count(
        _roster_variant_spec("country/KEN", "sv_a", member_count=1, carry_open_what_slot=False), 3
    )
    spec_b = _with_shape_member_count(
        _roster_variant_spec("country/KEN", "sv_a", member_count=1, carry_open_what_slot=True), 7
    )
    r_a = _with_population(
        _region(spec_a, "population of Kenya", earliest_index=0), _population(["sv_a", "sv_x"])
    )
    r_b = _with_population(
        _region(spec_b, "population of Kenya", earliest_index=1),
        _population(["sv_a", "sv_y"], exact=False),
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged_shape = effective[0].spec.resolved.shape
    assert merged_shape.member_digest is None
    assert merged_shape.member_count == 7


def test_collapse_same_shape_no_member_digest_when_a_member_has_no_population():
    """An unknown admitted population withholds the digest and keeps the maximum count."""
    spec_a = _with_shape_member_count(
        _roster_variant_spec("country/KEN", "sv_a", member_count=1, carry_open_what_slot=False), 2
    )
    spec_b = _with_shape_member_count(
        _roster_variant_spec("country/KEN", "sv_a", member_count=1, carry_open_what_slot=True), 9
    )
    r_a = _with_population(
        _region(spec_a, "population of Kenya", earliest_index=0), _population(["sv_a"])
    )
    r_b = _region(spec_b, "population of Kenya", earliest_index=1)
    assert r_b.member_population is None

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged_shape = effective[0].spec.resolved.shape
    assert merged_shape.member_digest is None
    assert merged_shape.member_count == 9


@pytest.mark.parametrize(
    ("uganda", "stated"),
    [
        pytest.param((2, "d"), (2, "d"), id="parts-agree"),
        pytest.param((3, "d"), (None, None), id="served-counts-differ"),
        pytest.param((2, "e"), (None, None), id="digests-differ"),
    ],
)
def test_collapse_same_shape_where_axis_merge_keeps_the_walk_its_parts_share(uganda, stated):
    """A where-axis merge binds one identity's members at more places, so the walk both
    parts settled stands for it; parts that disagree state nothing. The member count is the
    maximum."""
    spec_kenya = _stating(
        _with_shape_member_count(
            _where_bound_spec_with_member_count("country/KEN", "sv_kenya", member_count=5), 5
        ),
        2,
        "d",
    )
    spec_uganda = _stating(
        _with_shape_member_count(
            _where_bound_spec_with_member_count("country/UGA", "sv_uganda", member_count=7), 7
        ),
        *uganda,
    )
    r_a = _with_population(
        _region(spec_kenya, "population of Kenya", earliest_index=0), _population(["sv_kenya"])
    )
    r_b = _with_population(
        _region(spec_uganda, "population of Uganda", earliest_index=1), _population(["sv_uganda"])
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec.resolved
    assert (merged.stat_vars.served_count, merged.shape.member_digest) == stated
    assert merged.shape.member_count == 7


def test_collapse_same_shape_recomputed_member_count_does_not_move_spec_id():
    """Different merged counts leave `spec_id` unchanged."""

    def _merge(second_admitted: list[str]):
        spec_a = _roster_variant_spec(
            "country/KEN", "sv_a", member_count=1, carry_open_what_slot=False
        )
        spec_b = _roster_variant_spec(
            "country/KEN", "sv_a", member_count=1, carry_open_what_slot=True
        )
        r_a = _with_population(
            _region(spec_a, "population of Kenya", earliest_index=0), _population(["sv_a"])
        )
        r_b = _with_population(
            _region(spec_b, "population of Kenya", earliest_index=1), _population(second_admitted)
        )
        effective, _residual = collapse_same_shape([r_a, r_b])
        return effective[0].spec

    distinct = _merge(["sv_a", "sv_x"])
    shared = _merge(["sv_a"])

    assert distinct.resolved.shape.member_count != shared.resolved.shape.member_count
    assert distinct.spec_id == shared.spec_id
