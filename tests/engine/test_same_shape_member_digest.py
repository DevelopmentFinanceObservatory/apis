"""Tests merged `member_count` and `member_digest` over admitted populations.

Fixtures distinguish admitted members from StatVars confirmed at probe entities. No graph or LLM
calls are made. The fixture members sit outside the code index, where `build_spec` states no
digest, so the tests that read a merged digest stand the merged population in for the family
the index resolves (`_index_resolves`).
"""

from __future__ import annotations

import dataclasses

import pytest

from search.engine import spec_build
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


@pytest.fixture
def _index_resolves(monkeypatch):
    """Returns a function that makes the code index resolve `dcids` for every identity."""

    def resolve(dcids: list[str]) -> None:
        monkeypatch.setattr(
            spec_build, "identity_family_digest", lambda _identity: member_digest(dcids)
        )

    return resolve


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


def test_collapse_same_shape_single_slot_merge_publishes_member_digest_over_the_admitted_union(
    _index_resolves,
):
    """The digest covers five admitted dcids; only two StatVars are confirmed."""
    _index_resolves(["sv_a", "sv_x", "sv_y", "sv_b", "sv_z"])
    spec_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a")
    spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv_b")
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
    assert {sv.ref for sv in merged.resolved.stat_vars.items} == {"sv_a", "sv_b"}
    assert merged.resolved.shape.member_count == 5
    assert merged.resolved.shape.member_digest == member_digest(
        ["sv_a", "sv_x", "sv_y", "sv_b", "sv_z"]
    )


@pytest.mark.parametrize("index_resolves_the_union", [True, False])
def test_collapse_same_shape_roster_merge_publishes_member_digest_over_the_admitted_union(
    _index_resolves, index_resolves_the_union
):
    """A zero-diff roster merge (diff_key None): the two members' own
    admitted populations only partly overlap (sv_a shared, sv_x/sv_y each
    member's own) -- the merged shape publishes the union's own size/digest,
    not either member's alone. The digest is stated only when the union is the
    family the index resolves for the merged identity."""
    _index_resolves(["sv_a", "sv_x", "sv_y"] if index_resolves_the_union else ["sv_a"])
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
    merged_shape = effective[0].spec.resolved.shape
    assert merged_shape.member_count == 3
    assert merged_shape.member_digest == (
        member_digest(["sv_a", "sv_x", "sv_y"]) if index_resolves_the_union else None
    )


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


def test_collapse_same_shape_where_axis_merge_never_gets_member_digest():
    """Where-axis merges withhold the digest and use the maximum member count."""
    spec_kenya = _with_shape_member_count(
        _where_bound_spec_with_member_count("country/KEN", "sv_kenya", member_count=5), 5
    )
    spec_uganda = _with_shape_member_count(
        _where_bound_spec_with_member_count("country/UGA", "sv_uganda", member_count=7), 7
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
    merged_shape = effective[0].spec.resolved.shape
    assert merged_shape.member_digest is None
    assert merged_shape.member_count == 7


def test_collapse_same_shape_recomputed_member_digest_does_not_move_spec_id(_index_resolves):
    """Different merged counts and digests leave `spec_id` unchanged."""

    def _merge(second_admitted: list[str]):
        _index_resolves(second_admitted)
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
    assert distinct.resolved.shape.member_digest != shared.resolved.shape.member_digest
    assert distinct.spec_id == shared.spec_id
