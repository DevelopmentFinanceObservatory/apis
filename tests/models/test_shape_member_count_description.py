"""Checks that `Shape.member_count` describes the resolved population size."""

from __future__ import annotations

from search.models import Shape


def test_member_count_description_reads_as_a_population_not_a_family_tally():
    description = Shape.model_fields["member_count"].description
    assert description is not None
    assert "population" in description
    assert "fell into this family" not in description
