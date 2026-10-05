"""`match_dcids` against hand-built `SkillsIndex` fixtures: ontology matches
before provenance matches, each of the three ontology ordering tiers deciding in turn, a tie broken by skill id, the
minimum hop taken across more than one matching node, and no match
returning an empty list.
"""

from __future__ import annotations

import pytest

from skills_index.match import match_dcids
from skills_index.types import SkillMatch, SkillsIndex


def _index(node_to_skills: dict[str, tuple[SkillMatch, ...]]) -> SkillsIndex:
    return SkillsIndex(
        skills={}, node_to_skills=node_to_skills, issues=(), may_omit_published_skills=False
    )


def _match(skill_id: str, *, hop: int = 0, breadth: int = 10, provenance: bool = False):
    return SkillMatch(
        skill_id=skill_id,
        hop=hop,
        breadth=breadth,
        breadth_truncated=False,
        via_provenance=provenance,
    )


@pytest.mark.parametrize(
    ("node_to_skills", "expected"),
    [
        pytest.param(
            {"place/1": (_match("far", hop=3), _match("near", hop=0))},
            ["near", "far"],
            id="hop-ascending",
        ),
        pytest.param(
            {
                "provenance/CRS": (_match("dataset", breadth=0, provenance=True),),
                "crsPurpose/930": (_match("concept", hop=4, breadth=4000),),
            },
            ["concept", "dataset"],
            id="provenance-after-a-deeper-broader-ontology-match",
        ),
        pytest.param(
            {
                "provenance/B": (_match("zeta", breadth=0, provenance=True),),
                "provenance/A": (_match("alpha", breadth=9000, provenance=True),),
            },
            ["alpha", "zeta"],
            id="provenance-matches-order-by-id-not-breadth",
        ),
        pytest.param(
            {
                "provenance/CRS": (
                    _match("zeta-both", provenance=True),
                    _match("alpha-dataset", provenance=True),
                ),
                "crsPurpose/930": (_match("zeta-both", hop=4),),
            },
            ["zeta-both", "alpha-dataset"],
            id="an-ontology-match-outranks-the-same-skills-provenance-match",
        ),
    ],
)
def test_orders_ontology_matches_by_hop_then_provenance_matches_by_id(node_to_skills, expected):
    index = _index(node_to_skills)

    assert match_dcids(list(node_to_skills), index=index) == expected


def test_orders_by_breadth_ascending_when_hop_ties():
    index = _index(
        {
            "place/1": (
                SkillMatch(skill_id="broad", hop=1, breadth=4000, breadth_truncated=False),
                SkillMatch(skill_id="narrow", hop=1, breadth=10, breadth_truncated=False),
            )
        }
    )

    assert match_dcids(["place/1"], index=index) == ["narrow", "broad"]


def test_truncated_breadth_never_outranks_an_untruncated_one_at_equal_hop():
    # A truncated count of 200 is a floor, not the skill's real total --
    # it must never look narrower than a genuinely narrow skill's exact
    # count of 300, even though 200 < 300.
    index = _index(
        {
            "place/1": (
                SkillMatch(skill_id="truncated", hop=1, breadth=200, breadth_truncated=True),
                SkillMatch(skill_id="exact", hop=1, breadth=300, breadth_truncated=False),
            )
        }
    )

    assert match_dcids(["place/1"], index=index) == ["exact", "truncated"]


def test_orders_by_id_when_hop_and_breadth_tie():
    index = _index(
        {
            "place/1": (
                SkillMatch(skill_id="zeta", hop=1, breadth=10, breadth_truncated=False),
                SkillMatch(skill_id="alpha", hop=1, breadth=10, breadth_truncated=False),
            )
        }
    )

    assert match_dcids(["place/1"], index=index) == ["alpha", "zeta"]


def test_minimum_hop_used_across_multiple_matching_nodes():
    index = _index(
        {
            "place/1": (
                SkillMatch(skill_id="skill-a", hop=3, breadth=10, breadth_truncated=False),
            ),
            "place/2": (
                SkillMatch(skill_id="skill-a", hop=1, breadth=10, breadth_truncated=False),
            ),
        }
    )

    matches = match_dcids(["place/1", "place/2"], index=index)

    assert matches == ["skill-a"]


def test_minimum_hop_used_across_multiple_queried_dcids():
    index = _index(
        {
            "place/1": (
                SkillMatch(skill_id="skill-a", hop=2, breadth=10, breadth_truncated=False),
            ),
            "place/2": (
                SkillMatch(skill_id="skill-b", hop=0, breadth=10, breadth_truncated=False),
            ),
        }
    )

    matches = match_dcids(["place/1", "place/2"], index=index)

    assert matches == ["skill-b", "skill-a"]


def test_no_match_returns_empty_list():
    index = _index(
        {"place/1": (SkillMatch(skill_id="skill-a", hop=0, breadth=10, breadth_truncated=False),)}
    )

    assert match_dcids(["place/2"], index=index) == []


def test_every_matching_skill_is_returned_uncapped():
    index = _index(
        {
            "place/1": tuple(
                SkillMatch(skill_id=f"skill-{i}", hop=i, breadth=10, breadth_truncated=False)
                for i in range(10)
            )
        }
    )

    matches = match_dcids(["place/1"], index=index)

    assert len(matches) == 10
