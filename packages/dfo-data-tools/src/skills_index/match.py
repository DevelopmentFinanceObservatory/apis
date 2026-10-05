"""Match graph dcids against a built `SkillsIndex`."""

from __future__ import annotations

from collections.abc import Iterable

from skills_index.types import SkillMatch, SkillsIndex


def match_dcids(dcids: Iterable[str], *, index: SkillsIndex) -> list[str]:
    """Return all skills whose reach includes a queried dcid, most specific first.

    Skills matched through an ontology node come first, sorted by minimum hop,
    then exact before truncated breadth, ascending breadth, and skill id.
    Truncated breadths are lower bounds, so they rank after exact counts at equal
    hops. Skills matched only through a provenance attachment follow, sorted by
    skill id. A whole dataset is wider than any node inside it, whatever the hop.
    """
    best_key: dict[str, tuple[int, ...]] = {}
    for dcid in dcids:
        for match in index.node_to_skills.get(dcid, ()):
            key = _rank_key(match)
            current = best_key.get(match.skill_id)
            if current is None or key < current:
                best_key[match.skill_id] = key
    return sorted(best_key, key=lambda skill_id: (best_key[skill_id], skill_id))


def _rank_key(match: SkillMatch) -> tuple[int, ...]:
    """A match's rank, lower first. Every provenance match ranks after every ontology match."""
    if match.via_provenance:
        return (1,)
    return (0, match.hop, int(match.breadth_truncated), match.breadth)
