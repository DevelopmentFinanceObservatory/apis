"""Whether a set of StatVar rows reads a code the query names.

One rule serves recall's closure over reading families (`clients.graph.with_reading_families`),
the refusal on a term a rival family records (`region.phase_rank_bind._rival_held_qualifiers`)
and the family election (`bound_values.held_codes`, read by `query_phrases_read`): rows read a
code when one of them carries it, or one of its descendants (`CodeIndex.descendants`), on the
code's property. A measure value on a five-tuple axis ("grant equivalent" names
`measuredProperty` `grantEquivalent`) is read where the rows' five-tuple carries it, and one on
a constraint axis ("net" names `nettingBasis` `Net`) as a code is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet

from code_index.query import CodeIndex


def carried_values(
    constraints: Iterable[Mapping[str, str]], axes: Mapping[str, str | None] | None = None
) -> dict[str, frozenset[str]]:
    """The values `constraints`, rows' constraint maps, carry by property, with each
    five-tuple axis value of `axes` (property to dcid) under its property."""
    values: dict[str, set[str]] = {}
    for row in constraints:
        for prop, value in row.items():
            values.setdefault(prop, set()).add(value)
    for prop, value in (axes or {}).items():
        if value:
            values.setdefault(prop, set()).add(value)
    return {prop: frozenset(held) for prop, held in values.items()}


def reads_code(
    carried: Mapping[str, AbstractSet[str]], code_system: str, code_dcid: str, *, index: CodeIndex
) -> bool:
    """Whether rows carrying `carried` (`carried_values`) hold `code_dcid` or one of its
    descendants on `code_system`."""
    held = carried.get(code_system)
    return bool(held) and not held.isdisjoint(index.descendants(code_dcid))
