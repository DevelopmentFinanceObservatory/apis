"""The provenance a fixture facet carries when its fixture states none."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache

from code_index.query import CodeIndex
from code_index.shared import get_index
from dc_graph import ArcTarget


@cache
def _index_provenances(index: CodeIndex) -> dict[str, tuple[str, ...]]:
    """Keyed on the index object (hashed by identity), so a test that swaps in a synthetic
    index gets its own map and the fixture index's is untouched."""
    return {row.dcid: row.provenances for row in index.statvars}


def index_provenance(stat_var: str) -> str | None:
    """The one dataset the fixture code index records `stat_var` under, or None for a
    variable outside the index or under several datasets."""
    provenances = _index_provenances(get_index()).get(stat_var, ())
    return provenances[0] if len(provenances) == 1 else None


def facet_provenance(stat_var: str, raw: Mapping[str, object]) -> str | None:
    """The `provenanceId` a recorded facet states, else the dataset the code index records
    `stat_var` under. An index variable's observations are its dataset's on the graph and in
    coverage, and a mint and a Search read plan serve an index member from that dataset alone,
    so a fixture that names none means that one."""
    stated = raw.get("provenanceId")
    return stated if isinstance(stated, str) else index_provenance(stat_var)


def arc_targets(
    read: Mapping[str, Mapping[str, tuple[str, ...]] | None],
    nodes: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
    """`read`, a double's `node_arcs_many` answer, as `node_arc_targets_many` answers it: each
    value a target carrying the first `name` and the `typeOf` the double's `nodes` hold for
    it, as the mixer answers beside an arc's targets."""
    return {
        dcid: None
        if arcs is None
        else {
            prop: tuple(
                ArcTarget(
                    value=value,
                    name=next(iter(nodes.get(value, {}).get("name", ())), None),
                    types=tuple(nodes.get(value, {}).get("typeOf", ())),
                )
                for value in values
            )
            for prop, values in arcs.items()
        }
        for dcid, arcs in read.items()
    }
