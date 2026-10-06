"""The provenance a fixture facet carries when its fixture states none."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

from code_index.query import CodeIndex
from code_index.shared import get_index


@cache
def _index_provenances(index: CodeIndex) -> dict[str, tuple[str, ...]]:
    """Keyed on the index object (hashed by identity), so a test that swaps in a synthetic
    index gets its own map and the packaged one is untouched."""
    return {row.dcid: row.provenances for row in index.statvars}


def index_provenance(stat_var: str) -> str | None:
    """The one dataset the packaged code index records `stat_var` under, or None for a
    variable outside the index or under several datasets."""
    provenances = _index_provenances(get_index()).get(stat_var, ())
    return provenances[0] if len(provenances) == 1 else None


def facet_provenance(stat_var: str, raw: Mapping[str, object]) -> str | None:
    """The `provenanceId` a recorded facet states, else the dataset the code index records
    `stat_var` under. An index variable's observations are its dataset's on the graph and in
    coverage (`series_coverage.verify_index`), and a mint and a Search read plan serve an
    index member from that dataset alone, so a fixture that names none means that one."""
    stated = raw.get("provenanceId")
    return stated if isinstance(stated, str) else index_provenance(stat_var)
