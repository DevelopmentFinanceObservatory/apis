"""`index_family` returns, for every identity of a grid over the five-tuples several
datasets publish, the outcome the index rows decide (`family_grid.asked_family`).

The grid runs over the packaged index and over three synthetic ones: two datasets in one
family, the same two split into nested families by a sparse property, and two datasets
that each publish one row, the one carrying a property the other's lacks.
"""

from __future__ import annotations

import pytest

from code_index import default_index_path, load_index
from code_index.members import (
    SourceNotInFamily,
    deciding_query,
    identity_family_query,
    index_family,
)
from code_index.query import CodeIndex

from .family_grid import (
    asked_family,
    five_tuple_rows,
    grid_identities,
    role_class_rows,
    shared_five_tuples,
)
from .test_elect_dataset import (
    _CHANNEL,
    _PLAIN,
    _TABLES,
    _dataset,
    _index,
    _not_declared,
    _row,
    _split_index,
)


def _disjoint_split_index() -> CodeIndex:
    """`_PLAIN` publishes only a row carrying `_CHANNEL` and `_TABLES` only a `sector` row, so
    no row of `_PLAIN`'s family declares `sector` alone, and one of `_TABLES`' does."""
    return _index(
        _row("plain_health_ngo", {"sector": "Health", _CHANNEL: "NGO"}, _PLAIN),
        _row("tables_health", {"sector": "Health"}, _TABLES),
        datasets=(_dataset(_PLAIN, rank=0), _dataset(_TABLES, rank=1)),
        declarations=(_not_declared("sector"), _not_declared(_CHANNEL)),
    )


_INDEXES = {
    "packaged": lambda: load_index(default_index_path()),
    "one-family": lambda: _split_index(sparse=False),
    "split": lambda: _split_index(sparse=True),
    "disjoint-split": _disjoint_split_index,
}


def _resolved(index: CodeIndex, query) -> tuple:
    """`index_family`'s outcome for `query`: the refusal's datasets, no decision, or the
    elected dataset and its rows' dcids."""
    try:
        family = index_family(index, query)
    except SourceNotInFamily as refusal:
        return ("refused", frozenset(refusal.candidates))
    if family is None:
        return ("undecided",)
    return ("family", family.provenance, frozenset(row.dcid for row in family.rows))


@pytest.mark.parametrize("name", _INDEXES)
def test_the_index_rows_decide_the_family_of_every_grid_identity(name: str):
    """The named dataset's rows whose constraint properties lie within the identity's keys
    decide the outcome: the identity resolves to those that hold its bound values, refuses
    `SourceNotInFamily` when only another dataset has such rows, and is left to the graph
    walk when none has. `index_family` answers the deciding query alike. DAC1 and DAC2a
    share the five-tuple of net disbursements as a share of GNI until DAC2a's recipient
    denominators land, so the packaged grid holds a shared
    five-tuple with a denominator."""
    index = _INDEXES[name]()
    rows = {
        five_tuple: five_tuple_rows(index, five_tuple) for five_tuple in shared_five_tuples(index)
    }
    outcomes = dict.fromkeys(("refused", "undecided", "family", "no row", "fewer slots"), 0)
    disagreements = []

    for identity in grid_identities(index):
        query = identity_family_query(index, identity)
        asked = asked_family(rows[query.five_tuple], query)
        resolved = _resolved(index, query)
        if asked.refused is not None:
            expected = [("refused", asked.refused)]
        elif not asked.decided:
            expected = [("undecided",)]
        else:
            # A query naming no dataset elects one of those its keys select.
            expected = [
                ("family", dataset, role_class_rows(index, found, query.role_keys))
                for dataset, found in asked.rows.items()
                if found
            ] or [("family", None, frozenset())]
        outcomes["no row" if resolved[1:] == (None, frozenset()) else resolved[0]] += 1
        outcomes["fewer slots"] += deciding_query(index, query) is not query
        if resolved not in expected:
            disagreements.append((identity, resolved[:2], [found[:2] for found in expected]))
        elif (deciding := deciding_query(index, query)) is not query and (
            _resolved(index, deciding) != resolved
        ):
            disagreements.append((identity, "the deciding query resolves another family"))

    assert not disagreements, f"{len(disagreements)} disagree; first: {disagreements[0]!r}"
    # The agreement is vacuous for an outcome the grid never produces. Both datasets of the
    # one-family index publish its one family, and a provenance it holds no row of names no
    # dataset (`named_dataset`), so no identity there refuses.
    if name == "one-family":
        assert outcomes.pop("refused") == 0
    assert all(outcomes.values()), outcomes
