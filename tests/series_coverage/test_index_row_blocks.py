"""`CoverageIndex` lookups read each requested variable's row block by binary
search. This module checks every lookup against a reference index that
filters the whole table with membership masks, over a generated table and
random variable lists, role bindings and windows.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from series_coverage.index import BoundRoles, CoverageIndex, Role, _key_values
from series_coverage.schema import RoleNames
from tests.series_coverage.support import coverage_table, with_repeated_dictionaries

_VARIABLES = [f"sv/{name}" for name in random.Random(1).sample(range(1000), 40)]
_FACETS = ["facet-a", "facet-b", "facet-c"]
_DONORS = [f"donor/{d}" for d in range(6)]
_RECIPIENTS = [f"recipient/{r}" for r in range(8)]
_UNKNOWN_VARIABLES = ["sv/unknown-1", "sv/unknown-2"]
# Two role-name pairs, so `estimate` splits a request into several groups.
_ROLE_NAMES = {
    variable: RoleNames(donor="provider", recipient="recipient")
    if i % 3
    else RoleNames(donor="creditor", recipient="debtor")
    for i, variable in enumerate(_VARIABLES)
}


def _rows() -> list[tuple[str, str, str, str, list[int]]]:
    """Up to 12 series per variable, some variables with none, years
    spanning both mask words, in shuffled order so the index sorts them."""
    rng = random.Random(2)
    keys = list(itertools.product(_FACETS, _DONORS, _RECIPIENTS))
    rows = [
        (
            variable,
            facet,
            donor,
            recipient,
            sorted(rng.sample(range(1990, 2041), rng.randint(1, 6))),
        )
        for variable in _VARIABLES
        for facet, donor, recipient in rng.sample(keys, rng.choice([0, 1, 3, 12]))
    ]
    rng.shuffle(rows)
    return rows


def _is_in(column: pa.ChunkedArray, values: Iterable[str]) -> pa.ChunkedArray:
    return pc.is_in(column, value_set=pa.array(list(values), type=pa.string()))


def _reference_member_rows(
    index: CoverageIndex, *, variables: Sequence[str], bound: BoundRoles
) -> pa.Table:
    """`_member_rows` computed by masking every row of the table."""
    table = index._table
    mask = _is_in(table["variable"], variables)
    for role in ("donor", "recipient"):
        if role in bound:
            mask = pc.and_(mask, _is_in(table[role], bound[role]))
    return table.filter(mask)


@pytest.fixture(scope="module", params=["plain", "repeated-dictionaries"])
def indexes(request: pytest.FixtureRequest) -> tuple[CoverageIndex, CoverageIndex]:
    """The index under test and a reference index over the same plain-string
    table. The index under test reads either that table or the same data
    encoded over one-chunk dictionaries holding each value twice."""
    table, horizon = coverage_table(_rows())
    source = table if request.param == "plain" else with_repeated_dictionaries(table, chunks=1)
    index = CoverageIndex(source, _ROLE_NAMES, horizon=horizon)
    reference = CoverageIndex(table, _ROLE_NAMES, horizon=horizon)
    reference._member_rows = lambda *, variables, bound: _reference_member_rows(
        reference, variables=variables, bound=bound
    )
    assert horizon.mask_words == 2
    return index, reference


def _request(case: int) -> tuple[list[str], BoundRoles, int | None, int | None]:
    rng = random.Random(case)
    pool = _VARIABLES + _UNKNOWN_VARIABLES
    variables = rng.sample(pool, rng.choice([0, 1, 2, 5, 10, len(pool)]))
    if variables and rng.random() < 0.2:
        variables.append(variables[0])
    bound: dict[Role, frozenset[str]] = {}
    roles: tuple[tuple[Role, list[str]], ...] = (("donor", _DONORS), ("recipient", _RECIPIENTS))
    for role, dcids in roles:
        if rng.random() < 0.5:
            bound[role] = frozenset(rng.sample([*dcids, f"{role}/unknown"], rng.randint(0, 4)))
    first_year = rng.choice([None, 1985, 2000, 2020, 2024])
    last_year = rng.choice([None, 2045, 2023, 2020])
    if first_year is not None and last_year is not None and first_year > last_year:
        first_year, last_year = last_year, first_year
    return variables, bound, first_year, last_year


@pytest.mark.parametrize("case", range(300))
def test_block_reads_match_the_whole_table_filter(
    indexes: tuple[CoverageIndex, CoverageIndex], case: int
) -> None:
    index, reference = indexes
    variables, bound, first_year, last_year = _request(case)
    window = {"first_year": first_year, "last_year": last_year}

    filtered = index._filtered(variables=variables, bound=bound, **window)
    assert filtered.equals(reference._filtered(variables=variables, bound=bound, **window))
    # `PairRows` reads each column's one chunk. Empty results return before any read.
    assert filtered.num_rows == 0 or all(column.num_chunks == 1 for column in filtered.columns)

    assert index.filter_candidates(
        variables=variables, bound=bound, **window
    ) == reference.filter_candidates(variables=variables, bound=bound, **window)
    assert (
        index.pair_rows(variables=variables, bound=bound, **window).groups()
        == reference.pair_rows(variables=variables, bound=bound, **window).groups()
    )
    for axis in ("donor", "recipient"):
        assert index.open_axis(
            variables=variables, axis=axis, bound=bound, **window
        ) == reference.open_axis(variables=variables, axis=axis, bound=bound, **window)
    assert index.headline(variables=variables, bound=bound) == reference.headline(
        variables=variables, bound=bound
    )
    for estimate_bound in (bound, {}):
        for lanes in (1, 4):
            assert index.estimate(
                variables=variables, bound=estimate_bound, lanes=lanes, **window
            ) == reference.estimate(
                variables=variables, bound=estimate_bound, lanes=lanes, **window
            )
    roles = {"donor_role": "provider", "recipient_role": "recipient"}
    assert index.exhaustive_single_axis_filter(
        variables=variables, **roles
    ) == reference.exhaustive_single_axis_filter(variables=variables, **roles)


def test_the_generated_cases_reach_every_branch(
    indexes: tuple[CoverageIndex, CoverageIndex],
) -> None:
    """Guards the equivalence test against vacuous cases: some requests
    return rows and some none, and the requests include an empty variable
    list, unknown variables, and a role bound to an empty set."""
    index, _reference = indexes
    requests = [_request(case) for case in range(300)]
    sizes = [
        index._filtered(variables=v, bound=b, first_year=f, last_year=la).num_rows
        for v, b, f, la in requests
    ]
    assert sum(size > 0 for size in sizes) > 100
    assert sum(size == 0 for size in sizes) > 20
    assert any(not v for v, *_ in requests)
    assert any(dcids == frozenset() for _v, b, *_ in requests for dcids in b.values())
    assert any(set(v) & set(_UNKNOWN_VARIABLES) for v, *_ in requests)


def test_key_values_decodes_every_row_of_every_chunk() -> None:
    rng = random.Random(3)
    dictionary = pa.array([f"value/{i}" for i in range(50)])
    chunks = [
        pa.DictionaryArray.from_arrays(
            pa.array([rng.randrange(50) for _ in range(size)], type=pa.int32()), dictionary
        )
        for size in (0, 1, 7, 200)
    ]
    column = pa.chunked_array(chunks)
    assert _key_values(column) == column.to_pylist()
    plain = pa.chunked_array([pa.array(["a", "b"]), pa.array(["c"])])
    assert _key_values(plain) == ["a", "b", "c"]
