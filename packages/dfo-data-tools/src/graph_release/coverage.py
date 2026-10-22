"""Series coverage as a view over the `series` part.

`role_map` reads each variable's `observationProperties` from `edges` and names its donor and
recipient roles by one rule: a variable declaring two roles takes the first in sorted order as
its donor and must have that role on the origin axis and the second on the destination axis.
A variable declaring one role holds it on the column of its role's flow axis. `coverage_rows`
turns `series` into the coverage rows `series_coverage.CoverageIndex` holds (columns in
`series_coverage.schema`), one per series of a variable the role map names: `entity1` is the
donor, and the recipient for a lone destination-axis role, and `entity2` the recipient of a
two-role variable.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from types import MappingProxyType
from typing import Final, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from dfo_platform.roles import RoleAxis
from graph_release.format import MASK_EPOCH, MASK_WORDS
from graph_release.platform import PlatformRefusal
from series_coverage.schema import (
    ABSENT_ENTITY,
    DONOR_COLUMN,
    FACET_COLUMN,
    FIRST_YEAR_COLUMN,
    LAST_YEAR_COLUMN,
    N_OBS_COLUMN,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    RoleNames,
    YearHorizon,
)

type RoleMap = Mapping[str, RoleNames]
"""Each role-bearing variable's donor and recipient role names, by variable."""

_OBSERVATION_PROPERTIES: Final[str] = "observationProperties"
_KEY_COLUMNS: Final[tuple[str, ...]] = (
    VARIABLE_COLUMN,
    FACET_COLUMN,
    DONOR_COLUMN,
    RECIPIENT_COLUMN,
)
_EXAMPLES: Final[int] = 5

HORIZON: Final[YearHorizon] = YearHorizon(
    epoch_year=date.fromisoformat(MASK_EPOCH).year, mask_words=MASK_WORDS
)
"""The years the release's period masks cover, as the coverage schema names them."""


def role_map(edges: pa.Table, role_axes: Mapping[str, RoleAxis]) -> RoleMap:
    """The role names of every subject of `edges` that declares one or two
    `observationProperties` roles, sorted by variable. A subject declaring three or more is
    outside the coverage population and left out.

    Raises `PlatformRefusal` with code `missing_role_declaration` for a lone role on no flow
    axis in `role_axes`, and for a pair whose sorted order is not an origin-axis role then a
    destination-axis role.
    """
    declared = edges.filter(
        pc.equal(_strings(edges["predicate"]), pa.scalar(_OBSERVATION_PROPERTIES))
    )
    roles: dict[str, set[str]] = {}
    for variable, role in zip(
        cast("list[str]", _strings(declared["subject_id"]).to_pylist()),
        cast("list[str]", _strings(declared["object_id"]).to_pylist()),
        strict=True,
    ):
        roles.setdefault(variable, set()).add(role)

    names: dict[str, RoleNames] = {}
    refused: list[str] = []
    for variable in sorted(roles):
        match tuple(sorted(roles[variable])):
            case (role,):
                axis = role_axes.get(role)
                if axis is None:
                    refused.append(f"{variable} ({role})")
                    continue
                names[variable] = (
                    RoleNames(donor=role, recipient=None)
                    if axis == "origin"
                    else RoleNames(donor=None, recipient=role)
                )
            case (donor, recipient):
                if (role_axes.get(donor), role_axes.get(recipient)) != ("origin", "destination"):
                    refused.append(f"{variable} ({donor}, {recipient})")
                    continue
                names[variable] = RoleNames(donor=donor, recipient=recipient)
            case _:
                continue
    if refused:
        raise PlatformRefusal(
            "missing_role_declaration",
            "variables whose roles are not a lone flow-axis role or an origin-axis role then a "
            f"destination-axis role in sorted order: {_examples(refused)}",
        )
    return MappingProxyType(names)


def coverage_rows(series: pa.Table, role_map: RoleMap) -> pa.Table:
    """The coverage rows of `series` for the variables `role_map` names.

    The columns are `HORIZON.schema`'s, with the four key columns dictionary-encoded over sorted
    dictionaries as `CoverageIndex` reads them, and the rows sorted by that key. `series` may hold several chunks with their own dictionaries, as a
    loaded release does.

    Raises `PlatformRefusal` with code `missing_role_declaration` for a series whose `entity2`
    disagrees with its variable's role count, set for a variable outside the role map or
    declaring one role and absent for one declaring two.
    """
    ordered = sorted(role_map)
    variables = pa.array(ordered, pa.string())
    names = [role_map[variable] for variable in ordered]
    has_donor = np.array([role.donor is not None for role in names], dtype=bool)
    has_recipient = np.array([role.recipient is not None for role in names], dtype=bool)

    chunks = []
    mismatched: set[str] = set()
    for batch in series.to_batches():
        if not batch.num_rows:
            continue
        variable = _dictionary(batch.column("variable_measured"))
        position = pc.index_in(variable.dictionary, value_set=variables).fill_null(-1).to_numpy()
        known = position >= 0
        index = np.maximum(position, 0)
        codes = variable.indices.to_numpy(zero_copy_only=False)
        in_map = known[codes]
        donor_role = (known & has_donor[index])[codes]
        recipient_role = (known & has_recipient[index])[codes]
        two_roles = donor_role & recipient_role

        entity1 = _strings(batch.column("entity1"))
        entity2 = _strings(batch.column("entity2"))
        has_entity2 = pc.is_valid(entity2).to_numpy(zero_copy_only=False)
        wrong = has_entity2 != two_roles
        if wrong.any():
            mismatched.update(
                cast("list[str]", _strings(variable).filter(pa.array(wrong)).to_pylist())
            )
            continue

        donor = pc.if_else(pa.array(donor_role), entity1, pa.scalar(ABSENT_ENTITY))
        recipient = pc.if_else(
            pa.array(two_roles),
            entity2,
            pc.if_else(pa.array(recipient_role), entity1, pa.scalar(ABSENT_ENTITY)),
        )
        rows = pa.table(
            {
                VARIABLE_COLUMN: variable,
                FACET_COLUMN: _dictionary(batch.column("facet_id")),
                DONOR_COLUMN: pc.dictionary_encode(donor),
                RECIPIENT_COLUMN: pc.dictionary_encode(recipient),
                N_OBS_COLUMN: batch.column("observation_count").cast(pa.uint32()),
                FIRST_YEAR_COLUMN: _years(batch.column("earliest_date")),
                LAST_YEAR_COLUMN: _years(batch.column("latest_date")),
                **{
                    name: _unsigned(batch.column(f"period_mask_{word}"))
                    for word, name in enumerate(HORIZON.mask_columns)
                },
            }
        )
        chunks.append(rows.filter(pa.array(in_map)))
    if mismatched:
        raise PlatformRefusal(
            "missing_role_declaration",
            "series whose entity2 disagrees with their variable's role count: "
            f"{_examples(sorted(mismatched))}",
        )
    return _sorted_rows(chunks)


def _sorted_rows(chunks: list[pa.Table]) -> pa.Table:
    """`chunks` as one table in `HORIZON.schema`'s column order, key columns over sorted
    dictionaries, rows sorted by the key."""
    schema = HORIZON.schema
    if not chunks:
        return schema.empty_table()
    table = pa.concat_tables(chunks).unify_dictionaries().combine_chunks()
    ranks = {}
    columns: dict[str, pa.Array] = {}
    for name in _KEY_COLUMNS:
        encoded = _dictionary(table[name].combine_chunks())
        used = encoded.dictionary.take(pc.unique(encoded.indices))
        ordered = used.take(pc.array_sort_indices(used))
        rank = pc.index_in(encoded.dictionary, value_set=ordered).fill_null(0).to_numpy()
        ranks[name] = rank[encoded.indices.to_numpy()]
        columns[name] = ordered
    order = cast("pa.Int64Array", pa.array(np.lexsort([ranks[n] for n in reversed(_KEY_COLUMNS)])))
    arrays = []
    for field in schema:
        if field.name in columns:
            arrays.append(
                pa.DictionaryArray.from_arrays(
                    pa.array(ranks[field.name]).take(order), columns[field.name]
                )
            )
        else:
            arrays.append(table[field.name].combine_chunks().take(order))
    fields = [
        pa.field(field.name, arrays[i].type, nullable=False) for i, field in enumerate(schema)
    ]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _unsigned(mask: pa.Array) -> pa.Array:
    """A signed mask word as the unsigned word the coverage schema stores, bit for bit. A null
    word, a period the mask does not count, raises `ValueError`."""
    if mask.null_count:
        raise ValueError("a period mask word is null; coverage counts annual series only")
    return pc.cast(mask, pa.uint64(), safe=False)


def _dictionary(column: pa.Array) -> pa.DictionaryArray:
    if not pa.types.is_dictionary(column.type):
        column = pc.dictionary_encode(column)
    return cast("pa.DictionaryArray", column)


def _years(dates: pa.Array) -> pa.Array:
    """The calendar year of each `date32` value."""
    days = dates.to_numpy(zero_copy_only=False).astype("datetime64[Y]")
    return pa.array(days.astype(np.int64) + 1970, pa.uint16())


def _strings(column: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return column.cast(pa.string()) if pa.types.is_dictionary(column.type) else column


def _examples(items: list[str]) -> str:
    more = f" and {len(items) - _EXAMPLES} more" if len(items) > _EXAMPLES else ""
    return f"{items[:_EXAMPLES]}{more}"


__all__ = ["HORIZON", "RoleMap", "coverage_rows", "role_map"]
