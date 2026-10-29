"""The output digest of a derived frame (`output-digest` in the client contract's rules).

The digest names a frame's output alone: its values, with the variable, members, period and
unit each belongs to. Its document is the RFC 8785 (JCS) form of
`{"digest_version": "out2", "columns", "rows"}`, where `columns` are `stat_var`, then the
frame's role columns in the frame's order, then `time_period`, `value` and `unit`, and each row
holds the frame row's cells in those columns. A unit cell is its JSON form with
`recipe` dropped from every unit-request object, in the cell itself and in either part of a
composed unit. The rows are sorted by the UTF-8 bytes of their own JCS form, and a repeated row
stays. The digest is `out_` and the first 32 hex digits of the SHA-256 of the document.

The facet columns (`measurementMethod`, `provenance`) and `status` stay out, so a source
imported again or a conversion recipe revised moves no digest while no number moves. `stat_var`
stays in, so two frames holding the same values under different variables hold different
outputs.

`compare`, `sum_of`, `share_of` and `quotient_of` write their rows under a variable of their
own, spelt `dcid:` and the step's `ops_` id, which hashes the versions the step read and the
engine that ran it. The digest reads that variable under a name built from what was computed
(`minted_variable_names`): `mv_` and the first 32 hex digits of the SHA-256 of the JCS form
of `{"verb": ..., <slot>: [...]}`, each frame slot of the step holding the sorted names of the
variables its input carries. A root carries its frame's `stat_var` values, `filter`,
`aggregate`, `rank` and `growth` carry their input's, `join` the union of its inputs', and a
minting step its own name. So equal computations over equal variables share a digest across
recipes, engine releases and upstream rebuilds that move no number, and a step with its sides
swapped holds another. A name is 35 characters at any depth, since a nested step's document
holds its inputs' names and not their documents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import rfc8785
from dfo_frame_grammar import Frame, unit_cell_to_json, unit_cells, unit_codes
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
)

DIGEST_VERSION = "out2"
DIGEST_PREFIX = "out_"
_DIGEST_HEX_LENGTH = 32
_RECIPE_KEY = "recipe"
_COMPOSED_PARTS = ("numerator", "denominator")
_DCID_PREFIX = "dcid:"
_MINTING_VERBS = frozenset({"compare", "sum_of", "share_of", "quotient_of"})
_MINTED_PREFIX = "mv_"


def stat_vars(frame: Frame) -> frozenset[str]:
    """The distinct `stat_var` values of `frame`."""
    # The frame grammar holds no null `stat_var`.
    return frozenset(cast("list[str]", frame.table.column(STAT_VAR_COLUMN).unique().to_pylist()))


def minted_variable_names(
    tree: Mapping[str, Any], root_variables: Mapping[int, Iterable[str]]
) -> dict[str, str]:
    """The name of the variable each minting step of `tree` writes its rows under, keyed by
    the step's `ops_` id. See the module docstring.

    Args:
        tree: A recipe tree as computed (`Evaluation.tree`) or stored (a recipe's `tree`):
            each derived node with its `frame` pair, its `op_spec` verb and its `inputs`
            slots, each root node with its `root` index.
        root_variables: The `stat_var` values of each root index's frame.
    """
    names: dict[str, str] = {}

    def variables(node: Mapping[str, Any]) -> frozenset[str]:
        if "root" in node:
            return frozenset(root_variables[node["root"]])
        inputs = {slot: variables(child) for slot, child in node["inputs"].items()}
        verb = str(node["op_spec"]["verb"])
        if verb not in _MINTING_VERBS:
            return frozenset().union(*inputs.values())
        document = rfc8785.dumps(
            {"verb": verb, **{slot: sorted(held) for slot, held in inputs.items()}}
        )
        name = _MINTED_PREFIX + hashlib.sha256(document).hexdigest()[:_DIGEST_HEX_LENGTH]
        names[node["frame"]["spec_id"]] = name
        return frozenset({name})

    variables(tree)
    return names


def _without_recipe(unit: object) -> object:
    """A unit cell's JSON form with `recipe` dropped from every unit-request object."""
    if not isinstance(unit, Mapping):
        return unit
    if any(part in unit for part in _COMPOSED_PARTS):
        return {key: _without_recipe(value) for key, value in unit.items()}
    return {key: value for key, value in unit.items() if key != _RECIPE_KEY}


def _named(stat_var: str, names: Mapping[str, str]) -> str:
    """`stat_var`, or the name `names` gives the `ops_` id it spells."""
    if stat_var.startswith(_DCID_PREFIX):
        return names.get(stat_var.removeprefix(_DCID_PREFIX), stat_var)
    return stat_var


def digest_document(*, role_columns: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    """The JCS bytes of the digest document over `rows`, each holding a frame row's `stat_var`,
    its cells in `role_columns`, then `time_period`, `value` and the unit cell in its JSON
    form."""
    cells = [[*row[:-1], _without_recipe(row[-1])] for row in rows]
    ordered = sorted(cells, key=rfc8785.dumps)
    return rfc8785.dumps(
        {
            "digest_version": DIGEST_VERSION,
            "columns": [
                STAT_VAR_COLUMN,
                *role_columns,
                TIME_PERIOD_COLUMN,
                VALUE_COLUMN,
                UNIT_COLUMN,
            ],
            "rows": ordered,
        }
    )


def output_digest(frame: Frame, names: Mapping[str, str]) -> str:
    """`out_` and 32 hex digits naming the output of `frame`, each `stat_var` cell spelt
    `dcid:` and an `ops_` id that `names` holds read as its name there. See the module
    docstring."""
    role_columns = frame.columns[1 : frame.columns.index(TIME_PERIOD_COLUMN)]
    table = frame.table
    unit_column = table.column(UNIT_COLUMN)
    codes = unit_codes(unit_column)
    units = [
        None if cell is None else unit_cell_to_json(cell) for cell in unit_cells(unit_column, codes)
    ]
    stat_var_cells = cast("list[str]", table.column(STAT_VAR_COLUMN).to_pylist())
    columns = [
        [_named(stat_var, names) for stat_var in stat_var_cells],
        *(
            table.column(name).to_pylist()
            for name in (*role_columns, TIME_PERIOD_COLUMN, VALUE_COLUMN)
        ),
    ]
    rows = [
        [*(column[index] for column in columns), units[code]]
        for index, code in enumerate(cast("list[int]", codes.codes.to_pylist()))
    ]
    document = digest_document(role_columns=role_columns, rows=rows)
    return f"{DIGEST_PREFIX}{hashlib.sha256(document).hexdigest()[:_DIGEST_HEX_LENGTH]}"


__all__ = [
    "DIGEST_PREFIX",
    "DIGEST_VERSION",
    "digest_document",
    "minted_variable_names",
    "output_digest",
    "stat_vars",
]
