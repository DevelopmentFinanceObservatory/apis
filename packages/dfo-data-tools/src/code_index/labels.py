"""Labels the packaged indexes override: a measure partition, a StatVar family, and a node's
display label."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, cast

from code_index.query import CodeIndex
from code_index.types import MeasureIndex, MeasureValueRow, PartitionLabelMatch, label_for_pairs
from statvar_members.enumerate import five_tuple_from_identity

PRICE_BASIS_AXIS = "measurementQualifier"
"""The measure axis holding a StatVar's native price basis (`Nominal`). Unit conversion reads
it as the price basis it converts from."""

PricesKind = Literal["current", "constant"]
"""Current (nominal) or constant (real) prices."""

PRICES_BY_QUALIFIER: Mapping[str, PricesKind] = MappingProxyType(
    {"Nominal": "current", "RealValue": "constant"}
)
"""The `PRICE_BASIS_AXIS` values that state a price basis, and the prices each states. Another
value on that axis (`Adjusted`) is no price basis. No measure label states a price basis,
because the answer's unit phrase does."""


def is_price_basis(axis: str, value: str | None) -> bool:
    """Whether a measure axis value states a price basis (`PRICES_BY_QUALIFIER`)."""
    return axis == PRICE_BASIS_AXIS and value in PRICES_BY_QUALIFIER


# The identity `shape` member each `kind="shape"` measure axis reads.
_SHAPE_AXIS_FIELDS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "measurement_qualifier",
}


def normalize_label(dcid: str, raw_label: str, *, index: CodeIndex) -> str:
    """Return the index label for `dcid` when the index holds a row, otherwise `raw_label`.

    The graph names a code with its code-system suffix (`standard grant (CRS Finance)`). The
    index label drops it. Every surface that shows a graph node label passes it through here.
    """
    row = index.code_by_dcid(dcid)
    return row.label if row is not None else raw_label


def compose_measure_label(match: PartitionLabelMatch, values: Sequence[MeasureValueRow]) -> str:
    """The wire-facing label for one measure partition: `match.label` with each of
    `match.uncovered`'s `(axis, value)` pairs folded back in, except a price basis.

    An authored `MeasurePartitionLabel` omits any axis whose value it folds out of the
    rendered text (see `label_for_pairs`), so `match.label` alone can name two partitions
    identically. Each uncovered pair's `value` is looked up by dcid in `values` and appended
    parenthetically, in `uncovered`'s order, which keeps the published labels distinct.
    A price basis (`is_price_basis`) is skipped, because the answer's unit phrase states it
    and the nominal and real partitions read alike. Returns `match.label` unchanged when
    nothing else is uncovered. A `value` dcid absent from `values` is a measure index
    inconsistency and raises `KeyError`.
    """
    uncovered = [dcid for axis, dcid in match.uncovered if not is_price_basis(axis, dcid)]
    if not uncovered:
        return match.label
    by_dcid = {value.dcid: value for value in values}
    qualifiers = ", ".join(by_dcid[dcid].label for dcid in uncovered)
    return f"{match.label} ({qualifiers})"


def family_label(
    identity: Mapping[str, object], *, index: CodeIndex, measure_index: MeasureIndex
) -> str | None:
    """Name an identity's family: its measure partition label, then its bound slots' labels.

    The measure partition is read from the identity's shape and its value-bound measure-axis
    slots, and labelled with `compose_measure_label`. The other slots follow in slot order.
    `value` and `exact` bindings contribute their one ref, and a `set` binding joins its
    refs' labels with `/`. Measure-axis slots, slots the identity's five-tuple pins
    (`CodeIndex.five_tuple_pins`), unbound slots and refs the index holds no row for
    contribute nothing. Returns `None` when neither part has a label. The presentation door
    labels a collapsed family's `dcid:<spec_id>` ref with this, and the datasets door's
    `stat_vars_summed` warning carries the same string as `stat_var_label`.
    """
    shape = cast("Mapping[str, object]", identity.get("shape", {}))
    slots = cast("Sequence[Mapping[str, object]]", identity.get("slots", ()))
    bound = {cast(str, slot["property"]): _bound_refs(slot) for slot in slots}

    measure_axes = {axis.axis for axis in measure_index.axes}
    pairs = [
        (
            axis.axis,
            cast("str | None", shape.get(_SHAPE_AXIS_FIELDS[axis.axis]))
            if axis.kind == "shape"
            else _sole_ref(bound.get(axis.axis, ())),
        )
        for axis in measure_index.axes
    ]
    match = label_for_pairs(pairs, measure_index.partition_labels)
    parts = [compose_measure_label(match, measure_index.values)] if match is not None else []

    pinned = index.five_tuple_pins.get(five_tuple_from_identity(shape), {})
    for prop, refs in bound.items():
        if prop in measure_axes or prop in pinned:
            continue
        slot_labels = tuple(
            row.label for ref in refs if (row := index.code_by_dcid(ref)) is not None
        )
        if slot_labels:
            parts.append("/".join(slot_labels))
    return ", ".join(parts) if parts else None


def _bound_refs(slot: Mapping[str, object]) -> tuple[str, ...]:
    """The refs a slot's binding names: one for `value` and `exact`, all for `set`."""
    binding = cast("Mapping[str, object]", slot.get("binding", {}))
    state = binding.get("state")
    if state in ("value", "exact"):
        return (cast(str, binding["ref"]),)
    if state == "set":
        return tuple(cast("Sequence[str]", binding["refs"]))
    return ()


def _sole_ref(refs: tuple[str, ...]) -> str | None:
    return refs[0] if len(refs) == 1 else None
