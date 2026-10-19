"""Which arcs every reader counts, from raw Spanner `Edge` rows or from mixer `/v2/node` reads.

The rule has two stages. `mixer_visible` keeps the edges the mixer returns, so a part built from
Spanner rows holds what a live read can see. `dfo_visible` is DFO's normalisation, applied to mixer
rows and release rows alike. `visible_mask` applies both stages to Arrow columns. Every reader calls
these functions, so the release and a live read agree about each variable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    import pyarrow as pa

_CONSTRAINT_PROPERTIES: Final[str] = "constraintProperties"

MIXER_BLOCKED_PREDICATES: Final[frozenset[str]] = frozenset(
    {"linkedContainedInPlace", "linkedMember", "linkedMemberOf"}
)
"""Predicates the mixer never returns from `/v2/node` (`query_builder.go` in datacommonsorg/mixer).
"""

IMPORT_ADDED_CONSTRAINT_PROPERTIES: Final[frozenset[str]] = frozenset({"sourceIndicatorCode"})
"""Properties every arc read drops from a node's `constraintProperties`.

The Data Commons import tool (`McfMutator.mutate` in datacommonsorg/import) rewrites each
StatisticalVariable's `constraintProperties` to every property outside its own exclusion list,
replacing the list the import declared. `sourceIndicatorCode` names the source series a variable
came from and identifies no variable, yet it lands on the list and can carry two values (IMF WEO
GDP carries `NGDP` and `NGDPD`). Dropping it gives every reader the declared list. The property's
own arc is untouched.
"""


def mixer_visible(predicate: str) -> bool:
    """Stage one: the edges the mixer returns. A target with no `Node` row stays, typed `Thing`."""
    return predicate not in MIXER_BLOCKED_PREDICATES


def dfo_visible(predicate: str, object_id: str | None) -> bool:
    """Stage two: DFO normalisation, applied to mixer rows and release rows alike. `object_id` is
    the target's dcid, or `None` for a literal the mixer answers as a `value`."""
    return not (
        predicate == _CONSTRAINT_PROPERTIES and object_id in IMPORT_ADDED_CONSTRAINT_PROPERTIES
    )


def visible_mask(predicates: pa.Array, object_ids: pa.Array) -> pa.BooleanArray:
    """Both stages over Arrow columns, for the builder's gate and `ReleaseService` reads. A null
    predicate or object id hides nothing, as in the scalar functions: `is_in` answers false for a
    null, so the mask holds no nulls."""
    import pyarrow as pa
    import pyarrow.compute as pc

    blocked = pc.is_in(predicates, value_set=pa.array(sorted(MIXER_BLOCKED_PREDICATES)))
    import_added = pc.and_(
        pc.is_in(predicates, value_set=pa.array([_CONSTRAINT_PROPERTIES])),
        pc.is_in(object_ids, value_set=pa.array(sorted(IMPORT_ADDED_CONSTRAINT_PROPERTIES))),
    )
    return cast("pa.BooleanArray", pc.invert(pc.or_(blocked, import_added)))
