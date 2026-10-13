"""The rule that gives a node its kind, which resolve and the Graph API's node route both call,
so a node has one kind on every route.

`classify` reads, in order, and the first that holds wins:

1. the membership policy: a group it holds is `group`, whatever the graph says of the id;
2. `typeOf`: `StatisticalVariable` is `stat_var`; `Provenance`, `Source` and `Dataset` are
   `provenance`, `source` and `dataset`;
3. the code index: a code system's graph id is `code_system`, and a code row's dcid is `code`;
4. the membership policy: a region it holds is `continent` when its authority classes it a
   continent, else `subregion`, whatever its types (`MembershipPolicy.kind_of`);
5. `typeOf`: a unit or currency type is `unit`;
6. the code index's parties: a party whose kind is `total` or `residual`, by its row or by the
   kind its types give, is `aggregate`;
7. the place and organisation kind of its types (`types.entity_kind`); a node no step classes
   is `other`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from entity_resolution.types import Kind, KindIndex, entity_kind

if TYPE_CHECKING:
    from entity_resolution.membership import MembershipPolicy

_STAT_VAR_TYPE = "StatisticalVariable"
_SOURCE_KIND_BY_TYPE: dict[str, Kind] = {
    "Provenance": "provenance",
    "Source": "source",
    "Dataset": "dataset",
}
_UNIT_TYPES = frozenset({"UnitOfMeasure", "CurrencyUnit", "Currency"})
_AGGREGATE_PARTY_KINDS = frozenset({"total", "residual"})


def classify(
    *,
    dcid: str,
    types: Iterable[str],
    code_index: KindIndex | None,
    membership: MembershipPolicy | None,
) -> Kind:
    """The kind of the node `dcid` whose `typeOf` values are `types`, by the module's rule.

    `code_index` is `None` where no code index is wired, and then no node is a code system,
    code or aggregate party; `membership` is `None` where no policy is, and then no node is a
    group or region."""
    types = tuple(types)
    held = membership.kind_of(dcid) if membership is not None else None
    if held == "group":
        return held
    if _STAT_VAR_TYPE in types:
        return "stat_var"
    if source_kind := next(
        (_SOURCE_KIND_BY_TYPE[t] for t in types if t in _SOURCE_KIND_BY_TYPE), None
    ):
        return source_kind
    if code_index is not None and code_index.is_code_system(dcid):
        return "code_system"
    if code_index is not None and code_index.code_by_dcid(dcid) is not None:
        return "code"
    if held is not None:
        return held
    if _UNIT_TYPES.intersection(types):
        return "unit"
    if code_index is not None and code_index.party_kind_of(dcid, types) in _AGGREGATE_PARTY_KINDS:
        return "aggregate"
    return entity_kind(types)
