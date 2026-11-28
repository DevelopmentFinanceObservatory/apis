"""The facts a dataset id reads from sources that record them per member provenance.

From the read boundary on, a facet's provenance-axis value, a row's `provenance` cell and every
answer-level ref carry a dataset id (`code_index.shared.dataset_of`): a declared Dataset's id
for a series of one of its member provenances, the provenance itself for a plain table. The
origin catalogue, the graph's `lastDataRefreshDate` and `conversionRecipe` arcs, the packaged
`method_status.json` and a StatVar's declared defaults record their facts per provenance. Every
reader that joins the two goes through this module, which answers for the dataset a ref
belongs to, whichever of the two vocabularies the ref is in: it reads the members of
`dataset_of(ref)` (`members_of`), so a member provenance and its dataset's id read one fact.
A plain table is its own sole member, so each fold reads its own fact unchanged.

The rule of each fact:

- catalogue row facts (`dataset_origin`, `dataset_name`, `entry_licence`, `row_licence`): the
  value every member the catalogue lists gives (`shared_row_fact`), else `None`; the name
  falls back to the dataset's declared label.
- the partial mark (`partial_marks`, `partial_mark`): every member's mark, judged by the one
  that starts first, so the dataset holds back each period any member marks.
- projection methods (`projection_methods`, `is_projection`): the union of the members'
  declarations, so a facet's rows and its held runs agree whichever id they read by.
- reported status (`reported_sources`, `status_of`): a dataset's method-less values count as
  reported when every member is a reported provenance, so a member and its dataset agree.
- the refresh date (`refresh_nodes`, `newest_refresh`): the newest of the Dataset node's
  date and every member's.
- the conversion recipe (`dataset_recipe_id`): the Dataset node's own, else the one every
  member names, else none (`recipe_undeclared` at the mint).
- a declared default policy (`dataset_policy`, `dataset_policies`): a provenance default
  naming a member compares as the member's dataset.

Every function reads the member map of the bound release's code index and raises
`graph_release.context.ReleaseNotBound` when none is bound.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from code_index.shared import get_index, per_index
from materialise.method_status import (
    StatusClass,
    declared_projection_methods,
    method_status_classes,
    reported_provenances,
)
from materialise.partial_period import start_of
from origin_catalogue import Catalogue, DataCommons, ProvenanceRow, SourceRef, entry_ref
from origin_catalogue.catalogue import Licence, Undocumented, partial_period_of

if TYPE_CHECKING:
    from code_index.query import CodeIndex
    from dc_graph.client import FacetDefaultPolicy

__all__ = [
    "agreed",
    "dataset_name",
    "dataset_origin",
    "dataset_policies",
    "dataset_policy",
    "dataset_recipe_id",
    "entry_licence",
    "is_projection",
    "members_of",
    "newest_refresh",
    "partial_mark",
    "partial_marks",
    "projection_methods",
    "refresh_nodes",
    "reported_sources",
    "row_licence",
    "shared_row_fact",
    "status_of",
]

_PROVENANCE_AXIS = "provenance"


def members_of(ref: str) -> tuple[str, ...]:
    """The member provenances of the dataset `ref` belongs to in the bound release's code
    index (`CodeIndex.provenances_of` of `CodeIndex.dataset_of`), sorted: `(ref,)` for a
    plain table, and every member of the Dataset for a Dataset's id or a member's."""
    index = get_index()
    return index.provenances_of(index.dataset_of(ref))


def agreed[T](values: Iterable[T | None]) -> T | None:
    """The value every one of `values` holds, `None` when one is `None`, two differ or there
    are none."""
    found: list[T] = []
    for value in values:
        if value is None or (found and value != found[0]):
            return None
        found.append(value)
    return found[0] if found else None


def shared_row_fact[T](
    ref: str, *, catalogue: Catalogue, fact: Callable[[ProvenanceRow], T | None]
) -> T | None:
    """The `fact` every catalogue row of `ref`'s members (`members_of`) gives, a member the
    catalogue does not list adding none. `None` when no member is listed or the listed rows
    give different values."""
    rows = [row for member in members_of(ref) if (row := catalogue.provenance(member)) is not None]
    return agreed(fact(row) for row in rows)


def dataset_origin(ref: str, *, catalogue: Catalogue) -> tuple[SourceRef, DataCommons] | None:
    """The source and Data Commons the catalogue rows of `ref`'s members share
    (`shared_row_fact`). A citation (`provenance_doc.read_source_citations`) and a Search
    answer (`search.observations.provenance_ref`) both cite a dataset by it."""
    return shared_row_fact(
        ref, catalogue=catalogue, fact=lambda row: (row.source, row.data_commons)
    )


def dataset_name(ref: str, *, catalogue: Catalogue) -> str | None:
    """The name the catalogue rows of `ref`'s members share (`shared_row_fact`), else the
    declared label of `ref`'s dataset in the bound release's code index, else `None`."""
    if (name := shared_row_fact(ref, catalogue=catalogue, fact=lambda row: row.name)) is not None:
        return name
    index = get_index()
    declared = index.datasets.get(index.dataset_of(ref))
    return declared.label if declared is not None else None


def entry_licence(ref: str, *, catalogue: Catalogue) -> str | None:
    """The licence label the source-index entries of `ref`'s members share
    (`CardFacts.license_type` of each listed row's `entry_ref`), else `None`."""

    def licence(row: ProvenanceRow) -> str | None:
        facts = catalogue.dataset_facts(entry_ref(row))
        return facts.license_type if facts is not None else None

    return shared_row_fact(ref, catalogue=catalogue, fact=licence)


def row_licence(ref: str, *, catalogue: Catalogue) -> Licence | None:
    """The licence the catalogue rows of `ref`'s members share (`ProvenanceRow.licence`), else
    `None`. Rows that state no licence share an empty `Licence`."""
    return shared_row_fact(ref, catalogue=catalogue, fact=lambda row: row.licence)


def partial_marks(catalogue: Catalogue, source: str) -> tuple[str, ...]:
    """The distinct partial marks, sorted, of `source`'s members (`members_of`), each as the
    catalogue documents it (`origin_catalogue.partial_period_of`), a member the catalogue does
    not document adding none. Empty where none is marked.

    Raises:
        Undocumented: `catalogue` documents none of the members."""
    marks: set[str] = set()
    documented = False
    for provenance in members_of(source):
        try:
            mark = partial_period_of(catalogue, provenance)
        except Undocumented:
            continue
        documented = True
        if mark is not None:
            marks.add(mark)
    if not documented:
        raise Undocumented(source)
    return tuple(sorted(marks))


def partial_mark(catalogue: Catalogue, source: str) -> str | None:
    """The partial mark `source`'s rows are judged by: the one of `partial_marks` that starts
    first (`partial_period.start_of`), since it reaches every period a later one reaches, so a
    dataset holds back each period any member marks. `None` where none is marked.

    Raises:
        Undocumented: as `partial_marks` does."""
    return min(partial_marks(catalogue, source), key=start_of, default=None)


def projection_methods(provenance: str | None) -> frozenset[str]:
    """The `measurementMethod`s the members of `provenance`'s dataset (`members_of`) declare
    as projections (`method_status.declared_projection_methods`), joined. Empty for `None` and
    for a dataset no member of which declares any, whose estimates are actual rows."""
    if provenance is None:
        return frozenset()
    members = members_of(provenance)
    if len(members) == 1:
        return declared_projection_methods(members[0])
    return frozenset().union(*map(declared_projection_methods, members))


def is_projection(provenance: str | None, method: str | None) -> bool:
    """Whether a row under `method` and `provenance` is a projection (`projection_methods`)."""
    return method is not None and method in projection_methods(provenance)


def reported_sources() -> frozenset[str]:
    """The provenances and dataset ids whose values with no `measurementMethod` count as
    reported: each id whose dataset's every member is among `method_status.reported_provenances`,
    the dataset's own id included."""
    return _reported_sources(get_index())


@per_index
def _reported_sources(index: CodeIndex) -> frozenset[str]:
    listed = reported_provenances()
    reported: set[str] = set()
    for dataset in {index.dataset_of(provenance) for provenance in listed}:
        members = index.provenances_of(dataset)
        if all(member in listed for member in members):
            reported.update((dataset, *members))
    return frozenset(reported)


def status_of(method: str | None, provenance: str | None) -> StatusClass | None:
    """The status of the values a facet with `method` under `provenance` holds: the
    method's class (`method_status.method_status_classes`), or for no method `reported` when
    `provenance` is among `reported_sources`. `None` when unknown."""
    if method is None:
        return "reported" if provenance in reported_sources() else None
    return method_status_classes().get(method)


def refresh_nodes(ref: str) -> tuple[str, ...]:
    """The graph nodes whose `lastDataRefreshDate` dates `ref`'s dataset: the dataset's own
    node, then each member provenance (`members_of`). `(ref,)` for a plain table."""
    return tuple(dict.fromkeys((get_index().dataset_of(ref), *members_of(ref))))


def newest_refresh(dates: Iterable[str | None]) -> str | None:
    """The newest of `dates` (ISO dates, which order as strings), a missing one adding none.
    `None` when none is given."""
    return max((date for date in dates if date is not None), default=None)


def dataset_recipe_id(ref: str, *, recipe_of: Callable[[str], str | None]) -> str | None:
    """The conversion recipe id of `ref`'s dataset, `recipe_of` naming each node's own
    election: the dataset node's, else the one every member provenance names (`agreed`), else
    `None`. A plain table reads its own."""
    dataset = get_index().dataset_of(ref)
    if (own := recipe_of(dataset)) is not None:
        return own
    members = [member for member in members_of(ref) if member != dataset]
    return agreed(map(recipe_of, members)) if members else None


def dataset_policy(axis: str, policy: FacetDefaultPolicy) -> FacetDefaultPolicy:
    """`policy`, a default `axis` declares, as a reader of the dataset vocabulary compares it:
    a provenance default naming a member provenance names the member's dataset
    (`CodeIndex.dataset_of`). Any other axis, and a plain table's provenance, is unchanged."""
    if axis != _PROVENANCE_AXIS:
        return policy
    dataset = get_index().dataset_of(policy.resolved_ref)
    return policy if dataset == policy.resolved_ref else replace(policy, resolved_ref=dataset)


def dataset_policies(
    policies: Mapping[str, FacetDefaultPolicy | None],
) -> dict[str, FacetDefaultPolicy | None]:
    """`policies`, by axis, each read as `dataset_policy` reads it."""
    return {
        axis: None if policy is None else dataset_policy(axis, policy)
        for axis, policy in policies.items()
    }
