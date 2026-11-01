"""Build resolved scopes from member dcids and graph labels and types.

`build_scope` is pure. `fetch_scope_members` batches labels and types over the
union of member dcids, so a response with scopes needs two graph calls, not two
per spec. An answer's scope members are the ones its read plan serves
(`stated_scope_members`), the list coverage counts its breadth dimension from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialise.place_scope import scope_places
from search.engine.retrieval.coverage import scope_members
from search.models import EntityRoleKey, ResolvedEntity, ResolvedScope

if TYPE_CHECKING:
    from search.engine.clients.graph import EngineGraphClient
    from search.engine.retrieval.read_plan import ReadPlan
    from search.engine.retrieval.retrieve import Materialised


def scope_axis(mat_result: Materialised) -> EntityRoleKey:
    """Return `discovered_scope.axis`, or `observationAbout` for a standard scope."""
    if mat_result.discovered_scope is not None:
        return mat_result.discovered_scope.axis
    return "observationAbout"


def build_scope(
    axis: EntityRoleKey,
    member_dcids: list[str],
    labels: dict[str, str],
    types: dict[str, str],
) -> ResolvedScope:
    """Build a scope; missing labels or types remain `None`. A place scope
    (`observationAbout`) is in `scope_places` order, by dcid with Earth last, as the datasets
    door states a default scope; a discovered role scope is sorted by dcid."""
    ordered = scope_places(member_dcids) if axis == "observationAbout" else sorted(member_dcids)
    return ResolvedScope(
        axis=axis,
        members=[
            ResolvedEntity(
                ref=dcid,
                label=labels.get(dcid),
                entity_type=types.get(dcid),
                role=axis,
            )
            for dcid in ordered
        ],
    )


def stated_scope_members(mat_result: Materialised, plan: ReadPlan | None) -> list[str] | None:
    """The members an answer lists as its defaulted scope: those `plan`, its read plan under
    the request's facet filter, serves (`coverage.scope_members`), so the list holds the
    members the coverage breadth dimension counts. A carrier with no plan lists the members
    its resolver recorded. None when no scope was defaulted."""
    if mat_result.scope_member_dcids is None:
        return None
    if plan is None or mat_result.coverage_form is None:
        return mat_result.scope_member_dcids
    return scope_members(plan, mat_result.coverage_form)


def fetch_scope_members(
    member_dcids: list[str], *, graph: EngineGraphClient
) -> tuple[dict[str, str], dict[str, str]]:
    """Fetch labels and first types in two batches; empty input makes no graph calls.

    Callers pass the union of all specs' member dcids once per response.
    """
    if not member_dcids:
        return {}, {}
    labels = graph.node_labels_batch(member_dcids)
    types = graph.node_types_batch(member_dcids)
    return labels, {dcid: found[0] for dcid, found in types.items()}
