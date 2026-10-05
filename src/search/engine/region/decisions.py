"""Pre-pipeline gates for set references and standard-family probe places.

``standard_probe_union`` uses resolved dcids and performs no graph I/O.
``place_scope_defaulted_warning`` words the warning a defaulted scope carries.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from materialise.place_scope import DefaultPlaceScope
from search.engine.clients.graph import EngineGraphClient
from search.engine.place import default_place_scope_message
from search.engine.region.codes import PLACE_SCOPE_DEFAULTED, STANDARD_PROBE_PLACES_CLAMPED
from search.models import GraphRef, Warning

logger = logging.getLogger(__name__)


def detect_set_ref(
    *,
    value_dcids: list[str],
    graph: EngineGraphClient,
) -> GraphRef | None:
    """Return the taxonomy parent of a BindingSet that covers all its children.

    Checks one level: each member must have the same sole parent, whose full
    child set equals the members. Missing or ambiguous parents, partial matches,
    or an unreadable parent label return ``None``. Callers restrict this check
    to what/how axes.
    """
    if len(value_dcids) < 2:
        return None
    arcs = graph.node_arcs_batch(value_dcids)
    parent: str | None = None
    for dcid in value_dcids:
        member_arcs = arcs.get(dcid) or {}
        parent_nodes = member_arcs.get("isPartOf", {}).get("nodes", [])
        parent_dcids = [n["dcid"] for n in parent_nodes if "dcid" in n]
        if len(parent_dcids) != 1:
            return None
        p = parent_dcids[0]
        if parent is None:
            parent = p
        elif p != parent:
            return None
    if parent is None:
        return None
    children = set(graph.child_dcids(parent))
    if children != set(value_dcids):
        return None
    label = graph.node_labels_batch([parent]).get(parent)
    if label is None:
        return None
    return GraphRef(dcid=parent, label=label)


def standard_probe_union(
    place_dcids: list[str],
    place_children_by_dcid: Mapping[str, list[str]],
    *,
    max_probe_places: int,
) -> tuple[list[str], list[Warning]]:
    """Expand named places into a bounded, ordered probe union.

    Each place contributes its expanded members, already capped per region at
    ``SEARCH_MAX_CHILD_PLACES``, or its own dcid when expansion is empty.
    Expansions stay contiguous, then deduplicate in first-seen order.

    If the union exceeds ``max_probe_places``, trailing dcids are dropped and
    a ``STANDARD_PROBE_PLACES_CLAMPED`` warning is added alongside any
    per-region ``CHILD_PLACES_CLAMPED`` warnings. Return ``(dcids, warnings)``.
    Warnings are empty when no union clamp occurs.
    """
    union: list[str] = []
    for dcid in place_dcids:
        union.extend(place_children_by_dcid.get(dcid) or [dcid])
    deduped = list(dict.fromkeys(union))
    if len(deduped) <= max_probe_places:
        return deduped, []
    warning = Warning(
        code=STANDARD_PROBE_PLACES_CLAMPED,
        severity="warn",
        message=(
            f"{len(deduped)} probe places detected across {len(place_dcids)} named "
            f"place(s); resolved the first {max_probe_places}."
        ),
    )
    return deduped[:max_probe_places], [warning]


def place_scope_defaulted_warning(scope: DefaultPlaceScope) -> Warning:
    """The ``PLACE_SCOPE_DEFAULTED`` warning for a defaulted scope, naming its mode."""
    return Warning(
        code=PLACE_SCOPE_DEFAULTED,
        severity="warn",
        message=default_place_scope_message(scope),
    )
