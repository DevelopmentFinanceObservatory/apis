"""Pre-pipeline gates for set references and the places a Spec binds.

``PlaceSetRefusedError`` is the refusal of a shape that cannot bind its place sets: its
places exceed the one bound (``PlaceSetTooLargeError``), or a set has no reading every role
it fills admits (``PlaceSetNotAdmittedError``).
``place_scope_defaulted_warning`` words the warning a defaulted scope carries.
"""

from __future__ import annotations

import logging

from materialise.place_scope import DefaultPlaceScope
from search.engine.clients.graph import EngineGraphClient
from search.engine.degraded_reads import DegradedReads
from search.engine.place import default_place_scope_message
from search.engine.region.codes import PLACE_SCOPE_DEFAULTED
from search.engine.region.place_set import PlaceSet
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
    return GraphRef(ref=parent, label=label)


class PlaceSetRefusedError(Exception):
    """A shape cannot bind its place sets. An elected shape answers with no data, and an
    alternate, a sibling or a relaxation is dropped."""


class PlaceSetTooLargeError(PlaceSetRefusedError):
    """A shape's places exceed the one bound on the places a Spec binds
    (``SearchSettings.max_child_places``). ``count`` is the number of places counted against
    ``bound``, the distinct dcids per role summed over the roles.
    """

    def __init__(self, *, count: int, bound: int) -> None:
        super().__init__(f"{count} places exceed the bound of {bound}")
        self.count = count
        self.bound = bound


class PartyReadingWithdrawn(Exception):
    """A part read with a word naming a party (`place_roles.stated_party_words`) cannot
    answer from that party: no pooled shape binds it, or another shape reads a phrase the
    shapes binding it do not (`phase_rank_bind.with_party_reading`), or a read failed. The
    part is resolved again without the party reading (`pipeline.resolve_variable`), which
    carries `lost`, the withdrawn reading's record, without the scopes it consulted, which
    decided nothing the reply holds."""

    def __init__(self, *, lost: DegradedReads | None = None) -> None:
        super().__init__("party reading withdrawn")
        self.lost = lost if lost is not None else DegradedReads()


class PlaceSetNotAdmittedError(PlaceSetRefusedError):
    """Sets of a shape that no reading serves: the request asks for the members of each and
    a role it fills does not admit their type, or no reading's type is admitted on every
    role it fills. ``place_sets`` are those sets, in query order."""

    def __init__(self, place_sets: tuple[PlaceSet, ...]) -> None:
        super().__init__(f"no admitted reading for {', '.join(s.name for s in place_sets)}")
        self.place_sets = place_sets


def place_scope_defaulted_warning(scope: DefaultPlaceScope) -> Warning:
    """The ``PLACE_SCOPE_DEFAULTED`` warning for a defaulted scope, naming its mode."""
    return Warning(
        code=PLACE_SCOPE_DEFAULTED,
        severity="warn",
        message=default_place_scope_message(scope),
    )
