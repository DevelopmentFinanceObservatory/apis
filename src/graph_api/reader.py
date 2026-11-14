"""Narrow DCP read interface for the Graph doors, independent of the Datasets `GraphReader`.

`dc_graph.GraphClient` satisfies it directly. `node_arcs_many` confirms
nodes. `child_places_batch` expands regions. `resolve_by_arc` and
`node_names_many` also make the same object satisfy
`entity_resolution.GraphLookup`. `node_subjects` reads a Dataset's
Provenance children through the reverse `isPartOf` arc. `node_arc_subjects`,
`node_arc_subjects_many` and `node_arc_targets_many` read subjects and
targets with their names, types and the provenances that declared each arc,
which tell the instance's own terms from base Data Commons. Each method
accepts an absolute monotonic deadline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from dc_graph import ArcSubject, ArcSubjectPage, ArcTarget, SubjectsCoverageFlag
from materialise.graph import SubjectPage

type NodeArcsByDcid = dict[str, dict[str, tuple[str, ...]] | None]
"""`node_arcs_many`'s own return shape: every requested dcid as a key, a
dcid DCP does not hold maps to `None`, a held dcid maps to
`{property: (value_or_dcid, ...)}` with one entry per requested property
(a property the node carries no arc for maps to `()`, never omitted).
Each property's values are distinct, in first-occurrence order."""


class GraphNodeReader(Protocol):
    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> NodeArcsByDcid:
        """Read selected arcs in one batch. Each property's values are unique and ordered. An
        empty answer from a batch is read again in a group, then alone up to
        `individual_probe_budget` ids. With `None`, every `None` is the answer for an id read
        alone."""
        ...

    def child_places_batch(
        self,
        parent_dcids: Sequence[str],
        *,
        child_type: str,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        """Return child-place dcids by parent dcid."""
        ...

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Support `entity_resolution`'s fallback ladder."""
        ...

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Allow `deps.graph` to satisfy `entity_resolution.GraphLookup` structurally."""
        ...

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        """Read subjects with a matching arc, optionally filtered by their other properties."""
        ...

    def node_arc_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> ArcSubjectPage:
        """Read subjects as `node_subjects` does, each with its names and provenances."""
        ...

    def node_arc_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, tuple[tuple[ArcSubject, ...], SubjectsCoverageFlag]]:
        """Read many leads' subjects in one walk, each with a coverage flag."""
        ...

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        confirm_empty: bool = False,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        """Read selected arcs' targets with names, types and provenances, or `None` for an absent dcid.
        `confirm_empty` reads an empty answer again until the dcid is read alone."""
        ...
