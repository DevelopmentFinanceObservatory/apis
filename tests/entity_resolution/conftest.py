"""These tests use the real offline resolver and a call-recording GraphLookup fake."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from entity_resolution.shared import SharedResolver


class FakeGraphLookup:
    """In-memory GraphLookup with recorded calls and deadlines.

    `nodes` supplies arcs and names; `arcs` maps (arc, name) to dcids. Missing nodes return
    None, missing resolutions return [], and names combine `name` then `alternateName`.
    """

    def __init__(
        self,
        *,
        nodes: dict[str, dict[str, tuple[str, ...]]] | None = None,
        arcs: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        self.nodes = nodes or {}
        self.arcs = arcs or {}
        self.node_arcs_many_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.resolve_by_arc_calls: list[tuple[tuple[str, ...], str]] = []
        self.node_names_many_calls: list[tuple[str, ...]] = []
        self.deadline_calls: list[tuple[str, float | None]] = []

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self.node_arcs_many_calls.append((tuple(dcids), tuple(properties)))
        self.deadline_calls.append(("node_arcs_many", deadline))
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            node = self.nodes.get(dcid)
            if node is None:
                result[dcid] = None
            else:
                result[dcid] = {
                    prop: tuple(dict.fromkeys(node.get(prop, ()))) for prop in properties
                }
        return result

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        self.node_names_many_calls.append(tuple(dcids))
        self.deadline_calls.append(("node_names_many", deadline))
        return {
            dcid: tuple(
                dict.fromkeys(
                    value
                    for prop in ("name", "alternateName")
                    for value in self.nodes.get(dcid, {}).get(prop, ())
                )
            )
            for dcid in dict.fromkeys(dcids)
        }

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        self.resolve_by_arc_calls.append((tuple(names), arc))
        self.deadline_calls.append(("resolve_by_arc", deadline))
        return {name: list(dict.fromkeys(self.arcs.get((arc, name), []))) for name in names}


@pytest.fixture(scope="session")
def shared_resolver(_shared_resolvekit_pack: SharedResolver) -> SharedResolver:
    """The session-scoped composed SharedResolver."""
    return _shared_resolvekit_pack
