"""Shared fixtures and graph and revision readers for materialisation tests."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from graph_generation import FakeGeneration, GenerationStamp
from materialise.graph import Facet, FacetDefaultPolicy, SubjectPage
from materialise.revision import RevisionSnapshot

from ._scope_availability import ScopeAvailability

if TYPE_CHECKING:
    from materialise.frame_build import Frame
    from materialise.pipeline import MaterialisationResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def read_fixture_text(name: str) -> str:
    """Read a fixture file as text, without normalising its line endings."""
    return (FIXTURES_DIR / name).read_bytes().decode("utf-8")


def only_chunk(result: MaterialisationResult) -> Frame:
    """Return the sole frame chunk, asserting the result has exactly one.
    Fixtures contain fewer than the 100,000-row chunk target.
    """
    chunks = list(result.frame_chunks())
    assert len(chunks) == 1, f"expected exactly one frame chunk, got {len(chunks)}"
    return chunks[0]


# Each fake graph has a generation token of its own, so no fake shares another's memoised
# default scopes (`materialise.place_scope`).
_FAKE_GENERATIONS = itertools.count()

_PLACE_NODE: Mapping[str, tuple[str, ...]] = {"typeOf": ("Place",)}


class FakeGraphReader:
    """Graph test double with per-entity facets and recorded read calls.

    `facets` keys are `(stat_var, entity)`; `policies` use `(stat_var, axis)`.
    `arcs` and `subjects` map their dcid/property keys to graph values.
    `child_places` maps `(parent, child_type)` to the contained places. A StatVar has data at
    the entities holding a facet for it, in the order the read names them, and
    `presence_reads` records each presence read's `(stat_vars, entities)`.
    A place `facets` names is a node typed `Place` unless `arcs` holds it, as
    a live place holding observations is a graph node, so `scope_availability` scopes a
    request binding no place at every place holding a facet. Each instance has its own
    generation token (`generation_stamp`). Other missing nodes return `None`.
    `node_subjects` returns the `subjects` page for `(dcid, property)` when one is
    given, and otherwise scans `arcs` in reverse, in insertion order.
    Name reads combine `name` then `alternateName`, deduplicated. Calls are
    lock-protected for concurrent collection tests. Probe budgets and each
    `(dcids, properties)` read are recorded for assertions.
    """

    def __init__(
        self,
        *,
        facets: Mapping[tuple[str, str], list[Facet]] = {},
        policies: Mapping[tuple[str, str], FacetDefaultPolicy] = {},
        labels: Mapping[str, str] = {},
        arcs: Mapping[str, Mapping[str, tuple[str, ...]]] = {},
        subjects: Mapping[tuple[str, str], SubjectPage] = {},
        child_places: Mapping[tuple[str, str], list[str]] = {},
    ) -> None:
        self._facets = facets
        self._policies = policies
        self._labels = labels
        self._calls_lock = threading.Lock()
        self._arcs = arcs
        self._subjects = subjects
        self._child_places = child_places
        self.calls: dict[str, int] = {
            "node_label": 0,
            "variable_facets_batch": 0,
            "observation_facets_batch_many": 0,
            "child_places_batch": 0,
            "entities_with_data_by_variable": 0,
            "default_facet_policy": 0,
            "node_arc_values": 0,
            "node_subjects": 0,
            "node_arcs_many": 0,
            "node_names_many": 0,
        }
        self.last_individual_probe_budget: int | None = None
        self.node_arcs_many_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.presence_reads: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self._place_nodes = frozenset(entity for _stat_var, entity in facets)
        self._generation = FakeGeneration(token=f"fake-graph-{next(_FAKE_GENERATIONS)}")

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._generation.stamp()

    def un_availability(self) -> None:
        return None

    def scope_availability(self) -> ScopeAvailability:
        """Availability for the default place scope that agrees with these facets."""
        return ScopeAvailability(data=[key for key, found in self._facets.items() if found])

    def _count(self, method: str) -> None:
        with self._calls_lock:
            self.calls[method] += 1

    def node_label(self, dcid: str) -> str | None:
        self._count("node_label")
        return self._labels.get(dcid)

    def variable_facets_batch(
        self, *, stat_var: str, entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, list[Facet]]:
        self._count("variable_facets_batch")
        return {entity: self._facets.get((stat_var, entity), []) for entity in entities}

    def observation_facets_batch_many(
        self, *, stat_vars: Sequence[str], entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, list[Facet]]]:
        self._count("observation_facets_batch_many")
        return {
            stat_var: {entity: self._facets.get((stat_var, entity), []) for entity in entities}
            for stat_var in stat_vars
        }

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        self._count("child_places_batch")
        return {
            parent: list(self._child_places.get((parent, child_type), []))
            for parent in parent_dcids
        }

    def entities_with_data_by_variable(
        self,
        variable_dcids: Sequence[str],
        entity_dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        self._count("entities_with_data_by_variable")
        with self._calls_lock:
            self.presence_reads.append((tuple(variable_dcids), tuple(entity_dcids)))
        return {
            stat_var: [entity for entity in entity_dcids if self._facets.get((stat_var, entity))]
            for stat_var in variable_dcids
        }

    def declares_facet_defaults(self) -> bool:
        return bool(self._policies)

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        self._count("default_facet_policy")
        return self._policies.get((stat_var, axis))

    def un_observation_reader(self) -> None:
        return None

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        self._count("node_arc_values")
        node = self._arcs.get(dcid, {})
        return {prop: tuple(dict.fromkeys(node.get(prop, ()))) for prop in properties}

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        self._count("node_subjects")
        page = self._subjects.get((dcid, property))
        if page is not None:
            return SubjectPage(dcids=tuple(dict.fromkeys(page.dcids)), truncated=page.truncated)
        matches = [
            subject
            for subject, node in self._arcs.items()
            if dcid in node.get(property, ())
            and all(value in node.get(key, ()) for key, value in (filters or {}).items())
        ]
        return SubjectPage(dcids=tuple(matches[:max_results]), truncated=len(matches) > max_results)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self._count("node_arcs_many")
        self.last_individual_probe_budget = individual_probe_budget
        self.node_arcs_many_calls.append((tuple(dcids), tuple(properties)))
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            node = self._arcs.get(dcid, _PLACE_NODE if dcid in self._place_nodes else None)
            result[dcid] = (
                {prop: tuple(dict.fromkeys(node.get(prop, ()))) for prop in properties}
                if node is not None
                else None
            )
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return deduplicated names, with `name` values before `alternateName`."""
        self._count("node_names_many")
        result: dict[str, tuple[str, ...]] = {}
        for dcid in dict.fromkeys(dcids):
            node = self._arcs.get(dcid, {})
            result[dcid] = tuple(
                dict.fromkeys(
                    value for prop in ("name", "alternateName") for value in node.get(prop, ())
                )
            )
        return result


class FakeRevisionReader:
    """Return supplied snapshots in order and count reads; extra reads raise `IndexError`."""

    def __init__(self, snapshots: Sequence[RevisionSnapshot]) -> None:
        self._snapshots = list(snapshots)
        self.read_count = 0

    def read(self) -> RevisionSnapshot:
        snapshot = self._snapshots[self.read_count]
        self.read_count += 1
        return snapshot
