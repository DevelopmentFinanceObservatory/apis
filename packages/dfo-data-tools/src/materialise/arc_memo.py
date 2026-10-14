"""A `GraphReader` that remembers node arcs and reverse-arc pages for one request.

Conversion decisions read the same few nodes for every member: the table
provenance and its dataset, the native and target currency nodes, and the
currencies of the entities a role-shaped target serves. `MemoGraphReader`
serves a repeat `node_arcs_many` or `node_subjects` read of a property it was
told to remember from memory, so a request over thousands of members reads
each distinct node once. A read of any other property passes through, so a
change signal such as `lastDataRefreshDate`, which the rate-table version
guard reads on both sides of a fetch, always reaches the graph. Facet, label,
and policy reads pass through unchanged.

The memo holds graph state for the life of one request and has no expiry, so
build a fresh reader per request, and per retry after revision drift.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence

from graph_generation import GenerationStamp
from materialise.graph import (
    Facet,
    FacetDefaultPolicy,
    GraphReader,
    SubjectPage,
    UnObservationReader,
)
from origin_catalogue.graph import UnAvailabilityClient

type _SubjectKey = tuple[str, str, tuple[tuple[str, str], ...], int]


class MemoGraphReader:
    """Wraps a `GraphReader`, memoising `node_arcs_many` per node and property
    and `node_subjects` per query, for reads confined to `properties`. A
    memoised `node_arcs_many` call fetches only the dcids missing a requested
    property, in one call to the wrapped reader. Thread-safe: members planned
    or applied on pool threads share one memo.

    `batch` names nodes read together: the first memoised `node_arcs_many`
    read of any of them fetches `batch_properties` for all of them, together with that
    read's own nodes and properties, in one call, and a reader that never reads one makes
    no such call. Every node that call fetched is memoised for the union of the properties.
    """

    def __init__(
        self,
        reader: GraphReader,
        *,
        properties: Iterable[str],
        batch: Iterable[str] = (),
        batch_properties: Iterable[str] = (),
    ) -> None:
        self._reader = reader
        self._properties = frozenset(properties)
        self._lock = threading.Lock()
        self._nodes: dict[str, dict[str, tuple[str, ...]] | None] = {}
        self._fetched: dict[str, frozenset[str]] = {}
        self._subjects: dict[_SubjectKey, SubjectPage] = {}
        self._batch = tuple(dict.fromkeys(batch))
        self._batch_properties = tuple(dict.fromkeys(batch_properties))
        self._batch_pending = bool(self._batch and self._batch_properties)
        self._batch_lock = threading.Lock()

    def _read_batch(
        self, dcids: Sequence[str], properties: Sequence[str], deadline: float | None
    ) -> None:
        """Fetch the batch once, on the first read touching one of its nodes, folding in
        that read's `dcids` and `properties`. A concurrent read waits for the batch rather
        than fetching its own node."""
        if not self._batch_pending or set(dcids).isdisjoint(self._batch):
            return
        with self._batch_lock:
            if not self._batch_pending:
                return
            nodes = tuple(dict.fromkeys((*self._batch, *dcids)))
            union = tuple(dict.fromkeys((*self._batch_properties, *properties)))
            fetched = self._reader.node_arcs_many(nodes, properties=union, deadline=deadline)
            wanted = frozenset(union)
            with self._lock:
                for dcid in nodes:
                    self._store(dcid, wanted, fetched.get(dcid))
            self._batch_pending = False

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        wanted = frozenset(properties)
        if not wanted <= self._properties:
            return self._reader.node_arcs_many(
                dcids,
                properties=properties,
                individual_probe_budget=individual_probe_budget,
                deadline=deadline,
            )
        self._read_batch(dcids, properties, deadline)
        with self._lock:
            missing = [
                dcid
                for dcid in dict.fromkeys(dcids)
                if not wanted <= self._fetched.get(dcid, frozenset())
            ]
        if missing:
            fetched = self._reader.node_arcs_many(
                missing,
                properties=tuple(dict.fromkeys(properties)),
                individual_probe_budget=individual_probe_budget,
                deadline=deadline,
            )
            with self._lock:
                for dcid in missing:
                    self._store(dcid, wanted, fetched.get(dcid))
        with self._lock:
            return {dcid: self._view(dcid, wanted) for dcid in dcids}

    def _store(
        self, dcid: str, wanted: frozenset[str], arcs: dict[str, tuple[str, ...]] | None
    ) -> None:
        """Merge one fetched node into the memo. Caller holds the lock."""
        self._fetched[dcid] = self._fetched.get(dcid, frozenset()) | wanted
        if arcs is None:
            self._nodes.setdefault(dcid, None)
            return
        merged = dict(self._nodes.get(dcid) or {})
        merged.update(arcs)
        self._nodes[dcid] = merged

    def _view(self, dcid: str, wanted: frozenset[str]) -> dict[str, tuple[str, ...]] | None:
        """The memoised node narrowed to `wanted`. Caller holds the lock."""
        node = self._nodes.get(dcid)
        if node is None:
            return None
        return {prop: values for prop, values in node.items() if prop in wanted}

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        if property not in self._properties:
            return self._reader.node_subjects(
                dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
            )
        key: _SubjectKey = (dcid, property, tuple(sorted((filters or {}).items())), max_results)
        with self._lock:
            cached = self._subjects.get(key)
        if cached is not None:
            return cached
        page = self._reader.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )
        with self._lock:
            return self._subjects.setdefault(key, page)

    def un_observation_reader(self) -> UnObservationReader | None:
        return self._reader.un_observation_reader()

    def generation_stamp(self) -> GenerationStamp:
        return self._reader.generation_stamp()

    def source_key(self) -> str:
        return self._reader.source_key()

    def un_availability(self) -> UnAvailabilityClient | None:
        return self._reader.un_availability()

    def node_label(self, dcid: str) -> str | None:
        return self._reader.node_label(dcid)

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return self._reader.node_labels_batch(dcids)

    def variable_facets_batch(
        self, *, stat_var: str, entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, list[Facet]]:
        return self._reader.variable_facets_batch(
            stat_var=stat_var, entities=entities, deadline=deadline
        )

    def observation_facets_batch_many(
        self, *, stat_vars: Sequence[str], entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, list[Facet]]]:
        return self._reader.observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._reader.child_places_batch(parent_dcids, child_type, deadline=deadline)

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._reader.entities_with_data_by_variable(
            variable_dcids, entity_dcids, deadline=deadline
        )

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        return self._reader.default_facet_policy(stat_var=stat_var, axis=axis)

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        return self._reader.node_arc_values(dcid, properties=properties)

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        return self._reader.node_names_many(dcids, deadline=deadline)
