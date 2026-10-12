"""Graph and revision-reader doubles for datasets_api door tests."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_generation import FakeGeneration, GenerationStamp
from materialise.graph import Facet, FacetDefaultPolicy
from statvar_members import CANDIDATE_CAP


@dataclass(frozen=True)
class _FakeSubjectPage:
    """Structural `SubjectPage` stand-in with `dcids` and `truncated` fields."""

    dcids: tuple[str, ...]
    truncated: bool


# Each fake graph has a generation token of its own, so no fake shares another's memoised
# default scopes (`materialise.place_scope`).
_FAKE_GENERATIONS = itertools.count()

_PLACE_NODE: Mapping[str, tuple[str, ...]] = {"typeOf": ("Place",)}


class FakeMintGraphReader:
    """GraphReader and MemberArcReader double backed by one `arcs` map.

    `node_subjects` scans reverse arcs; `node_arcs_many` reads requested nodes and returns `None`
    for absent entries. A place `facets` names is a node typed `Place` unless `arcs` holds it, as
    a live place holding observations is a graph node. A StatVar has data at the entities
    holding a facet for it, and `presence_reads` records each presence read's
    `(stat_vars, entities)`. Each instance has its own generation token (`generation_stamp`)
    and reaches no UN deployment, unless `generation` supplies one a test advances. The locked
    `calls` counts are safe across concurrent materialisation workers.
    """

    def __init__(
        self,
        *,
        facets: Mapping[tuple[str, str], list[Facet]] = {},
        policies: Mapping[tuple[str, str], FacetDefaultPolicy] = {},
        labels: Mapping[str, str] = {},
        arcs: Mapping[str, Mapping[str, tuple[str, ...]]] = {},
        child_places: Mapping[tuple[str, str], list[str]] = {},
        generation: FakeGeneration | None = None,
    ) -> None:
        self._facets = facets
        self._child_places = child_places
        self._policies = policies
        self._labels = labels
        self._arcs = arcs
        self._place_nodes = frozenset(entity for _stat_var, entity in facets)
        self._generation = (
            generation
            if generation is not None
            else FakeGeneration(token=f"fake-mint-graph-{next(_FAKE_GENERATIONS)}")
        )
        self._calls_lock = threading.Lock()
        self.calls: dict[str, int] = {
            "node_label": 0,
            "node_labels_batch": 0,
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
        self.presence_reads: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def _count(self, method: str) -> None:
        with self._calls_lock:
            self.calls[method] += 1

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._generation.stamp()

    def un_availability(self) -> None:
        return None

    def node_label(self, dcid: str) -> str | None:
        self._count("node_label")
        return self._labels.get(dcid)

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        self._count("node_labels_batch")
        return {dcid: self._labels[dcid] for dcid in dcids if dcid in self._labels}

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
    ) -> _FakeSubjectPage:
        self._count("node_subjects")
        filters = filters or {}
        matches = sorted(
            candidate
            for candidate, node_arcs in self._arcs.items()
            if dcid in node_arcs.get(property, ())
            and all(
                filter_value in node_arcs.get(filter_property, ())
                for filter_property, filter_value in filters.items()
            )
        )
        truncated = len(matches) > max_results
        return _FakeSubjectPage(dcids=tuple(matches[:max_results]), truncated=truncated)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self._count("node_arcs_many")
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for candidate in dcids:
            node = self._arcs.get(
                candidate, _PLACE_NODE if candidate in self._place_nodes else None
            )
            result[candidate] = (
                None
                if node is None
                else {prop: tuple(dict.fromkeys(node.get(prop, ()))) for prop in properties}
            )
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        self._count("node_names_many")
        return {
            candidate: tuple(
                dict.fromkeys(
                    value
                    for prop in ("name", "alternateName")
                    for value in self._arcs.get(candidate, {}).get(prop, ())
                )
            )
            for candidate in dict.fromkeys(dcids)
        }


class FirstPageWithholdingGraphReader(FakeMintGraphReader):
    """A `FakeMintGraphReader` whose first reverse page, read at `CANDIDATE_CAP`, is truncated
    and leaves out `withheld`, as the live first page leaves out a root that sorts past it.
    Any other read, the member walk's full read among them, returns every match, truncated
    only past its own `max_results`."""

    def __init__(self, *, withheld: frozenset[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._withheld = withheld

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> _FakeSubjectPage:
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        if max_results != CANDIDATE_CAP:
            return page
        return _FakeSubjectPage(
            dcids=tuple(d for d in page.dcids if d not in self._withheld), truncated=True
        )


class BlockingFakeMintGraphReader:
    """Blocks `node_arc_values` until `release`; delegates other reads to the wrapped reader."""

    def __init__(self, delegate: FakeMintGraphReader) -> None:
        self._delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._delegate.generation_stamp()

    def un_availability(self) -> None:
        return None

    def node_label(self, dcid: str) -> str | None:
        return self._delegate.node_label(dcid)

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return self._delegate.node_labels_batch(dcids)

    def variable_facets_batch(
        self, *, stat_var: str, entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, list[Facet]]:
        return self._delegate.variable_facets_batch(
            stat_var=stat_var, entities=entities, deadline=deadline
        )

    def observation_facets_batch_many(
        self, *, stat_vars: Sequence[str], entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, list[Facet]]]:
        return self._delegate.observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._delegate.child_places_batch(parent_dcids, child_type, deadline=deadline)

    def entities_with_data_by_variable(
        self,
        variable_dcids: Sequence[str],
        entity_dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        return self._delegate.entities_with_data_by_variable(
            variable_dcids, entity_dcids, deadline=deadline
        )

    def declares_facet_defaults(self) -> bool:
        return self._delegate.declares_facet_defaults()

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        return self._delegate.default_facet_policy(stat_var=stat_var, axis=axis)

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        self.entered.set()
        self.release.wait(timeout=5.0)
        return self._delegate.node_arc_values(dcid, properties=properties)

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> _FakeSubjectPage:
        return self._delegate.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return self._delegate.node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        return self._delegate.node_names_many(dcids)
