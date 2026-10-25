"""Graph read protocols and re-exports for `materialise`.

`dc_graph.GraphClient` and `origin_catalogue.graph.OriginGraphClient`
implement `GraphReader`. This module re-exports the protocol's graph types
and errors for existing importers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from dc_graph import (
    UNRESOLVED_NODE,
    Facet,
    FacetDefaultPolicy,
    GraphDeadlineExceededError,
    GraphNotAdvancingError,
    GraphReadError,
    ProvenanceRefresh,
    SubjectPage,
    parse_arcs,
    reject_unsafe_property_token,
)
from graph_generation import GenerationStamp
from origin_catalogue.graph import UnAvailabilityClient

__all__ = [
    "UNRESOLVED_NODE",
    "Facet",
    "FacetDefaultPolicy",
    "GraphDeadlineExceededError",
    "GraphNotAdvancingError",
    "GraphReadError",
    "GraphReader",
    "ProvenanceRefresh",
    "SubjectPage",
    "UnObservationReader",
    "parse_arcs",
    "reject_unsafe_property_token",
]


class UnObservationReader(Protocol):
    """The origin graph client's dated, admitted facet read, which serves UN System Data
    Commons rows (`materialise.observation_rows`)."""

    def observation_facets_batch(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]: ...


class GraphReader(Protocol):
    """`SdmxDataResult.roles` supplies observation roles."""

    def un_observation_reader(self) -> UnObservationReader | None:
        """Returns the read that serves UN System Data Commons rows, or `None` when this
        reader cannot reach the UN deployment.

        A reader that wraps another returns the wrapped reader's, so wrapping never
        drops the capability.
        """
        ...

    def generation_stamp(self) -> GenerationStamp:
        """Returns the stamp the reader's caches key on, without I/O. A reader that wraps
        another returns the wrapped reader's."""
        ...

    def source_key(self) -> str:
        """Returns the backends the reader reads, equal for two readers exactly when they read
        the same backends. A reader that wraps another returns the wrapped reader's."""
        ...

    def un_availability(self) -> UnAvailabilityClient | None:
        """Returns the UN deployment's SDMX `/availability` read, which places a UN variable
        for a request that names no place (`materialise.place_scope`), or `None` when this
        reader cannot reach the UN deployment. A reader that wraps another returns the
        wrapped reader's."""
        ...

    def node_label(self, dcid: str) -> str | None:
        """Returns `None` for a 200 response with an absent or empty node.

        Transport errors raise `GraphReadError`.
        """
        ...

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        """Returns each labelled node's label, leaving out absent and unlabelled nodes.

        Transport errors raise `GraphReadError`.
        """
        ...

    def variable_facets_batch(
        self,
        *,
        stat_var: str,
        entities: Sequence[str],
        needs_dates: bool = False,
        deadline: float | None = None,
    ) -> dict[str, list[Facet]]:
        """Returns facets per entity from one POST. Empty entities return `{}`.

        An entity without observations maps to `[]`. `needs_dates` adds each facet's
        per-observation dates and values (`Facet.dates`); without it they are empty.
        Transport errors raise `GraphReadError`. An expired absolute monotonic `deadline`
        raises `GraphDeadlineExceededError` before the request. Otherwise the absolute
        deadline bounds the httpx timeout.
        """
        ...

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        """Returns facets for each StatVar/entity pair in one POST.

        Pairs without observations map to `[]`. Empty `stat_vars` or `entities`
        returns `{}`. `needs_dates` follows `variable_facets_batch`.
        Transport errors raise `GraphReadError`. `deadline` follows
        `variable_facets_batch`'s absolute-time contract.
        """
        ...

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Returns each parent's contained places of `child_type`, in graph order.

        A parent with none maps to `[]`. Transport errors raise `GraphReadError`.
        `deadline` follows `variable_facets_batch`'s absolute-time contract.
        """
        ...

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Returns, per StatVar, the entities where it has admitted data, in `entity_dcids`
        order. Empty inputs return `{}` without a request. Transport errors raise
        `GraphReadError`. `deadline` follows `variable_facets_batch`'s absolute-time
        contract."""
        ...

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        """Returns `None` when the graph declares no default for the axis."""
        ...

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        """Returns each requested property in one `/v2/node` call.

        Values are distinct and keep first-occurrence order. Missing arcs map
        to `()`, and empty `properties` returns `{}`. Transport errors raise
        `GraphReadError`.
        """
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
        """Returns distinct subjects in first-occurrence order. `filters` narrows them
        to nodes carrying the given value-bound arcs.

        A cursor remaining at the page bound raises `GraphReadError`. A stuck
        cursor raises `GraphNotAdvancingError`. `deadline` is one absolute
        instant bounding every page's httpx timeout.
        """
        ...

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """See `dc_graph.GraphClient.node_arcs_many` for the full wire
        contract this signature mirrors, including its per-property
        uniqueness guarantee and its `deadline` contract.
        """
        ...

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        """See `dc_graph.GraphClient.node_names_many` for the full wire
        contract this signature mirrors, including its uniqueness guarantee
        and its `deadline` contract.
        """
        ...
