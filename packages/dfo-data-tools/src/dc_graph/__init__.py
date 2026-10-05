"""Shared client for Data Commons `/v2/node`, `/v2/observation` and `/v2/resolve` reads.

`GraphClient` serves `materialise` and `graph_api` directly. Search adds an adapter for its own
result types. This module stays independent of Search so all consumers can share the client.
"""

from __future__ import annotations

from dc_graph.client import (
    CURRENCY_UNIT_OF_MEASURE_ISO_CODES,
    IMPORT_ADDED_CONSTRAINT_PROPERTIES,
    NODE_PATH,
    UNRESOLVED_NODE,
    Facet,
    FacetDefaultPolicy,
    GraphClient,
    ProvenanceRefresh,
    SubjectPage,
    SubjectsCoverageFlag,
    declared_role_refresh_deadline,
    earliest_date,
    is_front_door_refusal,
    latest_date,
    parse_arcs,
    reject_unsafe_filter_token,
    reject_unsafe_property_token,
)
from dc_graph.errors import (
    CapErrorFactory,
    GraphCapExceededError,
    GraphDeadlineExceededError,
    GraphNotAdvancingError,
    GraphReadError,
    ReadErrorFactory,
)

__all__ = [
    "CURRENCY_UNIT_OF_MEASURE_ISO_CODES",
    "IMPORT_ADDED_CONSTRAINT_PROPERTIES",
    "NODE_PATH",
    "UNRESOLVED_NODE",
    "CapErrorFactory",
    "Facet",
    "FacetDefaultPolicy",
    "GraphCapExceededError",
    "GraphClient",
    "GraphDeadlineExceededError",
    "GraphNotAdvancingError",
    "GraphReadError",
    "ProvenanceRefresh",
    "ReadErrorFactory",
    "SubjectPage",
    "SubjectsCoverageFlag",
    "declared_role_refresh_deadline",
    "earliest_date",
    "is_front_door_refusal",
    "latest_date",
    "parse_arcs",
    "reject_unsafe_filter_token",
    "reject_unsafe_property_token",
]
