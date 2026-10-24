"""Shared client for Data Commons `/v2/node`, `/v2/observation` and `/v2/resolve` reads.

`GraphClient` serves `materialise` and `graph_api` directly. Search adds an adapter for its own
result types. This module stays independent of Search so all consumers can share the client.
"""

from __future__ import annotations

from dc_graph.client import (
    CURRENCY_UNIT_OF_MEASURE_ISO_CODES,
    NODE_PATH,
    UNRESOLVED_NODE,
    Facet,
    FacetDefaultPolicy,
    FrontDoorRefused,
    GraphClient,
    ProvenanceRefresh,
    SubjectPage,
    SubjectsCoverageFlag,
    declared_role_refresh_deadline,
    display_name,
    earliest_date,
    is_front_door_refusal,
    label_or_self_name,
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
from dc_graph.visible_arcs import (
    IMPORT_ADDED_CONSTRAINT_PROPERTIES,
    MIXER_BLOCKED_PREDICATES,
    dfo_visible,
    mixer_visible,
    visible_mask,
)

__all__ = [
    "CURRENCY_UNIT_OF_MEASURE_ISO_CODES",
    "IMPORT_ADDED_CONSTRAINT_PROPERTIES",
    "MIXER_BLOCKED_PREDICATES",
    "NODE_PATH",
    "UNRESOLVED_NODE",
    "CapErrorFactory",
    "Facet",
    "FacetDefaultPolicy",
    "FrontDoorRefused",
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
    "dfo_visible",
    "display_name",
    "earliest_date",
    "is_front_door_refusal",
    "label_or_self_name",
    "latest_date",
    "mixer_visible",
    "parse_arcs",
    "reject_unsafe_filter_token",
    "reject_unsafe_property_token",
    "visible_mask",
]
