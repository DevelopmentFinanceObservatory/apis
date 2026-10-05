"""Build `GraphGeneration` from environment settings. Fixed token overrides Cloud Run.
Every stamp carries the build id of the origin catalogue the calling request is bound to.

The shared singleton calls this builder. Tests inject a source into `GraphGeneration` directly."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx

from graph_generation import config
from graph_generation.protocol import CatalogueBuilds
from graph_generation.provider import GraphGeneration
from graph_generation.source import (
    CloudRunGenerationSource,
    Credentials,
    FixedGenerationSource,
    GenerationSource,
)

logger = logging.getLogger(__name__)


def build_from_env(
    *,
    catalogue: CatalogueBuilds,
    expected_graph_base: str | None = None,
    transport: httpx.BaseTransport | None = None,
    credentials_factory: Callable[[], Credentials] | None = None,
) -> GraphGeneration:
    source: GenerationSource | None
    if config.DFO_GRAPH_GENERATION:
        source = FixedGenerationSource(token=config.DFO_GRAPH_GENERATION)
    elif config.DFO_GRAPH_GENERATION_SERVICE:
        if credentials_factory is not None:
            source = CloudRunGenerationSource(
                service=config.DFO_GRAPH_GENERATION_SERVICE,
                transport=transport,
                credentials_factory=credentials_factory,
            )
        else:
            source = CloudRunGenerationSource(
                service=config.DFO_GRAPH_GENERATION_SERVICE, transport=transport
            )
    else:
        logger.warning(
            "graph generation: neither DFO_GRAPH_GENERATION nor "
            "DFO_GRAPH_GENERATION_SERVICE is set; every graph-derived cache runs under "
            "the unknown generation (1h lifetime) for this process's whole lifetime"
        )
        source = None
    return GraphGeneration(
        source=source, catalogue=catalogue, expected_graph_base=expected_graph_base
    )
