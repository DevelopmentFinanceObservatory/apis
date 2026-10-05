"""Dependencies shared by the Graph doors and published during the surface lifespan.

The node, sources, and resolve doors share the graph reader, bounded DCP
executor, and semaphore. Resolve also uses the resolver and generation
provider. Resolvekit owns a separate single-worker executor. The sources
door reads currency from the coverage service when one is configured.

This module holds the bundle to keep door imports separate from the
composition root, which imports every door.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends

from dfo_platform.state import StateSlot
from entity_resolution import EntityResolver
from graph_api.reader import GraphNodeReader
from graph_generation import GenerationProvider

if TYPE_CHECKING:
    # `series_coverage.shared` needs numpy, which the `graph` extra does not install.
    from series_coverage.shared import CoverageService


class GraphDepsNotSet(Exception):
    """A request arrived before lifespan setup. The surface classifies this as retryable 503."""

    def __init__(self) -> None:
        super().__init__("Graph dependencies are not set: the surface's lifespan has not run")


@dataclass(frozen=True, slots=True)
class GraphDeps:
    graph: GraphNodeReader
    resolver: EntityResolver
    dcp_executor: ThreadPoolExecutor
    dcp_semaphore: threading.BoundedSemaphore
    generation: GenerationProvider
    # `None` when `COVERAGE_ARTIFACT_URI` is unset or the artifact failed to load.
    coverage: CoverageService | None = None


# A distinct slot name keeps composed surfaces separate. Missing dependencies map to 503.
GRAPH_DEPS_SLOT: StateSlot[GraphDeps] = StateSlot("graph_deps", unset_exception=GraphDepsNotSet)

# `Annotated` avoids a `Depends(...)` call in a default argument (ruff B008).
GraphDepsDep = Annotated[GraphDeps, Depends(GRAPH_DEPS_SLOT)]
