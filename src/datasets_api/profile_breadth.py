"""The whole-variable breadth reader a Datasets process binds around every serve that elects a
facet (`current.serve_current`), read from the process's shared facet profile cache
(`materialise.shared_profiles`), which Search and Graph share when they are mounted beside it and
name the same roots.

`deps.open_deps` acquires it with the surface's live graph and releases it after the executors
have drained. The operations worker is a process of its own and acquires the cache in its own
lifespan the same way.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack

from datasets_api.config import DatasetsBuildSettings, DatasetsSettings
from graph_generation import GenerationProvider
from materialise.facets import VariableBreadthReader
from materialise.shared_profiles import (
    ProfileSource,
    acquire_shared_profiles,
    release_shared_profiles,
)
from series_coverage.shared import ServedCoverage


def open_profile_breadth(
    stack: AsyncExitStack,
    *,
    settings: DatasetsBuildSettings,
    live_settings: DatasetsSettings,
    generation: GenerationProvider,
    coverage: ServedCoverage,
) -> VariableBreadthReader:
    """A reader over the process's profile cache, acquired over the Datasets graph and SDMX
    roots and released off the event loop when `stack` closes."""
    shared = acquire_shared_profiles(
        source=ProfileSource(
            graph_base_url=live_settings.graph_base_url,
            sdmx_base_url=live_settings.sdmx_base_url,
            un_base_url=live_settings.un_graph_base_url,
            token=settings.graph_token or None,
        ),
        generation=generation,
    )
    stack.push_async_callback(asyncio.to_thread, release_shared_profiles, shared)
    return shared.breadth(coverage.snapshot)
