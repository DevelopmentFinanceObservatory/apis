"""The process-wide graph generation the datasets graph clients key their caches on.

Search and the datasets doors acquire the same `graph_generation` provider, so a stamp key, and
with it a default-scope memo entry (`materialise.place_scope`), agrees across them in one process.
"""

from __future__ import annotations

import asyncio
import functools
from contextlib import AsyncExitStack

from graph_generation import GenerationProvider, acquire_shared, release_shared
from materialise.place_scope import clear_memo, memo_stats
from origin_catalogue.service import CatalogueService, get_catalogue_service

PLACE_SCOPE_MEMO_NAME = "datasets.default_place_scope"
"""The memo's registration name on the datasets surface. The worker app registers its own."""


async def enter_shared_generation(
    stack: AsyncExitStack, *, catalogue: CatalogueService | None, graph_base_url: str
) -> GenerationProvider:
    """Acquire the shared provider and release it when `stack` closes.

    The first acquire reads the generation source synchronously, so it runs off the event loop.
    `catalogue` defaults to the process-wide service."""
    generation = await asyncio.to_thread(
        functools.partial(
            acquire_shared,
            catalogue=catalogue if catalogue is not None else get_catalogue_service(),
            expected_graph_base=graph_base_url,
        )
    )
    stack.callback(release_shared, generation)
    return generation


def register_place_scope_memo(
    stack: AsyncExitStack, generation: GenerationProvider, *, name: str
) -> None:
    """Register the default-scope memo on `generation` until `stack` closes."""
    handle = generation.register(name, stats=memo_stats, clear=clear_memo)
    stack.callback(handle.close)
