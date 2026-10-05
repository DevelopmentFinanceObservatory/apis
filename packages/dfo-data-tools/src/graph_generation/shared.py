"""Reference-counted process-wide `GraphGeneration`.

The first acquire reads the source synchronously. Lifespan callers use `asyncio.to_thread`.
Unconfigured or unreachable sources already degrade to unknown, so the singleton needs no close
callback."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from dfo_platform.singleton import RefCountedSingleton
from graph_generation.build import build_from_env
from graph_generation.protocol import CatalogueBuilds
from graph_generation.provider import GraphGeneration
from graph_generation.source import Credentials


def _build(
    *,
    catalogue: CatalogueBuilds,
    expected_graph_base: str | None,
    transport: httpx.BaseTransport | None,
    credentials_factory: Callable[[], Credentials] | None,
) -> GraphGeneration:
    instance = build_from_env(
        catalogue=catalogue,
        expected_graph_base=expected_graph_base,
        transport=transport,
        credentials_factory=credentials_factory,
    )
    instance.refresh_now()
    return instance


_singleton: RefCountedSingleton[GraphGeneration] = RefCountedSingleton(
    build=_build, label="provider"
)


def acquire_shared(
    *,
    catalogue: CatalogueBuilds,
    expected_graph_base: str | None = None,
    transport: httpx.BaseTransport | None = None,
    credentials_factory: Callable[[], Credentials] | None = None,
) -> GraphGeneration:
    """Acquire the process-wide provider, building it from the environment on first use.

    `catalogue` is the origin catalogue the process serves: every stamp carries the build id of
    the catalogue the calling request is bound to, and a change of the loaded build clears every
    registered cache. Only the first caller's injected arguments apply.
    Reference counting closes it after the last release. Build `GraphGeneration` directly for
    test-specific inputs."""
    return _singleton.acquire(
        catalogue=catalogue,
        expected_graph_base=expected_graph_base,
        transport=transport,
        credentials_factory=credentials_factory,
    )


def release_shared(shared: GraphGeneration) -> None:
    """Release a reference and discard the instance at zero.

    Raise `RuntimeError` for stale or extra releases."""
    _singleton.release(shared)
