"""Process-wide resolvekit resolver, worker, and pack cache shared across surfaces."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from resolvekit import Resolver
from resolvekit.core.api import RoutingMode
from resolvekit.core.module_registry import register_module

from dfo_platform.singleton import RefCountedSingleton
from entity_resolution import pack

MODULE_IDS: tuple[str, ...] = (
    "geo.countries",
    "geo.regions",
    "geo.continents",
    "geo.continental_unions",
    pack.MODULE_ID,
)

_CACHE_DIR_ENV = "RESOLVEKIT_CACHE_DIR"
_OFFLINE_ENV = "RESOLVEKIT_OFFLINE"

T = TypeVar("T")


class _ResolverEnv(BaseSettings):
    """`acquire_shared`'s own env read: the pack cache directory, when the
    caller doesn't pass one explicitly.
    """

    model_config = SettingsConfigDict()

    cache_dir: str | None = Field(default=None, alias=_CACHE_DIR_ENV)


class SharedResolver:
    """Composed resolver with a dedicated single-worker executor and its pack cache directory.

    Resolvekit's query cache requires serial calls. Use `acquire_shared` and `release_shared`
    for its reference-counted lifecycle."""

    def __init__(
        self, *, resolver: Resolver, executor: ThreadPoolExecutor, cache_dir: Path
    ) -> None:
        self.resolver = resolver
        self.executor = executor
        self.cache_dir = cache_dir

    def run(self, fn: Callable[..., T], *args: object) -> Future[T]:
        """Submit `fn(*args)` to the single worker and return its future."""
        return self.executor.submit(fn, *args)

    def close(self) -> None:
        # Drain queued and running jobs before closing the resolver they use.
        self.executor.shutdown(wait=True)
        self.resolver.close()


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir
    env_value = _ResolverEnv().cache_dir
    if not env_value:
        raise RuntimeError(
            f"acquire_shared() needs a cache directory: pass cache_dir, or set "
            f"the {_CACHE_DIR_ENV} environment variable"
        )
    return Path(env_value)


def _build(cache_dir: Path | None) -> SharedResolver:
    resolved = _resolve_cache_dir(cache_dir)
    os.environ[_OFFLINE_ENV] = "1"
    resolved.mkdir(parents=True, exist_ok=True)
    pack_dir = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=resolved
    )
    register_module(pack.MODULE_ID, pack_dir)
    resolver = Resolver.lite(
        module_ids=list(MODULE_IDS), routing_mode=RoutingMode.EXPLICIT, cache_size=0
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resolvekit")
    return SharedResolver(resolver=resolver, executor=executor, cache_dir=resolved)


_singleton: RefCountedSingleton[SharedResolver] = RefCountedSingleton(
    build=_build, close=SharedResolver.close, label="resolver"
)


def acquire_shared(*, cache_dir: Path | None = None) -> SharedResolver:
    """Acquire the shared resolver, building it once and releasing it after the last user.

    Pass `cache_dir` or set `RESOLVEKIT_CACHE_DIR`. One is required."""
    return _singleton.acquire(cache_dir)


def release_shared(shared: SharedResolver) -> None:
    """Release one reference. Close and discard the instance when the count reaches zero."""
    _singleton.release(shared)
