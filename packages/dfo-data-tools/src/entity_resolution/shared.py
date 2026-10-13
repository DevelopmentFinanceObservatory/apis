"""Process-wide resolvekit resolver, worker and pack cache shared across surfaces, and the group
registry and membership policy, which live for the whole process.

The registry and policy are read once from resolvekit's bundled geo modules
(`GEO_MODULE_IDS`), through a resolver opened for the read and closed after it, and kept for
the life of the process. A caller that holds no shared resolver (a rendered sentence, an
options read, a resubmit, a surface built with an injected resolver) reads them without
building or closing one, and every shared resolver built in the process carries the same
snapshot."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from resolvekit import Resolver
from resolvekit.core.api import RoutingMode
from resolvekit.core.module_registry import register_module

from dfo_platform.singleton import RefCountedSingleton
from entity_resolution import pack
from entity_resolution.groups import GroupRegistry, build_group_registry
from entity_resolution.membership import MembershipPolicy, build_policy

GEO_MODULE_IDS: tuple[str, ...] = (
    "geo.countries",
    "geo.regions",
    "geo.continents",
    "geo.continental_unions",
)
"""Resolvekit's bundled geo modules, which hold every group's and region's membership."""

MODULE_IDS: tuple[str, ...] = (*GEO_MODULE_IDS, pack.MODULE_ID)

_CACHE_DIR_ENV = "RESOLVEKIT_CACHE_DIR"
_OFFLINE_ENV = "RESOLVEKIT_OFFLINE"

T = TypeVar("T")


class _ResolverEnv(BaseSettings):
    """`acquire_shared`'s own env read: the pack cache directory, when the
    caller doesn't pass one explicitly.
    """

    model_config = SettingsConfigDict()

    cache_dir: str | None = Field(default=None, alias=_CACHE_DIR_ENV)


@dataclass(frozen=True, slots=True)
class Memberships:
    """The group registry and the membership policy over it, immutable snapshots any thread
    can read."""

    groups: GroupRegistry
    policy: MembershipPolicy


def _build_memberships() -> Memberships:
    os.environ[_OFFLINE_ENV] = "1"
    resolver = Resolver.lite(
        module_ids=list(GEO_MODULE_IDS),
        routing_mode=RoutingMode.EXPLICIT,
        cache_size=0,
        warm=False,
    )
    # Resolvekit's stores are thread-affine, so the builders read on the thread that opened it.
    try:
        groups = build_group_registry(resolver)
        return Memberships(groups=groups, policy=build_policy(resolver, groups=groups))
    finally:
        resolver.close()


_memberships: Memberships | None = None
_memberships_lock = threading.Lock()


def shared_memberships() -> Memberships:
    """The process's group registry and membership policy, read on the first call and kept
    for the life of the process."""
    global _memberships
    with _memberships_lock:
        if _memberships is None:
            _memberships = _build_memberships()
        return _memberships


def shared_membership_policy() -> MembershipPolicy:
    """The process's membership policy (`shared_memberships`), for any caller, including one
    that holds no resolver: a resubmit, an options read, a rendered sentence."""
    return shared_memberships().policy


class SharedResolver:
    """Composed resolver with a dedicated single-worker executor, its pack cache directory and
    the exact index of the `dfo_graph` pack's names. `groups` and `policy` are the process's
    (`shared_memberships`).

    Resolvekit's query cache requires serial calls. Use `acquire_shared` and `release_shared`
    for its reference-counted lifecycle."""

    def __init__(
        self,
        *,
        resolver: Resolver,
        executor: ThreadPoolExecutor,
        cache_dir: Path,
        alias_index: pack.AliasIndex,
    ) -> None:
        self.resolver = resolver
        self.executor = executor
        self.cache_dir = cache_dir
        self.alias_index = alias_index

    @property
    def groups(self) -> GroupRegistry:
        return shared_memberships().groups

    @property
    def policy(self) -> MembershipPolicy:
        return shared_memberships().policy

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
    # Read the memberships with the resolver, so no request is the one that reads them.
    shared_memberships()
    os.environ[_OFFLINE_ENV] = "1"
    resolved.mkdir(parents=True, exist_ok=True)
    records_path, curated_path = pack.records_path(), pack.curated_path()
    pack_dir = pack.build_pack(
        records_path=records_path, curated_path=curated_path, out_dir=resolved
    )
    alias_index = pack.build_alias_index(records_path=records_path, curated_path=curated_path)
    register_module(pack.MODULE_ID, pack_dir)
    resolver = Resolver.lite(
        module_ids=list(MODULE_IDS), routing_mode=RoutingMode.EXPLICIT, cache_size=0
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resolvekit")
    return SharedResolver(
        resolver=resolver, executor=executor, cache_dir=resolved, alias_index=alias_index
    )


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
