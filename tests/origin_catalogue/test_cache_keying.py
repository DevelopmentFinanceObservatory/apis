"""Catalogue-dependent caches key on the build the request is bound to, and a swap clears them."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dfo_platform.single_flight_cache import SingleFlightCache
from graph_generation import FixedGenerationSource, GraphGeneration
from origin_catalogue.context import bound_catalogue
from origin_catalogue.service import CatalogueService
from tests.origin_catalogue.support import publish_sample


def _service(tmp_path: Path) -> tuple[CatalogueService, str]:
    uri = f"file://{tmp_path / 'catalogue'}"
    publish_sample(uri)
    return CatalogueService(artifact_uri=uri, recheck_seconds=0), uri


@pytest.mark.parametrize("source", [FixedGenerationSource(token="rev-1"), None])
def test_requests_bound_to_different_builds_never_share_an_entry_across_a_swap(
    tmp_path: Path, source: FixedGenerationSource | None
) -> None:
    service, uri = _service(tmp_path)
    generation = GraphGeneration(source=source, catalogue=service)
    generation.refresh_now()
    cache: SingleFlightCache[str, str] = SingleFlightCache(
        maxsize=8,
        flight_timeout_s=5.0,
        executor=ThreadPoolExecutor(max_workers=1),
        generation=generation,
    )
    generation.register("answers", stats=dict, clear=cache.clear)

    def answer_under_a_then_swap(_deadline: float) -> str:
        publish_sample(uri, build_id="build-b")
        assert service.snapshot_current().build_id == "build-b"
        return "admitted by build-a"

    with bound_catalogue(service.snapshot()) as snapshot_a:
        assert snapshot_a.build_id == "build-a"
        assert cache.get_or_fetch("gdp kenya", answer_under_a_then_swap) == "admitted by build-a"

    with bound_catalogue(service.snapshot()) as snapshot_b:
        assert snapshot_b.build_id == "build-b"
        assert cache.get_or_fetch("gdp kenya", lambda _deadline: "admitted by build-b") == (
            "admitted by build-b"
        )

    with bound_catalogue(snapshot_a):
        assert cache.get_or_fetch("gdp kenya", lambda _deadline: "refetched under build-a") == (
            "admitted by build-a"
        )


def test_a_swap_clears_the_registered_caches(tmp_path: Path) -> None:
    service, uri = _service(tmp_path)
    generation = GraphGeneration(source=FixedGenerationSource(token="rev-1"), catalogue=service)
    generation.refresh_now()
    cleared: list[str] = []
    generation.register("answers", stats=dict, clear=lambda: cleared.append("answers"))

    service.snapshot_current()
    assert cleared == []
    publish_sample(uri, build_id="build-b")
    service.snapshot_current()
    assert cleared == ["answers"]


def test_a_stamp_keys_on_the_bound_build_and_on_the_loaded_one_when_unbound(
    tmp_path: Path,
) -> None:
    service, uri = _service(tmp_path)
    generation = GraphGeneration(source=FixedGenerationSource(token="rev-1"), catalogue=service)
    generation.refresh_now()
    snapshot_a = service.snapshot()

    def unbound_key() -> str:
        # A fresh context binds nothing, as a pool thread without a copied context does.
        return contextvars.Context().run(lambda: generation.stamp().key)

    assert unbound_key() == "rev-1@build-a"
    publish_sample(uri, build_id="build-b")
    service.snapshot_current()
    assert unbound_key() == "rev-1@build-b"
    with bound_catalogue(snapshot_a):
        assert generation.stamp().key == "rev-1@build-a"
