"""Tests that the mint door uses an injected rate cache or creates a private cache by default."""

from __future__ import annotations

import datasets_api.serve as serve_module
from materialise.rate_tables import RateTableCache

from ._artifact_doubles import FakeArtifactStore
from .conftest import FakeDatasetStore
from .test_mint_spec_id import _STATVAR_SET_BODY, _client


def test_mint_door_serves_with_an_injected_rate_cache_and_shares_that_instance(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_group = serve_module.materialise_group

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", spy)

    injected = RateTableCache()
    client = _client(
        artifact_store=FakeArtifactStore(), store=FakeDatasetStore(), rate_cache=injected
    )

    resp = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert resp.status_code == 200, resp.text

    assert len(recorded) == 1
    assert recorded[0] is injected


def test_mint_door_serves_with_no_rate_cache_injected_by_building_its_own(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_group = serve_module.materialise_group

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", spy)

    client = _client(artifact_store=FakeArtifactStore(), store=FakeDatasetStore())

    resp = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert resp.status_code == 200, resp.text

    assert len(recorded) == 1
    assert recorded[0] is not None
    assert isinstance(recorded[0], RateTableCache)
