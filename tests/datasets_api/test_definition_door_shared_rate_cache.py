"""The definition door shares an injected rate cache with the mint door and creates a cache
when none is supplied. Tests exercise the retrieval route directly.
"""

from __future__ import annotations

import datasets_api.serve as serve_module
from materialise.rate_tables import RateTableCache

from ._artifact_doubles import FakeArtifactStore
from .test_mint_spec_id import _STATVAR_SET_BODY, _client


def test_datasets_surface_shares_one_injected_rate_cache_across_both_doors(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_group = serve_module.materialise_group

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", spy)

    injected = RateTableCache()
    client = _client(artifact_store=FakeArtifactStore(), rate_cache=injected)

    # Mint door: registers and materialises a fresh spec.
    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    # The GET re-materialises because no current head is recorded.
    resp = client.get(f"/api/datasets/{spec_id}")
    assert resp.status_code == 200, resp.text

    assert len(recorded) == 2
    assert recorded[0] is injected
    assert recorded[1] is injected


def test_definition_door_serves_with_no_rate_cache_injected_by_building_its_own(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_group = serve_module.materialise_group

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", spy)

    client = _client(artifact_store=FakeArtifactStore())

    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.get(f"/api/datasets/{spec_id}")
    assert resp.status_code == 200, resp.text

    assert len(recorded) == 2
    assert recorded[1] is not None
    assert isinstance(recorded[1], RateTableCache)
