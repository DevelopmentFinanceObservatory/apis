"""Each door reuses one `RateTableCache` across materialisations and passes it to the
materialise entry point. Spies check cache identity for definition, collection, and preview
calls.
"""

from __future__ import annotations

import datasets_api.preview as preview_module
import datasets_api.serve as serve_module
from materialise.rate_tables import RateTableCache

from .test_datasets_preview import _client as _preview_client
from .test_datasets_preview import _preview_body
from .test_mint_collection_id import _client as _collection_client
from .test_mint_collection_id import _collection_body, _member_sidecar, _register_collection
from .test_mint_collection_id import _graph as _collection_graph
from .test_mint_spec_id import _STATVAR_SET_BODY, _spec_id_body
from .test_mint_spec_id import _client as _spec_client


def test_definition_door_shares_one_rate_cache_across_two_materialisations(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_group = serve_module.materialise_group

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_group(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_group", spy)

    from ._artifact_doubles import FakeArtifactStore

    client = _spec_client(artifact_store=FakeArtifactStore())

    base = client.post("/api/datasets", json=_STATVAR_SET_BODY)
    assert base.status_code == 200, base.text
    spec_id = base.json()["definition"]["spec_id"]

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id))
    assert resp.status_code == 200, resp.text

    assert len(recorded) == 2
    assert recorded[0] is not None
    assert isinstance(recorded[0], RateTableCache)
    assert recorded[0] is recorded[1]


def test_collection_door_shares_one_rate_cache_across_two_materialisations(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_collection = serve_module.materialise_collection

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_collection(*args, **kwargs)

    monkeypatch.setattr(serve_module, "materialise_collection", spy)

    from ._artifact_doubles import FakeArtifactStore
    from .conftest import FakeDatasetStore
    from .test_mint_collection_id import _STAT_VAR_REF

    store = FakeDatasetStore()
    graph = _collection_graph()
    client = _collection_client(artifact_store=FakeArtifactStore(), store=store, graph=graph)
    _spec_ken, def_ken = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    _spec_uga, def_uga = _member_sidecar(
        _STAT_VAR_REF, entities={"recipient": ["country/UGA"]}, graph=graph
    )
    collection_id = _register_collection(client, def_ken, def_uga)
    body = _collection_body(collection_id)

    first = client.post("/api/datasets", json=body)
    assert first.status_code == 200, first.text
    second = client.post("/api/datasets", json=body)
    assert second.status_code == 200, second.text

    assert len(recorded) == 2
    assert recorded[0] is not None
    assert isinstance(recorded[0], RateTableCache)
    assert recorded[0] is recorded[1]


def test_preview_passes_the_doors_rate_cache(monkeypatch):
    recorded: list[RateTableCache | None] = []
    real_materialise_preview = preview_module.materialise_preview

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("rate_cache"))
        return real_materialise_preview(*args, **kwargs)

    monkeypatch.setattr(preview_module, "materialise_preview", spy)

    client, _artifact_store = _preview_client()
    from .test_datasets_preview import _SCOPE_DICT, _STAT_VAR_REF

    resp = client.post(
        "/api/datasets",
        json=_preview_body(
            {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE_DICT}, limit=10
        ),
    )
    assert resp.status_code == 200, resp.text

    assert len(recorded) == 1
    assert recorded[0] is not None
    assert isinstance(recorded[0], RateTableCache)
