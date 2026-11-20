"""The mint, options, and snapshot routes return `no-store` headers, and the definition GET
answers 200 with `no-cache`, its freshness and content version, and a strong ETag. The test
names these artifact-bearing routes explicitly, so new routes need to be added. Snapshot reads
use the shared `version_store` fixture.
"""

from __future__ import annotations

from datasets_api.headers import NO_STORE_HEADERS

from .test_definition_door import _SCOPE, _STAT_VAR_REF, _client, _graph


def test_every_artifact_bearing_route_carries_its_cache_control(version_store):
    client = _client(graph=_graph(), version_store=version_store)
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
        "options": {"bundle": False},
    }

    mint_resp = client.post("/api/datasets", json=body)
    assert mint_resp.status_code == 200, mint_resp.text
    manifest = mint_resp.json()
    spec_id = manifest["definition"]["spec_id"]
    content_version = manifest["content_version"]

    definition_resp = client.get(
        f"/api/datasets/{spec_id}?bundle=false", headers={"accept-encoding": "identity"}
    )
    options_resp = client.get(f"/api/datasets/{spec_id}/options")
    snapshot_resp = client.get(f"/api/datasets/{spec_id}/{content_version}")

    no_store = {
        "POST /api/datasets": mint_resp,
        "GET /api/datasets/{id}/options": options_resp,
        "GET /api/datasets/{id}/{content_version}": snapshot_resp,
    }
    for route, resp in no_store.items():
        assert resp.status_code == 200, f"{route}: {resp.text}"
        assert resp.headers["cache-control"] == "no-store", route
    assert definition_resp.status_code == 200, definition_resp.text
    headers = definition_resp.headers
    assert headers["cache-control"] == "no-cache"
    assert headers["x-dfo-freshness"] == "current"
    assert headers["x-dfo-content-version"] == content_version
    assert headers["etag"].startswith('"') and headers["etag"].endswith('"')


def test_no_store_headers_constant_is_the_single_source():
    assert NO_STORE_HEADERS == {"cache-control": "no-store"}
