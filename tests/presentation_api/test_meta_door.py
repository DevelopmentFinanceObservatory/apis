"""Checks the capability and catalogue versions served by the capabilities route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from assembler.catalogue import CatalogueRelease
from presentation_api.version import compute_presentation_capability_version


def test_capabilities_carries_the_real_computed_version(
    client: TestClient, release: CatalogueRelease
):
    resp = client.get("/api/presentation/capabilities")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"capability_version", "catalog_version", "build"}
    assert body["capability_version"] == compute_presentation_capability_version(release=release)
    assert body["capability_version"].startswith("pres_")
    assert body["catalog_version"] == release.catalog_version


def test_capabilities_route_is_in_the_openapi_paths(client: TestClient):
    resp = client.get("/openapi.json")

    assert resp.status_code == 200
    assert "/api/presentation/capabilities" in resp.json()["paths"]


def test_capabilities_response_is_cacheable(client: TestClient):
    first = client.get("/api/presentation/capabilities")
    second = client.get("/api/presentation/capabilities")

    assert first.status_code == 200
    assert "public" in first.headers["cache-control"]
    assert "max-age=300" in first.headers["cache-control"]
    assert first.headers.get("etag")
    assert first.headers["etag"] == second.headers["etag"]


def test_refusal_body_carries_the_same_capability_version_as_capabilities(client: TestClient):
    capabilities = client.get("/api/presentation/capabilities")
    refusal = client.get("/api/presentation/components/some-component-id")

    assert refusal.status_code == 400
    assert refusal.json()["capability_version"] == capabilities.json()["capability_version"]
