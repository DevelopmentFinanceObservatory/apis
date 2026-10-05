"""Tests for the health door: `GET /api/graph/healthz`."""

from __future__ import annotations

from dfo_platform.build import build_commit

from .conftest import make_client


def test_healthz_returns_ok_and_capability_version():
    client = make_client()

    resp = client.get("/api/graph/healthz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["capability_version"], str) and body["capability_version"]


def test_healthz_carries_the_shared_build_field():
    client = make_client()

    resp = client.get("/api/graph/healthz")

    assert resp.status_code == 200
    assert resp.json()["build"] == build_commit()
