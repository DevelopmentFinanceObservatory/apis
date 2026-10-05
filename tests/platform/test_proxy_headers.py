"""Checks that `X-Forwarded-Proto` sets the scheme for trailing-slash redirects."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.testclient import TestClient

from dfo_platform.app import Surface, create_app


def _make_surface() -> Surface:
    router = APIRouter()

    @router.get("/api/probe/ping")
    async def ping():
        return {"ok": True}

    return Surface(
        name="probe",
        prefix="/api/probe",
        router=router,
        capability_version="v1",
        codes={"probe_code": ("Probe", "probe detail")},
    )


def test_trailing_slash_redirect_preserves_forwarded_https():
    app = create_app([_make_surface()])
    client = TestClient(app, follow_redirects=False)
    resp = client.get(
        "/api/probe/ping/",
        headers={"x-forwarded-proto": "https"},
    )
    assert resp.status_code == 307
    assert resp.headers["location"] == "https://testserver/api/probe/ping"


def test_trailing_slash_redirect_without_forwarded_header_stays_http():
    """Pins that the scheme comes from the forwarded header rather than the
    middleware inventing https on its own.
    """
    app = create_app([_make_surface()])
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/api/probe/ping/")
    assert resp.status_code == 307
    assert resp.headers["location"] == "http://testserver/api/probe/ping"
