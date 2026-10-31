"""A bare spec id resolves to its dataset version from the records alone, so a read that decodes
no stored frame serves however large or old the stored frame is, and a read that decodes it
refuses by the bound its own surface applies."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.datasets_api._frame_doubles import link_frame1_only
from tests.live_refresh.composed_app import (
    DATASET,
    FRAME_OPTIONS,
    STARTUP_ENV,
    ComposedApp,
    compose,
)

# Past the 216,600,000 bytes a read that decodes a frame1 document alone takes at the default
# fold ceiling (`datasets_api.render.frame1_decode_max_bytes`).
_OVERSIZED = 250_000_000
_CALLER = {"x-forwarded-for": "198.51.100.1, 203.0.113.9"}


@pytest.fixture
def composed(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ComposedApp, TestClient]]:
    for name, value in STARTUP_ENV.items():
        monkeypatch.setenv(name, value)
    # The export door names the caller its copy counts against by this key.
    monkeypatch.setenv("DFO_CALLER_HMAC_KEY", "test-caller-key")
    composed = compose()
    with TestClient(composed.app, raise_server_exceptions=False) as client:
        yield composed, client


def _mint_with_an_oversized_frame1(composed: ComposedApp, client: TestClient) -> str:
    """Mint `DATASET` in the rendering a bare spec id reads and as a csv file, swap its
    canonical frame for the frame1 document of its rows, recorded at `_OVERSIZED` bytes, and
    return its content version."""
    versions = set()
    for options in (FRAME_OPTIONS, {"format": "csv", "bundle": False}):
        resp = client.post("/api/datasets", json=composed.mint_body(options))
        assert resp.status_code == 200, resp.text
        versions.add(resp.json()["content_version"])
    (content_version,) = versions
    digest = link_frame1_only(
        composed.version_store,
        composed.artifact_store,
        subject_id=DATASET,
        content_version=content_version,
    )
    with composed.version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_inventory SET size_bytes = %s WHERE artifact_digest = %s",
            (_OVERSIZED, digest),
        )
    return content_version


def test_a_bare_spec_export_of_an_existing_file_serves_under_an_oversized_frame1(composed):
    """The export copies the csv file the mint wrote, reading no frame."""
    app, client = composed
    content_version = _mint_with_an_oversized_frame1(app, client)

    resp = client.post(
        "/api/datasets/exports",
        json={"frame": {"spec_id": DATASET}, "format": "csv", "bundle": False},
        headers=_CALLER,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["frame"] == {"spec_id": DATASET, "content_version": content_version}


def test_a_bare_spec_frame_over_an_oversized_frame1_refuses_missing_rendering(composed):
    """Presentation resolves the bare id to the pair, then its frame store refuses the document
    it would decode, as it refuses the pair."""
    app, client = composed
    content_version = _mint_with_an_oversized_frame1(app, client)

    bare = client.get(f"/api/presentation/frames/{DATASET}")
    pair = client.get(f"/api/presentation/frames/{DATASET}/{content_version}")

    assert bare.status_code == pair.status_code == 404, bare.text
    assert bare.json()["cause"]["code"] == pair.json()["cause"]["code"] == "missing_rendering"
