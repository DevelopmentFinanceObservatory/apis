"""Pin and publish principals use separate allowlists. Pinning has no unpin route, so a
principal trusted to publish must not gain the pin grant's budget, and pins as any open caller
does. Tests cover surface wiring and the pin door's `DATASETS_PIN_FIRST_PARTY_PRINCIPAL`
fallback. A publish dependency
test also checks the all-or-nothing store configuration.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from artifact_store import Tier
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import pin as pin_door
from dfo_platform.caller import address_key
from spec_registry import LiveSpecRegistry

from ._app_factories import make_client, offline_allowances
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._snapshot_doubles import (
    FakeSnapshotReader,
    FakeSpecReader,
    _store_and_locate,
    _version,
    make_client_for_door,
)
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_NOW = datetime(2026, 6, 1, tzinfo=UTC)
_PIN_PRINCIPAL = "datasets-pin-key"
_PUBLISH_PRINCIPAL = "presentation-mcp"
_SPEC_ID = "spec_a"
_CONTENT_VERSION = "cv1"

_PUBLISH_SPEC_ID = "ops_abc123"
_PUBLISH_CONTENT_VERSION = "ocv_abc123"


def _publish_body() -> dict[str, object]:
    return {
        "schema_version": 1,
        "spec_id": _PUBLISH_SPEC_ID,
        "content_version": _PUBLISH_CONTENT_VERSION,
        "role_keys": ["recipient"],
        "rows": [
            {
                "stat_var": "dcid:Exemplar",
                "time_period": "2020",
                "value": 1.0,
                "recipient": "country/KEN",
            }
        ],
        "input_content_versions": ["cv_00000000000000000000000000abc123"],
    }


def test_surface_pin_and_publish_allowed_principals_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The surface wires each allowlist to one door. Each door grants its own principal, the
    publish door refuses the pin principal, and the pin door takes the publish principal for an
    open caller, named by its address."""
    monkeypatch.setenv("DFO_CALLER_HMAC_KEY", "test-caller-key")
    writer = FakeArtifactReferenceWriter()
    # Pinned-tier storage skips retention checks against the door's real clock.
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-vs-publish-separation", tier=Tier.pinned)
    reader = FakeSnapshotReader(
        versions={(_SPEC_ID, _CONTENT_VERSION): _version()},
        locations={(_SPEC_ID, _CONTENT_VERSION): [location]},
    )
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        spec_reader=FakeSpecReader(spec_ids=[_SPEC_ID]),
        graph=object(),
        sdmx=object(),
        artifact_store=store,
        snapshot_reader=reader,
        reference_writer=writer,
        pin_allowed_principals=frozenset({_PIN_PRINCIPAL}),
        publish_allowed_principals=frozenset({_PUBLISH_PRINCIPAL}),
    )

    pin_accepts_its_own = client.post(
        f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )
    assert pin_accepts_its_own.status_code == 200, pin_accepts_its_own.text

    pin_opens_to_the_publish_principal = client.post(
        f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PUBLISH_PRINCIPAL, "x-forwarded-for": "203.0.113.7"},
    )
    assert pin_opens_to_the_publish_principal.status_code == 200, (
        pin_opens_to_the_publish_principal.text
    )
    opened, granted = sorted({key[0] for key in writer.records})
    assert opened == address_key("203.0.113.7", hmac_key="test-caller-key")
    assert granted.startswith("secret-sha256:")

    publish_refuses_the_pin_principal = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(),
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )
    assert publish_refuses_the_pin_principal.status_code == 403, (
        publish_refuses_the_pin_principal.text
    )
    assert publish_refuses_the_pin_principal.json()["code"] == "publish_not_permitted"

    # The permission gate must accept this principal; the null writer may fail afterward.
    publish_accepts_its_own = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(),
        headers={"x-dfo-principal": _PUBLISH_PRINCIPAL},
    )
    assert publish_accepts_its_own.status_code != 403, publish_accepts_its_own.text


def test_pin_door_env_fallback_reads_its_own_constant() -> None:
    """`allowed_principals=None` resolves `pin_first_party_principal` from settings."""
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"pin-env-fallback-own-constant")
    reader = FakeSnapshotReader(
        versions={(_SPEC_ID, _CONTENT_VERSION): _version()},
        locations={(_SPEC_ID, _CONTENT_VERSION): [location]},
    )
    settings = DatasetsBuildSettings(pin_first_party_principal=_PIN_PRINCIPAL)
    door = pin_door.build_pin_door(
        deps=fake_deps_cell(
            settings=settings,
            spec_reader=FakeSpecReader(spec_ids=[_SPEC_ID]),
            snapshot_reader=reader,
            artifact_store=store,
            reference_writer=FakeArtifactReferenceWriter(),
            allowances=offline_allowances(settings),
            now=lambda: _NOW,
        ),
        settings=settings,
        allowed_principals=None,
    )
    client = make_client_for_door(door)

    resp = client.post(
        f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}/pin",
        json={"schema_version": 1, "artifact_digest": location.artifact_digest},
        headers={"x-dfo-principal": _PIN_PRINCIPAL},
    )

    assert resp.status_code == 200, resp.text
