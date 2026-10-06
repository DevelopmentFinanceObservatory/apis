"""Tests snapshot refusals, signed manifest fields, output options, route grammar, and cache
headers through `read_snapshot_manifest` and the HTTP door.
"""

from __future__ import annotations

import pytest

from artifact_store import Tier
from dataset_store.snapshot import (
    Revocation,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
)
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    SnapshotNotFound,
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.snapshot import build_snapshot_door, read_snapshot_manifest
from datasets_api.manifest import encode_version_body
from datasets_api.manifest_models import (
    CitationModel,
    CoverageModel,
    DefinitionModel,
    Manifest,
)
from datasets_api.render import (
    DATA_LAYOUT_VERSION,
    FRAME_LAYOUT,
    LAYOUT_KEY,
    RENDITION_KEY,
    rendition_token,
)
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from materialise import EnrichmentLookup
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.frame_build import build_frame
from materialise.profile import compute_output_profile

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import (
    _NOW,
    FakeSnapshotReader,
    FakeSpecReader,
    _location,
    _store_and_locate,
    _version,
    make_client_for_door,
)

_STAT_VAR_A = "dcid:GrantA"
# A bundle rendered under the current settings carries this rendition.
_RENDITION = rendition_token(DatasetsBuildSettings().site)


def _output_options_fixture_frame(*, stat_var: str = _STAT_VAR_A):
    return build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var,
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2015",
                "value": 1000.0,
            },
            {
                "stat_var": stat_var,
                "recipient": "country/KEN",
                "unit": "USD",
                "measurementMethod": "GrantEquivalent",
                "provenance": "OECD_DAC2A",
                "time_period": "2016",
                "value": 1500.0,
            },
        ],
    )


def _output_options_fixture_profile(*, stat_var: str = _STAT_VAR_A):
    return compute_output_profile(
        _output_options_fixture_frame(stat_var=stat_var),
        role_columns=("recipient",),
        enrichment=EnrichmentLookup(FakeMintGraphReader()),
    )


def _minimal_manifest_kwargs() -> dict[str, object]:
    return {
        "content_version": "cv1",
        "prepared_at": "2026-01-01T00:00:00Z",
        "coverage": CoverageModel(rows=2, dimensions=[]),
        "citation": CitationModel(text="", sources=[]),
        "trace": {},
        "artifacts": [],
    }


def test_unregistered_id_refuses_spec_not_found():
    with pytest.raises(SpecNotFound):
        read_snapshot_manifest(
            id="spec_unknown",
            content_version="cv1",
            spec_reader=FakeSpecReader(),
            snapshot_reader=FakeSnapshotReader(),
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


def test_registered_id_with_no_version_at_that_content_version_refuses_snapshot_not_found():
    with pytest.raises(SnapshotNotFound):
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv_unknown",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=FakeSnapshotReader(versions={("spec_a", "cv1"): _version()}),
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


def test_a_revoked_version_refuses_snapshot_revoked_even_with_servable_artifacts():
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [_location(stored_at=_NOW)]},
        revocations={("spec_a", "cv1"): Revocation(reason="takedown", revoked_at=_NOW)},
    )
    with pytest.raises(SnapshotRevoked):
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv1",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


def test_a_collapsed_versions_row_names_its_base_content_version():
    """Sets `base_content_version` directly on a `StoredVersion` for a collapsed version."""
    body = {
        **_minimal_manifest_kwargs(),
        "definition": {
            "schema_version": 1,
            "spec_id": "spec_a",
            "identity": {"identity_version": "spec1"},
        },
        "content_version": "cv_collapsed",
        "prepared_at": "2026-01-01T00:00:00Z",
    }
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"collapsed-bytes")
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv_collapsed"): _version(
                immutable_body=body, content_version="cv_collapsed", base_content_version="cv1"
            )
        },
        locations={("spec_a", "cv_collapsed"): [location]},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv_collapsed",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert manifest.content_version == "cv_collapsed"


def test_a_collapsed_version_refuses_snapshot_revoked_when_its_base_version_is_revoked():
    """Revoking the base version also revokes its collapsed version, since revocations are
    keyed by subject and content version.
    """
    body = {
        **_minimal_manifest_kwargs(),
        "definition": {
            "schema_version": 1,
            "spec_id": "spec_a",
            "identity": {"identity_version": "spec1"},
        },
        "content_version": "cv_collapsed",
        "prepared_at": "2026-01-01T00:00:00Z",
    }
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv_collapsed"): _version(
                immutable_body=body, content_version="cv_collapsed", base_content_version="cv1"
            )
        },
        locations={("spec_a", "cv_collapsed"): [_location(stored_at=_NOW)]},
        revocations={("spec_a", "cv1"): Revocation(reason="base-takedown", revoked_at=_NOW)},
    )

    with pytest.raises(SnapshotRevoked) as exc_info:
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv_collapsed",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )
    # The error names the requested collapsed version and the base version's reason.
    assert exc_info.value.content_version == "cv_collapsed"
    assert exc_info.value.reason == "base-takedown"


def test_a_collapsed_version_serves_fine_when_only_an_unrelated_version_is_revoked():
    """A revocation for an unrelated version leaves the collapsed version servable."""
    body = {
        **_minimal_manifest_kwargs(),
        "definition": {
            "schema_version": 1,
            "spec_id": "spec_a",
            "identity": {"identity_version": "spec1"},
        },
        "content_version": "cv_collapsed",
        "prepared_at": "2026-01-01T00:00:00Z",
    }
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"collapsed-bytes")
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv_collapsed"): _version(
                immutable_body=body, content_version="cv_collapsed", base_content_version="cv1"
            )
        },
        locations={("spec_a", "cv_collapsed"): [location]},
        revocations={("spec_a", "cv_other"): Revocation(reason="unrelated", revoked_at=_NOW)},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv_collapsed",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )
    assert manifest.content_version == "cv_collapsed"


def test_read_snapshot_currency_also_refuses_when_the_base_version_is_revoked():
    from datasets_api.doors.snapshot import read_snapshot_currency

    body = {
        **_minimal_manifest_kwargs(),
        "definition": {
            "schema_version": 1,
            "spec_id": "spec_a",
            "identity": {"identity_version": "spec1"},
        },
        "content_version": "cv_collapsed",
        "prepared_at": "2026-01-01T00:00:00Z",
    }
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv_collapsed"): _version(
                immutable_body=body, content_version="cv_collapsed", base_content_version="cv1"
            )
        },
        revocations={("spec_a", "cv1"): Revocation(reason="base-takedown", revoked_at=_NOW)},
    )

    with pytest.raises(SnapshotRevoked) as exc_info:
        read_snapshot_currency(
            id="spec_a",
            content_version="cv_collapsed",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            settings=DatasetsBuildSettings(),
        )
    assert exc_info.value.content_version == "cv_collapsed"
    assert exc_info.value.reason == "base-takedown"


class RevokingDuringLocationsReadSnapshotReader(FakeSnapshotReader):
    """Wraps `FakeSnapshotReader`, but `read_artifact_locations()` records a
    revocation as a side effect before returning. This stands in for a
    revocation committing in the gap between `read_snapshot_manifest`'s own
    early revocation check and its later pre-signing recheck, without
    needing real concurrency timing.
    """

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        self.revocations[(subject_id, content_version)] = Revocation(
            reason="revoked-before-sign", revoked_at=_NOW
        )
        return super().read_artifact_locations(subject_id, content_version)


def test_a_revocation_landing_after_the_early_check_is_caught_before_signing():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"revoked-before-sign")
    reader = RevokingDuringLocationsReadSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    with pytest.raises(SnapshotRevoked):
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv1",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


def test_every_working_artifact_expired_and_none_pinned_refuses_artifact_expired():
    from dataset_store.snapshot import WORKING_TIER_RETENTION

    long_ago = _NOW - WORKING_TIER_RETENTION - WORKING_TIER_RETENTION
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [_location(stored_at=long_ago)]},
    )
    with pytest.raises(ArtifactExpired):
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv1",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


def test_a_version_with_zero_artifacts_ever_linked_also_refuses_artifact_expired():
    reader = FakeSnapshotReader(versions={("spec_a", "cv1"): _version()})
    with pytest.raises(ArtifactExpired):
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv1",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


def test_an_expired_listable_rendering_refuses_artifact_expired_despite_an_unexpired_frame1_row():
    """Expiry checks listable locations. The frame1 row is filtered before
    `select_servable_locations`, leaving the expired CSV as the only candidate.
    """
    from dataset_store.snapshot import WORKING_TIER_RETENTION

    long_ago = _NOW - WORKING_TIER_RETENTION - WORKING_TIER_RETENTION
    csv_location = _location(stored_at=long_ago, media_type="text/csv")
    frame1_location = _location(
        digest="sha256:frame1",
        rendering_key="rk-frame1",
        rendering={"format": "frame1"},
        media_type=FRAME1_MEDIA_TYPE,
        stored_at=_NOW,
    )
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [csv_location, frame1_location]},
    )
    with pytest.raises(ArtifactExpired):
        read_snapshot_manifest(
            id="spec_a",
            content_version="cv1",
            spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
            snapshot_reader=reader,
            artifact_store=FakeArtifactStore(),
            now=lambda: _NOW,
            settings=DatasetsBuildSettings(),
        )


def test_a_pinned_artifact_is_always_servable_regardless_of_age():
    from dataset_store.snapshot import WORKING_TIER_RETENTION

    long_ago = _NOW - WORKING_TIER_RETENTION - WORKING_TIER_RETENTION
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"data", tier=Tier.pinned, stored_at=long_ago)
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].tier == "pinned"
    assert manifest.artifacts[0].artifact_digest == location.artifact_digest


def test_two_renderings_sharing_one_digest_both_survive_on_the_wire_manifest():
    """Two renderings can share a digest while remaining distinct manifest entries.

    The HTTP response verifies that the read path preserves both options.
    """
    store = FakeArtifactStore()
    stored = store.put(b"identical-bytes-two-renderings", tier=Tier.working)
    csv_location = _location(
        digest=stored.digest,
        rendering_key="rk-csv",
        rendering={"format": "csv"},
    )
    parquet_location = _location(
        digest=stored.digest,
        rendering_key="rk-parquet-bundle",
        rendering={"format": "parquet", "bundle": True, RENDITION_KEY: _RENDITION},
    )
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [csv_location, parquet_location]},
    )
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 2
    assert {artifact["artifact_digest"] for artifact in artifacts} == {stored.digest}
    assert {artifact["rendering"]["format"] for artifact in artifacts} == {"csv", "parquet"}


def test_a_version_with_one_csv_and_one_internal_frame1_location_lists_only_the_csv():
    """Frame1 is internal and has no file extension mapping, so the manifest must list only CSV."""
    store = FakeArtifactStore()
    csv_location = _store_and_locate(store, b"csv-bytes", media_type="text/csv")
    frame1_location = _store_and_locate(
        store,
        b"frame1-bytes",
        rendering_key="rk-frame1",
        rendering={"format": "frame1"},
        media_type=FRAME1_MEDIA_TYPE,
    )
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [csv_location, frame1_location]},
    )
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    artifacts = resp.json()["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["media_type"] == "text/csv"
    assert artifacts[0]["artifact_digest"] == csv_location.artifact_digest


def test_servable_artifacts_get_freshly_signed_urls_and_the_stored_body_round_trips():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"csv-bytes")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert manifest.content_version == "cv1"
    assert manifest.definition is not None
    assert manifest.definition.spec_id == "spec_a"
    assert len(manifest.artifacts) == 1
    artifact = manifest.artifacts[0]
    assert artifact.artifact_digest == location.artifact_digest
    assert artifact.tier == "working"
    assert artifact.url == f"fake://artifact/working/{location.artifact_digest}"


_CRS_CV = "cv_5c9dfd37e657d3a3c7000cc6acf49c43"


def _cited(*sources: tuple[str, str]) -> dict[str, object]:
    return {"text": "", "sources": [{"ref": ref, "label": label} for ref, label in sources]}


def _signed_file_names(
    *, id: str, version: StoredVersion, locations: list[StoredArtifactLocation]
) -> dict[str, str]:
    """Each signed artifact's `Content-Disposition`, keyed by its rendering's repr."""
    store = FakeArtifactStore()
    stored = [
        _store_and_locate(
            store,
            repr(location.rendering).encode(),
            rendering_key=location.rendering_key,
            rendering=location.rendering,
            media_type=location.media_type,
        )
        for location in locations
    ]
    read_snapshot_manifest(
        id=id,
        content_version=version.content_version,
        spec_reader=FakeSpecReader(spec_ids=[id]),
        snapshot_reader=FakeSnapshotReader(
            versions={(id, version.content_version): version},
            locations={(id, version.content_version): stored},
        ),
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )
    return {
        repr(location.rendering): store.signed_headers[(Tier.working, location.artifact_digest)][1]
        for location in stored
    }


def test_a_standalone_version_names_its_artifacts_by_source_first_mint_date_and_version():
    bundle = {"bundle": True, "format": "csv", RENDITION_KEY: _RENDITION}
    body = {
        **_version().immutable_body,
        "content_version": _CRS_CV,
        "prepared_at": "2026-10-04T09:00:00Z",
        "citation": _cited(
            ("provenance/OECDCreditorReportingSystem", "OECD Creditor Reporting System (CRS)")
        ),
    }
    version = StoredVersion(
        subject_id="spec_a",
        subject_kind="spec",
        content_version=_CRS_CV,
        schema_version=1,
        immutable_body=body,
    )

    names = _signed_file_names(
        id="spec_a",
        version=version,
        locations=[
            _location(rendering=bundle, media_type="application/zip"),
            _location(rendering={"bundle": False, "format": "csv"}, media_type="text/csv"),
            _location(
                rendering={"statistics": "series"}, media_type="application/vnd.apache.parquet"
            ),
        ],
    )

    prefix = 'attachment; filename="dfo_oecd-creditor-reporting-system_2026-10-04'
    assert names == {
        repr(bundle): f'{prefix}_csv_5c9dfd.zip"',
        repr({"bundle": False, "format": "csv"}): f'{prefix}_5c9dfd.csv"',
        repr({"statistics": "series"}): f'{prefix}_series-statistics_5c9dfd.parquet"',
    }


def test_a_version_citing_no_source_names_its_artifacts_dataset():
    bundle = {"bundle": True, "format": "xlsx", RENDITION_KEY: _RENDITION}
    names = _signed_file_names(
        id="spec_a",
        version=_version(),
        locations=[_location(rendering=bundle, media_type="application/zip")],
    )

    assert names == {repr(bundle): ('attachment; filename="dfo_dataset_2026-01-01_xlsx_cv1.zip"')}


# --- `output_options`: derived fresh from the stored `output_profile`
# at every read ---


def test_a_stored_profile_derives_output_options_from_the_current_capability_document():
    profile = _output_options_fixture_profile()
    manifest = Manifest(
        definition=DefinitionModel(schema_version=1, spec_id="spec_a", identity={}),
        **_minimal_manifest_kwargs(),
    )
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"csv-bytes")
    reader = FakeSnapshotReader(
        versions={
            ("spec_a", "cv1"): _version(
                immutable_body=encode_version_body(manifest, output_profile=profile)
            )
        },
        locations={("spec_a", "cv1"): [location]},
    )

    read_manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert read_manifest.output_options is not None
    assert read_manifest.output_options.capability_version == DATASETS_CAPABILITY_VERSION
    assert [c.name for c in read_manifest.output_options.columns] == [
        c.name for c in profile.columns
    ]


def test_a_legacy_stored_body_without_an_output_profile_serves_with_output_options_absent():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"csv-bytes")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},  # no "output_profile" key at all
        locations={("spec_a", "cv1"): [location]},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert manifest.output_options is None

    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    assert "output_options" not in resp.json()


# --- Route grammar: the {id:dfo_spec_id} convertor ---


@pytest.mark.parametrize("subject_id", ["not-a-valid-id", "col_a"])
def test_an_id_outside_the_grammar_refuses_the_platforms_own_not_found_before_the_handler(
    subject_id: str,
):
    """`SpecIdConvertor` sends any id outside the `spec_` namespace, a retired collection's
    `col_` id included, to the platform's generic `not_found` response."""
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=FakeSnapshotReader(),
        artifact_store=FakeArtifactStore(),
    )
    client = make_client_for_door(door)

    resp = client.get(f"/api/datasets/{subject_id}/cv1")

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "not_found"


def test_a_well_formed_but_unregistered_id_still_refuses_spec_not_found():
    """A `spec_`-prefixed id passes the convertor but names nothing durable.
    This door's own `spec_not_found` refusal fires, unchanged by the
    grammar constraint.
    """
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=FakeSnapshotReader(),
        artifact_store=FakeArtifactStore(),
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_totally_unregistered/cv1")

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "spec_not_found"


# --- Cache-Control: every response carries freshly signed URLs, so a cache
# handing a stored copy back later would serve a URL past its own
# `url_expires_at`, or for an artifact revoked/expired since.


# --- `rendering_key`: narrows to one stored rendering before signing, for
# the definition door's own head-hit path.


def _three_renderings_reader() -> tuple[
    FakeArtifactStore, FakeSnapshotReader, list[StoredArtifactLocation]
]:
    store = FakeArtifactStore()
    locations = [
        _store_and_locate(store, b"csv-bytes", rendering_key="rk-csv", media_type="text/csv"),
        _store_and_locate(
            store,
            b"parquet-bytes",
            rendering_key="rk-parquet",
            media_type="application/vnd.apache.parquet",
        ),
        _store_and_locate(
            store,
            b"zip-bytes",
            rendering_key="rk-zip",
            rendering={"format": "csv", "bundle": True, RENDITION_KEY: _RENDITION},
            media_type="application/zip",
        ),
    ]
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): locations},
    )
    return store, reader, locations


def test_rendering_key_left_none_lists_and_signs_every_rendering():
    store, reader, locations = _three_renderings_reader()

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert {a.artifact_digest for a in manifest.artifacts} == {
        loc.artifact_digest for loc in locations
    }
    assert len(store.signed_headers) == 3


def test_rendering_key_narrows_to_the_one_matching_rendering():
    store, reader, locations = _three_renderings_reader()
    target = locations[1]

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        rendering_key=target.rendering_key,
        settings=DatasetsBuildSettings(),
    )

    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].artifact_digest == target.artifact_digest
    assert len(store.signed_headers) == 1


def test_rendering_key_matching_nothing_returns_empty_artifacts_with_no_error():
    store, reader, _locations = _three_renderings_reader()

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        rendering_key="rk-does-not-exist",
        settings=DatasetsBuildSettings(),
    )

    assert manifest.artifacts == []
    assert store.signed_headers == {}


def test_a_successful_snapshot_response_carries_cache_control_no_store():
    store = FakeArtifactStore()
    location = _store_and_locate(store, b"cache-control-data")
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): [location]},
    )
    door = build_snapshot_door(
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
    )
    client = make_client_for_door(door)

    resp = client.get("/api/datasets/spec_a/cv1")

    assert resp.status_code == 200, resp.text
    assert resp.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("tier", "listed"),
    [(Tier.working, ["current", "statistics"]), (Tier.pinned, ["current", "old", "statistics"])],
    ids=["unpinned-is-left-out", "pinned-follows-the-current-bundle"],
)
def test_a_bundle_from_before_renditions_is_listed_only_when_pinned_after_the_current_one(
    tier: Tier, listed: list[str]
):
    """The reader lists the old bundle first; the snapshot puts the current layout first and
    keeps the statistics artifact last."""
    store = FakeArtifactStore()
    zip_type = "application/zip"
    locations = {
        "old": _store_and_locate(
            store,
            b"old-zip",
            tier=tier,
            rendering={"format": "csv", "bundle": True},
            media_type=zip_type,
        ),
        "current": _store_and_locate(
            store,
            b"current-zip",
            rendering={"format": "csv", "bundle": True, RENDITION_KEY: _RENDITION},
            media_type=zip_type,
        ),
        "statistics": _store_and_locate(
            store,
            b"statistics",
            rendering={"statistics": "series"},
            media_type="application/vnd.apache.parquet",
        ),
    }
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): list(locations.values())},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert [a.artifact_digest for a in manifest.artifacts] == [
        locations[name].artifact_digest for name in listed
    ]


def test_a_bare_file_in_an_old_layout_stays_listed_after_the_current_one():
    """The reader lists the old-layout file first. The snapshot puts the current long layout
    first, then the frame-layout file, keeps the old file listed unpinned, and keeps the
    statistics artifact last."""
    store = FakeArtifactStore()
    locations = {
        "old": _store_and_locate(
            store, b"old-csv", rendering={"format": "csv", "bundle": False}, media_type="text/csv"
        ),
        "frame": _store_and_locate(
            store,
            b"frame-parquet",
            rendering={"format": "parquet", "bundle": False, LAYOUT_KEY: FRAME_LAYOUT},
            media_type="application/vnd.apache.parquet",
        ),
        "current": _store_and_locate(
            store,
            b"current-csv",
            rendering={"format": "csv", "bundle": False, LAYOUT_KEY: DATA_LAYOUT_VERSION},
            media_type="text/csv",
        ),
        "statistics": _store_and_locate(
            store,
            b"statistics",
            rendering={"statistics": "series"},
            media_type="application/vnd.apache.parquet",
        ),
    }
    reader = FakeSnapshotReader(
        versions={("spec_a", "cv1"): _version()},
        locations={("spec_a", "cv1"): list(locations.values())},
    )

    manifest = read_snapshot_manifest(
        id="spec_a",
        content_version="cv1",
        spec_reader=FakeSpecReader(spec_ids=["spec_a"]),
        snapshot_reader=reader,
        artifact_store=store,
        now=lambda: _NOW,
        settings=DatasetsBuildSettings(),
    )

    assert [a.artifact_digest for a in manifest.artifacts] == [
        locations[name].artifact_digest for name in ("current", "frame", "old", "statistics")
    ]
