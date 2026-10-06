"""Pins checksums for applied Dataset Store and Presentation Store migrations."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset_store import migrate as dataset_migrate
from presentation_store import migrate as presentation_migrate

_DATASET_STORE = {
    "0001_bootstrap.sql": "6382bf44a1515ebe8be8a13c0ba4a7c2ebacc326d180a63dcae3fe98f2cf62e3",
    "0002_spec_registry.sql": "e72f23e04816b2e42712825d50ce8067aad284dbe540dab1f54b54d405ab9a55",
    "0003_artifact_inventory.sql": "9d7ea2130f5fc313a60917f338a72d3830c4852a32ad7704a1cdfa26167d3f65",
    "0004_version_records.sql": "7e3ab50a1ed75381cac47e75597d9c1641b668e22016742778e975e9b47bb7e7",
    "0006_version_revocation.sql": "4d15d9a90cc70df2cd071905109a3570cf08b3db9e8e54521666d1b0b6540672",
    "0007_artifact_references.sql": "a45c004a5f244a803f2a48969251351fdb82643764102031cfe446d21f2ba51e",
    "0008_artifact_location_tier_index.sql": "cd4f9d4e6e3d3e5fde416ba01e2391881c0330abc512cae3be7840189b794b70",
    "0009_subject_head.sql": "94342427fa7c85a217bc642d5aa723a3bb55649368db6237b699cdf586e24d5e",
    "0010_version_record_base_content_version.sql": (
        "75347f11aca60c3ce3f5be55c199bd04b4bdbd6c505001c22034c735ae6ba122"
    ),
    "0011_operation.sql": "0ea3f98f5cf405c36120173a524a30c7114c32d3a0abb45b8285fd05dad43460",
    "0012_worker_role.sql": "e4757f6042eebbcc4ab4bd127662faf542678255442bed18240f3ab3b2a4187a",
    "0013_operation_retryable_and_index.sql": (
        "c4b98ac9488b950f6ea5f51878fd41848844ef37411c70d3a303ff633c9965db"
    ),
    "0014_operation_dispatch.sql": (
        "0659559bebeff2cc61f4e1d430a1462b89725039193d46c217a02fe96c263a22"
    ),
    "0015_operation_origin_catalogue_build.sql": (
        "77adc60b13c6366dadf07b0f8b62d8dd179681970523c5b1dce78c51f94fe0b8"
    ),
    "0016_operation_refusal.sql": "3325dd126a53987bb850cba5588608e3d927065c99d247fa1dab6206fe06656e",
    "0017_trace_conversion_entries.sql": (
        "bed95a224cbd9edd4fcc6705d27b5c66a561b2fa2890104dbd2ecdc18be0e7f0"
    ),
    "0018_backfill_prepared_at.sql": (
        "22818fa8bfb477d7cddef422f2ba4451b7d1ac7cdd9e46521ddc0c20b941f571"
    ),
    "0019_backfill_prepared_at_after_deploy.sql": (
        "23e98a7de77f639c997f7268dfd2398cedd3fda15fdc837b1aa81bf1aa7556d3"
    ),
    "0020_trace_conversion_entries_after_deploy.sql": (
        "0f0219d5438f2a20e929eda90dbdd5c94c9ab77596fec406a3e6c95eb0ebb254"
    ),
    "0021_drop_collections.sql": (
        "9aa7182f4a1b6ed1087cb72086c0aac644072573f1214a3653c1b89b91991e84"
    ),
    "0022_drop_null_collection.sql": (
        "1ad07c270360607f2f58904689894c140351404dbc20961218329af91f78257f"
    ),
}

_PRESENTATION_STORE = {
    "0001_bootstrap.sql": "360fc23a1adc6f9f35229e5bd31b747a04083e4da2ff2fcad86929b5f26834c7",
    "0002_composition.sql": "a211e9c70d0e3db1188d51498e9155650dfb15e9b85ea7584cae6cd0645856f0",
    "0003_component.sql": "02f701d84f97212e2a53101cc4b428accf4e62602e1b6b83b323d0e6afcd67a7",
    "0004_draft_lease.sql": "801d6647f3dbf31b0249550ba95ea9b416057b3d5a37b91ed971f31bc1ebaeff",
    "0005_series_coordinates.sql": "7dd2ce59d4d405c5b0cd6440d77ef821f7f505743ce60534bbefa25748261a9e",
    "0006_bar_chart_axes.sql": "cea9ffe83879926a71d953004e1fdb1b1266332778aa0f99da12f55d0dae82a8",
    "0007_write_token.sql": "57f570237d6b957aad0ffc89152d75be7b104cfac8c537cef9be4e6094e184ae",
    "0008_idempotent_create.sql": (
        "2427bbee8b7452a791abd180c1d4c62d122cc335370f1b1151b46cc11abd4d76"
    ),
    "0009_drop_draft_lease.sql": (
        "396cb5bd69724e41eaef57fe31ff613cc41397b84372e30bed261a83e1e41b10"
    ),
    "0010_revision_formatter_profile.sql": (
        "41bac2559d6a5999ff756cf65e3e73133240fa0b463e0465f53becd445134b48"
    ),
}


@pytest.mark.parametrize(
    ("directory", "pinned"),
    [
        pytest.param(dataset_migrate.MIGRATIONS_DIR, _DATASET_STORE, id="dataset_store"),
        pytest.param(
            presentation_migrate.MIGRATIONS_DIR, _PRESENTATION_STORE, id="presentation_store"
        ),
    ],
)
def test_every_migration_file_matches_its_pinned_checksum(directory: Path, pinned: dict[str, str]):
    on_disk = {
        path.name: dataset_migrate._content_hash(path)
        for path in dataset_migrate.discover_migrations(directory)
    }

    assert on_disk == pinned
