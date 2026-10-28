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
    "0023_frame_recipe.sql": "886f346dd6db7d28dc72a320c2c5341406a877a9652b867b85902bc77bda17a1",
    "0024_operation_attempt_fencing.sql": (
        "eeb8ca879e64dc227e6eb72bf2103882ece76062fd8d753010bb20dc41f2a0e4"
    ),
    "0025_artifact_location_renewal.sql": (
        "f97466368768f08f1d82dafc15f6097edc999a5ef6689d158ef68dbba80861b9"
    ),
    "0026_artifact_reference_principal_label.sql": (
        "ee43e91445a16733facc8f00c13e6cf87ff218c609a561d296a154875745c7fe"
    ),
    "0027_materialisation_record.sql": (
        "4b9c3aa5168ce4532d49e1ea08fd88ee7e11f08817a5ee6f8f1f32528edfa23b"
    ),
    "0028_spec_alias.sql": "12ee2155239debed4f52725dba93b34af47142b53497a5022891a796bbec80d2",
    "0029_version_record_frame_grammar_version.sql": (
        "58e5fa6d931cfd4ba9131e10362aeb0631137f9041be4a2bcaaa814179151aee"
    ),
    "0030_version_record_facts.sql": (
        "ae645ef718676d96331267fc434a549f2157f240d3a48bc07a408e5e6dd8dfcb"
    ),
    "0031_version_record_facts_backfill.sql": (
        "b696e9b578ddb15aafccd1d567e4cc4440a47b1a2d0d7de91595f15acadd81f0"
    ),
    "0032_operation_confirmed_at.sql": (
        "9daa6be37ff91df3c06e49a633b4c8ee81c804025d725f4c0a67ce66d2f76227"
    ),
    "0033_output_profile_party_kind.sql": (
        "cb2b22abcc784b65e46c9d3ec21163ca218a2a6a0f94d29af8a70110c1c6ac32"
    ),
    "0034_description_record.sql": (
        "f30ecafdf90bfee3885f279884c92a5abe0e1c3b036ef7f27bb086dc72031336"
    ),
    "0035_strip_backfilled_documentation.sql": (
        "3384412073a9b99370a0bae3e7452cc951e5fbaf5353051a25a87da10ba7f418"
    ),
    "0036_version_record_continuations.sql": (
        "034edded49a5cc08f3c5c55c65d1523b618d896332d598e333a02fca40336511"
    ),
    "0037_version_record_recency.sql": (
        "6c9fe2aeeacc2ddc0be5972b4ae6ecab59bf2db0a36c0478f5d68000344ff7e8"
    ),
    "0038_version_record_frame_summed.sql": (
        "7236772faed0ec1328386a75b87a8ca85a18374b0e6affabf93d590478129664"
    ),
    "0039_caller_allowance.sql": (
        "f157414916f3846f52c50569bcf6b5c550bed67e123ec160b0cacc3024f0b706"
    ),
    "0040_caller_allowance_budgets.sql": (
        "29c6abfdfae3f078aa2f1d1de79cd745e116aca295707629567ccfa54938bbc7"
    ),
    "0041_analysis_answer.sql": (
        "f2868dc1a5d525e7d1d37415e53f339f649a9ae967e86a2ce285a94e0115df99"
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
    "0011_revision_frame_rendering.sql": (
        "956cc50b9e0285883089f082f663aa37b311974d0ac1b46cb1ee51e78a0889f0"
    ),
    "0012_share.sql": "af08f2b8183f5347fca21e7b4392adfd88161a854587c40cb7eef3fc6b134a51",
    "0013_share_slot_check.sql": (
        "e016def5442e46ed88a276ed0a8a21b46c8293f60add7135b8802a796735a3ac"
    ),
    "0014_share_slot_next_check.sql": (
        "65eb5bf176ffeaac8e951e0f5f076cd2a169c201d25653f4ed3aa88f60fe3ac0"
    ),
    "0015_share_create_token.sql": (
        "aed4d316afc7180b4b8491ec7f161a708d231805643ca1bca66d7e432a6730f3"
    ),
    "0016_share_slot_check_claim.sql": (
        "1b59cb708b9acedc7ade57cebde3210accc38be5fc7adb2014512e4fbd705d8c"
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
