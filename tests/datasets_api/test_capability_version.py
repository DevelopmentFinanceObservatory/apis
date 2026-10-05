"""Check which capability inputs change the token and which leave it unchanged.
Fresh subprocesses verify stability across Python hash seeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from code_index.types import CodeSystemDeclaration
from datasets_api import version
from datasets_api.capabilities import build_capabilities
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.config import DatasetsBuildSettings
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE


def _baseline_capability_document() -> dict:
    return build_capabilities(
        capability_version=version._EXEMPLAR_CAPABILITY_VERSION,
        build=version._EXEMPLAR_BUILD,
        code_index_stamp=version._EXEMPLAR_CODE_INDEX_STAMP,
        declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
        settings=DatasetsBuildSettings(),
    ).model_dump(mode="json")


class TestPerturbationsMoveTheToken:
    def test_sync_budget_rows_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["sync_budget_rows"] += 1

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_sync_budget_requests_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["sync_budget_requests"] += 1

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_async_max_rows_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["async_max_rows"] += 1

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_async_max_requests_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["async_max_requests"] += 1

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_catalogue_title_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        code, (title, detail) = next(iter(sorted(DATASETS_CODE_CATALOGUE.items())))
        mutated_catalogue = dict(DATASETS_CODE_CATALOGUE)
        mutated_catalogue[code] = (title + " (mutated)", detail)

        perturbed = version._compute_datasets_capability_version(code_catalogue=mutated_catalogue)

        assert perturbed != baseline

    def test_liveness_flag_flip_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["revision_provenance_live"] = True

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_warning_catalogue_message_edit_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        code, (title, detail, severity) = next(iter(sorted(DATASETS_WARNING_CATALOGUE.items())))
        mutated_catalogue = dict(DATASETS_WARNING_CATALOGUE)
        mutated_catalogue[code] = (title, detail + " (mutated)", severity)

        perturbed = version._compute_datasets_capability_version(
            warning_catalogue=mutated_catalogue
        )

        assert perturbed != baseline

    def test_default_facet_policy_field_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["default_facet_policy"]["version"] = "2"

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    @pytest.mark.parametrize(
        "field_name",
        [
            pytest.param("series_count", id="series-count"),
            pytest.param("output_options", id="output-options"),
        ],
    )
    def test_manifest_field_addition_moves_the_token(self, field_name):
        baseline = version._compute_datasets_capability_version()
        without_field_schema = version.build_schema_document()
        manifest_schema = dict(without_field_schema["manifest"])
        manifest_schema["properties"] = {
            key: value for key, value in manifest_schema["properties"].items() if key != field_name
        }
        without_field_schema = {
            **without_field_schema,
            "manifest": manifest_schema,
        }

        without_field = version._compute_datasets_capability_version(
            schema_document=without_field_schema
        )

        assert without_field != baseline

    def test_groupby_wire_shape_change_moves_the_token(self):
        # Removing `GroupbyOptions.by` checks that the groupby wire shape affects the token.
        baseline = version._compute_datasets_capability_version()
        perturbed_schema = version.build_schema_document()
        groupby_schema = dict(perturbed_schema["mint_request"]["$defs"]["GroupbyOptions"])
        groupby_schema["properties"] = {
            key: value for key, value in groupby_schema["properties"].items() if key != "by"
        }
        perturbed_schema = {
            **perturbed_schema,
            "mint_request": {
                **perturbed_schema["mint_request"],
                "$defs": {
                    **perturbed_schema["mint_request"]["$defs"],
                    "GroupbyOptions": groupby_schema,
                },
            },
        }

        perturbed = version._compute_datasets_capability_version(schema_document=perturbed_schema)

        assert perturbed != baseline

    def test_summability_declarations_live_flip_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["summability_declarations_live"] = not perturbed_document[
            "summability_declarations_live"
        ]

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_recipe_classes_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["recipe_classes"] = ["ConversionRecipeDAC"]

        perturbed = version._compute_datasets_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_schema_id_or_property_change_moves_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_schema = version.build_schema_document()
        perturbed_schema["manifest"] = {
            **perturbed_schema["manifest"],
            "$id": perturbed_schema["manifest"]["$id"] + "-mutated",
        }

        perturbed = version._compute_datasets_capability_version(schema_document=perturbed_schema)

        assert perturbed != baseline


class TestNonPerturbationsDoNotMoveTheToken:
    def test_schema_description_edit_does_not_move_the_token(self):
        baseline = version._compute_datasets_capability_version()
        perturbed_schema = version.build_schema_document()
        # Schema descriptions are stripped before hashing, so wording changes leave the token stable.
        perturbed_schema["mint_request"] = {
            **perturbed_schema["mint_request"],
            "description": "a reworded, purely cosmetic description",
        }

        unchanged = version._compute_datasets_capability_version(schema_document=perturbed_schema)

        assert unchanged == baseline

    def test_build_id_change_does_not_move_the_token(self):
        document_a = build_capabilities(
            capability_version="dset_whatever",
            build="build-A",
            code_index_stamp=version._EXEMPLAR_CODE_INDEX_STAMP,
            declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
        document_b = build_capabilities(
            capability_version="dset_whatever",
            build="build-B",
            code_index_stamp=version._EXEMPLAR_CODE_INDEX_STAMP,
            declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")

        token_a = version._compute_datasets_capability_version(capability_document=document_a)
        token_b = version._compute_datasets_capability_version(capability_document=document_b)

        assert token_a == token_b

    def test_decode_code_systems_change_does_not_move_the_token(self):
        # Changing code-system names can change this derived list without changing response behavior.
        # The capability token excludes index-derived fields.
        document_a = build_capabilities(
            capability_version="dset_whatever",
            build=version._EXEMPLAR_BUILD,
            code_index_stamp=version._EXEMPLAR_CODE_INDEX_STAMP,
            declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
        document_b = build_capabilities(
            capability_version="dset_whatever",
            build=version._EXEMPLAR_BUILD,
            code_index_stamp=replace(
                version._EXEMPLAR_CODE_INDEX_STAMP,
                code_counts={"futureCodeSystem": 3},
            ),
            declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
        assert document_a["decode_code_systems"] != document_b["decode_code_systems"]

        token_a = version._compute_datasets_capability_version(capability_document=document_a)
        token_b = version._compute_datasets_capability_version(capability_document=document_b)

        assert token_a == token_b

    @pytest.mark.parametrize(
        ("level_axis", "changed_fields"),
        [
            (False, ("summable_axes",)),
            (True, ("summable_axes", "groupby_level_axes")),
        ],
    )
    def test_summable_and_level_axes_changes_do_not_move_the_token(
        self, level_axis: bool, changed_fields: tuple[str, ...]
    ):
        # A fresh declaration can change these derived lists without changing response behavior.
        document_a = build_capabilities(
            capability_version="dset_whatever",
            build=version._EXEMPLAR_BUILD,
            code_index_stamp=version._EXEMPLAR_CODE_INDEX_STAMP,
            declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
        document_b = build_capabilities(
            capability_version="dset_whatever",
            build=version._EXEMPLAR_BUILD,
            code_index_stamp=version._EXEMPLAR_CODE_INDEX_STAMP,
            declarations={
                "futureCodeSystem": CodeSystemDeclaration(
                    code_system="futureCodeSystem",
                    hierarchical=level_axis,
                    disjoint_from_children=level_axis,
                    summable=True,
                    absent_reading="not_declared",
                    plain_name="axis",
                )
            },
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
        for field in changed_fields:
            assert document_a[field] != document_b[field]

        token_a = version._compute_datasets_capability_version(capability_document=document_a)
        token_b = version._compute_datasets_capability_version(capability_document=document_b)

        assert token_a == token_b

    def test_code_index_change_does_not_move_the_token(self):
        # New index metadata leaves behavior unchanged, so it does not move the capability token.
        document_a = build_capabilities(
            capability_version="dset_whatever",
            build=version._EXEMPLAR_BUILD,
            code_index_stamp=version._EXEMPLAR_CODE_INDEX_STAMP,
            declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")
        document_b = build_capabilities(
            capability_version="dset_whatever",
            build=version._EXEMPLAR_BUILD,
            code_index_stamp=replace(
                version._EXEMPLAR_CODE_INDEX_STAMP,
                content_hash="a-completely-different-content-hash",
                exported_at="2030-01-01T00:00:00Z",
                code_counts={"crsPurpose": 999},
            ),
            declarations=version._EXEMPLAR_CODE_SYSTEM_DECLARATIONS,
            settings=DatasetsBuildSettings(),
        ).model_dump(mode="json")

        token_a = version._compute_datasets_capability_version(capability_document=document_a)
        token_b = version._compute_datasets_capability_version(capability_document=document_b)

        assert token_a == token_b


def test_datasets_capability_version_is_dset_prefixed_and_deterministic():
    assert version.DATASETS_CAPABILITY_VERSION.startswith("dset_")
    assert version._compute_datasets_capability_version() == version.DATASETS_CAPABILITY_VERSION


def test_datasets_capability_version_is_stable_across_process_runs():
    """The token stays stable across fresh processes with different Python hash seeds."""
    code = "from datasets_api.version import DATASETS_CAPABILITY_VERSION as v\nprint(v)\n"
    first = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )
    second = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "2"},
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.strip() == second.stdout.strip()


def test_datasets_capability_version_imports_without_the_public_site_url(
    monkeypatch: pytest.MonkeyPatch,
):
    """The capability document does not carry the website origin, so importing the token
    must not require `DFO_PUBLIC_SITE_URL`."""
    monkeypatch.delenv("DFO_PUBLIC_SITE_URL", raising=False)
    result = subprocess.run(
        [sys.executable, "-c", "import datasets_api.version"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
