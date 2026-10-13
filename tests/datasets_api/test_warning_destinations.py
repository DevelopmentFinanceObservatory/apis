"""Where a new mint reports the six warning codes the description reroutes.

`processing_level_undeclared` and `origin_catalogue_stale` move to the manifest's
`diagnostics`. `stat_vars_summed` and `facet_default_selected` become steps of the version's
description record, and `deflator_projected` an `estimated` caveat, so none of them is a
manifest warning on a new mint. `revision_unavailable` has no emitter and survives only in stored
manifests. All six stay registered while a stored manifest can serve them: in their Datasets
catalogue, in the card envelope's `_WARNING_KIND_BY_CODE`, and active in the kit registry.
`accept-datasets.md` holds the table of all fifteen Datasets warning codes.
"""

from __future__ import annotations

import pytest
from dfo_spec_kit.registry import REGISTRY

from assembler.mint.provenance import _WARNING_KIND_BY_CODE
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from materialise.bundle import BUNDLE_CODE_CATALOGUE
from origin_catalogue.service import get_catalogue_service

from . import test_manifest_facet_warnings as facet_fixture
from ._description_doubles import (
    GROUPBY_OPTIONS,
    datasets_client,
    mint_definition,
    mint_family,
)
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_remint_returns_persisted_version import _StaleCatalogue

REROUTED = (
    "revision_unavailable",
    "processing_level_undeclared",
    "origin_catalogue_stale",
    "stat_vars_summed",
    "facet_default_selected",
    "deflator_projected",
)


def _require_diagnostics() -> None:
    from datasets_api.manifest_models import DiagnosticModel  # noqa: F401


def _codes(manifest: dict, member: str) -> set[str]:
    return {entry["code"] for entry in manifest[member]}


def _client(store: FakeVersionRecordDatasetStore, **kwargs):
    _require_diagnostics()
    return datasets_client(store, with_description_store=False, **kwargs)


def test_a_new_mint_reports_an_undeclared_processing_level_as_a_diagnostic() -> None:
    manifest = mint_definition(_client(FakeVersionRecordDatasetStore()))

    assert "processing_level_undeclared" not in _codes(manifest, "warnings")
    assert "processing_level_undeclared" in _codes(manifest, "diagnostics")


def test_a_new_mint_reports_a_stale_origin_catalogue_as_a_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(FakeVersionRecordDatasetStore())
    monkeypatch.setattr(get_catalogue_service(), "snapshot", lambda: _StaleCatalogue())

    manifest = mint_definition(client)

    assert "origin_catalogue_stale" not in _codes(manifest, "warnings")
    assert "origin_catalogue_stale" in _codes(manifest, "diagnostics")


def test_a_new_collapse_reports_its_sum_as_a_step_and_not_a_warning() -> None:
    store = FakeVersionRecordDatasetStore()
    client = _client(store, fixture="family")

    manifest = mint_family(client, options=GROUPBY_OPTIONS)
    snapshot = client.get(
        f"/api/datasets/{manifest['definition']['spec_id']}/{manifest['content_version']}"
    )

    assert snapshot.status_code == 200, snapshot.text
    for served in (manifest, snapshot.json()):
        assert "stat_vars_summed" not in _codes(served, "warnings")
        assert "stat_vars_summed" not in _codes(served, "diagnostics")


def test_a_new_mint_reports_its_choice_of_source_as_a_step_and_not_a_warning() -> None:
    _require_diagnostics()
    facets = {
        facet_fixture._STAT_VAR_A: [facet_fixture._WINNING_FACET, facet_fixture._DISAGREEING_FACET]
    }
    client = facet_fixture._client(
        graph=facet_fixture._graph(facets),
        csv_by_stat_var=facet_fixture._csv_by_stat_var(facets),
        coverage=facet_fixture._coverage_service(facets),
    )

    response = client.post(
        "/api/datasets", json=facet_fixture._body(statvars=[facet_fixture._REF_A])
    )

    assert response.status_code == 200, response.text
    manifest = response.json()
    assert facet_fixture._facet_entry(manifest, "provenance")["basis"] == "deployment_fallback"
    assert "facet_default_selected" not in _codes(manifest, "warnings")
    assert "facet_default_selected" not in _codes(manifest, "diagnostics")


def test_the_manifest_schema_declares_a_diagnostics_member() -> None:
    _require_diagnostics()
    from datasets_api.manifest_models import Manifest

    assert "diagnostics" in Manifest.model_json_schema()["properties"]


@pytest.mark.parametrize("code", REROUTED)
def test_a_rerouted_code_stays_registered_while_stored_manifests_serve_it(code: str) -> None:
    """`processing_level_undeclared` is catalogued with the bundle's codes, the rest with the
    Datasets warnings."""
    _require_diagnostics()
    assert code in DATASETS_WARNING_CATALOGUE or code in BUNDLE_CODE_CATALOGUE
    assert code in _WARNING_KIND_BY_CODE
    (entry,) = [entry for entry in REGISTRY.entries if entry.code == code]
    assert entry.lifecycle == "active"
    assert ("datasets.snapshot", "manifest_warning") in {
        (mapping.endpoint, mapping.channel) for mapping in entry.mappings
    }
