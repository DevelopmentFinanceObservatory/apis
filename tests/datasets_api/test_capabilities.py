"""Check published capability values against the constants and dispatch tables they describe."""

from __future__ import annotations

from dataclasses import replace

import datasets_api.render as render_module
from code_index.shared import get_index
from code_index.types import CodeSystemDeclaration, level_axes, summable_axes
from dataset_store.snapshot import WORKING_TIER_RETENTION
from datasets_api.capabilities import (
    _LIVE_RENDERING_FORMATS,
    ENABLED_ENRICHMENTS,
    Capabilities,
    build_capabilities,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.models import MAX_COLLECTION_MEMBERS
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_platform.build import build_commit
from materialise.facets import DEPLOYMENT_FACET_POLICY, DEPLOYMENT_FACET_POLICY_VERSION
from statvar_members.enumerate import CANDIDATE_CAP

_SETTINGS = DatasetsBuildSettings()


def _capabilities() -> Capabilities:
    return build_capabilities(
        capability_version=DATASETS_CAPABILITY_VERSION,
        build=build_commit(),
        code_index_stamp=get_index().stamp,
        declarations=get_index().declarations,
        settings=_SETTINGS,
    )


class TestPublishedCapsEqualEnforcedValues:
    def test_caps_equal_config_and_model_constants(self):
        caps = _capabilities().caps
        assert caps.max_collection_members == MAX_COLLECTION_MEMBERS
        assert caps.definition_candidate_cap == CANDIDATE_CAP
        assert caps.sync_budget_rows == _SETTINGS.sync_budget_rows
        assert caps.sync_budget_requests == _SETTINGS.sync_budget_requests
        assert caps.async_max_rows == _SETTINGS.async_max_rows
        assert caps.async_max_requests == _SETTINGS.async_max_requests
        assert caps.pivot_max_output_cells == _SETTINGS.pivot_max_output_cells
        assert caps.operation_retention_days == WORKING_TIER_RETENTION.days

    def test_operation_envelope_is_active(self):
        assert _capabilities().operation_envelope_active is True

    def test_capability_version_and_build_are_the_real_computed_values(self):
        document = _capabilities()
        assert document.capability_version == DATASETS_CAPABILITY_VERSION
        assert document.build == build_commit()

    def test_code_index_matches_the_loaded_index(self):
        stamp = get_index().stamp
        code_index = _capabilities().code_index
        assert code_index.content_hash == stamp.content_hash
        assert code_index.exported_at == stamp.exported_at
        assert code_index.code_counts == dict(stamp.code_counts)


class TestMintDoorLivenessMatchesWhatActuallyDispatches:
    """Published liveness flags must match the mint dispatch for each input kind and format."""

    def test_input_kind_liveness(self):
        kinds = {k.name: k.live for k in _capabilities().input_kinds}
        assert kinds == {
            "spec_id": True,
            "collection_id": True,
            "statvar_set": True,
            "source_codes": True,
        }

    def test_format_liveness(self):
        formats = {f.name: f.live for f in _capabilities().formats}
        assert formats == {"csv": True, "xlsx": True, "parquet": True}

    def test_live_rendering_formats_matches_the_render_dispatch_table(self):
        # Compare the hand-maintained set with the actual render dispatch table.
        assert frozenset(render_module._FORMAT_WRITERS) == _LIVE_RENDERING_FORMATS

    def test_bundle_is_live_pinning_is_not(self):
        document = _capabilities()
        assert document.bundle_active is True
        assert document.programmatic_pinning is False

    def test_recipe_classes_lists_the_conversion_recipe_members(self):
        # Keep the expected members literal so a wrong constant fails this test.
        assert _capabilities().recipe_classes == [
            "ConversionRecipeDAC",
            "ConversionRecipeIDS",
            "ConversionRecipeIMF",
        ]

    def test_enrichment_fields_matches_the_enabled_set(self):
        assert _capabilities().enrichment_fields == sorted(ENABLED_ENRICHMENTS)

    def test_enrichment_fields_derives_from_the_constant_not_a_hardcoded_literal(self, monkeypatch):
        # Changing the constant proves the published list derives from it.
        import datasets_api.capabilities as capabilities_module

        monkeypatch.setattr(capabilities_module, "ENABLED_ENRICHMENTS", frozenset({"iso3"}))
        caps = _capabilities()
        assert caps.enrichment_fields == ["iso3"]

    def test_decode_code_systems_matches_the_committed_indexs_own_declarations(self):
        # Compare against declarations directly, not the matching `code_counts` keys.
        assert _capabilities().decode_code_systems == sorted(get_index().declarations)

    def test_decode_code_systems_derives_from_the_stamp_not_a_hardcoded_literal(self):
        # A custom stamp proves the published list reads `code_counts` directly.
        stamp = replace(get_index().stamp, code_counts={"futureCodeSystem": 3, "crsPurpose": 1})
        caps = build_capabilities(
            capability_version=DATASETS_CAPABILITY_VERSION,
            build=build_commit(),
            code_index_stamp=stamp,
            declarations=get_index().declarations,
            settings=_SETTINGS,
        )
        assert caps.decode_code_systems == ["crsPurpose", "futureCodeSystem"]

    def test_revision_and_default_facet_policy_liveness_are_false(self):
        document = _capabilities()
        assert document.revision_provenance_live is False
        assert document.default_facet_policies_live is False

    def test_default_facet_policy_names_the_deployment_policy_even_though_no_graph_declared_one_is_live(
        self,
    ):
        # The deployment default resolves unbound axes without a graph-declared policy.
        policy = _capabilities().default_facet_policy
        assert policy is not None
        assert policy.policy == DEPLOYMENT_FACET_POLICY
        assert policy.version == DEPLOYMENT_FACET_POLICY_VERSION


def _declaration(
    code_system: str, *, summable: bool, hierarchical: bool = False, disjoint: bool = True
) -> CodeSystemDeclaration:
    return CodeSystemDeclaration(
        code_system=code_system,
        hierarchical=hierarchical,
        disjoint_from_children=disjoint,
        summable=summable,
        absent_reading="not_declared",
        plain_name="axis",
    )


class TestSummableAxes:
    """Test that declared summability controls published axes.

    All four CRS systems are summable, so a synthetic false declaration checks exclusion.
    """

    def test_all_four_real_crs_systems_are_summable(self):
        declarations = get_index().declarations
        assert {name for name, d in declarations.items() if d.summable} == {
            "crsPurpose",
            "crsTypeOfFinance",
            "crsCooperationModality",
            "crsTypeOfFlow",
        }

    def test_a_declared_summable_system_is_included(self):
        declarations = {"crsPurpose": _declaration("crsPurpose", summable=True)}
        assert summable_axes(declarations) == frozenset({"crsPurpose"})

    def test_a_declared_not_summable_system_is_excluded(self):
        declarations = {
            "crsPurpose": _declaration("crsPurpose", summable=True),
            "syntheticNotSummable": _declaration("syntheticNotSummable", summable=False),
        }
        assert summable_axes(declarations) == frozenset({"crsPurpose"})

    def test_an_undeclared_system_is_excluded_the_same_as_a_declared_false_one(self):
        # An undeclared axis is excluded like a declaration with `summable=False`.
        declarations = {"crsPurpose": _declaration("crsPurpose", summable=True)}
        assert "undeclaredAxis" not in summable_axes(declarations)


class TestGroupbyLevelAxes:
    """A level axis is declared summable, hierarchical, and disjoint from children."""

    def test_the_published_level_axes_are_the_real_hierarchical_crs_systems(self):
        assert _capabilities().groupby_level_axes == [
            "crsCooperationModality",
            "crsPurpose",
            "crsTypeOfFinance",
        ]

    def test_a_flat_unsummable_or_overlapping_system_is_no_level_axis(self):
        declarations = {
            "crsPurpose": _declaration("crsPurpose", summable=True, hierarchical=True),
            "flatSummable": _declaration("flatSummable", summable=True),
            "treeNotSummable": _declaration("treeNotSummable", summable=False, hierarchical=True),
            # A parent-coded record that includes its children would count them twice.
            "treeOverlapping": _declaration(
                "treeOverlapping", summable=True, hierarchical=True, disjoint=False
            ),
        }
        assert level_axes(declarations) == frozenset({"crsPurpose"})
