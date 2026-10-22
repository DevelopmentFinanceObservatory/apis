"""Checks which catalogue and policy inputs change the presentation capability version."""

from __future__ import annotations

import dataclasses

from assembler.catalogue import CatalogueRelease
from presentation_api import version


def test_adding_a_constraint_id_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    template = release.constraints[0]
    new_constraint = dataclasses.replace(
        template, constraint_id=template.constraint_id + "_perturbed"
    )
    perturbed_release = dataclasses.replace(
        release, constraints=(*release.constraints, new_constraint)
    )

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_catalog_version_change_alone_does_not_move_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    perturbed_release = dataclasses.replace(
        release, catalog_version=release.catalog_version + "-mutated"
    )

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed == baseline


def test_component_schema_title_property_removal_moves_the_token(release: CatalogueRelease):
    """Removing a member's `title` property changes its schema and capability token."""
    baseline = version.compute_presentation_capability_version(release=release)
    member, schema = next(iter(sorted(release.component_schemas.items())))
    mutated_properties = dict(schema["properties"])
    del mutated_properties["title"]
    mutated_schemas = dict(release.component_schemas)
    mutated_schemas[member] = {**schema, "properties": mutated_properties}
    perturbed_release = dataclasses.replace(release, component_schemas=mutated_schemas)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_component_schema_property_type_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    member, schema = next(iter(sorted(release.component_schemas.items())))
    mutated_properties = dict(schema["properties"])
    mutated_properties["series"] = {**mutated_properties["series"], "type": "object"}
    mutated_schemas = dict(release.component_schemas)
    mutated_schemas[member] = {**schema, "properties": mutated_properties}
    perturbed_release = dataclasses.replace(release, component_schemas=mutated_schemas)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_component_schema_required_list_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    member, schema = next(iter(sorted(release.component_schemas.items())))
    mutated_schemas = dict(release.component_schemas)
    mutated_schemas[member] = {**schema, "required": [*schema["required"], "_perturbed"]}
    perturbed_release = dataclasses.replace(release, component_schemas=mutated_schemas)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_intent_grammar_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    member, schema = next(iter(sorted(release.intent_schemas.items())))
    mutated_schemas = dict(release.intent_schemas)
    mutated_schemas[member] = {**schema, "title": "mutated"}
    perturbed_release = dataclasses.replace(release, intent_schemas=mutated_schemas)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_retry_mode_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    mutated_retry_modes = dict(version.CODE_RETRY_MODE)
    code = next(iter(sorted(mutated_retry_modes)))
    mutated_retry_modes[code] = (
        "do_not_retry" if mutated_retry_modes[code] != "do_not_retry" else "retry_after_delay"
    )

    perturbed = version.compute_presentation_capability_version(
        release=release, retry_modes=mutated_retry_modes
    )

    assert perturbed != baseline


def test_code_catalogue_title_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    code, (title, detail) = next(iter(sorted(version.PRESENTATION_CODE_CATALOGUE.items())))
    mutated_catalogue = dict(version.PRESENTATION_CODE_CATALOGUE)
    mutated_catalogue[code] = (title + " (mutated)", detail)

    perturbed = version.compute_presentation_capability_version(
        release=release, code_catalogue=mutated_catalogue
    )

    assert perturbed != baseline


def test_frame_unavailable_outcome_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    mutated_outcome = dict(version._FRAME_UNAVAILABLE_OUTCOME)
    reason = next(iter(sorted(mutated_outcome)))
    code, status, retryable = mutated_outcome[reason]
    mutated_outcome[reason] = (code, status, not retryable)

    perturbed = version.compute_presentation_capability_version(
        release=release, frame_unavailable_outcome=mutated_outcome
    )

    assert perturbed != baseline


def test_presentation_capability_version_is_pres_prefixed_and_deterministic(
    release: CatalogueRelease,
):
    computed = version.compute_presentation_capability_version(release=release)
    assert computed.startswith("pres_")
    assert computed == version.compute_presentation_capability_version(release=release)


def test_envelope_schema_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    mutated_schema = {**release.envelope_schema, "title": "mutated"}
    perturbed_release = dataclasses.replace(release, envelope_schema=mutated_schema)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_formatter_profile_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    mutated_profile = {**release.formatter_profile, "_test_perturbation": "mutated"}
    perturbed_release = dataclasses.replace(release, formatter_profile=mutated_profile)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_encoding_classes_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    field = next(iter(sorted(release.encoding_classes)))
    mutated_classes = {
        **release.encoding_classes,
        field: release.encoding_classes[field] + "_mutated",
    }
    perturbed_release = dataclasses.replace(release, encoding_classes=mutated_classes)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_grammar_members_change_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    perturbed_release = dataclasses.replace(
        release, grammar_members=(*release.grammar_members, "perturbed_member")
    )

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_constraint_retargeted_to_component_types_moves_the_token(release: CatalogueRelease):
    baseline = version.compute_presentation_capability_version(release=release)
    template = release.constraints[0]
    retargeted = dataclasses.replace(
        template, encoding_classes=(), component_types=template.encoding_classes
    )
    perturbed_release = dataclasses.replace(
        release, constraints=(retargeted, *release.constraints[1:])
    )

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_retry_mode_change_for_a_platform_code_moves_the_token(release: CatalogueRelease):
    """`timeout` is a platform code outside `PRESENTATION_CODE_CATALOGUE`, so
    flipping it reaches the hash only through the release document's own
    `retry_modes` field: the `retry_modes` parameter passed to
    `compute_presentation_capability_version` is threaded straight into
    that field."""
    baseline = version.compute_presentation_capability_version(release=release)
    mutated_retry_modes = dict(version.CODE_RETRY_MODE)
    mutated_retry_modes["timeout"] = "correct_and_retry"

    perturbed = version.compute_presentation_capability_version(
        release=release, retry_modes=mutated_retry_modes
    )

    assert perturbed != baseline


def test_component_schema_description_reword_moves_the_token(release: CatalogueRelease):
    """/api/presentation/release serves each component schema's own
    `description` whole, so a reword is client-observable and must move
    the token through the release document input."""
    baseline = version.compute_presentation_capability_version(release=release)
    member, schema = next(iter(sorted(release.component_schemas.items())))
    mutated_schemas = dict(release.component_schemas)
    mutated_schemas[member] = {**schema, "description": "a reworded description"}
    perturbed_release = dataclasses.replace(release, component_schemas=mutated_schemas)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline


def test_component_schema_nested_description_reword_moves_the_token(
    release: CatalogueRelease,
):
    """A `description` annotation on a property's own schema object, not
    just on the member schema's own top level, reaches the release
    document whole and so moves the token too."""
    baseline = version.compute_presentation_capability_version(release=release)
    member, schema = next(iter(sorted(release.component_schemas.items())))
    mutated_properties = dict(schema["properties"])
    mutated_properties["category_axis"] = {
        **mutated_properties["category_axis"],
        "description": "a reworded description",
    }
    mutated_schemas = dict(release.component_schemas)
    mutated_schemas[member] = {**schema, "properties": mutated_properties}
    perturbed_release = dataclasses.replace(release, component_schemas=mutated_schemas)

    perturbed = version.compute_presentation_capability_version(release=perturbed_release)

    assert perturbed != baseline
