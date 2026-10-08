"""Pin release loading, strict and tolerant schemas, and the baked registry."""

from __future__ import annotations

import pytest

from assembler.catalogue import BakedReleaseRegistry, load_release
from assembler.catalogue.constraints import CONSTRAINT_REGISTRY

_MEMBERS = ("line_chart", "bar_chart", "table", "stat_card", "note_narration", "source_card")
_GRAMMAR_MEMBERS = _MEMBERS


def test_load_release_is_not_run_at_import_time():
    """Importing the package does not load catalogue files."""
    import assembler.catalogue  # noqa: F401 -- import alone must not raise or read files


def test_load_release_populates_every_pinned_field(catalogue_dir):
    release = load_release(catalogue_dir=catalogue_dir)
    assert release.catalog_version == "2026.10.0"
    assert release.schema_version == 1
    assert release.envelope_schema["$id"].endswith("envelope.schema.json")
    assert set(release.component_schemas) == set(_MEMBERS)
    assert set(release.intent_schemas) == set(_MEMBERS)
    assert set(release.intent_schemas_tolerant) == set(_MEMBERS)
    assert set(release.minimal_intents) == set(_MEMBERS)
    assert set(release.grammar_members) == set(_GRAMMAR_MEMBERS)
    assert release.encoding_classes == {
        "line_chart": "position",
        "bar_chart": "position",
        "table": "none",
        "stat_card": "none",
        "note_narration": "none",
        "source_card": "none",
    }
    assert release.constraints == CONSTRAINT_REGISTRY
    assert "significant_digits" in release.formatter_profile
    assert release.retry_modes["schema_invalid"] == "correct_and_retry"
    assert release.retry_modes["release_mismatch"] == "retry_after_delay"


def test_tolerant_intent_schemas_declare_no_required_fields(release):
    for member in _MEMBERS:
        assert release.intent_schemas_tolerant[member]["required"] == []


@pytest.mark.parametrize(
    ("member", "required_fields"),
    [
        pytest.param("line_chart", {"type", "frame_ref", "select"}, id="line_chart"),
        pytest.param("bar_chart", {"type", "frame_ref", "select"}, id="bar_chart"),
        pytest.param("table", {"type", "frame_ref", "select"}, id="table"),
        pytest.param("stat_card", {"type", "frame_ref", "select"}, id="stat_card"),
        pytest.param("note_narration", {"type", "text"}, id="note_narration"),
        pytest.param("source_card", {"type", "frame_ref"}, id="source_card"),
    ],
)
def test_strict_intent_schemas_require_their_member_fields(release, member, required_fields):
    assert set(release.intent_schemas[member]["required"]) == required_fields


def test_tolerant_schema_admits_null_for_a_clearable_field(release):
    import jsonschema

    schema = release.intent_schemas_tolerant["line_chart"]
    jsonschema.Draft202012Validator(schema).validate({"window": None})
    jsonschema.Draft202012Validator(schema).validate({"title": None})


def test_tolerant_schema_refuses_null_for_non_clearable_fields(release):
    import jsonschema

    schema = release.intent_schemas_tolerant["line_chart"]
    for field in ("type", "frame_ref", "select"):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate({field: None})


def test_baked_release_registry_pinned_matches_load_release(catalogue_dir):
    registry = BakedReleaseRegistry(catalogue_dir=catalogue_dir)
    assert registry.pinned() == load_release(catalogue_dir=catalogue_dir)


def test_default_catalogue_dir_resolves_the_installed_package():
    """Without a directory override, `load_release()` reads the installed package."""
    pytest.importorskip("dfo_spec_kit_components")
    release = load_release()
    assert release.catalog_version
