"""Worked, minimal, and constraint fixtures validate against the release schemas."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from assembler.mint.validate import component_registry

_MEMBERS = ("line_chart", "bar_chart", "table", "stat_card", "note_narration", "source_card")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    ("source", "member"),
    [
        pytest.param(source, member, id=f"{source}-{member}")
        for source in ("worked", "minimal")
        for member in _MEMBERS
    ],
)
def test_intent_variants_validate_against_strict_schema(release, catalogue_dir, source, member):
    if source == "worked":
        intent = _read_json(catalogue_dir / "fixtures" / member / "intent.json")
    else:
        intent = release.minimal_intents[member]

    jsonschema.Draft202012Validator(release.intent_schemas[member]).validate(intent)


@pytest.mark.parametrize(
    ("member", "fixture_name"),
    [
        *((member, "payload.json") for member in _MEMBERS),
        ("stat_card", "payload_composed_unit.json"),
        ("table", "payload_composed_unit.json"),
    ],
)
def test_worked_payload_validates_against_envelope_and_component_schema(
    release, catalogue_dir, member, fixture_name
):
    payload = _read_json(catalogue_dir / "fixtures" / member / fixture_name)

    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(payload)
    jsonschema.Draft202012Validator(release.component_schemas[member]).validate(
        payload["component"]
    )
