"""Checks that the production extract rung matches the shipped prompt constant."""

from __future__ import annotations

import json

import jsonschema

from promptlab.compose import compose
from promptlab.manifest import PROMPTLAB_ROOT, load_rung
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT

EXTRACT_MANIFEST = PROMPTLAB_ROOT / "rungs" / "production-extract.json"
BIND_MANIFEST = PROMPTLAB_ROOT / "rungs" / "production-bind.json"
SCHEMA_PATH = PROMPTLAB_ROOT / "rung.schema.json"


def test_production_extract_composes_byte_for_byte_to_the_shipped_constant():
    rung = load_rung(EXTRACT_MANIFEST, BIND_MANIFEST)
    assert compose(rung.extract) == _EXTRACTION_SYSTEM_PROMPT


def test_production_extract_manifest_validates_against_the_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(json.loads(EXTRACT_MANIFEST.read_text()), schema)
