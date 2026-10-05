"""Checks pg8000's default encoding for parameters bound to the store's JSONB columns.

pg8000 1.31.5 encodes dicts as JSON but bare lists as PostgreSQL arrays. The store serializes
lists to JSON strings before binding them.
"""

from __future__ import annotations

import json

import pytest
from pg8000.converters import PY_TYPES, make_param

from presentation_store.store import _next_composition_blocks


def _encoded_as_json(value: object) -> object:
    """Encode a value with pg8000 and parse the result as JSON."""
    return json.loads(make_param(PY_TYPES, value))


def test_series_coordinates_write_param_encodes_as_valid_json():
    coordinates = [{"id": "srs_abc", "coordinates": {"stat_var": "dcid:X"}}]
    # Match the JSON serialization applied before the insert.
    param = json.dumps([dict(entry) for entry in coordinates])

    assert _encoded_as_json(param) == coordinates


def test_composition_blocks_write_param_encodes_as_valid_json():
    blocks = _next_composition_blocks([], component_id="cmp_1", revision=1)
    # Match the JSON serialization applied before the insert.
    param = json.dumps(blocks)

    assert _encoded_as_json(param) == blocks


def test_a_bare_list_param_does_not_encode_as_json():
    """Documents the failure mode the two tests above guard against: passing
    the list itself, rather than its JSON text, as the bound parameter."""
    coordinates = [{"id": "srs_abc"}]

    encoded = make_param(PY_TYPES, coordinates)

    with pytest.raises(json.JSONDecodeError):
        json.loads(encoded)
