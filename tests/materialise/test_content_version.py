"""Tests content-version minting and recomputation against grammar vectors."""

from __future__ import annotations

import importlib.resources
import json

from dfo_frame_grammar import canonical_frame_bytes

from materialise.content_version import mint_version, mint_version_from_chunks, recompute_version
from materialise.frame_build import build_frame


def _load_vector(name: str) -> dict:
    path = importlib.resources.files("dfo_frame_grammar").joinpath(f"vectors/frame/{name}")
    return json.loads(path.read_text())


def test_mint_version_and_recompute_match_installed_golden_vector():
    vector = _load_vector("frame-basic-single-row.json")
    frame = build_frame(role_keys=vector["role_keys"], rows=vector["rows"])

    assert mint_version(frame) == vector["expected_content_version"]
    assert recompute_version(canonical_frame_bytes(frame)) == vector["expected_content_version"]


def test_mint_version_from_chunks_matches_mint_version_over_the_concatenated_frame():
    """Hashing sorted, validated chunks matches the whole frame across a chunk boundary."""
    row_a = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/AAA",
        "time_period": "2015",
        "value": 1000.0,
    }
    row_b = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": "country/BBB",
        "time_period": "2015",
        "value": 2000.0,
    }
    whole_frame = build_frame(role_keys=("recipient",), rows=[row_b, row_a])
    chunk_a = build_frame(role_keys=("recipient",), rows=[row_a])
    chunk_b = build_frame(role_keys=("recipient",), rows=[row_b])

    assert mint_version_from_chunks((chunk_a, chunk_b)) == mint_version(whole_frame)
