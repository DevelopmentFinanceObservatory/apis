"""Tests for spec_registry.collection_id: server-side col1 minting over
the goldens in fixtures/collection_id_vectors.json (dedupe, sort/NFC
code-point order, singleton, empty-members refusal) -- kit-shaped, byte-exact
goldens, the same discipline tests/engine/test_identity_vectors.py applies
to the identity vector suite proper, but for the collection projection
dfo-spec-kit does not itself ship a helper for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dfo_spec_kit import identity as kit_identity

from spec_registry.collection_id import (
    collection_id_projection,
    mint_collection_id,
)
from spec_registry.errors import EmptyCollectionMembers

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "collection_id_vectors.json"
_VECTORS = json.loads(_FIXTURE_PATH.read_bytes())


@pytest.mark.parametrize("vector", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_collection_id_vector(vector: dict) -> None:
    if vector["classification"] == "negative":
        with pytest.raises(EmptyCollectionMembers):
            mint_collection_id(vector["members"])
        return

    projection = collection_id_projection(vector["members"])
    assert projection == vector["expected_projection"]

    canonical = bytes.fromhex(vector["canonical_bytes"])
    collection_id = mint_collection_id(vector["members"])
    assert collection_id == vector["collection_id"]

    # The projection's own canonical bytes, recomputed independently here,
    # must match the pinned golden -- proves the fixture and the minting
    # code agree on the exact JCS serialisation, not just the final id.
    assert kit_identity.canonicalize(projection) == canonical


def test_mint_collection_id_sorts_and_deduplicates() -> None:
    members = ["spec_bbb", "spec_aaa", "spec_bbb"]
    assert mint_collection_id(members) == mint_collection_id(["spec_aaa", "spec_bbb"])


def test_mint_collection_id_empty_members_refuses() -> None:
    with pytest.raises(EmptyCollectionMembers):
        mint_collection_id([])
