"""Tests for spec_registry.identity: the adapter over dfo_spec_kit.identity
that recomputes a spec_id from a wire definition's identity-core document.
"""

from __future__ import annotations

import hashlib

import pytest
from dfo_spec_kit.identity import IdentityError, canonicalize, mint_spec_id

from spec_registry.identity import _MAX_IDENTITY_DOCUMENT_BYTES, recompute_identity

from .conftest import identity_of_encoded_size, minimal_identity, shape_for


def test_recompute_identity_mints_the_same_id_the_kit_would() -> None:
    identity_definition = minimal_identity(
        shape=shape_for("developmentFinanceFlow"),
        slots=[
            {
                "property": "DevelopmentFinancePurpose",
                "binding": {"state": "value", "ref": "Malaria"},
            }
        ],
        entities={"provider": ["country/FRA"]},
    )

    recomputed = recompute_identity(identity_definition)

    expected_canonical = canonicalize(
        {
            "identity_version": "spec1",
            "shape": shape_for("developmentFinanceFlow"),
            "slots": [
                {
                    "property": "DevelopmentFinancePurpose",
                    "binding": {"state": "value", "ref": "Malaria"},
                }
            ],
            "entities": {"provider": ["country/FRA"]},
            "requested_window": {"kind": "all"},
        }
    )
    assert recomputed.identity_bytes == expected_canonical
    assert recomputed.spec_id == mint_spec_id(expected_canonical)
    assert recomputed.identity_digest == hashlib.sha256(expected_canonical).hexdigest()
    assert recomputed.spec_id == f"spec_{recomputed.identity_digest[:32]}"


def test_recompute_identity_is_deterministic() -> None:
    identity_definition = minimal_identity()

    first = recompute_identity(identity_definition)
    second = recompute_identity(identity_definition)

    assert first == second


def test_recompute_identity_propagates_identity_error_unwrapped() -> None:
    invalid_definition = minimal_identity(identity_version="spec2")

    with pytest.raises(IdentityError) as excinfo:
        recompute_identity(invalid_definition)
    assert excinfo.value.category == "unsupported_identity_version"


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(
            {"schema_version": 1, "spec_id": "spec_whatever", "identity": minimal_identity()},
            id="wire_envelope",
        ),
        pytest.param(
            {
                **minimal_identity(entities={"origin": ["country/FRA"]}),
                "entity_groups": {"origin": {"label": "France"}},
            },
            id="entity_groups",
        ),
    ],
)
def test_recompute_identity_refuses_a_member_outside_the_projection(document) -> None:
    # Only the identity-core document itself feeds the hash -- a caller
    # accidentally handing in the full wire envelope (spec_id/schema_version
    # alongside `identity`) must refuse (unknown_member), not silently hash
    # the envelope. This pins the exact byte-compare target: the identity
    # projection, never the wire definition. A named group is request context
    # beside an identity, so the document refuses it the same way.
    with pytest.raises(IdentityError) as excinfo:
        recompute_identity(document)
    assert excinfo.value.category == "unknown_member"


def test_recompute_identity_accepts_a_document_exactly_at_the_byte_cap() -> None:
    at_cap = identity_of_encoded_size(encoded_bytes=_MAX_IDENTITY_DOCUMENT_BYTES)

    recomputed = recompute_identity(at_cap)

    assert recomputed.spec_id.startswith("spec_")


def test_recompute_identity_refuses_a_document_one_byte_over_the_cap() -> None:
    over_cap = identity_of_encoded_size(encoded_bytes=_MAX_IDENTITY_DOCUMENT_BYTES + 1)

    with pytest.raises(IdentityError) as excinfo:
        recompute_identity(over_cap)
    assert excinfo.value.category == "input_limit_exceeded"
