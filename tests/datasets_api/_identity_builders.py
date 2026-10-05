"""Build register requests and minimal identities for datasets API tests.
The identity helpers cover pinned `stat_var` and described `statvar_schema` shapes.
"""

from __future__ import annotations


def register_body(*, definition: dict) -> dict:
    """Wrap a canonical definition in the register-request envelope."""
    return {"schema_version": 1, "definition": definition}


def pinned_identity(*, stat_var: str = "TestStatVar", **overrides: object) -> dict:
    """Build a minimal spec1 identity with the pinned `stat_var` shape.

    Tests can override fields such as `stat_var`.
    """
    doc: dict = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": stat_var},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    doc.update(overrides)
    return doc


def described_identity(**overrides: object) -> dict:
    """Build a minimal described-arm identity without `measured_property`.

    It resolves as `spec_not_exportable`. Enumeration tests must supply a measured property, and
    usually a population type, that the graph holds.
    """
    doc: dict = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "population_type": "Person"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    doc.update(overrides)
    return doc
