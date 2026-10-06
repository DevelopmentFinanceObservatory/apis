"""These tests pin `CurrencyRoleUndeclared` pointers across sidecar mints and stored-spec
retrieval.

Every refusal also reports the StatVar's declared roles as `alternatives`.
"""

from __future__ import annotations

from spec_registry.identity import recompute_identity

from ._identity_builders import pinned_identity, register_body
from .conftest import wire_definition_for
from .test_mint_error_classification import _STAT_VAR, _client, _spec_id_body

_BAD_UNIT_FACET = {
    "unit": {
        "state": "value",
        "request": {"currency": {"role": "origin"}, "prices": "current"},
    }
}


def test_sidecar_mint_refuses_request_invalid_with_the_spec_id_pointer():
    client = _client()
    identity = pinned_identity(stat_var=_STAT_VAR, facets=_BAD_UNIT_FACET)
    spec_id = recompute_identity(identity).spec_id
    definition = wire_definition_for(spec_id=spec_id, identity=identity)

    resp = client.post("/api/datasets", json=_spec_id_body(spec_id, definition=definition))

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/spec_id"
    assert problem["alternatives"] == [{"ref": "recipient", "label": "recipient"}]


def test_stored_spec_retrieval_refuses_request_invalid_with_no_pointer():
    client = _client()
    identity = pinned_identity(stat_var=_STAT_VAR, facets=_BAD_UNIT_FACET)
    spec_id = recompute_identity(identity).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )
    assert reg.status_code == 200, reg.text

    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert "pointer" not in problem
    assert "issues" not in problem or problem["issues"] is None
    assert problem["alternatives"] == [{"ref": "recipient", "label": "recipient"}]
