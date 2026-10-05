"""Tests that a collection member is served by the members its surgered identity resolves.

`scope.facets` replaces a member's facets whole, so it can drop or change the provenance that
names the member's dataset. The member below names the imputed multilateral dataset and
resolves one of its StatVars on the `attributedAmount` five-tuple. A scope naming CRS instead
names a dataset that publishes nothing on that five-tuple, so the surgered identity refuses.
The packaged code index decides the family.
"""

from __future__ import annotations

import pytest

from code_index.shared import get_index
from datasets_api.adapter import member_resolution_key
from spec_registry import mint_collection_id
from spec_registry.identity import recompute_identity
from tests.code_index.aid_identity import (
    CRS_PROVENANCE,
    IMPUTED_PROVENANCE,
    imputed_education_aid_identity,
)

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import pinned_identity
from .conftest import FakeDatasetStore, wire_definition_for
from .test_mint_collection_id import _client, _collection_body

_PRIMARY_EDUCATION = "CRSPurposeCode11110"
_CRS_SCOPE = {"provenance": {"ref": CRS_PROVENANCE}}


def _imputed_member_identity() -> dict:
    """An identity naming the imputed dataset, with the purpose narrowed to one StatVar."""
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    purpose = next(slot for slot in identity["slots"] if slot["property"] == "crsPurpose")
    purpose["binding"] = {"state": "value", "ref": _PRIMARY_EDUCATION}
    return identity


def _member() -> tuple[str, dict]:
    identity = _imputed_member_identity()
    spec_id = recompute_identity(identity).spec_id
    return spec_id, wire_definition_for(spec_id=spec_id, identity=identity)


def _surgered_spec_id() -> str:
    identity = _imputed_member_identity()
    identity["facets"] = {"provenance": {"state": "value", "ref": CRS_PROVENANCE}}
    return recompute_identity(identity).spec_id


def _body(collection_id: str, *, members: list[dict] | None, preview: bool) -> dict:
    body = _collection_body(collection_id, members=members, scope={"facets": _CRS_SCOPE})
    if preview:
        del body["options"]
        body["preview"] = {}
    return body


def _assert_refuses_the_crs_name(resp) -> None:
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "facet_not_available", body


@pytest.mark.parametrize("preview", [False, True], ids=["mint", "preview"])
def test_a_scope_that_renames_the_dataset_resolves_the_member_again(preview: bool):
    """The sidecar member resolves one imputed StatVar. Its surgered identity names CRS,
    which publishes nothing on the member's five-tuple, so the door refuses it and registers
    no surgered row."""
    store = FakeDatasetStore()
    client = _client(
        artifact_store=FakeArtifactStore(), store=store, graph=FakeMintGraphReader(arcs={})
    )
    spec_id, member = _member()
    collection_id = mint_collection_id([spec_id])

    resp = client.post(
        "/api/datasets", json=_body(collection_id, members=[member], preview=preview)
    )

    _assert_refuses_the_crs_name(resp)
    assert _surgered_spec_id() not in store.state.spec_registry
    assert mint_collection_id([_surgered_spec_id()]) not in store.state.collection_registry


@pytest.mark.parametrize("preview", [False, True], ids=["mint", "preview"])
def test_a_registered_member_resolves_again_under_a_scope_that_renames_its_dataset(
    preview: bool,
):
    store = FakeDatasetStore()
    client = _client(
        artifact_store=FakeArtifactStore(), store=store, graph=FakeMintGraphReader(arcs={})
    )
    spec_id, member = _member()
    registered = client.post(
        "/api/datasets/register", json={"schema_version": 1, "members": [member]}
    )
    assert registered.status_code == 200, registered.text
    collection_id = registered.json()["collection"]["collection_id"]
    assert collection_id == mint_collection_id([spec_id])

    resp = client.post("/api/datasets", json=_body(collection_id, members=None, preview=preview))

    _assert_refuses_the_crs_name(resp)
    assert _surgered_spec_id() not in store.state.spec_registry


def test_the_resolution_key_holds_what_member_resolution_reads():
    index = get_index()
    named = _imputed_member_identity()

    other_scope = {
        **named,
        "entities": {"destination": ["country/UGA"]},
        "requested_window": {"kind": "latest"},
        "facets": {**named["facets"], "unit": {"state": "value", "ref": "USD"}},
    }
    assert member_resolution_key(other_scope, code_index=index) == member_resolution_key(
        named, code_index=index
    )

    unnamed = {key: value for key, value in named.items() if key != "facets"}
    renamed = {**named, "facets": {"provenance": {"state": "value", "ref": CRS_PROVENANCE}}}
    rebound = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    keys = {
        member_resolution_key(identity, code_index=index)
        for identity in (named, unnamed, renamed, rebound)
    }
    assert len(keys) == 4


def test_a_pinned_identity_is_keyed_by_its_stat_var():
    index = get_index()
    kenya = pinned_identity(stat_var="SomeStatVar", entities={"recipient": ["country/KEN"]})
    uganda = pinned_identity(stat_var="SomeStatVar", entities={"recipient": ["country/UGA"]})
    other = pinned_identity(stat_var="AnotherStatVar", entities={"recipient": ["country/KEN"]})

    assert member_resolution_key(kenya, code_index=index) == member_resolution_key(
        uganda, code_index=index
    )
    assert member_resolution_key(kenya, code_index=index) != member_resolution_key(
        other, code_index=index
    )
