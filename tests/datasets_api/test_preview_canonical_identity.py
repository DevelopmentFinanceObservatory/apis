"""A preview materialises the canonical identity its mint registers and reads.

The spec kit hashes a facet bound as a `value` and as a `set` of one ref to one `spec_id`. A
sidecar may post either spelling, and the preview and the mint of that sidecar serve the same
result, at the `spec_id` door and for a collection member.
"""

from __future__ import annotations

import copy

import pytest

from spec_registry.collection_id import mint_collection_id

from . import test_mint_collection_id as collection_door
from . import test_mint_spec_id as spec_door
from ._artifact_doubles import FakeArtifactStore
from .conftest import FakeDatasetStore

_SPELLINGS = ("value", "set")


def _respelled(definition: dict, *, spelling: str) -> dict:
    """`definition` with its provenance facet bound in `spelling`, under the same `spec_id`."""
    definition = copy.deepcopy(definition)
    binding = definition["identity"]["facets"]["provenance"]
    assert binding == {"state": "value", "ref": "OECD_DAC2A"}
    if spelling == "set":
        definition["identity"]["facets"]["provenance"] = {"state": "set", "refs": [binding["ref"]]}
    return definition


def _preview_and_mint(make_client, body: dict) -> tuple[dict, dict]:
    """The preview entry and the mint manifest of `body`, each against an empty registry."""
    previewed = make_client().post(
        "/api/datasets", json={k: v for k, v in body.items() if k != "options"} | {"preview": {}}
    )
    minted = make_client().post("/api/datasets", json=body)
    assert previewed.status_code == 200, previewed.text
    assert minted.status_code == 200, minted.text
    (entry,) = previewed.json()["members"]
    return entry, minted.json()


def _served(entry: dict, manifest: dict) -> tuple[str, int]:
    assert entry["content_version"] == manifest["content_version"]
    assert entry["total_rows"] == manifest["coverage"]["rows"]
    return entry["content_version"], entry["total_rows"]


@pytest.mark.parametrize("spelling", _SPELLINGS)
def test_a_sidecar_previews_as_it_mints_under_either_provenance_spelling(spelling: str):
    graph = spec_door._graph()
    spec_id, definition = spec_door._sidecar_for(graph=graph)
    body = spec_door._spec_id_body(spec_id, definition=_respelled(definition, spelling=spelling))

    entry, manifest = _preview_and_mint(
        lambda: spec_door._client(
            artifact_store=FakeArtifactStore(), store=FakeDatasetStore(), graph=graph
        ),
        body,
    )

    assert manifest["definition"]["spec_id"] == spec_id
    assert _served(entry, manifest)[1] == 1


def test_both_provenance_spellings_of_a_sidecar_serve_one_result():
    graph = spec_door._graph()
    spec_id, definition = spec_door._sidecar_for(graph=graph)

    served = {
        _served(
            *_preview_and_mint(
                lambda: spec_door._client(
                    artifact_store=FakeArtifactStore(), store=FakeDatasetStore(), graph=graph
                ),
                spec_door._spec_id_body(
                    spec_id, definition=_respelled(definition, spelling=spelling)
                ),
            )
        )
        for spelling in _SPELLINGS
    }

    assert len(served) == 1


@pytest.mark.parametrize("spelling", _SPELLINGS)
def test_a_collection_member_previews_as_it_mints_under_either_provenance_spelling(
    spelling: str,
):
    graph = collection_door._graph()
    spec_id, member = collection_door._member_sidecar(
        collection_door._STAT_VAR_REF, entities={"recipient": ["country/KEN"]}, graph=graph
    )
    body = collection_door._collection_body(
        mint_collection_id([spec_id]), members=[_respelled(member, spelling=spelling)]
    )

    entry, manifest = _preview_and_mint(
        lambda: collection_door._client(
            artifact_store=FakeArtifactStore(), store=FakeDatasetStore(), graph=graph
        ),
        body,
    )

    assert entry["stat_var"] == collection_door._STAT_VAR_REF
    assert entry["total_rows"] == manifest["coverage"]["rows"]
