"""Tests code catalogue merging, registry lookups, and unknown-code responses."""

from __future__ import annotations

from typing import get_args

import pytest
from dfo_spec_kit.registry import REGISTRY
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dfo_apis.app import all_surfaces
from dfo_platform.app import create_app
from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, CodeCollision, merge_catalogues
from search.models import NoDataReason


def test_merge_disjoint_catalogues():
    a = {"code_a": ("Code A", "detail a")}
    b = {"code_b": ("Code B", "detail b")}
    merged = merge_catalogues(a, b)
    assert merged == {"code_a": ("Code A", "detail a"), "code_b": ("Code B", "detail b")}


def test_merge_raises_on_duplicate_code_with_differing_entries():
    a = {"dup": ("Title A", "detail a")}
    b = {"dup": ("Title B", "detail b")}
    with pytest.raises(CodeCollision) as exc_info:
        merge_catalogues(a, b)
    assert "dup" in str(exc_info.value)


def test_merge_raises_on_duplicate_code_even_with_identical_entries():
    a = {"dup": ("Same Title", "same detail")}
    b = {"dup": ("Same Title", "same detail")}
    with pytest.raises(CodeCollision) as exc_info:
        merge_catalogues(a, b)
    assert "dup" in str(exc_info.value)


def test_merge_with_no_catalogues_is_empty():
    assert merge_catalogues() == {}


def test_platform_code_catalogue_entries_are_non_empty_lower_snake_case_pairs():
    assert PLATFORM_CODE_CATALOGUE  # non-empty registry
    for code, entry in PLATFORM_CODE_CATALOGUE.items():
        assert code == code.lower()
        assert " " not in code
        assert code.strip("_") == code  # no leading/trailing underscore
        assert code.replace("_", "").isalnum()
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        title, detail = entry
        assert isinstance(title, str) and title
        assert isinstance(detail, str) and detail


@pytest.fixture(scope="module")
def app() -> FastAPI:
    return create_app(all_surfaces())


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _merged_catalogue() -> dict[str, tuple[str, str]]:
    return merge_catalogues(PLATFORM_CODE_CATALOGUE, *(s.codes for s in all_surfaces()))


def test_a_registry_only_no_data_reason_resolves_with_no_title(client: TestClient):
    """`entity_not_resolved` resolves through the `no_data_reason` channel with null title and detail."""
    resp = client.get("/codes/entity_not_resolved")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] == "refusal"
    assert any(mapping["channel"] == "no_data_reason" for mapping in body["mappings"])
    assert body["title"] is None
    assert body["detail"] is None


def test_a_warning_classified_registry_code_resolves(client: TestClient):
    resp = client.get("/codes/place_scope_defaulted")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] == "warning"


def test_a_code_in_both_sources_resolves_with_every_field(client: TestClient):
    resp = client.get("/codes/request_invalid")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] == "refusal"
    assert body["title"]
    assert body["detail"]
    assert body["mappings"]


def test_a_graph_catalogue_code_resolves_with_the_registry_backed_body(client: TestClient):
    """`node_not_held` is Graph's own catalogue code and also a registry
    entry, mapped onto `graph.node`."""
    resp = client.get("/codes/node_not_held")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] == "refusal"
    assert body["title"]
    assert body["detail"]
    assert body["mappings"]


def test_an_unknown_code_is_404_code_not_found(client: TestClient):
    resp = client.get("/codes/this_code_exists_nowhere")
    assert resp.status_code == 404
    assert resp.json()["code"] == "code_not_found"


def test_no_merged_catalogue_code_is_absent_from_the_registry():
    """Every merged platform and surface code has a packaged registry entry."""
    missing = {code for code in _merged_catalogue() if code not in REGISTRY}
    assert missing == set()


def test_every_no_data_reason_is_a_registry_code():
    """A part's `no_data.reason` is a registry code, resolvable at /codes/{code}."""
    assert {reason for reason in get_args(NoDataReason) if reason not in REGISTRY} == set()


@pytest.mark.parametrize("code", sorted(REGISTRY.codes))
def test_every_registry_code_resolves_200(client: TestClient, code: str):
    resp = client.get(f"/codes/{code}")
    assert resp.status_code == 200
    assert resp.json()["code"] == code
