"""Tests registry schema conformance and rejection of malformed data."""

from __future__ import annotations

import json
from importlib import resources

import jsonschema
import pytest

from search.engine.region import equivalent_series
from search.engine.region.equivalent_series import EquivalentSeriesRegistryError


def _shipped_document() -> dict:
    text = (
        resources.files(equivalent_series._PACKAGE) / equivalent_series._DATA_FILENAME
    ).read_text(encoding="utf-8")
    return json.loads(text)


def _shipped_schema() -> dict:
    text = (
        resources.files(equivalent_series._PACKAGE) / "equivalent_series.schema.json"
    ).read_text(encoding="utf-8")
    return json.loads(text)


def _write_registry(tmp_path, document: object) -> None:
    (tmp_path / equivalent_series._DATA_FILENAME).write_text(json.dumps(document), encoding="utf-8")


def test_shipped_registry_validates_against_schema():
    jsonschema.Draft202012Validator(_shipped_schema()).validate(_shipped_document())


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(EquivalentSeriesRegistryError, match="not found"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_invalid_json(tmp_path):
    (tmp_path / equivalent_series._DATA_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(EquivalentSeriesRegistryError, match="invalid JSON"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_document_missing_format_version(tmp_path):
    _write_registry(tmp_path, {"entries": []})
    with pytest.raises(EquivalentSeriesRegistryError, match="format_version"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_boolean_format_version(tmp_path):
    # Python treats True as 1, so the loader must reject booleans explicitly.
    _write_registry(
        tmp_path,
        {
            "format_version": True,
            "entries": [{"members": ["a/1", "a/2"], "note": "n"}],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="format_version"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_unknown_top_level_key(tmp_path):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [{"members": ["a/1", "a/2"], "note": "n"}],
            "junk": 1,
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="junk"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_non_list_entries(tmp_path):
    _write_registry(tmp_path, {"format_version": 1, "entries": "not-a-list"})
    with pytest.raises(EquivalentSeriesRegistryError, match="entries must be a non-empty array"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_empty_entries_list(tmp_path):
    _write_registry(tmp_path, {"format_version": 1, "entries": []})
    with pytest.raises(EquivalentSeriesRegistryError, match="entries must be a non-empty array"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_entry_that_is_not_an_object(tmp_path):
    _write_registry(tmp_path, {"format_version": 1, "entries": ["not-an-object"]})
    with pytest.raises(EquivalentSeriesRegistryError, match=r"entries\[0\] must be an object"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_unknown_entry_key(tmp_path):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [{"members": ["a/1", "a/2"], "note": "n", "junk": 1}],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="junk"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_cluster_with_fewer_than_two_members(tmp_path):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [{"members": ["only/one"], "note": "not a twin"}],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="at least two"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_non_hashable_member(tmp_path):
    # Reject the list as a registry error before `set(members_raw)` can raise TypeError.
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [{"members": ["a/1", ["b"]], "note": "n"}],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="distinct, non-empty strings"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_rejects_empty_note(tmp_path):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [{"members": ["a/1", "a/2"], "note": ""}],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="note"):
        equivalent_series._load_clusters(directory=tmp_path)


@pytest.mark.parametrize("preferred", ["a/3", 1])
def test_loader_rejects_a_preferred_dcid_outside_the_members(tmp_path, preferred):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [{"members": ["a/1", "a/2"], "note": "n", "preferred": preferred}],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="preferred"):
        equivalent_series._load_clusters(directory=tmp_path)


@pytest.mark.parametrize("currency", ["usd", "XYZ", 1])
def test_loader_rejects_a_currency_outside_iso_4217(tmp_path, currency):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [{"members": ["a/1", "a/2"], "note": "n", "currency": currency}],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="currency"):
        equivalent_series._load_clusters(directory=tmp_path)


def test_loader_reads_the_preferred_member_and_currency(tmp_path):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [
                {"members": ["a/1", "a/2"], "note": "n", "preferred": "a/2", "currency": "USD"},
                {"members": ["b/1", "b/2"], "note": "n"},
            ],
        },
    )
    clusters = equivalent_series._load_clusters(directory=tmp_path)
    assert [(cluster.preferred, cluster.currency) for cluster in clusters] == [
        ("a/2", "USD"),
        (None, None),
    ]


def test_loader_rejects_dcid_shared_across_clusters(tmp_path):
    _write_registry(
        tmp_path,
        {
            "format_version": 1,
            "entries": [
                {"members": ["shared/dcid", "a/2"], "note": "cluster one"},
                {"members": ["shared/dcid", "b/2"], "note": "cluster two"},
            ],
        },
    )
    with pytest.raises(EquivalentSeriesRegistryError, match="shared/dcid"):
        equivalent_series._load_clusters(directory=tmp_path)
