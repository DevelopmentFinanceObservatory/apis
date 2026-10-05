"""Tests code-index label overrides across single and batch graph lookups."""

from dataclasses import dataclass

import dc_graph
import search.engine.clients.graph_client as graph_client_module
from graph_generation.stamp import UNKNOWN_KEY
from search.engine.clients.graph_client import LiveGraphClient


@dataclass(frozen=True)
class _FakeCodeRow:
    label: str


class _FakeCodeIndex:
    """Implements the `code_by_dcid` method used by the label override."""

    def __init__(self, rows: dict[str, str]):
        self._rows = rows

    def code_by_dcid(self, dcid: str) -> _FakeCodeRow | None:
        label = self._rows.get(dcid)
        return _FakeCodeRow(label=label) if label is not None else None


def test_node_labels_batch_prefers_code_index_label_over_raw_graph_name(monkeypatch) -> None:
    """A code-index label replaces the graph label with its code-system suffix removed."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "CRSTypeOfFinance110": {
                    "arcs": {"name": {"nodes": [{"value": "standard grant (CRS Finance)"}]}}
                }
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    monkeypatch.setattr(
        graph_client_module,
        "get_index",
        lambda: _FakeCodeIndex({"CRSTypeOfFinance110": "standard grant"}),
    )
    client = LiveGraphClient(base="http://fake")

    result = client.node_labels_batch(["CRSTypeOfFinance110"])

    assert result == {"CRSTypeOfFinance110": "standard grant"}


def test_node_labels_batch_leaves_non_code_dcid_label_unchanged(monkeypatch) -> None:
    """Dcids absent from the code index keep the graph label."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "country/KEN": {"arcs": {"name": {"nodes": [{"value": "Kenya"}]}}},
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    monkeypatch.setattr(
        graph_client_module,
        "get_index",
        lambda: _FakeCodeIndex({"CRSTypeOfFinance110": "standard grant"}),
    )
    client = LiveGraphClient(base="http://fake")

    result = client.node_labels_batch(["country/KEN"])

    assert result == {"country/KEN": "Kenya"}


def test_node_label_and_node_labels_batch_agree_when_label_populates_cache(monkeypatch) -> None:
    """Single and batch lookups share the cache and code-index label."""
    post_called = []

    def fake_post(self, path, payload, *, deadline=None):
        post_called.append(payload)
        return {
            "data": {
                "CRSTypeOfFinance110": {
                    "arcs": {"name": {"nodes": [{"value": "standard grant (CRS Finance)"}]}}
                }
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    monkeypatch.setattr(
        graph_client_module,
        "get_index",
        lambda: _FakeCodeIndex({"CRSTypeOfFinance110": "standard grant"}),
    )
    client = LiveGraphClient(base="http://fake")

    single = client.node_label("CRSTypeOfFinance110")
    assert single == "standard grant"
    assert len(post_called) == 1

    batch = client.node_labels_batch(["CRSTypeOfFinance110"])
    assert batch == {"CRSTypeOfFinance110": "standard grant"}
    assert len(post_called) == 1, "the cache node_label populated must serve node_labels_batch"
    assert single == batch["CRSTypeOfFinance110"]


def test_node_label_and_node_labels_batch_agree_when_batch_populates_cache(monkeypatch) -> None:
    """Batch and single lookups share the cache and code-index label."""
    post_called = []

    def fake_post(self, path, payload, *, deadline=None):
        post_called.append(payload)
        return {
            "data": {
                "CRSTypeOfFinance110": {
                    "arcs": {"name": {"nodes": [{"value": "standard grant (CRS Finance)"}]}}
                }
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    monkeypatch.setattr(
        graph_client_module,
        "get_index",
        lambda: _FakeCodeIndex({"CRSTypeOfFinance110": "standard grant"}),
    )
    client = LiveGraphClient(base="http://fake")

    batch = client.node_labels_batch(["CRSTypeOfFinance110"])
    assert batch == {"CRSTypeOfFinance110": "standard grant"}
    assert len(post_called) == 1

    single = client.node_label("CRSTypeOfFinance110")
    assert single == "standard grant"
    assert len(post_called) == 1, "the cache node_labels_batch populated must serve node_label"
    assert single == batch["CRSTypeOfFinance110"]


def test_node_labels_batch_index_override_applies_to_cached_labels_too(monkeypatch) -> None:
    """The code-index override also corrects labels already in cache."""

    def fake_post(self, path, payload, *, deadline=None):
        raise AssertionError("no POST expected: dcid is already cached")

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    monkeypatch.setattr(
        graph_client_module,
        "get_index",
        lambda: _FakeCodeIndex({"CRSTypeOfFinance110": "standard grant"}),
    )
    client = LiveGraphClient(base="http://fake")
    client._client._ours._label_cache[(UNKNOWN_KEY, "CRSTypeOfFinance110")] = (
        "standard grant (CRS Finance)"
    )

    result = client.node_labels_batch(["CRSTypeOfFinance110"])

    assert result == {"CRSTypeOfFinance110": "standard grant"}
