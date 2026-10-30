"""`options.enrichment = "default"` mints the version's default enrichment columns, chosen from
the roles its frame holds once it is built (`datasets_api.default_enrichment`), and answers as a
mint naming those columns does. The rigs are the export tests' fetched datasets."""

from __future__ import annotations

import io
from typing import Any

import pyarrow.parquet as pq
import pytest

from artifact_store import Tier

from .test_exports import _DATASETS, _Rig

_FRAME = {"format": "parquet", "bundle": False, "layout": "frame"}
_FLOW = [
    "financingProvider_type",
    "financingRecipient_type",
    "party_class",
    "party_kind",
    "party_whole",
]


@pytest.fixture(autouse=True)
def _caller_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DFO_CALLER_HMAC_KEY", "test-caller-key")


def _file(manifest: dict[str, Any]) -> dict[str, Any]:
    [artifact] = [a for a in manifest["artifacts"] if a["rendering"].get("layout") == "frame"]
    return artifact


@pytest.mark.parametrize(
    ("dataset", "columns", "file_columns"),
    [
        pytest.param(
            "flow-binding-no-role",
            _FLOW,
            {
                "financingProvider_type",
                "financingProvider_party_kind",
                "financingRecipient_type",
                "financingRecipient_party_kind",
            },
            id="flow-binding-no-role",
        ),
        pytest.param("place", ["observationAbout_type"], {"observationAbout_type"}, id="place"),
    ],
)
def test_a_default_enrichment_mints_the_columns_of_the_roles_the_version_holds(
    dataset: str, columns: list[str], file_columns: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _Rig("fetched", monkeypatch, dataset=dataset)

    minted = _DATASETS[dataset].mint(rig.client, {**_FRAME, "enrichment": "default"})

    assert minted["content_version"] == rig.frame["content_version"]
    artifact = _file(minted)
    assert artifact["rendering"]["columns"] == columns
    assert "enrichment" not in artifact["rendering"]
    data = rig.artifact_store.get(artifact_digest=artifact["artifact_digest"], tier=Tier.working)
    assert file_columns <= set(pq.read_schema(io.BytesIO(data)).names)


@pytest.mark.parametrize("dataset", ["flow", "flow-binding-no-role", "place"])
def test_a_default_enrichment_answers_as_a_mint_naming_its_columns(
    dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _Rig("fetched", monkeypatch, dataset=dataset)
    by_default = _DATASETS[dataset].mint(rig.client, {**_FRAME, "enrichment": "default"})
    columns = _file(by_default)["rendering"]["columns"]

    named = _DATASETS[dataset].mint(rig.client, {**_FRAME, "columns": columns})

    assert named["content_version"] == by_default["content_version"]
    assert _file(named) == _file(by_default)


def test_a_first_mint_resolves_the_default_enrichment_against_the_version_it_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flow-binding-no-role identity is not yet minted on the flow rig, so this mint
    materialises and renders in one pass."""
    rig = _Rig("fetched", monkeypatch, dataset="flow")

    minted = _DATASETS["flow-binding-no-role"].mint(rig.client, {**_FRAME, "enrichment": "default"})

    assert minted["trace"]["served_from"] == "materialise"
    assert _file(minted)["rendering"]["columns"] == _FLOW
