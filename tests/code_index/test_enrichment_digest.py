"""`CodeIndex.enrichment_digest` moves with a party's kind or whole markers and with nothing else,
so an export that changes neither re-renders no party-kind file."""

from __future__ import annotations

import json
from typing import Any

import pytest

from code_index import CodeIndex, parse_index

from .conftest import FIXTURE_PATH

_DAC2A = "provenance/OECD_DAC2A"
_DAC2B = "provenance/OECD_DAC2B"
_WHOLE = [{"dataset": _DAC2A, "holds": ["party"], "years": [1960, 2023]}]


def _party(dcid: str, kind: str, **extra: Any) -> dict[str, Any]:
    return {
        "dcid": dcid,
        "kind": kind,
        "totals": None,
        "datasets": [_DAC2A],
        "axes": ["origin"],
        **extra,
    }


_ROWS = [
    _party("Gates", "party"),
    _party("DACOfficialDonors", "total", datasets=[_DAC2A, _DAC2B], whole=_WHOLE),
    _party("Africa", "total", totals="africa"),
]


def _index(parties: list[dict[str, Any]], **payload_overrides: Any) -> CodeIndex:
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["parties"] = parties
    payload.update(payload_overrides)
    return parse_index(payload)


def _replaced(row: int, **changes: Any) -> list[dict[str, Any]]:
    return [{**party, **changes} if at == row else party for at, party in enumerate(_ROWS)]


def test_the_digest_ignores_row_order_and_everything_outside_kind_and_whole_markers():
    baseline = _index(_ROWS).enrichment_digest

    assert _index(list(reversed(_ROWS))).enrichment_digest == baseline
    assert _index(_replaced(0, label="Gates Foundation")).enrichment_digest == baseline
    assert _index(_replaced(0, datasets=[_DAC2A, "provenance/OECD_DAC2B"])).enrichment_digest == (
        baseline
    )
    assert _index(_replaced(2, totals="Africa")).enrichment_digest == baseline
    restamped = json.loads(FIXTURE_PATH.read_text())["stamp"] | {"content_hash": "another-export"}
    assert _index(_ROWS, stamp=restamped).enrichment_digest == baseline


def _marked(**marker: Any) -> list[dict[str, Any]]:
    return _replaced(1, datasets=[_DAC2A, _DAC2B], whole=[{**_WHOLE[0], **marker}])


@pytest.mark.parametrize(
    "changed",
    [
        _replaced(0, kind="private"),
        _marked(holds=["party", "residual"]),
        _marked(years=[1960, 2022]),
        _marked(years=None),
        _marked(dataset=_DAC2B),
        _replaced(1, datasets=[_DAC2A, _DAC2B], whole=[*_WHOLE, {**_WHOLE[0], "dataset": _DAC2B}]),
        _replaced(1, whole=[]),
        _ROWS[:2],
    ],
    ids=[
        "kind",
        "holds",
        "window",
        "window_dropped",
        "dataset",
        "marker_added",
        "markers_dropped",
        "party_dropped",
    ],
)
def test_the_digest_moves_when_a_kind_a_marker_or_a_window_changes(changed: list[dict[str, Any]]):
    assert _index(changed).enrichment_digest != _index(_ROWS).enrichment_digest
