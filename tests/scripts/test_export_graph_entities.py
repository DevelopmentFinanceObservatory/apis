"""Pins the exporter's record merge, which folds a pinned source record into its target."""

from __future__ import annotations

import pytest

from scripts.export_graph_entities import merge_records


def _record(dcid: str, *, name: str = "Org", aliases: tuple[str, ...] = (), dac_code=None):
    return {
        "dcid": dcid,
        "name": name,
        "kind": "organisation",
        "aliases": list(aliases),
        "dac_code": dac_code,
    }


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [
                _record("Other", name="Other"),
                _record("Source", aliases=("X", "Z")),
                _record("Target", aliases=("Y", "X"), dac_code="896"),
            ],
            [
                _record("Other", name="Other"),
                _record("Target", aliases=("Y", "X", "Z"), dac_code="896"),
            ],
            id="aliases-and-source-removal",
        ),
        pytest.param(
            [_record("Source", dac_code="896"), _record("Target")],
            [_record("Target", dac_code="896")],
            id="source-dac-code-fallback",
        ),
    ],
)
def test_merge_records_preserves_target_fields_and_drops_source(records, expected):
    assert merge_records(records, {"Source": "Target"}) == expected


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([_record("Target")], id="missing-source"),
        pytest.param([_record("Source")], id="missing-target"),
        pytest.param([_record("Source"), _record("Target", name="Renamed")], id="renamed"),
        pytest.param(
            [_record("Source", dac_code="1"), _record("Target", dac_code="2")],
            id="conflicting-dac-code",
        ),
    ],
)
def test_merge_fails_loudly_when_the_pin_no_longer_holds(records):
    with pytest.raises(SystemExit, match="Source -> Target"):
        merge_records(records, {"Source": "Target"})
