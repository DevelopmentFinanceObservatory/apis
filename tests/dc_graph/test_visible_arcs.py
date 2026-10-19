"""`dc_graph.visible_arcs` gives the mixer path and the release path one arc set.

The fixture holds the Spanner `Edge` rows (joined to `Node`) of five IDS variables, whose
`constraintProperties` carry the import-added `sourceIndicatorCode`, and the dev mixer's `/v2/node`
`->*` response for the same variables. Its `source` field names where and when it was recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from dc_graph import MIXER_BLOCKED_PREDICATES, GraphClient, parse_arcs, visible_mask
from dc_graph.visible_arcs import mixer_target

_FIXTURE = Path(__file__).parents[1] / "fixtures/dc_graph/visible_arcs/ids_variables.json"

type Arc = tuple[str, str, str]


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(_FIXTURE.read_text())


def _target(row: dict) -> str:
    return mixer_target(row["object_id"], value=row["value"], types=row["types"])


def _release_arcs(rows: list[dict]) -> set[Arc]:
    table = pa.Table.from_pylist(rows)
    kept = table.filter(
        visible_mask(table["predicate"].combine_chunks(), table["object_id"].combine_chunks())
    )
    return {(row["subject_id"], row["predicate"], _target(row)) for row in kept.to_pylist()}


def _mixer_arcs(recorded: dict, monkeypatch: pytest.MonkeyPatch) -> set[Arc]:
    def fake_post(self, path, payload, *, deadline=None):
        assert payload["property"] == "->*"
        return recorded["mixer_response"]

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")
    arcs = client.node_arcs_batch(recorded["variables"])
    return {
        (dcid, prop, target)
        for dcid, node_arcs in arcs.items()
        for prop, targets in parse_arcs(node_arcs or {}, list(node_arcs or {})).items()
        for target in targets
    }


def test_mixer_and_release_paths_give_equal_arc_sets(
    recorded: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    mixer = _mixer_arcs(recorded, monkeypatch)
    release = _release_arcs(recorded["spanner_rows"])

    hidden = {(v, "constraintProperties", "sourceIndicatorCode") for v in recorded["variables"]}
    raw = {(r["subject_id"], r["predicate"], _target(r)) for r in recorded["spanner_rows"]}
    assert hidden <= raw, "the fixture must carry the import-added property on every variable"
    assert release == mixer
    assert not hidden & release


def test_mixer_visible_drops_blocked_predicates_and_keeps_targets_without_a_node_row(
    recorded: dict,
) -> None:
    rows = recorded["spanner_rows"]
    without_node = [r for r in rows if r["value"] is None and r["name"] is None and not r["types"]]
    assert without_node, "the fixture must hold an edge to a target with no Node row"
    blocked = [
        {**without_node[0], "predicate": predicate}
        for predicate in sorted(MIXER_BLOCKED_PREDICATES)
    ]

    kept = _release_arcs(without_node + blocked)

    assert kept == {(r["subject_id"], r["predicate"], r["object_id"]) for r in without_node}


@pytest.mark.parametrize(
    ("predicate", "object_id", "visible"),
    [
        ("constraintProperties", "sourceIndicatorCode", False),
        ("constraintProperties", "financialFlowType", True),
        ("sourceIndicatorCode", "sourceIndicatorCode", True),
        ("linkedMemberOf", "dc/g/Root", False),
        (None, "sourceIndicatorCode", True),
        ("constraintProperties", None, True),
    ],
)
def test_visible_mask_over_plain_and_dictionary_columns(
    predicate: str | None, object_id: str | None, visible: bool
) -> None:
    predicates = pa.array([predicate], pa.string())
    object_ids = pa.array([object_id], pa.string())
    for encode in (False, True):
        columns = (predicates, object_ids)
        if encode:
            columns = tuple(column.dictionary_encode() for column in columns)
        assert visible_mask(*columns).to_pylist() == [visible]
