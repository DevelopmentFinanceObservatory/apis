"""The committed eval build answers recall's hits as our mixer does.

`tests/fixtures/recorded_resolver_hit_arcs.json` holds what our dev mixer answered for 500
variables recall finds for the corpus queries, 250 DFO and 250 base
(`scripts/record_resolver_hit_arcs.py`): each variable's `->*` arcs as value sets per
property, its label and type, and, for every node those arcs name, its label and type and the
names the arcs carry for it. The arcs name a node with no name by its dcid and a node of
several names by its first, where its label is none or its last name. The build in
`tests/fixtures/origin_catalogue_eval` must answer each from its tables with the same value,
or leave it to a live read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from origin_catalogue import CatalogueSnapshot
from origin_catalogue.artifact import load_catalogue
from origin_catalogue.graph import _table_arcs, _table_label, _table_type, _Unmirrored
from scripts.stage_origin_catalogue_fixture import stage_fixture

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_RECORDED: dict[str, dict[str, dict[str, Any]]] = json.loads(
    (_FIXTURES / "recorded_resolver_hit_arcs.json").read_text()
)


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory: pytest.TempPathFactory) -> CatalogueSnapshot:
    return load_catalogue(
        stage_fixture(_FIXTURES / "origin_catalogue_eval", tmp_path_factory.mktemp("eval"))
    )


def _value_sets(arcs: Mapping[str, dict] | None) -> dict[str, list[str]] | None:
    if arcs is None:
        return None
    return {
        prop: sorted({node.get("dcid") or node.get("value") for node in entry["nodes"]})
        for prop, entry in arcs.items()
    }


def _answer(found: object) -> object:
    """A table read's answer as the live read gives it: an omitted fact is `None`."""
    return None if found is _Unmirrored.OMITTED else found


def test_the_recording_holds_250_dfo_and_250_base_variables_the_build_lists(catalogue) -> None:
    variables = _RECORDED["variables"]
    listed = {
        "dfo": catalogue.tables.dfo_variables,
        "base": catalogue.tables.base_variables.keys(),
    }

    assert sorted(entry["origin"] for entry in variables.values()) == ["base"] * 250 + ["dfo"] * 250
    assert all(dcid in listed[entry["origin"]] for dcid, entry in variables.items())


def test_the_tables_answer_every_recorded_variable_as_the_mixer_does(catalogue) -> None:
    for dcid, entry in _RECORDED["variables"].items():
        arcs = _table_arcs(catalogue, dcid)
        assert not isinstance(arcs, _Unmirrored), dcid
        assert _value_sets(arcs) == entry["arcs"], dcid
        assert _answer(_table_label(catalogue, dcid)) == entry["label"], dcid
        assert _answer(_table_type(catalogue, dcid)) == entry["type"], dcid


def test_the_tables_label_and_type_the_nodes_recorded_arcs_name_as_the_mixer_does(
    catalogue,
) -> None:
    mislabelled = set()
    for dcid, entry in _RECORDED["targets"].items():
        found_type = _table_type(catalogue, dcid)
        assert found_type is _Unmirrored.LIVE or _answer(found_type) == entry["type"], dcid
        label = _table_label(catalogue, dcid)
        if label is not _Unmirrored.LIVE and _answer(label) != entry["label"]:
            mislabelled.add(dcid)

    assert len(_RECORDED["targets"]) == 623
    assert mislabelled == set()


def test_the_tables_arcs_name_each_node_as_the_mixer_s_arcs_do(catalogue) -> None:
    targets = _RECORDED["targets"]
    names: dict[str, set[str]] = {}
    for dcid in _RECORDED["variables"]:
        arcs = _table_arcs(catalogue, dcid)
        assert isinstance(arcs, dict), dcid
        for entry in arcs.values():
            for node in entry["nodes"]:
                if node.get("dcid") in targets:
                    found = names.setdefault(node["dcid"], set())
                    if node.get("name"):
                        found.add(node["name"])

    assert {dcid: sorted(found) for dcid, found in names.items()} == {
        dcid: entry["arc_names"] for dcid, entry in targets.items()
    }
    # The recording holds nodes whose arcs and labels name them otherwise.
    assert targets["Years0"]["arc_names"] == ["Years0"] and targets["Years0"]["label"] is None
    assert targets["Alcohol"]["arc_names"] == ["Alcohol"]
    assert targets["Alcohol"]["label"] == "Alcohol (Drug)"
