"""The arc tables answer node reads with the values the graph gives.

`tests/fixtures/recorded_base_arcs.jsonl` holds 34 base nodes' `->*` answers
as base Data Commons gave them, drawn from a 1,000-variable sample: 30
variables the build lists (`"table": "base"`), three with no node, three
whose answers repeat a target and four with a denominator, and the four
denominators as schema nodes (`"table": "schema"`). One origin client reads
them live from a mixer serving those answers; another reads them from arc
tables built from the same answers and passed through the artifact's
encoding. Every read agrees, as value sets per property.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from dc_graph import GraphClient
from graph_generation import FakeGeneration
from origin_catalogue import BASE, CatalogueSnapshot
from origin_catalogue.arc_tables import ArcTables
from origin_catalogue.artifact import decode_arc_tables, encode_build
from origin_catalogue.graph import OriginGraphClient, UnGeneration
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue
from tests.origin_catalogue.support import build_arc_tables

_RECORDED = Path(__file__).parents[1] / "fixtures" / "recorded_base_arcs.jsonl"
_PROVENANCE = "dc/base/Recorded"
_UNLISTED = "Count_Person_NotInTheBuild"
_HOST = "graph.example.org"


def _recorded() -> dict[str, dict[str, dict[str, dict[str, list[dict]]]]]:
    """Each recorded node's `->*` answer in the wire shape of `/v2/node`, by arc table."""
    tables: dict[str, dict[str, dict[str, dict[str, list[dict]]]]] = {"base": {}, "schema": {}}
    for line in _RECORDED.read_text().splitlines():
        row = json.loads(line)
        tables[row["table"]][row["variable"]] = {
            prop: {"nodes": targets} for prop, targets in row["arcs"].items()
        }
    return tables


_TABLES = _recorded()
_ANSWERS = {**_TABLES["base"], **_TABLES["schema"]}
_VARIABLES = sorted(_TABLES["base"])
_NODES = sorted(_ANSWERS)


class _Mixer:
    """Our mixer serving the recorded answers to `->*`, `->prop` and `->[a,b]` reads,
    and `_UNLISTED` as a variable named for itself. It records every dcid it is sent."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.answers = {
            **_ANSWERS,
            _UNLISTED: {"name": {"nodes": [{"value": _UNLISTED}]}},
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        spec = payload["property"].removeprefix("->")
        wanted = None if spec == "*" else spec.strip("[]").split(",")
        data: dict = {}
        for dcid in payload["nodes"]:
            self.sent.append(dcid)
            arcs = self.answers.get(dcid, {})
            if wanted is not None:
                arcs = {prop: arcs[prop] for prop in wanted if prop in arcs}
            data[dcid] = {"arcs": arcs} if arcs else {}
        return httpx.Response(200, json={"data": data})


def _client(mixer: _Mixer) -> OriginGraphClient:
    transport = httpx.MockTransport(mixer.handler)
    return OriginGraphClient(
        ours=GraphClient(
            base_url=f"https://{_HOST}/core/api",
            transport=transport,
            generation=FakeGeneration(token="rev-1"),
        ),
        un=GraphClient(
            base_url="https://un.example.org/core/api",
            transport=transport,
            generation=UnGeneration(),
        ),
    )


def _catalogue(arcs: ArcTables | None) -> CatalogueSnapshot:
    """Lists every recorded variable as base, with `arcs` as its arc tables once encoded
    and decoded, or lists none and holds no arc table."""
    if arcs is None:
        return restrictive_catalogue(provenances=[provenance_row(_PROVENANCE, BASE)])
    listed = restrictive_catalogue(
        provenances=[provenance_row(_PROVENANCE, BASE)],
        base_variables={variable: ([_PROVENANCE], ()) for variable in _VARIABLES},
    )
    build = encode_build(
        listed.tables,
        build_id="recorded",
        built_at=datetime(2026, 10, 8, tzinfo=UTC),
        build_seconds=0.0,
        identity_inputs={},
        registry_hash="",
        arcs=arcs,
    )
    return restrictive_catalogue(
        provenances=[provenance_row(_PROVENANCE, BASE)],
        base_variables={variable: ([_PROVENANCE], ()) for variable in _VARIABLES},
        arcs=decode_arc_tables(build.files, label="recorded"),
    )


def _read_all(client: OriginGraphClient, dcids: list[str]) -> dict[str, object]:
    """Every node read the client answers from tables, over `dcids`, with arcs as value
    sets per property."""
    properties = sorted({prop for arcs in _ANSWERS.values() for prop in arcs})
    arcs = client.node_arcs_batch(dcids)
    return {
        "arcs": {
            dcid: None
            if found is None
            else {
                prop: {node.get("dcid") or node.get("value") for node in entry["nodes"]}
                for prop, entry in found.items()
            }
            for dcid, found in arcs.items()
        },
        "arcs_many": {
            dcid: None if found is None else {prop: set(values) for prop, values in found.items()}
            for dcid, found in client.node_arcs_many(dcids, properties=properties).items()
        },
        "labels": client.node_labels_batch(dcids),
        "types": client.node_types_batch(dcids),
        "names": {dcid: set(names) for dcid, names in client.node_names_many(dcids).items()},
        "observation_properties": client.node_observation_properties_batch(dcids),
        "exists": client.exists_batch(dcids),
    }


def _arcs() -> ArcTables:
    return build_arc_tables(dfo={}, base=_TABLES["base"], schema=_TABLES["schema"])


def test_the_tables_answer_every_recorded_node_as_the_graph_does(bind_catalogue) -> None:
    arcs = _arcs()
    with _client(live_mixer := _Mixer()) as live_client:
        bind_catalogue(_catalogue(None))
        expected = _read_all(live_client, _NODES)
    with _client(table_mixer := _Mixer()) as table_client:
        bind_catalogue(_catalogue(arcs))
        answered = _read_all(table_client, _NODES)

    assert answered == expected
    assert sum(found is None for found in answered["arcs"].values()) == 3
    assert set(live_mixer.sent) == set(_NODES)
    assert table_mixer.sent == []


def test_a_variable_absent_from_the_build_resolves_live(bind_catalogue) -> None:
    """A dcid outside the tables is out of their scope, so it reads from the graph,
    alone in a batch with held variables."""
    arcs = _arcs()
    mixer = _Mixer()
    bind_catalogue(_catalogue(arcs))
    with _client(mixer) as client:
        found = client.node_arcs_batch([_VARIABLES[0], _UNLISTED])
        label = client.node_label(_UNLISTED)

    assert found[_UNLISTED] == {"name": {"nodes": [{"value": _UNLISTED}]}}
    assert label == _UNLISTED
    assert mixer.sent == [_UNLISTED, _UNLISTED]
