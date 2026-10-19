"""Pins the DFO layer: the release-backed graph reader, the base parties, the DFO parts and the
declarations.

The reader is checked against the dev mixer's recorded `/v2/node` answer for five IDS variables
(`tests/fixtures/dc_graph/visible_arcs/ids_variables.json`). The parts are built from the
recorded slice with `tests.graph_release.recorded`'s declarations and parties: two parties the
release answers and two only the mixer does.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pytest

from code_index import declarations as code_index_declarations
from dc_graph import GraphClient, parse_arcs
from graph_release import declarations as release_declarations
from graph_release.declarations import DECLARATIONS, Declarations
from graph_release.dfo import (
    ReleaseGraphReader,
    base_party_ids,
    build_dfo_parts,
    read_base_parties,
    stage_dfo_parts,
)
from graph_release.format import conform_part
from graph_release.platform import PlatformRefusal
from tests.graph_release.recorded import (
    MIXER_PARTIES,
    SLICE_BASE_PARTIES,
    SLICE_DECLARATIONS,
    RecordedMixer,
    platform_parts,
    slice_platform_parts,
)

_IDS_FIXTURE = Path(__file__).parents[1] / "fixtures/dc_graph/visible_arcs/ids_variables.json"
_CRS = "provenance/OECDCreditorReportingSystem"


@pytest.fixture(scope="module")
def ids() -> dict:
    return json.loads(_IDS_FIXTURE.read_text())


@pytest.fixture(scope="module")
def ids_reader(ids: dict) -> ReleaseGraphReader:
    rows = pa.Table.from_pylist(ids["spanner_rows"])
    targets = {row["object_id"]: row for row in ids["spanner_rows"]}
    ends = sorted({*ids["variables"], *targets})
    nodes = pa.table(
        {
            "subject_id": ends,
            "name": [targets.get(end, {}).get("name") for end in ends],
            "types": pa.array(
                [targets.get(end, {}).get("types") or [] for end in ends], pa.list_(pa.string())
            ),
            "value": [targets.get(end, {}).get("value") for end in ends],
            "authored": [False] * len(ends),
        }
    )
    return ReleaseGraphReader(
        edges=conform_part(
            rows.select(["subject_id", "predicate", "object_id", "provenance"]), name="edges"
        ),
        nodes=conform_part(nodes, name="nodes"),
    )


@pytest.fixture
def mixer(ids: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    def fake_post(self, path, payload, *, deadline=None):
        return ids["mixer_response"]

    monkeypatch.setattr(GraphClient, "post", fake_post)
    return GraphClient(base_url="http://fake/core/api").node_arcs_batch(ids["variables"])


def test_reader_answers_each_arc_as_the_mixer_does(ids, ids_reader, mixer) -> None:
    def entries(arcs: dict) -> set[tuple[str, str, str]]:
        return {
            (prop, entry.get("dcid") or entry["value"], entry["provenanceId"])
            for prop, nodes in arcs.items()
            for entry in nodes["nodes"]
        }

    release = ids_reader.node_arcs_batch(ids["variables"])
    properties = sorted({prop for arcs in mixer.values() for prop in arcs})
    many = ids_reader.node_arcs_many(ids["variables"], properties=properties)

    for variable in ids["variables"]:
        assert entries(release[variable]) == entries(mixer[variable])
        assert many[variable] == {
            prop: tuple(sorted(values))
            for prop, values in parse_arcs(mixer[variable], properties).items()
        }
        assert "sourceIndicatorCode" not in many[variable]["constraintProperties"]


def test_reader_answers_nodes_without_edges_from_their_node_rows() -> None:
    parts = platform_parts()
    reader = ReleaseGraphReader(edges=parts.edges, nodes=parts.nodes)

    arcs = reader.node_arcs_many(["Count_Person", "absent"], properties=["typeOf", "name"])

    assert arcs == {
        "Count_Person": {"typeOf": ("StatisticalVariable",), "name": ("Total population",)},
        "absent": None,
    }


def test_reader_pages_subjects_and_narrows_them_by_filters() -> None:
    parts = platform_parts()
    reader = ReleaseGraphReader(edges=parts.edges, nodes=parts.nodes)
    every = reader.node_subjects("CRSTypeOfFlowEnum", property="typeOf", max_results=100)

    page = reader.node_subjects("CRSTypeOfFlowEnum", property="typeOf", max_results=3)
    narrowed = reader.node_subjects(
        "FinancialFlow",
        property="populationType",
        filters={"crsTypeOfFlow": "CRSTypeOfFlow10", "measuredProperty": "attributedAmount"},
        max_results=100,
    )

    assert len(every.dcids) > 3 and not every.truncated
    assert page.dcids == every.dcids[:3] and page.truncated
    assert narrowed.dcids and all(dcid.startswith("AttributedAmount_") for dcid in narrowed.dcids)
    assert not narrowed.truncated


def test_base_parties_are_the_parties_the_release_does_not_answer() -> None:
    mixer = RecordedMixer()

    ids = base_party_ids(slice_platform_parts(), SLICE_DECLARATIONS)
    stored = read_base_parties(list(reversed(ids)), graph=mixer)

    assert ids == tuple(sorted(MIXER_PARTIES))
    assert mixer.calls == [ids]
    assert stored == SLICE_BASE_PARTIES
    assert json.loads(stored) == {
        party: {prop: list(values) for prop, values in sorted(arcs.items())}
        for party, arcs in MIXER_PARTIES.items()
    }


def test_base_parties_let_a_mixer_error_propagate() -> None:
    class Failing:
        def node_arcs_many(self, dcids, *, properties):
            raise TimeoutError("mixer read timed out")

    with pytest.raises(TimeoutError):
        read_base_parties(["country/KGZ"], graph=Failing())


def test_dfo_parts_rebuild_byte_for_byte_and_join_coverage(tmp_path: Path) -> None:
    parts = slice_platform_parts()

    built = build_dfo_parts(parts, SLICE_DECLARATIONS, SLICE_BASE_PARTIES)
    staged = stage_dfo_parts(built, staging_dir=tmp_path, execution_id="exec-1")
    stored = {part.ref.name: part.path.read_bytes() for part in staged}
    again = build_dfo_parts(parts, SLICE_DECLARATIONS, stored["base_parties"])

    assert stored == built.blobs()
    assert again == built
    assert built.declarations == SLICE_DECLARATIONS.declarations_json()
    blocks = json.loads(gzip.decompress(built.code_index_blocks))
    assert "stamp" not in blocks
    assert {(row["dcid"].split("_")[0], *row["provenances"]) for row in blocks["statvars"]} == {
        ("Amount", _CRS),
        ("AttributedAmount", "provenance/ONEImputedMultilateralSectors"),
    }
    assert {row["dcid"]: (row["kind"], row["label"], row["axes"]) for row in blocks["parties"]} == {
        "AsianDevelopmentBank": ("party", "Asian Development Bank", ["origin"]),
        "DACAsiaUnallocated": ("residual", "Asia, regional", ["destination"]),
        "country/KGZ": ("party", "Kyrgyzstan", ["destination"]),
        "country/TJK": ("party", "Tajikistan", ["destination"]),
    }


def test_dfo_parts_refuse_a_party_neither_the_release_nor_the_mixer_answers() -> None:
    base = json.loads(SLICE_BASE_PARTIES)
    del base["country/TJK"]

    with pytest.raises(PlatformRefusal, match="country/TJK") as refused:
        build_dfo_parts(
            slice_platform_parts(), SLICE_DECLARATIONS, json.dumps(base).encode("utf-8")
        )

    assert refused.value.code == "code_index_export_refused"


@pytest.mark.parametrize(
    ("field", "empty"), [("role_axes", {}), ("code_systems", ()), ("datasets", ())]
)
def test_dfo_parts_refuse_empty_declarations(field: str, empty: object) -> None:
    declarations = dataclasses.replace(SLICE_DECLARATIONS, **{field: empty})

    with pytest.raises(ValueError, match=field):
        build_dfo_parts(slice_platform_parts(), declarations, SLICE_BASE_PARTIES)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda d, m: dataclasses.replace(d, role_axes={}), id="field"),
        pytest.param(
            lambda d, m: m.setattr(code_index_declarations, "CODE_SYNONYMS", {}) or d,
            id="code-index-module-table",
        ),
        pytest.param(
            lambda d, m: (
                m.setattr(release_declarations, "IMPORT_ADDED_CONSTRAINT_PROPERTIES", frozenset())
                or d
            ),
            id="import-added-properties",
        ),
    ],
)
def test_declarations_hash_moves_with_every_declaration(change, monkeypatch) -> None:
    before = DECLARATIONS.declarations_hash()

    changed: Declarations = change(DECLARATIONS, monkeypatch)

    assert changed.declarations_hash() != before
