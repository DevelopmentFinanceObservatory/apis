"""Pins the DFO layer: the release-backed graph reader, the base parties, the DFO parts and the
declarations.

The reader is checked against the dev mixer's recorded `/v2/node` answer for five IDS variables
(`tests/fixtures/dc_graph/visible_arcs/ids_variables.json`). The parts are built from the
recorded slice with `tests.graph_release.recorded`'s declarations and parties: two the release
holds by edges, and countries only the mixer answers, one of them a placeholder `Node` row typed
`Thing` that an edge names.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from code_index import declarations as code_index_declarations
from code_index import scope_conventions
from code_index.composites import provenance_datasets
from code_index.declarations import CRS_INCLUSIVE, WholePopulationTotal, WholeWindow
from code_index.export import (
    DATASET_MEMBER_CLAIMED_TWICE,
    DATASET_MEMBERS_SHARE_SERIES,
    WHOLE_MARKER_IN_COMPOSITE,
)
from code_index.loader import content_hash, parse_index
from dc_graph import GraphClient, parse_arcs
from graph_release import declarations as release_declarations
from graph_release.declarations import DECLARATIONS, Declarations
from graph_release.dfo import (
    GRAPH_ENDPOINT,
    MissingBaseParties,
    ReleaseGraphReader,
    _parties,
    base_party_ids,
    build_dfo_parts,
    dfo_inputs,
    read_base_parties,
    stage_dfo_parts,
)
from graph_release.format import conform_part
from graph_release.platform import PlatformParts, PlatformRefusal
from tests.graph_release.recorded import (
    MIXER_PARTIES,
    PLACEHOLDER_PARTY,
    RELEASE_PARTIES,
    SLICE_BASE_PARTIES,
    SLICE_DECLARATIONS,
    UNHELD_PARTY,
    RecordedMixer,
    platform_parts,
    slice_platform_parts,
)

_IDS_FIXTURE = Path(__file__).parents[1] / "fixtures/dc_graph/visible_arcs/ids_variables.json"
_CRS = "provenance/OECDCreditorReportingSystem"
_IMPUTED = "provenance/ONEImputedMultilateralSectors"


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


def test_reader_refuses_an_edge_to_a_node_the_nodes_part_lacks() -> None:
    parts = platform_parts()
    subjects = parts.edges["subject_id"].cast(pa.string())
    nodes = parts.nodes.filter(pc.invert(pc.is_in(parts.nodes["subject_id"], value_set=subjects)))

    with pytest.raises(PlatformRefusal) as refused:
        ReleaseGraphReader(edges=parts.edges, nodes=nodes)

    assert refused.value.code == "dangling_edge"


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

    parts = slice_platform_parts()
    placeholder = ReleaseGraphReader(edges=parts.edges, nodes=parts.nodes).node_arcs_many(
        [PLACEHOLDER_PARTY, UNHELD_PARTY], properties=["typeOf"]
    )

    ids = base_party_ids(dfo_inputs(parts, SLICE_DECLARATIONS))
    stored = read_base_parties(list(reversed(ids)), graph=mixer)

    assert placeholder == {PLACEHOLDER_PARTY: {"typeOf": ("Thing",)}, UNHELD_PARTY: None}
    assert ids == tuple(sorted(MIXER_PARTIES))
    assert {PLACEHOLDER_PARTY, UNHELD_PARTY} <= set(ids)
    assert mixer.calls == [ids]
    assert stored == SLICE_BASE_PARTIES
    assert json.loads(stored) == {
        party: {prop: list(values) for prop, values in sorted(arcs.items())}
        for party, arcs in MIXER_PARTIES.items()
    }


def test_base_parties_refuse_a_party_the_mixer_answers_with_no_node() -> None:
    class Empty:
        def node_arcs_many(self, dcids, *, properties):
            return dict.fromkeys(dcids)

    with pytest.raises(MissingBaseParties, match="country/KGZ"):
        read_base_parties(["country/KGZ"], graph=Empty())


def test_base_parties_let_a_mixer_error_propagate() -> None:
    class Failing:
        def node_arcs_many(self, dcids, *, properties):
            raise TimeoutError("mixer read timed out")

    with pytest.raises(TimeoutError):
        read_base_parties(["country/KGZ"], graph=Failing())


def test_dfo_parts_rebuild_byte_for_byte_and_join_coverage(tmp_path: Path) -> None:
    parts = slice_platform_parts()
    inputs = dfo_inputs(parts, SLICE_DECLARATIONS)

    built = build_dfo_parts(inputs, SLICE_BASE_PARTIES)
    staged = stage_dfo_parts(built, staging_dir=tmp_path, execution_id="exec-1")
    stored = {part.ref.name: part.path.read_bytes() for part in staged}
    again = build_dfo_parts(inputs, stored["base_parties"])

    assert stored == built.blobs()
    assert again == built
    assert built.declarations == SLICE_DECLARATIONS.declarations_json()
    document = json.loads(gzip.decompress(built.code_index_blocks))
    stamp = document.pop("stamp")
    blocks = document
    assert stamp == {
        "graph_endpoint": GRAPH_ENDPOINT,
        "exported_at": "",
        "code_counts": {"crsTypeOfFlow": len(blocks["codes"])},
        "statvar_shape_counts": {"crsTypeOfFlow": len(blocks["statvars"])},
        "content_hash": content_hash(blocks),
        "coverage_build": None,
        "statvars_without_coverage": 0,
    }
    assert {(row["dcid"].split("_")[0], *row["datasets"]) for row in blocks["statvars"]} == {
        ("Amount", _CRS),
        ("AttributedAmount", "provenance/ONEImputedMultilateralSectors"),
    }
    parties = {
        row["dcid"]: (
            row["kind"],
            row["label"],
            sorted({entry["axis"] for entry in row["published"]}),
        )
        for row in blocks["parties"]
    }
    assert parties.keys() == {*RELEASE_PARTIES, *MIXER_PARTIES}
    assert {
        party: parties[party] for party in (*RELEASE_PARTIES, PLACEHOLDER_PARTY, UNHELD_PARTY)
    } == {
        "AsianDevelopmentBank": ("party", "Asian Development Bank", ["origin"]),
        "DACAsiaUnallocated": ("residual", "Asia, regional", ["destination"]),
        "country/KGZ": ("party", "Kyrgyzstan", ["destination"]),
        "country/TJK": ("party", "Tajikistan", ["destination"]),
    }


def test_parties_publish_each_dataset_between_its_first_and_last_year() -> None:
    # `Donor` is on two facets of one dataset, whose spans merge, and on a facet of a second
    # dataset with its own span. `Other` sits on a facet of a dataset the call leaves out.
    # `Recipient` is a donor in the second dataset alone.
    inputs = dataclasses.replace(
        dfo_inputs(slice_platform_parts(), SLICE_DECLARATIONS),
        coverage=pa.table(
            {
                "donor": ["Donor", "Donor", "Donor", "Other", "", "Recipient"],
                "recipient": ["Recipient", "Recipient", "Recipient", "Recipient", "Recipient", ""],
                "facet_id": ["f1", "f2", "f3", "f4", "f1", "f3"],
                "first_year": [2000, 1995, 2010, 1990, 1980, 2012],
                "last_year": [2005, 2001, 2020, 2022, 2024, 2018],
            }
        ),
        facet_provenances={"f1": "d/A", "f2": "d/A", "f3": "d/B", "f4": "d/C"},
    )

    parties = _parties(inputs, {"d/A", "d/B"})

    assert set(parties) == {"Donor", "Recipient"}
    assert parties["Donor"].years == {"d/A": (1995, 2005), "d/B": (2010, 2020)}
    assert parties["Donor"].pairs == {("d/A", "origin"), ("d/B", "origin")}
    assert parties["Recipient"].years == {"d/A": (1980, 2024), "d/B": (2010, 2020)}
    assert parties["Recipient"].pairs == {
        ("d/A", "destination"),
        ("d/B", "destination"),
        ("d/B", "origin"),
    }


def test_dfo_parts_refuse_a_party_neither_the_release_nor_the_mixer_answers() -> None:
    base = json.loads(SLICE_BASE_PARTIES)
    del base["country/TJK"]

    with pytest.raises(PlatformRefusal, match="country/TJK") as refused:
        build_dfo_parts(
            dfo_inputs(slice_platform_parts(), SLICE_DECLARATIONS),
            json.dumps(base).encode("utf-8"),
        )

    assert refused.value.code == "code_index_export_refused"


def test_dfo_parts_carry_the_export_refusals_code_and_rows() -> None:
    crs, _imputed = provenance_datasets(CRS_INCLUSIVE)
    declarations = dataclasses.replace(
        SLICE_DECLARATIONS,
        whole_totals={
            "DACOfficialDonors": WholePopulationTotal(
                "origin", {crs: WholeWindow(("official",), None)}, "a"
            )
        },
    )

    with pytest.raises(PlatformRefusal) as refused:
        build_dfo_parts(dfo_inputs(slice_platform_parts(), declarations), SLICE_BASE_PARTIES)

    assert (refused.value.code, refused.value.detail_code, refused.value.rows) == (
        "code_index_export_refused",
        WHOLE_MARKER_IN_COMPOSITE,
        (("DACOfficialDonors", crs, CRS_INCLUSIVE),),
    )


def test_dfo_parts_read_a_base_partys_donor_code_from_the_mixer() -> None:
    # The placeholder country stands in for a foundation the base graph alone holds: its listed
    # code reaches `export_parties` only through `read_base_parties`.
    answers = {
        **MIXER_PARTIES,
        PLACEHOLDER_PARTY: {**MIXER_PARTIES[PLACEHOLDER_PARTY], "dacCode": ("1601",)},
    }

    class Mixer:
        def node_arcs_many(self, dcids, *, properties):
            return {
                dcid: {prop: answers[dcid].get(prop, ()) for prop in properties} for dcid in dcids
            }

    base = read_base_parties(sorted(MIXER_PARTIES), graph=Mixer())
    declarations = dataclasses.replace(
        SLICE_DECLARATIONS, private_providers={1601: PLACEHOLDER_PARTY}
    )

    built = build_dfo_parts(dfo_inputs(slice_platform_parts(), declarations), base)

    rows = {
        row["dcid"]: (row["kind"], row["party_class"])
        for row in json.loads(gzip.decompress(built.code_index_blocks))["parties"]
    }
    assert rows[PLACEHOLDER_PARTY] == ("party", "private")
    assert rows[UNHELD_PARTY] == ("party", "official")


def test_dfo_parts_carry_the_ratio_base_shape_slot_defaults_and_broad_readings() -> None:
    """The code index the release serves reads the declarations back, and a dataset that
    declares no ratio base writes no key."""
    crs, imputed = SLICE_DECLARATIONS.datasets
    declarations = dataclasses.replace(
        SLICE_DECLARATIONS,
        datasets=(dataclasses.replace(crs, ratio_base_axis="destination"), imputed),
        shape_slot_defaults=code_index_declarations.SHAPE_SLOT_DEFAULTS,
        broad_readings=code_index_declarations.BROAD_READINGS,
    )

    built = build_dfo_parts(dfo_inputs(slice_platform_parts(), declarations), SLICE_BASE_PARTIES)
    document = json.loads(gzip.decompress(built.code_index_blocks))
    index = parse_index(document)

    assert [row.get("ratio_base_axis") for row in document["datasets"]] == ["destination", None]
    assert "ratio_base_axis" not in document["datasets"][1]
    assert index.datasets[crs.id].ratio_base_axis == "destination"
    assert index.datasets[imputed.id].ratio_base_axis is None
    assert index.shape_slot_defaults == code_index_declarations.SHAPE_SLOT_DEFAULTS
    assert index.broad_readings == code_index_declarations.BROAD_READINGS
    assert built.declarations == declarations.declarations_json()


_LEDGER = "dataset/Ledger"
_NORTH = "provenance/LedgerNorth"
_SOUTH = "provenance/LedgerSouth"
_SOUTH_FACET = "ledger-south"


def _ledger_release(
    *, part_of: tuple[tuple[str, str], ...], shared_series: bool = False
) -> tuple[PlatformParts, Declarations]:
    """The slice with its CRS series split between two provenances, `_NORTH` and `_SOUTH` (the
    second half of its variables), the `isPartOf` edges `part_of` and a `Dataset` node
    `_LEDGER`, and the CRS declaration declared as `_LEDGER`. `shared_series` puts one
    `_NORTH` series under `_SOUTH` as well."""
    parts = slice_platform_parts()
    facets = parts.facets.to_pylist()
    (crs_facet,) = [row for row in facets if row["provenance"] == _CRS]
    crs_facet["provenance"] = _NORTH
    facets.append({**crs_facet, "facet_id": _SOUTH_FACET, "provenance": _SOUTH})
    series = parts.series.to_pylist()
    crs_rows = [row for row in series if row["facet_id"] == crs_facet["facet_id"]]
    variables = sorted({row["variable_measured"] for row in crs_rows})
    south = set(variables[len(variables) // 2 :])
    assert south and len(south) < len(variables)
    for row in crs_rows:
        if row["variable_measured"] in south:
            row["facet_id"] = _SOUTH_FACET
    if shared_series:
        north = next(row for row in crs_rows if row["facet_id"] != _SOUTH_FACET)
        series.append({**north, "facet_id": _SOUTH_FACET})
    nodes = pa.table(
        {
            "subject_id": [_LEDGER, _NORTH, _SOUTH],
            "name": ["Ledger", "Ledger north", "Ledger south"],
            "types": pa.array([["Dataset"], ["Provenance"], ["Provenance"]], pa.list_(pa.string())),
            "value": pa.array([None] * 3, pa.string()),
            "authored": [False] * 3,
        }
    )
    edges = pa.table(
        {
            "subject_id": pa.array([subject for subject, _ in part_of], pa.string()),
            "predicate": pa.array(["isPartOf"] * len(part_of), pa.string()),
            "object_id": pa.array([parent for _, parent in part_of], pa.string()),
            "provenance": pa.array(["provenance/DFOOntology"] * len(part_of), pa.string()),
        }
    )
    ledger_parts = dataclasses.replace(
        parts,
        facets=conform_part(pa.Table.from_pylist(facets), name="facets"),
        series=conform_part(pa.Table.from_pylist(series), name="series"),
        nodes=conform_part(pa.concat_tables([parts.nodes.cast(nodes.schema), nodes]), name="nodes"),
        edges=conform_part(pa.concat_tables([parts.edges.cast(edges.schema), edges]), name="edges"),
    )
    declarations = dataclasses.replace(
        SLICE_DECLARATIONS,
        datasets=tuple(
            dataclasses.replace(dataset, id=_LEDGER) if dataset.id == _CRS else dataset
            for dataset in SLICE_DECLARATIONS.datasets
        ),
        code_systems=tuple(
            dataclasses.replace(
                config,
                declared_by_every_statvar_of=tuple(
                    _LEDGER if dataset == _CRS else dataset
                    for dataset in config.declared_by_every_statvar_of
                ),
            )
            for config in SLICE_DECLARATIONS.code_systems
        ),
    )
    return ledger_parts, declarations


def test_dfo_parts_key_a_declared_dataset_by_the_provenances_part_of_it() -> None:
    """A declared Dataset owns the provenances whose `isPartOf` names it: its StatVar rows and
    parties carry the Dataset's id, and its row lists both members."""
    parts, declarations = _ledger_release(part_of=((_NORTH, _LEDGER), (_SOUTH, _LEDGER)))

    built = build_dfo_parts(dfo_inputs(parts, declarations), SLICE_BASE_PARTIES)
    document = json.loads(gzip.decompress(built.code_index_blocks))
    index = parse_index(document)

    assert {row["id"]: row["members"] for row in document["datasets"]} == {
        _LEDGER: [_NORTH, _SOUTH],
        "provenance/ONEImputedMultilateralSectors": ["provenance/ONEImputedMultilateralSectors"],
    }
    assert {(row["dcid"].split("_")[0], *row["datasets"]) for row in document["statvars"]} == {
        ("Amount", _LEDGER),
        ("AttributedAmount", "provenance/ONEImputedMultilateralSectors"),
    }
    published = {entry["dataset"] for row in document["parties"] for entry in row["published"]}
    assert published == {_LEDGER, "provenance/ONEImputedMultilateralSectors"}
    assert index.provenances_of(_LEDGER) == (_NORTH, _SOUTH)
    assert (index.dataset_of(_NORTH), index.dataset_of(_SOUTH)) == (_LEDGER, _LEDGER)
    assert index.dataset_of(_CRS) == _CRS


@pytest.mark.parametrize(
    ("part_of", "shared_series", "detail_code", "rows"),
    [
        pytest.param(
            ((_NORTH, _LEDGER), (_SOUTH, _LEDGER), (_IMPUTED, _LEDGER)),
            False,
            DATASET_MEMBER_CLAIMED_TWICE,
            ((_IMPUTED, _LEDGER, _IMPUTED),),
            id="declared-provenance-part-of-a-declared-dataset",
        ),
        pytest.param(
            ((_NORTH, _LEDGER), (_SOUTH, _LEDGER)),
            True,
            DATASET_MEMBERS_SHARE_SERIES,
            None,
            id="members-share-a-series",
        ),
    ],
)
def test_dfo_inputs_refuse_dataset_members_that_do_not_partition_the_dataset(
    part_of, shared_series, detail_code, rows
) -> None:
    parts, declarations = _ledger_release(part_of=part_of, shared_series=shared_series)

    with pytest.raises(PlatformRefusal) as refused:
        dfo_inputs(parts, declarations)

    assert (refused.value.code, refused.value.detail_code) == (
        "code_index_export_refused",
        detail_code,
    )
    if rows is not None:
        assert refused.value.rows == rows
    else:
        ((dataset, *_key, first, second),) = refused.value.rows
        assert (dataset, first, second) == (_LEDGER, _NORTH, _SOUTH)


@pytest.mark.parametrize(
    ("field", "empty"), [("role_axes", {}), ("code_systems", ()), ("datasets", ())]
)
def test_dfo_inputs_refuse_empty_declarations(field: str, empty: object) -> None:
    declarations = dataclasses.replace(SLICE_DECLARATIONS, **{field: empty})

    with pytest.raises(ValueError, match=field):
        dfo_inputs(slice_platform_parts(), declarations)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda d, m: dataclasses.replace(d, role_axes={}), id="field"),
        pytest.param(
            lambda d, m: m.setattr(code_index_declarations, "CODE_SYNONYMS", {}) or d,
            id="code-index-module-table",
        ),
        pytest.param(
            lambda d, m: m.setattr(code_index_declarations, "COMPOSITES", ()) or d,
            id="composites",
        ),
        pytest.param(
            lambda d, m: m.setattr(code_index_declarations, "CONTAINING_TOTALS", {}) or d,
            id="containing-totals",
        ),
        pytest.param(
            lambda d, m: m.setattr(scope_conventions, "SECTOR_GROUPINGS", ()) or d,
            id="scope-conventions-table",
        ),
        pytest.param(
            lambda d, m: (
                m.setattr(
                    scope_conventions,
                    "OPTION_LABELS",
                    dataclasses.replace(scope_conventions.OPTION_LABELS, sector_code="{label}"),
                )
                or d
            ),
            id="option-labels",
        ),
        pytest.param(
            lambda d, m: m.setattr(scope_conventions, "REGION_NAMES", ()) or d, id="region-names"
        ),
        pytest.param(
            lambda d, m: m.setattr(scope_conventions, "CONVENTION_CURRENCY", "EUR") or d,
            id="convention-currency",
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
def test_declarations_hash_moves_with_every_declaration(
    change, monkeypatch, fresh_composite_reads
) -> None:
    before = DECLARATIONS.declarations_hash()

    changed: Declarations = change(DECLARATIONS, monkeypatch)

    assert changed.declarations_hash() != before
