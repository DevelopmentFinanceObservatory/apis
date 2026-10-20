"""Pins `scripts/graph_release_parity.py` on the recorded release slice: a release and the
coverage build and code index of the same import report parity, and a difference in the rows,
the provenances or a code index block is a difference, whatever variable it names. Its origin
gate reports parity for a catalogue build read from the same rows as the release, and a
difference in a family, an arc, a member walk or the skills index is a difference."""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

import dc_graph
from artifact_publication.jobs import Ingestion
from code_index.loader import default_index_path
from graph_release.context import origin_release_enabled
from graph_release.format import LoadedRelease, open_release
from graph_release.shadow import PINNED_INDEX_VARIABLES
from origin_catalogue.arc_tables import ArcTables, ArcTablesBuilder
from origin_catalogue.catalogue import CatalogueSnapshot
from origin_catalogue.context import current_catalogue
from origin_catalogue.families import FamilyKey, FamilyTables
from scripts.graph_release_parity import (
    Refused,
    check_same_import,
    origin_parity_report,
    parity_report,
)
from series_coverage.loader import load_coverage, read_table
from skills_index.types import (
    Attachment,
    ReachedNode,
    SkillEntry,
    SkillFile,
    SkillFrontMatter,
    SkillReach,
    SkillsIndex,
    TupleMemberAttachment,
)
from statvar_members import MemberArcReader, SubjectPage
from tests.graph_release.recorded import FIXTURES, platform_parts, recorded_dfo_arcs
from tests.graph_release.support import (
    TEST_EXECUTION,
    build_test_release,
    build_view_release,
    publish_coverage_build,
)
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue
from tests.origin_catalogue.support import build_labelled
from tests.series_coverage.support import build_dir

_PINNED = next(iter(PINNED_INDEX_VARIABLES))


def _report(
    tmp_path: Path,
    *,
    published: Callable[[dict], None] = lambda _: None,
    facets: Callable[[dict], None] = lambda _: None,
    export: Callable[[dict], None] = lambda _: None,
) -> dict:
    """The report on `build_view_release` against the recorded coverage build and the packaged
    code index, after `published` edits the build's rows, `facets` its `facets.json` and
    `export` the code index document standing in for a fresh export."""
    manifest = build_view_release((tmp_path / "release").as_uri())
    release = open_release(
        (tmp_path / "release").as_uri(),
        local_dir=tmp_path / "local",
        release_id=manifest.release_id,
    )
    rows = json.loads((FIXTURES / "coverage_rows/published.json").read_text())
    published(rows)
    base = tmp_path / "coverage"
    publish_coverage_build(
        base, rows, ingestion=Ingestion(TEST_EXECUTION, "2026-10-18 23:50:00+00:00")
    )
    facets_file = build_dir(base, "build") / "facets.json"
    metadata = json.loads(facets_file.read_text())
    facets(metadata)
    facets_file.write_text(json.dumps(metadata))
    coverage = load_coverage(base.as_uri())
    document = json.loads(gzip.decompress(default_index_path().read_bytes()))
    export(document)
    return parity_report(
        release,
        coverage,
        read_table(build_dir(base, "build").as_uri(), coverage.manifest),
        document,
    )


def _extra_row(rows: dict) -> None:
    row = list(rows["rows"][0])
    row[rows["columns"].index("recipient")] = "country/XXX"
    rows["rows"].append(row)


def _moved_provenance(metadata: dict) -> None:
    facet = next(iter(metadata.values()))
    facet["provenance_id"] = "provenance/Elsewhere"


def _relabelled_statvar(document: dict) -> None:
    document["statvars"][0]["label"] = "another label"


def _pinned_row(document: dict) -> None:
    row = copy.deepcopy(document["listed_statvars"][0])
    row["dcid"] = _PINNED
    document["listed_statvars"].append(row)


def _pinned_and_another_source_code(document: dict) -> None:
    document["source_codes"][0]["dcids"].extend([_PINNED, "Another_Variable"])


def _prefixed_pinned_row(document: dict) -> None:
    row = copy.deepcopy(document["listed_statvars"][0])
    row["dcid"] = f"{_PINNED}_Other"
    document["listed_statvars"].append(row)


def test_equal_inputs_report_parity(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report["outcome"] == "parity"
    assert report["differences"] == 0
    categories = report["categories"]
    assert categories["coverage.rows"]["compared"] > 0
    assert categories["code_index.statvars"]["compared"] > 0
    assert all(category["differences"] == 0 for category in categories.values())


@pytest.mark.parametrize(
    ("edits", "category"),
    [
        ({"published": _extra_row}, "coverage.rows"),
        ({"facets": _moved_provenance}, "coverage.variable_provenances"),
        ({"export": _relabelled_statvar}, "code_index.statvars"),
        ({"export": _pinned_row}, "code_index.listed_statvars"),
        ({"export": _pinned_and_another_source_code}, "code_index.source_codes"),
        ({"export": _prefixed_pinned_row}, "code_index.listed_statvars"),
    ],
)
def test_a_difference_fails_parity(tmp_path: Path, edits: dict, category: str) -> None:
    report = _report(tmp_path, **edits)

    assert report["outcome"] == "differences"
    assert report["categories"][category]["differences"] >= 1


@pytest.mark.parametrize(
    ("ingestion", "code"),
    [
        (Ingestion("another-execution", "2026-10-18 23:50:00+00:00"), "import_mismatch"),
        (None, "coverage_ingestion_unrecorded"),
    ],
)
def test_a_coverage_build_of_another_import_is_refused(
    tmp_path: Path, ingestion: Ingestion | None, code: str
) -> None:
    manifest = build_view_release(tmp_path.as_uri())
    rows = json.loads((FIXTURES / "coverage_rows/published.json").read_text())
    coverage = load_coverage(publish_coverage_build(tmp_path / "coverage", rows)).manifest

    with pytest.raises(Refused) as refused:
        check_same_import(manifest, dataclasses.replace(coverage, ingestion=ingestion))

    assert refused.value.code == code


@pytest.mark.parametrize("artifact", ["coverage", "catalogue"])
def test_a_build_of_another_import_is_refused_whichever_artifact(
    tmp_path: Path, artifact: str
) -> None:
    manifest = build_view_release(tmp_path.as_uri())
    build = _catalogue().manifest

    with pytest.raises(Refused) as refused:
        check_same_import(
            manifest,
            dataclasses.replace(build, ingestion=Ingestion("another", "2026-10-18")),
            artifact=artifact,
        )
    assert refused.value.code == "import_mismatch"

    with pytest.raises(Refused) as refused:
        check_same_import(manifest, dataclasses.replace(build, ingestion=None), artifact=artifact)
    assert refused.value.code == f"{artifact}_ingestion_unrecorded"


# ---------------------------------------------------------------- origin

_FLOWS = FamilyKey("FinancialFlow", "amount", "measuredValue", "Nominal", None, False, True)
"""The recorded slice's largest DFO family."""
_UNLISTED = "Amount_FinancialFlow_Unlisted_Nominal"
_UNLISTED_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "populationType": ("FinancialFlow",),
    "measuredProperty": ("amount",),
    "statType": ("measuredValue",),
    "measurementQualifier": ("Nominal",),
    "observationProperties": ("financingRecipient",),
}
"""A base variable of `_FLOWS` that the catalogue does not list."""


def _catalogue() -> CatalogueSnapshot:
    """A catalogue build of the recorded slice's DFO variables, read as the catalogue job
    reads them from the rows the release copies, and holding no base member of their
    families, so the release holds each of them whole."""
    builder = ArcTablesBuilder()
    read = recorded_dfo_arcs(builder)
    denominators = {
        arc.dcid
        for rows in builder.table("dfo").values()
        for arc in rows
        if arc.property == "measurementDenominator"
        and arc.dcid is not None
        and arc.dcid not in read.variables
    }
    for denominator in sorted(denominators - {"Count_Person"}):
        builder.add("schema", denominator, None)
    builder.add("base", "Count_Person", {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}})
    return restrictive_catalogue(
        dfo_variables=read.variables,
        arcs=build_labelled(builder),
        ingestion=Ingestion(TEST_EXECUTION, "2026-10-18 23:50:00+00:00"),
    )


def _release(tmp_path: Path) -> LoadedRelease:
    uri = (tmp_path / "release").as_uri()
    manifest = build_test_release(uri, tables=platform_parts().tables())
    return open_release(uri, local_dir=tmp_path / "local", release_id=manifest.release_id)


def _skills_index(
    *, version: str = "sha256:a", tuple_property: str = "populationType", hop: int = 0
) -> SkillsIndex:
    entry = SkillEntry(
        front_matter=SkillFrontMatter("aid", "aid", "Aid flows", version),
        frontmatter={},
        url=None,
        attachments=(Attachment("FinancialFlow", TupleMemberAttachment(tuple_property)),),
        reach=SkillReach("aid", (ReachedNode("FinancialFlow", hop, None),), excluded=False),
        statvar_count=3,
        statvar_count_truncated=False,
        files=(SkillFile("SKILL.md", 1, "sha256:x", "x"),),
    )
    return SkillsIndex(
        skills={"aid": entry}, node_to_skills={}, issues=(), may_omit_published_skills=False
    )


def _skills(
    *, version: str = "sha256:a", tuple_property: str = "populationType", hop: int = 0
) -> Callable[[], SkillsIndex]:
    """A skills build answering an index of these fields with `GRAPH_RELEASE_ORIGIN` on and
    `_skills_index()` with it off. It reads the bound catalogue, as the build's graph reads
    do."""
    released = _skills_index(version=version, tuple_property=tuple_property, hop=hop)

    def build() -> SkillsIndex:
        current_catalogue()
        return released if origin_release_enabled() else _skills_index()

    return build


class _WithVariable:
    """`reader` and one more variable, `_UNLISTED`, which a reverse read finds wherever its
    arcs match the read's property and filters."""

    def __init__(self, reader: MemberArcReader) -> None:
        self._reader = reader

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        page = self._reader.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        wanted = {property: dcid, **(filters or {})}
        if all(_UNLISTED_ARCS.get(name) == (value,) for name, value in wanted.items()):
            return dc_graph.SubjectPage(dcids=(*page.dcids, _UNLISTED), truncated=page.truncated)
        return page

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        found = self._reader.node_arcs_many(
            [dcid for dcid in dcids if dcid != _UNLISTED], properties=properties
        )
        found[_UNLISTED] = {name: _UNLISTED_ARCS.get(name, ()) for name in properties}
        return {dcid: found[dcid] for dcid in dcids}


def _origin_report(
    tmp_path: Path,
    *,
    catalogue: Callable[[CatalogueSnapshot], CatalogueSnapshot] = lambda found: found,
    live: Callable[[MemberArcReader], MemberArcReader] = lambda reader: reader,
    skills: Callable[[], SkillsIndex] | None = None,
) -> dict:
    """The origin report on the recorded release against `_catalogue()` after `catalogue`
    edits it, with the release's own graph reader, after `live` wraps it, standing in for the
    mixer the flag off walks."""
    from graph_release.origin import release_origin

    loaded = _release(tmp_path)
    return origin_parity_report(
        loaded,
        catalogue(_catalogue()),
        live=live(release_origin(loaded).reader),
        build_skills=skills or _skills(),
        readings={},
    )


def _reordered_family(catalogue: CatalogueSnapshot) -> CatalogueSnapshot:
    families = catalogue.families
    flows = families.arcs[_FLOWS]
    reordered = dataclasses.replace(flows, members=flows.members[::-1])
    return dataclasses.replace(
        catalogue,
        families=FamilyTables(arcs={**families.arcs, _FLOWS: reordered}, un=families.un),
    )


def _renamed_target(catalogue: CatalogueSnapshot) -> CatalogueSnapshot:
    arcs = catalogue.arcs
    variable, rows = next((variable, rows) for variable, rows in sorted(arcs.dfo.items()) if rows)
    index = next(i for i, arc in enumerate(rows) if arc.dcid is not None and arc.name)
    renamed = (*rows[:index], dataclasses.replace(rows[index], name="Renamed"), *rows[index + 1 :])
    return dataclasses.replace(
        catalogue,
        arcs=ArcTables(
            dfo={**arcs.dfo, variable: renamed},
            base=arcs.base,
            schema=arcs.schema,
            labels=arcs.labels,
        ),
    )


def test_equal_origin_reads_report_parity(tmp_path: Path) -> None:
    report = _origin_report(tmp_path)

    assert report["outcome"] == "parity"
    categories = report["categories"]
    assert all(category["differences"] == 0 for category in categories.values())
    for name in (
        "origin.families",
        "origin.arcs",
        "origin.arc_targets",
        "origin.members",
        "skills.roster",
        "skills.attachments",
        "skills.reach",
    ):
        assert categories[name]["compared"] > 0, name
    assert report["member_routing"]["release"] == report["member_routing"]["five_tuples"]
    blind = report["blind_spots"]
    assert blind["unlisted_base_variables"]["count"] == 0
    assert blind["unobserved_spanner_variables"]["count"] == 1


@pytest.mark.parametrize(
    ("edits", "categories"),
    [
        ({"catalogue": _reordered_family}, {"origin.families"}),
        ({"catalogue": _renamed_target}, {"origin.arcs", "origin.arc_targets"}),
        ({"live": _WithVariable}, {"origin.members"}),
        ({"skills": _skills(version="sha256:b")}, {"skills.roster"}),
        ({"skills": _skills(tuple_property="measuredProperty")}, {"skills.attachments"}),
        ({"skills": _skills(hop=1)}, {"skills.reach"}),
    ],
)
def test_an_origin_difference_fails_parity(
    tmp_path: Path, edits: dict, categories: set[str]
) -> None:
    report = _origin_report(tmp_path, **edits)

    assert report["outcome"] == "differences"
    differing = {name for name, category in report["categories"].items() if category["differences"]}
    assert differing == categories


def test_a_variable_the_catalogue_does_not_list_is_a_blind_spot(tmp_path: Path) -> None:
    report = _origin_report(tmp_path, live=_WithVariable)

    unlisted = report["blind_spots"]["unlisted_base_variables"]
    assert unlisted["count"] == 1
    assert unlisted["examples"][0]["dcid"] == _UNLISTED
