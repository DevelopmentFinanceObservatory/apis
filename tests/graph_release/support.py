"""Builds graph releases in the bucket layout from rows given in code."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import graph_release.service as service_module
from artifact_publication.uri import read_uri_bytes_and_generation
from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from graph_release.authored import literal_id
from graph_release.context import bound_release, current_release
from graph_release.declarations import DECLARATIONS
from graph_release.format import (
    MASK_EPOCH,
    MASK_WORDS,
    PART_SCHEMAS,
    Manifest,
    StagedPart,
    conform_part,
    pointer_uri,
    write_blob_part,
    write_part,
)
from graph_release.origin import origin_view
from graph_release.publish import PublishStatus, hold_release, publish_release
from graph_release.service import Release, ReleaseService
from origin_catalogue.arc_reads import DFO_VARIABLE_TYPE
from origin_catalogue.arc_tables import WireArcs
from series_coverage.schema import RoleNames
from tests.code_index.support import INDEX_FIXTURE
from tests.graph_release.recorded import platform_parts

TEST_EXECUTION = "test-import-execution"
TEST_REVISION = "test-dc-revision"


def sample_tables() -> dict[str, pa.Table]:
    """Ten rows of each platform part, unsorted, over two provenances."""
    variables = [f"Var_{n}" for n in (3, 1, 4, 0, 2)]
    series = pa.table(
        {
            "variable_measured": [variables[n % 5] for n in range(10)],
            "entity1": ["country/GHA" if n % 2 else "country/DEU" for n in range(10)],
            "extra_entities_id": ["" if n < 7 else "x1" for n in range(10)],
            "facet_id": ["f1" if n % 3 else "f2" for n in range(10)],
            "entity2": [None if n % 4 else "country/KEN" for n in range(10)],
            "observation_count": list(range(1, 11)),
            "earliest_date": [date(1990 + n, 1, 1) for n in range(10)],
            "latest_date": [date(2010 + n, 1, 1) for n in range(10)],
            "period_mask_0": [(1 << n) | (1 << 63) for n in range(10)],
            "period_mask_1": [None if n == 4 else n for n in range(10)],
        },
        schema=pa.schema(
            [
                ("variable_measured", pa.string()),
                ("entity1", pa.string()),
                ("extra_entities_id", pa.string()),
                ("facet_id", pa.string()),
                ("entity2", pa.string()),
                ("observation_count", pa.int64()),
                ("earliest_date", pa.date32()),
                ("latest_date", pa.date32()),
                ("period_mask_0", pa.uint64()),
                ("period_mask_1", pa.int64()),
            ]
        ),
    )
    # Bit 63 set, stored signed as Spanner stores it.
    series = series.set_column(
        series.schema.get_field_index("period_mask_0"),
        "period_mask_0",
        series["period_mask_0"].cast(pa.int64(), safe=False),
    )
    facets = pa.table(
        {
            "facet_id": [f"f{n}" for n in (2, 1, 5, 3, 4, 9, 8, 6, 7, 0)],
            "provenance": ["dc/p2" if n % 2 else "dc/p1" for n in range(10)],
            "import_name": [f"Import{n}" for n in range(10)],
            "measurement_method": [None, "OECD", *(["Sum"] * 8)],
            "observation_period": ["P1Y"] * 10,
            "unit": ["USDollar"] * 9 + [None],
            "scaling_factor": [None] * 5 + ["1000000"] * 5,
            "is_dc_aggregate": [n == 3 for n in range(10)],
        }
    )
    edges = pa.table(
        {
            "subject_id": [variables[n % 5] for n in range(10)],
            "predicate": ["typeOf" if n % 2 else "measuredProperty" for n in range(10)],
            "object_id": [f"Node_{9 - n}" for n in range(10)],
            "provenance": ["dc/p2" if n % 3 else "dc/p1" for n in range(10)],
        }
    )
    nodes = pa.table(
        {
            "subject_id": [f"Node_{n}" for n in (7, 2, 9, 0, 4, 1, 8, 3, 6, 5)],
            "name": [None if n == 1 else f"Name {n}" for n in range(10)],
            "types": [["StatisticalVariable"], [], None, ["Thing", "Class"], *([["Thing"]] * 6)],
            "value": [None] * 8 + ["literal", "42"],
            "authored": [n >= 8 for n in range(10)],
        }
    )
    imports = pa.table(
        {
            "import_name": [f"Import{n}" for n in (4, 0, 9, 1, 6, 2, 8, 3, 7, 5)],
            "provenance": [f"dc/p{n}" if n % 2 else None for n in range(10)],
            "version": [f"v{n}" for n in range(10)],
            "state": ["SUCCESS"] * 10,
            "data_import_timestamp": [
                datetime(2026, 10, 1 + n, 12, 30, tzinfo=UTC) for n in range(10)
            ],
        }
    )
    return {
        "series": series,
        "facets": facets,
        "edges": edges,
        "nodes": nodes,
        "imports": imports,
    }


def build_test_release(
    root_uri: str,
    *,
    tables: Mapping[str, pa.Table],
    blobs: Mapping[str, bytes] | None = None,
    import_execution_id: str = TEST_EXECUTION,
    dc_serving_revision: str = TEST_REVISION,
    hold: bool = False,
) -> Manifest:
    """Stages `tables` and the serialised `blobs` as parts, publishes them under `root_uri`
    through `publish_release`, holds the release when `hold`, and returns
    the manifest. Raises `AssertionError` when a held pointer refuses it.

    `series` is written one file per facet provenance, so it needs `facets`.
    `edges` is written one file per its own `provenance` column.
    """
    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        staged: list[StagedPart] = []
        for name, table in tables.items():
            for provenance, rows in _provenance_files(name, table, tables):
                staged += write_part(
                    rows,
                    staging_dir=staging_dir,
                    name=name,
                    layer=PART_SCHEMAS[name].layer,
                    source="tests.graph_release.support",
                    produced_by_execution=import_execution_id,
                    provenance=provenance,
                )
        for name, data in (blobs or {}).items():
            staged.append(
                write_blob_part(
                    data,
                    staging_dir=staging_dir,
                    name=name,
                    source="tests.graph_release.support",
                    produced_by_execution=import_execution_id,
                )
            )
        manifest = Manifest(
            import_execution_id=import_execution_id,
            import_completed_at="2026-10-18T23:50:00+00:00",
            ingested_imports=("TestImport",),
            read_timestamp="2026-10-19T00:00:00+00:00",
            dc_serving_revision=dc_serving_revision,
            builder_code_hash="test-code-hash",
            declarations_hash="test-declarations-hash",
            parts=tuple(part.ref for part in staged),
        )
        found = read_uri_bytes_and_generation(pointer_uri(root_uri))
        outcome = publish_release(
            root_uri,
            staged=staged,
            manifest=manifest,
            expected_pointer_generation=found[1] if found is not None else None,
        )
    assert outcome.status is PublishStatus.PUBLISHED, outcome
    if hold:
        hold_release(root_uri, manifest.release_id)
    return manifest


def build_fixture_index_release(root_uri: str) -> Manifest:
    """Publishes a release under `root_uri` whose only part is the code index, the
    recorded index document (`INDEX_FIXTURE`) byte for byte."""
    return build_test_release(
        root_uri, tables={}, blobs={"code_index_blocks": INDEX_FIXTURE.read_bytes()}
    )


def build_view_release(root_uri: str, *, dc_serving_revision: str = TEST_REVISION) -> Manifest:
    """Publishes a release under `root_uri` whose views build: the recorded index document
    (`INDEX_FIXTURE`) as its code index part, and the recorded slice's platform parts and the
    declarations as the parts its coverage view reads (`tests.graph_release.recorded`)."""
    return build_test_release(
        root_uri,
        tables=platform_parts().tables(),
        blobs={
            "code_index_blocks": INDEX_FIXTURE.read_bytes(),
            "declarations": DECLARATIONS.declarations_json(),
        },
        dc_serving_revision=dc_serving_revision,
    )


COVERAGE_SLICE = Path(__file__).parents[1] / "fixtures" / "coverage_slice" / "format-3"
"""The recorded coverage slice: a coverage build of the variables the suite's fixtures and
corpora name, frozen at the recording."""


def slice_series_tables(slice_dir: Path = COVERAGE_SLICE) -> dict[str, pa.Table]:
    """The `series`, `edges` and `facets` rows whose coverage view is the coverage build under
    `slice_dir`, row for row: a series per coverage row, keyed by its donor and recipient as the
    view's role rule reads them back, its years moved onto the release's mask epoch, each
    variable's roles as `observationProperties` edges, and each facet's metadata. Columns are
    plain, not dictionary-encoded."""
    pointer = json.loads((slice_dir / "current.json").read_text())
    build = slice_dir / "versions" / pointer["build_id"]
    manifest = json.loads((build / "manifest.json").read_text())
    roles = {
        variable: RoleNames(**names)
        for variable, names in json.loads((build / "role_names.json").read_text()).items()
    }
    shift = manifest["epoch_year"] - date.fromisoformat(MASK_EPOCH).year
    words = range(manifest["mask_words"])
    series = []
    for row in pq.read_table(build / "coverage.parquet").to_pylist():
        names = roles[row["variable"]]
        two_roles = names.donor is not None and names.recipient is not None
        mask = sum(row[f"year_mask_{word}"] << (64 * word) for word in words) << shift
        series.append(
            {
                "variable_measured": row["variable"],
                "entity1": row["donor"] if names.donor is not None else row["recipient"],
                "extra_entities_id": row["recipient"] if two_roles else "",
                "facet_id": row["facet_id"],
                "entity2": row["recipient"] if two_roles else None,
                "observation_count": row["n_obs"],
                "earliest_date": date(row["first_year"], 1, 1),
                "latest_date": date(row["last_year"], 1, 1),
                **{
                    f"period_mask_{word}": _signed((mask >> (64 * word)) & _WORD)
                    for word in range(MASK_WORDS)
                },
            }
        )
    edges = [
        {
            "subject_id": variable,
            "predicate": "observationProperties",
            "object_id": role,
            "provenance": "provenance/DFOOntology",
        }
        for variable, names in roles.items()
        for role in (names.donor, names.recipient)
        if role is not None
    ]
    facets = [
        {
            "facet_id": facet_id,
            "provenance": facet["provenance_id"],
            "measurement_method": facet.get("measurement_method"),
            "observation_period": facet.get("observation_period"),
            "unit": facet.get("unit"),
            "scaling_factor": None,
            "is_dc_aggregate": False,
        }
        for facet_id, facet in json.loads((build / "facets.json").read_text()).items()
    ]
    return {
        name: pa.Table.from_pylist(rows, schema=_plain(PART_SCHEMAS[name].schema, rows[0]))
        for name, rows in (("series", series), ("edges", edges), ("facets", facets))
    }


def build_session_release(root_uri: str) -> Manifest:
    """Publishes under `root_uri` the release the suite serves (`tests/conftest.py`):
    `build_view_release`'s parts, its `series` the recorded coverage slice's rows alone and its
    `edges` and `facets` with the slice's added (`slice_series_tables`). Its coverage view is
    the slice's coverage, and its origin view holds no DFO variable, since no variable it
    declares has a series: a test that needs DFO variables binds a release holding them
    (`origin_release`)."""
    tables = dict(platform_parts().tables())
    for name, rows in slice_series_tables().items():
        base = tables[name] if name != "series" else tables[name].slice(0, 0)
        tables[name] = conform_part(
            pa.concat_tables([base.cast(_plain(base.schema)), rows], promote_options="default"),
            name=name,
        )
    return build_test_release(
        root_uri,
        tables=tables,
        blobs={
            "code_index_blocks": INDEX_FIXTURE.read_bytes(),
            "declarations": DECLARATIONS.declarations_json(),
        },
    )


_WORD = (1 << 64) - 1


def _signed(word: int) -> int:
    """An unsigned 64-bit mask word as the signed one the `series` part stores."""
    return word - (1 << 64) if word >> 63 else word


def _plain(schema: pa.Schema, row: Mapping[str, object] | None = None) -> pa.Schema:
    """`schema` with each dictionary column as its value type, limited to `row`'s columns when
    given."""
    return pa.schema(
        field.with_type(field.type.value_type) if pa.types.is_dictionary(field.type) else field
        for field in schema
        if row is None or field.name in row
    )


class _OneReleaseService:
    """A release service that serves `release` and never rechecks or swaps: the calls a surface
    makes on `ReleaseService`."""

    def __init__(self, release: Release) -> None:
        self._release = release

    def snapshot(self) -> Release:
        return self._release

    def serves(self, release: Release) -> bool:
        return release is self._release

    def register_swap_guard(
        self, guard: Callable[[], AbstractContextManager[object]]
    ) -> Callable[[], None]:
        return lambda: None

    def on_swap(self, callback: Callable[[Release], None]) -> Callable[[], None]:
        return lambda: None

    def close(self) -> None:
        pass


@contextmanager
def serving_release(release: Release) -> Iterator[Release]:
    """Binds `release` for the body of the `with` block, and installs as the process's release
    service (`graph_release.service.get_release_service`) one that serves it, so a surface built
    or requested in the block binds `release` as a production request binds its process's
    release. The session's service is back on exit. A surface built before the block keeps the
    service it was built with."""
    service = cast("ReleaseService", _OneReleaseService(release))
    with service_module._singleton.poisoned_for_test(service), bound_release(release) as bound:
        yield bound


@contextmanager
def serving_index(
    code_index: CodeIndex, measure_index: MeasureIndex | None = None
) -> Iterator[Release]:
    """Serves (`serving_release`) the bound release with `code_index`, and `measure_index` when
    given, as its views' indexes for the body of the `with` block, so `get_index()` and
    `get_measure_index()` serve them as they serve a release's. The release keeps its parts and
    its coverage view, and is served by no service, so it stays served."""
    release = current_release()
    views = dataclasses.replace(
        release.views,
        code_index=code_index,
        measure_index=measure_index if measure_index is not None else release.views.measure_index,
    )
    with serving_release(dataclasses.replace(release, views=views, service=None)) as served:
        yield served


def release_at_revision(revision: str) -> Release:
    """The bound release under the id DC serving revision `revision` gives it, with its parts and
    views, served by no service, for a test of what keys on the release id."""
    return release_with(dc_serving_revision=revision)


def release_with(
    *, blobs: Mapping[str, bytes] | None = None, **manifest_changes: object
) -> Release:
    """The bound release with its manifest changed by `manifest_changes` and the blobs in
    `blobs` replacing its own, its views unchanged, served by no service, for a test of what
    keys on the release id or its import key."""
    release = current_release()
    loaded = dataclasses.replace(
        release.loaded,
        manifest=dataclasses.replace(release.loaded.manifest, **manifest_changes),
        blobs={**release.loaded.blobs, **(blobs or {})},
    )
    return dataclasses.replace(release, loaded=loaded, service=None)


_NODE_ROWS = pa.schema(
    [
        ("subject_id", pa.string()),
        ("name", pa.string()),
        ("types", pa.list_(pa.string())),
        ("value", pa.string()),
        ("authored", pa.bool_()),
    ]
)
_ORIGIN_PROVENANCE = "provenance/DFOOntology"
"""The provenance of an `origin_release` edge whose target names none."""


def origin_release(variables: Mapping[str, WireArcs]) -> Release:
    """The bound release with an origin view holding `variables`, each with its `->*` arcs as
    a test graph answers them, in place of its own, under an id of its own, served by no
    service. Its `edges` and `nodes` rows are those DFO's Spanner holds for such a variable
    (`graph_release.origin`): an edge per target, a `Node` row per variable and per typed
    target, a literal row per value, and an `authored` row naming an untyped target by its
    dcid alone, which drops any name the test graph gives it, so the view answers each variable's arcs as `ArcTablesBuilder` holds the
    same answer, and its graph reader walks them. Each variable must declare `typeOf
    StatisticalVariable`, as a release holds only those."""
    edges: list[dict[str, str]] = []
    nodes: dict[str, dict[str, object]] = {}
    for variable, arcs in variables.items():
        types = [target.get("dcid") for target in arcs.get("typeOf", {}).get("nodes") or ()]
        assert DFO_VARIABLE_TYPE in types, f"{variable} declares no typeOf {DFO_VARIABLE_TYPE}"
        names = [target.get("value") for target in arcs.get("name", {}).get("nodes") or ()]
        nodes[variable] = _node(next(iter(names), None), types, variable)
        for prop, answer in arcs.items():
            for target in answer.get("nodes") or ():
                if (dcid := target.get("dcid")) is None:
                    dcid = literal_id(target["value"])
                    nodes[dcid] = _node("", [], target["value"])
                elif target.get("types"):
                    nodes[dcid] = _node(target.get("name"), list(target["types"]), dcid)
                else:
                    nodes.setdefault(dcid, _node(None, [], None, authored=True))
                provenance = target.get("provenanceId") or _ORIGIN_PROVENANCE
                edges.append(
                    {
                        "subject_id": variable,
                        "predicate": prop,
                        "object_id": dcid,
                        "provenance": provenance,
                    }
                )
    origin = origin_view(
        series=pa.table({"variable_measured": pa.array(list(variables), pa.string())}),
        edges=pa.Table.from_pylist(edges, schema=_plain(PART_SCHEMAS["edges"].schema)),
        nodes=pa.Table.from_pylist(
            [{"subject_id": dcid, **row} for dcid, row in nodes.items()],
            schema=_NODE_ROWS,
        ),
    )
    digest = hashlib.sha256(repr(sorted(edges, key=repr)).encode()).hexdigest()[:12]
    release = release_at_revision(f"origin-{digest}")
    return dataclasses.replace(release, origin=origin)


def _node(
    name: str | None, types: list[str], value: str | None, *, authored: bool = False
) -> dict[str, object]:
    return {"name": name, "types": types, "value": value, "authored": authored}


def _provenance_files(
    name: str, table: pa.Table, tables: Mapping[str, pa.Table]
) -> list[tuple[str | None, pa.Table]]:
    if name == "edges":
        provenance = table["provenance"]
    elif name == "series":
        facets = tables["facets"]
        provenance = pc.take(
            facets["provenance"], pc.index_in(table["facet_id"], value_set=facets["facet_id"])
        )
    else:
        return [(None, table)]
    return [
        (value, table.filter(pc.equal(provenance, value)))
        for value in sorted(pc.unique(provenance).to_pylist())
    ]
