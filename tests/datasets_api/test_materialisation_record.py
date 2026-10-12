"""Tests the materialisation record `serve_current` writes: one `datasets.materialisation` row
per run, the same dependency fingerprint whichever entry point ran it, and a head that names
the run that produced it. `LiveHeadStore` inserts the row and moves the head in one
transaction, so a head write that fails leaves no row behind.

The materialiser, `serve_current` and the Postgres writers are real. The graph, SDMX and
artifact store are fakes, and the store is `FakeVersionRecordDatasetStore`.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from typing import cast

import httpx
import pytest

from code_index.shared import get_index
from dataset_store import DatasetStore
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.adapter import build_identity_document, to_materialisation_request
from datasets_api.current import DefinitionDeps, Trigger, serve_current
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.shape import resolve_shape
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, MaterialisationRequest
from materialise.rate_tables import RateTableCache
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader, StoredDefinition
from statvar_members import MemberEnumerator

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore, materialisation_record
from .conftest import FakeDatasetStore
from .test_mint_statvar_set import _CSV_BODY, _FACET, _STAT_VAR, _STAT_VAR_REF, _coverage_service

_SPEC_ID = "spec_materialisation_record"
_REFRESH_DATE = "2026-10-07"


def _sdmx() -> SdmxClient:
    return SdmxClient(
        base_url="https://datacommons.one.org/core/api",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                text=_CSV_BODY,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
    )


def _graph_with_refresh_date() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            "OECD_DAC2A": {"lastDataRefreshDate": (_REFRESH_DATE,)},
        },
        facets={(_STAT_VAR, "country/KEN"): [_FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", "country/KEN": "Kenya"},
    )


def _deps(version_store: FakeVersionRecordDatasetStore) -> DefinitionDeps:
    store = FakeDatasetStore()
    graph = _graph_with_refresh_date()
    return DefinitionDeps(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=_sdmx(),
        artifact_store=FakeArtifactStore(),
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=4,
        enumerator=MemberEnumerator(),
        enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
        code_index=get_index(),
        rate_cache=RateTableCache(),
        coverage=_coverage_service(),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        head_store=LiveHeadStore(store=version_store),
        # Every call probes and rebuilds, so each trigger materialises once.
        head_recheck_interval=timedelta(0),
        head_max_age=timedelta(0),
    )


def _registered(deps: DefinitionDeps) -> tuple[StoredDefinition, MaterialisationRequest]:
    """The stored definition a `statvar_set` mint of `_STAT_VAR_REF` registers, and its request
    (`kinds.statvar_set.mint_statvar_set`)."""
    scope = Scope.model_validate(
        {
            "entities": {"recipient": ["country/KEN"]},
            "facets": {
                "provenance": {"ref": "OECD_DAC2A"},
                "measurementMethod": {"ref": "GrantEquivalent"},
                "unit": {"ref": "USD"},
            },
        }
    )
    shape = resolve_shape(_STAT_VAR_REF, graph=deps.graph)
    recomputed = recompute_identity(build_identity_document(shape, scope=scope, graph=deps.graph))
    deps.registry.register_definition(
        identity_definition=recomputed.identity,
        schema_version=1,
        supplied_spec_id=recomputed.spec_id,
    )
    stored = deps.spec_reader.read_definition(recomputed.spec_id)
    assert stored is not None
    request = to_materialisation_request(stored.identity, shape=shape, code_index=deps.code_index)
    return stored, request


_TRIGGERS: tuple[tuple[Trigger, MintOptions | None], ...] = (
    ("mint", MintOptions(bundle=False)),
    ("get", MintOptions(bundle=False)),
    ("options", None),
    ("refresh", MintOptions(bundle=False)),
    ("worker", MintOptions(bundle=False)),
)


@pytest.fixture
def version_store() -> Generator[FakeVersionRecordDatasetStore]:
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


def _materialisation_rows(store: FakeVersionRecordDatasetStore) -> list[Mapping[str, object]]:
    with store.transaction() as tx:
        return tx.fetch_all(
            'SELECT id, subject_id, content_version, "trigger", dependencies, selection, '
            "materialiser_build, frame_grammar_version, code_index_stamp, "
            "origin_catalogue_build_id, coverage_stamp, graph_generation, trace_id "
            "FROM datasets.materialisation ORDER BY id"
        )


def test_every_trigger_records_one_run_with_the_same_dependencies_and_the_head_names_it(
    version_store,
):
    deps = _deps(version_store)
    stored, request = _registered(deps)

    for trigger, options in _TRIGGERS:
        serve_current(
            stored,
            resolve_requests=lambda: [request],
            options=options,
            trigger=trigger,
            deps=deps,
            trace_id=f"trace-{trigger}",
        )

    rows = _materialisation_rows(version_store)
    assert [row["trigger"] for row in rows] == [trigger for trigger, _ in _TRIGGERS]
    assert [row["trace_id"] for row in rows] == [f"trace-{trigger}" for trigger, _ in _TRIGGERS]
    dependencies = rows[0]["dependencies"]
    assert all(row["dependencies"] == dependencies for row in rows)
    assert isinstance(dependencies, Mapping)
    assert dependencies["sources"] == [{"ref": "OECD_DAC2A", "token": _REFRESH_DATE}]
    first = rows[0]
    assert first["materialiser_build"] == "test-build"
    assert first["code_index_stamp"] == deps.code_index.stamp.content_hash
    assert first["coverage_stamp"] == deps.coverage.snapshot().stamp
    assert first["graph_generation"] == deps.graph.generation_stamp().token
    selection = first["selection"]
    assert isinstance(selection, Mapping)
    assert {facet["axis"] for facet in selection["facets"]} == {
        "provenance",
        "measurementMethod",
        "unit",
    }
    head = deps.head_store.read_head(stored.spec_id)
    assert head is not None
    assert head.materialisation_id == rows[-1]["id"]
    assert head.dependencies == dependencies
    assert head.content_version == rows[-1]["content_version"]


def test_a_new_manifest_trace_carries_the_run_generation_stamp_and_source_tokens(version_store):
    """The trace's `graph_revision` is the graph generation stamp the run read, the value its
    materialisation row records, and `source_revisions` the refresh token of each source its
    dependency fingerprint names."""
    deps = _deps(version_store)
    stored, request = _registered(deps)

    manifest = serve_current(
        stored,
        resolve_requests=lambda: [request],
        options=MintOptions(bundle=False),
        trigger="mint",
        deps=deps,
        trace_id=None,
    )

    (row,) = _materialisation_rows(version_store)
    assert manifest.trace["graph_revision"] == deps.graph.generation_stamp().token
    assert manifest.trace["graph_revision"] == row["graph_generation"]
    assert manifest.trace["source_revisions"] == [
        {"source": "OECD_DAC2A", "revision": _REFRESH_DATE}
    ]


class _FailingHeadWriteTransaction:
    """Raises on the `subject_head` statement, after the statements before it ran."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._inner.execute(sql, params)

    def fetch_one(self, sql: str, params: Sequence[object] = ()):
        if "datasets.subject_head" in sql:
            raise RuntimeError("head write failed")
        return self._inner.fetch_one(sql, params)

    def fetch_all(self, sql: str, params: Sequence[object] = ()):
        return self._inner.fetch_all(sql, params)


class _FailingHeadWriteStore:
    def __init__(self, inner: FakeVersionRecordDatasetStore) -> None:
        self._inner = inner

    @contextmanager
    def transaction(self):
        with self._inner.transaction() as tx:
            yield _FailingHeadWriteTransaction(tx)


def test_a_head_write_that_fails_leaves_no_materialisation_row(version_store):
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record (subject_id, subject_kind, content_version, "
            "schema_version, immutable_body, recorded_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (_SPEC_ID, "spec", "cv1", 1, {}, "2026-01-01T00:00:00+00:00"),
        )
    failing = LiveHeadStore(store=cast(DatasetStore, _FailingHeadWriteStore(version_store)))

    with pytest.raises(DatasetsStoreUnavailable):
        failing.advance_head(
            materialisation=materialisation_record(subject_id=_SPEC_ID, content_version="cv1"),
            expected_materialisation_id=None,
        )

    assert _materialisation_rows(version_store) == []
    assert LiveHeadStore(store=version_store).read_head(_SPEC_ID) is None
