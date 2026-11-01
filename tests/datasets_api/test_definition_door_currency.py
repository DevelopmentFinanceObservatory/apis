"""Freshness tests pin when GET routes confirm a stored head by its dependency fingerprint or
re-materialise it. `_FakeHeadStore` lets tests seed and age heads without waiting on the real
UTC clock.
"""

from __future__ import annotations

import dataclasses
import itertools
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest

import datasets_api.current as current_module
import datasets_api.deps as deps_module
from artifact_store import ArtifactNotFoundError, LiveArtifactInventory, StoredArtifact, Tier
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.render import CANONICAL_FRAME_RENDERING
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveVersionRecordWriter, rendering_key
from datasets_api.versions import SubjectHead
from dc_graph import GraphReadError
from materialise.graph import Facet
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.trace import RecordedFamily
from series_coverage.index import CoverageIndex
from series_coverage.schema import ABSENT_ENTITY, CoverageManifest, FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import build_coverage_index
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    DEFAULT_REFRESH_DATE,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)

from ._frame_doubles import _version_artifact_rows, link_frame1_only
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import pinned_identity, register_body
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore, wire_definition_for
from .test_definition_door import (
    _CSV_HEADER,
    _DESCRIBED_TWO_MEMBER_IDENTITY,
    _ODA_GRANTS,
    _ODA_LOANS,
    _RECIPIENT,
    _STAT_VAR,
    _STAT_VAR_B,
    _STAT_VAR_REF,
    _coverage_service,
    _described_two_member_graph,
    _mint_one_spec,
    _multi_stat_var_sdmx_transport,
)
from .test_definition_door import (
    _client as _base_client,
)
from .test_materialise_semaphore import (
    _FrameTrackingPermits,
)
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore


class _RenderableArtifactStore(_InventoryRecordingArtifactStore):
    """Adds stored frame reads and put counting to the inventory-recording store."""

    def __init__(self, *, inventory: LiveArtifactInventory) -> None:
        super().__init__(inventory=inventory)
        self.put_calls = 0

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        self.put_calls += 1
        return super().put(data, tier=tier)

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        key = (artifact_digest, tier.value)
        if key not in self._objects:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        return self._objects[key]


def _client(
    *,
    graph,
    sdmx_transport,
    head_writer=None,
    store=None,
    with_version_store=True,
    artifact_store=None,
    code_index=None,
    coverage=None,
):
    """Builds a definition-door client with live version stores by default. Tests can inject
    artifact stores, a `CodeIndex`, and coverage fixtures."""
    version_store = FakeVersionRecordDatasetStore() if with_version_store else None
    if version_store is not None and artifact_store is None:
        artifact_store = _RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        )
    return _base_client(
        graph=graph,
        sdmx_transport=sdmx_transport,
        head_writer=head_writer,
        store=store,
        artifact_store=artifact_store,
        version_store=version_store,
        code_index=code_index,
        coverage=coverage,
    )


_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "observationProperties": ("recipient",),
    "measuredProperty": ("amount",),
}

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

# A distinct candidate the graph reports in place of `_FACET` once the source refreshes, with a
# later period.
_FACET_CHANGED = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=2,
    earliest_date="2020",
    latest_date="2021",
)

_PAST_INTERVAL = timedelta(minutes=20)  # past 900s (15 min), well inside the 24h max age
_PAST_MAX_AGE = timedelta(hours=25)  # past the 86400s (24h) default maximum age


def _row(*, stat_var_ref: str, obs_value: str, period: str = "2020") -> str:
    """Builds one CSV row with a prefixed StatVar ref and bare `variableMeasured` value.
    Vary `period` to change the materialised rows."""
    bare_stat_var = stat_var_ref.removeprefix("dcid:")
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare_stat_var},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{period},{obs_value},,facet1\r\n"
    )


def _mutable_graph(
    facets: dict[tuple[str, str], list[Facet]],
    arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> FakeMintGraphReader:
    """Builds a graph over the caller's mutable facet mapping and, when given, its mutable
    `arcs`, which a test writes a provenance's `lastDataRefreshDate` into."""
    graph_arcs = arcs if arcs is not None else {}
    graph_arcs.update(
        {
            _STAT_VAR: {**_ARCS, "name": ("Aid",)},
            _STAT_VAR_B: {**_ARCS, "name": ("Aid B",)},
            # Named so a download's default name columns fill and add no render warning.
            "GrantEquivalent": {"name": ("Grant equivalent",)},
            _RECIPIENT: {"name": ("Kenya",)},
        }
    )
    return FakeMintGraphReader(
        arcs=graph_arcs,
        facets=facets,
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


class _CoverageFromFacets:
    """Rebuilds coverage from the mutable facet mapping on each `current()` call.
    Observation counts become consecutive years from `earliest_date`."""

    def __init__(
        self, facets: Mapping[tuple[str, str], list[Facet]], *, stamp: str = "test-coverage-stamp"
    ) -> None:
        self._facets = facets
        self.stamp = stamp

    def current(self) -> CoverageIndex:
        rows: list[tuple[str, str, str, str, list[int]]] = []
        role_names: dict[str, RoleNames] = {}
        facet_metadata: dict[str, FacetMetadata] = {}
        for (stat_var, recipient), candidates in self._facets.items():
            role_names[stat_var] = RoleNames(donor=None, recipient="recipient")
            for facet in candidates:
                obs_count = facet.obs_count if facet.obs_count else 1
                earliest = int(facet.earliest_date or facet.latest_date)
                years = list(range(earliest, earliest + obs_count))
                rows.append((stat_var, facet.facet_id, ABSENT_ENTITY, recipient, years))
                facet_metadata[facet.facet_id] = FacetMetadata(
                    provenance_id=facet.provenance_id,
                    measurement_method=facet.measurement_method,
                    unit=facet.unit,
                )
        return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)

    def snapshot(self) -> CoverageSnapshot:
        return CoverageSnapshot(
            index=self.current(),
            manifest=CoverageManifest(
                stamp=self.stamp,
                build_id=f"{self.stamp}-build",
                built_at="t",
                horizon=self.current().horizon,
            ),
            stamp=self.stamp,
        )


class _Counter:
    def __init__(self) -> None:
        self.n = 0


def _counting_transport(body_ref: dict[str, str]) -> tuple[httpx.MockTransport, _Counter]:
    """Reads a mutable response body and counts each request."""
    counter = _Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        counter.n += 1
        return httpx.Response(
            200,
            text=body_ref["text"],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler), counter


class _FakeHeadStore:
    """Models compare-and-set head writes and lets tests age confirmation and materialisation
    timestamps independently."""

    def __init__(self) -> None:
        self._heads: dict[str, SubjectHead] = {}
        self._runs = itertools.count(1)
        self.read_head_calls = 0
        self.advance_head_calls = 0
        self.confirm_head_calls = 0

    def read_head(self, subject_id: str) -> SubjectHead | None:
        self.read_head_calls += 1
        return self._heads.get(subject_id)

    def advance_head(
        self,
        *,
        materialisation,
        family,
        basis,
        expected_materialisation_id: int | None,
        fence=None,
    ) -> SubjectHead | None:
        subject_id = materialisation.subject_id
        content_version = materialisation.content_version
        self.advance_head_calls += 1
        existing = self._heads.get(subject_id)
        stored_run = existing.materialisation_id if existing is not None else None
        if stored_run != expected_materialisation_id:
            return None
        now = datetime.now(UTC)
        written = SubjectHead(
            subject_id=subject_id,
            content_version=content_version,
            confirmed_at=now,
            materialised_at=now,
            materialisation_id=next(self._runs),
            dependencies=materialisation.dependencies,
            family=RecordedFamily(stated=family, confirmed_at=now, basis=basis),
        )
        self._heads[subject_id] = written
        return written

    def confirm_head(
        self, *, subject_id: str, content_version: str, family, basis
    ) -> SubjectHead | None:
        self.confirm_head_calls += 1
        existing = self._heads.get(subject_id)
        if existing is None or existing.content_version != content_version:
            return None
        confirmed_at = datetime.now(UTC)
        written = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=confirmed_at,
            materialised_at=existing.materialised_at,
            materialisation_id=existing.materialisation_id,
            dependencies=existing.dependencies,
            family=RecordedFamily(stated=family, confirmed_at=confirmed_at, basis=basis),
        )
        self._heads[subject_id] = written
        return written

    def head_for(self, subject_id: str) -> SubjectHead:
        return self._heads[subject_id]

    def drop_family(self, subject_id: str) -> None:
        """Leave the head as an image recording no family wrote it."""
        self._heads[subject_id] = dataclasses.replace(self._heads[subject_id], family=None)

    def age_head(
        self, subject_id: str, *, age: timedelta, materialised_age: timedelta | None = None
    ) -> None:
        """Ages `confirmed_at` by `age` and `materialised_at` by
        `materialised_age`, defaulting the latter to `age` itself so a
        caller aging both together (every existing call site) need only
        name one delta. A caller wanting to age a head that has been
        repeatedly confirmed without ever re-materialising passes the two
        separately.
        """
        resolved_materialised_age = age if materialised_age is None else materialised_age
        existing = self._heads[subject_id]
        confirmed_at = datetime.now(UTC) - age
        self._heads[subject_id] = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=confirmed_at,
            materialised_at=datetime.now(UTC) - resolved_materialised_age,
            materialisation_id=existing.materialisation_id,
            dependencies=existing.dependencies,
            family=(
                dataclasses.replace(existing.family, confirmed_at=confirmed_at)
                if existing.family is not None
                else None
            ),
        )


class _RaisingHeadStore:
    """Raises the store error used when a malformed head row cannot be decoded."""

    def read_head(self, subject_id: str):
        raise DatasetsStoreUnavailable("malformed head row")

    def advance_head(
        self,
        *,
        materialisation,
        family,
        basis,
        expected_materialisation_id: int | None,
        fence=None,
    ) -> None:
        raise DatasetsStoreUnavailable("malformed head row")

    def confirm_head(self, *, subject_id: str, content_version: str, family, basis) -> None:
        raise DatasetsStoreUnavailable("malformed head row")


def _trace_sans_currency(trace: dict) -> dict:
    return {k: v for k, v in trace.items() if k not in ("served_from", "confirmed_at")}


# Signed URLs and creation times vary per response. Compare stable content fields only.
_CONTENT_FIELDS = (
    "schema_version",
    "definition",
    "collection",
    "content_version",
    "coverage",
    "warnings",
)


def _content_fields(body: dict) -> dict:
    return {k: body.get(k) for k in _CONTENT_FIELDS}


# --- The definition GET: fresh -> within interval -> past interval ---


def test_spec_currency_sequence():
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "OECD_DAC2A": {"lastDataRefreshDate": ("2026-01-01",)}
    }
    graph = _mutable_graph(facets, arcs)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)

    # Minting writes the head, so the first GET serves it with no read.
    first = client.get(f"/api/datasets/{spec_id}")
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["trace"]["served_from"] == "head"
    assert "confirmed_at" in first_body["trace"]
    assert head_store.head_for(spec_id).dependencies["sources"] == [
        {"ref": "OECD_DAC2A", "token": "2026-01-01"}
    ]
    graph_calls_after_first = dict(graph.calls)
    sdmx_calls_after_first = sdmx_calls.n

    # Inside the interval, the second GET makes no graph or SDMX call.
    second = client.get(f"/api/datasets/{spec_id}")
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["trace"]["served_from"] == "head"
    assert second_body["trace"]["confirmed_at"] == first_body["trace"]["confirmed_at"]
    assert _content_fields(second_body) == _content_fields(first_body)
    assert _trace_sans_currency(second_body["trace"]) == _trace_sans_currency(first_body["trace"])
    assert graph.calls == graph_calls_after_first
    assert sdmx_calls.n == sdmx_calls_after_first

    # An unchanged fingerprint confirms the head without moving `materialised_at`, at the cost
    # of one graph read and no SDMX call.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    materialised_at_before_third = head_store.head_for(spec_id).materialised_at
    graph_calls_before_third = dict(graph.calls)
    third = client.get(f"/api/datasets/{spec_id}")
    assert third.status_code == 200, third.text
    third_body = third.json()
    assert third_body["trace"]["served_from"] == "head"
    assert third_body["trace"]["confirmed_at"] != first_body["trace"]["confirmed_at"]
    assert (
        third_body["trace"]["confirmed_at"] == head_store.head_for(spec_id).confirmed_at.isoformat()
    )
    assert head_store.head_for(spec_id).materialised_at == materialised_at_before_third
    assert _content_fields(third_body) == _content_fields(first_body)
    # Rehydrating the members reads the StatVar's arcs; the fingerprint is one more read.
    assert graph.calls["node_arcs_many"] == graph_calls_before_third["node_arcs_many"] + 1
    assert (
        graph.calls["observation_facets_batch_many"]
        == graph_calls_before_third["observation_facets_batch_many"]
    )
    assert sdmx_calls.n == sdmx_calls_after_first

    # A moved source refresh date triggers a fetch and advances the head.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    arcs["OECD_DAC2A"] = {"lastDataRefreshDate": ("2026-02-01",)}
    facets[(_STAT_VAR, _RECIPIENT)] = [_FACET_CHANGED]
    body_ref["text"] = _CSV_HEADER + _row(
        stat_var_ref=_STAT_VAR_REF, obs_value="2000", period="2021"
    )
    graph_calls_before_fourth = dict(graph.calls)
    fourth = client.get(f"/api/datasets/{spec_id}")
    assert fourth.status_code == 200, fourth.text
    fourth_body = fourth.json()
    assert fourth_body["trace"]["served_from"] == "materialise"
    assert fourth_body["content_version"] != first_body["content_version"]
    assert (
        graph.calls["observation_facets_batch_many"]
        == graph_calls_before_fourth["observation_facets_batch_many"]
    )
    # Only the real fetch reaches SDMX.
    assert sdmx_calls.n == sdmx_calls_after_first + 1
    assert head_store.head_for(spec_id).content_version == fourth_body["content_version"]


def test_provenance_refresh_date_move_forces_a_rematerialise_despite_unchanged_facets():
    """#439: a change to `lastDataRefreshDate` forces re-materialisation when the facets are
    unchanged."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    arcs = {
        _STAT_VAR: _ARCS,
        "OECD_DAC2A": {"lastDataRefreshDate": ("2026-01-01",)},
    }
    graph = FakeMintGraphReader(
        arcs=arcs,
        facets=facets,
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)

    first = client.get(f"/api/datasets/{spec_id}")
    assert first.status_code == 200, first.text
    assert first.json()["trace"]["served_from"] == "head"
    assert head_store.head_for(spec_id).dependencies["sources"] == [
        {"ref": "OECD_DAC2A", "token": "2026-01-01"}
    ]
    sdmx_calls_after_first = sdmx_calls.n

    # An unchanged refresh date confirms the head with no SDMX call.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    unaffected = client.get(f"/api/datasets/{spec_id}")
    assert unaffected.status_code == 200, unaffected.text
    assert unaffected.json()["trace"]["served_from"] == "head"
    assert sdmx_calls.n == sdmx_calls_after_first

    # A changed `lastDataRefreshDate` forces re-materialisation while facet fields stay fixed.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    arcs["OECD_DAC2A"] = {"lastDataRefreshDate": ("2026-02-01",)}
    changed = client.get(f"/api/datasets/{spec_id}")
    assert changed.status_code == 200, changed.text
    assert changed.json()["trace"]["served_from"] == "materialise"
    assert head_store.head_for(spec_id).dependencies["sources"] == [
        {"ref": "OECD_DAC2A", "token": "2026-02-01"}
    ]


_DAC_RECIPE = "ConversionRecipeDAC"
_DAC_TABLE_PROVENANCE = RECIPE_BODIES[_DAC_RECIPE].table_provenance

_CURRENCY_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="CurrencyUSD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)


def _converted_spec_fixture(
    facets: dict[tuple[str, str], list[Facet]],
) -> tuple[FakeMintGraphReader, dict[str, dict[str, tuple[str, ...]]], RateTableServer]:
    """A graph and SDMX server that convert `_STAT_VAR`'s US-dollar rows to euros under the DAC
    recipe. The returned `arcs` mapping is live: a test moves the table provenance's
    `lastDataRefreshDate` by writing to it."""
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        _STAT_VAR: {**_ARCS, **CONVERTIBLE_STAT_VAR_ARCS},
        **conversion_arcs(
            recipe_id=_DAC_RECIPE,
            provenance_id="OECD_DAC2A",
            currencies=("CurrencyUSD", "CurrencyEUR"),
        ),
    }
    graph = FakeMintGraphReader(arcs=arcs, facets=facets, labels={_RECIPIENT: "Kenya"})
    observation_body = _CSV_HEADER + (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},CurrencyUSD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    )

    def observations(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=observation_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    server = RateTableServer(
        recipe_table_bodies(
            _DAC_RECIPE, rates={"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}}
        ),
        fallback=observations,
    )
    return graph, arcs, server


def test_a_converted_spec_minted_through_the_door_confirms_its_head_until_the_rate_date_moves():
    """The mint stores the rate table's admitted `lastDataRefreshDate` as a conversion source in
    the head's fingerprint, so a recheck with an unchanged date confirms the head and fetches
    nothing. A moved date forces re-materialisation."""
    facets = {(_STAT_VAR, _RECIPIENT): [_CURRENCY_FACET]}
    graph, arcs, server = _converted_spec_fixture(facets)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=server.transport(),
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [_STAT_VAR_REF],
                "scope": {
                    "entities": {"recipient": [_RECIPIENT]},
                    "facets": {"unit": {"currency": "EUR", "prices": "current"}},
                },
            },
            "options": {"bundle": False},
        },
    )
    assert resp.status_code == 200, resp.text
    spec_id = resp.json()["definition"]["spec_id"]
    minted_version = resp.json()["content_version"]
    assert head_store.head_for(spec_id).dependencies["conversion_sources"] == [
        {"ref": _DAC_TABLE_PROVENANCE, "token": DEFAULT_REFRESH_DATE}
    ]
    requests_after_mint = len(server.requests)

    # Past the recheck interval with an unchanged rate date, the head is confirmed: no table
    # load, no observation fetch, no new head write.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    advances_before = head_store.advance_head_calls
    confirmed = client.get(f"/api/datasets/{spec_id}")
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["trace"]["served_from"] == "head"
    assert confirmed.json()["content_version"] == minted_version
    assert len(server.requests) == requests_after_mint
    assert head_store.advance_head_calls == advances_before

    # Only the rate-provenance date moves. Facet candidates stay fixed.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    arcs[_DAC_TABLE_PROVENANCE] = {"lastDataRefreshDate": ("2026-02-01T00:00:00Z",)}
    changed = client.get(f"/api/datasets/{spec_id}")
    assert changed.status_code == 200, changed.text
    assert changed.json()["trace"]["served_from"] == "materialise"
    assert head_store.head_for(spec_id).dependencies["conversion_sources"] == [
        {"ref": _DAC_TABLE_PROVENANCE, "token": "2026-02-01T00:00:00Z"}
    ]


def test_an_unconverted_head_names_no_conversion_source_and_serves_from_head():
    """An unconverted run records no conversion source or recipe, so its recheck reads only
    its data source and serves from head."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    dependencies = head_store.head_for(spec_id).dependencies
    assert (dependencies["conversion_sources"], dependencies["recipes"]) == ([], [])
    sdmx_calls_after_first = sdmx_calls.n

    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    second = client.get(f"/api/datasets/{spec_id}")
    assert second.status_code == 200, second.text
    assert second.json()["trace"]["served_from"] == "head"
    assert sdmx_calls.n == sdmx_calls_after_first


def test_member_with_no_provenance_refresh_properties_keeps_serving_from_head():
    """#439: a provenance without a refresh date records a `None` token, which equals itself,
    so the recheck confirms the head."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)  # `arcs` carries no "OECD_DAC2A" entry at all.
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)

    first = client.get(f"/api/datasets/{spec_id}")
    assert first.status_code == 200, first.text
    assert head_store.head_for(spec_id).dependencies["sources"] == [
        {"ref": "OECD_DAC2A", "token": None}
    ]
    sdmx_calls_after_first = sdmx_calls.n

    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    second = client.get(f"/api/datasets/{spec_id}")
    assert second.status_code == 200, second.text
    assert second.json()["trace"]["served_from"] == "head"
    assert sdmx_calls.n == sdmx_calls_after_first


def test_spec_past_the_maximum_age_re_materialises_despite_an_unchanged_fingerprint():
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    first = client.get(f"/api/datasets/{spec_id}")
    assert first.status_code == 200, first.text
    sdmx_calls_after_first = sdmx_calls.n

    # The fingerprint matches. The maximum age forces re-materialisation.
    head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "materialise"
    assert body["content_version"] == first.json()["content_version"]
    # Only the real fetch reaches SDMX.
    assert sdmx_calls.n == sdmx_calls_after_first + 1


def test_repeated_confirmations_do_not_extend_the_maximum_age():
    """Repeated confirmations do not extend the maximum age measured from `materialised_at`."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    first = client.get(f"/api/datasets/{spec_id}")
    assert first.status_code == 200, first.text
    sdmx_calls_after_first = sdmx_calls.n

    # Confirmation is recent, but materialisation is past its age limit and the fingerprint
    # matches.
    head_store.age_head(spec_id, age=_PAST_INTERVAL, materialised_age=_PAST_MAX_AGE)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "materialise"
    assert body["content_version"] == first.json()["content_version"]
    # Only the real fetch reaches SDMX.
    assert sdmx_calls.n == sdmx_calls_after_first + 1


# --- The options route: same states, no writes ---


def test_options_route_currency_confirms_and_advances_the_head_as_a_get_does():
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "OECD_DAC2A": {"lastDataRefreshDate": ("2026-01-01",)}
    }
    graph = _mutable_graph(facets, arcs)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    # A prior GET (not /options) establishes a real head to check against.
    definition_resp = client.get(f"/api/datasets/{spec_id}")
    assert definition_resp.status_code == 200, definition_resp.text
    writes_after_definition_get = head_store.advance_head_calls
    sdmx_calls_after_definition_get = sdmx_calls.n

    # Within the interval: serves from head, no read.
    within = client.get(f"/api/datasets/{spec_id}/options")
    assert within.status_code == 200, within.text
    assert within.json()["trace"]["served_from"] == "head"
    assert within.json()["trace"]["family_member_digest"] is not None
    assert head_store.advance_head_calls == writes_after_definition_get
    # A head with no recorded family states it unknown, as null keys on the wire.
    head_store.drop_family(spec_id)
    unknown = client.get(f"/api/datasets/{spec_id}/options").json()["trace"]
    assert (unknown["family_member_digest"], unknown["family_member_count"]) == (None, None)

    # `/options` confirms the head when the fingerprint matches, as a GET does.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    aged_confirmed_at = head_store.head_for(spec_id).confirmed_at.isoformat()
    graph_calls_before = dict(graph.calls)
    past_interval = client.get(f"/api/datasets/{spec_id}/options")
    assert past_interval.status_code == 200, past_interval.text
    past_interval_body = past_interval.json()
    assert past_interval_body["trace"]["served_from"] == "head"
    assert past_interval_body["trace"]["confirmed_at"] != aged_confirmed_at
    assert head_store.advance_head_calls == writes_after_definition_get
    # The comparison reads the source's refresh date once and reaches no SDMX.
    assert graph.calls["node_arcs_many"] == graph_calls_before["node_arcs_many"] + 1
    assert sdmx_calls.n == sdmx_calls_after_definition_get

    # A moved refresh date re-materialises, records the version and advances the head to it.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    arcs["OECD_DAC2A"] = {"lastDataRefreshDate": ("2026-02-01",)}
    facets[(_STAT_VAR, _RECIPIENT)] = [_FACET_CHANGED]
    body_ref["text"] = _CSV_HEADER + _row(
        stat_var_ref=_STAT_VAR_REF, obs_value="2000", period="2021"
    )
    changed = client.get(f"/api/datasets/{spec_id}/options")
    assert changed.status_code == 200, changed.text
    assert changed.json()["trace"]["served_from"] == "materialise"
    assert head_store.advance_head_calls == writes_after_definition_get + 1
    assert head_store.head_for(spec_id).content_version == changed.json()["content_version"]
    assert changed.json()["content_version"] != definition_resp.json()["content_version"]


def test_definition_and_options_head_hits_sign_no_artifacts():
    """A definition GET lists its artifacts under their download links and a within-interval
    `/options` head hit reads only the version and revocation state, so neither signs."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _base_client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)

    definition_resp = client.get(f"/api/datasets/{spec_id}")
    assert definition_resp.status_code == 200, definition_resp.text
    assert definition_resp.json()["artifacts"]

    within = client.get(f"/api/datasets/{spec_id}/options")
    assert within.status_code == 200, within.text
    assert within.json()["trace"]["served_from"] == "head"
    assert artifact_store.signings == []


# --- Store failure and route isolation ---


def test_malformed_head_row_surfaces_as_the_existing_store_failure_response():
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    store = FakeDatasetStore()
    minting_client = _client(
        graph=graph,
        sdmx_transport=transport,
        store=store,
        with_version_store=False,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(minting_client)
    retrieving_client = _client(
        graph=graph,
        sdmx_transport=transport,
        store=store,
        head_writer=_RaisingHeadStore(),
        with_version_store=False,
        coverage=_CoverageFromFacets(facets),
    )

    resp = retrieving_client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True


def test_snapshot_route_is_unaffected_by_currency():
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    definition_resp = client.get(f"/api/datasets/{spec_id}")
    content_version = definition_resp.json()["content_version"]

    snapshot_resp = client.get(f"/api/datasets/{spec_id}/{content_version}")

    assert snapshot_resp.status_code == 200, snapshot_resp.text
    snapshot_body = snapshot_resp.json()
    assert "served_from" not in snapshot_body["trace"]
    assert "confirmed_at" not in snapshot_body["trace"]
    assert _content_fields(snapshot_body) == _content_fields(definition_resp.json())


# --- Rendering on demand from a stored frame, on a head hit ---


@pytest.mark.parametrize("stored_frame", ["frame1", "canonical"])
def test_head_hit_renders_a_missing_rendering_on_demand_and_reuses_it(stored_frame: str):
    """A head hit renders a missing Parquet variant from the stored frame, a canonical frame or
    a frame1 document alone, and records it. The next identical GET reuses it without another
    artifact write."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _base_client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    if stored_frame == "frame1":
        link_frame1_only(
            version_store,
            artifact_store,
            subject_id=spec_id,
            content_version=head_store.head_for(spec_id).content_version,
        )
    sdmx_calls_after_mint = sdmx_calls.n
    puts_after_mint = artifact_store.put_calls

    first = client.get(f"/api/datasets/{spec_id}?format=parquet&bundle=false")

    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["trace"]["served_from"] == "head"
    assert len(first_body["artifacts"]) == 2
    requested = next(
        a for a in first_body["artifacts"] if a["rendering"] != {"statistics": "series"}
    )
    assert requested["media_type"] == "application/vnd.apache.parquet"
    assert sdmx_calls.n == sdmx_calls_after_mint
    puts_after_render = artifact_store.put_calls
    assert puts_after_render > puts_after_mint

    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=first_body["content_version"]
    )
    # The newly rendered `format=parquet` rendering, the statistics and a canonical frame all
    # use the Parquet media type. `rendering` distinguishes them.
    parquet_rows = [
        row
        for row in rows
        if row["media_type"] == "application/vnd.apache.parquet"
        and row["rendering"] not in ({"statistics": "series"}, CANONICAL_FRAME_RENDERING)
    ]
    assert len(parquet_rows) == 1

    second = client.get(f"/api/datasets/{spec_id}?format=parquet&bundle=false")

    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["trace"]["served_from"] == "head"
    assert len(second_body["artifacts"]) == 2
    requested_second = next(
        a for a in second_body["artifacts"] if a["rendering"] != {"statistics": "series"}
    )
    assert requested_second["media_type"] == "application/vnd.apache.parquet"
    assert artifact_store.put_calls == puts_after_render
    rows_after_second = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=first_body["content_version"]
    )
    assert len(rows_after_second) == len(rows)


def test_head_hit_reads_and_renders_the_stored_frame_inside_a_materialise_slot(
    monkeypatch: pytest.MonkeyPatch,
):
    """The frame read and the rendered artifact's write both happen on the work executor
    while the request holds a `bounded_materialise_slot`, and the slot is released once the
    response is built."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    version_store = FakeVersionRecordDatasetStore()
    slots_held = {"count": 0}
    store_calls_in_slot: list[tuple[str, bool]] = []
    frame_read_threads: list[str] = []

    class _SlotObservingArtifactStore(_RenderableArtifactStore):
        def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
            store_calls_in_slot.append(("put", slots_held["count"] > 0))
            return super().put(data, tier=tier)

        def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
            store_calls_in_slot.append(("get", slots_held["count"] > 0))
            frame_read_threads.append(threading.current_thread().name)
            return super().get(artifact_digest=artifact_digest, tier=tier)

    client = _base_client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=_FakeHeadStore(),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=_SlotObservingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    store_calls_in_slot.clear()
    real_slot = current_module.bounded_materialise_slot

    @contextmanager
    def counting_slot(semaphore, **kwargs) -> Iterator[None]:
        with real_slot(semaphore, **kwargs):
            slots_held["count"] += 1
            try:
                yield
            finally:
                slots_held["count"] -= 1

    monkeypatch.setattr(current_module, "bounded_materialise_slot", counting_slot)

    resp = client.get(f"/api/datasets/{spec_id}?format=parquet&bundle=false")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert ("get", True) in store_calls_in_slot
    assert ("put", True) in store_calls_in_slot
    assert all(in_slot for _call, in_slot in store_calls_in_slot)
    assert slots_held["count"] == 0
    assert frame_read_threads
    assert all(name.startswith("datasets-work") for name in frame_read_threads), frame_read_threads


def test_a_head_hit_render_frees_the_stored_frame_before_its_permit_is_released(
    monkeypatch: pytest.MonkeyPatch,
):
    """A head renders a missing format from its stored frame. When the render gives its
    permit back, nothing references the frame's bytes."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    version_store = FakeVersionRecordDatasetStore()
    permits = _FrameTrackingPermits(1)
    monkeypatch.setattr(deps_module, "MaterialisePermits", lambda _permits: permits)
    tracking: list[bool] = []

    class _TrackingArtifactStore(_RenderableArtifactStore):
        def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
            data = super().get(artifact_digest=artifact_digest, tier=tier)
            return permits.track(data) if tracking else data

    client = _base_client(
        graph=_mutable_graph(facets),
        sdmx_transport=transport,
        head_writer=_FakeHeadStore(),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=_TrackingArtifactStore(inventory=LiveArtifactInventory(store=version_store)),
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    tracking.append(True)

    resp = client.get(f"/api/datasets/{spec_id}?format=parquet&bundle=false")

    assert resp.status_code == 200, resp.text
    assert len(permits.read) == 1
    assert permits.freed_at_release == [True]


def test_head_hit_serves_an_already_stored_rendering_without_rendering_on_demand():
    """A stored CSV rendering serves directly on a head hit without an artifact write."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _base_client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    puts_after_mint = artifact_store.put_calls

    resp = client.get(f"/api/datasets/{spec_id}?format=csv&bundle=false")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "head"
    assert len(body["artifacts"]) == 2
    requested = next(a for a in body["artifacts"] if a["rendering"] != {"statistics": "series"})
    assert requested["media_type"] == "text/csv"
    assert artifact_store.put_calls == puts_after_mint


def test_head_hit_with_no_stored_frame_re_materialises_instead_of_rendering_on_demand():
    """Without its canonical frame, a head hit cannot render a missing format and must
    re-materialise."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _base_client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    sdmx_calls_after_mint = sdmx_calls.n
    with version_store.transaction() as tx:
        tx.execute(
            "DELETE FROM datasets.version_artifact WHERE subject_id = %s AND rendering_key = %s",
            (spec_id, rendering_key(CANONICAL_FRAME_RENDERING)),
        )

    resp = client.get(f"/api/datasets/{spec_id}?format=parquet&bundle=false")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "materialise"
    # The facet probe reads coverage. Only the real fetch reaches SDMX.
    assert sdmx_calls.n == sdmx_calls_after_mint + 1
    assert len(body["artifacts"]) == 2
    requested = next(a for a in body["artifacts"] if a["rendering"] != {"statistics": "series"})
    assert requested["media_type"] == "application/vnd.apache.parquet"


_GROUPBY_CODE_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-09T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)

_PURPOSE_HEALTH = CodeRow(
    dcid="dcid:CRSPurposeCode122",
    code_system="crsPurpose",
    code="122",
    label="Basic health care",
    synonyms=(),
    parent=None,
)
_PURPOSE_MALARIA = CodeRow(
    dcid="dcid:CRSPurposeCode12250",
    code_system="crsPurpose",
    code="12250",
    label="Malaria control",
    synonyms=(),
    parent=None,
)

_GROUPBY_CODE_INDEX = CodeIndex(
    stamp=_GROUPBY_CODE_STAMP,
    declarations=(
        CodeSystemDeclaration(
            code_system="crsPurpose",
            hierarchical=True,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="sector",
        ),
        # Summable but unused by the fixture. It lets `crsPurpose` collapse with `stat_var`.
        CodeSystemDeclaration(
            code_system="crsCooperationModality",
            hierarchical=False,
            disjoint_from_children=False,
            summable=True,
            absent_reading="not_declared",
            plain_name="modality",
        ),
    ),
    codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA),
    statvars=(
        # Decoding uses dcids alone, but enumeration checks the five-tuple. The synthetic
        # measured property sends membership resolution through the graph fixture.
        StatVarRow(
            dcid=_ODA_GRANTS,
            population_type="DevelopmentFinance",
            measured_property="amount_groupby_decode_fixture_only",
            stat_type=None,
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=("recipient",),
            constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid},
        ),
        StatVarRow(
            dcid=_ODA_LOANS,
            population_type="DevelopmentFinance",
            measured_property="amount_groupby_decode_fixture_only",
            stat_type=None,
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=("recipient",),
            constraint_properties={"crsPurpose": _PURPOSE_MALARIA.dcid},
        ),
    ),
)


def test_head_hit_stored_frame_groupby_refuses_a_zero_row_resolved_member_missing_from_it():
    """`_ODA_LOANS` resolves but contributes no rows. A stored-frame groupby must include it
    in the completeness check and refuse `rollup_not_summable`, as a fresh materialisation does."""
    graph = _described_two_member_graph()
    client = _client(
        graph=graph,
        sdmx_transport=_multi_stat_var_sdmx_transport(
            {
                _ODA_GRANTS: _CSV_HEADER
                + f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_GRANTS},{_RECIPIENT},USD,GrantEquivalent,"
                "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n",
                _ODA_LOANS: _CSV_HEADER,  # facet-viable, but zero data rows
            },
            probe_only_rows_by_ref={
                _ODA_LOANS: f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_LOANS},{_RECIPIENT},USD,"
                "GrantEquivalent,NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n",
            },
        ),
        head_writer=_FakeHeadStore(),
        code_index=_GROUPBY_CODE_INDEX,
        coverage=_coverage_service(stat_vars=(_ODA_GRANTS, _ODA_LOANS)),
    )
    spec_id = recompute_identity(_DESCRIBED_TWO_MEMBER_IDENTITY).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(
            definition=wire_definition_for(spec_id=spec_id, identity=_DESCRIBED_TWO_MEMBER_IDENTITY)
        ),
    )
    assert reg.status_code == 200, reg.text

    first = client.get(f"/api/datasets/{spec_id}")
    assert first.status_code == 200, first.text

    resp = client.get(f"/api/datasets/{spec_id}?groupby.by=crsCooperationModality&groupby.op=sum")

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "rollup_not_summable"
    assert problem["pointer"] == "/options/groupby"
    assert "Malaria control" in problem["detail"]


def test_a_head_hit_lists_only_the_matching_rendering_among_several_stored():
    """A head hit narrows to the ONE rendering its own request names
    (`read_snapshot_manifest`'s own `rendering_key` narrowing) plus the
    statistics artifact, always kept independent of that narrowing, so
    it lists exactly those two regardless of how many other
    renderings the version actually carries stored, and signs none. Three data renderings
    end up stored here: the mint's own `csv, bundle=false` plus two
    rendered on demand by the two GETs preceding the one this test
    measures.
    """
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _base_client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        artifact_store=artifact_store,
        coverage=_CoverageFromFacets(facets),
    )
    spec_id = _mint_one_spec(client)
    for query in ("format=parquet&bundle=false", "format=xlsx&bundle=false"):
        extra = client.get(f"/api/datasets/{spec_id}?{query}")
        assert extra.status_code == 200, extra.text
    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=extra.json()["content_version"]
    )
    # 5 rows: the 3 requested renderings, the mint's own `series-statistics/1`
    # artifact, and the record-only canonical frame row every mint stores regardless
    # of `options` (`render_and_store_canonical_frame`).
    listable_media_types = {
        row["media_type"] for row in rows if row["rendering"] != CANONICAL_FRAME_RENDERING
    }
    assert listable_media_types == {
        "text/csv",
        "application/vnd.apache.parquet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

    resp = client.get(f"/api/datasets/{spec_id}?format=csv&bundle=false")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "head"
    # The matching csv rendering, plus the statistics artifact -- always
    # included independent of narrowing (`read_snapshot_manifest`'s own
    # docstring).
    assert len(body["artifacts"]) == 2
    assert artifact_store.signings == []


_UNIT_STAT_VAR = "CurrencyRecheckDisbursements"

_UNIT_GRAPH_ARCS = {
    # The nominal qualifier and `CurrencyUSD` arcs satisfy native USD conversion and ISO
    # currency resolution without a rate table.
    _UNIT_STAT_VAR: {**_ARCS, "measurementQualifier": ("Nominal",)},
    "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
}


def _unit_request_identity() -> dict:
    """Builds a request-form unit binding. Scope-based mint bodies cannot express this shape,
    so tests post it as `kind: "spec_id"`."""
    return pinned_identity(
        stat_var=_UNIT_STAT_VAR,
        entities={"recipient": [_RECIPIENT]},
        facets={"unit": {"state": "value", "request": {"currency": "USD", "prices": "current"}}},
    )


def _unit_row(*, unit: str) -> str:
    """Builds a CSV row with the given unit. Other facet axes use the sole candidate."""
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_UNIT_STAT_VAR},{_RECIPIENT},{unit},GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    )


def _unit_sdmx_transport(body_ref: dict[str, str]) -> httpx.MockTransport:
    """Returns a mutable SDMX body for head priming and recheck probes."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_CSV_HEADER + body_ref["text"],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def test_a_rebuild_after_the_native_unit_turns_non_currency_remints_the_422():
    """When the elected unit changes from native `CurrencyUSD` to `Percent` with no source
    date moving, the head serves until its maximum age. The rebuild then returns 422
    `facet_binding_unsupported` without a pointer because the binding belongs to the stored
    identity."""
    graph = FakeMintGraphReader(
        arcs=_UNIT_GRAPH_ARCS,
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )
    body_ref = {"text": _unit_row(unit="CurrencyUSD")}
    facets = {
        (_UNIT_STAT_VAR, _RECIPIENT): [
            Facet(
                facet_id="facet1",
                provenance_id="OECD_DAC2A",
                measurement_method="GrantEquivalent",
                unit="CurrencyUSD",
                obs_count=1,
                earliest_date="2020",
                latest_date="2020",
            )
        ]
    }
    head_store = _FakeHeadStore()
    client = _client(
        graph=graph,
        sdmx_transport=_unit_sdmx_transport(body_ref),
        head_writer=head_store,
        coverage=_CoverageFromFacets(facets),
    )

    identity = _unit_request_identity()
    spec_id = recompute_identity(identity).spec_id
    definition = wire_definition_for(spec_id=spec_id, identity=identity)

    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": definition},
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text

    # The unit changes to `Percent`, and the head reaches its maximum age.
    head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    body_ref["text"] = _unit_row(unit="Percent")
    facets[(_UNIT_STAT_VAR, _RECIPIENT)] = [
        replace(facets[(_UNIT_STAT_VAR, _RECIPIENT)][0], unit="Percent")
    ]

    rechecked = client.get(f"/api/datasets/{spec_id}")

    assert rechecked.status_code == 422, rechecked.text
    problem = rechecked.json()
    assert problem["code"] == "facet_binding_unsupported"
    assert problem["retryable"] is False
    assert "pointer" not in problem


class _ReReadGraph(FakeMintGraphReader):
    """A graph whose withheld-provenance re-read fails while `re_read_fails`, and otherwise
    finds nothing withheld."""

    re_read_fails = True

    def withheld_at(self, variables, entities, *, deadline=None, max_bytes=None):
        if self.re_read_fails:
            raise GraphReadError("503", upstream_status=503)
        return {}


def test_an_empty_probe_writes_nothing_until_its_registry_check_completes():
    stat_var = "Count_Person"
    facets: dict[tuple[str, str], list[Facet]] = {}
    graph = _ReReadGraph(
        arcs={
            stat_var: {"typeOf": ("StatisticalVariable",), "measuredProperty": ("count",)},
            _RECIPIENT: {"typeOf": ("Country",), "name": ("Kenya",)},
        },
        facets=facets,
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )
    header = _CSV_HEADER.replace(",recipient,", ",observationAbout,")
    transport, sdmx_calls = _counting_transport(
        {"text": header + _row(stat_var_ref=f"dcid:{stat_var}", obs_value="7")}
    )
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=version_store))
    client = _base_client(
        graph=graph,
        sdmx_transport=transport,
        head_writer=head_store,
        version_store=version_store,
        artifact_store=artifact_store,
    )
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [f"dcid:{stat_var}"],
            "scope": {"entities": {"observationAbout": [_RECIPIENT]}},
        },
        "options": {"bundle": False},
    }

    # The re-read that would tell withheld data from no data fails, so the mint refuses and
    # writes no artifact, version or head.
    refused = client.post("/api/datasets", json=body)
    assert refused.status_code == 502, refused.text
    assert (refused.json()["code"], refused.json()["retryable"]) == ("upstream_error", True)
    assert artifact_store.put_calls == 0
    assert head_store.advance_head_calls == 0

    # The check completes and finds nothing withheld, so the empty version is written.
    graph.re_read_fails = False
    minted = client.post("/api/datasets", json=body)
    assert minted.status_code == 200, minted.text
    empty = minted.json()
    spec_id = empty["definition"]["spec_id"]
    assert empty["coverage"]["rows"] == 0
    assert head_store.head_for(spec_id).content_version == empty["content_version"]
    assert sdmx_calls.n == 0

    # Data arrives with no source date to show it, so the head serves until its maximum age,
    # when the rebuild finds a facet and a new version replaces the empty one.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    facets[(stat_var, _RECIPIENT)] = [_FACET]
    held = client.get(f"/api/datasets/{spec_id}")
    assert held.status_code == 200, held.text
    assert held.json()["content_version"] == empty["content_version"]
    head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    facets[(stat_var, _RECIPIENT)] = [_FACET]
    arrived = client.get(f"/api/datasets/{spec_id}")
    assert arrived.status_code == 200, arrived.text
    assert arrived.json()["trace"]["served_from"] == "materialise"
    assert arrived.json()["coverage"]["rows"] == 1
    assert arrived.json()["content_version"] != empty["content_version"]
    assert head_store.head_for(spec_id).content_version == arrived.json()["content_version"]
