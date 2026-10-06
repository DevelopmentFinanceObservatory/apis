"""Currency tests pin when GET routes confirm a stored head or re-materialise it.
`_FakeHeadStore` lets tests seed and age heads without waiting on the real UTC clock.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest

import datasets_api.doors.definition as definition_module
from artifact_store import ArtifactNotFoundError, LiveArtifactInventory, StoredArtifact, Tier
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import MemberSignal, SubjectHead
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.graph import Facet
from materialise.recipe_bodies import RECIPE_BODIES
from series_coverage.index import CoverageIndex
from series_coverage.schema import CoverageManifest, FacetMetadata, RoleNames
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

from ._frame1_doubles import _version_artifact_rows
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
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore


class _RenderableArtifactStore(_InventoryRecordingArtifactStore):
    """Adds frame1 reads and put counting to the inventory-recording store."""

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

# A distinct candidate a mutated probe reports in place of `_FACET`.
# It has a different `obs_count`/`latest_date`, so `_member_signal` derives
# a different `MemberSignal` from it.
_FACET_CHANGED = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=2,
    earliest_date="2020",
    latest_date="2021",
)

_SIGNAL = (MemberSignal("facet1", 1, "2020", "2020"),)

_PAST_INTERVAL = timedelta(minutes=20)  # past 900s (15 min), well inside the 24h max age
_PAST_MAX_AGE = timedelta(hours=25)  # past the 86400s (24h) default maximum age


def _row(*, stat_var_ref: str, obs_value: str, period: str = "2020") -> str:
    """Builds one CSV row with a prefixed StatVar ref and bare `variableMeasured` value.
    Vary `period` to change the derived signal's latest date."""
    bare_stat_var = stat_var_ref.removeprefix("dcid:")
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare_stat_var},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{period},{obs_value},,facet1\r\n"
    )


def _mutable_graph(facets: dict[tuple[str, str], list[Facet]]) -> FakeMintGraphReader:
    """Builds a graph over the caller's mutable facet mapping."""
    return FakeMintGraphReader(
        arcs={
            _STAT_VAR: {**_ARCS, "name": ("Aid",)},
            _STAT_VAR_B: {**_ARCS, "name": ("Aid B",)},
            # Named so a download's default name columns fill and add no render warning.
            "GrantEquivalent": {"name": ("Grant equivalent",)},
            _RECIPIENT: {"name": ("Kenya",)},
        },
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
            role_names[stat_var] = RoleNames(donor="provider", recipient="recipient")
            for facet in candidates:
                obs_count = facet.obs_count if facet.obs_count else 1
                earliest = int(facet.earliest_date or facet.latest_date)
                years = list(range(earliest, earliest + obs_count))
                rows.append((stat_var, facet.facet_id, "country/USA", recipient, years))
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
                source="fake",
                row_counts={},
                files={},
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
        self.read_head_calls = 0
        self.advance_head_calls = 0
        self.confirm_head_calls = 0

    def read_head(self, subject_id: str) -> SubjectHead | None:
        self.read_head_calls += 1
        return self._heads.get(subject_id)

    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal,
        expected_content_version: str | None,
    ) -> SubjectHead | None:
        self.advance_head_calls += 1
        existing = self._heads.get(subject_id)
        stored_content_version = existing.content_version if existing is not None else None
        if stored_content_version != expected_content_version:
            return None
        now = datetime.now(UTC)
        written = SubjectHead(
            subject_id=subject_id,
            content_version=content_version,
            confirmed_at=now,
            materialised_at=now,
            signal=tuple(signal),
        )
        self._heads[subject_id] = written
        return written

    def confirm_head(self, *, subject_id: str, content_version: str) -> SubjectHead | None:
        self.confirm_head_calls += 1
        existing = self._heads.get(subject_id)
        if existing is None or existing.content_version != content_version:
            return None
        written = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=datetime.now(UTC),
            materialised_at=existing.materialised_at,
            signal=existing.signal,
        )
        self._heads[subject_id] = written
        return written

    def head_for(self, subject_id: str) -> SubjectHead:
        return self._heads[subject_id]

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
        self._heads[subject_id] = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=datetime.now(UTC) - age,
            materialised_at=datetime.now(UTC) - resolved_materialised_age,
            signal=existing.signal,
        )


class _RaisingHeadStore:
    """Raises the store error used when a malformed head row cannot be decoded."""

    def read_head(self, subject_id: str):
        raise DatasetsStoreUnavailable("malformed head row")

    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal,
        expected_content_version: str | None,
    ) -> None:
        raise DatasetsStoreUnavailable("malformed head row")

    def confirm_head(self, *, subject_id: str, content_version: str) -> None:
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

    # Minting primes the head, so the first GET serves without probing.
    first = client.get(f"/api/datasets/{spec_id}")
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["trace"]["served_from"] == "head"
    assert "confirmed_at" in first_body["trace"]
    assert head_store.head_for(spec_id).signal == _SIGNAL
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

    # An unchanged probe confirms the head without moving `materialised_at`. Coverage supplies
    # candidates, so the probe makes no graph or SDMX call.
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
    assert (
        graph.calls["observation_facets_batch_many"]
        == graph_calls_before_third["observation_facets_batch_many"]
    )
    assert sdmx_calls.n == sdmx_calls_after_first

    # A changed coverage signal triggers a fetch and advances the head.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
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
    # The facet probe reads coverage. Only the real fetch reaches SDMX.
    assert sdmx_calls.n == sdmx_calls_after_first + 1
    assert head_store.head_for(spec_id).content_version == fourth_body["content_version"]


def test_provenance_refresh_date_move_forces_a_rematerialise_despite_an_unchanged_probe():
    """#439: a change to `lastDataRefreshDate` forces re-materialisation when the facet signal
    is unchanged."""
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
    assert head_store.head_for(spec_id).signal == (
        MemberSignal("facet1", 1, "2020", "2020", provenance_last_refresh="2026-01-01"),
    )
    sdmx_calls_after_first = sdmx_calls.n

    # An unchanged refresh date confirms the head. Coverage needs no SDMX call.
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
    assert head_store.head_for(spec_id).signal == (
        MemberSignal("facet1", 1, "2020", "2020", provenance_last_refresh="2026-02-01"),
    )


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
    """The mint stores the rate table's admitted `lastDataRefreshDate` in the head signal, so a
    recheck with an unchanged date confirms the head and fetches nothing. A moved date forces
    re-materialisation."""
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
    assert head_store.head_for(spec_id).signal[0].rate_provenance_refresh == (
        (_DAC_TABLE_PROVENANCE, DEFAULT_REFRESH_DATE),
    )
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
    assert head_store.head_for(spec_id).signal[0].rate_provenance_refresh == (
        (_DAC_TABLE_PROVENANCE, "2026-02-01T00:00:00Z"),
    )


def test_stored_head_without_rate_provenance_refresh_keeps_todays_behaviour():
    """A signal without rate provenance skips rate-date rechecks and serves from head."""
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
    assert head_store.head_for(spec_id).signal[0].rate_provenance_refresh == ()
    sdmx_calls_after_first = sdmx_calls.n

    # The facet probe reaches coverage, not SDMX.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    second = client.get(f"/api/datasets/{spec_id}")
    assert second.status_code == 200, second.text
    assert second.json()["trace"]["served_from"] == "head"
    assert sdmx_calls.n == sdmx_calls_after_first


def test_member_with_no_provenance_refresh_properties_keeps_serving_from_head():
    """#439: provenances without refresh-date fields leave both signal values `None`, so an
    unchanged probe can confirm the head."""
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
    assert head_store.head_for(spec_id).signal == (
        MemberSignal(
            "facet1",
            1,
            "2020",
            "2020",
            provenance_last_refresh=None,
            provenance_next_refresh=None,
        ),
    )
    sdmx_calls_after_first = sdmx_calls.n

    # The facet probe reaches coverage, not SDMX.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    second = client.get(f"/api/datasets/{spec_id}")
    assert second.status_code == 200, second.text
    assert second.json()["trace"]["served_from"] == "head"
    assert sdmx_calls.n == sdmx_calls_after_first
    assert head_store.head_for(spec_id).signal[0].provenance_last_refresh is None


def test_spec_past_the_maximum_age_re_materialises_despite_an_unchanged_probe():
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

    # The signal matches. The maximum age forces re-materialisation.
    head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "materialise"
    assert body["content_version"] == first.json()["content_version"]
    # The facet probe reads coverage. Only the real fetch reaches SDMX.
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

    # Confirmation is recent, but materialisation is past its age limit and the signal matches.
    head_store.age_head(spec_id, age=_PAST_INTERVAL, materialised_age=_PAST_MAX_AGE)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "materialise"
    assert body["content_version"] == first.json()["content_version"]
    # The facet probe reads coverage. Only the real fetch reaches SDMX.
    assert sdmx_calls.n == sdmx_calls_after_first + 1


# --- The definition GET over a collection: same states, per-member signal ---


# --- The options route: same states, no writes ---


def test_options_route_currency_never_writes_the_head():
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
    # A prior GET (not /options) establishes a real head to check against.
    definition_resp = client.get(f"/api/datasets/{spec_id}")
    assert definition_resp.status_code == 200, definition_resp.text
    writes_after_definition_get = head_store.advance_head_calls
    sdmx_calls_after_definition_get = sdmx_calls.n

    # Within the interval: serves from head, no probe.
    within = client.get(f"/api/datasets/{spec_id}/options")
    assert within.status_code == 200, within.text
    assert within.json()["trace"]["served_from"] == "head"
    assert head_store.advance_head_calls == writes_after_definition_get

    # `/options` leaves `confirmed_at` unchanged when the probe matches.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    aged_confirmed_at = head_store.head_for(spec_id).confirmed_at.isoformat()
    graph_calls_before = dict(graph.calls)
    past_interval = client.get(f"/api/datasets/{spec_id}/options")
    assert past_interval.status_code == 200, past_interval.text
    past_interval_body = past_interval.json()
    assert past_interval_body["trace"]["served_from"] == "head"
    assert past_interval_body["trace"]["confirmed_at"] == aged_confirmed_at
    assert head_store.advance_head_calls == writes_after_definition_get
    # This role-bearing StatVar's facet candidates come from coverage, an
    # in-memory table, so the probe reaches neither the graph nor SDMX.
    assert (
        graph.calls["observation_facets_batch_many"]
        == graph_calls_before["observation_facets_batch_many"]
    )
    assert sdmx_calls.n == sdmx_calls_after_definition_get

    # A changed probe re-materialises without writing a version record or head.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    facets[(_STAT_VAR, _RECIPIENT)] = [_FACET_CHANGED]
    body_ref["text"] = _CSV_HEADER + _row(
        stat_var_ref=_STAT_VAR_REF, obs_value="2000", period="2021"
    )
    changed = client.get(f"/api/datasets/{spec_id}/options")
    assert changed.status_code == 200, changed.text
    assert changed.json()["trace"]["served_from"] == "materialise"
    assert head_store.advance_head_calls == writes_after_definition_get
    # The head retains the mint's content version.
    assert head_store.head_for(spec_id).content_version == definition_resp.json()["content_version"]


class _SigningCountingArtifactStore(_RenderableArtifactStore):
    """Counts signed URL requests while retaining the frame1 reads required by the definition GET."""

    def __init__(self, *, inventory: LiveArtifactInventory) -> None:
        super().__init__(inventory=inventory)
        self.signing_calls = 0

    def signed_get_url(
        self, digest: str, *, tier, ttl, response_type: str, response_disposition: str
    ) -> str:
        self.signing_calls += 1
        return super().signed_get_url(
            digest,
            tier=tier,
            ttl=ttl,
            response_type=response_type,
            response_disposition=response_disposition,
        )


def test_options_head_hit_signs_no_artifacts():
    """The definition GET signs artifacts. A within-interval `/options` head hit signs none
    because it reads only the version and revocation state."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    head_store = _FakeHeadStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _SigningCountingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
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
    assert artifact_store.signing_calls > 0
    signing_calls_after_definition_get = artifact_store.signing_calls

    within = client.get(f"/api/datasets/{spec_id}/options")
    assert within.status_code == 200, within.text
    assert within.json()["trace"]["served_from"] == "head"
    assert artifact_store.signing_calls == signing_calls_after_definition_get


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


# --- Rendering on demand from a stored frame1, on a head hit ---


def test_head_hit_renders_a_missing_rendering_on_demand_and_reuses_it():
    """A head hit renders a missing Parquet variant from stored frame1 and records it. The
    next identical GET reuses it without another artifact write."""
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
    # Both the newly rendered `format=parquet` rendering and the statistics
    # Both artifacts use the Parquet media type. `rendering` distinguishes them.
    parquet_rows = [
        row
        for row in rows
        if row["media_type"] == "application/vnd.apache.parquet"
        and row["rendering"] != {"statistics": "series"}
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


def test_head_hit_reads_and_renders_the_stored_frame1_inside_a_materialise_slot(
    monkeypatch: pytest.MonkeyPatch,
):
    """The frame1 read and the rendered artifact's write both happen while the request holds
    a `bounded_materialise_slot`, and the slot is released once the response is built."""
    facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
    graph = _mutable_graph(facets)
    body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
    transport, _sdmx_calls = _counting_transport(body_ref)
    version_store = FakeVersionRecordDatasetStore()
    slots_held = {"count": 0}
    store_calls_in_slot: list[tuple[str, bool]] = []

    class _SlotObservingArtifactStore(_RenderableArtifactStore):
        def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
            store_calls_in_slot.append(("put", slots_held["count"] > 0))
            return super().put(data, tier=tier)

        def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
            store_calls_in_slot.append(("get", slots_held["count"] > 0))
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
    real_slot = definition_module.bounded_materialise_slot

    @contextmanager
    def counting_slot(semaphore, **kwargs) -> Iterator[None]:
        with real_slot(semaphore, **kwargs):
            slots_held["count"] += 1
            try:
                yield
            finally:
                slots_held["count"] -= 1

    monkeypatch.setattr(definition_module, "bounded_materialise_slot", counting_slot)

    resp = client.get(f"/api/datasets/{spec_id}?format=parquet&bundle=false")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert ("get", True) in store_calls_in_slot
    assert ("put", True) in store_calls_in_slot
    assert all(in_slot for _call, in_slot in store_calls_in_slot)
    assert slots_held["count"] == 0


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


def test_head_hit_with_no_stored_frame1_re_materialises_instead_of_rendering_on_demand():
    """Without its frame1 artifact, a head hit cannot render a missing format and must
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
            "DELETE FROM datasets.version_artifact WHERE subject_id = %s AND media_type = %s",
            (spec_id, FRAME1_MEDIA_TYPE),
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


def test_signing_counts_only_the_matching_rendering_among_several_stored():
    """A head hit narrows to the ONE rendering its own request names
    (`read_snapshot_manifest`'s own `rendering_key` narrowing) plus the
    statistics artifact, always kept independent of that narrowing, so
    signing reaches exactly those two regardless of how many other
    renderings the version actually carries stored. Three data renderings
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
    artifact_store = _SigningCountingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
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
    # artifact, and the record-only frame1 row every mint stores regardless
    # of `options` (`render_and_store_frame1`).
    listable_media_types = {
        row["media_type"] for row in rows if row["media_type"] != FRAME1_MEDIA_TYPE
    }
    assert listable_media_types == {
        "text/csv",
        "application/vnd.apache.parquet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    signing_calls_before = artifact_store.signing_calls

    resp = client.get(f"/api/datasets/{spec_id}?format=csv&bundle=false")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "head"
    # The matching csv rendering, plus the statistics artifact -- always
    # included independent of narrowing (`read_snapshot_manifest`'s own
    # docstring) -- both get signed.
    assert len(body["artifacts"]) == 2
    assert artifact_store.signing_calls == signing_calls_before + 2


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


def test_head_recheck_declines_on_a_non_currency_native_unit_and_serve_remints_the_422():
    """When the elected unit changes from native `CurrencyUSD` to plain `USD`, the head
    recheck declines comparison. Re-materialisation returns 422 `facet_binding_unsupported`
    without a pointer because the binding belongs to the stored identity."""
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
    assert head_store.head_for(spec_id).signal == (MemberSignal("facet1", 1, "2020", "2020"),)

    # The unit changes to plain `USD`. The recheck declines comparison before the 422 response.
    head_store.age_head(spec_id, age=_PAST_INTERVAL)
    body_ref["text"] = _unit_row(unit="USD")
    facets[(_UNIT_STAT_VAR, _RECIPIENT)] = [
        replace(facets[(_UNIT_STAT_VAR, _RECIPIENT)][0], unit="USD")
    ]

    rechecked = client.get(f"/api/datasets/{spec_id}")

    assert rechecked.status_code == 422, rechecked.text
    problem = rechecked.json()
    assert problem["code"] == "facet_binding_unsupported"
    assert problem["retryable"] is False
    assert "pointer" not in problem
