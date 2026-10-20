"""Tests a `latest` head's election and the year coverage a version records.

A `latest` run records the year the coverage index elected for each member in its dependency
fingerprint, with the facets and omitted entities the election counted. Past the recheck window
the head check elects again from the call's index over those series, and the head rebuilds only
when the index elects another year. A version's `coverage` states `recent_years`,
`elected_year` and `fullest_year` as the index its run elected over counted them, stored with
the version, so a rebuild, a head hit and a stored read state the same ones. A version stored
before versions held them gains them once, from one rebuild of its head.

The materialiser, `serve_current_head`, `serve_current` and the stores are real. The graph is
`FakeMintGraphReader`, the coverage index is built in memory, and SDMX is a mock transport
serving the rows `_Rig.live` holds, which a test sets apart from the index's.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from artifact_store import LiveArtifactInventory
from code_index.shared import get_index
from dataset_store.snapshot import LiveSnapshotReader, ManifestFacts
from datasets_api.adapter import build_identity_document, to_materialisation_request
from datasets_api.current import DefinitionDeps, serve_current, serve_current_head
from datasets_api.manifest import version_document, year_coverage
from datasets_api.manifest_models import Manifest
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.shape import resolve_shape
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from datasets_api.versions import Trigger, VersionRecord
from dc_sdmx import SdmxClient
from dfo_platform.timings import PhaseTimings, collecting
from materialise import EnrichmentLookup, MaterialisationRequest, MaterialisationResult
from materialise.graph import Facet
from materialise.rate_tables import RateTableCache
from materialise.window import LatestYear, ReadSeries, RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader, StoredDefinition
from statvar_members import MemberEnumerator
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_AID = "GrossOdaDisbursements"
_LOANS = "GrossOdaLoans"
_MWI = "country/MWI"
_PROVIDER = "financingProvider"
_RECIPIENT = "financingRecipient"
_PROVENANCE = "OECD_DAC2A"
_RECHECK = timedelta(seconds=900)
_START = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
_LATEST = RequestedWindow(kind="latest")
_CSV = MintOptions(bundle=False)

type Held = Mapping[str, Sequence[int]]
"""The years each donor's series to Malawi holds."""


def _donors(count: int, *, years: Sequence[int], first: int = 1) -> dict[str, Sequence[int]]:
    return {f"country/D{i:02d}": years for i in range(first, first + count)}


# Nine donors report 2023, eight of them 2024 and two 2025: 2025 holds under half of 2023's
# nine, so the three years elect 2024.
_MALAWI: Held = {
    **_donors(2, years=range(2021, 2026)),
    **_donors(6, years=range(2021, 2025), first=3),
    **_donors(1, years=range(2021, 2024), first=9),
}
# The same nine donors under a unit the request leaves out, each reporting 2025: counted
# with the dispatched facet's series, nine hold 2025 and the three years elect it.
_OTHER_UNIT: Held = _donors(9, years=(2025,))
# Every donor reports 2025 under the dispatched facet, which then elects it.
_ALL_REPORT_2025: Held = _donors(9, years=range(2021, 2026))


def _index(held: Mapping[str, Held], *, other_unit: Held = _OTHER_UNIT) -> CoverageIndex:
    """An index holding each StatVar of `held` at Malawi under the facet the requests bind
    (`usd`), and the first of them under a facet of another unit (`eur`) as `other_unit`."""
    rows = [
        (stat_var, "usd", donor, _MWI, list(years))
        for stat_var, donors in held.items()
        for donor, years in donors.items()
    ]
    rows += [
        (stat_var, "eur", donor, _MWI, list(years))
        for stat_var in list(held)[:1]
        for donor, years in other_unit.items()
    ]
    return build_coverage_index(
        rows,
        role_names=dict.fromkeys(held, RoleNames(donor=_PROVIDER, recipient=_RECIPIENT)),
        facet_metadata={
            unit: FacetMetadata(
                provenance_id=_PROVENANCE,
                measurement_method="GrantEquivalent",
                unit=ref,
                observation_period="P1Y",
            )
            for unit, ref in (("usd", "USD"), ("eur", "EUR"))
        },
        build_year=2026,
    )


class _Clock:
    def __init__(self) -> None:
        self.at = _START

    def __call__(self) -> datetime:
        return self.at


class _Rig:
    """Mints of aid to Malawi from every donor, served as `kinds.spec_id.serve_mint` serves
    them, against shared fakes and a clock. `live` holds the rows SDMX serves, the index's
    own unless a test sets them apart, and `reads` each data read's `c[TIME_PERIOD]`."""

    def __init__(
        self,
        version_store: FakeVersionRecordDatasetStore,
        *,
        indexed: Mapping[str, Held],
        live: Mapping[str, Held] | None = None,
        indicators: Mapping[str, Sequence[int]] = {},
    ) -> None:
        self.clock = _Clock()
        self.live = dict(live if live is not None else indexed)
        self.indicators = indicators
        self.reads: list[str | None] = []
        graph = FakeMintGraphReader(
            arcs={
                stat_var: {
                    "typeOf": ("StatisticalVariable",),
                    "observationProperties": (_PROVIDER, _RECIPIENT),
                    "measuredProperty": ("amount",),
                }
                for stat_var in indexed
            }
            | {stat_var: {"typeOf": ("StatisticalVariable",)} for stat_var in indicators}
            | {_PROVENANCE: {"lastDataRefreshDate": ("2026-10-07",)}},
            facets={
                (stat_var, _MWI): [
                    Facet(
                        facet_id="usd",
                        provenance_id=_PROVENANCE,
                        measurement_method="GrantEquivalent",
                        unit="USD",
                        obs_count=len(years),
                        earliest_date=str(min(years)),
                        latest_date=str(max(years)),
                    )
                ]
                for stat_var, years in indicators.items()
            },
            labels={_PROVENANCE: "OECD DAC2a", _MWI: "Malawi"},
        )
        store = FakeDatasetStore()
        self.deps = DefinitionDeps(
            registry=LiveSpecRegistry(store=store),
            spec_reader=LiveSpecReader(store=store),
            graph=graph,
            sdmx=SdmxClient(
                base_url="https://datacommons.one.org/core/api",
                transport=httpx.MockTransport(self._respond),
            ),
            artifact_store=_InventoryRecordingArtifactStore(
                inventory=LiveArtifactInventory(store=version_store)
            ),
            materialiser_build="test-build",
            capability_version="dset_test",
            ceiling=1_000_000,
            member_concurrency=1,
            enumerator=MemberEnumerator(),
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            code_index=get_index(),
            rate_cache=RateTableCache(),
            coverage=FakeCoverageService(_index(indexed)),
            version_writer=LiveVersionRecordWriter(store=version_store),
            snapshot_reader=LiveSnapshotReader(store=version_store),
            head_store=LiveHeadStore(store=version_store, now=self.clock),
            head_recheck_interval=_RECHECK,
            head_max_age=timedelta(hours=24),
            now=self.clock,
        )
        self.version_store = version_store
        self.stored: StoredDefinition
        self.requests: dict[str, MaterialisationRequest] = {}
        for stat_var in indexed:
            self.stored, self.requests[stat_var] = self._registered(stat_var, _RECIPIENT)
        for stat_var in indicators:
            self.stored, self.requests[stat_var] = self._registered(stat_var, "observationAbout")

    def _respond(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        period = params.get("c[TIME_PERIOD]")
        self.reads.append(period)
        stat_vars = params["c[variableMeasured]"].split(",")
        roles = (
            "observationAbout" if stat_vars[0] in self.indicators else f"{_PROVIDER},{_RECIPIENT}"
        )
        body = (
            f"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,{roles},unit,"
            "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
            "facetId\r\n"
        )
        for stat_var in stat_vars:
            series = (
                {_MWI: self.indicators[stat_var]}
                if stat_var in self.indicators
                else {f"{donor},{_MWI}": years for donor, years in self.live[stat_var].items()}
            )
            for entities, years in series.items():
                held = [str(year) for year in years]
                if period == "LATEST":
                    held = held[-1:]
                elif period is not None:
                    held = [year for year in held if year in period.split(",")]
                for year in held:
                    body += (
                        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{entities},USD,"
                        f"GrantEquivalent,P1Y,{_PROVENANCE},{year},1,,usd\r\n"
                    )
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    def _registered(
        self, stat_var: str, role: str
    ) -> tuple[StoredDefinition, MaterialisationRequest]:
        scope = Scope.model_validate(
            {
                "window": {"kind": "latest"},
                "entities": {role: [_MWI]},
                "facets": {
                    "provenance": {"ref": _PROVENANCE},
                    "measurementMethod": {"ref": "GrantEquivalent"},
                    "unit": {"ref": "USD"},
                },
            }
        )
        shape = resolve_shape(f"dcid:{stat_var}", graph=self.deps.graph)
        recomputed = recompute_identity(
            build_identity_document(shape, scope=scope, graph=self.deps.graph)
        )
        self.deps.registry.register_definition(
            identity_definition=recomputed.identity,
            schema_version=1,
            supplied_spec_id=recomputed.spec_id,
        )
        stored = self.deps.spec_reader.read_definition(recomputed.spec_id)
        assert stored is not None
        return stored, to_materialisation_request(
            stored.identity, shape=shape, code_index=self.deps.code_index
        )

    def index(self, held: Mapping[str, Held], **changes: Held) -> None:
        """Replace the coverage index the service serves."""
        self.deps.coverage = FakeCoverageService(_index(held, **changes))

    def mint(
        self,
        window: RequestedWindow = _LATEST,
        *,
        trigger: Trigger = "mint",
        options: MintOptions = _CSV,
    ) -> tuple[Manifest, dict[str, str]]:
        """Serve every registered StatVar as one frame under `window`, as a mint unless
        `trigger` names another caller, returning the response and the call's timing labels."""
        requests = [replace(request, window=window) for request in self.requests.values()]
        snapshot = self.deps.coverage.snapshot()
        timings = PhaseTimings()
        with collecting(timings):
            served = serve_current_head(
                self.stored,
                requests=requests,
                options=options,
                trigger=trigger,
                deps=self.deps,
                trace_id=None,
                coverage_snapshot=snapshot,
            )
            if not isinstance(served, Manifest):
                served = serve_current(
                    self.stored,
                    resolve_requests=lambda: requests,
                    options=options,
                    trigger=trigger,
                    deps=self.deps,
                    trace_id=None,
                    coverage_snapshot=snapshot,
                    head_read=served,
                )
        return served, dict(item.split("=", 1) for item in timings.as_log_field().split())

    def past_recheck(self) -> None:
        self.clock.at += _RECHECK + timedelta(seconds=1)

    def head_dependencies(self) -> dict[str, object]:
        head = self.deps.head_store.read_head(self.stored.spec_id)
        assert head is not None and head.dependencies is not None
        return dict(head.dependencies)

    def stored_coverage(self) -> dict[str, object]:
        """The `coverage` the store holds of the one version on record, which its
        `immutable_body` must repeat."""
        with self.version_store.transaction() as tx:
            row = tx.fetch_one("SELECT coverage, immutable_body FROM datasets.version_record")
        assert row is not None
        coverage, body = cast("tuple[dict[str, object], ...]", (*row.values(),))
        assert body["coverage"] == coverage
        return coverage

    def uncount(self) -> dict[str, object]:
        """Store the one version on record as a mint did before versions held year coverage:
        no year members and no member nouns. Returns the coverage it held."""
        counted = self.stored_coverage()
        dimensions = cast("list[dict[str, object]]", counted["dimensions"])
        coverage = {
            "rows": counted["rows"],
            "dimensions": [{k: v for k, v in d.items() if k != "noun"} for d in dimensions],
            "window": counted["window"],
        }
        with self.version_store.transaction() as tx:
            body = tx.fetch_one("SELECT immutable_body FROM datasets.version_record")
            assert body is not None
            tx.execute(
                "UPDATE datasets.version_record SET coverage = %s, immutable_body = %s",
                (
                    coverage,
                    {**cast("dict[str, object]", body["immutable_body"]), "coverage": coverage},
                ),
            )
        return counted

    def set_head_dependencies(self, dependencies: Mapping[str, object]) -> None:
        with self.version_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.subject_head SET dependencies = %s WHERE subject_id = %s",
                (dependencies, self.stored.spec_id),
            )


@pytest.fixture
def version_store() -> Generator[FakeVersionRecordDatasetStore]:
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


def _window(manifest: Manifest) -> tuple[object, object]:
    window = manifest.coverage.window or {}
    return window.get("start_year"), window.get("end_year")


def _year_coverage(manifest: Manifest) -> dict[str, object]:
    """The year coverage members of `manifest`'s wire `coverage`."""
    coverage = manifest.model_dump(mode="json")["coverage"]
    return {
        name: coverage[name]
        for name in ("recent_years", "elected_year", "fullest_year")
        if name in coverage
    }


def _counts(*years: tuple[int, int]) -> list[dict[str, object]]:
    """`recent_years` entries of `(year, donors)`, each donor one series to Malawi."""
    return [
        {"year": year, "series": donors, "role_counts": {_PROVIDER: donors, _RECIPIENT: 1}}
        for year, donors in years
    ]


def test_a_latest_head_holds_while_the_index_elects_the_year_it_recorded(version_store):
    """The dispatched facet's series elect 2024 and every facet's elect 2025, so a check
    counting every facet would rebuild the head on each pass."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    minted, fields = rig.mint()
    assert fields["rebuild_reason"] == "no_head"
    assert _window(minted) == (2024, 2024)
    assert rig.head_dependencies()["latest_years"] == [
        {
            "stat_var": f"dcid:{_AID}",
            "elected": 2024,
            "served": 2024,
            "facet_ids": ["usd"],
            "omitted": [],
        }
    ]

    for _ in range(2):
        rig.past_recheck()
        rig.reads.clear()
        again, fields = rig.mint()

        assert again.trace["served_from"] == "head"
        assert "rebuild_reason" not in fields
        assert again.content_version == minted.content_version
        assert rig.reads == []


def test_a_latest_head_rebuilds_when_the_index_elects_another_year(version_store):
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    minted, _ = rig.mint()
    rig.live[_AID] = _ALL_REPORT_2025
    rig.index({_AID: _ALL_REPORT_2025})
    rig.past_recheck()

    rebuilt, fields = rig.mint()

    assert fields["rebuild_reason"] == "latest_years"
    assert rebuilt.trace["served_from"] == "materialise"
    assert (_window(minted), _window(rebuilt)) == ((2024, 2024), (2025, 2025))


@pytest.mark.parametrize(
    "held",
    [
        pytest.param(_MALAWI, id="elects-an-earlier-year"),
        pytest.param(_ALL_REPORT_2025, id="elects-the-newest-period"),
    ],
)
def test_a_latest_head_that_records_no_election_rebuilds_once(version_store, held: Held):
    """The index elects a year for the head's StatVar and the head records none, whether or
    not that year is the one the newest period falls in."""
    rig = _Rig(version_store, indexed={_AID: held})
    minted, _ = rig.mint()
    recorded = rig.head_dependencies()
    rig.set_head_dependencies({k: v for k, v in recorded.items() if k != "latest_years"})
    rig.past_recheck()

    rebuilt, rebuilt_fields = rig.mint()
    rig.past_recheck()
    again, again_fields = rig.mint()

    assert rebuilt_fields["rebuild_reason"] == "latest_years"
    assert rebuilt.content_version == minted.content_version
    assert rig.head_dependencies() == recorded
    assert again.trace["served_from"] == "head"
    assert "rebuild_reason" not in again_fields


def _an_indicator(version_store: FakeVersionRecordDatasetStore) -> _Rig:
    return _Rig(version_store, indexed={}, indicators={"LifeExpectancy": (2022, 2023)})


def _a_variable_the_index_dropped(version_store: FakeVersionRecordDatasetStore) -> _Rig:
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    rig.mint()
    rig.index({_LOANS: _MALAWI})
    return rig


@pytest.mark.parametrize("build", [_an_indicator, _a_variable_the_index_dropped])
def test_a_latest_head_that_records_no_election_holds_when_the_index_elects_none(
    version_store, build
):
    rig = build(version_store)
    minted, _ = rig.mint()
    recorded = rig.head_dependencies()
    rig.set_head_dependencies({k: v for k, v in recorded.items() if k != "latest_years"})
    rig.past_recheck()

    again, fields = rig.mint()

    assert minted.coverage.rows > 0
    assert again.trace["served_from"] == "head"
    assert "rebuild_reason" not in fields


def test_a_head_states_the_year_coverage_its_run_recorded_when_the_index_moves(version_store):
    """Within the recheck window the head serves 2024 while the index now elects 2025."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    minted, _ = rig.mint()
    rig.index({_AID: _ALL_REPORT_2025})

    again, _ = rig.mint()

    assert again.trace["served_from"] == "head"
    assert _window(again) == _window(minted) == (2024, 2024)
    assert (
        _year_coverage(again)
        == _year_coverage(minted)
        == {"recent_years": _counts((2025, 2), (2024, 8), (2023, 9)), "elected_year": 2024}
    )


def test_a_head_that_fell_back_holds_and_states_no_year_coverage(version_store):
    """The index dates eight donors at 2024 and SDMX holds none of them there, so the run
    serves the newest period any series holds. The head records the year elected, which the
    index still elects, and the response states no count of an index that disagrees with
    its rows."""
    live = {donor: [year for year in years if year != 2024] for donor, years in _MALAWI.items()}
    rig = _Rig(version_store, indexed={_AID: _MALAWI}, live={_AID: live})
    minted, _ = rig.mint()
    assert _window(minted) == (2025, 2025)
    assert rig.reads == ["LATEST", "2024"]
    assert [
        (entry["elected"], entry["served"]) for entry in rig.head_dependencies()["latest_years"]
    ] == [(2024, 2025)]

    for _ in range(2):
        rig.past_recheck()
        again, fields = rig.mint()

        assert again.trace["served_from"] == "head"
        assert "rebuild_reason" not in fields
        assert _year_coverage(minted) == _year_coverage(again) == {}


def test_a_mint_states_its_year_coverage_from_a_rebuild_and_from_the_head(version_store):
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    minted, _ = rig.mint()
    from_head, _ = rig.mint()
    rendered, _ = rig.mint(options=MintOptions(bundle=False, format="parquet"))

    assert [m.trace["served_from"] for m in (minted, from_head, rendered)] == [
        "materialise",
        "head",
        "head",
    ]
    assert from_head.content_version == rendered.content_version == minted.content_version
    assert (
        _year_coverage(minted)
        == _year_coverage(from_head)
        == _year_coverage(rendered)
        == {"recent_years": _counts((2025, 2), (2024, 8), (2023, 9)), "elected_year": 2024}
    )
    # The frame counts its own donors under the label `role_counts` is keyed by, and names
    # them from its flow family, here none the code index knows.
    assert [
        (d.count, d.noun.plural if d.noun is not None else None)
        for m in (minted, from_head)
        for d in m.coverage.dimensions
        if d.label == _PROVIDER
    ] == [(8, "providers"), (8, "providers")]


def test_a_version_stores_its_year_coverage_and_every_read_states_it(version_store):
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    rebuilt, _ = rig.mint(trigger="get")
    from_head, _ = rig.mint(trigger="get")
    stored = rig.deps.snapshot_reader.read_version(rig.stored.spec_id, rebuilt.content_version)

    assert (rebuilt.trace["served_from"], from_head.trace["served_from"]) == ("materialise", "head")
    assert stored is not None
    document = cast("dict[str, object]", version_document(stored)["coverage"])
    assert (
        _year_coverage(rebuilt)
        == _year_coverage(from_head)
        == {name: document[name] for name in ("recent_years", "elected_year")}
        == {"recent_years": _counts((2025, 2), (2024, 8), (2023, 9)), "elected_year": 2024}
    )


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        pytest.param(
            RequestedWindow(kind="range", start_year=2025, end_year=2025),
            {"recent_years": _counts((2025, 2), (2024, 8), (2023, 9)), "elected_year": 2024},
            id="named-year-after-the-latest-windows",
        ),
        pytest.param(
            RequestedWindow(kind="range", start_year=2024, end_year=2024),
            {"recent_years": _counts((2025, 2), (2024, 8), (2023, 9)), "elected_year": 2024},
            id="named-year-the-latest-window-serves",
        ),
        pytest.param(
            RequestedWindow(kind="range", start_year=2023, end_year=2023),
            {},
            id="named-year-before-the-latest-windows",
        ),
        pytest.param(RequestedWindow(kind="range", start_year=2023, end_year=2024), {}, id="range"),
        pytest.param(RequestedWindow(kind="all"), {}, id="all"),
        pytest.param(
            RequestedWindow(kind="range", start_year=2025, end_year=2030),
            {},
            id="range-holding-one-year",
        ),
    ],
)
def test_year_coverage_by_window(version_store, window: RequestedWindow, expected):
    """A named year counts the series of the facet its run dispatched, from a rebuild and
    from the head. The index holds nine more at 2025 under a unit the request leaves out.
    A `latest` window serves 2024. A frame of that year or a later one states the
    members the `latest` frame states, and an earlier year states none.
    2023 is the fullest of the last ten years and one of the three stated, so no
    `fullest_year` repeats it."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    minted, _ = rig.mint(window)
    from_head, _ = rig.mint(window)

    assert (minted.trace["served_from"], from_head.trace["served_from"]) == ("materialise", "head")
    assert _year_coverage(minted) == _year_coverage(from_head) == expected


def test_a_named_year_states_the_year_a_latest_window_serves(version_store):
    rig = _Rig(version_store, indexed={_AID: _MALAWI})

    latest, _ = rig.mint()
    rig.past_recheck()
    named, _ = rig.mint(RequestedWindow(kind="range", start_year=2025, end_year=2025))

    assert (_window(latest), _window(named)) == ((2024, 2024), (2025, 2025))
    assert _year_coverage(named)["elected_year"] == _year_coverage(latest)["elected_year"] == 2024


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(
            RequestedWindow(kind="range", start_year=2025, end_year=2025), id="named-year"
        ),
        pytest.param(RequestedWindow(kind="range", start_year=2023, end_year=2024), id="range"),
    ],
)
def test_a_range_head_records_no_series_read_and_holds(version_store, window: RequestedWindow):
    """A range run's fingerprint holds no `read_series`, so the head matches past the
    recheck window."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    minted, _ = rig.mint(window)
    assert set(rig.head_dependencies()) == {
        "schema",
        "members",
        "sources",
        "conversion_sources",
        "recipes",
        "window",
    }
    rig.past_recheck()

    again, fields = rig.mint(window)

    assert again.trace["served_from"] == "head"
    assert "rebuild_reason" not in fields
    assert again.content_version == minted.content_version


def test_a_range_version_stored_before_year_coverage_is_left_as_it_is(version_store):
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    window = RequestedWindow(kind="range", start_year=2023, end_year=2024)
    rig.mint(window)
    rig.uncount()
    uncounted = rig.stored_coverage()

    within, within_fields = rig.mint(window)
    rig.past_recheck()
    again, fields = rig.mint(window)

    assert (within.trace["served_from"], again.trace["served_from"]) == ("head", "head")
    assert "rebuild_reason" not in within_fields | fields
    assert rig.stored_coverage() == uncounted


_YEAR_2025 = RequestedWindow(kind="range", start_year=2025, end_year=2025)
_YEAR_2023 = RequestedWindow(kind="range", start_year=2023, end_year=2023)
_LATEST_MEMBERS = {"recent_years": _counts((2025, 2), (2024, 8), (2023, 9)), "elected_year": 2024}


@pytest.mark.parametrize(
    ("window", "fingerprinted", "reason", "members"),
    [
        pytest.param(_LATEST, True, "year_coverage", _LATEST_MEMBERS, id="latest"),
        pytest.param(_LATEST, False, "latest_years", _LATEST_MEMBERS, id="latest-no-election"),
        pytest.param(_YEAR_2025, True, "year_coverage", _LATEST_MEMBERS, id="named-newest"),
        pytest.param(_YEAR_2023, True, "year_coverage", {}, id="named-earlier"),
    ],
)
def test_a_version_stored_before_year_coverage_gains_it_from_one_rebuild(
    version_store, window: RequestedWindow, fingerprinted: bool, reason: str, members
):
    """The head's version holds no year members and no member nouns, as one stored before
    versions held them, and `latest-no-election`'s head records no election either. The
    first read rebuilds, makes the same version and stores its coverage on it. An earlier
    named year gains the nouns alone, and that is enough to hold its head."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    minted, _ = rig.mint(window)
    counted = rig.uncount()
    if not fingerprinted:
        recorded = rig.head_dependencies()
        rig.set_head_dependencies({k: v for k, v in recorded.items() if k != "latest_years"})
        rig.past_recheck()

    rebuilt, rebuilt_fields = rig.mint(window)
    again, again_fields = rig.mint(window)
    rig.past_recheck()
    later, later_fields = rig.mint(window)

    assert rebuilt_fields["rebuild_reason"] == reason
    assert [m.trace["served_from"] for m in (rebuilt, again, later)] == [
        "materialise",
        "head",
        "head",
    ]
    assert "rebuild_reason" not in again_fields | later_fields
    assert {m.content_version for m in (rebuilt, again, later)} == {minted.content_version}
    assert rig.stored_coverage() == counted
    wire = [m.model_dump(mode="json")["coverage"] for m in (rebuilt, again, later)]
    assert wire == [counted] * 3
    assert _year_coverage(later) == members
    assert [d.get("noun", {}).get("plural") for d in wire[0]["dimensions"]] == [
        "providers",
        "recipients",
        None,
    ]


def test_a_second_writer_of_a_version_leaves_the_year_coverage_the_first_stored(version_store):
    """Two writers hold coverage counted from different indexes for one version stored with
    none. The first write stores its coverage. The second finds the version counted and
    writes nothing, so both read back the first's."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI})
    minted, _ = rig.mint(_YEAR_2025)
    counted = rig.uncount()
    stored = rig.deps.snapshot_reader.read_version(rig.stored.spec_id, minted.content_version)
    assert stored is not None and isinstance(stored.facts, ManifestFacts)

    def write(coverage: Mapping[str, object]) -> None:
        created = rig.deps.version_writer.record_version(
            VersionRecord(
                subject_id=stored.subject_id,
                subject_kind=stored.subject_kind,
                content_version=stored.content_version,
                schema_version=stored.schema_version,
                facts=replace(stored.facts, coverage=coverage),
            )
        )
        assert not created

    write(counted)
    write({**counted, "elected_year": 2025})
    # A writer holding an uncounted coverage stores nothing either.
    write(stored.facts.coverage)

    assert rig.stored_coverage() == counted


def _counted(
    index: CoverageIndex, *, year: int, window: RequestedWindow | None = None
) -> dict[str, object]:
    """The year coverage of a frame of aid to Malawi in `year` whose run read `usd`, under
    a window naming that year unless `window` is given."""
    request = MaterialisationRequest(
        stat_var=f"dcid:{_AID}",
        role_bindings={_RECIPIENT: [_MWI]},
        declared_roles=(_PROVIDER, _RECIPIENT),
        facet_bindings={},
        window=window or RequestedWindow(kind="range", start_year=year, end_year=year),
    )
    result = SimpleNamespace(
        coverage=SimpleNamespace(window={"start_year": year, "end_year": year}),
        read_series={request.stat_var: ReadSeries(facet_ids=("usd",))},
        latest_years={request.stat_var: LatestYear(elected=None, served=year)},
    )
    return year_coverage(cast("MaterialisationResult", result), requests=[request], index=index)


@pytest.mark.parametrize(
    ("build_year", "year", "stated"),
    [
        pytest.param(2026, 2028, False, id="after-the-build-year"),
        pytest.param(2026, 2026, True, id="the-build-year"),
        pytest.param(None, 2028, True, id="no-build-year"),
    ],
)
def test_a_named_year_after_the_build_year_states_no_year_coverage(
    build_year: int | None, year: int, stated: bool
):
    index = build_coverage_index(
        [(_AID, "usd", donor, _MWI, [2026, 2027, 2028]) for donor in _MALAWI],
        role_names={_AID: RoleNames(donor=_PROVIDER, recipient=_RECIPIENT)},
        build_year=build_year,
    )

    assert _counted(index, year=year).get("elected_year") == (year if stated else None)


def test_a_facet_finer_than_a_year_states_no_year_coverage():
    index = build_coverage_index(
        [(_AID, "usd", donor, _MWI, [2025]) for donor in _MALAWI],
        role_names={_AID: RoleNames(donor=_PROVIDER, recipient=_RECIPIENT)},
        facet_metadata={"usd": FacetMetadata(observation_period="P3M")},
        build_year=2026,
    )

    assert _counted(index, year=2025) == {}


def test_a_latest_variable_that_elected_no_year_states_no_year_coverage():
    assert _counted(_index({_AID: _MALAWI}), year=2025, window=_LATEST) == {}


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(_LATEST, id="latest"),
        pytest.param(RequestedWindow(kind="range", start_year=2024, end_year=2024), id="named"),
    ],
)
@pytest.mark.parametrize(
    ("donors_2019", "fullest"),
    [
        pytest.param(17, _counts((2019, 17))[0], id="under-the-share"),
        pytest.param(16, None, id="at-the-share"),
    ],
)
def test_the_fullest_year_is_stated_when_the_served_year_holds_under_the_share_of_it(
    version_store, donors_2019: int, fullest, window: RequestedWindow
):
    """Eight donors report 2024. Seventeen in 2019 leave it under half, and sixteen do not."""
    earlier = _donors(donors_2019 - 9, years=(2019,), first=10)
    held = {donor: (2019, *years) for donor, years in _MALAWI.items()} | earlier
    rig = _Rig(version_store, indexed={_AID: held})

    minted, _ = rig.mint(window)

    assert _window(minted) == (2024, 2024)
    assert _year_coverage(minted).get("fullest_year") == fullest


def test_a_frame_of_two_years_states_no_year_coverage(version_store):
    """Loans elect 2025 and aid 2024, so the frame holds both years."""
    rig = _Rig(version_store, indexed={_AID: _MALAWI, _LOANS: _ALL_REPORT_2025})

    minted, _ = rig.mint()

    assert _window(minted) == (2024, 2025)
    assert _year_coverage(minted) == {}


@pytest.mark.parametrize(
    ("aid", "loans", "year"),
    [
        pytest.param(_MALAWI, _donors(9, years=range(2018, 2021)), 2024, id="at-the-newer"),
        pytest.param(
            _donors(9, years=range(2019, 2023)),
            _donors(9, years=range(2019, 2022)),
            2021,
            id="at-the-older",
        ),
    ],
)
def test_a_named_year_of_variables_electing_different_years_states_no_year_coverage(
    version_store, aid: Held, loans: Held, year: int
):
    """A `latest` frame of the two holds both their years and states nothing."""
    rig = _Rig(version_store, indexed={_AID: aid, _LOANS: loans})

    minted, _ = rig.mint(RequestedWindow(kind="range", start_year=year, end_year=year))

    assert _window(minted) == (year, year)
    assert _year_coverage(minted) == {}


def test_a_frame_of_two_variables_at_one_year_counts_both(version_store):
    rig = _Rig(version_store, indexed={_AID: _MALAWI, _LOANS: _MALAWI})

    minted, _ = rig.mint()

    assert _window(minted) == (2024, 2024)
    assert _year_coverage(minted) == {
        "recent_years": [
            {"year": year, "series": 2 * donors, "role_counts": {_PROVIDER: donors, _RECIPIENT: 1}}
            for year, donors in ((2025, 2), (2024, 8), (2023, 9))
        ],
        "elected_year": 2024,
    }
