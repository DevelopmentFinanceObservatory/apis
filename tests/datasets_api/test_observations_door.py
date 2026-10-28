"""Tests `GET /api/datasets/{spec_id}/{content_version}/observations` over in-memory fakes: each
refusal, each preview mode, labels, conversion, and a pinned version read past expiry."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from fastapi.testclient import TestClient

import search.engine.region.observations.bulk_role_series
import search.observations
import tests.fixtures
from artifact_store import Tier
from code_index import CodeIndex, CodeRow, IndexStamp, StatVarRow
from code_index.types import PartyRow
from dataset_store.snapshot import WORKING_TIER_RETENTION, Revocation, StoredVersion
from datasets_api.admission import MaterialisePermits
from datasets_api.codes import DATASETS_CODE_CATALOGUE
from datasets_api.doors.observations import build_observations_door, observations_body
from datasets_api.doors.pin import pin_artifact
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_platform.app import Surface, create_app
from materialise.formats import CANONICAL_FRAME_MEDIA_TYPE, FRAME1_MEDIA_TYPE
from materialise.graph import GraphReadError
from materialise.series_conversion import SeriesConversion, pack_series_conversions
from origin_catalogue import DFO, bound_catalogue
from search.engine.clients.graph import Facet
from search.engine.retrieval.member_facts import ArcsMemberFacts, IndexMemberFacts
from search.models import TimeWindow
from search.observations import ObservationSettings
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import build_observations_planned
from tests.fixtures import FakeGraph
from tests.materialise._canonical_frame import canonical_frame_of_frame1
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row

from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import (
    FakeSnapshotReader,
    FakeSpecReader,
    StoreBackedSnapshotReader,
    _store_and_locate,
)
from ._version_facts import facts_from_body
from .test_materialise_semaphore import _FrameTrackingPermits
from .test_pin_door import FakeArtifactReferenceWriter

_NOW = datetime(2026, 6, 1, tzinfo=UTC)
_SPEC = "spec_a"
_CV = "cv1"
_PATH = f"/api/datasets/{_SPEC}/{_CV}/observations"
_PROV = "OECD_DAC2A"
_COLUMNS_STANDARD = (
    "stat_var",
    "observationAbout",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)
_COLUMNS_FLOW = (
    "stat_var",
    "provider",
    "recipient",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)
_HEALTH_SV = "Aid_Health"
_EDU_SV = "Aid_Education"
_LABELS = {
    "country/KEN": "Kenya",
    "country/GHA": "Ghana",
    "country/NOR": "Norway",
    "country/FRA": "France",
    "Count_Person": "Population",
    "CRSPurposeCode121": "Basic health (CRS Purpose)",
    "CRSPurposeCode111": "Education, level unspecified (CRS Purpose)",
    "CurrencyEUR": "Euro",
    "GrantEquivalent": "Grant equivalent",
    _HEALTH_SV: "Raw health StatVar name",
    _EDU_SV: "Raw education StatVar name",
}


def _index() -> CodeIndex:
    """One CRS code row whose label drops the graph's suffix, the health StatVar's row, and the
    flow's parties. The education StatVar is left to the graph's arcs."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=(
            CodeRow(
                dcid="CRSPurposeCode121",
                code_system="crsPurpose",
                code="121",
                label="Basic health",
                synonyms=(),
                parent=None,
            ),
        ),
        statvars=(
            StatVarRow(
                dcid=_HEALTH_SV,
                population_type="ODA",
                measured_property="amount",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("provider", "recipient"),
                constraint_properties={"crsPurpose": "CRSPurposeCode121"},
            ),
        ),
        parties=(
            PartyRow(dcid="country/FRA", kind="party"),
            PartyRow(dcid="country/GHA", kind="party"),
            PartyRow(dcid="country/KEN", kind="party"),
        ),
    )


def _graph(*, fail: bool = False) -> FakeMintGraphReader:
    graph = FakeMintGraphReader(
        labels=_LABELS,
        arcs={
            _EDU_SV: {
                "constraintProperties": ("crsPurpose",),
                "crsPurpose": ("CRSPurposeCode111",),
            }
        },
    )
    if fail:

        def broken(dcids: list[str]) -> dict[str, str]:
            raise GraphReadError("graph down")

        graph.node_labels_batch = broken
    return graph


@pytest.fixture(autouse=True, params=["canonical", "frame1"])
def frame_storage(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs every case over a version storing its canonical frame and over one storing only its
    frame1 document."""
    monkeypatch.setattr(_World, "storage", request.param)
    return request.param


def _frame(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    return json.dumps(
        {"frame_version": "frame1", "columns": list(columns), "rows": [list(r) for r in rows]}
    ).encode()


def _standard_row(sv: str, place: str, year: str, value: float, *, unit: object = "USD"):
    return (f"dcid:{sv}", place, year, value, unit, "GrantEquivalent", _PROV, None)


def _flow_row(sv: str, recipient: str, year: str, value: float, *, unit: object = "USD"):
    return (f"dcid:{sv}", "country/FRA", recipient, year, value, unit, None, _PROV, None)


def _body(
    *,
    entities: Mapping[str, Sequence[str]],
    roles: Sequence[str],
    members: Sequence[str],
    places: int,
    series_count: int | None,
    window: str = "all",
    unit_request: Mapping[str, object] | None = None,
    conversions: Mapping | None = None,
    default_scope_plural: bool | None = False,
) -> dict[str, object]:
    """A definition version body. `default_scope_plural=None` leaves the key out, as a body
    minted before it stored the fact does."""
    identity: dict[str, object] = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "measured_property": "amount"},
        "slots": [],
        "entities": dict(entities),
        "requested_window": {"kind": window},
    }
    if unit_request is not None:
        identity["facets"] = {"unit": {"state": "value", "request": dict(unit_request)}}
    trace: dict[str, object] = {
        "resolved_stat_vars": [
            {"ref": f"dcid:{sv}", "roles": list(roles), "observation_count": 1} for sv in members
        ]
    }
    if unit_request is not None:
        trace["conversion"] = [{"recipe": "usd-to-eur"}]
    body: dict[str, object] = {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": _SPEC, "identity": identity},
        "content_version": _CV,
        "prepared_at": "2026-01-01T00:00:00Z",
        "coverage": {
            "rows": 1,
            "dimensions": [{"label": role, "count": places} for role in roles],
        },
        "citation": {"text": "", "sources": []},
        "trace": trace,
        "warnings": [],
        "series_count": series_count,
    }
    if conversions:
        body["series_conversions"] = pack_series_conversions(conversions)
    if default_scope_plural is not None:
        body["default_scope_plural"] = default_scope_plural
    return body


class _World:
    """One stored version, its stored frame and a listable CSV, behind the door on a test app
    whose request scope binds a catalogue listing `_PROV`. `frame` is a frame1 document, stored
    as given or as its canonical frame (`storage`)."""

    storage = "canonical"

    def __init__(
        self,
        body: Mapping[str, object],
        frame: bytes | None,
        *,
        graph: FakeMintGraphReader | None = None,
        reader: FakeSnapshotReader | None = None,
        store: FakeArtifactStore | None = None,
        now: datetime = _NOW,
        base_content_version: str | None = None,
        permits: MaterialisePermits | None = None,
    ) -> None:
        self.permits = permits if permits is not None else MaterialisePermits(1)
        self.store = store if store is not None else FakeArtifactStore()
        self.reader = reader if reader is not None else FakeSnapshotReader()
        self.reader.versions[(_SPEC, _CV)] = StoredVersion(
            subject_id=_SPEC,
            subject_kind="spec",
            content_version=_CV,
            schema_version=1,
            facts=facts_from_body(body),
            base_content_version=base_content_version,
        )
        locations = [_store_and_locate(self.store, b"csv")]
        if frame is not None and self.storage == "canonical":
            locations.append(
                _store_and_locate(
                    self.store,
                    canonical_frame_of_frame1(frame),
                    rendering={"format": "frame"},
                    media_type=CANONICAL_FRAME_MEDIA_TYPE,
                )
            )
        elif frame is not None:
            locations.append(
                _store_and_locate(
                    self.store,
                    frame,
                    rendering={"format": "frame1"},
                    media_type=FRAME1_MEDIA_TYPE,
                )
            )
        self.reader.locations[(_SPEC, _CV)] = locations
        self.graph = graph if graph is not None else _graph()
        self.now = now

    def client(self) -> TestClient:
        door = build_observations_door(
            deps=fake_deps_cell(
                spec_reader=FakeSpecReader(spec_ids=[_SPEC]),
                snapshot_reader=self.reader,
                artifact_store=self.store,
                graph=self.graph,
                code_index=_index(),
                now=lambda: self.now,
                materialise_semaphore=self.permits,
            )
        )
        catalogue = FakeCatalogue(provenances={_PROV: provenance_row(_PROV, DFO, source="OECD")})
        surface = Surface(
            name="datasets",
            prefix="/api/datasets",
            router=door.router,
            capability_version=DATASETS_CAPABILITY_VERSION,
            codes=DATASETS_CODE_CATALOGUE,
            classify=door.classify,
            exception_types=door.exception_types,
            exception_exemplars=door.exception_exemplars,
            request_scope=lambda: bound_catalogue(catalogue),
        )
        return TestClient(create_app([surface]), raise_server_exceptions=False)

    def get(self, path: str = _PATH):
        with self.client() as client:
            return client.get(path)


@pytest.fixture
def small_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A preview cap of two series and a role bulk of two shown series."""
    monkeypatch.setattr(
        search.observations,
        "_SETTINGS",
        ObservationSettings(obs_preview_max_pairs=2, role_bulk_series_max=2),
    )


def _standard_world(**overrides) -> _World:
    rows = [
        _standard_row("Count_Person", "country/KEN", "2020", 50.0),
        _standard_row("Count_Person", "country/KEN", "2021", 52.0),
        _standard_row("Count_Person", "country/GHA", "2021", 30.0),
    ]
    body = _body(
        entities={"observationAbout": ["country/KEN", "country/GHA"]},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=2,
        series_count=2,
    )
    return _World(body, _frame(_COLUMNS_STANDARD, rows), **overrides)


def _series_brief(payload: Mapping) -> list[tuple[str, list[tuple[str, float]]]]:
    return [
        (
            ",".join(entity["ref"] for entity in series["entities"]),
            [(point["date"], point["value"]) for point in series["points"]],
        )
        for series in payload["series"]
    ]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _unregistered(world: _World) -> str:
    return "/api/datasets/spec_other/cv1/observations"


def _unknown_version(world: _World) -> str:
    return f"/api/datasets/{_SPEC}/cv_missing/observations"


def _revoked(world: _World) -> str:
    world.reader.revocations[(_SPEC, _CV)] = Revocation(reason="takedown", revoked_at=_NOW)
    return _PATH


def _expired(world: _World) -> str:
    world.now = _NOW + WORKING_TIER_RETENTION + timedelta(days=1)
    return _PATH


def _only_a_superseded_bundle(world: _World) -> str:
    """The listable CSV is replaced by an unpinned bundle with no rendition, which a snapshot
    does not list, and the stored frame, which a snapshot lists when it is the canonical frame,
    is past working-tier retention, so the version has no listable rendering left."""
    bundle = _store_and_locate(
        world.store,
        b"zip",
        rendering={"format": "csv", "bundle": True},
        media_type="application/zip",
    )
    frame = world.reader.locations[(_SPEC, _CV)][1]
    expired = replace(frame, stored_at=_NOW - WORKING_TIER_RETENTION - timedelta(seconds=1))
    world.reader.locations[(_SPEC, _CV)] = [bundle, expired]
    return _PATH


def _collection(world: _World) -> str:
    return "/api/datasets/col_a/cv1/observations"


def _graph_down(world: _World) -> str:
    world.graph = _graph(fail=True)
    return _PATH


@pytest.mark.parametrize(
    ("arrange", "status", "code"),
    [
        (_unregistered, 404, "spec_not_found"),
        (_unknown_version, 404, "snapshot_not_found"),
        (_revoked, 410, "snapshot_revoked"),
        (_expired, 410, "artifact_expired"),
        (_only_a_superseded_bundle, 410, "artifact_expired"),
        # A retired `col_` id is outside the id grammar, so it matches no route.
        (_collection, 404, "not_found"),
        (_graph_down, 503, "service_unavailable"),
    ],
)
def test_snapshot_and_route_refusals(arrange, status: int, code: str):
    world = _standard_world()
    response = world.get(arrange(world))

    assert response.status_code == status
    assert response.json()["code"] == code


def test_a_graph_failure_is_retryable():
    world = _standard_world(graph=_graph(fail=True))

    assert world.get().json()["retryable"] is True


def test_a_version_without_a_frame1_refuses_observations_unavailable():
    body = _body(
        entities={"observationAbout": ["country/KEN"]},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=1,
        series_count=1,
    )
    response = _World(body, None).get()

    assert (response.status_code, response.json()["code"]) == (404, "observations_unavailable")


def test_a_groupby_collapse_refuses_observations_unavailable():
    world = _standard_world(base_content_version="cv_base")
    response = world.get()

    assert (response.status_code, response.json()["code"]) == (404, "observations_unavailable")
    assert world.store.gets == []


def _converted_body(conversions: Mapping | None, *, places_named: bool = True) -> dict[str, object]:
    return _body(
        entities={"observationAbout": ["country/KEN", "country/GHA"]} if places_named else {},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=2,
        series_count=2,
        unit_request={"currency": "EUR", "prices": "current"},
        conversions=conversions,
        default_scope_plural=not places_named,
    )


_EUR = {"currency": "EUR", "prices": "current"}
_CONVERTED_ROWS = [
    _standard_row("Count_Person", "country/KEN", "2021", 10.0, unit=_EUR),
    _standard_row("Count_Person", "country/GHA", "2021", 20.0, unit=_EUR),
]


def _conversion(fallbacks: int) -> SeriesConversion:
    return SeriesConversion(
        recipe_id="usd-to-eur",
        table_provenance_id=_PROV,
        exchange_rate_table_version="v7",
        deflator_table_version=None,
        fallback_count=fallbacks,
        fallback_examples=(("country/KEN", "Earth"),) if fallbacks else (),
        projected_count=0,
        projected_examples=(),
    )


_KEN_KEY = ("Count_Person", (("observationAbout", "country/KEN"),))
_GHA_KEY = ("Count_Person", (("observationAbout", "country/GHA"),))


@pytest.mark.parametrize(
    ("conversions", "places_named", "expected"),
    [
        # A scope-wide version answers live's converted-bulk omission, which reads no
        # records, so a version that stores none answers it too.
        pytest.param(
            None, False, (200, "observations_converted_scope_unsupported"), id="no-records-bulk"
        ),
        pytest.param(
            {_KEN_KEY: _conversion(1)},
            True,
            (404, "observations_unavailable"),
            id="a-series-without-its-record",
        ),
    ],
)
def test_a_converted_version_refuses_only_for_a_shown_series_without_its_record(
    conversions, places_named, expected
):
    body = _converted_body(conversions, places_named=places_named)
    response = _World(body, _frame(_COLUMNS_STANDARD, _CONVERTED_ROWS)).get()

    payload = response.json()
    code = payload["omitted"]["reason"] if response.status_code == 200 else payload["code"]
    assert (response.status_code, code) == expected


def test_a_series_stored_under_two_facets_fails_as_a_corrupt_version():
    rows = [
        _standard_row("Count_Person", "country/KEN", "2020", 50.0),
        _standard_row("Count_Person", "country/KEN", "2021", 52.0, unit="EUR"),
    ]
    body = _body(
        entities={"observationAbout": ["country/KEN"]},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=1,
        series_count=1,
    )
    response = _World(body, _frame(_COLUMNS_STANDARD, rows)).get()

    assert (response.status_code, response.json()["code"]) == (500, "internal_error")


# ---------------------------------------------------------------------------
# Preview modes
# ---------------------------------------------------------------------------


def test_a_per_series_version_returns_every_series_labelled_and_sourced():
    world = _standard_world()
    response = world.get()

    assert response.status_code == 200
    assert response.headers["server-timing"].startswith("total;")
    payload = response.json()
    assert (payload["total_obs"], payload["truncated"]) == (3, False)
    assert "omitted" not in payload
    assert _series_brief(payload) == [
        ("country/KEN", [("2020", 50.0), ("2021", 52.0)]),
        ("country/GHA", [("2021", 30.0)]),
    ]
    kenya = payload["series"][0]
    assert kenya["stat_var"] == {"ref": "Count_Person", "label": "Population"}
    assert kenya["entities"] == [
        {"role": "observationAbout", "ref": "country/KEN", "label": "Kenya"}
    ]
    assert kenya["total_obs"] == 2
    # The request-scoped catalogue reached the worker thread, so the provenance carries its row.
    assert [(ref["ref"], ref["source"]["ref"]) for ref in kenya["source_refs"]] == [(_PROV, "OECD")]
    assert kenya["facet"]["measurementMethod"] == {
        "ref": "GrantEquivalent",
        "label": "Grant equivalent",
    }
    assert kenya["facet"]["unit"]["ref"] == "USD"


def test_a_scope_defaulted_version_shows_one_latest_point_per_place():
    rows = [
        _standard_row("Count_Person", "country/KEN", "2020", 50.0),
        _standard_row("Count_Person", "country/KEN", "2021", 52.0),
        _standard_row("Count_Person", "country/GHA", "2021", 30.0),
        _standard_row("Count_Person", "country/NOR", "2019", 5.0),
    ]
    body = _body(
        entities={},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=3,
        series_count=3,
        default_scope_plural=True,
    )
    payload = _World(body, _frame(_COLUMNS_STANDARD, rows)).get().json()

    assert _series_brief(payload) == [
        ("country/KEN", [("2021", 52.0)]),
        ("country/GHA", [("2021", 30.0)]),
        ("country/NOR", [("2019", 5.0)]),
    ]
    assert (payload["total_obs"], payload["truncated"]) == (4, True)


def test_a_plural_default_scope_shows_one_latest_point_per_place():
    """Coverage holds one place, as when only one place reported under a latest window. The
    mint's default scope was plural, so the preview reads in bulk as live does."""
    rows = [
        _standard_row("Count_Person", "country/KEN", "2020", 50.0),
        _standard_row("Count_Person", "country/KEN", "2021", 52.0),
    ]
    body = _body(
        entities={},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=1,
        series_count=1,
        default_scope_plural=True,
    )
    payload = _World(body, _frame(_COLUMNS_STANDARD, rows)).get().json()

    assert _series_brief(payload) == [("country/KEN", [("2021", 52.0)])]
    assert (payload["total_obs"], payload["truncated"]) == (2, True)


def test_a_body_that_cannot_say_whether_its_scope_was_plural_refuses():
    body = _body(
        entities={},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=3,
        series_count=1,
        default_scope_plural=None,
    )
    world = _World(body, _frame(_COLUMNS_STANDARD, []))
    response = world.get()

    assert (response.status_code, response.json()["code"]) == (404, "observations_unavailable")
    assert world.store.gets == []


@pytest.mark.parametrize(
    ("entities", "places", "unit_request"),
    [
        ({"observationAbout": ["country/KEN", "country/GHA"]}, 2, None),
        ({}, 3, None),
        # A converted mint with no rows stores no per-series records, and none is missing.
        ({"observationAbout": ["country/KEN", "country/GHA"]}, 2, _EUR),
    ],
    ids=["per-series", "scope-bulk", "converted-per-series"],
)
def test_a_version_whose_frame_holds_no_rows_returns_the_bare_empty_result(
    entities, places, unit_request
):
    body = _body(
        entities=entities,
        roles=["observationAbout"],
        members=["Count_Person"],
        places=places,
        series_count=0,
        unit_request=unit_request,
        default_scope_plural=not entities,
    )
    response = _World(body, _frame(_COLUMNS_STANDARD, [])).get()

    assert response.status_code == 200
    assert response.json() == {"series": [], "total_obs": 0, "truncated": False}


def test_a_role_bulk_version_whose_frame_holds_no_rows_returns_the_bare_empty_result(small_caps):
    body = _body(
        entities={"provider": ["country/FRA", "country/DEU", "country/NOR"]},
        roles=["provider", "recipient"],
        members=[_HEALTH_SV, _EDU_SV],
        places=0,
        series_count=0,
    )
    response = _World(body, _frame(_COLUMNS_FLOW, [])).get()

    assert response.status_code == 200
    assert response.json() == {"series": [], "total_obs": 0, "truncated": False}


_FLOW_POINTS = (
    (_HEALTH_SV, "country/KEN", "2021", 3.0),
    (_HEALTH_SV, "country/GHA", "2021", 9.0),
    (_EDU_SV, "country/KEN", "2020", 1.0),
    (_EDU_SV, "country/KEN", "2021", 2.0),
)


def _flow_rows() -> list[tuple]:
    return [_flow_row(*point) for point in _FLOW_POINTS]


def _flow_body(*, series_count: int) -> dict[str, object]:
    return _body(
        entities={"provider": ["country/FRA"]},
        roles=["provider", "recipient"],
        members=[_HEALTH_SV, _EDU_SV],
        places=0,
        series_count=series_count,
    )


# The flow frame holds four rows. At a gate of four the door ranks every series and shows
# health's two 2021 points, the largest first. Below it the door takes one series from each
# StatVar in turn, both dated 2021 so education leads by dcid, and shows them largest first.
_ROLE_BULK_ORDERS = [
    pytest.param(4, [(_HEALTH_SV, "country/GHA"), (_HEALTH_SV, "country/KEN")], id="ranked"),
    pytest.param(3, [(_HEALTH_SV, "country/GHA"), (_EDU_SV, "country/KEN")], id="fixed"),
]


@pytest.mark.parametrize(("gate", "expected"), _ROLE_BULK_ORDERS)
def test_a_role_bearing_version_over_the_cap_shows_the_series_its_row_gate_selects(
    small_caps, search_settings, gate: int, expected: list[tuple[str, str]]
):
    search_settings(role_bulk_rank_max_rows=gate)
    payload = _World(_flow_body(series_count=3), _frame(_COLUMNS_FLOW, _flow_rows())).get().json()

    assert [
        (series["stat_var"]["ref"], series["entities"][1]["ref"]) for series in payload["series"]
    ] == expected
    assert (payload["total_obs"], payload["truncated"]) == (4, False)
    assert payload["order"] == ("ranked" if gate == 4 else "fixed")


def test_member_labels_compose_from_index_rows_and_graph_arcs():
    payload = _World(_flow_body(series_count=3), _frame(_COLUMNS_FLOW, _flow_rows())).get().json()

    labels = {
        series["stat_var"]["ref"]: series["stat_var"]["label"] for series in payload["series"]
    }
    # The health StatVar's constraint comes from its index row and its value's label from the
    # index's code row. The education StatVar's constraint comes from graph arcs and keeps the
    # graph's label.
    assert labels == {
        _HEALTH_SV: "Basic health",
        _EDU_SV: "Education, level unspecified (CRS Purpose)",
    }


def _wire_series(series: Sequence[Mapping]) -> list[dict[str, object]]:
    """Each series' StatVar, role entities, axes and points: what the StatVar's constraints
    and the label batch decide."""
    return [{key: s[key] for key in ("stat_var", "entities", "axes", "points")} for s in series]


# Under a 2019-2020 window the door's frame holds the three rows inside it. Health's Ghana
# series has its one point in 2021, so the fixed selection passes over it and takes health's
# Kenya series beside education's newer Ghana series.
_WINDOWED_FLOW_POINTS = (
    (_HEALTH_SV, "country/GHA", "2021", 9.0),
    (_HEALTH_SV, "country/KEN", "2019", 3.0),
    (_EDU_SV, "country/GHA", "2020", 2.0),
    (_EDU_SV, "country/KEN", "2020", 1.0),
)

# Under `latest` each StatVar keeps its own latest year: health's two 2021 series and
# education's 2020 Kenya series, three rows of the five. A gate of three ranks them.
_LATEST_FLOW_POINTS = (
    (_HEALTH_SV, "country/GHA", "2021", 9.0),
    (_HEALTH_SV, "country/KEN", "2020", 1.0),
    (_HEALTH_SV, "country/KEN", "2021", 3.0),
    (_EDU_SV, "country/GHA", "2019", 2.0),
    (_EDU_SV, "country/KEN", "2020", 1.0),
)

_PARITY_CASES = [
    *(
        pytest.param(param.values[0], None, _FLOW_POINTS, *param.values[1:], id=param.id)
        for param in _ROLE_BULK_ORDERS
    ),
    pytest.param(
        2,
        TimeWindow(start_year=2019, end_year=2020),
        _WINDOWED_FLOW_POINTS,
        [(_EDU_SV, "country/GHA"), (_HEALTH_SV, "country/KEN")],
        id="fixed-range",
    ),
    pytest.param(
        3,
        "latest",
        _LATEST_FLOW_POINTS,
        [(_HEALTH_SV, "country/GHA"), (_HEALTH_SV, "country/KEN")],
        id="ranked-latest",
    ),
]


def _kept_rows(
    flow_points: Sequence[tuple[str, str, str, float]], window: TimeWindow | str | None
) -> list[tuple]:
    """The frame rows a mint under `window` keeps: the rows inside a range, each StatVar's
    own latest year under `latest`."""
    if isinstance(window, TimeWindow):
        kept = [p for p in flow_points if window.start_year <= int(p[2]) <= window.end_year]
    elif window == "latest":
        latest = {}
        for sv, _recipient, year, _value in flow_points:
            latest[sv] = max(latest.get(sv, year), year)
        kept = [p for p in flow_points if p[2] == latest[p[0]]]
    else:
        kept = list(flow_points)
    return [_flow_row(*point) for point in kept]


@pytest.mark.parametrize(("gate", "window", "flow_points", "expected"), _PARITY_CASES)
def test_the_door_and_the_live_role_bulk_preview_agree_on_each_series_axes(
    small_caps,
    search_settings,
    monkeypatch,
    gate: int,
    window: TimeWindow | str | None,
    flow_points: Sequence[tuple[str, str, str, float]],
    expected: list[tuple[str, str]],
):
    """The live role-bulk read and the door over the rows its mint keeps select and give
    the same series and state the same order, ranked or fixed, on every window. The live
    path takes constraints from member facts (the health index row, the education graph
    arcs); the door from its code index, else graph arcs. Both label through one code
    index, as both read the fixture index in production."""
    search_settings(role_bulk_rank_max_rows=gate)
    monkeypatch.setattr(tests.fixtures, "get_index", _index)
    monkeypatch.setattr(search.engine.region.observations.bulk_role_series, "get_index", _index)
    points: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for sv, recipient, year, value in flow_points:
        points.setdefault((sv, recipient), []).append((year, value))
    coverage = patch_coverage(
        monkeypatch,
        [
            (sv, "F1", "country/FRA", recipient, [int(year) for year, _ in series])
            for (sv, recipient), series in points.items()
        ],
        role_names={
            sv: RoleNames(donor="provider", recipient="recipient") for sv in (_HEALTH_SV, _EDU_SV)
        },
    )
    sdmx_points: dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]] = {}
    for (sv, recipient), series in points.items():
        role_tuple = (("provider", "country/FRA"), ("recipient", recipient))
        sdmx_points.setdefault(sv, {})[role_tuple] = [
            Facet(
                earliest_date=series[0][0],
                latest_date=series[-1][0],
                obs_count=len(series),
                dates=[year for year, _ in series],
                values=[value for _, value in series],
                facet_id="F1",
            )
        ]
    live, warn = build_observations_planned(
        [_HEALTH_SV, _EDU_SV],
        {"provider": ["country/FRA"], "recipient": ["country/GHA", "country/KEN"]},
        window=window,
        graph=FakeGraph(nodes={dcid: {"label": label} for dcid, label in _LABELS.items()}),
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=FakeSdmxClient(observations=sdmx_points),
        coverage=FakeCoverageService(coverage).snapshot(),
        member_facts={
            _HEALTH_SV: IndexMemberFacts(row=_index().statvars[0]),
            _EDU_SV: ArcsMemberFacts(
                arcs={
                    "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}]},
                    "crsPurpose": {"nodes": [{"dcid": "CRSPurposeCode111"}]},
                }
            ),
        },
    )
    door = (
        _World(_flow_body(series_count=3), _frame(_COLUMNS_FLOW, _kept_rows(flow_points, window)))
        .get()
        .json()
    )

    assert _wire_series(observations_body(live)["series"]) == _wire_series(door["series"])
    assert [(s["stat_var"]["ref"], s["entities"][1]["ref"]) for s in door["series"]] == expected
    assert warn is not None and warn.params is not None
    assert live.order == door["order"] == warn.params["order"]
    axes = {
        _HEALTH_SV: {"property": "crsPurpose", "ref": "CRSPurposeCode121", "label": "Basic health"},
        _EDU_SV: {
            "property": "crsPurpose",
            "ref": "CRSPurposeCode111",
            "label": "Education, level unspecified (CRS Purpose)",
        },
    }
    assert [s["axes"] for s in door["series"]] == [[axes[sv]] for sv, _recipient in expected]


def test_a_converted_per_series_version_carries_each_series_conversion():
    conversions = {_KEN_KEY: _conversion(1), _GHA_KEY: _conversion(0)}
    payload = (
        _World(_converted_body(conversions), _frame(_COLUMNS_STANDARD, _CONVERTED_ROWS))
        .get()
        .json()
    )

    by_place = {series["entities"][0]["ref"]: series for series in payload["series"]}
    kenya = by_place["country/KEN"]
    assert kenya["facet"]["unit"] == {"ref": "CurrencyEUR", "label": "Euro"}
    assert kenya["conversion"]["recipe_id"] == "usd-to-eur"
    assert kenya["conversion"]["exchange_rate_table_version"] == "v7"
    assert kenya["conversion"]["table_provenance"]["ref"] == _PROV
    assert (
        kenya["conversion"]["fallback_count"],
        by_place["country/GHA"]["conversion"]["fallback_count"],
    ) == (1, 0)
    assert kenya["conversion"]["fallback_examples"] == [
        {"entity": "country/KEN", "fallback_entity": "Earth"}
    ]


def _too_large() -> tuple[dict[str, object], str]:
    body = _body(
        entities={"observationAbout": ["country/KEN", "country/GHA", "country/NOR"]},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=3,
        series_count=3,
    )
    return body, "observations_preview_too_large"


def _windowed_scope() -> tuple[dict[str, object], str]:
    body = _body(
        entities={},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=3,
        series_count=3,
        window="range",
        default_scope_plural=True,
    )
    return body, "observations_windowed_scope_unsupported"


def _converted_scope() -> tuple[dict[str, object], str]:
    body = _body(
        entities={},
        roles=["observationAbout"],
        members=["Count_Person"],
        places=3,
        series_count=3,
        unit_request=_EUR,
        conversions={_KEN_KEY: _conversion(0)},
        default_scope_plural=True,
    )
    return body, "observations_converted_scope_unsupported"


@pytest.mark.parametrize(
    "frame", [_frame(_COLUMNS_STANDARD, []), None], ids=["frame1", "no-frame1"]
)
@pytest.mark.parametrize("arrange", [_too_large, _windowed_scope, _converted_scope])
def test_an_omitted_preview_reads_no_frame_bytes(small_caps, arrange, frame):
    """A version too large for a frame1 answers live's omission too."""
    body, reason = arrange()
    world = _World(body, frame)
    response = world.get()

    assert response.status_code == 200
    payload = response.json()
    assert (payload["series"], payload["omitted"]["reason"]) == ([], reason)
    assert world.store.gets == []


def test_the_too_large_omission_counts_the_stored_series(small_caps):
    body, _reason = _too_large()

    omitted = _World(body, _frame(_COLUMNS_STANDARD, [])).get().json()["omitted"]

    assert (omitted["pairs"], omitted["cap"]) == (3, 2)


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_an_observations_read_frees_its_frame_before_its_permit_is_released():
    """The read loads and decodes the stored frame under one permit. When it gives the permit
    back, nothing references the frame's bytes."""
    permits = _FrameTrackingPermits(1)

    class _TrackingStore(FakeArtifactStore):
        def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
            data = super().get(artifact_digest=artifact_digest, tier=tier)
            # A `bytearray` stands in for the bytes, so the test can hold a weak reference.
            return cast(bytes, permits.track(data))

    resp = _standard_world(store=_TrackingStore(), permits=permits).get()

    assert resp.status_code == 200, resp.text
    assert len(permits.read) == 1
    assert permits.freed_at_release == [True]


def test_a_version_pinned_then_read_31_days_later_returns_its_observations():
    store = FakeArtifactStore()
    reader = StoreBackedSnapshotReader(store=store)
    world = _standard_world(reader=reader, store=store)
    csv = reader.locations[(_SPEC, _CV)][0]
    pin_artifact(
        id=_SPEC,
        content_version=_CV,
        artifact_digest=csv.artifact_digest,
        principal="first-party-key",
        spec_reader=FakeSpecReader(spec_ids=[_SPEC]),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=FakeArtifactReferenceWriter(),
        daily_max=1_000_000,
        now=lambda: _NOW,
    )
    world.now = _NOW + timedelta(days=31)

    response = world.get()

    assert response.status_code == 200
    assert len(response.json()["series"]) == 2
    assert store.gets[-1][1] is Tier.pinned
