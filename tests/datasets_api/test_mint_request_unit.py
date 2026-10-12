"""Tests request-form currency conversion through `POST /api/datasets`, including
identity generation, conversion, and invalid unit requests. The mint door
and materialisation are real; graph and SDMX are faked. The fixture uses the
DAC recipe and the `recipient` role (`destination` flow axis).
"""

from __future__ import annotations

import csv
import io

import httpx
import pyarrow.parquet as pq
import pytest

from dc_sdmx import SdmxClient
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._identity_builders import pinned_identity
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"

_MEMBER_PROVENANCE = "OECD_DAC2A"
_DAC_RECIPE = "ConversionRecipeDAC"

_ROLE_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _role_row(*, unit: str, value: str, year: str = "2015") -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},{unit},GrantEquivalent,"
        f"NotApplicable,{_MEMBER_PROVENANCE},{year},{value},,facet1\r\n"
    )


def _currency_arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        _STAT_VAR: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
            **CONVERTIBLE_STAT_VAR_ARCS,
        },
        _MEMBER_PROVENANCE: {"conversionRecipe": (_DAC_RECIPE,), "isPartOf": ()},
        **conversion_arcs(
            recipe_id=_DAC_RECIPE,
            provenance_id=_MEMBER_PROVENANCE,
            currencies={"CurrencyUSD": [], "CurrencyEUR": [], "CurrencyKES": [_RECIPIENT]},
        ),
    }


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs=_currency_arcs(), labels={_MEMBER_PROVENANCE: "OECD DAC2a", _RECIPIENT: "Kenya"}
    )


def _sdmx(*, native_unit: str = "CurrencyUSD", eur_rate: float = 0.9) -> SdmxClient:
    role_body = _ROLE_HEADER + _role_row(unit=native_unit, value="1000")

    def observations(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=role_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    server = RateTableServer(
        recipe_table_bodies(
            _DAC_RECIPE,
            rates={
                "CurrencyUSD": {2015: 1.0},
                "CurrencyEUR": {2015: eur_rate},
                "CurrencyKES": {2015: 112.0},
            },
        ),
        fallback=observations,
    )
    return server.client(base_url=_BASE_URL)


def _coverage_service(*, native_unit: str = "CurrencyUSD") -> FakeCoverageService:
    """Builds one 2015 coverage facet for `_STAT_VAR` and `native_unit`."""
    index = build_coverage_index(
        [(_STAT_VAR, "facet1", ABSENT_ENTITY, _RECIPIENT, [2015])],
        role_names={_STAT_VAR: RoleNames(donor=None, recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id=_MEMBER_PROVENANCE,
                measurement_method="GrantEquivalent",
                unit=native_unit,
            )
        },
    )
    return FakeCoverageService(index)


def _client(
    *,
    store=None,
    sdmx: SdmxClient | None = None,
    graph=None,
    coverage: FakeCoverageService | None = None,
    artifact_store: FakeArtifactStore | None = None,
):
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph if graph is not None else _graph(),
        sdmx=sdmx if sdmx is not None else _sdmx(),
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage if coverage is not None else _coverage_service(),
    )


def _first_value(manifest: dict, artifact_store: FakeArtifactStore) -> float:
    """Reads `first_value` from the manifest's `series-statistics/1` Parquet artifact."""
    statistics_artifact = next(
        a for a in manifest["artifacts"] if a["rendering"] == {"statistics": "series"}
    )
    table = pq.read_table(io.BytesIO(artifact_store.fetch(statistics_artifact["url"])))
    return table.to_pylist()[0]["first_value"]


def _statvar_set_body(*, unit: dict) -> dict:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {"entities": {"recipient": [_RECIPIENT]}, "facets": {"unit": unit}},
        },
        "options": {"bundle": False},
    }


def _spec_id_body(spec_id: str, *, scope: dict) -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": spec_id, "scope": scope},
        "options": {"bundle": False},
    }


def test_currency_string_request_on_a_dollar_facet_mints_and_converts():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _statvar_set_body(unit={"currency": "EUR", "prices": "current"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    identity = manifest["definition"]["identity"]
    assert identity["facets"]["unit"] == {
        "state": "value",
        "request": {"currency": "EUR", "prices": "current"},
    }
    # A mint through this door and a spec built directly over the
    # identity-core document it ought to produce hash to the same spec_id.
    # `facets_to_identity` matches the pinned spec-kit's own
    # canonicalisation of the request form.
    expected_identity = pinned_identity(
        stat_var=_STAT_VAR,
        entities={"recipient": [_RECIPIENT]},
        facets={"unit": {"state": "value", "request": {"currency": "EUR", "prices": "current"}}},
    )
    assert manifest["definition"]["spec_id"] == recompute_identity(expected_identity).spec_id
    # The conversion ran: USD 1000 -> EUR at 0.9.
    assert _first_value(manifest, artifact_store) == 900.0


_UNIT_COLUMNS = ("unit", "unit_label", "currency", "price_basis", "base_year")


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        pytest.param(
            {"currency": "EUR", "prices": "current"},
            ("CurrencyEUR", "Current euros", "EUR", "current", ""),
            id="iso-currency",
        ),
        # A role-currency request converts each row to its entity's own currency, which the
        # frame's unit cell names (`country/KEN` uses `CurrencyKES`).
        pytest.param(
            {"currency": {"role": "destination"}, "prices": "current"},
            ("CurrencyKES", "Current Kenyan shillings", "KES", "current", ""),
            id="role-currency",
        ),
    ],
)
def test_a_converted_csv_artifact_splits_the_unit_cell(unit, expected):
    """The frame's converted `unit` cell is written as the unit node, its label, the currency,
    the price basis and the base year."""
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _statvar_set_body(unit=unit)
    body["options"] = {"format": "csv", "bundle": False}

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    csv_artifact = next(a for a in resp.json()["artifacts"] if a["media_type"] == "text/csv")
    rows = list(csv.DictReader(io.StringIO(artifact_store.fetch(csv_artifact["url"]).decode())))
    assert [tuple(row[column] for column in _UNIT_COLUMNS) for row in rows] == [expected]


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        pytest.param(
            {"currency": "EUR", "prices": "current"},
            '{"currency":"EUR","prices":"current"}',
            id="iso-currency",
        ),
        pytest.param(
            {"currency": {"role": "destination"}, "prices": "current"},
            '{"currency":"KES","prices":"current"}',
            id="role-currency",
        ),
    ],
)
def test_a_frame_layout_artifact_carries_the_unit_cell_as_canonical_json(unit, expected):
    """A frame-layout file keeps the frame's columns, `status` included, and writes the
    converted `unit` cell as the frame grammar's JCS serialisation of the unit request."""
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _statvar_set_body(unit=unit)
    body["options"] = {"format": "parquet", "bundle": False, "layout": "frame"}

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    (artifact,) = (a for a in resp.json()["artifacts"] if a["rendering"] == body["options"])
    table = pq.read_table(io.BytesIO(artifact_store.fetch(artifact["url"])))
    assert table.column_names == [
        "stat_var",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    ]
    assert table.column("unit").to_pylist() == [expected]


def test_role_shaped_currency_naming_a_role_the_spec_carries_mints():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    body = _statvar_set_body(unit={"currency": {"role": "destination"}, "prices": "current"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["identity"]["facets"]["unit"] == {
        "state": "value",
        "request": {"currency": {"role": "destination"}, "prices": "current"},
    }
    # `recipient` (country/KEN) uses CurrencyKES, rate 112.0 against the
    # USD anchor of 1.0: USD 1000 -> KES 112000.
    assert _first_value(manifest, artifact_store) == 112000.0


def test_a_role_the_spec_lacks_refuses_request_invalid():
    # `_STAT_VAR` declares only `recipient` (the `destination` axis).
    # `origin` names no role this spec carries.
    client = _client()
    body = _statvar_set_body(unit={"currency": {"role": "origin"}, "prices": "current"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["alternatives"] == [{"ref": "recipient", "label": "recipient"}]


def test_a_constant_price_request_without_base_year_refuses_request_invalid():
    client = _client()
    body = _statvar_set_body(unit={"currency": "EUR", "prices": "constant"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/scope/facets/unit/base_year"


def test_a_current_price_request_with_a_base_year_refuses_request_invalid():
    client = _client()
    body = _statvar_set_body(unit={"currency": "EUR", "prices": "current", "base_year": 2015})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/scope/facets/unit/base_year"


def test_a_lowercase_currency_code_refuses_request_invalid():
    client = _client()
    body = _statvar_set_body(unit={"currency": "eur", "prices": "current"})

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"].startswith("/input/scope/facets/unit/currency")


def test_the_request_form_on_a_non_unit_axis_refuses_request_invalid():
    client = _client()
    body = {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [_STAT_VAR_REF],
            "scope": {"facets": {"provenance": {"currency": "EUR", "prices": "current"}}},
        },
        "options": {"bundle": False},
    }

    resp = client.post("/api/datasets", json=body)

    assert resp.status_code == 400, resp.text
    problem = resp.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/scope"
    assert "provenance" in problem["detail"]


def test_scope_surgery_with_a_request_form_unit_round_trips():
    artifact_store = FakeArtifactStore()
    client = _client(artifact_store=artifact_store)
    base = client.post(
        "/api/datasets", json=_statvar_set_body(unit={"currency": "USD", "prices": "current"})
    )
    assert base.status_code == 200, base.text
    base_spec_id = base.json()["definition"]["spec_id"]

    resp = client.post(
        "/api/datasets",
        json=_spec_id_body(
            base_spec_id, scope={"facets": {"unit": {"currency": "EUR", "prices": "current"}}}
        ),
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["definition"]["spec_id"] != base_spec_id
    assert manifest["definition"]["identity"]["facets"]["unit"] == {
        "state": "value",
        "request": {"currency": "EUR", "prices": "current"},
    }
    recomputed = recompute_identity(manifest["definition"]["identity"])
    assert recomputed.spec_id == manifest["definition"]["spec_id"]
    assert _first_value(manifest, artifact_store) == 900.0
