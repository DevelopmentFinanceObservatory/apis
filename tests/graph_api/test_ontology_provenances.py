"""`build_provenances` over a small code index and coverage, and the route that serves it."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from code_index import CodeIndex, DatasetDeclaration, IndexStamp, StatVarRow
from code_index.shared import get_index
from graph_api.doors import ontology as ontology_door
from graph_api.ontology_provenances import SHAPE_FIELDS, build_provenances
from series_coverage.schema import FacetMetadata
from statvar_members import FiveTuple

from .conftest import FakeGraphNodeReader, make_client, wait_until

_CRS = "provenance/CRS"
_DAC = "provenance/DAC1"
_ROUTE = "/api/graph/ontology/provenances"
_IDENTITY = {"Accept-Encoding": "identity"}

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _row(
    dcid: str,
    constraints: Mapping[str, str],
    *,
    provenances: tuple[str, ...] = (_CRS,),
    observation_properties: tuple[str, ...] = ("financingRecipient",),
) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=observation_properties,
        constraint_properties=constraints,
        provenances=provenances,
    )


_ROWS = (
    _row("Amount_B", {"crsPurpose": "CRSPurposeCode120", "flowType": "ODA"}),
    _row("Amount_A", {"flowType": "ODA", "crsPurpose": "CRSPurposeCode12220"}),
    _row(
        "Amount_C",
        {"crsPurpose": "CRSPurposeCode120", "flowType": "OOF"},
        observation_properties=("financingRecipient", "financingProvider"),
    ),
    _row("Amount_Shared", {"flowType": "ODA"}, provenances=(_CRS, _DAC)),
)

_DECLARATION = DatasetDeclaration(
    provenance=_CRS,
    label="Creditor Reporting System",
    description="Aid by project.",
    opt_in=False,
    absent_observation="zero",
    account="provider",
    rank=0,
)


def _index(*rows: StatVarRow) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP, declarations=(), codes=(), statvars=rows, datasets=(_DECLARATION,)
    )


@dataclass
class _FakeCoverageIndex:
    facet_metadata: Mapping[str, FacetMetadata] = field(default_factory=dict)
    provenance_currencies: Mapping[str, frozenset[str]] = field(default_factory=dict)


_COVERAGE = _FakeCoverageIndex(
    facet_metadata={
        "f2": FacetMetadata(
            provenance_id=_CRS,
            unit="CurrencyUSD",
            scaling_factor="1000000",
            observation_period="P1Y",
        ),
        "f1": FacetMetadata(
            provenance_id=_CRS, unit="CurrencyEUR", measurement_method="Commitment"
        ),
        "f3": FacetMetadata(provenance_id=_CRS, unit="CurrencyUSD", observation_period="P1Y"),
        "f4": FacetMetadata(provenance_id="provenance/Other", unit="Percent"),
        "f5": FacetMetadata(unit="Count"),
    },
    provenance_currencies={_CRS: frozenset({"USD", "EUR"}), "provenance/Other": frozenset()},
)


def _built(index: CodeIndex | None = None, coverage=_COVERAGE) -> dict[str, dict]:
    if index is None:
        index = _index(*_ROWS)
    built = build_provenances(code_index=index, coverage=coverage).model_dump(mode="json")
    return {entry["dcid"]: entry for entry in built["provenances"]}


def test_one_entry_per_provenance_with_its_declaration() -> None:
    built = _built()

    assert list(built) == [_CRS, _DAC]
    assert built[_CRS]["label"] == "Creditor Reporting System"
    assert built[_CRS]["description"] == "Aid by project."
    assert built[_DAC]["label"] is None
    assert built[_DAC]["description"] is None


def test_a_statvar_with_two_provenances_counts_under_both() -> None:
    built = _built()

    assert built[_CRS]["stat_var_count"] == 4
    assert built[_DAC]["stat_var_count"] == 1
    assert [p["examples"] for p in built[_DAC]["patterns"]] == [["Amount_Shared"]]


def test_patterns_group_rows_and_split_on_observation_properties() -> None:
    patterns = _built()[_CRS]["patterns"]

    assert [
        (
            p["constraint_properties"],
            p["observation_properties"],
            p["stat_var_count"],
            p["examples"],
        )
        for p in patterns
    ] == [
        (["crsPurpose", "flowType"], ["financingRecipient"], 2, ["Amount_A", "Amount_B"]),
        (
            ["crsPurpose", "flowType"],
            ["financingProvider", "financingRecipient"],
            1,
            ["Amount_C"],
        ),
        (["flowType"], ["financingRecipient"], 1, ["Amount_Shared"]),
    ]
    assert patterns[0]["population_type"] == "FinancialFlow"
    assert patterns[0]["measured_property"] == "amount"
    assert patterns[0]["stat_type"] == "measuredValue"
    assert patterns[0]["measurement_qualifier"] is None


def test_examples_are_the_first_three_dcids_in_sorted_order() -> None:
    rows = [_row(f"Amount_{n}", {"flowType": "ODA"}) for n in "edcba"]

    (pattern,) = _built(_index(*rows))[_CRS]["patterns"]

    assert pattern["stat_var_count"] == 5
    assert pattern["examples"] == ["Amount_a", "Amount_b", "Amount_c"]


def test_constraint_values_count_statvars_per_value() -> None:
    assert _built()[_CRS]["constraint_values"] == [
        {
            "property": "crsPurpose",
            "values": [
                {"value": "CRSPurposeCode120", "stat_var_count": 2},
                {"value": "CRSPurposeCode12220", "stat_var_count": 1},
            ],
        },
        {
            "property": "flowType",
            "values": [
                {"value": "ODA", "stat_var_count": 3},
                {"value": "OOF", "stat_var_count": 1},
            ],
        },
    ]


def test_facets_hold_only_the_provenances_own_facets() -> None:
    assert _built()[_CRS]["facets"] == {
        "units": ["CurrencyEUR", "CurrencyUSD"],
        "scaling_factors": ["1000000"],
        "measurement_methods": ["Commitment"],
        "observation_periods": ["P1Y"],
        "currencies": ["EUR", "USD"],
    }


def test_a_provenance_with_no_coverage_gets_empty_facets() -> None:
    assert _built()[_DAC]["facets"] == {
        "units": [],
        "scaling_factors": [],
        "measurement_methods": [],
        "observation_periods": [],
        "currencies": [],
    }


def test_reordered_inputs_give_the_same_body() -> None:
    reordered = _FakeCoverageIndex(
        facet_metadata=dict(reversed(list(_COVERAGE.facet_metadata.items()))),
        provenance_currencies=_COVERAGE.provenance_currencies,
    )

    assert _built(_index(*reversed(_ROWS)), reordered) == _built()


# The route


@dataclass
class _Manifest:
    build_id: str


@dataclass
class _Snapshot:
    index: _FakeCoverageIndex
    manifest: _Manifest


class _FakeCoverage:
    """Stands in for `ServedCoverage`; the route reads `snapshot()`. Set `build` to swap it."""

    def __init__(self, build: _Snapshot) -> None:
        self.build = build

    def snapshot(self) -> _Snapshot:
        return self.build


def _coverage_of(unit: str) -> _Snapshot:
    """A coverage build of its own id, since the route's memo outlives one test."""
    provenance = next(iter(get_index().variables_by_provenance))
    return _Snapshot(
        index=_FakeCoverageIndex(
            facet_metadata={"f1": FacetMetadata(provenance_id=provenance, unit=unit)}
        ),
        manifest=_Manifest(build_id=str(uuid.uuid4())),
    )


def _units(response) -> list[list[str]]:
    return [entry["facets"]["units"] for entry in response.json()["provenances"]]


def test_the_route_answers_from_the_indexes_with_an_etag_and_no_graph_read() -> None:
    graph = FakeGraphNodeReader()
    coverage = _FakeCoverage(_coverage_of("CurrencyUSD"))

    with make_client(graph=graph, coverage=coverage) as client:
        first = client.get(_ROUTE, headers=_IDENTITY)
        etag = first.headers["ETag"]
        again = client.get(_ROUTE, headers={**_IDENTITY, "If-None-Match": etag})
        gzipped = client.get(_ROUTE, headers={"Accept-Encoding": "gzip"})

    assert first.status_code == 200, first.text
    assert first.headers["Cache-Control"].startswith("public, max-age=")
    assert again.status_code == 304
    assert again.headers["ETag"] == etag
    assert gzipped.headers["Content-Encoding"] == "gzip"
    assert gzipped.json() == first.json()
    assert ["CurrencyUSD"] in _units(first)
    assert set(graph.calls.values()) == {0}


def test_the_body_is_built_once_per_coverage_build(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[object] = []
    real = ontology_door.build_provenances

    def counted(**kwargs):
        builds.append(kwargs["coverage"])
        return real(**kwargs)

    monkeypatch.setattr(ontology_door, "build_provenances", counted)
    coverage = _FakeCoverage(_coverage_of("CurrencyUSD"))

    with make_client(coverage=coverage) as client:
        first = client.get(_ROUTE, headers=_IDENTITY)
        repeat = client.get(_ROUTE, headers=_IDENTITY)
        coverage.build = _coverage_of("CurrencyEUR")
        changed = client.get(_ROUTE, headers=_IDENTITY)

    assert len(builds) == 2
    assert repeat.content == first.content
    assert ["CurrencyUSD"] in _units(first)
    assert ["CurrencyEUR"] in _units(changed)
    assert changed.headers["ETag"] != first.headers["ETag"]


def test_a_miss_builds_the_body_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    on_loop: list[bool] = []
    real = ontology_door.build_provenances

    def recorded(**kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop.append(False)
        else:
            on_loop.append(True)
        return real(**kwargs)

    monkeypatch.setattr(ontology_door, "build_provenances", recorded)

    with make_client(coverage=_FakeCoverage(_coverage_of("CurrencyUSD"))) as client:
        response = client.get(_ROUTE, headers=_IDENTITY)

    assert response.status_code == 200, response.text
    assert on_loop == [False]


def test_concurrent_first_requests_share_one_build(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    builds: list[object] = []
    real = ontology_door.build_provenances

    def blocked(**kwargs):
        builds.append(kwargs["coverage"])
        started.set()
        release.wait(timeout=5.0)
        return real(**kwargs)

    monkeypatch.setattr(ontology_door, "build_provenances", blocked)
    responses: list[int] = []

    with make_client(coverage=_FakeCoverage(_coverage_of("CurrencyUSD"))) as client:
        cache = client.app.state.graph_deps.ontology_provenances

        def send() -> None:
            responses.append(client.get(_ROUTE, headers=_IDENTITY).status_code)

        first = threading.Thread(target=send)
        first.start()
        assert started.wait(timeout=5.0), "the build never started"
        second = threading.Thread(target=send)
        second.start()
        wait_until(lambda: dict(cache.stats())["misses"] == 2)
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)

    assert responses == [200, 200]
    assert len(builds) == 1


def test_a_build_slower_than_the_request_answers_503_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    real = ontology_door.build_provenances

    def blocked(**kwargs):
        release.wait(timeout=5.0)
        return real(**kwargs)

    monkeypatch.setattr(ontology_door, "build_provenances", blocked)
    monkeypatch.setattr(ontology_door, "GRAPH_NODE_TIMEOUT_SECONDS", 0.2)

    with make_client(coverage=_FakeCoverage(_coverage_of("CurrencyUSD"))) as client:
        pending = client.get(_ROUTE, headers=_IDENTITY)
        release.set()
        cache = client.app.state.graph_deps.ontology_provenances
        wait_until(lambda: dict(cache.stats())["entries"] == 1)
        landed = client.get(_ROUTE, headers=_IDENTITY)

    assert pending.status_code == 503, pending.text
    assert pending.json()["code"] == "service_unavailable"
    assert pending.json()["retryable"] is True
    assert pending.headers["Retry-After"] == "5"
    assert landed.status_code == 200, landed.text


def test_shape_fields_are_five_tuple_fields_and_statvar_row_attributes() -> None:
    five_tuple = {field.name for field in dataclasses.fields(FiveTuple)}
    row_fields = {field.name for field in dataclasses.fields(StatVarRow)}

    assert set(SHAPE_FIELDS) <= five_tuple
    assert set(SHAPE_FIELDS) <= row_fields
