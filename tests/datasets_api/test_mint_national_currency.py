"""Tests a unit bound to each place's national currency through `POST /api/datasets` (#1273).

The mint door, the identity core and materialisation are real; the graph and SDMX serve the
recorded WEO nominal GDP fixture (`tests.materialise._weo_recorded`).
"""

from __future__ import annotations

from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise._weo_recorded import (
    FRANCE,
    GDP,
    KENYA,
    WEO,
    recorded_currencies,
    recorded_facets,
    recorded_values,
    sdmx,
)

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_UNNAMED = "country/XNA"
"""A constructed place no currency names, holding France's facets."""

_NATIONAL_CURRENCY = {"state": "value", "national_currency": {"role": "observationAbout"}}


def _client():
    facets_by_place = recorded_facets()
    facets_by_place[_UNNAMED] = list(facets_by_place[FRANCE])
    values = recorded_values()
    for candidate in facets_by_place[FRANCE]:
        assert candidate.facet_id is not None
        values[(_UNNAMED, candidate.facet_id)] = dict(values[(FRANCE, candidate.facet_id)])
    named: dict[str, list[str]] = {}
    for place, units in recorded_currencies().items():
        for unit in units:
            named.setdefault(unit, []).append(place)
    arcs = {
        GDP: {"typeOf": ("StatisticalVariable",), "measuredProperty": ("amount",)},
        **{
            unit: {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": tuple(places)}
            for unit, places in named.items()
        },
    }
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=FakeMintGraphReader(
            facets={(GDP, place): facets for place, facets in facets_by_place.items()},
            arcs=arcs,
        ),
        sdmx=sdmx(facets_by_place, values),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=FakeCoverageService(build_coverage_index([], role_names={})),
    )


def _body(places: list[str], facets: dict) -> dict:
    return {
        "schema_version": 1,
        "input": {
            "kind": "statvar_set",
            "statvars": [f"dcid:{GDP}"],
            "scope": {
                "entities": {"observationAbout": places},
                "window": {"kind": "range", "start_year": 2016, "end_year": 2024},
                "facets": facets,
            },
        },
        "options": {"bundle": False},
    }


_SERVED = {"unit": {"national_currency": {"role": "observationAbout"}}}


def test_a_mint_serves_each_place_its_own_currency_and_names_the_place_it_leaves_out():
    """The wire form mints the kit's `national_currency` identity, each served place keeps
    its own currency from one provenance, nothing is converted, and the trace names the place
    no currency names."""
    response = _client().post(
        "/api/datasets",
        json=_body([FRANCE, KENYA, _UNNAMED], _SERVED),
    )

    assert response.status_code == 200, response.text
    manifest = response.json()
    identity = manifest["definition"]["identity"]
    assert identity["facets"]["unit"] == _NATIONAL_CURRENCY
    assert manifest["definition"]["spec_id"] == recompute_identity(identity).spec_id
    trace = manifest["trace"]
    assert "conversion" not in trace
    assert trace["coverage_gaps"] == [
        {"role": "observationAbout", "reason": "no_national_currency", "entities": [_UNNAMED]}
    ]
    assert [entry["ref"] for entry in trace["facets"] if entry["axis"] == "provenance"] == [WEO]
