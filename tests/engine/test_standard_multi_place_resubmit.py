"""Resubmission preserves posted place sets, reads each posted place at its own dcid, and
holds the set to the bound on the places a Spec binds.

A posted region is read at its node. A set over the bound is refused whole: no posted place
is dropped to fit.
"""

from __future__ import annotations

import re

import pytest

from search.engine.errors import EngineInputError
from search.engine.region import resolve_spec_resubmit
from search.models import (
    BindingSet,
    BindingValue,
    CoverageExact,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    make_identity,
    subject_entity,
)
from tests.engine._builders import gref as _gref
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph, fixture_provenance_names
from tests.origin_catalogue.fake_catalogue import listing

pytestmark = pytest.mark.usefixtures("request_log")

# Count_Person has no constraintProperties. These tests exercise the where axis.
_SV = "Count_Person"

_NODES: dict = {
    _SV: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/GTM": {"label": "Guatemala", "type": "Country"},
    "country/SLV": {"label": "El Salvador", "type": "Country"},
    "country/HND": {"label": "Honduras", "type": "Country"},
    # Boreal is a Continent with two member countries.
    "boreal": {"label": "Boreal", "type": "Continent"},
    "country/B1": {"label": "B1", "type": "Country"},
    "country/B2": {"label": "B2", "type": "Country"},
}
_CHILDREN: dict = {
    "boreal": {"Country": ["country/B1", "country/B2"]},
}


# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_OBS_PROVENANCE = "dc/base/WorldDevelopmentIndicators"


@pytest.fixture(autouse=True)
def _listed_catalogue(bind_catalogue) -> None:
    bind_catalogue(listing(fixture_provenance_names()))


def _where_value_slot(dcid: str, label: str) -> Slot:
    return Slot(
        key=SlotKey(axis="where", property=None, label="place"),
        binding=BindingValue(value=SlotValue(ref=_gref(dcid, label), value_kind="entity")),
    )


def _where_set_slot(pairs: list[tuple[str, str]]) -> Slot:
    return Slot(
        key=SlotKey(axis="where", property=None, label="place"),
        binding=BindingSet(
            values=[SlotValue(ref=_gref(dcid, label), value_kind="entity") for dcid, label in pairs]
        ),
    )


def _graph(obs: dict) -> FakeGraph:
    return FakeGraph(nodes=_NODES, obs=obs, children=_CHILDREN)


def _where_slot(region):
    return next(s for s in region.specs[0].resolved.slots if s.key.axis == "where")


def _resubmit(where_slot: Slot, obs: dict):
    binding = where_slot.binding
    values = [binding.value] if isinstance(binding, BindingValue) else binding.values
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            PERSON_COUNT_SHAPE,
            entities=[subject_entity(value.ref.ref, value.ref.label) for value in values],
        ),
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=_graph(obs),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(_graph(obs)),
    )


# Posted place sets retain their values and roles.


def test_posted_two_place_set_survives_as_binding_set_with_both_roles():
    """A posted 2-place set retains both entities with observationAbout roles."""
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
        f"{_SV}|country/SLV": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("country/SLV", "El Salvador")])
    region = _resubmit(where_slot, obs)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    where = _where_slot(region)
    assert isinstance(where.binding, BindingSet)
    assert [v.ref.ref for v in where.binding.values] == ["country/GTM", "country/SLV"]
    assert {e.ref for e in spec.resolved.entities} == {"country/GTM", "country/SLV"}


def test_posted_set_whose_data_sits_at_its_second_place_answers():
    """Guatemala holds no data and El Salvador does: the probe reads both places of the
    posted set, so the resubmit answers with El Salvador's point."""
    obs = {
        f"{_SV}|country/SLV": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("country/SLV", "El Salvador")])
    region = _resubmit(where_slot, obs)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    coverage = region.specs[0].resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert coverage.has_data is True
    assert coverage.observation_count == 1


# A posted region is read at its own node.


def _held_at(place: str) -> dict:
    """One observation of the series, held at `place`."""
    return {
        f"{_SV}|{place}": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }


@pytest.mark.parametrize(
    "where_slot",
    [
        _where_value_slot("boreal", "Boreal"),
        _where_set_slot([("country/GTM", "Guatemala"), ("boreal", "Boreal")]),
    ],
    ids=["region-alone", "region-beside-a-country"],
)
def test_posted_region_whose_members_alone_hold_data_has_no_observations(where_slot):
    """Boreal's node holds nothing and its member B1 holds the series. The resubmit reads
    the posted dcids only, so it finds no observation."""
    region = _resubmit(where_slot, _held_at("country/B1"))

    assert (region.status, region.no_data_reason) == ("no_data", "no_observations")


@pytest.mark.parametrize(
    ("where_slot", "bound"),
    [
        (_where_value_slot("boreal", "Boreal"), ["boreal"]),
        (
            _where_set_slot([("country/GTM", "Guatemala"), ("boreal", "Boreal")]),
            # Canonical identity order puts boreal before country/GTM.
            ["boreal", "country/GTM"],
        ),
    ],
    ids=["region-alone", "region-beside-a-country"],
)
def test_posted_region_is_read_at_its_node(where_slot, bound):
    """Boreal's node holds the series: the resubmit answers with the node's observation
    and keeps the posted places as its where binding."""
    region = _resubmit(where_slot, _held_at("boreal"))

    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    binding = _where_slot(region).binding
    values = [binding.value] if isinstance(binding, BindingValue) else binding.values
    assert [value.ref.ref for value in values] == bound
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.observation_count == 1


# A posted where set takes precedence over a base entity.


def test_unresolvable_base_entity_alongside_valid_where_set_is_inert():
    """The posted where set supplies the entity anchor when the base contains country/ZZZ.

    Resolution uses the set's first member.
    """
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
        f"{_SV}|country/SLV": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("country/SLV", "El Salvador")])
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            PERSON_COUNT_SHAPE,
            entities=[
                subject_entity(value.ref.ref, value.ref.label)
                for value in where_slot.binding.values
            ],
        ),
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=_graph(obs),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(_graph(obs)),
    )
    assert region.status == "definite"
    where = _where_slot(region)
    assert isinstance(where.binding, BindingSet)
    assert {v.ref.ref for v in where.binding.values} == {"country/GTM", "country/SLV"}


# The bound on the places a Spec binds holds on resubmit.


def _numbers_named(exc: EngineInputError) -> set[int]:
    """The numbers the refusal's message names."""
    return {int(number) for number in re.findall(r"\d+", str(exc))}


def test_posted_places_over_the_place_bound_refuse(search_settings):
    """A role over the bound returns request_invalid at that role, naming the count and
    the bound."""
    search_settings(max_child_places=2)
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot(
        [
            ("country/GTM", "Guatemala"),
            ("country/SLV", "El Salvador"),
            ("country/HND", "Honduras"),
        ]
    )
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(where_slot, obs)
    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.status == 400
    assert _numbers_named(exc_info.value) == {3, 2}
    assert exc_info.value.pointer == "/input/identity/entities/observationAbout"


def test_posted_region_counts_as_one_place_against_the_bound(search_settings):
    """A posted region binds one dcid, so a bound of 1 admits it whatever its members
    number."""
    search_settings(max_child_places=1)

    region = _resubmit(_where_value_slot("boreal", "Boreal"), _held_at("boreal"))

    assert region.status == "definite"


# Single-value resubmits keep their scalar binding.


def test_posted_single_value_where_unchanged():
    """A single-place resubmit keeps its value binding."""
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_value_slot("country/GTM", "Guatemala")
    region = _resubmit(where_slot, obs)
    assert region.status == "definite"
    where = _where_slot(region)
    assert isinstance(where.binding, BindingValue)
    assert where.binding.value.ref.ref == "country/GTM"
