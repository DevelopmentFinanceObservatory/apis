"""Tests options-surface membership and response assembly with derived populations.

Covers place, facet, and directional options, coverage probes, refusals, and wire fields."""

from __future__ import annotations

import dataclasses
import time

import pydantic
import pytest
from fastapi.testclient import TestClient

from materialise.facets import ElectedDataset
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import GraphReadError, SubjectPage
from origin_catalogue import note_un_blocked_places
from search.engine.config import settings
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError, GraphCapExceededError
from search.engine.options import surface
from search.engine.options.population import OptionsPopulation
from search.engine.options.resolve import ResolvedOptionsTarget, resolve_options_target
from search.engine.options.surface import build_options_response
from search.engine.region import OPTION_COVERAGE_PROBE_FAILED
from search.engine.region.codes import UN_SYSTEM_UNAVAILABLE
from search.models import (
    BaseIdentity,
    CoverageBreadth,
    IdentityBindingSet,
    IdentityBindingUnbound,
    IdentityBindingValue,
    IdentityFacetValueAbsent,
    IdentitySlot,
    IdentityWindowAll,
    IdentityWindowRange,
    OptionsFor,
    OptionsRequest,
    ResolvedEntity,
)
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import (
    CoverageRow,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.materialise.conftest import FakeGraphReader

# ---------------------------------------------------------------------------
# These synthetic five-tuples are absent from the code index, so tests use graph enumeration.
# Life expectancy has one subject member; gender has two constrained members.
# ---------------------------------------------------------------------------

_LIFE_EXPECTANCY_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "lifeExpectancy",
    "stat_type": "measuredValue",
}
_LIFE_EXPECTANCY_SV = "LifeExpectancy_Person"
_LIFE_EXPECTANCY_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    _LIFE_EXPECTANCY_SV: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
    },
}

_GENDER_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "count",
    "stat_type": "measuredValue",
}
_GENDER_SV_MALE = "Count_Person_Male"
_GENDER_SV_FEMALE = "Count_Person_Female"
_GENDER_SLOTS = [IdentitySlot(property="gender", binding=IdentityBindingUnbound())]
_GENDER_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    _GENDER_SV_MALE: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Male",),
    },
    _GENDER_SV_FEMALE: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Female",),
    },
}


def _subjects_from_arcs(
    arcs: dict[str, dict[str, tuple[str, ...]]],
) -> dict[tuple[str, str], SubjectPage]:
    """Builds reverse-query entries from `arcs` for `FakeGraphReader`.

    Forward-arc confirmation applies five-tuple and slot filters."""
    by_measured_property: dict[str, list[str]] = {}
    for dcid, node_arcs in arcs.items():
        for measured_property in node_arcs.get("measuredProperty", ()):
            by_measured_property.setdefault(measured_property, []).append(dcid)
    return {
        (measured_property, "measuredProperty"): SubjectPage(
            dcids=tuple(sorted(dcids)), truncated=False
        )
        for measured_property, dcids in by_measured_property.items()
    }


def _reader(arcs: dict, *, facets: dict | None = None) -> FakeGraphReader:
    """Builds a reader for population enumeration and optional facet probes."""
    return FakeGraphReader(arcs=arcs, subjects=_subjects_from_arcs(arcs), facets=facets or {})


def _place_identity(
    *,
    shape: dict = _LIFE_EXPECTANCY_SHAPE,
    slots: list[IdentitySlot] | None = None,
    entities: dict[str, list[str]] | None = None,
    requested_window=None,
    facets=None,
) -> BaseIdentity:
    return BaseIdentity(
        shape=shape,
        slots=slots or [],
        entities=entities or {},
        requested_window=requested_window or IdentityWindowAll(),
        facets=facets,
    )


def _place_request(
    *,
    shape: dict = _LIFE_EXPECTANCY_SHAPE,
    slots: list[IdentitySlot] | None = None,
    entities: dict[str, list[str]] | None = None,
    scope: str | None = None,
    include_coverage: bool = False,
    requested_window=None,
    facets=None,
    role: str = "observationAbout",
) -> OptionsRequest:
    identity = _place_identity(
        shape=shape,
        slots=slots,
        entities=entities,
        requested_window=requested_window,
        facets=facets,
    )
    return OptionsRequest(
        identity=identity,
        **{"for": OptionsFor(role=role, scope=scope)},
        include_coverage=include_coverage,
    )


def _materialise_graph_for(graph: FakeGraph, *, arcs: dict) -> FakeGraphReader:
    """Builds materialise facets from `graph._obs` so presence and facet reads share fixture data."""
    facets: dict[tuple[str, str], list[MaterialiseFacet]] = {}
    for key, raw_list in graph._obs.items():
        if "|" not in key:
            continue
        sv, entity = key.split("|", 1)
        facets[(sv, entity)] = [
            MaterialiseFacet(
                # The mixer names every facet. A fixture without `facetId` takes the position
                # id `FakeGraph.observation_facets` gives the same facet.
                facet_id=raw.get("facetId") or f"f{i}",
                provenance_id=raw.get("provenanceId"),
                measurement_method=raw.get("measurementMethod"),
                unit=raw.get("unit"),
                obs_count=raw.get("obsCount", 0),
                earliest_date=raw.get("earliestDate"),
                latest_date=raw.get("latestDate"),
            )
            for i, raw in enumerate(raw_list)
        ]
    return FakeGraphReader(arcs=arcs, subjects=_subjects_from_arcs(arcs), facets=facets)


def _options_response(
    req: OptionsRequest,
    *,
    graph: FakeGraph,
    sdmx: FakeSdmxClient | None = None,
    materialise_graph: FakeGraphReader | None = None,
    population_arcs: dict = _LIFE_EXPECTANCY_ARCS,
):
    sdmx = sdmx if sdmx is not None else FakeSdmxClient(observations={})
    materialise_graph = (
        materialise_graph
        if materialise_graph is not None
        else _materialise_graph_for(graph, arcs=population_arcs)
    )
    target = resolve_options_target(req, graph=graph, materialise_graph=materialise_graph)
    return target, build_options_response(
        req,
        target,
        graph=graph,
        start_ms=now_ms(),
        materialise_graph=materialise_graph,
        coverage=empty_coverage_snapshot(),
    )


# ---------------------------------------------------------------------------
# observationAbout scopes enumerate places with data for the resolved population.
# ---------------------------------------------------------------------------

_PLACE_NODES = {
    "country/GTM": {"label": "Guatemala", "type": "Country"},
    "country/USA": {"label": "United States", "type": "Country"},
    "country/VAT": {"label": "Vatican City", "type": "Country"},
    "country/ABW": {"label": "Aruba", "type": "Country"},
    "africa": {"label": "Africa", "type": "Continent"},
    "asia": {"label": "Asia", "type": "Continent"},
    "Earth": {"label": "World", "type": "Planet"},
    "wikidataId/Q178136": {"label": "Sololá Department", "type": "AdministrativeArea1"},
    "wikidataId/Q21296012": {"label": "Amatitlán Department", "type": "AdministrativeArea1"},
    "wikidataId/Q466061": {"label": "Petén Department", "type": "AdministrativeArea1"},
}

_PLACE_CHILDREN = {
    "Earth": {
        "Country": ["country/GTM", "country/USA", "country/VAT", "country/ABW"],
        "Continent": ["africa", "asia"],
    },
    "country/GTM": {
        "AdministrativeArea1": [
            "wikidataId/Q178136",
            "wikidataId/Q21296012",
            "wikidataId/Q466061",
        ],
    },
}

# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_OBS_PROVENANCE = "dc/base/WorldDevelopmentIndicators"

_PLACE_OBS = {
    f"{_LIFE_EXPECTANCY_SV}|country/GTM": [
        {
            "earliestDate": "1960",
            "latestDate": "2024",
            "obsCount": 65,
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_LIFE_EXPECTANCY_SV}|country/USA": [
        {
            "earliestDate": "1960",
            "latestDate": "2024",
            "obsCount": 65,
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_LIFE_EXPECTANCY_SV}|country/ABW": [
        {
            "earliestDate": "2000",
            "latestDate": "2024",
            "obsCount": 25,
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
}


def _place_graph(obs: dict | None = None) -> FakeGraph:
    return FakeGraph(
        nodes=_PLACE_NODES,
        obs=obs if obs is not None else _PLACE_OBS,
        detect={},
        resolve={},
        children=_PLACE_CHILDREN,
        enum_members={},
    )


class TestObservationAboutPeersScope:
    def test_country_anchor_happy_path(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        _target, resp = _options_response(req, graph=_place_graph())
        assert resp.options.truncated is False
        dcids = {i.value.ref.dcid for i in resp.options.items}
        # GTM, USA, ABW have data; VAT does not.
        assert dcids == {"country/GTM", "country/USA", "country/ABW"}
        labels = {i.value.ref.dcid: i.value.ref.label for i in resp.options.items}
        assert labels["country/GTM"] == "Guatemala"

    def test_continent_anchor_empty_result_is_valid(self):
        """Neither continent in the fixture (africa, asia) has data -- the
        peers enumeration correctly reports a valid, empty result rather
        than a refusal."""
        req = _place_request(entities={"observationAbout": ["africa"]})
        _target, resp = _options_response(req, graph=_place_graph())
        assert resp.options.items == []
        assert resp.options.truncated is False

    def test_unsupported_anchor_type_refuses(self):
        req = _place_request(
            entities={"observationAbout": ["wikidataId/Q178136"]}  # AdministrativeArea1
        )
        with pytest.raises(EngineInputError) as exc_info:
            _options_response(req, graph=_place_graph())
        assert exc_info.value.code == "unsupported_peers_anchor_type"
        assert exc_info.value.status == 400

    def test_no_bound_entity_defaults_to_country(self):
        """With no anchor, peer options default to countries with data."""
        req = _place_request(entities={})
        _target, resp = _options_response(req, graph=_place_graph())
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/GTM", "country/USA", "country/ABW"}

    def test_several_same_typed_anchors_agree_regardless_of_posted_order(self):
        req_gtm_first = _place_request(
            entities={"observationAbout": ["country/GTM", "country/USA"]}
        )
        req_usa_first = _place_request(
            entities={"observationAbout": ["country/USA", "country/GTM"]}
        )
        _t1, resp1 = _options_response(req_gtm_first, graph=_place_graph())
        _t2, resp2 = _options_response(req_usa_first, graph=_place_graph())
        dcids1 = {i.value.ref.dcid for i in resp1.options.items}
        dcids2 = {i.value.ref.dcid for i in resp2.options.items}
        assert dcids1 == dcids2 == {"country/GTM", "country/USA", "country/ABW"}

    def test_differing_anchor_types_refuse_mixed_peers_anchor_types(self):
        req = _place_request(entities={"observationAbout": ["country/GTM", "africa"]})
        with pytest.raises(EngineInputError) as exc_info:
            _options_response(req, graph=_place_graph())
        assert exc_info.value.code == "mixed_peers_anchor_types"
        assert exc_info.value.pointer == "/identity/entities/observationAbout"

    def test_duplicate_anchor_dcids_do_not_trigger_mixed_peers_anchor_types(self):
        req = _place_request(entities={"observationAbout": ["country/GTM", "country/GTM"]})
        _target, resp = _options_response(req, graph=_place_graph())
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/GTM", "country/USA", "country/ABW"}

    def test_deterministic_ordering(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        graph = _place_graph()
        _t1, resp1 = _options_response(req, graph=graph)
        _t2, resp2 = _options_response(req, graph=graph)
        dcids1 = [i.value.ref.dcid for i in resp1.options.items]
        dcids2 = [i.value.ref.dcid for i in resp2.options.items]
        assert dcids1 == dcids2
        assert dcids1 == ["country/GTM", "country/USA", "country/ABW"]

    def test_truncation_sets_truncated_true(self, search_settings):
        search_settings(max_options=2)
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        _target, resp = _options_response(req, graph=_place_graph())
        assert resp.options.truncated is True
        assert len(resp.options.items) <= 2

    def test_coverage_populated_only_under_the_flag(self):
        req_no_coverage = _place_request(entities={"observationAbout": ["country/GTM"]})
        _target, resp = _options_response(req_no_coverage, graph=_place_graph())
        assert all(item.coverage is None for item in resp.options.items)

        req_coverage = _place_request(
            entities={"observationAbout": ["country/GTM"]}, include_coverage=True
        )
        _target2, resp2 = _options_response(req_coverage, graph=_place_graph())
        by_dcid = {i.value.ref.dcid: i for i in resp2.options.items}
        assert by_dcid["country/GTM"].coverage.has_data is True
        assert by_dcid["country/GTM"].coverage.observation_count == 65


# ---------------------------------------------------------------------------
# Union membership admits a candidate with data for either gender member.
# ---------------------------------------------------------------------------

_DEPT_UNION_OBS = {
    f"{_GENDER_SV_MALE}|wikidataId/Q178136": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_GENDER_SV_FEMALE}|wikidataId/Q178136": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    # Amatitlán has data for one member and qualifies under union membership.
    f"{_GENDER_SV_MALE}|wikidataId/Q21296012": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    # Petén has no matching member and is excluded.
}


class TestObservationAboutChildrenScope:
    def test_country_anchor_department_union(self):
        """Union across the two resolved StatVars: Sololá (both members) and
        Amatitlán (one member only) both qualify; Petén (neither) does not."""
        req = _place_request(
            shape=_GENDER_SHAPE,
            slots=_GENDER_SLOTS,
            entities={"observationAbout": ["country/GTM"]},
            scope="children",
        )
        graph = _place_graph(obs=_DEPT_UNION_OBS)
        _target, resp = _options_response(req, graph=graph, population_arcs=_GENDER_ARCS)
        assert resp.options.truncated is False
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"wikidataId/Q178136", "wikidataId/Q21296012"}

    def test_empty_children_is_valid_empty_result(self):
        req = _place_request(
            entities={"observationAbout": ["country/USA"]},  # no AdministrativeArea1 entry
            scope="children",
        )
        _target, resp = _options_response(req, graph=_place_graph())
        assert resp.options.items == []
        assert resp.options.truncated is False

    def test_unsupported_anchor_type_refuses(self):
        req = _place_request(
            entities={"observationAbout": ["wikidataId/Q178136"]},  # AdministrativeArea1
            scope="children",
        )
        with pytest.raises(EngineInputError) as exc_info:
            _options_response(req, graph=_place_graph())
        assert exc_info.value.code == "unsupported_children_anchor_type"
        assert exc_info.value.status == 400

    def test_no_bound_entity_refuses(self):
        req = _place_request(entities={}, scope="children")
        with pytest.raises(EngineInputError) as exc_info:
            _options_response(req, graph=_place_graph())
        assert exc_info.value.code == "children_anchor_required"
        assert exc_info.value.status == 400

    def test_multiple_distinct_anchors_refuse_children_anchor_ambiguous(self):
        req = _place_request(
            entities={"observationAbout": ["country/GTM", "country/USA"]}, scope="children"
        )
        with pytest.raises(EngineInputError) as exc_info:
            _options_response(req, graph=_place_graph())
        assert exc_info.value.code == "children_anchor_ambiguous"
        assert exc_info.value.pointer == "/identity/entities/observationAbout"
        assert exc_info.value.code != "children_anchor_required"

    def test_duplicate_anchor_dcids_do_not_trigger_children_anchor_ambiguous(self):
        req = _place_request(
            entities={"observationAbout": ["country/GTM", "country/GTM"]}, scope="children"
        )
        _target, resp = _options_response(req, graph=_place_graph())
        assert resp.options.items == []
        assert resp.options.truncated is False


# ---------------------------------------------------------------------------
# Facet summaries lack point dates, so windowed coverage is a breadth estimate.
# ---------------------------------------------------------------------------


class TestObservationAboutWindowedCoverage:
    def test_windowed_request_demotes_to_breadth_not_an_exact_count(self):
        req = _place_request(
            entities={"observationAbout": ["country/GTM"]},
            requested_window=IdentityWindowRange(start_year=2015, end_year=2019),
            include_coverage=True,
        )
        _target, resp = _options_response(req, graph=_place_graph())
        item = next(i for i in resp.options.items if i.value.ref.dcid == "country/GTM")
        assert isinstance(item.coverage, CoverageBreadth), (
            "a windowed coverage request over materialise's own dateless facet probe must "
            f"demote to CoverageBreadth; got {type(item.coverage).__name__}"
        )
        assert item.coverage.has_data is True


# ---------------------------------------------------------------------------
# A bound facet narrows membership; a candidate needs a matching facet on one member.
# ---------------------------------------------------------------------------

_UNIT_BOUND_OBS = {
    f"{_LIFE_EXPECTANCY_SV}|country/GTM": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 100,
            "unit": "EUR",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_LIFE_EXPECTANCY_SV}|country/USA": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 50,
            "unit": "EUR",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
}


def _unit_bound_request(
    *,
    include_coverage: bool = False,
    shape: dict = _LIFE_EXPECTANCY_SHAPE,
    slots: list[IdentitySlot] | None = None,
) -> OptionsRequest:
    return _place_request(
        shape=shape,
        slots=slots,
        entities={},
        facets={"unit": IdentityBindingValue(ref="USD")},
        include_coverage=include_coverage,
    )


_TWO_SV_UNIT_BOUND_OBS = {
    f"{_GENDER_SV_MALE}|country/GTM": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    # GTM's second StatVar only carries a EUR facet.
    f"{_GENDER_SV_FEMALE}|country/GTM": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "EUR",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_GENDER_SV_MALE}|country/USA": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_GENDER_SV_FEMALE}|country/USA": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
}


class TestObservationAboutFacetBoundMembership:
    def test_unit_bound_identity_excludes_non_matching_candidates(self):
        req = _unit_bound_request()
        _target, resp = _options_response(req, graph=_place_graph(obs=_UNIT_BOUND_OBS))
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/GTM"}

    def test_membership_admitted_when_any_one_population_member_satisfies(self):
        """GTM qualifies from its first StatVar's USD facet.
        Its second StatVar has only EUR, and union membership still admits it.
        USA has USD facets on both StatVars."""
        req = _unit_bound_request(shape=_GENDER_SHAPE, slots=_GENDER_SLOTS)
        graph = _place_graph(obs=_TWO_SV_UNIT_BOUND_OBS)
        _target, resp = _options_response(req, graph=graph, population_arcs=_GENDER_ARCS)
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/GTM", "country/USA"}

    def test_coverage_reflects_only_matching_facets(self):
        req = _unit_bound_request(include_coverage=True)
        _target, resp = _options_response(req, graph=_place_graph(obs=_UNIT_BOUND_OBS))
        by_dcid = {i.value.ref.dcid: i for i in resp.options.items}
        assert set(by_dcid) == {"country/GTM"}
        # The USD facet contributes 5 observations.
        # The EUR facet's 100 observations fall outside the binding.
        assert by_dcid["country/GTM"].coverage.observation_count == 5

    def test_a_null_method_pin_counts_only_the_facets_with_no_method(self):
        """Guatemala carries a recorded series with no method and an Estimated projection.
        Under a pin on the facets that declare no method, its coverage counts the recorded
        series' 5 observations and none of the projection's 3."""
        obs = {
            f"{_LIFE_EXPECTANCY_SV}|country/GTM": [
                {
                    "earliestDate": "2010",
                    "latestDate": "2014",
                    "obsCount": 5,
                    "provenanceId": "dc/base/WHO",
                },
                {
                    "earliestDate": "2025",
                    "latestDate": "2027",
                    "obsCount": 3,
                    "measurementMethod": "Estimated",
                    "provenanceId": "dc/base/WHO",
                },
            ],
        }
        req = _place_request(
            entities={},
            facets={"measurementMethod": IdentityFacetValueAbsent(ref=None)},
            include_coverage=True,
        )
        _target, resp = _options_response(req, graph=_place_graph(obs=obs))
        by_dcid = {i.value.ref.dcid: i for i in resp.options.items}
        assert set(by_dcid) == {"country/GTM"}
        assert by_dcid["country/GTM"].coverage.observation_count == 5

    def test_a_null_method_pin_offers_only_places_with_a_facet_declaring_no_method(self):
        """Guatemala carries a recorded series with no method; the United States only an
        Estimated projection. Pinning the facets that declare no method offers Guatemala."""
        obs = {
            f"{_LIFE_EXPECTANCY_SV}|country/GTM": [
                {"earliestDate": "2010", "latestDate": "2014", "obsCount": 5},
            ],
            f"{_LIFE_EXPECTANCY_SV}|country/USA": [
                {
                    "earliestDate": "2025",
                    "latestDate": "2027",
                    "obsCount": 3,
                    "measurementMethod": "Estimated",
                },
            ],
        }
        req = _place_request(
            entities={}, facets={"measurementMethod": IdentityFacetValueAbsent(ref=None)}
        )
        _target, resp = _options_response(req, graph=_place_graph(obs=obs))
        assert [i.value.ref.dcid for i in resp.options.items] == ["country/GTM"]

    @pytest.mark.parametrize(
        ("elected", "offered"),
        [("dc/base/WHO", ["country/GTM"]), (None, ["country/GTM", "country/USA"])],
        ids=["elected_dataset", "no_election"],
    )
    def test_a_pin_offers_only_places_the_members_elected_dataset_serves(self, elected, offered):
        """Guatemala's facet is the WHO dataset's and the United States' another dataset's.
        A member whose family elected WHO is read from WHO alone, as a mint reads it, so a
        pin that both facets satisfy offers Guatemala only, and its coverage counts the WHO
        facet. With no election both places are offered and ranking picks the larger facet."""
        obs = {
            f"{_LIFE_EXPECTANCY_SV}|country/GTM": [
                {
                    "earliestDate": "2010",
                    "latestDate": "2014",
                    "obsCount": 5,
                    "provenanceId": "dc/base/WHO",
                },
                {
                    "earliestDate": "2000",
                    "latestDate": "2014",
                    "obsCount": 15,
                    "provenanceId": "dc/base/Other",
                },
            ],
            f"{_LIFE_EXPECTANCY_SV}|country/USA": [
                {
                    "earliestDate": "2010",
                    "latestDate": "2014",
                    "obsCount": 5,
                    "provenanceId": "dc/base/Other",
                },
            ],
        }
        req = _place_request(
            entities={},
            facets={"measurementMethod": IdentityFacetValueAbsent(ref=None)},
            include_coverage=True,
        )
        graph = _place_graph(obs=obs)
        materialise_graph = _materialise_graph_for(graph, arcs=_LIFE_EXPECTANCY_ARCS)
        target = resolve_options_target(req, graph=graph, materialise_graph=materialise_graph)
        if elected is not None:
            target = dataclasses.replace(
                target,
                population=dataclasses.replace(
                    target.population,
                    datasets={
                        _LIFE_EXPECTANCY_SV: ElectedDataset(
                            provenance=elected,
                            member_provenances=("dc/base/Other", "dc/base/WHO"),
                        )
                    },
                ),
            )

        resp = build_options_response(
            req,
            target,
            graph=graph,
            start_ms=now_ms(),
            materialise_graph=materialise_graph,
            coverage=empty_coverage_snapshot(),
        )

        by_dcid = {i.value.ref.dcid: i for i in resp.options.items}
        assert sorted(by_dcid) == offered
        # Coverage counts the one source the plan elects: WHO's, or the larger by ranking.
        expected_count = 5 if elected is not None else 15
        assert by_dcid["country/GTM"].coverage.observation_count == expected_count

    def test_no_facet_bindings_leaves_options_unchanged(self):
        graph = _place_graph(obs=_UNIT_BOUND_OBS)
        req = _place_request(entities={})
        _target, resp = _options_response(req, graph=graph)
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/GTM", "country/USA"}


# ---------------------------------------------------------------------------
# A set binding requires every member on one StatVar's facet list.
# ---------------------------------------------------------------------------

_SET_BOUND_SINGLE_SV_OBS = {
    f"{_LIFE_EXPECTANCY_SV}|country/GTM": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_LIFE_EXPECTANCY_SV}|country/USA": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 7,
            "unit": "EUR",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
}

_SET_BOUND_TWO_SV_OBS = {
    f"{_GENDER_SV_MALE}|country/GTM": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    # GTM's second StatVar carries only EUR -- neither of GTM's two StatVars
    # carries both members on its own, even though pooling across both would.
    f"{_GENDER_SV_FEMALE}|country/GTM": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "EUR",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_GENDER_SV_MALE}|country/USA": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "EUR",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
    f"{_GENDER_SV_FEMALE}|country/USA": [
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "USD",
            "provenanceId": _OBS_PROVENANCE,
        },
        {
            "earliestDate": "2010",
            "latestDate": "2020",
            "obsCount": 5,
            "unit": "EUR",
            "provenanceId": _OBS_PROVENANCE,
        },
    ],
}


def _set_bound_request(
    refs: list[str],
    *,
    shape: dict = _LIFE_EXPECTANCY_SHAPE,
    slots: list[IdentitySlot] | None = None,
) -> OptionsRequest:
    return _place_request(
        shape=shape, slots=slots, entities={}, facets={"unit": IdentityBindingSet(refs=refs)}
    )


class TestObservationAboutSetBindingMembership:
    def test_partial_member_on_a_stat_var_excludes_the_candidate(self):
        """GTM's only StatVar carries USD but not EUR -- a `{USD, EUR}`
        binding excludes it even though its one USD facet would satisfy an
        any-member reading. USA, whose StatVar carries both members, stays."""
        req = _set_bound_request(["USD", "EUR"])
        _target, resp = _options_response(req, graph=_place_graph(obs=_SET_BOUND_SINGLE_SV_OBS))
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/USA"}

    def test_multi_stat_var_split_members_excludes_the_candidate(self):
        """A `{USD, EUR}` binding where GTM's first StatVar carries only USD
        and its second carries only EUR must exclude GTM: neither StatVar's
        own facet list covers both members, even though pooling facets
        ACROSS the two StatVars would -- unaffected by the any-one-StatVar
        union fix, since NEITHER StatVar individually satisfies the set.
        USA, whose every StatVar carries both members, stays."""
        req = _set_bound_request(["USD", "EUR"], shape=_GENDER_SHAPE, slots=_GENDER_SLOTS)
        graph = _place_graph(obs=_SET_BOUND_TWO_SV_OBS)
        _target, resp = _options_response(req, graph=graph, population_arcs=_GENDER_ARCS)
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/USA"}


# ---------------------------------------------------------------------------
# Presence reads use the SDMX byte cap and return typed refusals on cap trips.
# ---------------------------------------------------------------------------


class _SpyPresenceGraph(FakeGraph):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entities_with_data_max_bytes: list[int | None] = []

    def entities_with_data(self, variable_dcids, entity_dcids, *, max_bytes=None):
        self.entities_with_data_max_bytes.append(max_bytes)
        return super().entities_with_data(variable_dcids, entity_dcids, max_bytes=max_bytes)


class _CapExceededPresenceGraph(FakeGraph):
    def entities_with_data(self, variable_dcids, entity_dcids, *, max_bytes=None):
        raise GraphCapExceededError(
            "simulated cap trip",
            cap_kind="bytes",
            cap=max_bytes or 0,
            observed=(max_bytes or 0) + 1,
        )


class TestObservationAboutPresenceReadIsBounded:
    def test_entities_with_data_is_called_with_the_sdmx_byte_cap(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        graph = _SpyPresenceGraph(
            nodes=_PLACE_NODES,
            obs=_PLACE_OBS,
            detect={},
            resolve={},
            children=_PLACE_CHILDREN,
            enum_members={},
        )
        _target, _resp = _options_response(req, graph=graph)
        assert graph.entities_with_data_max_bytes == [settings().sdmx_max_bytes]

    def test_cap_trip_yields_a_typed_refusal_not_a_500(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        graph = _CapExceededPresenceGraph(
            nodes=_PLACE_NODES,
            obs=_PLACE_OBS,
            detect={},
            resolve={},
            children=_PLACE_CHILDREN,
            enum_members={},
        )
        with pytest.raises(EngineInputError) as exc_info:
            _options_response(req, graph=graph)
        assert exc_info.value.code == "scope_too_broad"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/for/role"


# ---------------------------------------------------------------------------
# Directional options use the declared role and filter by the opposite anchor.
# ---------------------------------------------------------------------------

_DIRECTIONAL_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "EconomicActivity",
    "measured_property": "amount",
    "stat_type": "measuredValue",
}
_DIRECTIONAL_SV_DCID = "TESTFIXTURE/OptionsSurfaceRoleAxes/BothAxes"
_DIRECTIONAL_DEST_DCID = "country/KEN"
_DIRECTIONAL_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    _DIRECTIONAL_SV_DCID: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("EconomicActivity",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": ("financingProvider", "financingRecipient"),
    },
}


def _directional_graph() -> FakeGraph:
    nodes = {
        "country/KEN": {"label": "Kenya", "type": "Country"},
        "country/UGA": {"label": "Uganda", "type": "Country"},
        "country/FRA": {"label": "France", "type": "Country"},
        "country/DEU": {"label": "Germany", "type": "Country"},
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _directional_request(
    *,
    role: str,
    entities: dict[str, list[str]] | None = None,
    scope: str | None = None,
    requested_window=None,
) -> OptionsRequest:
    identity = _place_identity(
        shape=_DIRECTIONAL_SHAPE,
        entities=entities if entities is not None else {"destination": [_DIRECTIONAL_DEST_DCID]},
        requested_window=requested_window,
    )
    return OptionsRequest(identity=identity, **{"for": OptionsFor(role=role, scope=scope)})


_DIRECTIONAL_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_DIRECTIONAL_UNBOUND_DONOR = "TESTFIXTURE/OptionsSurfaceRoleAxes/UnboundDonor"


def _directional_coverage_rows(
    *,
    role: str,
    members: list[str],
    opposite_dcids: list[str] | None = None,
    noise: list[tuple[str, str]] = (),
    facet_id: str = "f1",
) -> list[CoverageRow]:
    """Builds role-axis coverage rows crossed with opposite anchors.

    `noise` adds a separate-facet pair to test opposite-anchor filtering."""
    coverage_role = "recipient" if role == "destination" else "donor"
    opposite = opposite_dcids or [_DIRECTIONAL_UNBOUND_DONOR]
    rows: list[CoverageRow] = [
        (
            _DIRECTIONAL_SV_DCID,
            facet_id,
            *((opp, member) if coverage_role == "recipient" else (member, opp)),
            [2020],
        )
        for member in members
        for opp in opposite
    ]
    rows += [
        (_DIRECTIONAL_SV_DCID, "noise", donor, recipient, [2020]) for donor, recipient in noise
    ]
    return rows


def _directional_response(
    *, role: str, monkeypatch: pytest.MonkeyPatch, rows: list[CoverageRow], requested_window=None
):
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, rows, role_names={_DIRECTIONAL_SV_DCID: _DIRECTIONAL_ROLE_NAMES}
    )
    graph = _directional_graph()
    sdmx = FakeSdmxClient(observations={}, availability={})
    req = _directional_request(role=role, requested_window=requested_window)
    target = resolve_options_target(req, graph=graph, materialise_graph=_reader(_DIRECTIONAL_ARCS))
    resp = build_options_response(
        req,
        target,
        graph=graph,
        start_ms=now_ms(),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    return target, resp, sdmx


class TestDirectionalOptions:
    def test_places_the_un_refused_during_the_read_are_reported(self, monkeypatch):
        build_set = surface._directional_options_set

        def refusing(*args, **kwargs):
            note_un_blocked_places(["country/XYZ"])
            return build_set(*args, **kwargs)

        monkeypatch.setattr(surface, "_directional_options_set", refusing)
        rows = _directional_coverage_rows(role="destination", members=["country/KEN"])
        _target, resp, _sdmx = _directional_response(
            role="destination", monkeypatch=monkeypatch, rows=rows
        )

        (warning,) = [w for w in resp.diagnostics.warnings if w.code == UN_SYSTEM_UNAVAILABLE]
        assert "country/XYZ" in warning.message

    def test_destination_request_returns_coverage_members_with_labels(self, monkeypatch):
        rows = _directional_coverage_rows(
            role="destination", members=["country/KEN", "country/UGA"]
        )
        _target, resp, _sdmx = _directional_response(
            role="destination", monkeypatch=monkeypatch, rows=rows
        )
        dcids = [i.value.ref.dcid for i in resp.options.items]
        assert dcids == ["country/KEN", "country/UGA"]
        labels = {i.value.ref.dcid: i.value.ref.label for i in resp.options.items}
        assert labels["country/UGA"] == "Uganda"
        assert resp.options.truncated is False

    def test_destination_request_is_windowed_by_the_requested_window(self, monkeypatch):
        """A 2015-2025 window includes Uganda's 2020 row and excludes Kenya's 2010 row."""
        rows = [
            (_DIRECTIONAL_SV_DCID, "f1", _DIRECTIONAL_UNBOUND_DONOR, "country/KEN", [2010]),
            (_DIRECTIONAL_SV_DCID, "f1", _DIRECTIONAL_UNBOUND_DONOR, "country/UGA", [2020]),
        ]
        window = IdentityWindowRange(start_year=2015, end_year=2025)
        _target, resp, _sdmx = _directional_response(
            role="destination", monkeypatch=monkeypatch, rows=rows, requested_window=window
        )
        dcids = [i.value.ref.dcid for i in resp.options.items]
        assert dcids == ["country/UGA"]

    def test_origin_request_is_filtered_by_the_bound_destination(self, monkeypatch):
        rows = _directional_coverage_rows(
            role="origin",
            members=["country/DEU", "country/FRA"],
            opposite_dcids=[_DIRECTIONAL_DEST_DCID],
            # A donor with data only at a DIFFERENT recipient: proves the
            # bound-destination filter actually narrows the coverage
            # lookup, not merely that the seeded members come back.
            noise=[("country/JPN", "country/UGA")],
        )
        _target, resp, _sdmx = _directional_response(
            role="origin", monkeypatch=monkeypatch, rows=rows
        )
        dcids = [i.value.ref.dcid for i in resp.options.items]
        assert dcids == ["country/DEU", "country/FRA"]

    def test_empty_coverage_read_is_a_valid_empty_result(self, monkeypatch):
        _target, resp, _sdmx = _directional_response(
            role="destination", monkeypatch=monkeypatch, rows=[]
        )
        assert resp.options.items == []
        assert resp.options.truncated is False


# ---------------------------------------------------------------------------
# A role-bearing population served end to end through the mounted route,
# proving the whole app wiring (materialise_graph population derivation,
# typed anchors, the directional coverage read) composes correctly -- not
# just the unit-level pieces above.
# ---------------------------------------------------------------------------


class TestDirectionalOptionsRouteWiring:
    def test_bound_destination_route_returns_filtered_origin_options(self, monkeypatch):
        """The mounted route derives a directional population and returns filtered origin options."""
        rows = _directional_coverage_rows(
            role="origin",
            members=["country/DEU", "country/FRA"],
            opposite_dcids=[_DIRECTIONAL_DEST_DCID],
        )
        _auto_coverage_index_1 = patch_coverage(
            monkeypatch, rows, role_names={_DIRECTIONAL_SV_DCID: _DIRECTIONAL_ROLE_NAMES}
        )
        graph = _directional_graph()
        sdmx = FakeSdmxClient(observations={}, availability={})
        req = _directional_request(role="origin")
        client = TestClient(
            search_app(
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=sdmx,
                materialise_graph=_reader(_DIRECTIONAL_ARCS),
            ),
            raise_server_exceptions=False,
        )
        resp = client.post("/api/search/options", json=req.model_dump(by_alias=True, mode="json"))
        assert resp.status_code == 200, resp.text
        items = resp.json()["options"]["items"]
        assert [i["value"]["ref"]["dcid"] for i in items] == ["country/DEU", "country/FRA"]
        assert [i["value"]["ref"]["label"] for i in items] == ["Germany", "France"]


# ---------------------------------------------------------------------------
# A bound-facet read is required for membership and propagates on failure.
# ---------------------------------------------------------------------------


class _FailingFacetsGraphReader(FakeGraphReader):
    def observation_facets_batch(self, *, stat_var, entities, deadline: float | None = None):
        raise GraphReadError("simulated transport error during the coverage probe")

    def observation_facets_batch_many(self, *, stat_vars, entities, deadline: float | None = None):
        raise GraphReadError("simulated transport error during the coverage probe")


class TestFacetBoundMembershipFetchFailsClosed:
    def test_binding_present_infra_error_surfaces_as_503_not_presence_only_options(self):
        req = _unit_bound_request()
        graph = _place_graph(obs=_UNIT_BOUND_OBS)
        client = TestClient(
            search_app(
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}),
                materialise_graph=_FailingFacetsGraphReader(
                    arcs=_LIFE_EXPECTANCY_ARCS, subjects=_subjects_from_arcs(_LIFE_EXPECTANCY_ARCS)
                ),
            ),
            raise_server_exceptions=False,
        )
        resp = client.post("/api/search/options", json=req.model_dump(by_alias=True, mode="json"))
        assert resp.status_code == 503
        body = resp.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True

    def test_materialise_graph_uninitialised_surfaces_as_503_via_the_none_guard(self):
        """The route's materialise-graph guard returns 503 before target resolution."""
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        graph = _place_graph()
        client = TestClient(
            search_app(
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}),
            ),
            raise_server_exceptions=False,
        )
        resp = client.post("/api/search/options", json=req.model_dump(by_alias=True, mode="json"))
        assert resp.status_code == 503
        body = resp.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is True

    def test_binding_present_non_infra_exception_also_propagates(self):
        class _BuggyGraphReader(FakeGraphReader):
            def observation_facets_batch(
                self, *, stat_var, entities, deadline: float | None = None
            ):
                raise ValueError("simulated bug, not a graph transport failure")

            def observation_facets_batch_many(
                self, *, stat_vars, entities, deadline: float | None = None
            ):
                raise ValueError("simulated bug, not a graph transport failure")

        req = _unit_bound_request()
        graph = _place_graph(obs=_UNIT_BOUND_OBS)
        with pytest.raises(ValueError):
            _options_response(
                req,
                graph=graph,
                materialise_graph=_BuggyGraphReader(
                    arcs=_LIFE_EXPECTANCY_ARCS, subjects=_subjects_from_arcs(_LIFE_EXPECTANCY_ARCS)
                ),
            )


# ---------------------------------------------------------------------------
# Optional coverage failures return candidates with a warning.
# ---------------------------------------------------------------------------


class TestCoverageProbeFailsOpen:
    def test_observation_about_probe_failure_fails_open_with_a_warning(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]}, include_coverage=True)
        graph = _place_graph()
        client = TestClient(
            search_app(
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}),
                materialise_graph=_FailingFacetsGraphReader(
                    arcs=_LIFE_EXPECTANCY_ARCS, subjects=_subjects_from_arcs(_LIFE_EXPECTANCY_ARCS)
                ),
            )
        )
        resp = client.post("/api/search/options", json=req.model_dump(by_alias=True, mode="json"))
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["options"]["items"], "the candidate place list must still be returned"
        assert all("coverage" not in item for item in payload["options"]["items"])
        codes = [w["code"] for w in payload["diagnostics"]["warnings"]]
        assert OPTION_COVERAGE_PROBE_FAILED in codes


# ---------------------------------------------------------------------------
# The wire must accept the published alias `for` only. `for_` is the internal
# Python attribute name; a client that discovers and posts it directly must
# be rejected exactly like any other undeclared field.
# ---------------------------------------------------------------------------


class TestForAliasWireRejection:
    def test_for_underscore_body_is_rejected(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        client = TestClient(
            search_app(
                graph=FakeGraph(nodes=_PLACE_NODES, obs={}, detect={}, resolve={}),
                resolver=make_resolver(
                    FakeGraph(nodes=_PLACE_NODES, obs={}, detect={}, resolve={})
                ),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}),
            ),
            raise_server_exceptions=False,
        )
        body = req.model_dump(by_alias=True, mode="json")
        body["for_"] = body.pop("for")
        resp = client.post("/api/search/options", json=body)
        assert resp.status_code == 400
        assert resp.json()["code"] == "request_invalid"

    def test_for_alias_body_is_accepted(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        graph = _place_graph()
        client = TestClient(
            search_app(
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}),
                materialise_graph=_materialise_graph_for(graph, arcs=_LIFE_EXPECTANCY_ARCS),
            )
        )
        body = req.model_dump(by_alias=True, mode="json")
        assert "for" in body
        resp = client.post("/api/search/options", json=body)
        assert resp.status_code == 200, resp.text

    def test_model_validate_rejects_the_python_attribute_name(self):
        with pytest.raises(pydantic.ValidationError):
            OptionsRequest.model_validate(
                {
                    "identity": _place_identity().model_dump(mode="json"),
                    "for_": {"role": "observationAbout"},
                }
            )


# ---------------------------------------------------------------------------
# Route timing starts before target resolution.
# ---------------------------------------------------------------------------


class TestDiagnosticsReflectTheWholeRequest:
    def test_route_measures_total_time_from_before_resolution_not_after(self):
        real_resolve = resolve_options_target

        def _slow_resolve(*args, **kwargs):
            time.sleep(0.05)
            return real_resolve(*args, **kwargs)

        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        graph = _place_graph()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("search.engine.app.resolve_options_target", _slow_resolve)
            client = TestClient(
                search_app(
                    graph=graph,
                    resolver=make_resolver(graph),
                    llm=FakeLLM(),
                    sdmx=FakeSdmxClient(observations={}),
                    materialise_graph=_materialise_graph_for(graph, arcs=_LIFE_EXPECTANCY_ARCS),
                )
            )
            resp = client.post(
                "/api/search/options", json=req.model_dump(by_alias=True, mode="json")
            )
        assert resp.status_code == 200
        assert resp.json()["diagnostics"]["timing_ms"]["total"] >= 50


# ---------------------------------------------------------------------------
# Population truncation propagates independently of candidate-list truncation.
# ---------------------------------------------------------------------------


class TestPopulationTruncationPropagatesToOptionsSet:
    def test_observation_about_reports_truncated_even_when_candidates_are_not(self):
        req = _place_request(entities={"observationAbout": ["country/GTM"]})
        graph = _place_graph()
        population = OptionsPopulation(
            stat_var_dcids=(_LIFE_EXPECTANCY_SV,),
            declared_roles=("observationAbout",),
            source="graph",
            truncated=True,
        )
        target = ResolvedOptionsTarget(
            population=population,
            anchors=(
                ResolvedEntity(ref="country/GTM", entity_type="Country", role="observationAbout"),
            ),
            facets=None,
            requested_window=None,
            role_names_by_axis={},
        )
        resp = build_options_response(
            req, target, graph=graph, start_ms=now_ms(), coverage=empty_coverage_snapshot()
        )
        assert resp.options.truncated is True
        # Proves this reports truncated on the population's own strength,
        # not because the candidate list itself was clipped.
        assert len(resp.options.items) < settings().max_options

    def test_directional_reports_truncated_from_the_population_alone(self, monkeypatch):
        rows = _directional_coverage_rows(role="destination", members=["country/KEN"])
        _auto_coverage_index_1 = patch_coverage(
            monkeypatch, rows, role_names={_DIRECTIONAL_SV_DCID: _DIRECTIONAL_ROLE_NAMES}
        )
        graph = _directional_graph()
        population = OptionsPopulation(
            stat_var_dcids=(_DIRECTIONAL_SV_DCID,),
            declared_roles=("financingProvider", "financingRecipient"),
            source="graph",
            truncated=True,
        )
        target = ResolvedOptionsTarget(
            population=population,
            anchors=(),
            facets=None,
            requested_window=None,
            role_names_by_axis={
                "origin": "financingProvider",
                "destination": "financingRecipient",
            },
        )
        req = _directional_request(role="destination", entities={})
        resp = build_options_response(
            req,
            target,
            graph=graph,
            start_ms=now_ms(),
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
        assert resp.options.truncated is True
        assert resp.options.items, "a genuine non-empty result, not the empty-read case"


# ---------------------------------------------------------------------------
# Facet reads are chunked and concurrency-limited, with no StatVar-count cap.
# ---------------------------------------------------------------------------


class TestFacetReadNoLongerBoundedByScopeMaxMembers:
    def test_load_bearing_probe_runs_regardless_of_population_size(self):
        req = _unit_bound_request(shape=_GENDER_SHAPE, slots=_GENDER_SLOTS)
        graph = _place_graph(obs=_TWO_SV_UNIT_BOUND_OBS)
        materialise_graph = _materialise_graph_for(graph, arcs=_GENDER_ARCS)

        _target, resp = _options_response(
            req, graph=graph, materialise_graph=materialise_graph, population_arcs=_GENDER_ARCS
        )

        # test_membership_admitted_when_any_one_population_member_satisfies'
        # own known-good result: no precheck exists to change it.
        dcids = {i.value.ref.dcid for i in resp.options.items}
        assert dcids == {"country/GTM", "country/USA"}

    def test_coverage_only_probe_also_runs_regardless_of_population_size(self):
        req = _place_request(
            shape=_GENDER_SHAPE,
            slots=_GENDER_SLOTS,
            entities={"observationAbout": ["country/GTM"]},
            include_coverage=True,
        )
        graph = _place_graph(obs=_TWO_SV_UNIT_BOUND_OBS)
        materialise_graph = _materialise_graph_for(graph, arcs=_GENDER_ARCS)

        _target, resp = _options_response(
            req, graph=graph, materialise_graph=materialise_graph, population_arcs=_GENDER_ARCS
        )

        assert resp.options.items, "the candidate list must still be returned"
        # The probe ran, and real coverage attached.
        assert any(item.coverage is not None for item in resp.options.items)
        assert OPTION_COVERAGE_PROBE_FAILED not in [w.code for w in resp.diagnostics.warnings]

    def test_deadline_is_threaded_into_the_facet_read(self):
        """build_options_response's own remaining-budget deadline
        (start_ms + SEARCH_REQUEST_TIMEOUT_S) reaches
        observation_facets_via_materialise's own chunked read, the
        same way it already reaches _directional_options_set's availability
        call (TestDirectionalOptions.test_deadline_is_threaded_into_the_
        availability_call)."""

        class _DeadlineCapturingGraphReader(FakeGraphReader):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.deadlines: list[float | None] = []

            def observation_facets_batch(self, *, stat_var, entities, deadline=None):
                self.deadlines.append(deadline)
                return super().observation_facets_batch(
                    stat_var=stat_var, entities=entities, deadline=deadline
                )

            def observation_facets_batch_many(self, *, stat_vars, entities, deadline=None):
                self.deadlines.append(deadline)
                return super().observation_facets_batch_many(
                    stat_vars=stat_vars, entities=entities, deadline=deadline
                )

        req = _unit_bound_request()
        graph = _place_graph(obs=_UNIT_BOUND_OBS)
        materialise_graph = _DeadlineCapturingGraphReader(
            arcs=_LIFE_EXPECTANCY_ARCS,
            subjects=_subjects_from_arcs(_LIFE_EXPECTANCY_ARCS),
        )
        target = resolve_options_target(req, graph=graph, materialise_graph=materialise_graph)

        before = time.monotonic()
        build_options_response(
            req,
            target,
            graph=graph,
            start_ms=now_ms(),
            materialise_graph=materialise_graph,
            coverage=empty_coverage_snapshot(),
        )

        assert materialise_graph.deadlines, "the facet read must have been issued at all"
        assert all(d is not None for d in materialise_graph.deadlines)
        assert all(d > before for d in materialise_graph.deadlines)
