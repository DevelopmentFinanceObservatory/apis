"""Standard queries bind places as a set and probe the expanded country union.

Separate caps govern named and probe places.
Region members enter probes. Bindings retain named regions.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from search.engine.bind import _UniversalBindOutput
from search.engine.region import (
    CHILD_PLACES_CLAMPED,
    MULTI_RECIPIENT_TRUNCATED,
    STANDARD_PROBE_PLACES_CLAMPED,
    RegionResult,
    resolve_variable,
)
from search.models import BindingSet, BindingValue, CoverageExact
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

# A single SV resolved via the standard resolve path.
# The arcs provide populationType and measuredProperty for derive_shapes.
_SV = "Count_Person"
_DETECT_QUERY = "population"

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
    # "Guatemala City area" is a listed alternate name, so the fallback
    # ladder admits the second surface name for this same dcid.
    "country/GTM": {
        "label": "Guatemala",
        "type": "Country",
        "arcs": {"alternateName": {"nodes": [{"value": "Guatemala City area"}]}},
    },
    "country/SLV": {"label": "El Salvador", "type": "Country"},
    "country/HND": {"label": "Honduras", "type": "Country"},
    # Continent types route both names through place expansion.
    "boreal": {"label": "Boreal", "type": "Continent", "arcs": {}},
    "austral": {"label": "Austral", "type": "Continent", "arcs": {}},
}
_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV], "entities": []}}
_RESOLVE: dict = {
    "Guatemala": "country/GTM",
    "El Salvador": "country/SLV",
    "Honduras": "country/HND",
}
_PLACES: dict = {
    "Boreal": {"dcid": "boreal", "type_of": "Continent"},
    "Austral": {"dcid": "austral", "type_of": "Continent"},
}
# Three members each so a SEARCH_MAX_CHILD_PLACES=2 monkeypatch clamps BOTH
# regions (see test_probe_union_cap_overflow_composes_with_per_region_clamp).
_CHILDREN: dict = {
    "boreal": {"Country": ["country/B1", "country/B2", "country/B3"]},
    "austral": {"Country": ["country/A1", "country/A2", "country/A3"]},
}

# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_SOURCE = "dc/base/PlaceFixture"


class _FixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings, unconditionally."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=None,
            ask=None,
        ), None


def _run(
    entities: list[str],
    *,
    role_query: str,
    obs: dict,
    resolve: dict | None = None,
    emit=None,
    variable_index: int = 0,
) -> RegionResult:
    """Run resolve_variable with an inline FakeGraph over the fixtures above."""
    graph = FakeGraph(
        nodes=_NODES,
        detect=_DETECT,
        resolve=resolve if resolve is not None else _RESOLVE,
        obs=obs,
        places=_PLACES,
        children=_CHILDREN,
    )
    return asyncio.run(
        resolve_variable(
            _DETECT_QUERY,
            entities=entities,
            date_request=None,
            detect_query=_DETECT_QUERY,
            role_query=role_query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_steps=[],
            base_timing={},
            emit=emit,
            variable_index=variable_index,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def _where_slot(result: RegionResult):
    return next(s for s in result.specs[0].resolved.slots if s.key.axis == "where")


# A multi-place binding states its places sorted by dcid (`place_roles.place_order`), as the
# identity stores them, so a first answer and its resubmit name them alike.


def test_two_named_countries_bind_as_set_with_distinct_country_count():
    """Two countries bind as a set. Guatemala's 2 facets count as one country."""
    obs = {
        f"{_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
            {"earliestDate": "2019", "latestDate": "2019", "obsCount": 1, "provenanceId": _SOURCE},
        ],
        f"{_SV}|country/SLV": [
            {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Guatemala", "El Salvador"],
        role_query="population Guatemala El Salvador",
        obs=obs,
    )
    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    where = _where_slot(result)
    assert isinstance(where.binding, BindingSet)
    assert [v.ref.dcid for v in where.binding.values] == ["country/GTM", "country/SLV"]
    assert {e.ref for e in spec.resolved.entities} == {"country/GTM", "country/SLV"}
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.dimensions is not None
    countries_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "countries")
    assert countries_dim.count == 2  # Three facets represent 2 distinct countries.


def test_probe_union_covers_region_mix():
    """The binding keeps Boreal. The probe covers Guatemala and Boreal's member countries."""
    obs = {
        f"{_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Guatemala", "Boreal"],
        role_query="population Guatemala Boreal",
        obs=obs,
    )
    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    where = _where_slot(result)
    assert isinstance(where.binding, BindingSet)
    assert [v.ref.dcid for v in where.binding.values] == ["boreal", "country/GTM"]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.dimensions is not None
    countries_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "countries")
    assert countries_dim.count == 1  # Guatemala contributes the one country in coverage.


def test_probe_union_covers_two_regions():
    """The probe union covers member countries from both named regions."""
    obs = {
        f"{_SV}|country/B1": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
        f"{_SV}|country/A1": [
            {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Boreal", "Austral"],
        role_query="population Boreal Austral",
        obs=obs,
    )
    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    where = _where_slot(result)
    assert isinstance(where.binding, BindingSet)
    assert [v.ref.dcid for v in where.binding.values] == ["austral", "boreal"]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.dimensions is not None
    countries_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "countries")
    assert countries_dim.count == 2  # one positive member from EACH region


# Place and probe caps.


def test_named_place_cap_overflow_truncates_and_warns(search_settings):
    """Naming more places than SEARCH_MAX_STANDARD_PLACES drops the trailing ones in query
    order and fires MULTI_RECIPIENT_TRUNCATED naming the counts."""
    search_settings(max_standard_places=2)
    obs = {
        f"{_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Guatemala", "El Salvador", "Honduras"],
        role_query="population Guatemala El Salvador Honduras",
        obs=obs,
    )
    truncated = [w for w in result.warnings if w.code == MULTI_RECIPIENT_TRUNCATED]
    assert len(truncated) == 1
    assert "3 places detected" in truncated[0].message
    assert "first 2 used" in truncated[0].message
    assert result.status == "definite"
    where = _where_slot(result)
    assert isinstance(where.binding, BindingSet)
    assert [v.ref.dcid for v in where.binding.values] == ["country/GTM", "country/SLV"]


def test_named_place_cap_overflow_entities_match_capped_where_set(search_settings):
    """Resolved entities must match the capped where binding."""
    search_settings(max_standard_places=2)
    obs = {
        f"{_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Guatemala", "El Salvador", "Honduras"],
        role_query="population Guatemala El Salvador Honduras",
        obs=obs,
    )
    assert result.status == "definite"
    truncated = [w for w in result.warnings if w.code == MULTI_RECIPIENT_TRUNCATED]
    assert len(truncated) == 1
    where = _where_slot(result)
    assert isinstance(where.binding, BindingSet)
    where_dcids = {v.ref.dcid for v in where.binding.values}
    entity_dcids = {e.ref for e in result.specs[0].resolved.entities}
    assert where_dcids == {"country/GTM", "country/SLV"}
    assert entity_dcids == where_dcids  # HND falls beyond the 2-place cap.


def test_probe_union_cap_overflow_composes_with_per_region_clamp(search_settings):
    """The union cap adds to each region's child-place clamp warning."""
    search_settings(max_child_places=2)
    search_settings(max_standard_probe_places=3)
    obs = {
        f"{_SV}|country/B1": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Boreal", "Austral"],
        role_query="population Boreal Austral",
        obs=obs,
    )
    child_clamped = [w for w in result.warnings if w.code == CHILD_PLACES_CLAMPED]
    probe_clamped = [w for w in result.warnings if w.code == STANDARD_PROBE_PLACES_CLAMPED]
    assert len(child_clamped) == 2  # Boreal (3->2) and Austral (3->2)
    assert len(probe_clamped) == 1  # post-clamp union of 4 capped to 3
    assert "4 probe places detected across 2 named place(s)" in probe_clamped[0].message
    assert "first 3" in probe_clamped[0].message
    assert result.status == "definite"  # country/B1 survives the union clamp


# Scalar bindings for one resolved place.


def test_two_surface_names_one_dcid_stays_scalar():
    """Listed name and alternate name for one DCID collapse to a scalar binding."""
    obs = {
        f"{_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Guatemala", "Guatemala City area"],
        role_query="population Guatemala Guatemala City area",
        obs=obs,
        resolve={
            "Guatemala": "country/GTM",
            "Guatemala City area": "country/GTM",
        },
    )
    assert result.status == "definite"
    where = _where_slot(result)
    assert isinstance(where.binding, BindingValue)
    assert where.binding.value.ref.dcid == "country/GTM"
    assert not any(w.code == MULTI_RECIPIENT_TRUNCATED for w in result.warnings)


def test_single_named_place_stays_scalar_value_binding():
    """One named place keeps a scalar where binding."""
    obs = {
        f"{_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(["Guatemala"], role_query="population Guatemala", obs=obs)
    assert result.status == "definite"
    where = _where_slot(result)
    assert isinstance(where.binding, BindingValue)
    assert where.binding.value.ref.dcid == "country/GTM"
    assert not any(w.code == MULTI_RECIPIENT_TRUNCATED for w in result.warnings)


# Sentence events preserve the multi-place binding.


def test_sentence_event_carries_both_places():
    """Count_Person emits a post-materialisation event whose where slot keeps both places."""
    events: list = []

    async def _emit(event):
        events.append(event)

    obs = {
        f"{_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE},
        ],
    }
    result = _run(
        ["Guatemala", "El Salvador"],
        role_query="population Guatemala El Salvador",
        obs=obs,
        emit=_emit,
    )
    assert result.status == "definite"
    sentence_events = [e for e in events if type(e).__name__ == "SentenceEvent"]
    assert sentence_events, "expected at least one SentenceEvent"
    where = next(s for s in sentence_events[0].slots if s.key.axis == "where")
    assert isinstance(where.binding, BindingSet)
    assert {v.ref.dcid for v in where.binding.values} == {"country/GTM", "country/SLV"}


# Candidate specs retain the full place set.


class TestCandidatesEveryShapeCarriesFullPlaceSet:
    """Every candidate on a 2-place query retains the full where set."""

    _SV_A = "Count_Person"
    _SV_B = "Count_Household"
    _NODES: ClassVar[dict] = {
        _SV_A: {
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
        # A different population type creates a second ShapeDraft.
        _SV_B: {
            "label": "Count Household",
            "type": "StatisticalVariable",
            "arcs": {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "name": {"nodes": [{"value": "Count Household"}]},
                "populationType": {"nodes": [{"dcid": "Household"}]},
                "measuredProperty": {"nodes": [{"dcid": "count"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
            },
        },
        "country/GTM": {"label": "Guatemala", "type": "Country"},
        "country/SLV": {"label": "El Salvador", "type": "Country"},
    }
    _DETECT_QUERY = "population"
    # Scores elect _SV_A first and mark _SV_B as the alternate.
    _DETECT: ClassVar[dict] = {
        _DETECT_QUERY: {"svs": [_SV_A, _SV_B], "cosine_scores": [0.9, 0.85]},
    }
    _RESOLVE: ClassVar[dict] = {"Guatemala": "country/GTM", "El Salvador": "country/SLV"}
    _OBS: ClassVar[dict] = {
        f"{_SV_A}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE}
        ],
        f"{_SV_A}|country/SLV": [
            {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "provenanceId": _SOURCE}
        ],
        f"{_SV_B}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "provenanceId": _SOURCE}
        ],
        f"{_SV_B}|country/SLV": [
            {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "provenanceId": _SOURCE}
        ],
    }

    class _CandidatesBindLLM:
        """Elect the first shape and flag the second as also plausible."""

        async def generate_structured(self, *, prompt, system, schema):
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[],
                also_plausible=[1],
                ask=None,
            ), None

    def test_every_candidate_where_binding_carries_both_places(self):
        graph = FakeGraph(
            nodes=self._NODES,
            detect=self._DETECT,
            resolve=self._RESOLVE,
            obs=self._OBS,
        )
        result = asyncio.run(
            resolve_variable(
                self._DETECT_QUERY,
                entities=["Guatemala", "El Salvador"],
                date_request=None,
                detect_query=self._DETECT_QUERY,
                role_query="population Guatemala El Salvador",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=self._CandidatesBindLLM(),
                base_steps=[],
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "candidates", f"expected candidates, got {result.status!r}"
        assert len(result.specs) == 2
        for spec in result.specs:
            _assert_where_is_full_place_set(spec)


def _assert_where_is_full_place_set(spec) -> None:
    where = next(s for s in spec.resolved.slots if s.key.axis == "where")
    assert isinstance(where.binding, BindingSet), (
        f"expected BindingSet for spec {spec.resolved.stat_vars.items[0].ref!r}, "
        f"got {type(where.binding).__name__}"
    )
    assert {v.ref.dcid for v in where.binding.values} == {"country/GTM", "country/SLV"}
