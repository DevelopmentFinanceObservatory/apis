"""Tests entity resolution, place expansion, and batched observation probes."""

from __future__ import annotations

import asyncio
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest

import dc_graph
from entity_resolution import Group, Resolution, Resolved
from entity_resolution.membership import MembershipPolicy
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.core import resolve_async
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.extract import ExtractedEntity, Extraction, ExtractionResult
from search.engine.place import PlaceExpansion, expand_places
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult, resolve_variable
from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.shape import ShapeDraft
from search.engine.source_scope import SourceReference, SourceScope
from search.models import (
    BindingSet,
    BindingValue,
    CoverageExact,
    RawTextInput,
    ResolveRequest,
    ResolveResponse,
    SourceMention,
)
from search.render.no_data import no_data_phrase
from tests.engine._binders import elected_reply
from tests.engine._builders import bound_place_context, subject_place_roles
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, FakeLLM, make_resolver, with_regions

_FAKE_LLM = FakeLLM()

# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_SOURCE = "dc/base/PlaceFixture"
_CRS_PROVENANCE = "provenance/OECDCreditorReportingSystem"


class _FixedBindLLM:
    """Elects shape 0 without bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=None), None


def _resolve(query: str, graph: FakeGraph | None = None) -> ResolveResponse:
    """Runs ``resolve_async`` offline with a fixed extraction date."""
    req = ResolveRequest(input=RawTextInput(query=query))
    g = graph if graph is not None else FakeGraph()
    return asyncio.run(resolve_async(req, graph=g, llm=_FAKE_LLM, sdmx=FakeSdmxClient()))


def _set_binding_dcids(binding: BindingSet) -> list[str]:
    """Returns each value's required dcid in binding order."""
    dcids = []
    for v in binding.values:
        assert v.ref is not None
        dcids.append(v.ref.ref)
    return dcids


class TestPartialEntityResolution:
    def test_unknown_entity_fires_entity_not_resolved(self):
        result = _resolve("health ODA grants from USA to Atlantis")
        inner = result.parts[0]
        assert inner.status == "no_data"
        assert inner.no_data.reason == "entity_not_resolved"

    def test_partial_resolution_fires_entity_not_resolved(self):
        result = _resolve("health ODA from USA to Atlantis")
        inner = result.parts[0]
        assert inner.status == "no_data"
        assert inner.no_data.reason == "entity_not_resolved"


class TestLiveResolveEmptyCandidatesReachesEntityNotResolved:
    """A compliant empty ``/v2/resolve`` response reaches ``entity_not_resolved``.

    The test uses the live client's response parser."""

    def test_live_client_empty_candidates_response_lands_entity_not_resolved(
        self, monkeypatch
    ) -> None:
        def fake_post(self, path, payload, *, deadline=None):
            if path.endswith("/node"):
                return {"data": {dcid: {} for dcid in payload["nodes"]}}
            return {"entities": [{"node": name, "candidates": []} for name in payload["nodes"]]}

        monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
        graph = LiveGraphClient(base="http://fake")

        result = _resolve("health ODA grants from USA to Atlantis", graph=graph)

        inner = result.parts[0]
        assert inner.status == "no_data"
        assert inner.no_data.reason == "entity_not_resolved"


class TestEntityNotResolvedAdvice:
    def test_advises_a_different_place_name_for_a_region_miss(self) -> None:
        assert no_data_phrase("entity_not_resolved", entity_terms=["West Africa"]) == (
            'Could not use "West Africa" as a place for this query. Try a different place name.'
        )


class TestDonorFallback:
    def test_bare_single_entity_becomes_recipient(self):
        result = _resolve("health ODA grants Kenya")
        inner = result.parts[0]
        assert inner.status in ("definite", "no_data")
        if inner.status == "definite":
            spec = inner.spec
            slot_map = {s.key.property.ref: s for s in spec.resolved.slots if s.key.property}
            recipient_slot = slot_map.get("DevelopmentFinanceRecipient")
            assert recipient_slot is not None
            binding = recipient_slot.binding
            assert isinstance(binding, BindingValue)
            ref = binding.value.ref
            assert ref is not None
            assert ref.ref == "country/KEN"


class TestPreResolvedDedup:
    """Multiple variables share one resolver call over the deduplicated entity set."""

    def test_duplicate_entity_resolved_in_one_call(self):
        """Duplicate names are sent to the resolver once."""
        fake_graph = FakeGraph()
        resolver = make_resolver(fake_graph)
        resolve_calls: list[list[str]] = []

        _original_resolve = resolver.resolve

        async def _tracked_resolve(texts: list[str]) -> list[Resolution]:
            resolve_calls.append(list(texts))
            return await _original_resolve(texts)

        resolver.resolve = _tracked_resolve  # ty: ignore[invalid-assignment]

        async def _stub_resolve(variable, **_kwargs):
            return RegionResult(
                variable_text=variable,
                status="no_data",
                specs=(),
                no_data_reason="variable_not_resolved",
                warnings=(),
                timing_by_step={},
            )

        fake_extraction = AsyncMock(
            return_value=(
                ExtractionResult(
                    extraction=Extraction(
                        kind="resolution",
                        variables=["v1", "v2"],
                        entities=[ExtractedEntity(text=t) for t in ("Kenya", "Kenya")],
                        dates=[],
                    ),
                    qualifiers={},
                ),
                None,
            )
        )
        req = ResolveRequest(input=RawTextInput(query="test"))
        with (
            patch("search.engine.entry.raw_text.extract", fake_extraction),
            patch("search.engine.fanout.resolve_variable", new=_stub_resolve),
        ):
            asyncio.run(
                resolve_async(
                    req,
                    graph=fake_graph,
                    resolver=resolver,
                    llm=FakeLLM(),
                    sdmx=FakeSdmxClient(observations={}, availability={}),
                )
            )

        assert resolve_calls == [["Kenya"]]

    def test_unique_entities_resolved_in_the_same_one_call(self):
        """Distinct names share one resolver call."""
        fake_graph = FakeGraph()
        resolver = make_resolver(fake_graph)
        resolve_calls: list[list[str]] = []

        _original_resolve = resolver.resolve

        async def _tracked_resolve(texts: list[str]) -> list[Resolution]:
            resolve_calls.append(list(texts))
            return await _original_resolve(texts)

        resolver.resolve = _tracked_resolve  # ty: ignore[invalid-assignment]

        async def _stub_resolve(variable, **_kwargs):
            return RegionResult(
                variable_text=variable,
                status="no_data",
                specs=(),
                no_data_reason="variable_not_resolved",
                warnings=(),
                timing_by_step={},
            )

        fake_extraction = AsyncMock(
            return_value=(
                ExtractionResult(
                    extraction=Extraction(
                        kind="resolution",
                        variables=["v1", "v2"],
                        entities=[ExtractedEntity(text=t) for t in ("Kenya", "Uganda")],
                        dates=[],
                    ),
                    qualifiers={},
                ),
                None,
            )
        )
        req = ResolveRequest(input=RawTextInput(query="test"))
        with (
            patch("search.engine.entry.raw_text.extract", fake_extraction),
            patch("search.engine.fanout.resolve_variable", new=_stub_resolve),
        ):
            asyncio.run(
                resolve_async(
                    req,
                    graph=fake_graph,
                    resolver=resolver,
                    llm=FakeLLM(),
                    sdmx=FakeSdmxClient(observations={}, availability={}),
                )
            )

        assert resolve_calls == [["Kenya", "Uganda"]]


class TestExpandPlaces:
    """A region's members come from the membership policy."""

    _POLICY = with_regions(
        MembershipPolicy([]),
        {
            "testonia": ["country/T1", "country/T2", "country/T3"],
            "boreal": ["country/B1"],
        },
    )

    def test_each_region_takes_the_policys_countries(self):
        results = expand_places(["testonia", "boreal"], membership=self._POLICY)
        assert results == {
            "testonia": PlaceExpansion(
                children=["country/T1", "country/T2", "country/T3"], child_type="Country"
            ),
            "boreal": PlaceExpansion(children=["country/B1"], child_type="Country"),
        }

    def test_returns_every_member_past_the_place_bound(self, search_settings):
        """Expansion truncates nothing. The bound on the places a Spec binds refuses the
        shape that reads them (`place_context`)."""
        search_settings(max_child_places=2)
        expansion = expand_places(["testonia"], membership=self._POLICY)["testonia"]
        assert expansion is not None
        assert expansion.children == ["country/T1", "country/T2", "country/T3"]

    def test_a_region_the_policy_omits_reads_as_its_node_alone(self):
        """A region the policy omits holds no country, so it reads as its node alone; graph
        containment is not read in its place, and the answer is not degraded."""
        policy = MembershipPolicy(list(self._POLICY), omitted_regions={"antarctica"})
        with recording_degraded_reads() as lost:
            results = expand_places(["antarctica"], membership=policy)
        assert results == {"antarctica": None}
        assert lost.reads == []

    def test_empty_input_expands_nothing(self):
        assert expand_places([], membership=self._POLICY) == {}


# Packaged GDP arcs exercise the normal five-tuple derivation path.
_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_NODE = {
    "label": "Nominal gross domestic product",
    "type": "StatisticalVariable",
    "arcs": {
        "constraintProperties": {
            "nodes": [{"name": "activitySource", "types": ["Property"], "dcid": "activitySource"}]
        },
        "activitySource": {
            "nodes": [{"name": "GrossDomesticProduction", "dcid": "GrossDomesticProduction"}]
        },
        "measuredProperty": {"nodes": [{"name": "amount", "dcid": "amount"}]},
        "measurementQualifier": {"nodes": [{"name": "Nominal", "dcid": "Nominal"}]},
        "statType": {"nodes": [{"name": "measuredValue", "dcid": "measuredValue"}]},
        "typeOf": {"nodes": [{"name": "StatisticalVariable", "dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Nominal gross domestic product"}]},
        "populationType": {"nodes": [{"name": "EconomicActivity", "dcid": "EconomicActivity"}]},
    },
}


class TestPlaceExpansionEndToEnd:
    """Place-set members are probed in one batched call."""

    _COUNTRIES: ClassVar[list[str]] = ["country/T1", "country/T2", "country/T3"]

    @pytest.fixture(autouse=True)
    def _regions(self, declare_regions):
        declare_regions({"testonia": self._COUNTRIES})

    def _graph(self, obs: dict) -> FakeGraph:
        return FakeGraph(
            nodes={
                "testonia": {"label": "Testonia", "type": "Continent", "arcs": {}},
                _GDP_SV: _GDP_NODE,
            },
            obs=obs,
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={"Testonia": {"dcid": "testonia", "type_of": "Continent"}},
            children={"testonia": {"Country": self._COUNTRIES}},
        )

    def test_region_expands_to_per_country_coverage(self):
        obs = {
            f"{_GDP_SV}|country/T1": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
            f"{_GDP_SV}|country/T2": [
                {
                    "earliestDate": "2021",
                    "latestDate": "2021",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Testonia"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Testonia",
                pac=True,
                graph=self._graph(obs),
                resolver=make_resolver(self._graph(obs)),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "definite"
        spec = result.specs[0]
        assert spec.resolved.members[0].ref.ref == _GDP_SV
        # The region's node holds no observation, so the Spec binds its member countries,
        # and coverage counts those that hold data.
        assert [e.ref for e in spec.resolved.entities] == self._COUNTRIES
        assert isinstance(spec.resolved.coverage, CoverageExact)
        assert spec.resolved.coverage.dimensions is not None
        countries_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "countries")
        assert countries_dim.count == 2

    def test_region_multi_facet_country_reports_distinct_count_not_facet_count(self):
        """The countries dimension counts each country once across multiple facets."""
        obs = {
            f"{_GDP_SV}|country/T1": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                },
                {
                    "earliestDate": "2019",
                    "latestDate": "2019",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                },
            ],
            f"{_GDP_SV}|country/T2": [
                {
                    "earliestDate": "2021",
                    "latestDate": "2021",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Testonia"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Testonia",
                pac=True,
                graph=self._graph(obs),
                resolver=make_resolver(self._graph(obs)),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "definite"
        spec = result.specs[0]
        assert isinstance(spec.resolved.coverage, CoverageExact)
        assert spec.resolved.coverage.dimensions is not None
        countries_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "countries")
        assert countries_dim.count == 2

    def test_region_with_no_member_data_is_no_data_not_a_crash(self):
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Testonia"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Testonia",
                pac=True,
                graph=self._graph(obs={}),
                resolver=make_resolver(self._graph(obs={})),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"

    def test_query_place_unknown_to_both_resolve_entity_and_resolve_place(self):
        """A place name unresolved by both resolvers returns ``entity_not_resolved``."""
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Republic of Atlantis"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Republic of Atlantis",
                pac=True,
                graph=self._graph(obs={}),
                resolver=make_resolver(self._graph(obs={})),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"

    @pytest.mark.parametrize(
        "source_scope",
        [
            None,
            SourceScope(
                allowed=frozenset({_CRS_PROVENANCE}),
                named=True,
                mentions=(SourceMention(text="CRS", ref=_CRS_PROVENANCE, polarity="include"),),
                references=(SourceReference(ref=_CRS_PROVENANCE, label="CRS", kind="provenance"),),
                expansions=(frozenset({_CRS_PROVENANCE}),),
            ),
            SourceScope(
                allowed=frozenset({_SOURCE}),
                named=False,
                mentions=(SourceMention(text="CRS", ref=_CRS_PROVENANCE, polarity="exclude"),),
                references=(SourceReference(ref=_CRS_PROVENANCE, label="CRS", kind="provenance"),),
                expansions=(frozenset({_CRS_PROVENANCE}),),
            ),
        ],
        ids=["no-source-scope", "included-source", "excluded-source"],
    )
    def test_named_entity_miss_refuses_before_shape_and_bind(self, source_scope):
        """An unresolved place refuses before shape and bind and retains its source scope."""
        graph = FakeGraph(
            nodes={_GDP_SV: _GDP_NODE},
            obs={},
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children={"Earth": {"Country": ["country/T1"]}},
        )
        llm = _FixedBindLLM()

        with (
            patch.object(llm, "generate_structured", wraps=llm.generate_structured) as bind_call,
            patch(
                "search.engine.region.materialise_step.materialise",
                side_effect=AssertionError("materialise must not run for a named entity miss"),
            ),
        ):
            result: RegionResult = asyncio.run(
                resolve_variable(
                    "gdp",
                    entities=["Republic of Zedonia"],
                    date_request=None,
                    detect_query="gdp",
                    role_query="gdp Republic of Zedonia",
                    pac=True,
                    graph=graph,
                    resolver=make_resolver(graph),
                    sdmx=FakeSdmxClient(observations={}, availability={}),
                    llm=llm,
                    base_timing={},
                    source_scope=source_scope,
                    coverage=empty_coverage_snapshot(),
                )
            )

        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"
        assert result.unresolved_entity_names == ("Republic of Zedonia",)
        assert result.source_scope == source_scope
        assert result.settled_dataset is None
        bind_call.assert_not_called()
        assert result.elected_stat_vars is None
        assert not any(warning.code == "query_qualifier_dropped" for warning in result.warnings)
        assert "shape" not in result.timing_by_step
        assert "bind" not in result.timing_by_step

    @pytest.mark.parametrize(
        ("dcid", "graph_type"),
        [("DACAfrica", "DACGroup"), ("qqq/IncomeGroup", "WorldBankIncomeGroup")],
    )
    def test_a_total_or_residual_party_the_graph_confirms_resolves_as_an_aggregate(
        self, dcid: str, graph_type: str
    ):
        """Search's resolver reads the code index production passes, so a party the index
        lists as a total, by its row or by its graph type, is an `aggregate` where its graph
        type alone would class it `other`."""
        graph = FakeGraph(
            nodes={dcid: {"type": graph_type, "label": "Qqq party"}},
            obs={},
            detect={},
            resolve={},
            places={},
            resolve_any={"Qqq party": dcid},
        )

        (resolution,) = asyncio.run(make_resolver(graph).resolve(["Qqq party"]))

        assert isinstance(resolution, Resolved), resolution
        assert resolution.entity.dcid == dcid
        assert resolution.entity.kind == "aggregate"

    def test_least_developed_countries_resolve_from_the_registry_with_no_member_held(self):
        """The membership registry, not the graph, confirms a group and its members, so a
        graph holding none of the Least Developed Countries still resolves the group."""
        graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={}, places={})

        (resolution,) = asyncio.run(make_resolver(graph).resolve(["Least Developed Countries"]))

        assert isinstance(resolution, Group), resolution
        assert resolution.dcid == "groups/UN.LDC"
        assert resolution.member_count == 44

    def test_empty_entity_list_still_reaches_default_scope_and_materialise(self):
        """No named entity reaches default scope and materialisation."""
        graph = FakeGraph(
            nodes={_GDP_SV: _GDP_NODE},
            obs={
                f"{_GDP_SV}|country/T1": [
                    {
                        "earliestDate": "2020",
                        "latestDate": "2020",
                        "obsCount": 1,
                        "provenanceId": _SOURCE,
                    }
                ]
            },
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            children={"Earth": {"Country": ["country/T1"]}},
        )
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=[],
                date_request=None,
                detect_query="gdp",
                role_query="gdp",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=_FixedBindLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert any(w.code == "place_scope_defaulted" for w in result.warnings)
        assert "materialise" in result.timing_by_step

    def test_unresolvable_variable_and_place_reports_entity_not_resolved(self):
        """An unresolvable place outranks an unrecognised variable."""
        graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={}, places={})
        result: RegionResult = asyncio.run(
            resolve_variable(
                "left-handedness rate",
                entities=["Republic of Atlantis"],
                date_request=None,
                detect_query="left-handedness rate",
                role_query="left-handedness rate Republic of Atlantis",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"

    def test_ungeoregion_name_resolves_and_expands_end_to_end(self):
        """A ``UNGeoRegion`` resolves and, holding no observation at its node, binds the
        countries the membership policy holds for it, whatever its graph children are."""
        obs = {
            f"{_GDP_SV}|country/ZAF": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        graph = FakeGraph(
            nodes={
                _GDP_SV: _GDP_NODE,
                "SouthernAfrica": {"label": "Southern Africa", "type": "UNGeoRegion"},
            },
            obs=obs,
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children={"SouthernAfrica": {"Country": self._COUNTRIES}},
        )
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Southern Africa"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Southern Africa",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "definite"
        spec = result.specs[0]
        assert [e.ref for e in spec.resolved.entities] == [
            "country/BWA",
            "country/LSO",
            "country/NAM",
            "country/SWZ",
            "country/ZAF",
        ]

    def test_partial_entity_resolution_names_only_unresolved_ordered_and_deduped(self):
        """Unresolved names appear once each in query order."""
        obs = {
            f"{_GDP_SV}|country/T1": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        graph = FakeGraph(
            nodes={
                _GDP_SV: _GDP_NODE,
                "SouthernAfrica": {"label": "Southern Africa", "type": "UNGeoRegion"},
            },
            obs=obs,
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children={"SouthernAfrica": {"Country": self._COUNTRIES}},
        )
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=[
                    "Republic of Zedonia",
                    "Southern Africa",
                    "Republic of Atlantis",
                    "Republic of Zedonia",
                ],
                date_request=None,
                detect_query="gdp",
                role_query=("gdp Republic of Zedonia Southern Africa Republic of Atlantis"),
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"
        assert result.unresolved_entity_names == ("Republic of Zedonia", "Republic of Atlantis")

    def test_european_union_absent_from_the_graph_resolves_to_its_registry_group(self):
        """The European Union is a group the membership registry confirms, so a graph holding
        nothing still resolves it, and the request fails on its variable, not its entity."""
        graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={}, places={})

        (resolution,) = asyncio.run(make_resolver(graph).resolve(["European Union"]))

        assert isinstance(resolution, Group), resolution
        assert resolution.dcid == "EuropeanUnion"
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["European Union"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp European Union",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"

    def test_bare_arc_match_to_a_non_place_node_counts_as_resolved_in_this_slice(self):
        """A named non-place node counts as resolved.

        With no recalled variable, the result is ``variable_not_resolved``."""
        graph = FakeGraph(
            nodes={
                "sv/SomeIndicator": {"label": "Some Indicator Name", "type": "StatisticalVariable"}
            },
            obs={},
            detect={},
            resolve={},
            places={},
            resolve_any={"Some Indicator Name": "sv/SomeIndicator"},
        )
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Some Indicator Name"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Some Indicator Name",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"

    def test_mixed_unknown_and_uncovered_type_names_report_entity_not_resolved(self):
        """A NotFound and a NotHeld result combine to ``entity_not_resolved``."""
        graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={}, places={})
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["European Union", "Republic of Atlantis"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp European Union Republic of Atlantis",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"

    def test_mapped_region_empty_membership_is_not_entity_not_resolved(self):
        """A resolved region with no members reaches ordinary no-observations handling."""
        graph = FakeGraph(
            nodes={
                _GDP_SV: _GDP_NODE,
                "emptyregion": {"label": "Empty Region", "type": "UNGeoRegion"},
            },
            obs={},
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={"Empty Region": {"dcid": "emptyregion", "type_of": "UNGeoRegion"}},
            children={"emptyregion": {"Country": []}},
        )
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Empty Region"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Empty Region",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "no_data"
        assert result.no_data_reason != "entity_not_resolved"


class _CountingChildPlacesGraph(FakeGraph):
    """Records arguments for each ``child_places_batch`` call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.child_places_batch_calls: list[tuple[list[str], str]] = []

    def child_places_batch(
        self,
        parent_dcids: list[str],
        child_type: str,
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        self.child_places_batch_calls.append((list(parent_dcids), child_type))
        return super().child_places_batch(parent_dcids, child_type)


class TestMultiPlaceSetExpansion:
    """Named place sets take their members from the membership policy, and no graph
    containment is read for them."""

    _TESTONIA_COUNTRIES: ClassVar[list[str]] = ["country/T1", "country/T2"]
    _TESTLANDIA_COUNTRIES: ClassVar[list[str]] = ["country/L1", "country/L2"]

    @pytest.fixture(autouse=True)
    def _regions(self, declare_regions):
        declare_regions(
            {"testonia": self._TESTONIA_COUNTRIES, "testlandia": self._TESTLANDIA_COUNTRIES}
        )

    def _graph(self, **kwargs) -> _CountingChildPlacesGraph:
        defaults = {
            "nodes": {
                "testonia": {"label": "Testonia", "type": "Continent", "arcs": {}},
                "testlandia": {"label": "Testlandia", "type": "Continent", "arcs": {}},
                _GDP_SV: _GDP_NODE,
            },
            "obs": {},
            "detect": {"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            "resolve": {},
            "places": {
                "Testonia": {"dcid": "testonia", "type_of": "Continent"},
                "Testlandia": {"dcid": "testlandia", "type_of": "Continent"},
            },
            "children": {
                "testonia": {"Country": self._TESTONIA_COUNTRIES},
                "testlandia": {"Country": self._TESTLANDIA_COUNTRIES},
            },
        }
        defaults.update(kwargs)
        return _CountingChildPlacesGraph(**defaults)

    def test_two_regions_named_in_one_query_read_no_graph_containment(self):
        graph = self._graph()
        asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Testonia", "Testlandia"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Testonia Testlandia",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert graph.child_places_batch_calls == []

    def test_a_country_and_a_region_read_no_graph_containment(self):
        # Kenya resolves as a country. Testlandia expands through the membership policy.
        graph = self._graph(resolve={"Kenya": "country/KEN"})
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Kenya", "Testlandia"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Kenya Testlandia",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status in ("definite", "no_data")
        assert graph.child_places_batch_calls == []

    def test_two_place_query_binds_a_set_in_query_order(self):
        """The where set holds Kenya and the members of Testlandia, whose node holds no
        observation, in place order."""
        graph = self._graph(
            resolve={"Kenya": "country/KEN"},
            nodes={
                "testonia": {"label": "Testonia", "type": "Continent", "arcs": {}},
                "testlandia": {"label": "Testlandia", "type": "Continent", "arcs": {}},
                "country/KEN": {"label": "Kenya", "type": "Country"},
                _GDP_SV: _GDP_NODE,
            },
            obs={
                f"{_GDP_SV}|country/KEN": [
                    {
                        "earliestDate": "2020",
                        "latestDate": "2020",
                        "obsCount": 1,
                        "provenanceId": _SOURCE,
                    }
                ]
            },
        )
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Kenya", "Testlandia"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Kenya Testlandia",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "definite", f"expected definite, got {result.status!r}"
        assert {e.ref for e in result.specs[0].resolved.entities} == {
            "country/KEN",
            "country/L1",
            "country/L2",
        }
        where_slot = next(s for s in result.specs[0].resolved.slots if s.key.axis == "where")
        assert isinstance(where_slot.binding, BindingSet)
        assert _set_binding_dcids(where_slot.binding) == ["country/KEN", "country/L1", "country/L2"]

    def test_two_place_query_finds_data_regardless_of_naming_order(self):
        """Kenya's data resolves when the data-less region appears first."""
        graph = self._graph(
            resolve={"Kenya": "country/KEN"},
            nodes={
                "testonia": {"label": "Testonia", "type": "Continent", "arcs": {}},
                "testlandia": {"label": "Testlandia", "type": "Continent", "arcs": {}},
                "country/KEN": {"label": "Kenya", "type": "Country"},
                _GDP_SV: _GDP_NODE,
            },
            obs={
                f"{_GDP_SV}|country/KEN": [
                    {
                        "earliestDate": "2020",
                        "latestDate": "2020",
                        "obsCount": 1,
                        "provenanceId": _SOURCE,
                    }
                ]
            },
        )
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Testlandia", "Kenya"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Testlandia Kenya",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert result.status == "definite", f"expected definite, got {result.status!r}"
        where_slot = next(s for s in result.specs[0].resolved.slots if s.key.axis == "where")
        assert isinstance(where_slot.binding, BindingSet)
        # Places are stated in the one place order, by dcid, whatever order they were named.
        assert _set_binding_dcids(where_slot.binding) == ["country/KEN", "country/L1", "country/L2"]

    def test_regions_over_the_place_bound_answer_entity_set_too_large(self, search_settings):
        """Two regions whose members exceed the bound are expanded from the membership
        policy and refused whole, and no member list is truncated."""
        search_settings(max_child_places=2)
        graph = self._graph()
        result: RegionResult = asyncio.run(
            resolve_variable(
                "gdp",
                entities=["Testonia", "Testlandia"],
                date_request=None,
                detect_query="gdp",
                role_query="gdp Testonia Testlandia",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=FakeLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert graph.child_places_batch_calls == []
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_set_too_large"
        assert result.specs == ()


class _CountingCachingGraph(FakeGraph):
    """Caches observation probes and counts simulated network requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe_cache: dict[tuple[str, str, bool], list] = {}
        self.post_count = 0

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, deadline=None):
        result: dict[str, dict[str, list]] = {sv: {} for sv in stat_vars}
        miss_pairs: set[tuple[str, str]] = set()
        for sv in stat_vars:
            for entity in entities:
                key = (sv, entity, needs_dates)
                if key in self._probe_cache:
                    result[sv][entity] = self._probe_cache[key]
                else:
                    miss_pairs.add((sv, entity))
        if not miss_pairs:
            return result
        self.post_count += 1
        miss_svs = list(dict.fromkeys(sv for sv, _ in miss_pairs))
        miss_entities = list(dict.fromkeys(e for _, e in miss_pairs))
        fresh = super().observation_facets_batch(miss_svs, miss_entities, needs_dates=needs_dates)
        for sv in miss_svs:
            for entity in miss_entities:
                facets = fresh.get(sv, {}).get(entity, [])
                self._probe_cache[(sv, entity, needs_dates)] = facets
                result[sv][entity] = facets
        return result


class TestPlaceExpansionBatchingGuard:
    """A place set issues one observation POST across all member countries."""

    _COUNTRIES: ClassVar[list[str]] = [f"country/T{i}" for i in range(10)]  # a 10-country region

    def _shape(self) -> ShapeDraft:
        return ShapeDraft(
            shape_id="gdp_shape",
            label="gdp_shape",
            pop_type_dcid="EconomicActivity",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            sv_arc_facts={"gdp_sv": {}},
            representative_score=0.9,
        )

    def test_region_expansion_issues_one_post(self):
        obs = {
            f"gdp_sv|{c}": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ]
            for c in self._COUNTRIES[:5]
        }
        graph = _CountingCachingGraph(nodes={}, obs=obs, detect={}, resolve={})

        result = materialise(
            self._shape(),
            [],
            {},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            date_request=None,
            place_dcids=self._COUNTRIES,
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert graph.post_count == 1, (
            f"expected exactly one batched observation POST for a 10-country "
            f"region expansion; got {graph.post_count}"
        )
        assert result.has_data is True
        assert isinstance(result.coverage, CoverageExact)
        assert result.coverage.dimensions is not None
        countries_dim = next(d for d in result.coverage.dimensions if d.label == "countries")
        assert countries_dim.count == 5


class TestAltShapesBatchingGuard:
    """Alternate shapes share one batched observation POST."""

    _RECIPIENT = "country/USA"

    def _shape(self, shape_id: str, sv: str) -> ShapeDraft:
        return ShapeDraft(
            shape_id=shape_id,
            label=shape_id,
            pop_type_dcid="EconomicActivity",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            sv_arc_facts={sv: {}},
            representative_score=0.9,
        )

    def test_multi_alt_shape_fan_out_issues_one_post(self):
        from tests.engine._builders import materialise_alternates

        n_shapes = 4
        shapes = [self._shape(f"shape_{i}", f"sv_{i}") for i in range(n_shapes)]
        obs = {
            f"sv_{i}|{self._RECIPIENT}": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ]
            for i in range(n_shapes)
        }
        graph = _CountingCachingGraph(nodes={}, obs=obs, detect={}, resolve={})

        results, unsupported = materialise_alternates(
            shapes,
            places=[bound_place_context(subject_place_roles(self._RECIPIENT))] * len(shapes),
            elected_bindings=[],
            elected_has_data=True,
            named_place_dcids=[self._RECIPIENT],
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            date_request=None,
            coverage=empty_coverage_snapshot(),
        )

        assert unsupported is None
        assert len(results) == n_shapes, (
            f"expected all {n_shapes} alt shapes to materialise data; got {len(results)}"
        )
        assert graph.post_count == 1, (
            f"expected exactly one batched observation POST for a "
            f"{n_shapes}-alt-shape fan-out; got {graph.post_count}"
        )
