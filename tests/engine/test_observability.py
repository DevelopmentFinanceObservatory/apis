"""Tests for resolution observability fields and their effect on `spec_id`."""

from __future__ import annotations

import pytest

from search.engine.assemble import build_no_data_part
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.conjoin import assemble_region
from search.engine.envelope import make_diagnostics, now_ms
from search.engine.region import RegionResult
from search.engine.retrieval.read_plan import plan_standard
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import (
    build_shape_model,
    build_slot,
    build_spec,
    build_stat_vars,
)
from search.models import (
    BindingSet,
    BindingValue,
    CoverageBare,
    Diagnostics,
    Entity,
    EntityRoleSubject,
    GraphRef,
    NearestRealSpec,
    QueryEcho,
    RawTextInput,
    Relaxation,
    ResolveRequest,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
    Timing,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve


def _query_echo() -> QueryEcho:
    return QueryEcho(
        entry_path="raw_text",
        raw_query="test",
        normalized_query="test",
        extract_skipped=False,
    )


def _diagnostics(**kw) -> Diagnostics:
    return Diagnostics(
        engine_build="test", trace_id="test-trace", warnings=[], timing_ms=Timing(total=0), **kw
    )


def _sv_ref(dcid: str) -> GraphRef:
    return GraphRef(dcid=dcid, label=dcid)


def _facet(
    earliest: str | None,
    latest: str | None,
    *,
    facet_id: str = "f1",
    provenance_id: str = "dc/base/Source",
    obs_count: int = 10,
) -> Facet:
    return Facet(
        earliest_date=earliest,
        latest_date=latest,
        obs_count=obs_count,
        facet_id=facet_id,
        provenance_id=provenance_id,
    )


def _plan(facets_by_sv: dict[str, list[Facet]]):
    """A dateless plan reading each StatVar's facets at one place."""
    return plan_standard(
        {(sv, "country/KEN"): facets for sv, facets in facets_by_sv.items()},
        members=list(facets_by_sv),
        years=None,
    )


def _to_nearest_real(spec) -> NearestRealSpec:
    return NearestRealSpec(
        spec_id=spec.spec_id,
        definition=spec.definition,
        resolved=spec.resolved,
        relaxation=Relaxation(
            axis="what", property_dcid="DevelopmentFinanceScheme", dropped_value_dcids=["ODAGrants"]
        ),
    )


class TestLLMUsage:
    def test_passes_through_explicit_usage(self):
        """make_diagnostics carries an explicit llm_usage into Diagnostics."""
        usage = {"input_tokens": 100, "output_tokens": 50, "cached_tokens": 10}
        d = make_diagnostics("build", [], {}, 10, llm_usage=usage)
        assert d.llm_usage == usage

    def test_none_when_usage_omitted(self):
        """The no-LLM early-return paths omit llm_usage and get None."""
        d = make_diagnostics("build", [], {}, 10)
        assert d.llm_usage is None


def _minimal_spec():
    shape_draft = ShapeDraft(
        shape_id="test_minimal_shape",
        label="test shape",
        pop_type_dcid="DevelopmentFinance",
        meas_prop_dcid="DevelopmentFinanceFlow",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="what", property_dcid="DevelopmentFinanceScheme", label="scheme"),
        ),
    )
    five_tuple_refs = {
        "DevelopmentFinance": GraphRef(dcid="DevelopmentFinance", label="Dev Finance"),
        "DevelopmentFinanceFlow": GraphRef(dcid="DevelopmentFinanceFlow", label="Flow"),
        "measuredValue": GraphRef(dcid="measuredValue", label="Measured Value"),
    }
    scheme_draft = SlotBindingDraft(
        axis="what",
        property_dcid="DevelopmentFinanceScheme",
        kind="value",
        value_dcids=["ODAGrants"],
    )
    prop_ref = GraphRef(dcid="DevelopmentFinanceScheme", label="DevelopmentFinanceScheme")
    slot = build_slot(
        shape_draft.slot_keys[0],
        scheme_draft,
        [GraphRef(dcid="ODAGrants", label="ODA Grants")],
        property_ref=prop_ref,
    )
    shape_model = build_shape_model(
        shape_draft, [slot.key], five_tuple_refs, member_count=1, members=None
    )
    sv_ref = GraphRef(dcid="TEST_SV", label="Test SV")
    stat_vars = [
        StatVar(ref=sv_ref, shape_id=shape_draft.shape_id, slot_values=[], constraint_properties=[])
    ]
    entity = Entity(
        ref=GraphRef(dcid="country/ETH", label="Ethiopia"),
        entity_type=None,
        role=EntityRoleSubject(),
    )
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape_model,
        slots=[slot],
        stat_vars=stat_vars,
        entities=[entity],
        coverage=CoverageBare(has_data=True),
    )


class TestDataDateRangeAndDestination:
    def test_date_range_spans_the_served_facets(self):
        """Two facets from one source serve together, so the range runs from the earlier
        start to the later end."""
        stat_vars = build_stat_vars(
            sv_refs=[_sv_ref("SV_A")],
            shape_id="shape1",
            slots=[],
            plan=_plan({"SV_A": [_facet("2010", "2015"), _facet("2012", "2020", facet_id="f2")]}),
            arcs_by_sv={},
        )
        assert len(stat_vars) == 1
        dr = stat_vars[0].data_date_range
        assert dr is not None
        assert (dr.start, dr.end) == ("2010", "2020")

    def test_date_range_leaves_out_a_facet_the_plan_does_not_serve(self):
        """The plan serves the larger facet; another source's earlier facet does not date
        the StatVar."""
        stat_vars = build_stat_vars(
            sv_refs=[_sv_ref("SV_A")],
            shape_id="shape1",
            slots=[],
            plan=_plan(
                {
                    "SV_A": [
                        _facet("2018", "2020", obs_count=30),
                        _facet("2000", "2004", facet_id="f2", provenance_id="dc/base/Other"),
                    ]
                }
            ),
            arcs_by_sv={},
        )
        dr = stat_vars[0].data_date_range
        assert dr is not None
        assert (dr.start, dr.end) == ("2018", "2020")

    def test_date_range_none_without_a_plan(self):
        """An answer with no plan reads no facet, so data_date_range stays None."""
        sv_ref = _sv_ref("SV_A")
        stat_vars = build_stat_vars(sv_refs=[sv_ref], shape_id="s", slots=[], arcs_by_sv={})
        assert stat_vars[0].data_date_range is None

    def test_date_range_none_when_the_plan_serves_nothing_of_the_sv(self):
        """A SV the plan serves no series of gets date_range=None."""
        sv_ref = _sv_ref("SV_ABSENT")
        stat_vars = build_stat_vars(
            sv_refs=[sv_ref],
            shape_id="s",
            slots=[],
            plan=_plan({"SV_OTHER": [_facet("2010", "2020")]}),
            arcs_by_sv={},
        )
        assert stat_vars[0].data_date_range is None

    def test_destination_confirmed_true(self):
        """SV in destination_confirmed set → data_confirmed_at_destination=True."""
        sv_ref = _sv_ref("SV_A")
        stat_vars = build_stat_vars(
            sv_refs=[sv_ref],
            shape_id="s",
            slots=[],
            destination_confirmed={"SV_A", "SV_B"},
            arcs_by_sv={},
        )
        assert stat_vars[0].data_confirmed_at_destination is True

    def test_destination_confirmed_false_for_origin_only(self):
        """SV absent from destination_confirmed set → data_confirmed_at_destination=False."""
        sv_ref = _sv_ref("SV_ORIGIN_ONLY")
        stat_vars = build_stat_vars(
            sv_refs=[sv_ref],
            shape_id="s",
            slots=[],
            destination_confirmed={"SV_DIFFERENT"},
            arcs_by_sv={},
        )
        assert stat_vars[0].data_confirmed_at_destination is False

    def test_destination_confirmed_none_when_not_checked(self):
        """When destination_confirmed is None, data_confirmed_at_destination stays None."""
        sv_ref = _sv_ref("SV_A")
        stat_vars = build_stat_vars(sv_refs=[sv_ref], shape_id="s", slots=[], arcs_by_sv={})
        assert stat_vars[0].data_confirmed_at_destination is None

    def test_date_range_partial_dates(self):
        """If only earliest_date is set, end is None (and vice versa)."""
        sv_ref = _sv_ref("SV_A")
        stat_vars = build_stat_vars(
            sv_refs=[sv_ref],
            shape_id="s",
            slots=[],
            plan=_plan({"SV_A": [_facet("2015", None)]}),
            arcs_by_sv={},
        )
        dr = stat_vars[0].data_date_range
        assert dr is not None
        assert dr.start == "2015"
        assert dr.end is None


def _constraint_arcs(prop: str, value: str) -> dict:
    """Build one constraint-property/value pair in `node_arcs` form."""
    return {
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value}]},
    }


def _value_slot(prop_dcid: str, value_dcid: str) -> Slot:
    return Slot(
        key=SlotKey(
            axis="what", property=GraphRef(dcid=prop_dcid, label=prop_dcid), label=prop_dcid
        ),
        binding=BindingValue(
            value=SlotValue(
                ref=GraphRef(dcid=value_dcid, label=value_dcid), value_kind="enum_value"
            )
        ),
    )


class TestBuildStatVarsSlotValuesArcDerived:
    def test_only_the_asserting_stat_var_carries_the_bound_value(self):
        """Only StatVars whose arcs assert a bound value include it in `slot_values`."""
        slot = _value_slot("gender", "schema:Male")
        arcs_by_sv = {
            "Count_Person_Female": _constraint_arcs("gender", "schema:Male"),
            "Count_Person_Urban": _constraint_arcs("residenceType", "schema:Urban"),
        }
        stat_vars = build_stat_vars(
            sv_refs=[_sv_ref("Count_Person_Female"), _sv_ref("Count_Person_Urban")],
            shape_id="s",
            slots=[slot],
            arcs_by_sv=arcs_by_sv,
        )
        by_ref = {sv.ref.dcid: sv for sv in stat_vars}
        assert [v.value.ref.dcid for v in by_ref["Count_Person_Female"].slot_values] == [
            "schema:Male"
        ]
        assert by_ref["Count_Person_Urban"].slot_values == []

    def test_set_binding_contributes_only_the_member_each_stat_var_asserts(self):
        """Each StatVar carries only its asserted member from a shared set binding."""
        slot = Slot(
            key=SlotKey(
                axis="how",
                property=GraphRef(dcid="financingSource", label="financingSource"),
                label="financingSource",
            ),
            binding=BindingSet(
                values=[
                    SlotValue(
                        ref=GraphRef(dcid="SourceA", label="SourceA"), value_kind="enum_value"
                    ),
                    SlotValue(
                        ref=GraphRef(dcid="SourceB", label="SourceB"), value_kind="enum_value"
                    ),
                ]
            ),
        )
        arcs_by_sv = {
            "SV_A": _constraint_arcs("financingSource", "SourceA"),
            "SV_B": _constraint_arcs("financingSource", "SourceB"),
        }
        stat_vars = build_stat_vars(
            sv_refs=[_sv_ref("SV_A"), _sv_ref("SV_B")],
            shape_id="s",
            slots=[slot],
            arcs_by_sv=arcs_by_sv,
        )
        by_ref = {sv.ref.dcid: sv for sv in stat_vars}
        assert [v.value.ref.dcid for v in by_ref["SV_A"].slot_values] == ["SourceA"]
        assert [v.value.ref.dcid for v in by_ref["SV_B"].slot_values] == ["SourceB"]


class TestNearestReal:
    def test_nearest_real_passed_through_build_no_data_part(self):
        """build_no_data_part threads nearest_real into NoData.nearest_real."""
        spec = _minimal_spec()
        part = build_no_data_part(
            "no_observations",
            variable_text="test",
            warnings=[],
            nearest_real=[_to_nearest_real(spec)],
        )
        assert part.status == "no_data"
        assert part.no_data.nearest_real is not None
        assert len(part.no_data.nearest_real) == 1
        assert part.no_data.nearest_real[0].spec_id == spec.spec_id

    def test_nearest_real_none_when_nothing_relaxes(self):
        """build_no_data_part with nearest_real=None leaves NoData.nearest_real as None."""
        part = build_no_data_part(
            "no_observations",
            variable_text="test",
            warnings=[],
            nearest_real=None,
        )
        assert part.status == "no_data"
        assert part.no_data.nearest_real is None

    def test_nearest_real_preserved_through_assemble_region(self):
        """`nearest_real` survives assembly into the no-data response."""
        spec = _minimal_spec()
        region = RegionResult(
            variable_text="test",
            status="no_data",
            specs=(),
            no_data_reason="no_observations",
            warnings=(),
            timing_by_step={},
            nearest_real=(_to_nearest_real(spec),),
        )
        response = assemble_region(
            region,
            query="test",
            extra_warnings=[],
            start_ms=now_ms(),
            engine_build="test-build",
            max_candidates=6,
        )
        inner = response.parts[0]
        assert inner.status == "no_data"
        assert inner.no_data.nearest_real is not None
        assert len(inner.no_data.nearest_real) == 1
        assert inner.no_data.nearest_real[0].spec_id == spec.spec_id


class TestLLMUsageEndToEnd:
    def test_llm_usage_non_none_after_resolve(self):
        """A full resolve carries extract and bind usage into diagnostics."""
        result = offline_resolve(
            ResolveRequest(input=RawTextInput(query="health ODA grants from USA to Ethiopia"))
        )
        diag = result.diagnostics
        assert diag.llm_usage is not None, (
            "llm_usage was None; the extract/bind usage was not threaded through"
        )
        assert set(diag.llm_usage.keys()) == {"input_tokens", "output_tokens", "cached_tokens"}

    def test_llm_usage_non_none_standard_golden(self):
        """The standard-DC path reports extract usage without a bind call."""
        result = offline_resolve(ResolveRequest(input=RawTextInput(query="total population India")))
        diag = result.diagnostics
        assert diag.llm_usage is not None
        assert set(diag.llm_usage.keys()) == {"input_tokens", "output_tokens", "cached_tokens"}


class TestSpecIdUnchanged:
    @pytest.mark.parametrize(
        ("query", "expected_spec_id", "sdmx"),
        [
            pytest.param(
                "fertility rate in Kenya",
                "spec_d76ae772591c93f87a07fffe1fdbc0da",
                FakeSdmxClient(),
                id="fertility_rate_kenya",
            ),
            pytest.param(
                "total population India",
                "spec_f8ff0ca5ad202870904fd51149b171a6",
                None,
                id="total_population_india",
            ),
        ],
    )
    def test_spec_id_pinned(self, query, expected_spec_id, sdmx):
        """Definite goldens retain their exact, role-aware spec identities."""
        result = offline_resolve(ResolveRequest(input=RawTextInput(query=query)), sdmx=sdmx)
        assert result.parts[0].status == "definite"
        spec = result.parts[0].spec
        assert spec.spec_id == expected_spec_id


class TestStatVarObservabilityNonFallbackPaths:
    def test_standard_path_populates_statvar_observability(self):
        """A standard subject probe derives a date range from the facets its plan serves."""
        result = offline_resolve(ResolveRequest(input=RawTextInput(query="total population India")))
        assert result.parts[0].status == "definite"
        stat_vars = result.parts[0].spec.resolved.members
        assert stat_vars, "expected at least one resolved StatVar"
        sv = stat_vars[0]
        assert sv.data_date_range is not None, (
            "standard path read observations but data_date_range came through None"
        )
        assert sv.data_date_range.start is not None
        assert sv.data_confirmed_at_destination is None


class TestNearestRealProbeFailureIsContained:
    def test_probe_graph_error_still_yields_200_no_data(self, monkeypatch):
        import asyncio

        from search.engine.bind import _UniversalBindOutput
        from search.engine.errors import GraphInfraError
        from search.engine.region import resolve_variable
        from search.engine.retrieval.retrieve import NoDataDraft
        from tests.fixtures import FakeGraph, make_resolver

        query = "fertility rate of test subjects"
        sv = "Test_FertilityRate_Person_Female"

        class _ElectSingleShapeLLM:
            """Bind the sole constraint slot for the nearest-real probe."""

            async def generate_structured(self, *, prompt, system, schema):
                assert schema.__name__ == "_UniversalBindOutput"
                return (
                    _UniversalBindOutput(
                        chosen_shape_index=0,
                        bindings=[
                            {
                                "axis": "how",
                                "property_dcid": "gender",
                                "kind": "value",
                                "value_dcids": ["Female"],
                            }
                        ],
                        also_plausible=[],
                        ask=None,
                    ),
                    None,
                )

        class _BatchRaisingGraph(FakeGraph):
            """Raise on the optional facet probe; primary materialisation is stubbed."""

            def observation_facets_batch(self, *_args, **_kwargs):
                raise GraphInfraError("simulated 503 in relaxation probe", upstream_status=503)

        graph = _BatchRaisingGraph(
            nodes={
                "country/TEST": {"label": "Testland", "type": "Country"},
                sv: {
                    "label": "Fertility rate of test subjects",
                    "type": "StatisticalVariable",
                    "arcs": {
                        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                        "name": {"nodes": [{"value": "Fertility rate of test subjects"}]},
                        "populationType": {"nodes": [{"dcid": "Person"}]},
                        "measuredProperty": {"nodes": [{"dcid": "fertilityRate"}]},
                        "statType": {"nodes": [{"dcid": "measuredValue"}]},
                        "constraintProperties": {"nodes": [{"dcid": "gender", "name": "gender"}]},
                        "gender": {"nodes": [{"dcid": "Female", "name": "Female"}]},
                    },
                },
            },
            detect={query: {"svs": [sv], "cosine_scores": [0.9]}},
            resolve={"Testland": "country/TEST"},
            obs={},
        )

        monkeypatch.setattr(
            "search.engine.region.materialise_step.materialise",
            lambda *args, **kwargs: NoDataDraft(reason="no_observations"),
        )

        result = asyncio.run(
            resolve_variable(
                query,
                entities=["Testland"],
                date_request=None,
                detect_query=query,
                role_query=query,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_ElectSingleShapeLLM(),
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "no_data", (
            "probe GraphInfraError leaked and changed the region's status (expected no_data)"
        )
        assert result.no_data_reason == "no_observations"
        assert not result.nearest_real
