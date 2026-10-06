"""Tests shape derivation when StatVars lack schema-defining arcs."""

from __future__ import annotations

import asyncio
import logging

import pytest

from search.engine.errors import RoleSignatureError
from search.engine.region.pipeline import resolve_variable
from search.engine.retrieval.derive import derive_shapes
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver


class TestDeriveShapesZeroMemberGroup:
    """Zero-member groups cannot form identity shapes and are skipped."""

    def _zero_member_arcs(self) -> dict:
        """Builds a confirmable node with no schema arcs."""
        return {"name": {"nodes": [{"value": "Untyped fixture node"}]}}

    def test_zero_member_group_alongside_a_real_group_is_skipped(self):
        """A zero-member group drops out beside a valid group."""
        zero_member_sv = "Test_ZeroMemberAlongsideReal"
        real_sv = "Count_Person_ZeroMemberSibling"
        nodes = {
            zero_member_sv: {"label": "Untyped", "arcs": self._zero_member_arcs()},
            real_sv: {
                "label": "Population",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[zero_member_sv, real_sv], graph=graph)

        assert len(shapes) == 1
        assert shapes[0].pop_type_dcid == "Person"

    def test_zero_member_group_logs_and_is_not_counted_as_mixed(self, caplog):
        """The skip logs its SV at debug level and increments ``zero_member_groups``."""
        zero_member_sv = "Test_ZeroMemberLogged"
        real_sv = "Count_Person_ZeroMemberLoggedSibling"
        nodes = {
            zero_member_sv: {"label": "Untyped", "arcs": self._zero_member_arcs()},
            real_sv: {
                "label": "Population",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        with caplog.at_level(logging.DEBUG, logger="search.engine.retrieval.derive"):
            shapes = derive_shapes(confirmed_svs=[zero_member_sv, real_sv], graph=graph)

        assert len(shapes) == 1
        skip_records = [r for r in caplog.records if "zero-member shape group" in r.getMessage()]
        assert len(skip_records) == 1
        assert zero_member_sv in skip_records[0].getMessage()

    def test_every_candidate_zero_member_returns_empty_list(self):
        """An all-zero-member pool produces no shapes."""
        sv_a = "Test_ZeroMemberOnlyA"
        sv_b = "Test_ZeroMemberOnlyB"
        nodes = {
            sv_a: {"label": "Untyped A", "arcs": self._zero_member_arcs()},
            sv_b: {"label": "Untyped B", "arcs": self._zero_member_arcs()},
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[sv_a, sv_b], graph=graph)

        assert shapes == []

    def test_unresolved_denominator_alone_is_not_zero_member(self):
        """An unresolved denominator supplies a schema member and preserves the group."""
        sv_dcid = "Test_UnresolvedDenominatorOnly"
        nodes = {
            sv_dcid: {
                "label": "Unresolved-denominator-only fixture",
                "arcs": {
                    "measurementDenominator": {
                        "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
                    },
                },
            }
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert len(shapes) == 1
        assert shapes[0].meas_denom_unresolved is True


class TestDeriveShapesZeroMemberGroupMixedWithRoleMismatch:
    """Role-signature errors count only groups with schema members."""

    def _zero_member_arcs(self) -> dict:
        return {"name": {"nodes": [{"value": "Untyped fixture node"}]}}

    def _person_arcs(self, *, observation_properties: list[str] | None = None) -> dict:
        arcs: dict = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        if observation_properties is not None:
            arcs["observationProperties"] = {"nodes": [{"dcid": p} for p in observation_properties]}
        return arcs

    def _econ_arcs(self, *, observation_properties: list[str] | None = None) -> dict:
        arcs: dict = {
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        if observation_properties is not None:
            arcs["observationProperties"] = {"nodes": [{"dcid": p} for p in observation_properties]}
        return arcs

    def test_zero_member_plus_mismatched_group_raises_counting_only_the_viable_group(self):
        """The error counts only the group whose roles share a flow axis."""
        zero_member_sv = "Test_ZeroMemberPlusMismatchedOnly"
        mixed_a = "Count_Person_ZeroPlusMismatchedA"
        mixed_b = "Count_Person_ZeroPlusMismatchedB"
        nodes = {
            zero_member_sv: {"label": "Untyped", "arcs": self._zero_member_arcs()},
            mixed_a: {
                "label": "Mixed A",
                "arcs": self._person_arcs(observation_properties=["debtor"]),
            },
            mixed_b: {
                "label": "Mixed B",
                "arcs": self._person_arcs(observation_properties=["provider", "recipient"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        with pytest.raises(RoleSignatureError, match=r"all 1 viable shape group"):
            derive_shapes(confirmed_svs=[zero_member_sv, mixed_a, mixed_b], graph=graph)

    def test_zero_member_plus_two_role_classes_plus_another_group_builds_three_shapes(self):
        """Each role class and the agreeing group build a shape beside a zero-member group."""
        zero_member_sv = "Test_ZeroMemberPlusMismatchedSurviving"
        mixed_a = "Count_Person_ZeroPlusSurvivingMixedA"
        mixed_b = "Count_Person_ZeroPlusSurvivingMixedB"
        agree_a = "Amount_EconomicActivity_ZeroPlusSurvivingAgree"
        nodes = {
            zero_member_sv: {"label": "Untyped", "arcs": self._zero_member_arcs()},
            mixed_a: {"label": "Mixed A", "arcs": self._person_arcs()},
            mixed_b: {
                "label": "Mixed B",
                "arcs": self._person_arcs(observation_properties=["provider", "recipient"]),
            },
            agree_a: {"label": "Agree A", "arcs": self._econ_arcs()},
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(
            confirmed_svs=[zero_member_sv, mixed_a, mixed_b, agree_a], graph=graph
        )

        assert {shape.shape_id: list(shape.sv_arc_facts or {}) for shape in shapes} == {
            "person_count_measuredvalue": [mixed_a],
            "person_count_measuredvalue_by_role": [mixed_b],
            "economicactivity_amount_measuredvalue": [agree_a],
        }


class _UnusedBindLLM:
    """Raises if binding runs after an empty shape derivation."""

    async def generate_structured(self, *, prompt, system, schema):
        raise AssertionError("bind's LLM must not be called when derive_shapes returns []")


class TestDeriveShapesZeroMemberGroupPipelineOutcome:
    """An empty shape derivation returns ``variable_not_resolved`` before binding."""

    def test_zero_member_only_recall_reports_variable_not_resolved(self):
        zero_member_sv = "Test_ZeroMemberPipelineOnly"
        nodes = {
            zero_member_sv: {
                "label": "Untyped",
                "arcs": {"name": {"nodes": [{"value": "Untyped fixture node"}]}},
            },
        }
        graph = FakeGraph(
            nodes=nodes,
            detect={"pre-primary enrollment": {"svs": [zero_member_sv], "cosine_scores": [0.9]}},
            resolve={},
            obs={},
        )

        result = asyncio.run(
            resolve_variable(
                "pre-primary enrollment",
                entities=[],
                date_request=None,
                detect_query="pre-primary enrollment",
                role_query="pre-primary enrollment",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_UnusedBindLLM(),
                sdmx=FakeSdmxClient(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"
