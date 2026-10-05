"""Tests declared-role stamps, the split of a five-tuple by role class, and degenerate role arcs."""

from __future__ import annotations

import logging

import pytest

from search.engine.errors import RoleSignatureError
from search.engine.retrieval.derive import derive_shapes
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# derive_shapes: declared_roles stamp
# ---------------------------------------------------------------------------


class TestDeriveShapesDeclaredRoles:
    """A five-tuple's subject-only members and its role-bearing members are separate shapes."""

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

    def test_standard_group_stamps_observation_about(self):
        sv_dcid = "Count_Person_DeclaredRolesDefault"
        nodes = {sv_dcid: {"label": "Population", "arcs": self._person_arcs()}}
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert shapes[0].declared_roles == ("observationAbout",)

    def test_role_bearing_group_stamps_its_declared_roles(self):
        """A neutral test StatVar declares provider and recipient roles."""
        sv_dcid = "Amount_EconomicActivity_RoleBearingFixture"
        arcs = self._person_arcs(observation_properties=["provider", "recipient"])
        nodes = {sv_dcid: {"label": "Role-bearing fixture", "arcs": arcs}}
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert shapes[0].declared_roles == ("provider", "recipient")

    def test_only_the_subject_only_shape_carries_member_coverage(self):
        """The place probe reads subject facets, so a role-bearing member's coverage is
        unknown however the probe answers."""
        subject_sv, role_sv = "Count_Person_SubjectCoverage", "Count_Person_RoleCoverage"
        nodes = {
            subject_sv: {"label": "Subject", "arcs": self._person_arcs()},
            role_sv: {
                "label": "Role",
                "arcs": self._person_arcs(observation_properties=["provider", "recipient"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(
            confirmed_svs=[subject_sv, role_sv], graph=graph, place_dcids=["country/KEN"]
        )

        assert {s.declared_roles: s.member_coverage for s in shapes} == {
            ("observationAbout",): {subject_sv: {}},
            ("provider", "recipient"): None,
        }

    def test_representative_degenerate_arc_raises_role_signature_error(self):
        """A degenerate role arc leaves no viable group and raises ``RoleSignatureError``."""
        sv_dcid = "Count_Person_DegenerateRepresentative"
        arcs = self._person_arcs(observation_properties=["observationAbout"])
        nodes = {sv_dcid: {"label": "Degenerate", "arcs": arcs}}
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        with pytest.raises(RoleSignatureError):
            derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

    def test_degenerate_arc_skips_its_group_and_the_subject_only_members_keep_their_shape(self):
        """A degenerate arc reads as role-bearing, so it leaves the subject-only shape whole."""
        good_sv = "Count_Person_GoodRepresentative"
        bad_sibling_sv = "Count_Person_BadSibling"
        nodes = {
            good_sv: {"label": "Good", "arcs": self._person_arcs()},
            bad_sibling_sv: {
                "label": "Bad sibling",
                "arcs": self._person_arcs(observation_properties=["observationAbout"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[good_sv, bad_sibling_sv], graph=graph)

        assert [(list(s.sv_arc_facts or {}), s.declared_roles) for s in shapes] == [
            ([good_sv], ("observationAbout",))
        ]

    def test_degenerate_arc_logs_error_and_differing_role_classes_log_none(self, caplog):
        """Degenerate role arcs log errors.

        Members of different role classes produce no error logs."""
        bad_sv = "Count_Person_DegenerateLogged"
        nodes = {
            bad_sv: {
                "label": "Degenerate",
                "arcs": self._person_arcs(observation_properties=["observationAbout"]),
            }
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        with (
            caplog.at_level(logging.ERROR, logger="search.engine.retrieval.derive"),
            pytest.raises(RoleSignatureError),
        ):
            derive_shapes(confirmed_svs=[bad_sv], graph=graph)

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        assert bad_sv in error_records[0].getMessage()
        assert "observationAbout" in error_records[0].getMessage()

        caplog.clear()
        mixed_a = "Count_Person_MixedLoggedA"
        mixed_b = "Count_Person_MixedLoggedB"
        mixed_nodes = {
            mixed_a: {"label": "Mixed A", "arcs": self._person_arcs()},
            mixed_b: {
                "label": "Mixed B",
                "arcs": self._person_arcs(observation_properties=["provider", "recipient"]),
            },
        }
        mixed_graph = FakeGraph(nodes=mixed_nodes, obs={}, detect={}, resolve={})

        with caplog.at_level(logging.ERROR, logger="search.engine.retrieval.derive"):
            derive_shapes(confirmed_svs=[mixed_a, mixed_b], graph=mixed_graph)

        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

    def test_subject_only_and_role_bearing_members_of_a_five_tuple_are_two_shapes(self):
        """The subject-only shape keeps the five-tuple's id and the role-bearing shape takes
        a suffix."""
        mixed_a = "Count_Person_MixedA"
        mixed_b = "Count_Person_MixedB"
        agree_a = "Amount_EconomicActivity_AgreeA"
        nodes = {
            mixed_a: {"label": "Mixed A", "arcs": self._person_arcs()},
            mixed_b: {
                "label": "Mixed B",
                "arcs": self._person_arcs(observation_properties=["provider", "recipient"]),
            },
            agree_a: {"label": "Agree A", "arcs": self._econ_arcs()},
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[mixed_a, mixed_b, agree_a], graph=graph)

        assert {s.shape_id: (list(s.sv_arc_facts or {}), s.declared_roles) for s in shapes} == {
            "person_count_measuredvalue": ([mixed_a], ("observationAbout",)),
            "person_count_measuredvalue_by_role": ([mixed_b], ("provider", "recipient")),
            "economicactivity_amount_measuredvalue": ([agree_a], ("observationAbout",)),
        }

    def test_role_bearing_members_of_several_role_sets_share_one_shape(self):
        """A recipient-only member and a provider-and-recipient member form one shape that
        declares both roles. The five-tuple's primary class is subject-only, since the index
        does not carry it, so the role-bearing shape takes the `_by_role` suffix."""
        recipient_sv = "Amount_EconomicActivity_RecipientOnly"
        pair_sv = "Amount_EconomicActivity_ProviderAndRecipient"
        nodes = {
            recipient_sv: {
                "label": "Recipient only",
                "arcs": self._econ_arcs(observation_properties=["recipient"]),
            },
            pair_sv: {
                "label": "Provider and recipient",
                "arcs": self._econ_arcs(observation_properties=["provider", "recipient"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[recipient_sv, pair_sv], graph=graph)

        assert [(s.shape_id, list(s.sv_arc_facts or {}), s.declared_roles) for s in shapes] == [
            (
                "economicactivity_amount_measuredvalue_by_role",
                [recipient_sv, pair_sv],
                ("provider", "recipient"),
            )
        ]

    def test_two_roles_on_one_flow_axis_skip_the_group(self):
        """Members naming the destination axis by two roles leave no role to bind a place
        under, so the group is skipped."""
        debtor_sv = "Amount_EconomicActivity_Debtor"
        recipient_sv = "Amount_EconomicActivity_Recipient"
        nodes = {
            debtor_sv: {
                "label": "Debtor",
                "arcs": self._econ_arcs(observation_properties=["debtor"]),
            },
            recipient_sv: {
                "label": "Recipient",
                "arcs": self._econ_arcs(observation_properties=["recipient"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        with pytest.raises(RoleSignatureError):
            derive_shapes(confirmed_svs=[debtor_sv, recipient_sv], graph=graph)

    @pytest.mark.parametrize(
        "order",
        [
            ("Count_Person_PermA", "Count_Person_PermB"),
            ("Count_Person_PermB", "Count_Person_PermA"),
        ],
    )
    def test_role_class_split_is_order_independent(self, order):
        """Either input order yields the same shape ids over the same members."""
        nodes = {
            "Count_Person_PermA": {"label": "A", "arcs": self._person_arcs()},
            "Count_Person_PermB": {
                "label": "B",
                "arcs": self._person_arcs(observation_properties=["provider", "recipient"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=list(order), graph=graph)

        assert {s.shape_id: list(s.sv_arc_facts or {}) for s in shapes} == {
            "person_count_measuredvalue": ["Count_Person_PermA"],
            "person_count_measuredvalue_by_role": ["Count_Person_PermB"],
        }

    def test_agreeing_role_bearing_members_stamp_the_shared_signature(self):
        """Matching member signatures are stamped on the shape."""
        agree_a = "Amount_EconomicActivity_RoleAgreeA"
        agree_b = "Amount_EconomicActivity_RoleAgreeB"
        nodes = {
            agree_a: {
                "label": "A",
                "arcs": self._econ_arcs(observation_properties=["provider", "recipient"]),
            },
            agree_b: {
                "label": "B",
                "arcs": self._econ_arcs(observation_properties=["provider", "recipient"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[agree_a, agree_b], graph=graph)

        assert len(shapes) == 1
        assert shapes[0].declared_roles == ("provider", "recipient")

    @pytest.mark.parametrize(
        "order",
        [
            ("Amount_EconomicActivity_DispatchA", "Amount_EconomicActivity_DispatchB"),
            ("Amount_EconomicActivity_DispatchB", "Amount_EconomicActivity_DispatchA"),
        ],
    )
    def test_agreeing_members_dispatch_the_same_in_either_order(self, order):
        nodes = {
            "Amount_EconomicActivity_DispatchA": {
                "label": "A",
                "arcs": self._econ_arcs(observation_properties=["provider", "recipient"]),
            },
            "Amount_EconomicActivity_DispatchB": {
                "label": "B",
                "arcs": self._econ_arcs(observation_properties=["provider", "recipient"]),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=list(order), graph=graph)

        assert len(shapes) == 1
        assert shapes[0].declared_roles == ("provider", "recipient")
