"""Tests options resolution refusals for role collisions, invalid identities, and anchors."""

from __future__ import annotations

import pytest

from search.engine.errors import EngineInputError
from search.engine.identity_edit import open_requested_role
from search.engine.options.population import OptionsPopulation, options_population
from search.engine.options.resolve import (
    _check_entity_role_admission,
    _declared_axis_roles,
    resolve_options_target,
)
from search.models import (
    BaseIdentity,
    IdentityBindingExact,
    IdentityBindingUnbound,
    IdentityShapeSchema,
    IdentitySlot,
    IdentityWindowAll,
    OptionsFor,
    OptionsRequest,
)
from tests.code_index.aid_identity import education_aid_identity
from tests.engine._index_slots import unbound_index_slots
from tests.engine.test_options_population import (
    _IDS_BY_CREDITOR,
    _IDS_MEMBERS,
    _ids_identity,
    _ids_index,
    _ids_reader,
    _use_index,
)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader
from tests.statvar_members.conftest import FakeMemberArcReader, kenya_reader

# ---------------------------------------------------------------------------
# Two declared roles on one flow axis must raise a typed refusal.
# ---------------------------------------------------------------------------


def _population(declared_roles: tuple[str, ...]) -> OptionsPopulation:
    return OptionsPopulation(
        stat_var_dcids=("TESTFIXTURE/OptionsResolve/SameAxisCollision",),
        declared_roles=declared_roles,
        source="graph",
    )


class TestDeclaredAxisRolesCatchesSameAxisCollision:
    def test_declared_axis_roles_refuses_role_signature_incompatible(self):
        """A provider/creditor axis collision becomes a typed input error."""
        with pytest.raises(EngineInputError) as exc_info:
            _declared_axis_roles(_population(("provider", "creditor")))
        assert exc_info.value.code == "role_signature_incompatible"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/for/role"

    def test_non_colliding_roles_resolve_cleanly(self):
        result = _declared_axis_roles(_population(("financingProvider", "financingRecipient")))
        assert result == {"origin": "financingProvider", "destination": "financingRecipient"}


# ---------------------------------------------------------------------------
# A pinned-ask shape has no population to derive.
# ---------------------------------------------------------------------------


class TestPinnedAskRefusesBeforeDerivation:
    def test_pin_arm_shape_refuses_request_invalid(self):
        pin_identity = BaseIdentity(
            shape={"kind": "stat_var", "id": "Count_Person_Female"},
            slots=[],
            entities={},
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=pin_identity, **{"for": OptionsFor(role="observationAbout")})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(
                req, graph=FakeGraph(), materialise_graph=FakeGraphReader(arcs={}, subjects={})
            )

        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.status == 400
        assert exc_info.value.pointer == "/identity/shape"


class TestCompositeProvenanceRefusesBeforeDerivation:
    def test_a_composite_provenance_refuses_request_invalid(self):
        identity = BaseIdentity.model_validate(
            education_aid_identity(source="provenance/ONEBilateralAndImputedMultilateral")
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="financingProvider")})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(
                req, graph=FakeGraph(), materialise_graph=FakeGraphReader(arcs={}, subjects={})
            )

        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.status == 400
        assert exc_info.value.pointer == "/identity/facets/provenance"


# ---------------------------------------------------------------------------
# The identity core rejects duplicate slot properties before population derivation.
# ---------------------------------------------------------------------------

_NON_INDEX_SHAPE = IdentityShapeSchema(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)


class TestIdentityCoreValidationRunsBeforeDerivation:
    def test_duplicate_slot_properties_refuse_request_invalid(self):
        """Duplicate slot properties must be refused before dict conversion can collapse them."""
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[
                IdentitySlot(property="gender", binding=IdentityBindingUnbound()),
                IdentitySlot(property="gender", binding=IdentityBindingUnbound()),
            ],
            entities={},
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(
                req, graph=FakeGraph(), materialise_graph=FakeGraphReader(arcs={}, subjects={})
            )

        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.status == 400
        assert exc_info.value.pointer == "/identity"

    @pytest.mark.parametrize(
        ("counts", "pointer"),
        [
            pytest.param(
                {"observationAbout": 257}, "/identity/entities/observationAbout", id="one-role"
            ),
            pytest.param(
                {"destination": 200, "origin": 200},
                "/identity/entities",
                id="same-refs-on-two-roles",
            ),
        ],
    )
    def test_an_identity_over_the_place_bound_refuses_before_any_read(self, counts, pointer):
        """The options door holds the bound on the places a Spec binds, as the resubmit
        does: distinct dcids per role, summed over the roles, refused before the graph
        is read."""

        class _NoGraph:
            def __getattr__(self, name):
                raise AssertionError(f"graph must not be read for the place bound: {name}")

        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[],
            entities={
                role: [f"country/P{i}" for i in range(count)] for role, count in counts.items()
            },
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(req, graph=_NoGraph(), materialise_graph=_NoGraph())

        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.status == 400
        assert exc_info.value.pointer == pointer


# ---------------------------------------------------------------------------
# Options requests do not support `exact` slot bindings.
# ---------------------------------------------------------------------------


class TestExactBindingRefusesBeforeDerivation:
    def test_exact_slot_binding_refuses_exact_binding_unsupported(self):
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[
                IdentitySlot(property="gender", binding=IdentityBindingExact(ref="Female")),
            ],
            entities={},
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(
                req, graph=FakeGraph(), materialise_graph=FakeGraphReader(arcs={}, subjects={})
            )

        assert exc_info.value.code == "exact_binding_unsupported"
        assert exc_info.value.status == 501
        assert exc_info.value.pointer == "/identity/slots/0/binding"


# ---------------------------------------------------------------------------
# Anchor validation precedes population derivation.
# ---------------------------------------------------------------------------


class _ExplodingMaterialiseGraph:
    def __getattr__(self, name: str):  # pragma: no cover -- only reached on a defect
        raise AssertionError(
            f"materialise_graph.{name} was called; anchors must be typed and "
            "checked BEFORE options_population ever reads the materialise graph"
        )


class TestAnchorsAreTypedBeforePopulationIsDerived:
    def test_an_invalid_entity_ref_refuses_without_ever_deriving_the_population(self):
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())],
            entities={"observationAbout": ["country/DOES_NOT_EXIST"]},
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(
                req, graph=FakeGraph(), materialise_graph=_ExplodingMaterialiseGraph()
            )

        assert exc_info.value.code == "invalid_role_ref"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/identity/entities/observationAbout"


# ---------------------------------------------------------------------------
# Anchors must match a role declared by the resolved population.
# ---------------------------------------------------------------------------


def _population_for_admission(declared_roles: tuple[str, ...]) -> OptionsPopulation:
    return OptionsPopulation(
        stat_var_dcids=("TESTFIXTURE/OptionsResolve/EntityRoleAdmission",),
        declared_roles=declared_roles,
        source="graph",
    )


class TestCheckEntityRoleAdmission:
    def test_no_entities_bound_is_a_no_op_for_either_shape(self):
        _check_entity_role_admission(
            BaseIdentity(
                shape=_NON_INDEX_SHAPE, slots=[], entities={}, requested_window=IdentityWindowAll()
            ),
            {},
        )
        _check_entity_role_admission(
            BaseIdentity(
                shape=_NON_INDEX_SHAPE, slots=[], entities={}, requested_window=IdentityWindowAll()
            ),
            {"origin": "financingProvider", "destination": "financingRecipient"},
        )

    def test_a_directional_entity_against_a_subject_only_population_refuses(self):
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[],
            entities={"destination": ["country/KEN"]},
            requested_window=IdentityWindowAll(),
        )
        with pytest.raises(EngineInputError) as exc_info:
            _check_entity_role_admission(identity, {})
        assert exc_info.value.code == "role_incompatible"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/identity/entities/destination"

    def test_an_observationabout_entity_against_a_role_bearing_population_refuses(self):
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[],
            entities={"observationAbout": ["country/KEN"]},
            requested_window=IdentityWindowAll(),
        )
        with pytest.raises(EngineInputError) as exc_info:
            _check_entity_role_admission(
                identity, {"origin": "financingProvider", "destination": "financingRecipient"}
            )
        assert exc_info.value.code == "role_incompatible"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/identity/entities/observationAbout"

    def test_an_entity_on_an_axis_the_population_declares_no_role_for_refuses(self):
        """An origin anchor is invalid when the population declares only a destination role."""
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[],
            entities={"origin": ["country/KEN"]},
            requested_window=IdentityWindowAll(),
        )
        with pytest.raises(EngineInputError) as exc_info:
            _check_entity_role_admission(identity, {"destination": "financingRecipient"})
        assert exc_info.value.code == "role_incompatible"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/identity/entities/origin"


class TestEntityRoleAdmissionThroughResolveOptionsTarget:
    def test_a_destination_entity_against_a_subject_only_identity_refuses_through_resolve(self):
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())],
            entities={"destination": ["country/KEN"]},
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})
        graph = FakeGraph(nodes={"country/KEN": {"label": "Kenya", "type": "Country"}})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(req, graph=graph, materialise_graph=kenya_reader())

        assert exc_info.value.code == "role_incompatible"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/identity/entities/destination"


# ---------------------------------------------------------------------------
# The door reads the population of the identity with the requested flow axis open.
# ---------------------------------------------------------------------------

_KEN_GRAPH_NODES = {"country/KEN": {"label": "Kenya", "type": "Country"}}


@pytest.mark.parametrize(
    ("entities", "open_roles", "role", "expected"),
    [
        ({}, None, "origin", ["origin"]),
        ({}, ["destination"], "origin", ["destination", "origin"]),
        ({"origin": ["country/CHN"]}, None, "origin", None),
        ({}, ["origin"], "origin", ["origin"]),
        ({}, None, "observationAbout", None),
        ({}, None, "creditor", ["origin"]),
    ],
)
def test_open_requested_role_opens_only_an_unnamed_flow_axis(entities, open_roles, role, expected):
    identity = _ids_identity(entities, open_roles=open_roles)

    opened = open_requested_role(identity, role=role)

    assert opened.open_roles == expected
    assert opened.entities == identity.entities


class TestRequestedFlowAxisOpensThePopulation:
    @pytest.mark.parametrize("source", ["graph", "index"])
    def test_an_ids_total_asked_for_origin_reads_the_by_creditor_twin(
        self, monkeypatch, unlisted_index, source
    ):
        """The posted total declares only the debtor; asking for creditors reads the twin an
        edit binding the creditor elects, so the role-target check accepts `origin`."""
        if source == "index":
            _use_index(monkeypatch, _ids_index(_IDS_MEMBERS))
            reader = FakeMemberArcReader(arcs={})
        else:
            reader = _ids_reader(_IDS_MEMBERS)
        req = OptionsRequest(
            identity=_ids_identity({"destination": ["country/KEN"]}),
            **{"for": OptionsFor(role="origin")},
        )

        target = resolve_options_target(
            req, graph=FakeGraph(nodes=_KEN_GRAPH_NODES), materialise_graph=reader
        )

        assert target.population.stat_var_dcids == (_IDS_BY_CREDITOR,)
        assert target.role_names_by_axis == {
            "origin": "financingProvider",
            "destination": "financingRecipient",
        }
        assert target.identity.open_roles == ["origin"]
        assert target.identity.entities == {"destination": ["country/KEN"]}

    def test_a_debtor_of_a_kind_no_recipient_takes_refuses_naming_its_kind(self):
        """A flow-role anchor whose kind none of the code index's recipients take refuses
        once the population declares the role (`search.role_kinds`)."""
        req = OptionsRequest(
            identity=_ids_identity({"destination": ["Count_Person"]}),
            **{"for": OptionsFor(role="origin")},
        )
        graph = FakeGraph(
            nodes={"Count_Person": {"label": "Total population", "type": "StatisticalVariable"}}
        )

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(req, graph=graph, materialise_graph=_ids_reader(_IDS_MEMBERS))

        assert exc_info.value.code == "invalid_role_ref"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/identity/entities/destination"
        assert str(exc_info.value).startswith("Count_Person is a statistical variable;")

    def test_a_crs_population_is_unchanged_by_the_opened_axis(self):
        identity = BaseIdentity(
            shape=IdentityShapeSchema(
                population_type="FinancialFlow",
                measured_property="grantEquivalent",
                stat_type="measuredValue",
                measurement_qualifier="Nominal",
            ),
            slots=unbound_index_slots(
                population_type="FinancialFlow",
                measured_property="grantEquivalent",
                measurement_qualifier="Nominal",
            ),
            entities={},
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="destination")})

        target = resolve_options_target(req, graph=FakeGraph(), materialise_graph=kenya_reader())

        assert target.population == options_population(identity, kenya_reader())
        assert target.identity.open_roles == ["destination"]

    def test_a_subject_only_identity_asked_for_origin_still_refuses(self):
        """No StatVar declares the opened axis, so the election drops it and the population
        stays subject-only."""
        identity = BaseIdentity(
            shape=_NON_INDEX_SHAPE,
            slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())],
            entities={"observationAbout": ["country/KEN"]},
            requested_window=IdentityWindowAll(),
        )
        req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="origin")})

        with pytest.raises(EngineInputError) as exc_info:
            resolve_options_target(
                req, graph=FakeGraph(nodes=_KEN_GRAPH_NODES), materialise_graph=kenya_reader()
            )

        assert exc_info.value.code == "option_target_invalid"
        assert exc_info.value.pointer == "/for/role"
        assert [a.ref for a in exc_info.value.alternatives] == ["observationAbout"]
