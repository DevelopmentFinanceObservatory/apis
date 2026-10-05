"""Tests `resolve_options_target` population, anchor, facet, window, and role derivation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.errors import EngineInputError
from search.engine.options.resolve import ResolvedOptionsTarget, resolve_options_target
from search.models import (
    BaseIdentity,
    IdentityBindingUnbound,
    IdentityShapeSchema,
    IdentitySlot,
    IdentityWindowAll,
    IdentityWindowRange,
    OptionsFor,
    OptionsRequest,
    TimeWindow,
)
from tests.engine._index_slots import unbound_index_slots
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader
from tests.statvar_members.conftest import FakeMemberArcReader, kenya_reader

# This non-index shape resolves through the fixture-backed member reader.
_LIFE_EXPECTANCY_SHAPE = IdentityShapeSchema(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)

# This indexed shape declares `financingProvider` and `financingRecipient`.
_CRS_SHAPE = IdentityShapeSchema(
    population_type="FinancialFlow",
    measured_property="grantEquivalent",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)

_KENYA_DCID = "country/KEN"


def _life_expectancy_identity(
    *,
    entities: dict[str, list[str]] | None = None,
    requested_window: IdentityWindowAll | IdentityWindowRange | None = None,
) -> BaseIdentity:
    return BaseIdentity(
        shape=_LIFE_EXPECTANCY_SHAPE,
        slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())],
        entities=entities or {},
        requested_window=requested_window or IdentityWindowAll(),
    )


def _crs_identity(*, entities: dict[str, list[str]] | None = None) -> BaseIdentity:
    return BaseIdentity(
        shape=_CRS_SHAPE,
        slots=unbound_index_slots(
            population_type="FinancialFlow",
            measured_property="grantEquivalent",
            measurement_qualifier="Nominal",
        ),
        entities=entities or {},
        requested_window=IdentityWindowAll(),
    )


def _request(identity: BaseIdentity, *, role: str = "observationAbout") -> OptionsRequest:
    return OptionsRequest(identity=identity, **{"for": OptionsFor(role=role)})


def _place_graph() -> FakeGraph:
    return FakeGraph(nodes={_KENYA_DCID: {"label": "Kenya", "type": "Country"}})


def test_a_non_index_identity_resolves_with_no_resubmit_and_no_seed():
    identity = _life_expectancy_identity(entities={"observationAbout": [_KENYA_DCID]})
    req = _request(identity)

    target = resolve_options_target(req, graph=_place_graph(), materialise_graph=kenya_reader())

    assert isinstance(target, ResolvedOptionsTarget)
    assert target.population.source == "graph"
    assert target.population.stat_var_dcids == ("LifeExpectancy_Person",)
    assert [a.ref for a in target.anchors] == [_KENYA_DCID]
    assert target.anchors[0].entity_type == "Country"
    assert target.anchors[0].role == "observationAbout"
    assert target.role_names_by_axis == {}
    assert target.facets is None
    assert target.requested_window is None


def test_a_role_bearing_identity_derives_role_names_by_axis_from_the_population():
    """No resubmit Spec, no fresh graph.node_arcs_batch read -- role_names_by_axis
    comes straight off the population's own declared_roles, computed by
    options_population itself while deriving the population."""
    identity = _crs_identity()
    req = _request(identity, role="destination")

    target = resolve_options_target(req, graph=_place_graph(), materialise_graph=kenya_reader())

    assert target.population.source == "index"
    assert target.population.declared_roles == ("financingProvider", "financingRecipient")
    assert target.role_names_by_axis == {
        "origin": "financingProvider",
        "destination": "financingRecipient",
    }


def test_requested_window_translates_a_range_to_engine_form():
    identity = _life_expectancy_identity(
        requested_window=IdentityWindowRange(start_year=2015, end_year=2020)
    )
    req = _request(identity)

    target = resolve_options_target(req, graph=_place_graph(), materialise_graph=kenya_reader())

    assert target.requested_window == TimeWindow(start_year=2015, end_year=2020)


def test_observationabout_against_a_role_bearing_shape_refuses():
    identity = _crs_identity()
    req = _request(identity, role="observationAbout")

    with pytest.raises(EngineInputError) as exc_info:
        resolve_options_target(req, graph=_place_graph(), materialise_graph=kenya_reader())

    assert exc_info.value.code == "option_target_invalid"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/for/role"
    assert {a.ref for a in exc_info.value.alternatives} == {"origin", "destination"}


def test_a_directional_role_against_a_subject_only_shape_still_refuses():
    """The mirror direction: a subject-only population accepts only
    observationAbout."""
    identity = _life_expectancy_identity(entities={"observationAbout": [_KENYA_DCID]})
    req = _request(identity, role="destination")

    with pytest.raises(EngineInputError) as exc_info:
        resolve_options_target(req, graph=_place_graph(), materialise_graph=kenya_reader())

    assert exc_info.value.code == "option_target_invalid"
    assert exc_info.value.pointer == "/for/role"
    assert {a.ref for a in exc_info.value.alternatives} == {"observationAbout"}


def test_resolved_stat_var_dcids_is_no_longer_a_field():
    identity = _life_expectancy_identity()
    with pytest.raises(ValidationError):
        OptionsRequest(
            identity=identity,
            resolved_stat_var_dcids=["LifeExpectancy_Person"],
            **{"for": OptionsFor(role="observationAbout")},
        )


def test_response_options_is_no_longer_a_field():
    identity = _life_expectancy_identity()
    with pytest.raises(ValidationError):
        OptionsRequest(
            identity=identity, response_options=None, **{"for": OptionsFor(role="observationAbout")}
        )


def test_include_coverage_replaces_response_options_include_option_coverage():
    identity = _life_expectancy_identity()
    default_req = OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})
    assert default_req.include_coverage is False

    opted_in_req = OptionsRequest(
        identity=identity,
        include_coverage=True,
        **{"for": OptionsFor(role="observationAbout")},
    )
    assert opted_in_req.include_coverage is True


def test_a_posted_entity_ref_the_graph_does_not_carry_refuses_invalid_role_ref():
    identity = _life_expectancy_identity(entities={"observationAbout": ["country/DOES_NOT_EXIST"]})
    req = _request(identity)

    with pytest.raises(EngineInputError) as exc_info:
        resolve_options_target(req, graph=_place_graph(), materialise_graph=kenya_reader())

    assert exc_info.value.code == "invalid_role_ref"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/identity/entities/observationAbout"


def test_a_present_but_typeless_entity_ref_also_refuses_invalid_role_ref():
    """node_types_batch omits a present-but-type-less node exactly like an
    absent one -- the two read identically through this one batched call
    (see _typed_anchors's own docstring)."""
    identity = _life_expectancy_identity(entities={"observationAbout": ["country/TYPELESS"]})
    req = _request(identity)
    graph = FakeGraph(nodes={"country/TYPELESS": {"label": "No Type"}})

    with pytest.raises(EngineInputError) as exc_info:
        resolve_options_target(req, graph=graph, materialise_graph=kenya_reader())

    assert exc_info.value.code == "invalid_role_ref"


# Role-target validation runs after the identity-core checks.
def test_a_malformed_role_alongside_an_invalid_identity_core_refuses_under_the_earlier_check():
    """Two slots naming the same property fail identity_edit.
    validate_edited_identity -- population._member_slot_bindings would
    otherwise collapse this last-write-wins into a single binding rather
    than refuse. Posting that identity together with a role this
    definition's population never declares still reads as request_invalid,
    never option_target_invalid: the role check runs last."""
    identity = BaseIdentity(
        shape=_LIFE_EXPECTANCY_SHAPE,
        slots=[
            IdentitySlot(property="gender", binding=IdentityBindingUnbound()),
            IdentitySlot(property="gender", binding=IdentityBindingUnbound()),
        ],
        entities={},
        requested_window=IdentityWindowAll(),
    )
    req = _request(identity, role="not-a-declared-role")

    with pytest.raises(EngineInputError) as exc_info:
        resolve_options_target(
            req, graph=_place_graph(), materialise_graph=FakeGraphReader(arcs={}, subjects={})
        )

    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.status == 400
    assert exc_info.value.pointer == "/identity"


# ---------------------------------------------------------------------------
# A role-bearing family declaring only one flow axis: _check_role_target's
# declared_axes comes straight off role_names_by_axis, so a destination-only
# population accepts exactly "destination" and refuses "origin".
# ---------------------------------------------------------------------------

_DESTINATION_ONLY_SHAPE = IdentityShapeSchema(
    population_type="EconomicActivity", measured_property="amount", stat_type="measuredValue"
)

_DESTINATION_ONLY_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "Amount_EconomicActivity_DestinationOnlyFixture": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("EconomicActivity",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": ("financingRecipient",),
    }
}


def _destination_only_identity() -> BaseIdentity:
    return BaseIdentity(
        shape=_DESTINATION_ONLY_SHAPE, slots=[], entities={}, requested_window=IdentityWindowAll()
    )


def _destination_only_reader() -> FakeMemberArcReader:
    return FakeMemberArcReader(arcs=_DESTINATION_ONLY_ARCS)


def test_a_destination_only_family_accepts_its_one_declared_axis():
    identity = _destination_only_identity()
    req = _request(identity, role="destination")

    target = resolve_options_target(
        req, graph=_place_graph(), materialise_graph=_destination_only_reader()
    )

    assert target.population.declared_roles == ("financingRecipient",)
    assert target.role_names_by_axis == {"destination": "financingRecipient"}


def test_a_destination_only_family_refuses_the_undeclared_origin_axis():
    identity = _destination_only_identity()
    req = _request(identity, role="origin")

    with pytest.raises(EngineInputError) as exc_info:
        resolve_options_target(
            req, graph=_place_graph(), materialise_graph=_destination_only_reader()
        )

    assert exc_info.value.code == "option_target_invalid"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/for/role"
    assert {a.ref for a in exc_info.value.alternatives} == {"destination"}
