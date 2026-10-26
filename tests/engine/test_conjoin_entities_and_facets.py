"""Tests same-shape merge rules for identities, entities, windows, and facet state.
Merges refuse groups whose members differ on more than one identity constraint.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from search.engine.binders import GeminiBinder
from search.engine.concurrency import scoped_concurrency
from search.engine.conjoin import BINDING_FOLDED, combine_regions
from search.engine.enrichment import _reconcile_after_collapse
from search.engine.entry.multi import _stream_combined_regions
from search.engine.entry.params import EntryParams
from search.engine.envelope import now_ms
from search.engine.errors import GraphInfraError
from search.engine.events import FacetOptionsEvent, ResponseEvent
from search.engine.facet_refusals import facet_refusal
from search.engine.region import OPTION_COVERAGE_PROBE_FAILED, RegionResult
from search.engine.region.facet_probe import facet_pins, facet_probe, merge_facet_probes
from search.engine.region.option_coverage import _facet_options, _maybe_build_facet_options
from search.engine.region.relaunch import _relaunch_slot_options
from search.engine.region.un_system import NO_LABELS
from search.engine.same_shape import collapse_same_shape
from search.engine.spec_build import NamedGroup, build_spec
from search.models import (
    BindingSet,
    BindingValue,
    CoverageBare,
    Entity,
    EntityGroup,
    EntityRoleDirectional,
    EntityRoleSubject,
    Expansion,
    FacetOptions,
    GraphRef,
    IdentityBindingValue,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ResolvedEntity,
    ResolvedFacet,
    ResolvedScope,
    RoleAxis,
    Shape,
    Slot,
    SlotKey,
    SlotOption,
    SlotValue,
    StatVar,
    TimeWindow,
)
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._builders import constraint_value_slot as _value_slot
from tests.engine._builders import gref as _gr
from tests.engine._conjoin_fixtures import _PERSON_SHAPE, _region
from tests.engine._fake_coverage import build_coverage_index, empty_coverage_snapshot
from tests.fixtures import FakeGraph

_KEN = "country/KEN"


def _spec_with_facet(
    shape: Shape,
    constraint_val: str,
    sv_dcid: str,
    facet_bindings: dict,
    open_roles: tuple[RoleAxis, ...] = (),
):
    """Like _spec, but with a door-posted facet binding (definition.identity.facets) and
    open roles, the concrete cases a same-shape merge must compare and carry forward."""
    return build_spec(
        open_roles=open_roles,
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[_value_slot("constraintProp", constraint_val)],
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        facet_bindings=facet_bindings,
    )


def _spec_with_resolved_facets(
    shape: Shape,
    constraint_val: str,
    sv_dcid: str,
    facets: dict,
    facet_options: list | None = None,
    unit_lexicon_version: str | None = None,
):
    """Build a spec with resolved facet state, distinct from posted identity facets.
    State `unit_lexicon_version` after `build_spec`.
    """
    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[_value_slot("constraintProp", constraint_val)],
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        facets=facets,
        facet_options=facet_options,
    )
    if unit_lexicon_version is not None:
        spec = spec.model_copy(
            update={"resolved": spec.resolved.stating_unit_lexicon_version(unit_lexicon_version)}
        )
    return spec


def _with_declaration_version(spec, version: str):
    return spec.model_copy(update={"resolved": spec.resolved.stating_declaration_version(version)})


def _spec_with_scope(spec, member_dcids: list[str]):
    """Install a defaulted observationAbout scope on a conjoin test Spec."""
    scope = ResolvedScope(
        axis="observationAbout",
        members=[
            ResolvedEntity(ref=dcid, label=None, entity_type=None, role="observationAbout")
            for dcid in member_dcids
        ],
    )
    return spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"scope": scope})})


_RECIPIENT_PROP = "DevelopmentFinanceRecipient"


def _recipient_only_shape() -> Shape:
    """A recipient where-slot whose identity is carried in `entities`."""
    return Shape(
        shape_id="recipient_only_test_shape",
        label="Test recipient-only shape",
        population_type=_gr("Person"),
        measured_property=_gr("count"),
        stat_type=_gr("measuredValue"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[
            SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        ],
        member_count=1,
    )


def _destination_entity(dcid: str) -> Entity:
    return Entity(
        ref=_gr(dcid),
        role=EntityRoleDirectional(role=_gr(_RECIPIENT_PROP), axis="destination"),
    )


def _where_bound_spec(shape: Shape, recipient_dcid: str, sv_dcid: str):
    where_slot = Slot(
        key=SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        binding=BindingValue(value=SlotValue(ref=_gr(recipient_dcid), value_kind="entity")),
    )
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[where_slot],
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[_destination_entity(recipient_dcid)],
        coverage=CoverageBare(has_data=True),
    )


def _directional_entity(dcid: str, role_prop: str, axis: str) -> Entity:
    return Entity(ref=_gr(dcid), role=EntityRoleDirectional(role=_gr(role_prop), axis=axis))


def _where_bound_spec_with_provider(
    shape: Shape, recipient_dcid: str, provider_dcid: str, sv_dcid: str
):
    """Add a provider entity, which has no slot on this recipient-only shape."""
    where_slot = Slot(
        key=SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        binding=BindingValue(value=SlotValue(ref=_gr(recipient_dcid), value_kind="entity")),
    )
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[where_slot],
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[
            _destination_entity(recipient_dcid),
            _directional_entity(provider_dcid, _RECIPIENT_PROP, "origin"),
        ],
        coverage=CoverageBare(has_data=True),
    )


def _directional_member(
    provider_dcid: str,
    recipient_dcid: str,
    sv_dcid: str,
    variable_text: str,
    earliest_index: int,
) -> RegionResult:
    """Build a recipient-slot member with a provider entity outside its slots.
    The merge gate must catch disagreements in both identity buckets.
    """
    spec = _where_bound_spec_with_provider(
        _recipient_only_shape(), recipient_dcid, provider_dcid, sv_dcid
    )
    return _region(spec, variable_text, earliest_index=earliest_index)


# ---------------------------------------------------------------------------
# collapse_same_shape — entity union and identity-entities
# ---------------------------------------------------------------------------


def test_collapse_same_shape_refuses_merge_with_disjoint_resolved_scopes():
    """Each defaulted scope belongs to its own StatVar; collapse must not
    union StatVars while retaining only the first member's scope."""
    spec_a = _spec_with_scope(
        _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a"),
        ["country/AAA"],
    )
    spec_b = _spec_with_scope(
        _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv_b"),
        ["country/BBB"],
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert effective == [r_a]
    assert residual == [("var B", 1)]


def test_collapse_same_shape_merges_with_identical_resolved_scopes():
    """The scope gate preserves normal collapse when every member agrees."""
    member_dcids = ["country/AAA", "country/BBB"]
    spec_a = _spec_with_scope(_spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a"), member_dcids)
    spec_b = _spec_with_scope(_spec(_PERSON_SHAPE, "ValB", sv_dcid="sv_b"), member_dcids)
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    assert effective[0].spec.resolved.scope == spec_a.resolved.scope


def test_collapse_same_shape_where_axis_merge_distinct_from_base():
    """A recipient merge unions identity entities because `where` is absent from identity slots.
    The merged ID must differ from each single-recipient ID.
    """
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")

    r_a = _region(spec_kenya, "aid to Kenya", earliest_index=0)
    r_b = _region(spec_ethiopia, "aid to Ethiopia", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec
    merged_binding = merged.resolved.slots[0].binding
    assert isinstance(merged_binding, BindingSet)
    assert {v.ref.ref for v in merged_binding.values} == {"country/KEN", "country/ETH"}

    assert merged.spec_id != spec_kenya.spec_id
    assert merged.spec_id != spec_ethiopia.spec_id
    assert set(merged.definition.identity.entities["destination"]) == {"country/KEN", "country/ETH"}


def test_collapse_same_shape_unions_entities():
    """A recipient merge must union the members' resolved entities.
    `where` is absent from identity slots, so each recipient contributes to identity.
    """
    shape = _recipient_only_shape()
    spec_a = _where_bound_spec(shape, "country/KEN", "sv_a")
    spec_b = _where_bound_spec(shape, "country/ETH", "sv_b")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]
    merged_entity_dcids = [e.ref for e in merged.spec.resolved.entities]
    assert merged_entity_dcids == ["country/KEN", "country/ETH"]


def _stating_unserved(spec, *dcids: str):
    """`spec` stating each of `dcids` unserved as its subject."""
    pairs = tuple(("observationAbout", dcid) for dcid in dcids)
    return spec.model_copy(update={"resolved": spec.resolved.stating_unserved(pairs)})


@pytest.mark.parametrize(
    ("unserved_by_part", "unserved"),
    [
        ((("country/GHA",), ("country/GHA",)), ("country/GHA",)),
        ((("country/GHA",), ()), ()),
        (((), ()), ()),
    ],
    ids=["neither_part_serves_ghana", "one_part_serves_ghana", "both_serve_both"],
)
def test_a_merged_answer_states_the_places_no_part_serves(
    unserved_by_part: tuple[tuple[str, ...], tuple[str, ...]], unserved: tuple[str, ...]
) -> None:
    """Two parts naming Kenya and Ghana merge on their constraint value. Each part states
    Ghana unserved in an `entity_unavailable` departure on the where token when neither
    part's series serves it, and nothing once one part serves it. Its sentence carries no
    note either way."""

    def places() -> list[Entity]:
        return [
            Entity(ref=GraphRef(ref=dcid, label=label), role=EntityRoleSubject())
            for dcid, label in (("country/KEN", "Kenya"), ("country/GHA", "Ghana"))
        ]

    regions = [
        _region(
            _stating_unserved(
                _spec(_PERSON_SHAPE, value, sv_dcid=f"sv_{value}", entities=places()), *unserved
            ),
            value,
            earliest_index=index,
        )
        for index, (value, unserved) in enumerate(
            zip(("ValA", "ValB"), unserved_by_part, strict=True)
        )
    ]

    response, _effective = combine_regions(
        regions,
        query="ValA and ValB in Kenya and Ghana",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )

    merged = response.parts[0].spec
    assert merged.spec_id == response.parts[1].spec.spec_id
    assert merged.spec_id not in {region.spec.spec_id for region in regions}
    assert not any(token.kind == "note" for token in merged.resolved.sentence_plan or [])
    assert merged.resolved.unserved_entities == tuple(
        ("observationAbout", dcid) for dcid in unserved
    )
    expected = [("entity_unavailable", "where:", "Ghana")] if unserved else []
    for part in response.parts:
        assert [(d.code, d.binding_ref, d.asked) for d in part.departures] == expected


@pytest.mark.parametrize(
    ("sv_b", "named"),
    [("sv_ValB", ["Left flow"]), ("sv_Left", [])],
    ids=["no_part_serves_it", "the_other_part_serves_it"],
)
def test_a_merged_answer_names_the_members_no_part_serves(sv_b: str, named: list[str]) -> None:
    """Two parts merge on their constraint value, the first leaving out `sv_Left`. The
    merged answer names it in a `member_unavailable` departure while no part serves it, and
    names none once the second part's member is `sv_Left`."""
    left = GraphRef(ref="sv_Left", label="Left flow")
    first = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_ValA")
    first = first.model_copy(update={"resolved": first.resolved.stating_unserved_members((left,))})
    regions = [
        _region(first, "ValA", earliest_index=0),
        _region(_spec(_PERSON_SHAPE, "ValB", sv_dcid=sv_b), "ValB", earliest_index=1),
    ]

    response, _effective = combine_regions(
        regions,
        query="ValA and ValB in Kenya",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )

    assert response.parts[0].spec.spec_id == response.parts[1].spec.spec_id
    for part in response.parts:
        assert [d.asked for d in part.departures if d.code == "member_unavailable"] == named


def test_collapse_same_shape_refuses_merge_when_dcid_flips_role():
    """A dcid changing from recipient to provider changes two identity buckets and blocks merging."""
    shape = _recipient_only_shape()
    spec_a = _where_bound_spec_with_provider(shape, "country/KEN", "country/XYZ", "sv_a")
    spec_b = _where_bound_spec_with_provider(shape, "country/ETH", "country/KEN", "sv_b")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


def test_collapse_same_shape_refuses_merge_with_different_donors():
    """Different donors and recipients change two identity buckets.
    The merge refuses the cross-product.
    """
    r_us = _directional_member("country/USA", "country/KEN", "sv_us", "var US", 0)
    r_fr = _directional_member("country/FRA", "country/ETH", "sv_fr", "var FR", 1)

    effective, residual = collapse_same_shape([r_us, r_fr])

    assert residual == [("var FR", 1)]
    assert len(effective) == 1
    assert effective[0] is r_us  # unmerged: first member kept unchanged


def test_collapse_same_shape_refuses_directional_reversal():
    """Opposite donor-recipient pairs over the same four countries produce distinct identities.
    Each group differs on both the recipient slot and provider entity, so both merges refuse.
    """
    # Group A: donor differs between the two members (USA vs France); the
    # where-axis recipient is still the merge's one differing SLOT.
    r_a1 = _directional_member("country/USA", "country/ETH", "sv_a1", "var A1", 0)
    r_a2 = _directional_member("country/FRA", "country/KEN", "sv_a2", "var A2", 1)

    # Group B: the exact reversal -- same four countries, donor and
    # recipient swapped in both pairs.
    r_b1 = _directional_member("country/ETH", "country/USA", "sv_b1", "var B1", 0)
    r_b2 = _directional_member("country/KEN", "country/FRA", "sv_b2", "var B2", 1)

    effective_a, residual_a = collapse_same_shape([r_a1, r_a2])
    effective_b, residual_b = collapse_same_shape([r_b1, r_b2])

    # Refused, not merged: first member kept verbatim, second goes to residual.
    assert residual_a == [("var A2", 1)]
    assert residual_b == [("var B2", 1)]
    assert effective_a == [r_a1]
    assert effective_b == [r_b1]

    # Two opposite donor/recipient stories must not mint the same identity.
    assert effective_a[0].spec.spec_id != effective_b[0].spec.spec_id


def test_collapse_same_shape_refuses_merge_when_entity_flips_role():
    """An entity that is the RECIPIENT in one member and the PROVIDER in
    another must refuse the merge outright -- the sharpest form of the
    cross-product risk: merging "A's aid to B" with "B's aid to C" would
    otherwise yield provider {A}, recipient {B, C}, silently dropping B's
    provider role from the merged identity.
    """
    r_1 = _directional_member("country/AAA", "country/BBB", "sv_1", "var 1", 0)
    # country/BBB flips from recipient (member 1) to provider (member 2).
    r_2 = _directional_member("country/BBB", "country/CCC", "sv_2", "var 2", 1)

    effective, residual = collapse_same_shape([r_1, r_2])

    assert residual == [("var 2", 1)]
    assert len(effective) == 1
    assert effective[0] is r_1  # unmerged: first member kept unchanged


def test_collapse_same_shape_reversed_provider_recipient_stays_distinct():
    """Opposite provider-recipient identities stay distinct when resolved entities share one subject.
    `identity_entities` carries the direction omitted from `resolved.entities`. A member's
    named group survives the merge on the role its identity binds the group's members on.
    """
    usa_to_eth = [
        _directional_entity("country/USA", "provider", "origin"),
        _directional_entity("country/ETH", "recipient", "destination"),
    ]
    eth_to_usa = [
        _directional_entity("country/ETH", "provider", "origin"),
        _directional_entity("country/USA", "recipient", "destination"),
    ]
    collapsed_subject = [Entity(ref=_gr("flow/placeholder"), role=EntityRoleSubject())]

    usa_group = NamedGroup(
        label="the test group", dcid=None, member_dcids=frozenset({"country/USA"})
    )

    def _place_constraint_off_spec(constraint_val, sv_dcid, identity_entities):
        return build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=_PERSON_SHAPE,
            slots=[_value_slot("constraintProp", constraint_val)],
            stat_vars=[
                StatVar(
                    ref=_gr(sv_dcid),
                    shape_id=_PERSON_SHAPE.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=collapsed_subject,
            identity_entities=identity_entities,
            coverage=CoverageBare(has_data=True),
            named_groups=(usa_group,),
        )

    group1 = [
        _region(
            _place_constraint_off_spec("ValA", "sv_1a", usa_to_eth),
            "var 1a",
            earliest_index=0,
        ),
        _region(
            _place_constraint_off_spec("ValB", "sv_1b", usa_to_eth),
            "var 1b",
            earliest_index=1,
        ),
    ]
    group2 = [
        _region(
            _place_constraint_off_spec("ValA", "sv_2a", eth_to_usa),
            "var 2a",
            earliest_index=0,
        ),
        _region(
            _place_constraint_off_spec("ValB", "sv_2b", eth_to_usa),
            "var 2b",
            earliest_index=1,
        ),
    ]

    merged1, _ = collapse_same_shape(group1)
    merged2, _ = collapse_same_shape(group2)

    assert len(merged1) == 1
    assert len(merged2) == 1
    assert merged1[0].spec.spec_id != merged2[0].spec.spec_id
    served = EntityGroup(ref=None, label="the test group")
    assert merged1[0].spec.resolved.entity_groups == {"origin": served}
    assert merged2[0].spec.resolved.entity_groups == {"destination": served}


def test_collapse_same_shape_divergent_entities_on_non_where_axis_stays_residual():
    """A value-slot and entity difference cannot collapse into a value-place cross-product.
    `where` is absent from identity slots, so entity identity is checked separately.
    """
    kenya = [_directional_entity("country/KEN", _RECIPIENT_PROP, "destination")]
    ethiopia = [_directional_entity("country/ETH", _RECIPIENT_PROP, "destination")]

    def _place_constraint_off_spec(constraint_val, sv_dcid, identity_entities):
        return build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=_PERSON_SHAPE,
            slots=[_value_slot("constraintProp", constraint_val)],
            stat_vars=[
                StatVar(
                    ref=_gr(sv_dcid),
                    shape_id=_PERSON_SHAPE.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=[Entity(ref=_gr("flow/placeholder"), role=EntityRoleSubject())],
            identity_entities=identity_entities,
            coverage=CoverageBare(has_data=True),
        )

    spec_a = _place_constraint_off_spec("ValA", "sv_a", kenya)
    spec_b = _place_constraint_off_spec("ValB", "sv_b", ethiopia)
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    # Divergent entities on a non-where axis: residual, NOT a merged BindingSet.
    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


def test_collapse_same_shape_window_survives_merge():
    """The merged spec keeps the requested window in its identity and trace."""
    window = TimeWindow(start_year=2015, end_year=2020)
    spec_a = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValA")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_a"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        requested_window=window,
    )
    spec_b = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValB")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_b"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        requested_window=window,
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec
    assert merged.definition.identity.requested_window.kind == "range"
    assert (
        merged.definition.identity.requested_window == spec_a.definition.identity.requested_window
    )
    assert merged.resolved.trace.requested_window == window


def test_collapse_same_shape_refuses_merge_on_window_disagreement():
    """Different requested windows refuse the merge so one member's dates cannot govern both.

    The merge refuses instead of raising, which would turn the response into a 500.
    """
    window_a = TimeWindow(start_year=2015, end_year=2020)
    window_b = TimeWindow(start_year=2018, end_year=2020)
    spec_a = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValA")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_a"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        requested_window=window_a,
    )
    spec_b = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValB")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_b"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        requested_window=window_b,
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


def test_combine_regions_keeps_window_refusal_parts_and_effective_regions_distinct():
    """A same-five-tuple window refusal keeps each visible part and its own RegionResult.

    `collapse_same_shape` alone determines that this group is not mergeable.
    The combiner must not replace the refused member's response or effective
    region with the first member just because they share a five-tuple.
    """
    window_a = TimeWindow(start_year=2015, end_year=2020)
    window_b = TimeWindow(start_year=2018, end_year=2020)
    spec_a = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValA")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_a"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        requested_window=window_a,
    )
    spec_b = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValB")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_b"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        requested_window=window_b,
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    response, effective = combine_regions(
        [r_a, r_b],
        query="var A and var B",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )

    assert [part.spec.spec_id for part in response.parts] == [spec_a.spec_id, spec_b.spec_id]
    assert all(
        BINDING_FOLDED not in [warning.code for warning in part.warnings] for part in response.parts
    )
    assert effective == [r_a, r_b]
    assert effective[0] is r_a
    assert effective[1] is r_b


# ---------------------------------------------------------------------------
# collapse_same_shape — facet identity (definition.identity.facets)
# ---------------------------------------------------------------------------


def test_collapse_same_shape_same_facets_merge_carries_facets_forward():
    """Two members bound to the identical unit facet merge, and the merged
    spec's identity carries that facet forward -- both in
    definition.identity.facets and in a spec_id distinct from the same merge
    with no facet bound at all, proving the facet actually fed the hash."""
    unit_facets = {"unit": IdentityBindingValue(ref="unit/USDollar")}
    spec_a = _spec_with_facet(_PERSON_SHAPE, "ValA", "sv_a", unit_facets)
    spec_b = _spec_with_facet(_PERSON_SHAPE, "ValB", "sv_b", unit_facets)
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec
    assert merged.definition.identity.facets == unit_facets

    unconstrained_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv_a")
    unconstrained_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv_b")
    unconstrained_effective, _ = collapse_same_shape(
        [
            _region(unconstrained_a, "var A", earliest_index=0),
            _region(unconstrained_b, "var B", earliest_index=1),
        ]
    )
    assert merged.spec_id != unconstrained_effective[0].spec.spec_id


def test_collapse_same_shape_refuses_merge_on_differing_facets():
    """Two members bound to DIFFERENT unit facets must not merge -- collapsing
    them would erase one member's own facet constraint under a single merged
    identity that names only one of the two units actually asked for. Both
    members survive separately, same as any other eligibility refusal."""
    facets_a = {"unit": IdentityBindingValue(ref="unit/USDollar")}
    facets_b = {"unit": IdentityBindingValue(ref="unit/Euro")}
    spec_a = _spec_with_facet(_PERSON_SHAPE, "ValA", "sv_a", facets_a)
    spec_b = _spec_with_facet(_PERSON_SHAPE, "ValB", "sv_b", facets_b)
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


@pytest.mark.parametrize(
    ("open_a", "open_b", "merged_open_roles"),
    [
        (("origin",), ("origin",), ["origin"]),
        (("origin",), (), None),
        (("origin",), ("destination",), None),
    ],
    ids=["equal", "one-open", "different"],
)
def test_collapse_same_shape_merges_only_members_asking_the_same_open_roles(
    open_a, open_b, merged_open_roles
):
    """The merged identity states one set of open roles for the group, so members that
    ask different ones stay apart, and members that ask the same carry them forward."""
    r_a = _region(
        _spec_with_facet(_PERSON_SHAPE, "ValA", "sv_a", {}, open_roles=open_a),
        "var A",
        earliest_index=0,
    )
    r_b = _region(
        _spec_with_facet(_PERSON_SHAPE, "ValB", "sv_b", {}, open_roles=open_b),
        "var B",
        earliest_index=1,
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    if merged_open_roles is None:
        assert effective == [r_a]
        assert residual == [("var B", 1)]
    else:
        (merged,) = effective
        assert residual == []
        assert merged.spec.definition.identity.open_roles == merged_open_roles


def test_collapse_same_shape_merges_when_neither_member_has_facets():
    """When no member carries a facet binding, the merge proceeds exactly as
    before and the merged identity carries no facets member -- the facet
    eligibility gate must not affect the no-facets case."""
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    assert effective[0].spec.definition.identity.facets is None


# ---------------------------------------------------------------------------
# collapse_same_shape — resolved facet state (resolved.facets,
# resolved.facet_options, resolved.unit_lexicon_version)
# ---------------------------------------------------------------------------


def test_collapse_same_shape_same_resolved_facets_merge_carries_state_forward():
    """Equal facets merge, and the merged Spec keeps its lexicon version."""
    resolved_facets = {"unit": ResolvedFacet(basis="bound", ref="unit/USDollar", label="US Dollar")}
    spec_a = _spec_with_resolved_facets(
        _PERSON_SHAPE, "ValA", "sv_a", resolved_facets, None, "lexicon_v1"
    )
    spec_b = _spec_with_resolved_facets(
        _PERSON_SHAPE, "ValB", "sv_b", resolved_facets, None, "lexicon_v1"
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec
    assert merged.resolved.facets == resolved_facets
    assert merged.resolved.unit_lexicon_version == "lexicon_v1"
    assert merged.resolved.facet_options == []


def _unit_options(*units: str) -> list[FacetOptions]:
    return [
        FacetOptions(
            axis="unit",
            options=[
                SlotOption(value=SlotValue(ref=_gr(unit), value_kind="enum_value"))
                for unit in units
            ],
            truncated=False,
            source="observed",
        )
    ]


def _unit_facet(unit: str) -> dict:
    return {
        "facetId": unit,
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "dates": ["2020"],
        "provenanceId": "dc/base/TestSource",
        "unit": unit,
    }


# Each member's StatVar carries US dollars and one currency of its own, so only US dollars is
# a unit every merged StatVar carries.
_MERGE_GRAPH = FakeGraph(
    obs={
        f"sv_a|{_KEN}": [_unit_facet("unit/USDollar"), _unit_facet("unit/Euro")],
        f"sv_b|{_KEN}": [_unit_facet("unit/USDollar"), _unit_facet("unit/Yen")],
    }
)


def _member(constraint_val: str, sv_dcid: str, units: tuple[str, ...], index: int):
    spec = _spec_with_resolved_facets(
        _PERSON_SHAPE, constraint_val, sv_dcid, {}, _unit_options(*units) if units else None
    )
    return dataclasses.replace(
        _region(spec, f"var {constraint_val}", earliest_index=index),
        facet_probe=facet_probe([sv_dcid], {"observationAbout": [_KEN]}, years=None),
    )


def test_merged_facet_options_offer_only_what_a_pin_on_the_merged_spec_serves():
    """Each member offers its own currency, but the merged spec answers for both StatVars,
    so a pin on either currency is refused and only US dollars is offered."""
    r_a = _member("ValA", "sv_a", ("unit/USDollar", "unit/Euro"), 0)
    r_b = _member("ValB", "sv_b", ("unit/USDollar", "unit/Yen"), 1)

    (merged,), residual = collapse_same_shape(
        [r_a, r_b],
        facet_options_for=functools.partial(
            _maybe_build_facet_options,
            graph=_MERGE_GRAPH,
            coverage=build_coverage_index([], role_names={}),
            deadline=None,
        ),
    )

    assert residual == []
    assert merged.facet_probe is not None
    assert merged.facet_probe.sv_dcids == ("sv_a", "sv_b")
    options = {
        fo.axis: [o.value.ref.ref for o in fo.options if o.value.ref is not None]
        for fo in merged.spec.resolved.facet_options
    }
    assert options["unit"] == ["unit/USDollar"]
    for unit, refused in (("unit/USDollar", False), ("unit/Euro", True), ("unit/Yen", True)):
        refusal = facet_refusal(
            _MERGE_GRAPH,
            ["sv_a", "sv_b"],
            [_KEN],
            {"unit": IdentityBindingValue(ref=unit)},
            deadline=None,
        )
        assert (refusal is not None) is refused, unit


def test_a_merged_probe_reads_every_members_places_and_elected_units():
    """`merge_facet_probes` unions the members' entities per role and their elected units per
    StatVar, so the merged spec's options are judged over every place and every election."""
    usd = {
        "unit": IdentityFacetValueRequest(
            request=IdentityUnitRequest(currency="USD", prices="current")
        )
    }

    def _probe(sv: str, place: str, elected: str | None = None):
        pins = (
            facet_pins(usd, elected_units={sv: frozenset({elected})})
            if elected is not None
            else facet_pins(None)
        )
        return facet_probe([sv], {"observationAbout": [place]}, years=None, pins=pins)

    def _options(probe, graph: FakeGraph) -> dict[str, list[str]]:
        return {
            fo.axis: [o.value.ref.ref for o in fo.options if o.value.ref is not None]
            for fo in _facet_options(
                probe.sv_dcids,
                probe.probe_entities,
                years=probe.years,
                graph=graph,
                coverage=build_coverage_index([], role_names={}),
                pins=probe.pins,
                deadline=None,
            )
        }

    merged = merge_facet_probes(
        [_probe("sv_a", _KEN, "unit/Euro"), _probe("sv_b", "country/ETH", "unit/Yen")],
        years=None,
    )
    assert merged.probe_entities == {"observationAbout": (_KEN, "country/ETH")}
    assert merged.pins.elected_units == {
        "sv_a": frozenset({"unit/Euro"}),
        "sv_b": frozenset({"unit/Yen"}),
    }

    # A where-merge of one StatVar at Kenya and Ethiopia: only Ethiopia carries yen.
    where_merge = merge_facet_probes(
        [_probe("sv_x", _KEN), _probe("sv_x", "country/ETH")], years=None
    )
    where_graph = FakeGraph(
        obs={
            f"sv_x|{_KEN}": [_unit_facet("unit/USDollar")],
            "sv_x|country/ETH": [_unit_facet("unit/USDollar"), _unit_facet("unit/Yen")],
        }
    )
    assert _options(where_merge, where_graph)["unit"] == ["unit/USDollar", "unit/Yen"]

    # Two StatVars at Kenya under a US dollar request, each electing its own native unit.
    usd_merge = merge_facet_probes(
        [_probe("sv_a", _KEN, "unit/Euro"), _probe("sv_b", _KEN, "unit/Yen")], years=None
    )
    usd_graph = FakeGraph(
        obs={
            f"sv_a|{_KEN}": [_unit_facet("unit/Euro")],
            f"sv_b|{_KEN}": [_unit_facet("unit/Yen")],
        }
    )
    assert _options(usd_merge, usd_graph)["provenance"] == ["dc/base/TestSource"]


def test_a_merge_whose_members_offer_facet_options_needs_a_reader():
    r_a = _member("ValA", "sv_a", ("unit/USDollar",), 0)
    r_b = _member("ValB", "sv_b", ("unit/USDollar",), 1)

    with pytest.raises(ValueError, match="facet_options_for"):
        collapse_same_shape([r_a, r_b])


class _FacetReadFailingGraph(FakeGraph):
    def observation_facets_batch(self, *args, **kwargs):
        raise GraphInfraError("facet read failed")


def test_a_failed_merged_facet_read_warns_on_the_merged_spec():
    """The merged spec's facet options cannot be read, so its warning names that spec."""
    r_a = _member("ValA", "sv_a", ("unit/USDollar", "unit/Euro"), 0)
    r_b = _member("ValB", "sv_b", ("unit/USDollar", "unit/Yen"), 1)

    (merged,), _residual = collapse_same_shape(
        [r_a, r_b],
        facet_options_for=functools.partial(
            _maybe_build_facet_options,
            graph=_FacetReadFailingGraph(),
            coverage=build_coverage_index([], role_names={}),
            deadline=None,
        ),
    )

    assert (OPTION_COVERAGE_PROBE_FAILED, merged.spec.spec_id) in [
        (w.code, w.spec_id) for w in merged.warnings
    ]


def test_a_multi_variable_answer_offers_the_merged_facet_options():
    """The multi-variable entry merges the two members and reads the merged spec's facet
    options, so the answer offers US dollars, the one currency both StatVars serve."""
    r_a = _member("ValA", "sv_a", ("unit/USDollar", "unit/Euro"), 0)
    r_b = _member("ValB", "sv_b", ("unit/USDollar", "unit/Yen"), 1)
    params = EntryParams(
        deadline=time.monotonic() + 30,
        max_candidates=5,
        pac=True,
        stream_option_coverage=False,
        include_option_coverage=True,
        stream_observations=False,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        coverage=empty_coverage_snapshot(),
        binder=GeminiBinder(),
    )

    async def _response():
        async for event in _stream_combined_regions(
            [r_a, r_b],
            initial_llm_usage=None,
            query="a and b",
            extra_warnings=[],
            echo=None,
            timed_out=False,
            graph=_MERGE_GRAPH,
            start_ms=0,
            params=params,
        ):
            if isinstance(event, ResponseEvent):
                return event.response
        raise AssertionError("no response event")

    response = asyncio.run(_response())
    unit_options = [
        [o.value.ref.ref for o in fo.options if o.value.ref is not None]
        for part in response.parts
        if getattr(part, "spec", None) is not None
        for fo in part.spec.resolved.facet_options
        if fo.axis == "unit"
    ]
    assert unit_options and all(refs == ["unit/USDollar"] for refs in unit_options)


def test_collapse_same_shape_refuses_merge_on_differing_resolved_facets():
    """Different resolved facets block merging even when both identity facet sets are empty."""
    facets_a = {"unit": ResolvedFacet(basis="bound", ref="unit/USDollar", label="US Dollar")}
    facets_b = {"unit": ResolvedFacet(basis="bound", ref="unit/Euro", label="Euro")}
    spec_a = _spec_with_resolved_facets(_PERSON_SHAPE, "ValA", "sv_a", facets_a)
    spec_b = _spec_with_resolved_facets(_PERSON_SHAPE, "ValB", "sv_b", facets_b)
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


def test_collapse_same_shape_refuses_merge_on_differing_unit_lexicon_version():
    """A differing unit lexicon version blocks merging because the lexicon build is process-wide."""
    facets = {"unit": ResolvedFacet(basis="bound", ref="unit/USDollar", label="US Dollar")}
    spec_a = _spec_with_resolved_facets(
        _PERSON_SHAPE, "ValA", "sv_a", facets, unit_lexicon_version="lexicon_v1"
    )
    spec_b = _spec_with_resolved_facets(
        _PERSON_SHAPE, "ValB", "sv_b", facets, unit_lexicon_version="lexicon_v2"
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


def test_collapse_same_shape_merges_declaration_version_onto_merged_spec():
    """The merged Spec keeps the declaration version shared by both members."""
    spec_a = _spec(_PERSON_SHAPE, "ValA", "sv_a")
    spec_a = _with_declaration_version(spec_a, "index_v1")
    spec_b = _spec(_PERSON_SHAPE, "ValB", "sv_b")
    spec_b = _with_declaration_version(spec_b, "index_v1")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    assert effective[0].spec.resolved.declaration_version == "index_v1"


def test_collapse_same_shape_refuses_merge_on_differing_declaration_version():
    """The process-wide code index must have one declaration version for every member."""
    spec_a = _spec(_PERSON_SHAPE, "ValA", "sv_a")
    spec_a = _with_declaration_version(spec_a, "index_v1")
    spec_b = _spec(_PERSON_SHAPE, "ValB", "sv_b")
    spec_b = _with_declaration_version(spec_b, "index_v2")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


def test_collapse_same_shape_merge_preserves_slot_expansion():
    """The merge replaces only the slot binding; its expansion metadata must survive."""
    expansion = Expansion(added=[GraphRef(ref="added/x", label="X")])
    spec_a = _spec(_PERSON_SHAPE, "ValA", "sv_a")
    patched_slots = [
        s.model_copy(update={"expansion": expansion})
        if s.key.property is not None and s.key.property.ref == "constraintProp"
        else s
        for s in spec_a.resolved.slots
    ]
    spec_a = spec_a.model_copy(
        update={"resolved": spec_a.resolved.model_copy(update={"slots": patched_slots})}
    )
    spec_b = _spec(_PERSON_SHAPE, "ValB", "sv_b")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged_slot = next(
        s
        for s in effective[0].spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == "constraintProp"
    )
    assert merged_slot.expansion == expansion


def test_collapse_same_shape_no_resolved_facet_state_merge_unchanged():
    """Members without resolved facet state still merge with empty defaults."""
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec
    assert merged.resolved.facets == {}
    assert merged.resolved.facet_options == []
    assert merged.resolved.unit_lexicon_version is None
    assert merged.resolved.declaration_version is None


def test_collapse_same_shape_merges_differing_fallback_resolved_facets():
    """Different fallback picks merge because each applies to its own series.
    The merged facet records `selection="per_series"` instead of one member's pick.
    """
    facets_a = {
        "measurementMethod": ResolvedFacet(basis="fallback", ref="method/A", label="Method A")
    }
    facets_b = {
        "measurementMethod": ResolvedFacet(basis="fallback", ref="method/B", label="Method B")
    }
    spec_a = _spec_with_resolved_facets(_PERSON_SHAPE, "ValA", "sv_a", facets_a)
    spec_b = _spec_with_resolved_facets(_PERSON_SHAPE, "ValB", "sv_b", facets_b)
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec
    assert merged.resolved.facets == {
        "measurementMethod": ResolvedFacet(basis="fallback", selection="per_series")
    }


def test_collapse_same_shape_refuses_merge_on_differing_bound_facets_but_not_fallback():
    """A bound-facet mismatch blocks merging, regardless of fallback differences."""
    facets_a = {
        "unit": ResolvedFacet(basis="bound", ref="unit/USDollar", label="US Dollar"),
        "measurementMethod": ResolvedFacet(basis="fallback", ref="method/A", label="Method A"),
    }
    facets_b = {
        "unit": ResolvedFacet(basis="bound", ref="unit/Euro", label="Euro"),
        "measurementMethod": ResolvedFacet(basis="fallback", ref="method/B", label="Method B"),
    }
    spec_a = _spec_with_resolved_facets(_PERSON_SHAPE, "ValA", "sv_a", facets_a)
    spec_b = _spec_with_resolved_facets(_PERSON_SHAPE, "ValB", "sv_b", facets_b)
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == [("var B", 1)]


def test_collapse_same_shape_merge_outcome_invariant_to_facet_population():
    """Observation enrichment must not change merge eligibility or the resulting BindingSet."""
    facets_a = {
        "measurementMethod": ResolvedFacet(basis="fallback", ref="method/A", label="Method A")
    }
    facets_b = {
        "measurementMethod": ResolvedFacet(basis="fallback", ref="method/B", label="Method B")
    }

    def _merged_what_dcids(facets_a: dict, facets_b: dict) -> frozenset[str]:
        spec_a = _spec_with_resolved_facets(_PERSON_SHAPE, "ValA", "sv_a", facets_a)
        spec_b = _spec_with_resolved_facets(_PERSON_SHAPE, "ValB", "sv_b", facets_b)
        r_a = _region(spec_a, "var A", earliest_index=0)
        r_b = _region(spec_b, "var B", earliest_index=1)
        effective, residual = collapse_same_shape([r_a, r_b])
        assert residual == []
        merged_slot = next(s for s in effective[0].spec.resolved.slots if s.key.axis == "what")
        return frozenset(v.ref.ref for v in merged_slot.binding.values if v.ref)

    populated = _merged_what_dcids(facets_a, facets_b)
    empty = _merged_what_dcids({}, {})

    assert populated == empty == frozenset({"ValA", "ValB"})


def test_a_merged_relaunch_judges_options_over_the_merged_probe():
    """The trailing relaunch of a merged region receives the merged probe, so streamed facet
    options cover every member's StatVars, as the buffered options do."""
    seen: list = []

    def _relaunch(spec_id, slot_options, facet_probe=None):
        seen.append(facet_probe)
        return None

    r_a = dataclasses.replace(_member("ValA", "sv_a", (), 0), slot_options_relaunch=_relaunch)
    r_b = _member("ValB", "sv_b", (), 1)
    effective, _residual = collapse_same_shape([r_a, r_b])

    _reconcile_after_collapse([r_a, r_b], effective)

    assert [probe.sv_dcids for probe in seen] == [("sv_a", "sv_b")]
    assert dict(seen[0].probe_entities) == {"observationAbout": (_KEN,)}


def test_a_merged_relaunch_streams_options_over_every_members_places():
    """The real relaunch of a where-merge judges its streamed options over the merged probe,
    not the survivor's own: Kenya carries US dollars and Ethiopia US dollars and yen, so the
    unit event lists both."""
    graph = FakeGraph(
        obs={
            f"sv_x|{_KEN}": [_unit_facet("unit/USDollar")],
            "sv_x|country/ETH": [_unit_facet("unit/USDollar"), _unit_facet("unit/Yen")],
        }
    )
    kenya = facet_probe(["sv_x"], {"observationAbout": [_KEN]}, years=None)
    ethiopia = facet_probe(["sv_x"], {"observationAbout": ["country/ETH"]}, years=None)
    state = SimpleNamespace(
        date_request=None,
        graph=graph,
        sdmx=None,
        materialise_graph=None,
        coverage=SimpleNamespace(index=build_coverage_index([], role_names={})),
        deadline=None,
    )

    async def _events():
        async with scoped_concurrency():
            return await _relaunch_slot_options(
                "spec_merged",
                [],
                merge_facet_probes([kenya, ethiopia], years=None),
                state=cast(Any, state),
                shape_draft=cast(Any, None),
                bindings=[],
                entities={},
                place_dcids=None,
                entity_options=None,
                default_facet_probe=kenya,
                facets_pinned=False,
                variable_labels=NO_LABELS,
                open_axes=frozenset(),
            )

    events, _warnings = asyncio.run(_events())
    assert [
        [o.value.ref.ref for o in event.options if o.value.ref is not None]
        for event in events
        if isinstance(event, FacetOptionsEvent) and event.axis == "unit"
    ] == [["unit/USDollar", "unit/Yen"]]
