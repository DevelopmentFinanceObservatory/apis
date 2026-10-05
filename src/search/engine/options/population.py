"""Derive an options request's StatVar population and declared role signature.

The packaged index supplies the population of an identity whose membership it
decides (`index_decides_membership`, the Datasets mint's test). Its synthetic
`ShapeDraft` sets `sv_arc_facts_complete=False` so `shape_member_facts` reads
the family the index resolves; an empty indexed result remains empty. Other
identities use deterministic `statvar_members.enumerate_members` graph
enumeration with registry-defined absent readings, as the mint does.

Index populations are uncapped; the ODA-to-Kenya family has 3,812 members.
Graph enumeration uses `CANDIDATE_CAP` and `settings().options_member_cap`.
An empty complete result raises `population_empty`; an empty truncated result
raises `scope_too_broad`. Confirmed members from truncated pages are retained,
and `truncated` is false when antichain reduction proves the set complete.
Role disagreement raises `role_signature_incompatible`; enumeration data
integrity failures raise `request_invalid`. The search-owned
`population_empty` code is registered on `search.options` as a 422 response.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from code_index.query import dedupe_first_seen
from code_index.shared import get_index
from materialise.facets import ElectedDataset
from search.engine.config import settings
from search.engine.errors import EngineInputError, RoleSignatureError
from search.engine.retrieval.bound_values import (
    declared_role_signature,
    identity_query,
    index_decides_membership,
    shape_member_facts,
    synthetic_shape_for_identity,
)
from search.engine.retrieval.elected_dataset import ElectedDatasets, elected_datasets
from search.models import (
    BaseIdentity,
    IdentityBindingExact,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityShapeDenominatorRef,
    IdentityShapeDenominatorUnresolved,
    IdentityShapeSchema,
    IdentitySlot,
)
from statvar_members import (
    AbsentReading,
    DegenerateObservationProperties,
    MemberArcReader,
    MemberEnumerationTooLarge,
    MemberQueryUnanchored,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    SlotBinding,
    enumerate_members,
)
from statvar_members import FiveTuple as MemberFiveTuple


@dataclass(frozen=True, slots=True)
class OptionsPopulation:
    """The StatVar population and agreed role signature for one identity shape.

    `source` identifies index or graph derivation; neither source falls back
    to the other. `declared_roles` is shared by all members, including
    `("observationAbout",)` for subject-only populations.

    `truncated` is true when graph enumeration hit `CANDIDATE_CAP`, confirmed
    members, and antichain reduction could not prove the set complete. It is
    false for index results, complete pages, or pages whose antichain proves
    no unseen candidate dominates a confirmed member. That proof does not
    rule out an unreported duplicate. The surface carries this flag into
    `OptionsSet.truncated`.

    `datasets` holds the elected dataset of each member of an index-resolved family, which
    every facet read of the population is kept to (`retrieval.elected_dataset`).
    """

    stat_var_dcids: tuple[str, ...]
    declared_roles: tuple[str, ...]
    source: Literal["index", "graph"]
    truncated: bool = False
    datasets: ElectedDatasets = dataclasses.field(default_factory=dict, compare=False)


def _absent_readings() -> Mapping[str, AbsentReading]:
    """Every declared code system's `absent_reading`, keyed by `code_system`: the
    registry-backed mapping `retrieval.read_plan._absent_readings` and
    `datasets_api.adapter._absent_readings` also build, and the one every production
    `enumerate_members` call site must pass
    (`tests/conformance/test_absent_reading_call_sites.py`).
    """
    return {
        code_system: declaration.absent_reading
        for code_system, declaration in get_index().declarations.items()
    }


def _member_five_tuple(shape: IdentityShapeSchema) -> MemberFiveTuple:
    """`shape`'s five schema members, translated into `statvar_members`'
    own `FiveTuple` -- a field-for-field mirror by construction
    (`statvar_members`'s own package docstring), so this is a retag, never
    a re-derivation. The `measurement_denominator` sub-union unpacks the
    same way `retrieval.bound_values.synthetic_shape_for_identity` unpacks
    it for `ShapeDraft`: a resolved ref's own dcid, or `None` with
    `denominator_unresolved=True` for the unresolved-arc marker.
    """
    denominator = shape.measurement_denominator
    return MemberFiveTuple(
        population_type=shape.population_type,
        measured_property=shape.measured_property,
        stat_type=shape.stat_type,
        measurement_qualifier=shape.measurement_qualifier,
        measurement_denominator=(
            denominator.ref if isinstance(denominator, IdentityShapeDenominatorRef) else None
        ),
        denominator_unresolved=isinstance(denominator, IdentityShapeDenominatorUnresolved),
    )


def _member_slot_bindings(slots: Sequence[IdentitySlot]) -> dict[str, SlotBinding]:
    """`identity.slots`, retagged into `statvar_members`'s own `SlotBinding`
    triad -- the identical value/set/unbound states `IdentitySlot.binding`
    already carries, just reshaped from the wire's tagged-object form into
    `enumerate_members`'s tuple form.

    `IdentityBindingExact` has no arm here: `engine.identity_edit
    .validate_edited_identity` (run by `options/resolve.py` ahead of this
    function, at `resolve_options_target`) already refuses a posted `exact`
    slot with `exact_binding_unsupported`, so this function never sees one
    in practice. Raising rather than falling through to `("unbound",)`
    keeps that true structurally too -- an `exact` binding silently read as
    unbound would then be widened by `_widen_hierarchical_slots` as if it
    were `value`, the same failure mode `_binding_from_identity`
    (`resubmit_request.py`) guards against for the wire-binding side of the
    same union.
    """
    result: dict[str, SlotBinding] = {}
    for slot in slots:
        binding = slot.binding
        if isinstance(binding, IdentityBindingValue):
            result[slot.property] = ("value", binding.ref)
        elif isinstance(binding, IdentityBindingSet):
            result[slot.property] = ("set", tuple(binding.refs))
        elif isinstance(binding, IdentityBindingExact):
            raise ValueError(  # pragma: no cover -- unreachable: refused earlier
                f"unsupported IdentityBinding type: {type(binding)!r}"
            )
        else:
            result[slot.property] = ("unbound",)
    return result


def _widen_hierarchical_slots(slots: Mapping[str, SlotBinding]) -> dict[str, SlotBinding]:
    """Expand value and set bindings to self-inclusive code descendants.

    Leaves, flat code systems, and unknown dcids pass through unchanged.
    This accepts typed `IdentitySlot` bindings; the shared wire adapter
    accepts raw mappings, so only the widening loop is repeated here.
    """
    index = get_index()
    widened: dict[str, SlotBinding] = {}
    for prop, binding in slots.items():
        match binding:
            case ("value", ref):
                refs = index.descendants(ref)
                widened[prop] = ("set", refs) if len(refs) > 1 else binding
            case ("set", refs):
                widened[prop] = ("set", dedupe_first_seen(index.descendants(ref) for ref in refs))
            case _:
                widened[prop] = binding
    return widened


def _role_signature_incompatible(source: str, cause: Exception) -> EngineInputError:
    return EngineInputError(
        f"the {source} population's members disagree on declared entity-role "
        f"signatures, so no single role signature describes it: {cause}",
        code="role_signature_incompatible",
        status=422,
        pointer="/identity",
    )


def options_population(identity: BaseIdentity, graph: MemberArcReader) -> OptionsPopulation:
    """The StatVar population `identity`'s shape denotes, server-derived.

    `identity.shape` must already be the described arm (`IdentityShapeSchema`)
    -- the caller runs `identity_edit.check_described_arm` first, exactly as
    every other Search door does, so the pinned arm (`IdentityShapePin`)
    never reaches this function; a pinned identity has one StatVar and no
    population to derive at all.

    Raises:
        EngineInputError: the population is empty and, for a non-index
            enumeration, its own candidate page read in full
            (`code="population_empty"`, 422 -- a new Search-own code; see
            this module's own docstring for the registry-lag note), a
            non-index enumeration exceeds `settings().options_member_cap` or truncates its
            own candidate page while confirming no member
            (`code="scope_too_broad"`, 422, mirroring `options/surface.py`'s
            own use of that code for the same "too broad to enumerate"
            shape), the elected members disagree on declared entity roles on
            either source (`code="role_signature_incompatible"`,
            422, mirroring `options/resolve.py`'s own use of that code), or a
            non-index enumeration hits one of `enumerate_members`'s other
            data-integrity refusals (`code="request_invalid"`, 400).
    """
    shape = cast(IdentityShapeSchema, identity.shape)
    synthetic_shape = dataclasses.replace(
        synthetic_shape_for_identity(shape), sv_arc_facts_complete=False
    )

    source: Literal["index", "graph"]
    truncated = False
    datasets: dict[str, ElectedDataset] = {}
    if index_decides_membership(identity):
        source = "index"
        # The index decides this identity's membership, so the population is the elected
        # dataset's family, empty when the bound values match no row or the election refuses.
        member_facts, _ = shape_member_facts(synthetic_shape, identity_query(identity))
        try:
            declared_roles = declared_role_signature(member_facts)
        except RoleSignatureError as exc:
            raise _role_signature_incompatible("index-sourced", exc) from exc
        stat_var_dcids = tuple(sorted(member_facts))
        datasets = elected_datasets(member_facts)
    else:
        source = "graph"
        five_tuple = _member_five_tuple(shape)
        slots = _widen_hierarchical_slots(_member_slot_bindings(identity.slots))
        try:
            group = enumerate_members(
                five_tuple=five_tuple,
                slots=slots,
                graph=graph,
                readings=_absent_readings(),
                member_cap=settings().options_member_cap,
                role_keys=frozenset(identity.entities),
            )
        except MemberEnumerationTooLarge as exc:
            raise EngineInputError(
                f"{exc.count} StatVars are consistent with this identity, exceeding the "
                f"{exc.member_cap}-member enumeration cap; bind an additional what/how slot "
                "to narrow the population",
                code="scope_too_broad",
                status=422,
                pointer="/identity",
            ) from exc
        except MemberRoleDisagreement as exc:
            raise _role_signature_incompatible("graph-enumerated", exc) from exc
        except (
            MemberQueryUnanchored,
            MultiValuedConstraint,
            NotAStatisticalVariable,
            DegenerateObservationProperties,
        ) as exc:
            raise EngineInputError(
                str(exc),
                code="request_invalid",
                status=400,
                pointer="/identity",
            ) from exc
        if not group.members and group.truncated:
            # A truncated page confirmed no member from inside its own
            # candidate window -- not proof the identity denotes no
            # population, since a dominating member could sit past the
            # window (see `MemberGroup.truncated`'s own docstring). Reading
            # this as `population_empty` would tell the caller the identity
            # names nothing, which the graph never established; reading it
            # as `scope_too_broad`, the same code the index and directional
            # branches already raise for "too wide to tell," says only what
            # is actually known.
            raise EngineInputError(
                "the identity's raw candidate population exceeds the graph "
                "enumeration window before any consistent member was confirmed, so "
                "whether the identity denotes a real, narrower population cannot be told "
                "from an empty one; bind an additional what/how slot to narrow it",
                code="scope_too_broad",
                status=422,
                pointer="/identity",
            )
        stat_var_dcids = tuple(sorted(member.stat_var for member in group.members))
        declared_roles = group.members[0].declared_roles if group.members else ("observationAbout",)
        # A truncated page that still confirmed a member is used as-is (see
        # this module's own docstring) -- but the population it came from is
        # a sample, not a confirmed-complete membership, unless the
        # antichain reduction itself proves otherwise
        # (`group.proven_complete`, `statvar_members.enumerate
        # ._antichain_proven_complete`'s own docstring has the domination
        # argument). That state carries forward on the result rather than
        # being discarded once the scope_too_broad-vs-population_empty
        # branch above has read it.
        truncated = group.truncated and not group.proven_complete

    if not stat_var_dcids:
        raise EngineInputError(
            "the posted identity's shape denotes no StatVar population at all; "
            "there is nothing to enumerate options against",
            code="population_empty",
            status=422,
            pointer="/identity",
        )

    return OptionsPopulation(
        stat_var_dcids=stat_var_dcids,
        declared_roles=declared_roles,
        source=source,
        truncated=truncated,
        datasets=datasets,
    )
