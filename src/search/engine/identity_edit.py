"""Validate a resubmit edit against its optional base identity.

Checks verify the base hash, preserve its declared slot keys, account for
removed facets and entity roles, reject shape changes, and refuse pinned
StatVar identities. These checks use parsed request models and the packaged
index's dataset declarations; they never merge base state into the posted
identity. The resubmit and options doors share the same validation functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dfo_spec_kit import identity as _identity

from materialise.roles import roles_by_axis
from search.engine.errors import EngineInputError
from search.engine.spec_id import build_spec_identity
from search.models import (
    BaseIdentity,
    BindingSet,
    BindingUnbound,
    BindingValue,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    IdentityBindingExact,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityShape,
    IdentityShapePin,
    IdentitySlot,
    RemovedParts,
    RequestedWindow,
    Slot,
    SlotKey,
    SlotValue,
)

# The facet axis vocabulary `removed.facets` may name -- every axis this
# engine models (see spec_id.bound_facet_axes / _facets_member for the
# source-slot-derived provenance axis; measurementMethod/unit arrive in
# ResubmitIdentityPayload.identity.facets -- see check_edit_incomplete below).
_FACET_AXES = frozenset({"provenance", "measurementMethod", "unit"})

# The identity core's fixed entity-role vocabulary (see spec_id.entity_role_key).
_ENTITY_ROLE_KEYS = frozenset({"observationAbout", "origin", "destination"})


def check_no_exact_bindings(identity: BaseIdentity, *, pointer: str) -> None:
    """Refuse the first `exact` slot binding at its JSON pointer.

    Search readers do not handle this identity-core binding; only dataset
    readers do. Run this before core validation, which accepts `exact`.
    Callers must check both the posted identity and its base identity.
    """
    for i, slot in enumerate(identity.slots):
        if isinstance(slot.binding, IdentityBindingExact):
            raise EngineInputError(
                f"slot {slot.property!r} carries an 'exact' binding; no Search "
                "door serves it -- only the datasets doors read it",
                code="exact_binding_unsupported",
                status=501,
                pointer=f"{pointer}/slots/{i}/binding",
            )


def validate_edited_identity(identity: BaseIdentity, *, pointer: str = "/input/identity") -> None:
    """Validate a posted identity through the identity core.

    `BaseIdentity` mirrors most of the core schema but deliberately does not
    duplicate each cross-member invariant -- duplicate slot properties
    (collapsed silently by a naive dict build rather than refused) chief
    among them. The posted identity must therefore pass the authoritative
    core before any door reads its shape, slots, entities, window, or
    facets. `pointer` defaults to the resubmit path's own nested identity field;
    a caller whose posted identity lives under a different wire shape (e.g.
    options/resolve.py's `OptionsRequest.identity`) passes its own pointer,
    the same convention `check_described_arm` already follows.

    Refuses `exact_binding_unsupported` (501) first, ahead of the core
    check, for any slot bound `exact` (`check_no_exact_bindings`).
    """
    check_no_exact_bindings(identity, pointer=pointer)
    try:
        _identity.validate_identity_core(identity.core_document())
    except ValueError as exc:
        raise EngineInputError(
            f"identity is not a well-formed identity document: {exc}",
            code="request_invalid",
            pointer=pointer,
        ) from exc


def verify_base_identity(
    base_spec_id: str | None,
    base_identity: BaseIdentity | None,
) -> None:
    """Verify the posted base pair, or return when neither value is present.

    `base_identity` without `base_spec_id` raises `request_invalid`/400;
    `base_spec_id` without an identity raises `base_not_found`/422 because no
    spec registry can resolve a bare id. A hash mismatch raises
    `base_identity_mismatch`/422. The core revalidates the client-supplied
    identity so schema gaps become `request_invalid`/400, not server errors. That does not
    hold for engine-built projections, where a rejection is an engine bug and stays a 500.
    Pydantic rejects malformed wire shapes before this function runs.
    """
    if base_identity is None:
        if base_spec_id is not None:
            raise EngineInputError(
                "base_spec_id was posted with no base_identity to verify it "
                "against; no spec registry exists to look up a bare base_spec_id",
                code="base_not_found",
                status=422,
                pointer="/input/base_spec_id",
            )
        return None
    if base_spec_id is None:
        raise EngineInputError(
            "base_identity was posted with no base_spec_id to verify it against",
            code="request_invalid",
            pointer="/input/base_identity",
        )
    try:
        computed = _identity.spec_id_from_value(base_identity.core_document())
    except ValueError as exc:
        # Backstop for any identity-core rule BaseIdentity's schema does not
        # mirror. Deliberately broad: the argument is the CLIENT posted this
        # document, so a rejection from the core -- the authority on identity
        # well-formedness -- is bad input, not a server fault. IdentityError
        # subclasses ValueError, so this covers the core's own error type too.
        raise EngineInputError(
            f"base_identity is not a well-formed identity document: {exc}",
            code="request_invalid",
            pointer="/input/base_identity",
        ) from exc
    if computed != base_spec_id:
        raise EngineInputError(
            "base_identity does not canonicalise and hash to the posted base_spec_id",
            code="base_identity_mismatch",
            status=422,
            pointer="/input/base_identity",
        )
    return None


def check_shape_unchanged(base_identity: BaseIdentity | None, posted_shape: IdentityShape) -> None:
    """A base-tracked edit never changes shape: a different shape is a new
    construction, posted without a base. Compares the two shape union values
    structurally (pydantic model equality across every member of whichever arm
    each side carries) -- never by a display slug. No-op when no base was posted.
    """
    if base_identity is None:
        return
    if base_identity.shape != posted_shape:
        raise EngineInputError(
            "shape differs from the base's; a shape change is a new "
            "construction, posted without a base",
            code="shape_mismatch",
            status=422,
            pointer="/input/identity/shape",
        )


def check_described_arm(shape: IdentityShape, *, pointer: str = "/input/shape") -> None:
    """Refuse a pinned-ask shape (`kind="stat_var"`) posted to a Search door.

    Search doors (a dataset_resubmit identity entry, options) resolve described
    asks only -- the pin arm is Datasets' statvar_set identity, never a Search
    input. `pointer` defaults to the resubmit path's identity shape
    (`ResubmitIdentityPayload.identity.shape`); a caller whose posted value lives under a
    different wire shape (e.g. options/resolve.py's `OptionsRequest.identity.shape`)
    passes its own pointer so the refusal lands where the client can act on it.
    """
    if isinstance(shape, IdentityShapePin):
        raise EngineInputError(
            "a pinned StatVar identity (kind='stat_var') was posted to a Search "
            "door; Search doors resolve described asks only",
            code="request_invalid",
            pointer=pointer,
        )


def check_slots_declared(
    base_slots: Sequence[IdentitySlot],
    posted_slots: Sequence[IdentitySlot],
) -> None:
    """Require every base-declared what/how slot in the posted identity.

    Compare properties only; no graph read is needed. A posted key may be
    unbound to signal no narrowing, but omitting the key is ambiguous. Extra
    posted keys are allowed. Empty `base_slots` is a no-op. A missing key
    raises `request_invalid`/400 for the first property in base order.
    """
    posted_props = {slot.property for slot in posted_slots}
    for base_slot in base_slots:
        if base_slot.property in posted_props:
            continue
        detail = (
            "the posted identity is missing a base-declared slot property "
            f"({base_slot.property!r}); every what/how slot the base identity "
            "declares must appear, at minimum as unbound"
        )
        raise EngineInputError(
            detail,
            code="request_invalid",
            pointer="/input/identity/slots",
            detail=detail,
        )


def validate_removed(
    removed: RemovedParts | None,
    base_spec_id: str | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Validate `removed` and return its two categories as canonical-name sets:
    (facets_removed, entities_removed).

    Raises EngineInputError, always code="request_invalid" (400) -- a
    malformed or unrecognised `removed` entry is a shape violation, not a
    graph-resolution failure (`invalid_role_ref` is reserved for a structured
    role ref the graph does not hold; nothing here ever touches the graph):
      - `removed` is posted (non-None) with no base_spec_id;
      - either category names the same canonical part twice;
      - a `facets` entry is not a recognised facet axis name, or an
        `entities` entry is not a recognised entity role key.

    Returns (frozenset(), frozenset()) when removed is None, or names nothing
    in either category (the common case, and the vacuous {} a client might
    post to say "removing nothing" explicitly).
    """
    if removed is None:
        return frozenset(), frozenset()
    facets = removed.facets or []
    entities = removed.entities or []
    if not facets and not entities:
        return frozenset(), frozenset()
    if base_spec_id is None:
        raise EngineInputError(
            "removed was posted with no base_spec_id; there is no base to remove parts from",
            code="request_invalid",
            pointer="/input/removed",
        )
    if len(facets) != len(set(facets)):
        raise EngineInputError(
            "removed.facets carries a duplicate entry",
            code="request_invalid",
            pointer="/input/removed/facets",
        )
    if len(entities) != len(set(entities)):
        raise EngineInputError(
            "removed.entities carries a duplicate entry",
            code="request_invalid",
            pointer="/input/removed/entities",
        )
    for name in facets:
        if name not in _FACET_AXES:
            raise EngineInputError(
                f"{name!r} is not a recognised facet axis name",
                code="request_invalid",
                pointer="/input/removed/facets",
            )
    for name in entities:
        if name not in _ENTITY_ROLE_KEYS:
            raise EngineInputError(
                f"{name!r} is not a recognised entity role key",
                code="request_invalid",
                pointer="/input/removed/entities",
            )
    return frozenset(facets), frozenset(entities)


def check_edit_incomplete(
    base_identity: BaseIdentity | None,
    *,
    posted_facets: dict[str, IdentityFacetBinding] | None = None,
    posted_entity_roles: set[str],
    removed_facets: frozenset[str],
    removed_entities: frozenset[str],
) -> None:
    """Diff the posted identity's facets and entities against the base's.

    Slots carry no accounting here: every slot property the base identity
    declares always appears in the posted identity (see
    `check_slots_declared`), at minimum unbound, and that unbound state is
    itself the removal signal. Slots need no `removed` acknowledgement.

    `posted_facets` is the nested identity's canonical sparse facet map. A
    present key is bound by schema construction, including provenance, so
    there is no wire-slot adapter to reconstruct before accounting.

    Raises EngineInputError:
      - code="edit_incomplete" (422) for a base-bound facet axis or entity
        role that arrives neither re-sent (still bound in the posted
        facets/entity roles) nor named in `removed`.
      - code="request_invalid" (400) for a `removed` entry naming a part the
        posted definition STILL binds -- a contradictory accounting (you
        can't both keep it and remove it).

    No-op when no base was posted -- nothing to account for.
    """
    if base_identity is None:
        return
    posted_facet_axes = frozenset(posted_facets or ())
    # base_identity is the identity-core projection (see models.BaseIdentity):
    # its own `facets`/`entities` members are already the canonical bound-part
    # names (a sparse dict keyed by facet axis, a dict keyed by role), so
    # there is nothing left to derive from a Slot/Entity list here -- read them
    # straight off the base.
    for axis in base_identity.facets or {}:
        if axis in removed_facets:
            if axis in posted_facet_axes:
                raise EngineInputError(
                    f"{axis!r} is named in removed.facets but the posted definition still binds it",
                    code="request_invalid",
                    pointer="/input/removed/facets",
                )
            continue
        if axis not in posted_facet_axes:
            detail = (
                f"{axis!r} was bound in the base and arrives neither re-sent "
                "nor named in removed.facets"
            )
            raise EngineInputError(
                detail,
                code="edit_incomplete",
                status=422,
                pointer="/input/identity/facets",
                detail=detail,
            )
    for role in base_identity.entities:
        if role in removed_entities:
            if role in posted_entity_roles:
                raise EngineInputError(
                    f"{role!r} is named in removed.entities but the posted "
                    "definition still binds it",
                    code="request_invalid",
                    pointer="/input/removed/entities",
                )
            continue
        if role not in posted_entity_roles:
            detail = (
                f"{role!r} was bound in the base and arrives neither re-sent "
                "nor named in removed.entities"
            )
            raise EngineInputError(
                detail,
                code="edit_incomplete",
                status=422,
                pointer="/input/identity/entities",
                detail=detail,
            )


def _legacy_slots_with_provenance(
    slots: list[Slot],
    facets: Mapping[str, IdentityFacetBinding] | None,
) -> tuple[list[Slot], dict[str, IdentityFacetBinding] | None]:
    """Move legacy ``facets.provenance`` onto the canonical source slot.

    ``build_spec_identity`` intentionally hashes provenance only from a
    source-axis wire Slot.  The legacy corpus adapter predates the nested
    identity request and can still carry both forms, so it performs the same
    single-channel rule before minting: an existing *bound* source slot and a
    provenance facet are contradictory; an unbound source slot is filled, or
    one is appended when the legacy record has none.  measurementMethod and
    unit remain explicit facets.
    """
    provenance = (facets or {}).get("provenance")
    explicit_facets = {
        axis: binding for axis, binding in (facets or {}).items() if axis != "provenance"
    }
    if provenance is None:
        return slots, explicit_facets or None

    source_slots = [slot for slot in slots if slot.key.axis == "source"]
    if any(not isinstance(slot.binding, BindingUnbound) for slot in source_slots):
        raise ValueError("a source-axis slot and facets.provenance cannot both be bound")

    if isinstance(provenance, IdentityBindingValue):
        values = [
            SlotValue(ref=GraphRef(dcid=provenance.ref, label=provenance.ref), value_kind="source")
        ]
    elif isinstance(provenance, IdentityBindingSet):
        values = [
            SlotValue(ref=GraphRef(dcid=dcid, label=dcid), value_kind="source")
            for dcid in provenance.refs
        ]
    else:  # IdentityFacetBinding excludes unbound and request-form provenance.
        raise ValueError(f"invalid provenance facet binding: {type(provenance)!r}")

    binding = BindingValue(value=values[0]) if len(values) == 1 else BindingSet(values=values)
    if source_slots:
        return (
            [
                slot.model_copy(update={"binding": binding}) if slot.key.axis == "source" else slot
                for slot in slots
            ],
            explicit_facets or None,
        )
    return (
        [
            *slots,
            Slot(
                key=SlotKey(axis="source", property=None, label="data source"),
                binding=binding,
            ),
        ],
        explicit_facets or None,
    )


def synthesize_base(
    shape: Mapping[str, object],
    slots: list[Slot],
    *,
    entity_dcids: list[str] | None,
    origin_dcids: list[str] | None = None,
    resolved_stat_var_dcids: list[str] | None,
    requested_window: RequestedWindow | None = None,
    declared_roles: tuple[str, ...] | None = None,
    facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> tuple[str | None, BaseIdentity | None]:
    """Build a base identity from legacy promote fields for corpus evaluation.

    `shape` is passed unchanged to `build_spec_identity`. With no declared
    non-subject roles, `entity_dcids` become `observationAbout` entities;
    otherwise they become destination entities. `origin_dcids` add origin
    entities. When `declared_roles` is absent, the role-free legacy shape uses
    `observationAbout`.

    Return `(None, None)` when no anchor or StatVar was supplied, no declared
    role can hold the posted entities, or an unsupported what/how slot value
    prevents hashing. Other invariant failures propagate. The returned base
    is only a diff baseline; callers carry the complete entities and
    `resolved_stat_var_dcids` on the resubmit payload. StatVar dcids are not
    part of the hashed `BaseIdentity`.
    """
    if not entity_dcids and not origin_dcids and not resolved_stat_var_dcids:
        return None, None
    slots, explicit_facets = _legacy_slots_with_provenance(slots, facets)
    entities: list[Entity] = []
    non_subject_roles = () if declared_roles in (None, ("observationAbout",)) else declared_roles
    if not non_subject_roles and not origin_dcids:
        entities.extend(
            Entity(ref=GraphRef(dcid=dcid, label=dcid), role=EntityRoleSubject())
            for dcid in entity_dcids or []
        )
    else:
        roles_by_direction = roles_by_axis(non_subject_roles)
        # Legacy corpus records predate declared role metadata. An origin
        # member unambiguously marks their standard-promote seed directional;
        # its canonical identity role keys are the axes themselves.
        destination_name = roles_by_direction.get("destination") or (
            "destination" if entity_dcids and origin_dcids else None
        )
        origin_name = roles_by_direction.get("origin") or ("origin" if origin_dcids else None)
        if (entity_dcids and destination_name is None) or (origin_dcids and origin_name is None):
            return None, None
        if destination_name is not None:
            destination_role = EntityRoleDirectional(
                role=GraphRef(dcid=destination_name, label=destination_name), axis="destination"
            )
            entities.extend(
                Entity(ref=GraphRef(dcid=dcid, label=dcid), role=destination_role)
                for dcid in entity_dcids or []
            )
        if origin_name is not None:
            origin_role = EntityRoleDirectional(
                role=GraphRef(dcid=origin_name, label=origin_name), axis="origin"
            )
            entities.extend(
                Entity(ref=GraphRef(dcid=dcid, label=dcid), role=origin_role)
                for dcid in origin_dcids or []
            )
    try:
        base_spec_id, base_identity = build_spec_identity(
            shape,
            slots,
            entities=entities or None,
            requested_window=requested_window,
            explicit_facets=explicit_facets,
        )
    except ValueError as exc:
        # Narrowed to the one documented case: a literal or time-window
        # SlotValue on a what/how axis, which spec_id.py's
        # `_slot_binding_member` raises with this exact phrase for both the
        # BindingValue and BindingSet arms. Every other ValueError this
        # adapter can raise (a slot missing its constraint property, 2+
        # source-axis slots, an empty requested_window, an unknown binding
        # type) is a real invariant violation upstream and must propagate,
        # never be swallowed as "nothing to synthesize".
        if "must never reach identity" not in str(exc):
            raise
        return None, None
    return base_spec_id, base_identity
