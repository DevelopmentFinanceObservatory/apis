"""Map engine models to the spec1 identity document and mint ids through
``dfo_spec_kit.identity``. Canonicalisation, validation, and hashing live in that
package. A spec id is 37 characters: ``spec_`` plus 32 lowercase SHA-256 hex digits.

The adapter maps ``shape``, ``slots``, ``entities``, ``open_roles``, ``facets``, and
``requested_window``. The rules below explain which model fields feed each member.

The caller's requested window represents the ``when`` axis separately from the
coverage-narrowed slot. Entities represent the ``where`` axis. Hashing both would
duplicate destination identity. The ``source`` axis maps to
``facets.provenance``. What/how slots require a constraint property. A missing
property raises ``ValueError``.

Bindings map to ``value``/``ref``, ``set``/``refs``, or ``unbound``. A missing ref
raises ``ValueError``. Set taxonomy refs stay outside identity as provenance.
Inapplicable properties are graph state, reported in ``ResolvedSpec.inapplicable_slots``
and excluded from identity.

Subject entities use the ``observationAbout`` role. Directional entities use their
``origin`` or ``destination`` axis as the role key. The directional role property
records provenance. Identity normalisation sorts and deduplicates entity dcids. Absent
entities map to ``{}``. ``open_roles`` names the flow axes asked for member by member, and
the member is omitted when none is.

The source slot supplies ``facets.provenance``. Explicit facets supply
``measurementMethod`` and ``unit``. Explicit ``provenance`` entries are ignored.
The facets member is omitted when empty.

An identity records the resolved selection. A part that names a table binds that table as
``facets.provenance``. An exclusion is not an identity facet. When excluding a table changes
which table serves, the identity binds the table that serves. When it changes nothing, the
identity is the plain question's and mints the same spec_id, so the exclusion does not persist
if the serving order later changes.

A missing requested window maps to ``{"kind": "all"}``, ``"latest"`` maps to that
kind, and a ``TimeWindow`` maps to a range with its bounds. Empty ranges raise
``ValueError``.

Every identity built here is canonical (`code_index.members.canonical_identity`): each
slot an invariant of the dataset the identity names fixes binds the invariant's value, so an
answer, an alternate candidate and a suggestion naming one dataset mint one spec_id whichever
invariants their bindings hold. The module
reads the release's code index for that election and is otherwise pure apart from the
identity package call.

The id hashes the whole ``BaseIdentity`` document, dumped with null members omitted.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from dfo_spec_kit import identity

from code_index.members import canonical_identity
from code_index.shared import get_index
from search.models import (
    BaseIdentity,
    BindingSet,
    BindingUnbound,
    BindingValue,
    Entity,
    EntityRoleKey,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    RequestedWindow,
    RoleAxis,
    Slot,
    TimeWindow,
)

# Axes excluded from slot hashing are `when` (time rides on requested_window
# only), `where` (the property-less subject slot, since the entity itself is
# what identity hashes, via Spec.resolved.entities), and `source` (a facet,
# see _facets_member).
_EXCLUDED_SLOT_AXES = frozenset({"when", "where", "source"})


def _slot_binding_member(binding: object) -> dict:
    """Map a slot binding to the identity core's binding shape."""
    if isinstance(binding, BindingValue):
        ref = binding.value.ref
        if ref is None:
            raise ValueError(
                "BindingValue has no ref; a literal/time-window SlotValue must never reach identity"
            )
        return {"state": "value", "ref": ref.ref}

    if isinstance(binding, BindingSet):
        refs: list[str] = []
        for member in binding.values:
            if member.ref is None:
                raise ValueError(
                    "BindingSet member has no ref; a literal/time-window "
                    "SlotValue must never reach identity"
                )
            refs.append(member.ref.ref)
        return {"state": "set", "refs": refs}

    if isinstance(binding, BindingUnbound):
        return {"state": "unbound"}

    raise ValueError(f"Unknown binding type: {type(binding)!r}")


def _slots_member(slots: list[Slot]) -> list[dict]:
    """Build the identity core's what/how slots, sorted by property dcid.

    Sorting keeps the served document in hash order. A surviving slot without a
    constraint property raises ValueError.
    """
    result: list[dict] = []
    for slot in slots:
        if slot.key.axis in _EXCLUDED_SLOT_AXES:
            continue
        prop = slot.key.property
        if prop is None:
            raise ValueError(
                f"{slot.key.axis!r}-axis slot has no constraint property; "
                "every what/how slot must carry one"
            )
        result.append(
            {
                "property": prop.ref,
                "binding": _slot_binding_member(slot.binding),
            }
        )
    result.sort(key=lambda entry: entry["property"])
    return result


def entity_role_key(entity: Entity) -> EntityRoleKey:
    """Return ``observationAbout`` for subjects or the flow axis for directional entities."""
    role = entity.role
    if role.kind == "subject":
        return "observationAbout"
    if role.kind == "directional":
        return role.axis
    raise ValueError(f"unknown entity role kind: {role.kind!r}")


def _entities_member(entities: list[Entity] | None) -> dict[str, list[str]]:
    """Map role keys to dcids. Empty input gives ``{}``, and the package sorts buckets."""
    result: dict[str, list[str]] = {}
    for entity in entities or []:
        result.setdefault(entity_role_key(entity), []).append(entity.ref.ref)
    return result


def _identity_facet_binding_member(binding: IdentityFacetBinding) -> dict:
    """Map a measurementMethod or unit binding to the identity core's binding shape."""
    if isinstance(binding, IdentityBindingValue | IdentityFacetValueAbsent):
        return {"state": "value", "ref": binding.ref}
    if isinstance(binding, IdentityBindingSet):
        return {"state": "set", "refs": list(binding.refs)}
    if isinstance(binding, IdentityFacetValueRequest):
        return {"state": "value", "request": binding.request.model_dump(exclude_none=True)}
    raise ValueError(f"Unknown IdentityFacetBinding type: {type(binding)!r}")


def _facets_member(
    slots: list[Slot],
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> dict[str, dict]:
    """Merge source-slot provenance with explicit measurementMethod/unit facets.

    Ignores explicit provenance and returns ``{}`` when neither source contributes.
    """
    source_slots = [s for s in slots if s.key.axis == "source"]
    if len(source_slots) > 1:
        # Multiple source slots violate the shape grammar and make provenance ambiguous.
        raise ValueError(f"expected at most one source-axis slot, got {len(source_slots)}")
    source_slot = source_slots[0] if source_slots else None

    result: dict[str, dict] = {}
    if source_slot is not None and not isinstance(source_slot.binding, BindingUnbound):
        result["provenance"] = _slot_binding_member(source_slot.binding)
    for axis, binding in (explicit_facets or {}).items():
        if axis == "provenance":
            continue
        result[axis] = _identity_facet_binding_member(binding)
    return result


def bound_facet_axes(
    slots: list[Slot],
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> frozenset[str]:
    """Return bound facet axes using the same mapping as identity hashing.

    Shared with ``engine.identity_edit`` so edit accounting matches the hash.
    """
    return frozenset(_facets_member(slots, explicit_facets))


def _requested_window_member(requested_window: RequestedWindow | None) -> dict:
    """The identity core's `requested_window` tagged union."""
    if requested_window is None:
        return {"kind": "all"}
    if requested_window == "latest":
        return {"kind": "latest"}

    tw: TimeWindow = requested_window
    if tw.start_year is None and tw.end_year is None:
        # Structurally unreachable -- TimeWindow.at_least_one_bound already
        # refuses to construct this -- but this mapping fails loud rather
        # than trusting that invariant silently (see module docstring).
        raise ValueError("TimeWindow has neither start_year nor end_year")
    result: dict[str, object] = {"kind": "range"}
    if tw.start_year is not None:
        result["start_year"] = tw.start_year
    if tw.end_year is not None:
        result["end_year"] = tw.end_year
    return result


def build_identity_projection(
    shape: Mapping[str, object],
    slots: list[Slot],
    *,
    entities: list[Entity] | None = None,
    open_roles: Collection[RoleAxis],
    requested_window: RequestedWindow | None = None,
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> dict:
    """Build the canonical spec1 identity document without minting a hash.

    ``shape`` is an identity-core wire member. ``open_roles`` are the axes asked for member
    by member, written sorted and omitted when empty. ``explicit_facets`` adds posted
    measurementMethod/unit bindings beside source-slot provenance. Each slot an invariant of
    the dataset the identity names fixes, left unbound, binds the invariant's value
    (`code_index.members.canonical_identity`). The
    caller validates the completed projection when it becomes a document.
    """
    projection: dict = {
        # "spec1" is the only identity_version this adapter builds.
        # The identity API exposes no version constant. This adapter pins spec1 here.
        "identity_version": "spec1",
        "shape": dict(shape),
        "slots": _slots_member(slots),
        "entities": _entities_member(entities),
        "requested_window": _requested_window_member(requested_window),
    }
    if open_roles:
        projection["open_roles"] = sorted(set(open_roles))
    facets = _facets_member(slots, explicit_facets)
    if facets:
        projection["facets"] = facets
    return dict(canonical_identity(get_index(), projection))


def compute_spec_id(
    shape: Mapping[str, object],
    slots: list[Slot],
    *,
    entities: list[Entity] | None = None,
    open_roles: Collection[RoleAxis],
    requested_window: RequestedWindow | None = None,
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> str:
    """Build and hash the spec1 identity document.

    Slot order and duplicate entity dcids do not affect the hash. Identity-schema
    errors propagate as ``identity.IdentityError``. Adapter invariant failures raise
    ``ValueError``. Returns a 37-character ``spec_`` id.
    """
    projection = build_identity_projection(
        shape,
        slots,
        entities=entities,
        open_roles=open_roles,
        requested_window=requested_window,
        explicit_facets=explicit_facets,
    )
    return identity.spec_id_from_value(projection)


def mint_identity(projection: Mapping[str, object]) -> tuple[str, BaseIdentity]:
    """Normalise, validate, and hash a projection as one identity document.

    Hashes the validated ``BaseIdentity`` after the identity package's NFC,
    sorting, deduplication, and singleton-set collapse, so its hash matches the
    returned definition.
    """
    definition = BaseIdentity.model_validate(identity.build_projection(projection))
    return identity_spec_id(definition), definition


def identity_spec_id(definition: BaseIdentity) -> str:
    """The spec id of `definition`, hashed whole."""
    return identity.spec_id_from_value(definition.model_dump(exclude_none=True))


def build_spec_identity(
    shape: Mapping[str, object],
    slots: list[Slot],
    *,
    entities: list[Entity] | None = None,
    open_roles: Collection[RoleAxis],
    requested_window: RequestedWindow | None = None,
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> tuple[str, BaseIdentity]:
    """Mint the id and return the ``BaseIdentity`` document it hashes.

    Both derive from one normalised projection through ``mint_identity``. The served
    document is in hash order, so clients can reproduce the id with JCS and SHA-256.
    Use ``compute_spec_id`` when only the id is needed.
    """
    projection = build_identity_projection(
        shape,
        slots,
        entities=entities,
        open_roles=open_roles,
        requested_window=requested_window,
        explicit_facets=explicit_facets,
    )
    return mint_identity(projection)


def slots_in_identity_order(slots: list[Slot], identity_doc: BaseIdentity) -> list[Slot]:
    """``slots`` with each set binding's values in the order ``identity_doc`` holds its
    refs. The identity core sorts a set's refs, and a resubmit rebuilds the set from
    them, so a Spec built from the request states its set in the order its resubmit
    will. A value the identity does not hold, such as a literal, keeps its place after
    those it does."""
    set_refs = {
        slot.property: slot.binding.refs
        for slot in identity_doc.slots
        if isinstance(slot.binding, IdentityBindingSet)
    }
    ordered: list[Slot] = []
    for slot in slots:
        binding = slot.binding
        refs = set_refs.get(slot.key.property.ref) if slot.key.property is not None else None
        if refs is None or not isinstance(binding, BindingSet):
            ordered.append(slot)
            continue
        rank = {dcid: i for i, dcid in enumerate(refs)}
        values = sorted(
            binding.values,
            key=lambda v: rank.get(v.ref.ref, len(rank)) if v.ref is not None else len(rank),
        )
        ordered.append(
            slot.model_copy(update={"binding": binding.model_copy(update={"values": values})})
        )
    return ordered
