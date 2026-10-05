"""Adapters for registering identities and materialising resolved StatVar shapes.

`build_identity_document` and `to_materialisation_request` handle one shape.
Described identities call them per member. `resolve_definition_members`
resolves pinned or described identities from the graph or code index.
`check_reproducible` validates roles with a resolved shape.
`rehydrate_definition` combines resolution and validation for stored definitions.
`verify_sidecar_definition` checks a supplied identity against its id.
`pinned_member_ref` extracts pinned refs for batch shape resolution.

`prime_head_signal` reads the head, probes facets, then derives a signal.
Its returned content version anchors the later head compare-and-set.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from dfo_spec_kit.identity import IdentityError, validate_identity_core

from code_index.members import (
    FamilyQuery,
    SourceNotInFamily,
    antichain_minimal_rows,
    canonical_identity,
    identity_family_query,
    index_covers_slot_keys,
    index_family,
)
from code_index.query import CodeIndex
from datasets_api.admission import bounded_materialise_slot
from datasets_api.codes_materialise import (
    CurrencyRoleUndeclared,
    DefinitionNotReproducible,
    InvalidRoleRef,
    MemberNotResolvable,
    RoleIncompatible,
    RoleSignatureIncompatible,
    SpecNotExportable,
)
from datasets_api.currency import derive_signals
from datasets_api.mint_models import (
    FacetRefBinding,
    FacetUnitRequestBinding,
    Scope,
    UnitRequestBindingBase,
    WindowAll,
    WindowLatest,
    WindowRange,
)
from datasets_api.models import MAX_COLLECTION_MEMBERS, CanonicalDefinition
from datasets_api.shape import StatVarShape, resolve_shape
from datasets_api.timings import phase
from datasets_api.versions import HeadReader, MemberSignal
from dc_sdmx import EntityFilter, SdmxClient
from materialise import CollectionFacetProbe, MaterialisationRequest, probe_collection_facets
from materialise.errors import (
    FacetNotAvailableError,
    OverCeilingError,
)
from materialise.facets import ElectedDataset, FacetBinding, UnitRequest
from materialise.graph import GraphReader
from materialise.recipes import (
    RecipeNotDeclaredError,
    RoleAxisConflictError,
    SpecRole,
    graph_role_for_spec_role,
)
from materialise.revision import RevisionReader
from materialise.roles import axis_ambiguous_for, axis_for_role, declared_role_for
from materialise.trace import MemberFamily
from materialise.window import RequestedWindow, election_window
from series_coverage.index import CoverageIndex
from spec_registry import SpecIdMismatch, recompute_identity
from spec_registry.reader import StoredDefinition
from statvar_members import (
    AbsentReading,
    DegenerateObservationProperties,
    Member,
    MemberEnumerationTooLarge,
    MemberEnumerator,
    MemberQueryUnanchored,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    declares_only_placing_roles,
    five_tuple_from_identity,
    member_digest,
    root_can_prove_complete,
)

# Identity documents store bare dcids. Shapes carry the `dcid:` prefix.
_STAT_VAR_PREFIX = "dcid:"

logger = logging.getLogger(__name__)

# Maximum candidate ids named in a `MemberNotResolvable` refusal.
_MAX_NAMED_INCONSISTENT_CANDIDATES = 10

# The subject role is valid only when it is the StatVar's sole declared role.
_OBSERVATION_ABOUT_ROLE = "observationAbout"

# The identity facet axis that names a dataset, and its wire pointer.
_PROVENANCE_AXIS = "provenance"
_PROVENANCE_POINTER = f"/facets/{_PROVENANCE_AXIS}"

# Probe role keys with `typeOf` to distinguish graph properties from other nodes.
_ROLE_EXISTENCE_PROBE = "typeOf"

# The count exceeds `statvar_set`'s cap. Narrowing the five-tuple requires
# another slot binding. Scope and window leave that count unchanged.
_MEMBER_CARDINALITY_OVER_CAP_REMEDY = (
    "Add another slot binding to the identity's own `slots` list to "
    "narrow the member set. A member_cardinality ceiling is fixed by the "
    "identity's own five-tuple and slots, so a narrower scope.entities or "
    "scope.window does not clear it; a narrower identity hashes to a "
    "different spec_id, so this registers a new definition rather than "
    "editing this one. A caller holding only spec_id gets that identity "
    "document from GET /api/datasets/{spec_id}, in the response's "
    "definition.identity field."
)

# A truncated reverse query yields a candidate window. The member total is unmeasured.
# `spec_id.resolved_stat_var_dcids` takes bare dcids, the form of a Search ref, and refuses a
# `dcid:` prefix. `statvar_set.statvars` requires the prefix, mints one pinned spec per
# StatVar, and accepts at most `MAX_COLLECTION_MEMBERS`. Wider sets need another slot.
_CANDIDATE_WINDOW_SEED_REMEDY = (
    "Post this same spec_id again with resolved_stat_var_dcids set to the StatVar dcid(s) "
    "search already resolved, written bare as resolved.stat_vars.items[].ref gives them (no "
    "dcid: prefix): a posted root, a member constrained by exactly the identity's bound "
    "slots that declares only the roles its entities are bound under, proves the family "
    "complete and this spec mints. Alternatively "
)
_CANDIDATE_WINDOW_STATVAR_SET_REMEDY = (
    'POST /api/datasets with kind "statvar_set" and statvars set to the StatVar dcid(s) '
    "search already resolved, each resolved.stat_vars.items[].ref written dcid:<ref>; that "
    "mints a different, pinned spec per StatVar, and statvars caps at "
    f"{MAX_COLLECTION_MEMBERS} per request. For a wider family, add another slot binding "
    "to the identity's own `slots` list instead."
)


def _candidate_window_remedy(*, root_can_prove: bool, accepts_seed: bool) -> str:
    """The remedy for a truncated, unproven page. The seed remedy keeps the caller's spec and
    leads when a posted root can prove the family complete (`root_can_prove_complete` and
    `_posted_root_can_elect`) and the refused subject is a spec, which `kind: "spec_id"`
    mints with a seed. Otherwise no
    posted StatVar clears the refusal, and `statvar_set` is the only direct mint: a collection
    member takes no seed.
    """
    if root_can_prove and accepts_seed:
        return f"{_CANDIDATE_WINDOW_SEED_REMEDY}{_CANDIDATE_WINDOW_STATVAR_SET_REMEDY}"
    return f"Mint the StatVars directly instead: {_CANDIDATE_WINDOW_STATVAR_SET_REMEDY}"


def _posted_root_can_elect(members: Sequence[Member], role_keys: frozenset[str]) -> bool:
    """Whether a posted root of the family `members` were read from would elect its role
    class (`statvar_members.declares_only_placing_roles`), so posting it can prove the
    family complete. With no member read, a root can elect only under a bound role key:
    with none, the root must be subject-only, which a role-bearing family has none of."""
    if members:
        return all(
            declares_only_placing_roles(member.declared_roles, role_keys) for member in members
        )
    return bool(role_keys)


def _absent_readings(index: CodeIndex) -> Mapping[str, AbsentReading]:
    """Map code systems to absent readings for member enumeration.

    The values distinguish aggregate members from populations omitting the
    property. Use the caller's index so enumeration and rendering share one snapshot.
    """
    return {
        code_system: declaration.absent_reading
        for code_system, declaration in index.declarations.items()
    }


def build_identity_document(
    shape: StatVarShape, *, scope: Scope | None, graph: GraphReader | None = None
) -> dict[str, object]:
    """Build one pinned identity-core document with an empty `slots` list and the request's scope.

    Shape constraints and declared roles guide materialisation outside the hash.
    `graph` classifies undeclared scope roles. When absent, those roles map to `InvalidRoleRef`.
    """
    entities = _identity_entities(shape, scope, graph=graph)
    document: dict[str, object] = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": shape.stat_var.removeprefix(_STAT_VAR_PREFIX)},
        "slots": [],
        "entities": entities,
        "requested_window": _identity_window(scope),
    }
    facets = _identity_facets(scope)
    if facets:
        document["facets"] = facets
    return document


def _identity_entities(
    shape: StatVarShape, scope: Scope | None, *, graph: GraphReader | None
) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    if scope is not None and scope.entities is not None:
        pinned_values = dict(shape.slots)
        # Track resolved roles separately because `entities` keeps caller spellings.
        resolved_keys: dict[str, str] = {}
        for role, refs in scope.entities.items():
            declared_role = declared_role_for(role, shape.declared_roles)
            if declared_role is None:
                if role == _OBSERVATION_ABOUT_ROLE:
                    raise RoleIncompatible(
                        stat_var=shape.stat_var, role=role, declared_roles=shape.declared_roles
                    )
                raise _undeclared_role_refusal(
                    shape.stat_var, role, declared_roles=shape.declared_roles, graph=graph
                )
            pinned_value = pinned_values.get(declared_role)
            if pinned_value is not None:
                raise RoleIncompatible(
                    stat_var=shape.stat_var,
                    role=role,
                    pinned_value=pinned_value,
                    declared_roles=shape.declared_roles,
                )
            collides_with = resolved_keys.get(declared_role)
            if collides_with is not None:
                raise _colliding_roles_refusal(
                    shape.stat_var,
                    role,
                    collides_with=collides_with,
                    declared_role=declared_role,
                    declared_roles=shape.declared_roles,
                )
            # Preserve caller spelling in the hashed identity. Translate at materialisation.
            resolved_keys[declared_role] = role
            entities[role] = list(refs)
    return entities


def _colliding_roles_refusal(
    stat_var: str,
    role: str,
    *,
    collides_with: str,
    declared_role: str,
    declared_roles: tuple[str, ...],
) -> RoleIncompatible:
    """Refuse keys that map to the same declared role and would collapse into one wire filter.

    `role` is the later key, `collides_with` the earlier one, and
    `declared_role` their shared role. The wire pointer uses `role`.
    """
    exc = RoleIncompatible(
        stat_var=stat_var,
        role=role,
        declared_roles=declared_roles,
        message=(
            f"stat_var {stat_var!r} scope.entities keys {collides_with!r} and {role!r} "
            f"name the same declared role {declared_role!r}"
        ),
    )
    return exc


def _ambiguous_axis_message(stat_var: str, role: str, declared_roles: tuple[str, ...]) -> str:
    """Name both declared roles when a scope key aliases their shared flow axis."""
    axis_roles = [declared for declared in declared_roles if axis_for_role(declared) == role]
    return (
        f"stat_var {stat_var!r} declares two roles, {axis_roles[0]!r} and {axis_roles[1]!r}, "
        f"on the {role!r} axis (its own observationProperties arc); the {role!r} binding is "
        "ambiguous and no single role can serve it"
    )


def _undeclared_role_refusal(
    stat_var: str, role: str, *, declared_roles: tuple[str, ...], graph: GraphReader | None
) -> InvalidRoleRef | RoleIncompatible:
    """Classify a scope key absent from the shape's declared roles.

    `observationAbout` is handled before this call.

    Return `RoleIncompatible` for a graph property and `InvalidRoleRef` for
    another node type or a missing graph. The returned error carries
    `declared_roles`. Ambiguous aliases preserve the graph-based error type
    and name both roles on the axis.
    """
    if graph is not None:
        arcs = graph.node_arc_values(role, properties=(_ROLE_EXISTENCE_PROBE,))
        if "Property" in arcs.get(_ROLE_EXISTENCE_PROBE, ()):
            return RoleIncompatible(
                stat_var=stat_var,
                role=role,
                declared_roles=declared_roles,
                message=(
                    _ambiguous_axis_message(stat_var, role, declared_roles)
                    if axis_ambiguous_for(role, declared_roles)
                    else None
                ),
            )
    if axis_ambiguous_for(role, declared_roles):
        return InvalidRoleRef(
            stat_var=stat_var,
            role=role,
            declared_roles=declared_roles,
            message=_ambiguous_axis_message(stat_var, role, declared_roles),
        )
    return InvalidRoleRef(stat_var=stat_var, role=role, declared_roles=declared_roles)


def _identity_window(scope: Scope | None) -> dict[str, object]:
    if scope is None or scope.window is None:
        return {"kind": "all"}
    return window_to_identity(scope.window)


def window_to_identity(window: WindowAll | WindowLatest | WindowRange) -> dict[str, object]:
    """Convert a wire `scope.window` to the identity-core `requested_window` form."""
    if window.kind == "range":
        result: dict[str, object] = {"kind": "range"}
        if window.start_year is not None:
            result["start_year"] = window.start_year
        if window.end_year is not None:
            result["end_year"] = window.end_year
        return result
    return {"kind": window.kind}


def _identity_facets(scope: Scope | None) -> dict[str, object]:
    if scope is None or scope.facets is None:
        return {}
    return facets_to_identity(scope.facets)


def facets_to_identity(
    facets: Mapping[str, FacetRefBinding | FacetUnitRequestBinding],
) -> dict[str, object]:
    """Convert facet bindings to identity form: unit requests use `request`, others use `ref`."""
    return {
        axis: _unit_request_to_identity(binding)
        if isinstance(binding, UnitRequestBindingBase)
        else {"state": "value", "ref": binding.ref}
        for axis, binding in facets.items()
    }


def _unit_request_to_identity(binding: UnitRequestBindingBase) -> dict[str, object]:
    """Build the identity request.

    Preserve ISO or role currency and omit an unset base year.
    """
    currency: str | dict[str, object] = (
        binding.currency if isinstance(binding.currency, str) else {"role": binding.currency.role}
    )
    request: dict[str, object] = {"currency": currency, "prices": binding.prices}
    base_year = getattr(binding, "base_year", None)
    if base_year is not None:
        request["base_year"] = base_year
    return {"state": "value", "request": request}


def to_materialisation_request(
    identity: Mapping[str, object], *, shape: StatVarShape, code_index: CodeIndex
) -> MaterialisationRequest:
    """Convert an identity document and resolved shape to a materialisation request.

    StatVar, declared roles, and the elected dataset come from `shape`. Entity, facet,
    and window bindings come from the identity. Entity keys retain their hashed spelling
    in storage and map to declared graph roles for SDMX filters. Reproducibility
    checks reject unsupported roles before this call. `index_sourced` is
    derived from the same `CodeIndex`. Pinned shapes use the graph path.

    A facet uses one form, `ref` or a unit-axis `request`. Malformed bindings
    raise `IdentityError`. An undeclared role currency raises
    `CurrencyRoleUndeclared`. The caller adds the wire pointer.
    Validation covers facet-binding shape only.
    """
    entities = cast("Mapping[str, list[str]]", identity.get("entities", {}))
    role_bindings: dict[str, EntityFilter] = {}
    for role, refs in entities.items():
        declared_role = declared_role_for(role, shape.declared_roles)
        assert declared_role is not None, (
            f"identity {identity!r} binds entity role {role!r} that shape "
            f"declared_roles {shape.declared_roles!r} no longer declares; "
            "check_reproducible runs before materialisation and refuses this"
        )
        role_bindings[declared_role] = list(refs)
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets", {}))
    facet_bindings: dict[str, FacetBinding] = {}
    for axis, binding in facets.items():
        has_ref = "ref" in binding
        has_request = "request" in binding
        if has_ref and has_request:
            raise IdentityError(
                "invalid_binding_shape",
                f"{axis} facet binding cannot carry both 'ref' and 'request'",
            )
        if has_request:
            facet_bindings[axis] = FacetBinding(
                state="request",
                request=_unit_request_from_identity(
                    cast("Mapping[str, object]", binding["request"]), shape=shape
                ),
            )
        elif has_ref:
            # A null ref binds the facets that declare no value on the axis; the identity
            # core admits it on measurementMethod and unit only.
            facet_bindings[axis] = FacetBinding(
                state="bound", ref=cast("str | None", binding["ref"])
            )
        else:
            raise IdentityError(
                "schema_violation", f"{axis} facet binding requires 'ref' or 'request'"
            )
    window = _requested_window(
        cast("Mapping[str, object]", identity.get("requested_window", {"kind": "all"}))
    )
    return MaterialisationRequest(
        stat_var=shape.stat_var,
        role_bindings=role_bindings,
        declared_roles=shape.declared_roles,
        facet_bindings=facet_bindings,
        window=window,
        index_sourced=_identity_is_index_sourced(identity, code_index=code_index),
        dataset=shape.dataset,
        family=_request_family(identity, shape),
    )


def _request_family(identity: Mapping[str, object], shape: StatVarShape) -> MemberFamily | None:
    """Return the population `identity` admits, for the trace. A pinned identity admits its
    one StatVar. A described identity admits the family member resolution stated on `shape`."""
    if cast("Mapping[str, object]", identity.get("shape", {})).get("kind") == "stat_var":
        return _member_family(frozenset((shape.stat_var.removeprefix(_STAT_VAR_PREFIX),)))
    return shape.family


def _identity_is_index_sourced(identity: Mapping[str, object], *, code_index: CodeIndex) -> bool:
    """Return whether `resolve_definition_members` resolves a described arm from `code_index`.

    The test is `index_covers_slot_keys` on the identity's slot keys. Pinned arms resolve
    against the graph.
    """
    shape_member = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape_member.get("kind") == "stat_var":
        return False
    slots = cast("Sequence[Mapping[str, object]]", identity.get("slots", ()))
    return index_covers_slot_keys(
        code_index,
        five_tuple_from_identity(shape_member),
        {cast(str, slot["property"]) for slot in slots},
    )


def _unit_request_from_identity(
    request: Mapping[str, object], *, shape: StatVarShape
) -> UnitRequest:
    """Map a unit-facet request to `UnitRequest`.

    Role currencies must match a declared role. Undeclared or ambiguous
    roles raise `CurrencyRoleUndeclared`. Ambiguity names both roles. The
    caller adds the wire pointer and member id.
    """
    currency_value = request["currency"]
    if isinstance(currency_value, Mapping):
        role = cast(str, cast("Mapping[str, object]", currency_value)["role"])
        try:
            graph_role_for_spec_role(
                spec_role=cast("SpecRole", role), declared_roles=shape.declared_roles
            )
        except RoleAxisConflictError as exc:
            # Name both roles because the alias maps to their shared axis.
            axis_roles = [
                declared for declared in shape.declared_roles if axis_for_role(declared) == role
            ]
            raise CurrencyRoleUndeclared(
                stat_var=shape.stat_var,
                role=role,
                declared_roles=shape.declared_roles,
                message=(
                    f"stat_var {shape.stat_var!r} declares two roles, {axis_roles[0]!r} and "
                    f"{axis_roles[1]!r}, on the {role!r} axis named by a unit request's "
                    "role-shaped currency; no single role can serve it"
                ),
            ) from exc
        except RecipeNotDeclaredError as exc:
            raise CurrencyRoleUndeclared(
                stat_var=shape.stat_var, role=role, declared_roles=shape.declared_roles
            ) from exc
        currency = role
    else:
        currency = cast(str, currency_value)
    return UnitRequest(
        currency=currency,
        prices=cast('Literal["current", "constant"]', request["prices"]),
        base_year=cast("int | None", request.get("base_year")),
    )


def _requested_window(window: Mapping[str, object]) -> RequestedWindow:
    """The window a request reads for an identity's `requested_window`: a range with no end
    year ends at the current year (`materialise.window.election_window`), the years a
    Search answer for the same identity elects and reads under. The identity keeps its open
    end."""
    kind = cast('Literal["all", "latest", "range"]', window.get("kind", "all"))
    return election_window(
        RequestedWindow(
            kind=kind,
            start_year=cast("int | None", window.get("start_year")),
            end_year=cast("int | None", window.get("end_year")),
        )
    )


def pinned_member_ref(identity: Mapping[str, object]) -> str | None:
    """Return a pinned identity's `dcid:` ref, or `None` for a described arm.

    Graph access is unnecessary, so callers can batch pinned refs.
    """
    shape_member = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape_member.get("kind") != "stat_var":
        return None
    return f"{_STAT_VAR_PREFIX}{shape_member['id']}"


@dataclass(frozen=True)
class ResolvedMembers:
    """The members an identity resolves to and the family they were reduced from."""

    shapes: tuple[StatVarShape, ...]
    """The antichain-minimal members, one shape each. The mint serves these."""
    family: frozenset[str]
    """Bare dcids of every member before antichain reduction. It holds each `shapes` dcid
    and every member a broader member dominates."""


def _member_family(dcids: frozenset[str]) -> MemberFamily:
    """State a whole admitted population by the digest Search states it by
    (`statvar_members.member_digest`) and its count."""
    return MemberFamily(digest=member_digest(dcids), count=len(dcids))


def _resolve_index_members(
    query: FamilyQuery, *, spec_id: str, code_index: CodeIndex
) -> ResolvedMembers | None:
    """Resolve a described identity's `query` from the in-memory code index, or return
    `None` when the index does not decide its membership (`code_index.members.index_family`).

    The family is one role class of one dataset's rows, elected by
    `code_index.members.elect_dataset`: the dataset the query names, else the one elected
    among the five-tuple's publishers, and its rows whose roles cover the query's role keys
    and declare the fewest roles. Return its antichain-minimal rows as shapes. Each shape
    carries the elected dataset, every dataset the index records its row under, and the
    dcids of every family row. Raise `SpecNotExportable` when none match,
    `FacetNotAvailableError` when the query names a dataset that does not publish its
    five-tuple, `RoleSignatureIncompatible` when the election ties between role sets, or
    `MemberNotResolvable` for `observationAbout`. The in-memory path has no page
    truncation or member-count ceiling.
    """
    five_tuple, slots = query.five_tuple, query.slots
    try:
        elected = index_family(code_index, query)
    except SourceNotInFamily as exc:
        raise FacetNotAvailableError(
            f"spec {spec_id!r} binds provenance {exc.source!r}, which does not publish "
            f"this five-tuple; it is published by {list(exc.candidates)!r}",
            pointer=_PROVENANCE_POINTER,
        ) from exc
    if elected is None:
        return None
    family = elected.rows
    rows = antichain_minimal_rows(code_index, family)
    for row in rows:
        if row.observation_properties and _OBSERVATION_ABOUT_ROLE in row.observation_properties:
            raise MemberNotResolvable(members=(spec_id,))
    if not rows:
        raise SpecNotExportable(
            spec_id=spec_id,
            detail=(
                f"spec {spec_id!r} enumerates no member series against the index for "
                f"{five_tuple!r} consistent with slots {slots!r}"
            ),
        )
    roles_by_dcid = {
        row.dcid: row.observation_properties or (_OBSERVATION_ABOUT_ROLE,) for row in rows
    }
    if len({frozenset(roles) for roles in roles_by_dcid.values()}) > 1:
        raise RoleSignatureIncompatible(spec_id=spec_id, roles_by_stat_var=roles_by_dcid)
    family_dcids = frozenset(row.dcid for row in family)
    # The index holds every row of the family, so the whole of it is stated.
    stated = _member_family(family_dcids)
    shapes = tuple(
        StatVarShape(
            stat_var=f"{_STAT_VAR_PREFIX}{row.dcid}",
            declared_roles=roles_by_dcid[row.dcid],
            slots=tuple(sorted(row.constraint_properties.items())),
            measured_property=five_tuple.measured_property,
            population_type=five_tuple.population_type,
            # A family with a row has an elected provenance.
            dataset=ElectedDataset(
                provenance=cast(str, elected.provenance), member_provenances=row.provenances
            ),
            family=stated,
        )
        for row in rows
    )
    return ResolvedMembers(shapes=shapes, family=family_dcids)


def member_resolution_key(identity: Mapping[str, object], *, code_index: CodeIndex) -> Hashable:
    """Return the inputs `resolve_definition_members` reads from `identity`, as one key.

    A pinned arm's key is its StatVar. A described arm's key is the `FamilyQuery` of its
    canonical form (`code_index.members.canonical_identity`): the five-tuple, the widened
    slot bindings, the role keys its entities are bound under and the dataset the identity
    names. Two identities with equal keys resolve the same members, so a resolved shape
    serves another identity only under an equal key, and identities that bind different pins
    of one family share a key. The entities bound under each role key,
    the window and every facet that names no dataset stay out of the key."""
    shape_member = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape_member.get("kind") == "stat_var":
        return shape_member["id"]
    query = identity_family_query(code_index, canonical_identity(code_index, identity))
    return (
        query.five_tuple,
        tuple(sorted(query.slots.items())),
        query.source,
        tuple(sorted(query.role_keys)),
    )


def resolve_definition_members(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
    seed: Sequence[str] | None = None,
    accepts_seed: bool = True,
    role_keys: Collection[str] | None = None,
) -> ResolvedMembers:
    """Resolve the members denoted by an identity's shape against the graph or code index.

    `ResolvedMembers.shapes` holds the antichain-minimal members and
    `ResolvedMembers.family` every member before that reduction: the pinned dcid,
    every matching index row of the elected dataset, or every confirmed walked or seeded
    candidate.

    `code_index` is the caller's resolved index, shared by enumeration and
    frame rendering so both use the same declarations.

    `seed` adds bare caller-supplied dcids to described-arm candidates.
    Empty seeds leave candidates unchanged. Pinned and index-backed paths
    ignore seeds. Every seed uses the same confirmation and antichain
    reduction as walked candidates. Only matching candidates survive. A root
    seed can prove a page complete by dominating its candidates. Other seeds
    confirm membership only.

    `role_keys` replaces the role keys of the identity's own `entities` in the
    role-class election, for a caller whose scope binds other entities. `None`
    elects under the identity's own.

    `accepts_seed` says whether the caller's door can take a seed for this
    identity. A spec can: `kind: "spec_id"` mints it with
    `resolved_stat_var_dcids`. A collection member cannot, so its
    `candidate_window` refusal leaves the seed remedy out.

    A pinned arm resolves to one `StatVarShape`. A described arm widens
    hierarchical `value` and `set` slots to the self-inclusive subtree via
    `CodeIndex.descendants`. Leaves, flat systems, and unknown codes pass
    through unchanged. The graph walk builds all shapes in one read.

    If the identity's slot keys hold every constraint property of at least one
    index row of the five-tuple (`index_covers_slot_keys`), resolve the elected
    dataset's minimal rows in memory; `facets.provenance` names the dataset when it
    binds one distinct ref (`code_index.members.named_dataset`). An identity whose keys cover a row and whose values match none
    refuses `SpecNotExportable` on this path. This bounded index path shares
    empty-result and role-signature refusals with the graph walk. The graph walk
    enforces the enumerator's member cap and refuses incomplete candidate pages.

    Raises `IdentityError` before inspecting the shape if the identity is invalid.
    The pinned arm maps `resolve_shape` refusals to `MemberNotResolvable` for `spec_id`.
    The described arm raises `SpecNotExportable` for an unanchored query or empty result.
    Enumeration over its member cap raises `OverCeilingError` with
    the exact count. A truncated, unproven page raises `candidate_window`
    with the count read and page cap. Membership beyond that window is
    unmeasured. A proven-complete antichain is accepted. Same-keyset
    duplicates beyond the window may be absent from the answer.

    Invalid candidate arcs raise `MemberNotResolvable` for `spec_id`.
    Inconsistent forward and reverse arcs raise it with up to 10 candidate
    ids and `truncated=True` when more exist, unless the group holds a
    confirmed root (`MemberGroup.root_confirmed`) that dominates every one of
    them (`MemberGroup.undismissed_candidates` is empty). A candidate that may
    duplicate the root still refuses, since its roles are unverified.
    Both paths elect one role class (`statvar_members.elect_role_class`): the
    members whose declared roles cover the role keys `entities` is bound under,
    least-constrained first, then those declaring the fewest roles, so a member
    lacking a role its sibling carries answers as the aggregate over that role.
    `RoleSignatureIncompatible` reports a tie that election cannot break, between
    members whose role sets do not contain one another, and takes precedence under
    a confirmed root, where an unverified candidate cannot remove a member.

    An unresolved denominator (`{"unresolved": true}`) matches a target with
    no dcid or one typed exactly `ProvisionalNode`, as Search reads it.

    An identity that binds a slot to an invariant of the dataset it names and one that
    leaves it unbound resolve the same members, since their canonical forms agree
    (`code_index.members.canonical_identity`).
    """
    validate_identity_core(identity)
    shape_member = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape_member.get("kind") == "stat_var":
        ref = f"{_STAT_VAR_PREFIX}{shape_member['id']}"
        try:
            shape = resolve_shape(ref, graph=graph)
        except MemberNotResolvable as exc:
            raise MemberNotResolvable(members=(spec_id,)) from exc
        return ResolvedMembers(
            shapes=(shape,), family=frozenset((shape.stat_var.removeprefix(_STAT_VAR_PREFIX),))
        )

    query = identity_family_query(code_index, identity)
    if role_keys is not None:
        query = dataclasses.replace(query, role_keys=frozenset(role_keys))
    five_tuple, slots = query.five_tuple, query.slots
    resolved = _resolve_index_members(query, spec_id=spec_id, code_index=code_index)
    if resolved is not None:
        logger.debug(
            "datasets.member_resolution path=index spec_id=%s member_count=%d "
            "index_content_hash=%s",
            spec_id,
            len(resolved.shapes),
            code_index.stamp.content_hash,
        )
        return resolved
    try:
        group = enumerator.enumerate(
            five_tuple=five_tuple,
            slots=slots,
            readings=_absent_readings(code_index),
            extra_candidates=seed or (),
            role_keys=query.role_keys,
        )
    except MemberQueryUnanchored as exc:
        raise SpecNotExportable(
            spec_id=spec_id,
            detail=f"spec {spec_id!r} names no measured_property to anchor a member query",
        ) from exc
    except MemberEnumerationTooLarge as exc:
        raise OverCeilingError(
            f"spec {spec_id!r} enumerates {exc.count} member(s) consistent with its own "
            f"five-tuple and slots, exceeding member_cap={exc.member_cap}. "
            f"{_MEMBER_CARDINALITY_OVER_CAP_REMEDY}",
            ceiling_kind="member_cardinality",
            count=exc.count,
            cap=exc.member_cap,
        ) from exc
    except MemberRoleDisagreement as exc:
        raise RoleSignatureIncompatible(
            spec_id=spec_id, roles_by_stat_var=exc.roles_by_stat_var
        ) from exc
    except (
        DegenerateObservationProperties,
        MultiValuedConstraint,
        NotAStatisticalVariable,
    ) as exc:
        raise MemberNotResolvable(members=(spec_id,)) from exc
    if group.truncated and not group.proven_complete:
        raise OverCeilingError(
            f"spec {spec_id!r} enumerates against {five_tuple!r} past the "
            f"{group.candidate_cap}-candidate reverse-arc window this deployment reads: "
            f"this read returned {group.candidates_considered} candidates of the "
            f"{group.candidate_cap} that window allows, and the member count past it is "
            "unknown, not zero. "
            + _candidate_window_remedy(
                root_can_prove=root_can_prove_complete(
                    slots=slots, readings=_absent_readings(code_index)
                )
                and _posted_root_can_elect(group.members, query.role_keys),
                accepts_seed=accepts_seed,
            ),
            ceiling_kind="candidate_window",
            count=group.candidates_considered,
            cap=group.candidate_cap,
        )
    if group.undismissed_candidates:
        candidates = group.undismissed_candidates
        named = candidates[:_MAX_NAMED_INCONSISTENT_CANDIDATES]
        raise MemberNotResolvable(
            members=tuple(f"{_STAT_VAR_PREFIX}{candidate}" for candidate in named),
            truncated=len(candidates) > _MAX_NAMED_INCONSISTENT_CANDIDATES,
        )
    if not group.members:
        raise SpecNotExportable(
            spec_id=spec_id,
            detail=(
                f"spec {spec_id!r} enumerates no member series against the graph for "
                f"{five_tuple!r} consistent with slots {slots!r}"
            ),
        )

    walked = frozenset(group.family)
    # A truncated page proven complete by a root leaves candidates past the window unread,
    # and a member that duplicates another's keys can sit among them, so the walk states
    # the family only when it read every candidate.
    stated = None if group.truncated else _member_family(walked)
    shapes = tuple(
        StatVarShape(
            stat_var=f"{_STAT_VAR_PREFIX}{member.stat_var}",
            declared_roles=member.declared_roles,
            slots=tuple(sorted(member.constraints.items())),
            measured_property=five_tuple.measured_property,
            population_type=five_tuple.population_type,
            family=stated,
        )
        for member in group.members
    )
    logger.debug(
        "datasets.member_resolution path=walk spec_id=%s member_count=%d index_content_hash=None",
        spec_id,
        len(shapes),
    )
    return ResolvedMembers(shapes=shapes, family=walked)


def check_reproducible(
    identity: Mapping[str, object], *, spec_id: str, shape: StatVarShape
) -> None:
    """Check identity role keys against the resolved shape without graph access.

    Keys may use raw graph roles or flow-axis aliases. `declared_role_for`
    maps both. Described members share one declared-role set. A pinned
    identity's hash follows its dcid, so roles are its reproducibility check.
    Raises `DefinitionNotReproducible` with `spec_id` and unsupported roles.
    """
    bound_entities = cast("Mapping[str, object]", identity.get("entities", {}))
    undeclared = sorted(
        role for role in bound_entities if declared_role_for(role, shape.declared_roles) is None
    )
    if undeclared:
        raise DefinitionNotReproducible(
            spec_ids=(spec_id,),
            undeclared_roles=tuple(undeclared),
            detail=(
                f"spec {spec_id!r} binds entity role(s) {', '.join(undeclared)} that "
                f"{shape.stat_var!r} no longer declares"
            ),
        )


def rehydrate_definition(
    stored: StoredDefinition,
    *,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
    seed: Sequence[str] | None = None,
    accepts_seed: bool = True,
) -> tuple[tuple[StatVarShape, ...], tuple[MaterialisationRequest, ...]]:
    """Re-resolve a stored identity, check its roles, and build one request per shape.

    Pinned identities return one shape and request. Described identities
    return one pair per current member. `index_sourced` is shared across the
    requests and lets multi-member facet grouping accept shared provenance
    and unit. Reuse the caller's `code_index`.

    Pass the original `seed` when posted StatVar dcids proved completeness at
    mint. Callers recover it from the stored head or from the operation that
    carries it. A seedless re-read can exceed the candidate window and refuse
    the identity. `accepts_seed` is `False` for a collection member.
    """
    shapes = resolve_definition_members(
        stored.identity,
        spec_id=stored.spec_id,
        graph=graph,
        enumerator=enumerator,
        code_index=code_index,
        seed=seed,
        accepts_seed=accepts_seed,
    ).shapes
    check_reproducible(stored.identity, spec_id=stored.spec_id, shape=shapes[0])
    requests = tuple(
        to_materialisation_request(stored.identity, shape=shape, code_index=code_index)
        for shape in shapes
    )
    return shapes, requests


def verify_sidecar_definition(
    definition: CanonicalDefinition, *, spec_id: str
) -> dict[str, object]:
    """Confirm the identity hashes to `spec_id` and return its canonical document.

    The canonical document is the one a registration stores. Raise `SpecIdMismatch` on
    disagreement. Malformed identities raise `IdentityError`.
    """
    recomputed = recompute_identity(definition.identity)
    if recomputed.spec_id != spec_id:
        raise SpecIdMismatch(supplied=spec_id, computed=recomputed.spec_id)
    return recomputed.identity


@dataclass(frozen=True)
class HeadPriming:
    """Facet probe and signal plus the head version read before probing."""

    probe: CollectionFacetProbe
    signal: tuple[MemberSignal, ...] | None
    expected_content_version: str | None


def prime_head_signal(
    requests: Sequence[MaterialisationRequest],
    *,
    subject_id: str,
    graph: GraphReader,
    sdmx: SdmxClient,
    combine: bool,
    head_store: HeadReader,
    materialise_wait_seconds: float,
    revision_reader: RevisionReader | None = None,
    materialise_semaphore: threading.BoundedSemaphore | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
) -> HeadPriming:
    """Read the head, probe facet candidates, and derive a signal from that probe.

    The returned content version predates the probe and anchors the later
    head compare-and-set. `combine=True` returns one signal. `False` returns
    one per request in order. A request that binds no place is probed at its
    default place scope (`materialise.place_scope`), read through `sdmx`.
    `None` indicates a member facet failure or
    failure to find one facet for the whole set. It leaves the head unchanged,
    and materialisation runs and can raise its refusal.

    Pass the same coverage snapshot used for admission. Role-bearing requests
    require it. Missing coverage raises `MaterialisationUsageError`. An
    optional `revision_reader` stamps the probe before its first graph call.

    The head read and probe wait for `materialise_semaphore` up to the
    configured wait or `deadline`. The probe also receives the deadline, and
    the following rebuild takes a second slot under that same deadline.
    """
    with bounded_materialise_slot(
        materialise_semaphore, wait_seconds=materialise_wait_seconds, deadline=deadline
    ):
        with phase("head_read"):
            head = head_store.read_head(subject_id)
        with phase("facet_probe"):
            probe = probe_collection_facets(
                requests,
                graph=graph,
                availability=sdmx,
                settle=True,
                revision_reader=revision_reader,
                coverage=coverage,
                deadline=deadline,
            )
    with phase("signal"):
        signal = derive_signals(requests, probe, graph=graph, combine=combine)
    return HeadPriming(
        probe=probe,
        signal=signal,
        expected_content_version=head.content_version if head is not None else None,
    )
