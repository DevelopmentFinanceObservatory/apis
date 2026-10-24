"""Adapters for registering identities and materialising resolved StatVar shapes.

`build_identity_document` and `to_materialisation_request` handle one shape.
Described identities call them per member. `resolve_definition_members`
resolves pinned or described identities from the graph or code index.
`check_reproducible` validates roles with a resolved shape.
`rehydrate_definition` combines resolution and validation for stored definitions.
`verify_sidecar_definition` checks a supplied identity against its id.
`pinned_member_ref` extracts pinned refs for batch shape resolution.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, cast

from dfo_spec_kit.identity import IdentityError, validate_identity_core

from code_index.members import (
    FamilyQuery,
    SourceNotInFamily,
    antichain_minimal_rows,
    identity_family_query,
    index_decides,
    index_family,
    listed_group,
    pinned_refs,
)
from code_index.query import CodeIndex
from datasets_api.codes_materialise import (
    CurrencyRoleUndeclared,
    DefinitionNotReproducible,
    InvalidRoleRef,
    MemberNotResolvable,
    RoleIncompatible,
    RoleSignatureIncompatible,
    SpecNotExportable,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.mint_models import (
    FacetNationalCurrencyBinding,
    FacetRefBinding,
    FacetUnitRequestBinding,
    Scope,
    UnitRequestBindingBase,
    WindowAll,
    WindowLatest,
    WindowRange,
)
from datasets_api.models import CanonicalDefinition
from datasets_api.shape import StatVarShape, resolve_shape
from dc_sdmx import EntityFilter
from dfo_platform.named import cut
from materialise import MaterialisationRequest
from materialise.errors import (
    FacetNotAvailableError,
    OverCeilingError,
)
from materialise.facets import ElectedDataset, FacetBinding, NationalCurrency, UnitRequest
from materialise.graph import GraphReader
from materialise.recipes import (
    RecipeNotDeclaredError,
    RoleAxisConflictError,
    SpecRole,
    graph_role_for_spec_role,
)
from materialise.roles import axis_ambiguous_for, axis_for_role, declared_role_for
from materialise.trace import MemberFamily
from materialise.window import RequestedWindow, election_window
from origin_catalogue import scoped_reads
from spec_registry import SpecIdMismatch, recompute_identity
from spec_registry.reader import StoredDefinition
from statvar_members import (
    AbsentReading,
    DegenerateObservationProperties,
    MemberEnumerationTooLarge,
    MemberEnumerator,
    MemberQueryUnanchored,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    member_digest,
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

# The member count exceeds `member_cap`. Narrowing the five-tuple requires
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

# A walk refuses `candidate_window` when the whole filtered reverse set holds more subjects than
# the walk reads. The member total is unmeasured. `statvar_set.statvars` requires the `dcid:`
# prefix and names one StatVar, which it mints as a pinned spec.
_CANDIDATE_WINDOW_REMEDY = (
    "Add another slot binding to the identity's own `slots` list to narrow the set the walk "
    "reads; a narrower identity hashes to a different spec_id. Alternatively "
    'POST /api/datasets with kind "statvar_set" and statvars set to one StatVar dcid, '
    "written dcid:<ref>; that mints a different, pinned spec for that StatVar, one request "
    "per StatVar."
)


@dataclass(frozen=True)
class MemberReadBounds:
    """How far a described arm's member walk reads.

    A first reverse page that truncates without a confirmed root makes the walk read the
    whole filtered set, up to `subject_ceiling` subjects. Every graph read of the walk ends
    by `deadline`, a `time.monotonic()` instant."""

    subject_ceiling: int
    deadline: float


def member_read_bounds(
    settings: DatasetsBuildSettings, *, request_deadline: float | None
) -> MemberReadBounds:
    """The bounds of a walk starting now: `settings.member_subject_ceiling`, and
    `settings.member_read_seconds` from now or `request_deadline`, whichever is earlier."""
    deadline = time.monotonic() + settings.member_read_seconds
    if request_deadline is not None:
        deadline = min(deadline, request_deadline)
    return MemberReadBounds(subject_ceiling=settings.member_subject_ceiling, deadline=deadline)


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
    facets: Mapping[str, FacetRefBinding | FacetUnitRequestBinding | FacetNationalCurrencyBinding],
) -> dict[str, object]:
    """Convert facet bindings to identity form: unit requests use `request`, a
    national-currency binding `national_currency`, and others `ref`."""
    return {axis: _facet_to_identity(binding) for axis, binding in facets.items()}


def _facet_to_identity(
    binding: FacetRefBinding | FacetUnitRequestBinding | FacetNationalCurrencyBinding,
) -> dict[str, object]:
    if isinstance(binding, UnitRequestBindingBase):
        return _unit_request_to_identity(binding)
    if isinstance(binding, FacetNationalCurrencyBinding):
        return {"state": "value", "national_currency": {"role": binding.national_currency.role}}
    return {"state": "value", "ref": binding.ref}


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

    A facet uses one form, `ref` or a unit-axis `request`, and the provenance facet may
    also bind a set (`refs`). A set of one distinct ref binds it as `ref` does. A set of
    several leaves provenance unbound for selection: the read is scoped to its refs
    (`pinned_provenance_scope`), and selection elects among them. Malformed bindings
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
        if axis == _PROVENANCE_AXIS and "refs" in binding:
            refs = tuple(dict.fromkeys(cast("Sequence[str]", binding["refs"])))
            if len(refs) == 1:
                facet_bindings[axis] = FacetBinding(state="bound", ref=refs[0])
            continue
        has_ref = "ref" in binding
        has_request = "request" in binding
        has_national_currency = "national_currency" in binding
        if has_ref + has_request + has_national_currency > 1:
            raise IdentityError(
                "invalid_binding_shape",
                f"{axis} facet binding carries more than one of 'ref', 'request' and "
                "'national_currency'",
            )
        if has_national_currency:
            role = cast("Mapping[str, object]", binding["national_currency"])["role"]
            _require_currency_role(cast(str, role), shape=shape)
            facet_bindings[axis] = FacetBinding(
                state="national_currency",
                national_currency=NationalCurrency(role=cast("SpecRole", role)),
            )
        elif has_request:
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
                "schema_violation",
                f"{axis} facet binding requires 'ref', 'request' or 'national_currency'",
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
        family=admitted_family(identity, shape),
    )


def provenance_pin(identity: Mapping[str, object]) -> frozenset[str] | None:
    """The provenances `identity`'s `facets.provenance` binds, as a `ref` or a set of
    `refs` (`code_index.members.pinned_refs`), or `None` when it binds none."""
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets", {}))
    return frozenset(pinned_refs(facets)) or None


def pinned_provenance_scope(identity: Mapping[str, object]) -> AbstractContextManager[object]:
    """Bind the request's catalogue narrowed to `identity`'s pinned provenances
    (`provenance_pin`, `origin_catalogue.scoped_reads`) for every read of a mint of it, so
    a pinned source is admitted before DFO precedence and a base pin serves at a place
    where a DFO table also holds data. Binds nothing for an identity that pins no provenance."""
    return scoped_reads(provenance_pin(identity))


def admitted_family(identity: Mapping[str, object], shape: StatVarShape) -> MemberFamily | None:
    """Return the population `identity` admits, as the trace states it. A pinned identity
    admits its one StatVar. A described identity admits the family member resolution stated on `shape`."""
    if cast("Mapping[str, object]", identity.get("shape", {})).get("kind") == "stat_var":
        return _member_family(frozenset((shape.stat_var.removeprefix(_STAT_VAR_PREFIX),)))
    return shape.family


def _identity_is_index_sourced(identity: Mapping[str, object], *, code_index: CodeIndex) -> bool:
    """Return whether `resolve_definition_members` resolves a described arm from `code_index`.

    The test is `index_decides` on the identity's query, the one `index_family` applies.
    Pinned arms resolve against the graph.
    """
    shape_member = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape_member.get("kind") == "stat_var":
        return False
    return index_decides(code_index, identity_family_query(code_index, identity))


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
        currency = cast(str, cast("Mapping[str, object]", currency_value)["role"])
        _require_currency_role(currency, shape=shape)
    else:
        currency = cast(str, currency_value)
    return UnitRequest(
        currency=currency,
        prices=cast('Literal["current", "constant"]', request["prices"]),
        base_year=cast("int | None", request.get("base_year")),
    )


def _require_currency_role(role: str, *, shape: StatVarShape) -> None:
    """Raise `CurrencyRoleUndeclared` unless `role`, the role a unit binding takes each
    entity's currency from, names exactly one role `shape` declares. An ambiguous role
    names both."""
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
                f"{axis_roles[1]!r}, on the {role!r} axis named by a unit binding's "
                "currency role; no single role can serve it"
            ),
        ) from exc
    except RecipeNotDeclaredError as exc:
        raise CurrencyRoleUndeclared(
            stat_var=shape.stat_var, role=role, declared_roles=shape.declared_roles
        ) from exc


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


def shape_measurement_qualifier(identity: Mapping[str, object]) -> str | None:
    """The `measurementQualifier` dcid a described identity's shape states, which every member
    shares, or `None` when the shape states none or the identity is pinned."""
    shape_member = cast("Mapping[str, object]", identity.get("shape", {}))
    qualifier = shape_member.get("measurement_qualifier")
    return qualifier if isinstance(qualifier, str) else None


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
    among the five-tuple's publishers, and its rows of the role class the shared election
    reads for the query's role keys (`statvar_members.elect_role_class`). Return its
    antichain-minimal rows as shapes. Each shape carries the elected dataset, every dataset
    the index records its row under, and the dcids of every family row. Raise
    `SpecNotExportable` when none match, `FacetNotAvailableError` when the query names a
    dataset that does not publish its five-tuple, `RoleSignatureIncompatible` when that
    election ties (`DatasetFamily.tied`), naming the roles of the antichain-minimal rows
    among those that place the role keys, or `MemberNotResolvable` for `observationAbout`.
    The in-memory path has no page truncation or member-count ceiling.
    """
    five_tuple, slots = query.five_tuple, query.slots
    try:
        elected = index_family(code_index, query)
    except SourceNotInFamily as exc:
        raise FacetNotAvailableError(
            f"spec {cut(spec_id)!r} binds provenance {cut(exc.source)!r}, which does not publish "
            f"this five-tuple; it is published by {list(exc.candidates)!r}",
            facet_pointer=_PROVENANCE_POINTER,
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
    if elected.tied:
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


def resolve_definition_members(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
    bounds: MemberReadBounds,
    role_keys: Collection[str] | None = None,
    open_role_keys: Collection[str] | None = None,
) -> ResolvedMembers:
    """Resolve the members denoted by an identity's shape against the graph or code index.

    `ResolvedMembers.shapes` holds the antichain-minimal members and
    `ResolvedMembers.family` every member before that reduction: the pinned dcid,
    every matching index row of the elected dataset, or every confirmed walked candidate.

    `code_index` is the caller's resolved index, shared by enumeration and
    frame rendering so both use the same declarations.

    `bounds` lets the graph walk settle a first page that truncates without a confirmed
    root by reading the whole filtered set up to `bounds.subject_ceiling`, every read ending
    by `bounds.deadline`. Every caller walks under the bounds of a mint
    (`member_read_bounds`), on every door, so one identity resolves the same members
    wherever it is read. Membership is a function of the identity, the graph and the code
    index.

    `role_keys` replaces the role keys of the identity's own `entities` in the
    role-class election, for a caller whose scope binds other entities, and
    `open_role_keys` replaces its own `open_roles`. `None` elects under the
    identity's own.

    A pinned arm resolves to one `StatVarShape`. A described arm widens
    hierarchical `value` and `set` slots to the self-inclusive subtree via
    `CodeIndex.expansion`. Leaves, flat systems, codes of a system whose parent row is
    the total of its children, and unknown codes pass through unchanged. The graph walk builds all shapes in one read.

    A family holding a row the index lists outside its code systems, as IDS families do,
    resolves in memory to the members the graph walk would elect, with no graph read
    (`code_index.members.listed_group`).

    If the identity's slot keys hold every constraint property of at least one
    index row of the five-tuple (`index_decides`), resolve the elected
    dataset's minimal rows in memory. `facets.provenance` names the dataset when it
    binds one distinct ref the index holds rows of
    (`code_index.members.named_dataset`). A pin on a provenance only listed rows record,
    as on IDS, scopes the reads alone (`pinned_provenance_scope`) and resolves the
    unpinned members. An identity whose keys cover a row and whose values match none
    refuses `SpecNotExportable` on this path. This bounded index path shares
    empty-result and role-signature refusals with the graph walk. The graph walk
    enforces the enumerator's member cap and refuses incomplete candidate pages.

    Raises `IdentityError` before inspecting the shape if the identity is invalid.
    The pinned arm maps `resolve_shape` refusals to `MemberNotResolvable` for `spec_id`.
    The described arm raises `SpecNotExportable` for an unanchored query or empty result.
    Enumeration over its member cap raises `OverCeilingError` with
    the exact count. A walk that ends truncated and unproven, past the first page's window
    or past `bounds.subject_ceiling`, raises `candidate_window` with the count read and the
    window it read. Membership beyond the window is unmeasured. A walk past
    `bounds.deadline` raises `MemberReadDeadline`, a transient timeout. A
    proven-complete antichain is accepted. Same-keyset duplicates beyond the first page may
    be absent from the answer.

    Invalid candidate arcs raise `MemberNotResolvable` for `spec_id`.
    Inconsistent forward and reverse arcs raise it with up to 10 candidate
    ids and `truncated=True` when more exist, unless the group holds a
    confirmed root (`MemberGroup.root_confirmed`) that dominates every one of
    them (`MemberGroup.undismissed_candidates` is empty). A candidate that may
    duplicate the root still refuses, since its roles are unverified. The refusal is
    retryable when a later walk can settle a candidate it names
    (`MemberGroup.undismissed_rereadable`).
    Both paths elect one role class (`statvar_members.elect_role_class`): the
    members whose declared roles cover the role keys `entities` is bound under,
    least-constrained first, then those declaring the fewest roles, so a member
    lacking a role its sibling carries answers as the aggregate over that role.
    `RoleSignatureIncompatible` reports a tie that election cannot break, between
    members whose role sets do not contain one another or whose fewest-role members
    hold constraint keys the others lack, and takes precedence under a confirmed
    root, where an unverified candidate cannot remove a member.

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
    if open_role_keys is not None:
        query = dataclasses.replace(query, open_role_keys=frozenset(open_role_keys))
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
        group = listed_group(code_index, query, member_cap=enumerator.member_cap)
        path = "listed"
        if group is None:
            path = "walk"
            group = enumerator.enumerate(
                graph=graph,
                stamp_key=graph.generation_stamp().key,
                five_tuple=five_tuple,
                slots=slots,
                readings=_absent_readings(code_index),
                role_keys=query.role_keys,
                open_role_keys=query.open_role_keys,
                subject_ceiling=bounds.subject_ceiling,
                deadline=bounds.deadline,
                spec_id=spec_id,
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
            "unknown, not zero. " + _CANDIDATE_WINDOW_REMEDY,
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
            retryable=group.undismissed_rereadable,
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
    stated = _member_family(group.stated_family) if group.stated_family is not None else None
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
        "datasets.member_resolution path=%s spec_id=%s member_count=%d index_content_hash=None",
        path,
        spec_id,
        len(shapes),
    )
    return ResolvedMembers(shapes=shapes, family=walked)


def check_reproducible(
    identity: Mapping[str, object], *, spec_id: str, shape: StatVarShape
) -> None:
    """Check identity role keys and open roles against the resolved shape without graph
    access.

    Keys may use raw graph roles or flow-axis aliases. `declared_role_for`
    maps both. Described members share one declared-role set. A pinned
    identity's hash follows its dcid, so roles are its reproducibility check.
    An open role must be declared too, since the identity asks for every member
    on it. Raises `DefinitionNotReproducible` with `spec_id` and unsupported roles.
    """
    roles = [
        *cast("Mapping[str, object]", identity.get("entities", {})),
        *cast("Sequence[str]", identity.get("open_roles", ())),
    ]
    undeclared = sorted(
        role for role in roles if declared_role_for(role, shape.declared_roles) is None
    )
    if undeclared:
        raise DefinitionNotReproducible(
            spec_ids=(spec_id,),
            undeclared_roles=tuple(undeclared),
            detail=(
                f"spec {spec_id!r} binds or opens role(s) {', '.join(undeclared)} that "
                f"{shape.stat_var!r} no longer declares"
            ),
        )


def rehydrate_definition(
    stored: StoredDefinition,
    *,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
    bounds: MemberReadBounds,
) -> tuple[tuple[StatVarShape, ...], tuple[MaterialisationRequest, ...]]:
    """Re-resolve a stored identity, check its roles, and build one request per shape.

    Pinned identities return one shape and request. Described identities
    return one pair per current member. `index_sourced` is shared across the
    requests and lets multi-member facet grouping accept shared provenance
    and unit. Reuse the caller's `code_index`. `bounds` bounds the member walk as in
    `resolve_definition_members`.
    """
    shapes = resolve_definition_members(
        stored.identity,
        spec_id=stored.spec_id,
        graph=graph,
        enumerator=enumerator,
        code_index=code_index,
        bounds=bounds,
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
