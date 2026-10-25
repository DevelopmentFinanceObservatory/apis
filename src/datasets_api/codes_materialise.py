"""Refusal codes raised by the mint and definition doors.

Materialise and format codes reuse their source catalogues' text. Shared platform codes cover
facet, role, and currency-conversion refusals. Redeclaring them would raise
`CodeCollision`. `SpecNotFound` and reshape refusals are registered here because this package
raises their exception classes.

`spec_not_exportable` covers described-arm asks with no graph member matching their shape.
`definition_not_reproducible` covers stored identities that fail to reproduce against the graph.
`role_signature_incompatible` covers enumerated members whose `declared_roles` differ.
`rollup_not_summable` and pivot/column refusals come from the reshape paths wired by this surface.
`excluded_by_registry` replaces `default_facet_undefined` when the only data for the requested
series sits under provenances the origin registry does not admit (`datasets_api.origin`).
`UnSystemUnavailable` replaces it with the shared `upstream_error` when the UN could not serve a UN
series at some of its places.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dfo_platform.named import named_capped
from dfo_platform.problem import Alternative, ProblemIssue, Refusal
from dfo_platform.roles import axis_for_role, declared_role_for
from materialise.codes import MATERIALISE_CODE_CATALOGUE
from materialise.errors import MaterialisationError
from materialise.formats import FORMATS_CODE_CATALOGUE
from materialise.recipes import RecipeNotDeclaredError, SpecRole, graph_role_for_spec_role

MATERIALISE_DOOR_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    "excluded_by_registry": (
        "Excluded By Registry",
        "The requested variable has data at the requested places only under provenances the "
        "origin registry does not admit; the refusal names those provenances.",
    ),
    "member_not_resolvable": (
        "Member Not Resolvable",
        "A StatVar did not resolve to exactly one registered shape.",
    ),
    "definition_not_reproducible": (
        "Definition Not Reproducible",
        "This registered definition no longer reproduces its own id against the "
        "graph's current account of its StatVar.",
    ),
    "spec_not_exportable": (
        "Spec Not Exportable",
        "This spec resolves to a breadth-only coverage region with no enumerable member series.",
    ),
    "rollup_not_summable": (
        "Rollup Not Summable",
        "The requested groupby roll-up fails summability or one of its preconditions.",
    ),
    "pivot_not_unique": (
        "Pivot Not Unique",
        "The requested pivot produces duplicate cells under its index and column keys.",
    ),
    "column_not_droppable": (
        "Column Not Droppable",
        "The requested column selection drops a grain, unit, or provenance column.",
    ),
    "column_unknown": (
        "Column Unknown",
        "The requested columns/pivot/enrichment reference a column the frame does not "
        "hold at this reshape point.",
    ),
    "pivot_over_column_cap": (
        "Pivot Over Column Cap",
        "The requested pivot's output width exceeds the target format's column cap.",
    ),
    "pivot_over_cell_cap": (
        "Pivot Over Cell Cap",
        "The requested pivot's dense output would exceed this deployment's output-cell cap.",
    ),
    "pivot_not_homogeneous": (
        "Pivot Not Homogeneous",
        "The requested pivot would merge rows that differ on a grain, unit, or provenance "
        "column it drops.",
    ),
    "excel_limit": FORMATS_CODE_CATALOGUE["excel_limit"],
    "excel_illegal_character": FORMATS_CODE_CATALOGUE["excel_illegal_character"],
    "parquet_size_bound": FORMATS_CODE_CATALOGUE["parquet_size_bound"],
    "entities_unbound": MATERIALISE_CODE_CATALOGUE["entities_unbound"],
    "default_facet_undefined": MATERIALISE_CODE_CATALOGUE["default_facet_undefined"],
    "over_ceiling": MATERIALISE_CODE_CATALOGUE["over_ceiling"],
    "collection_probe_ceiling": MATERIALISE_CODE_CATALOGUE["collection_probe_ceiling"],
    "members_not_held": MATERIALISE_CODE_CATALOGUE["members_not_held"],
}


class ExcludedByRegistry(MaterialisationError):
    """422 `excluded_by_registry`. Raised in place of `default_facet_undefined` when every facet
    for the requested variables at the requested places sits under a provenance the origin
    registry does not admit. `provenances` names them in first-occurrence order."""

    code = "excluded_by_registry"
    status = 422

    def __init__(self, *, variables: tuple[str, ...], provenances: tuple[str, ...]) -> None:
        self.variables = variables
        self.provenances = provenances
        detail = (
            f"{', '.join(variables)} has data at the requested places only under provenances "
            f"the origin registry does not admit: {', '.join(provenances)}"
        )
        super().__init__(detail, detail=detail)


class UnSystemUnavailable(MaterialisationError):
    """502 `upstream_error`, retryable. Raised in place of `default_facet_undefined` when a UN
    series has no candidate and the UN could not serve it at some of its places, because the
    front door refused them or the read failed, so the empty answer says nothing about whether
    data exists there. `places` names them all in first-occurrence order, and the detail names
    the first five (`named_capped`)."""

    code = "upstream_error"
    status = 502
    retryable = True

    def __init__(self, *, variable: str, places: tuple[str, ...]) -> None:
        self.variable = variable
        self.places = places
        detail = (
            f"the UN System Data Commons could not serve observations of {variable} at "
            f"{named_capped(places)}; retry"
        )
        super().__init__(detail, detail=detail)


def _role_keys(declared_roles: tuple[str, ...]) -> list[str]:
    """Every `scope.entities` key that binds one of `declared_roles`.

    The `origin` and `destination` aliases come first, in declared order, for each role an
    alias resolves to alone; the declared role names follow."""
    aliases = [
        axis
        for role in declared_roles
        if (axis := axis_for_role(role)) is not None
        and declared_role_for(axis, declared_roles) == role
    ]
    return [*aliases, *declared_roles]


def _currency_role_keys(declared_roles: tuple[str, ...]) -> list[str]:
    """The `_role_keys` a unit request's role-shaped `currency` accepts for `declared_roles`:
    those `graph_role_for_spec_role` maps to one declared role."""
    keys: list[str] = []
    for key in _role_keys(declared_roles):
        try:
            # A key outside `SpecRole` raises `ValueError`, which skips it.
            graph_role_for_spec_role(spec_role=cast("SpecRole", key), declared_roles=declared_roles)
        except (RecipeNotDeclaredError, ValueError):
            continue
        keys.append(key)
    return keys


def _scope_role_issue(role: str, declared_roles: tuple[str, ...]) -> ProblemIssue:
    """The issue for a `scope.entities` role, listing the keys that bind a declared role."""
    return ProblemIssue(
        pointer=f"/input/scope/entities/{role}",
        member=role,
        axis=role,
        alternatives=[Alternative(ref=key, label=key) for key in _role_keys(declared_roles)],
        truncated=False,
    )


class MemberNotResolvable(Refusal):
    """Raised when StatVars resolve to zero or multiple registered shapes.

    This maps to `member_not_resolvable` with one issue per member at `issue_pointer`, which
    `kinds.spec_id` sets and which defaults to `kinds.statvar_set`'s `/input/statvars`.
    `truncated` marks a capped list of disagreeing members. `retryable` marks members a
    member walk left unsettled on reads that can drop an arc
    (`MemberGroup.undismissed_rereadable`), which a later walk can settle.
    """

    code = "member_not_resolvable"
    issue_pointer = "/input/statvars"

    @property
    def issues(self) -> tuple[ProblemIssue, ...]:
        return tuple(ProblemIssue(pointer=self.issue_pointer, member=m) for m in self.members)

    def __init__(
        self, *, members: tuple[str, ...], truncated: bool = False, retryable: bool = False
    ) -> None:
        self.members = members
        self.truncated = truncated
        self.retryable = retryable
        detail = f"member(s) did not resolve to exactly one registered shape: {members!r}"
        if truncated:
            detail += "; more members disagreed than are named here"
        super().__init__(detail)


class DefinitionNotReproducible(Refusal):
    """A stored identity fails against the graph's current StatVar shape.

    `undeclared_roles` lists bound or open roles the shape lacks. It is empty without a
    role-level cause. One issue names each spec, and one more each spec and undeclared role.
    """

    code = "definition_not_reproducible"

    def __init__(
        self,
        *,
        spec_ids: tuple[str, ...],
        detail: str,
        undeclared_roles: tuple[str, ...] = (),
    ) -> None:
        super().__init__(detail)
        self.spec_ids = spec_ids
        self.detail = detail
        self.undeclared_roles = undeclared_roles
        self.issues = (
            *(ProblemIssue(member=spec_id) for spec_id in spec_ids),
            *(
                ProblemIssue(member=spec_id, axis=role)
                for spec_id in spec_ids
                for role in undeclared_roles
            ),
        )


class SpecNotExportable(Refusal):
    """A described-arm identity has no pinned StatVar series to export."""

    code = "spec_not_exportable"

    def __init__(self, *, spec_id: str, detail: str) -> None:
        super().__init__(detail)
        self.spec_id = spec_id
        self.detail = detail


class InvalidRoleRef(Refusal):
    """`scope.entities` names a role absent from the shape or a graph node whose type differs from
    `Property`. This maps to `invalid_role_ref`, 422, with one issue per role.

    `declared_roles` carries the shape's role list, or is empty when no shape is available. A
    custom `message` can describe an ambiguous-axis binding. `role` and `declared_roles` remain
    available to the wire mapper. The issue's alternatives are the keys that bind a declared
    role: the `origin` and `destination` aliases, then the declared role names.
    """

    code = "invalid_role_ref"

    def __init__(
        self,
        *,
        stat_var: str,
        role: str,
        declared_roles: tuple[str, ...] = (),
        message: str | None = None,
    ) -> None:
        self.stat_var = stat_var
        self.role = role
        self.declared_roles = declared_roles
        self.issues = (_scope_role_issue(role, declared_roles),)
        super().__init__(
            message
            if message is not None
            else f"stat_var {stat_var!r} does not declare role {role!r}"
        )


class RoleIncompatible(Refusal):
    """A runtime binding is refused when `slots` pins the role, even if the binding repeats that
    value. This also covers `observationAbout` on shapes with other roles and graph properties
    absent from the shape. It maps to `role_incompatible`, 422, with one issue per role.

    `declared_roles` carries the shape's roles. A custom `message` describes colliding spellings
    or ambiguous axes. The wire fields remain set. The issue's alternatives are the keys that
    bind a declared role: the `origin` and `destination` aliases, then the declared role names.
    """

    code = "role_incompatible"

    def __init__(
        self,
        *,
        stat_var: str,
        role: str,
        pinned_value: str | None = None,
        declared_roles: tuple[str, ...] = (),
        message: str | None = None,
    ) -> None:
        self.stat_var = stat_var
        self.role = role
        self.pinned_value = pinned_value
        self.declared_roles = declared_roles
        self.issues = (_scope_role_issue(role, declared_roles),)
        if message is not None:
            pass
        elif pinned_value is not None:
            message = (
                f"stat_var {stat_var!r} pins role {role!r} to {pinned_value!r}; "
                "scope.entities cannot rebind it"
            )
        else:
            message = f"stat_var {stat_var!r} does not declare role {role!r}"
        super().__init__(message)


class CurrencyRoleUndeclared(Refusal):
    """The unit request's role-shaped `currency` names a role absent from the StatVar shape or one
    that shares its axis with another role. `graph_role_for_spec_role` raises `role_undeclared`,
    which this exception maps to `request_invalid`, 400.

    Request kinds attach `pointer`. Stored-spec retrieval leaves `pointer` unset. A custom message
    can name conflicting roles. This concerns a role inside the unit value. `InvalidRoleRef` and
    `RoleIncompatible` cover `scope.entities`. Roles outside `_CURRENCY_ROLES` fail identity
    validation first. The top-level alternatives are the currency roles that bind a declared
    role (`origin`, `destination`, `observationAbout`).
    """

    code = "request_invalid"

    def __init__(
        self,
        *,
        stat_var: str,
        role: str,
        declared_roles: tuple[str, ...],
        message: str | None = None,
    ) -> None:
        self.role = role
        self.declared_roles = declared_roles
        self.alternatives = [
            Alternative(ref=key, label=key) for key in _currency_role_keys(declared_roles)
        ]
        super().__init__(
            message
            if message is not None
            else (
                f"stat_var {stat_var!r} does not declare role {role!r}, named by a unit "
                "request's role-shaped currency"
            )
        )


class RoleSignatureIncompatible(Refusal):
    """Described-arm members disagree on `declared_roles`.

    This maps to `role_signature_incompatible` with one issue per member.
    """

    code = "role_signature_incompatible"

    def __init__(self, *, spec_id: str, roles_by_stat_var: Mapping[str, tuple[str, ...]]) -> None:
        self.spec_id = spec_id
        self.roles_by_stat_var = dict(roles_by_stat_var)
        self.issues = tuple(ProblemIssue(member=sv) for sv in sorted(self.roles_by_stat_var))
        super().__init__(
            f"spec {spec_id!r} enumerates members disagreeing on declared_roles: "
            f"{dict(sorted(self.roles_by_stat_var.items()))!r}"
        )


class CapabilityUnavailable(Refusal):
    """The request uses a disabled format, reshape, or conversion capability. It maps to
    `capability_unavailable`. `pointer` identifies a disabled enrichment field. `groupby`
    leaves it unset because the option is a boolean.
    """

    code = "capability_unavailable"

    def __init__(self, *, detail: str, pointer: str | None = None) -> None:
        self.detail = detail
        self.pointer = pointer
        super().__init__(detail)


class SpecNotFound(Refusal):
    """The registry and request sidecar lack this definition. It maps to `spec_not_found`.

    The raising kind sets `pointer` to `/input/spec_id`.
    """

    code = "spec_not_found"

    def __init__(self, *, subject_id: str, pointer: str) -> None:
        self.subject_id = subject_id
        self.pointer = pointer
        super().__init__(f"{subject_id!r} is not registered, and nothing was supplied to mint it")


class MalformedSidecarIdentity(Refusal):
    """A sidecar fails identity-core validation. It maps to `request_invalid` with a request
    pointer, which the raising kind sets.
    """

    code = "request_invalid"

    def __init__(self, *, detail: str, pointer: str) -> None:
        self.pointer = pointer
        super().__init__(detail)


# Match recorded exception class names. Credential/certificate defects are permanent.
# `RefreshError` and `DefaultCredentialsError` stay transient because google-auth also uses them
# for metadata-server outages with no distinguishing status.
_PERMANENT_SIGNING_FAILURE_CLASS_NAMES = frozenset(
    {
        "MalformedError",
        "InvalidValue",
        "InvalidType",
        "InvalidOperation",
        "InvalidResource",
        "ClientCertError",
    }
)


def signing_failure_is_permanent(*, error_class_name: str | None, status_code: int | None) -> bool:
    """Identify permission and malformed-credential failures that an operator can fix.

    `401` and `403` are permanent. GCS `403` can also mean `rateLimitExceeded`, but the signing
    error exposes no reason, so the classifier fails closed on broken IAM. Listed malformed
    credential classes are permanent without a status. Other statuses, unknown class/status pairs,
    and `RefreshError` or `DefaultCredentialsError` without a status are transient.

    Artifact bytes and inventory are already committed here. A false transient costs a retry. A
    false permanent refusal could block a request that would succeed on retry.
    """
    if status_code in (401, 403):
        return True
    return error_class_name in _PERMANENT_SIGNING_FAILURE_CLASS_NAMES
