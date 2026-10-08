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

from dfo_platform.named import named_capped
from materialise.codes import MATERIALISE_CODE_CATALOGUE
from materialise.errors import MaterialisationError
from materialise.formats import FORMATS_CODE_CATALOGUE

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

    def __init__(self, *, variables: tuple[str, ...], provenances: tuple[str, ...]) -> None:
        self.variables = variables
        self.provenances = provenances
        detail = (
            f"{', '.join(variables)} has data at the requested places only under provenances "
            f"the origin registry does not admit: {', '.join(provenances)}"
        )
        super().__init__(detail, code="excluded_by_registry", status=422, detail=detail)


class UnSystemUnavailable(MaterialisationError):
    """502 `upstream_error`, retryable. Raised in place of `default_facet_undefined` when a UN
    series has no candidate and the UN could not serve it at some of its places, because the
    front door refused them or the read failed, so the empty answer says nothing about whether
    data exists there. `places` names them all in first-occurrence order, and the detail names
    the first five (`named_capped`)."""

    def __init__(self, *, variable: str, places: tuple[str, ...]) -> None:
        self.variable = variable
        self.places = places
        detail = (
            f"the UN System Data Commons could not serve observations of {variable} at "
            f"{named_capped(places)}; retry"
        )
        super().__init__(detail, code="upstream_error", status=502, detail=detail, retryable=True)


class MemberNotResolvable(Exception):
    """Raised when StatVars resolve to zero or multiple registered shapes.

    This maps to `member_not_resolvable`, 422, with one issue per member. `pointer` is set by
    `kinds.spec_id`. `kinds.statvar_set` uses `/input/statvars`. `truncated` marks
    a capped list of disagreeing members.
    """

    pointer: str | None = None

    def __init__(self, *, members: tuple[str, ...], truncated: bool = False) -> None:
        self.members = members
        self.truncated = truncated
        detail = f"member(s) did not resolve to exactly one registered shape: {members!r}"
        if truncated:
            detail += "; more members disagreed than are named here"
        super().__init__(detail)


class DefinitionNotReproducible(Exception):
    """A stored identity fails against the graph's current StatVar shape.

    `undeclared_roles` lists bound or open roles the shape lacks. It is empty without a
    role-level cause.
    """

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


class SpecNotExportable(Exception):
    """A described-arm identity has no pinned StatVar series to export."""

    def __init__(self, *, spec_id: str, detail: str) -> None:
        super().__init__(detail)
        self.spec_id = spec_id
        self.detail = detail


class InvalidRoleRef(Exception):
    """`scope.entities` names a role absent from the shape or a graph node whose type differs from
    `Property`. This maps to `invalid_role_ref`, 422, with one issue per role.

    `declared_roles` carries the shape's role list, or is empty when no shape is available. A
    custom `message` can describe an ambiguous-axis binding. `role` and `declared_roles` remain
    available to the wire mapper.
    """

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
        super().__init__(
            message
            if message is not None
            else f"stat_var {stat_var!r} does not declare role {role!r}"
        )


class RoleIncompatible(Exception):
    """A runtime binding is refused when `slots` pins the role, even if the binding repeats that
    value. This also covers `observationAbout` on shapes with other roles and graph properties
    absent from the shape. It maps to `role_incompatible`, 422, with one issue per role.

    `declared_roles` carries the shape's roles. A custom `message` describes colliding spellings
    or ambiguous axes. The wire fields remain set.
    """

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


class CurrencyRoleUndeclared(Exception):
    """The unit request's role-shaped `currency` names a role absent from the StatVar shape or one
    that shares its axis with another role. `graph_role_for_spec_role` raises `role_undeclared`,
    which this exception maps to `request_invalid`, 400.

    Request kinds attach `pointer`. Stored-spec retrieval leaves `pointer` unset. A custom message
    can name conflicting roles. This concerns a role inside the unit value. `InvalidRoleRef` and
    `RoleIncompatible` cover `scope.entities`. Roles outside `_CURRENCY_ROLES` fail identity
    validation first.
    """

    pointer: str | None = None

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
        super().__init__(
            message
            if message is not None
            else (
                f"stat_var {stat_var!r} does not declare role {role!r}, named by a unit "
                "request's role-shaped currency"
            )
        )


class RoleSignatureIncompatible(Exception):
    """Described-arm members disagree on `declared_roles`.

    This maps to `role_signature_incompatible`, 422, with one issue per member listing its roles.
    """

    def __init__(self, *, spec_id: str, roles_by_stat_var: Mapping[str, tuple[str, ...]]) -> None:
        self.spec_id = spec_id
        self.roles_by_stat_var = dict(roles_by_stat_var)
        super().__init__(
            f"spec {spec_id!r} enumerates members disagreeing on declared_roles: "
            f"{dict(sorted(self.roles_by_stat_var.items()))!r}"
        )


class CapabilityUnavailable(Exception):
    """The request uses a disabled format, reshape, or conversion capability. It maps to
    `capability_unavailable`, 501. `pointer` identifies a disabled enrichment field. `groupby`
    leaves it unset because the option is a boolean.
    """

    def __init__(self, *, detail: str, pointer: str | None = None) -> None:
        self.detail = detail
        self.pointer = pointer
        super().__init__(detail)


class SpecNotFound(Exception):
    """The registry and request sidecar lack this definition. It maps to `spec_not_found`, 404.

    The raising kind sets `pointer` to `/input/spec_id`.
    """

    def __init__(self, *, subject_id: str, pointer: str) -> None:
        self.subject_id = subject_id
        self.pointer = pointer
        super().__init__(f"{subject_id!r} is not registered, and nothing was supplied to mint it")


class MalformedSidecarIdentity(Exception):
    """A sidecar fails identity-core validation. It maps to `request_invalid`, 400, with a
    request pointer, which the raising kind sets.
    """

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


class RenderedArtifactSigningFailed(Exception):
    """A signed URL failed after the rendered bytes were stored.

    The raise site copies safe error fields outside the caught exception so `__context__` stays
    unset. `error_class_name` and `status_code` distinguish permanent causes from outages.
    """

    def __init__(
        self,
        *,
        bucket: str,
        object_name: str,
        digest: str,
        error_class_name: str | None,
        status_code: int | None,
    ) -> None:
        self.bucket = bucket
        self.object_name = object_name
        self.digest = digest
        self.error_class_name = error_class_name
        self.status_code = status_code
        detail = (
            f"failed to sign a GET url for a rendered artifact object {object_name!r} "
            f"in bucket {bucket!r} (digest {digest!r})"
        )
        if error_class_name is not None:
            detail += f"; underlying failure class {error_class_name!r}"
        if status_code is not None:
            detail += f", status {status_code!r}"
        super().__init__(detail)
