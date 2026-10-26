"""Validate a resubmitted identity.

A posted identity passes the identity core before any door reads it, and Search doors
refuse the `exact` binding, the pinned-StatVar arm and an identity binding more places than
a Spec holds. A composite provenance is read as the identity its option was set on
(`composite_table_identity`). The resubmit and options doors share these checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dfo_spec_kit import identity as _identity

from code_index.composites import CompositeNotServable, composite_ref, table_identity
from code_index.shared import get_index
from dfo_platform.named import cut
from dfo_platform.roles import role_key_axis
from search.engine.config import settings
from search.engine.errors import EngineInputError
from search.models import (
    BaseIdentity,
    IdentityBindingExact,
    IdentityShape,
    IdentityShapePin,
)


def check_no_exact_bindings(identity: BaseIdentity, *, pointer: str) -> None:
    """Refuse the first `exact` slot binding at its JSON pointer.

    Search readers do not handle this identity-core binding; only dataset
    readers do. Run this before core validation, which accepts `exact`.
    """
    for i, slot in enumerate(identity.slots):
        if isinstance(slot.binding, IdentityBindingExact):
            raise EngineInputError(
                f"slot {cut(slot.property)!r} carries an 'exact' binding; no Search "
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
        _identity.validate_identity_core(identity.model_dump(exclude_none=True))
    except ValueError as exc:
        raise EngineInputError(
            f"identity is not a well-formed identity document: {cut(str(exc))}",
            code="request_invalid",
            pointer=pointer,
        ) from exc


def open_requested_role(identity: BaseIdentity, *, role: str) -> BaseIdentity:
    """Return `identity` with the flow axis `role` names in `open_roles`.

    An options request for a flow axis asks for that axis member by member, so the
    population it reads is the one an edit binding that axis elects, such as the by-creditor
    twin of an IDS total. `identity` comes back unchanged when `role` names no flow axis
    (`observationAbout`) or when an `entities` key or an open role already names the axis.
    Keys match by axis (`dfo_platform.roles.role_key_axis`), since they keep the caller's
    spelling. A population whose StatVars declare no role on the axis drops the open role
    at election, so the door's role-target check still refuses it.
    """
    axis = role_key_axis(role)
    if axis is None:
        return identity
    named = {role_key_axis(key) for key in identity.entities} | set(identity.open_roles or ())
    if axis in named:
        return identity
    return BaseIdentity.model_validate(
        identity.model_dump(exclude_none=True)
        | {"open_roles": [*(identity.open_roles or ()), axis]}
    )


def refuse_over_place_bound(
    entities: Mapping[str, Sequence[str]], *, pointer: str = "/input/identity/entities"
) -> None:
    """Refuse with `request_invalid` an identity that binds more places than a Spec holds
    (`SearchSettings.max_child_places`). Places count as distinct dcids per role, summed
    over the roles.

    `pointer` is the door's own path to the identity's entity map. The refusal names the
    first role that exceeds the bound alone, and the whole entity map when only the sum
    does."""
    bound = settings().max_child_places
    counts = {role: len(set(dcids)) for role, dcids in entities.items()}
    for role, count in counts.items():
        if count > bound:
            raise EngineInputError(
                f"{count} places on {role} exceed the {bound}-place limit a dataset binds; "
                "re-post with fewer places",
                code="request_invalid",
                pointer=f"{pointer}/{role}",
            )
    if (total := sum(counts.values())) > bound:
        raise EngineInputError(
            f"{total} places across {len(counts)} roles exceed the {bound}-place limit a "
            "dataset binds; re-post with fewer places",
            code="request_invalid",
            pointer=pointer,
        )


def composite_table_identity(identity: BaseIdentity, *, pointer: str) -> BaseIdentity:
    """The identity a Search door reads for `identity`: for a composite
    (`code_index.composites`), the identity its option was set on
    (`code_index.composites.table_identity`), whose one table every graph and index read
    names; `identity` itself otherwise. Refuses a composite whose components cannot be derived
    (`CompositeNotServable`) as `request_invalid`, at `pointer`, the posted identity, joined
    with the member at fault."""
    document = identity.model_dump(mode="json", exclude_none=True)
    try:
        read = table_identity(get_index(), document)
    except CompositeNotServable as exc:
        raise EngineInputError(
            str(exc), code="request_invalid", pointer=f"{pointer}{exc.pointer}"
        ) from exc
    return identity if read is document else BaseIdentity.model_validate(read)


def check_not_composite(identity: BaseIdentity, *, pointer: str) -> None:
    """Refuse an identity whose `facets.provenance` names a declared composite
    (`code_index.composites`), at `pointer`, the posted identity's provenance facet.

    No table publishes a composite, so its pin scopes every read to nothing: a Search door
    resolves the identity its option was set on (`composite_table_identity`) in its place,
    and this guards the resolve that reads it."""
    try:
        ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable as exc:
        raise EngineInputError(str(exc), code="request_invalid", pointer=pointer) from exc
    if ref is not None:
        raise EngineInputError(
            f"provenance {ref!r} is a composite of other tables' series; post each of its "
            "component identities instead",
            code="request_invalid",
            pointer=pointer,
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
