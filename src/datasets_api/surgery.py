"""Apply a mint request's `scope` to a stored identity-core document.
Supplied members replace their whole stored value. Unset members carry
forward. The caller canonicalises and hashes the result, then writes it
when its id changes.

A scope that binds a role on the axis of one of the identity's `open_roles` closes it, and
the role leaves `open_roles`. Every other open role stays (`open_roles_left_open`), and
the scoped identity elects under it, open, beside the scope's bound keys.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import cast

from datasets_api.adapter import facets_to_identity, window_to_identity
from datasets_api.mint_models import Scope
from dfo_platform.roles import role_key_axis


def open_roles_left_open(identity: Mapping[str, object], bound_roles: Collection[str]) -> list[str]:
    """`identity`'s open roles less each one whose axis a key of `bound_roles` names."""
    bound_axes = {role_key_axis(role) for role in bound_roles}
    open_roles = cast("Sequence[str]", identity.get("open_roles", ()))
    return [role for role in open_roles if role not in bound_axes]


def apply_scope_surgery(identity: Mapping[str, object], *, scope: Scope) -> dict[str, object]:
    """Replace supplied scope members and preserve unset ones.

    `scope.window` maps to `requested_window`. Empty entities clear all
    bound roles and leave open roles open. An open role the scope binds leaves
    `open_roles`, which is removed when none stays. Empty facets remove the key,
    matching fresh identity documents.
    """
    document = dict(identity)

    if scope.entities is not None:
        document["entities"] = {role: list(refs) for role, refs in scope.entities.items()}
        open_roles = open_roles_left_open(identity, scope.entities)
        if open_roles:
            document["open_roles"] = open_roles
        else:
            document.pop("open_roles", None)

    if scope.window is not None:
        document["requested_window"] = window_to_identity(scope.window)

    if scope.facets is not None:
        if scope.facets:
            document["facets"] = facets_to_identity(scope.facets)
        else:
            document.pop("facets", None)

    return document
