"""Apply a mint request's `scope` to a stored identity-core document.
Supplied members replace their whole stored value. Unset members carry
forward. The caller canonicalises and hashes the result, then writes it
when its id changes.
"""

from __future__ import annotations

from collections.abc import Mapping

from datasets_api.adapter import facets_to_identity, window_to_identity
from datasets_api.mint_models import Scope


def apply_scope_surgery(identity: Mapping[str, object], *, scope: Scope) -> dict[str, object]:
    """Replace supplied scope members and preserve unset ones.

    `scope.window` maps to `requested_window`. Empty entities clear all
    roles. Empty facets remove the key, matching fresh identity documents.
    """
    document = dict(identity)

    if scope.entities is not None:
        document["entities"] = {role: list(refs) for role, refs in scope.entities.items()}

    if scope.window is not None:
        document["requested_window"] = window_to_identity(scope.window)

    if scope.facets is not None:
        if scope.facets:
            document["facets"] = facets_to_identity(scope.facets)
        else:
            document.pop("facets", None)

    return document
