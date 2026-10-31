"""`SpecDefinition`, the self-contained definition a Spec was resolved from."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel
from .identity import BaseIdentity
from .vocab import SCHEMA_VERSION


class SpecDefinition(StrictModel):
    """The canonical definition a Spec was resolved from: `{schema_version,
    spec_id, identity}`. This is the form the registry stores and permalinks
    carry, so it is self-contained. A definition lifted out of its Spec still
    names which spec it is.

    `identity` is the identity-core projection (see `BaseIdentity`) this
    Spec's spec_id was minted from, so hashing `identity` again reproduces
    `spec_id` by construction.

    `spec_id` is carried here as well as on `Spec.spec_id` deliberately: the
    duplication is what makes the definition independently meaningful once
    detached from the response that delivered it. The one exception is an
    `unsupported` response's definition, which omits it: the request was
    valid and understood but never became a resolvable construction, so no
    spec_id was ever legitimately minted for it.
    """

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="contract schema version this definition was published under; exactly 1",
    )
    spec_id: str = Field(
        description=(
            "the spec_id minted from `identity`; equals the sibling Spec.spec_id, "
            "and is what a detached definition (registry row, permalink) is keyed by"
        )
    )
    identity: BaseIdentity = Field(
        description=(
            "the identity-core projection this Spec's spec_id was minted from, "
            "serialised once and never re-derived; post it back byte-for-byte as "
            "ResubmitIdentityPayload.identity to edit this Spec. Directional edits "
            "change identity.entities.origin or identity.entities.destination directly"
        )
    )
