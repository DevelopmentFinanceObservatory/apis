"""Resolves one StatVar and mints its fully pinned spec.

The shape resolves before any write. An unresolved StatVar refuses `member_not_resolvable`.
"""

from __future__ import annotations

from collections.abc import Sequence

from datasets_api.adapter import build_identity_document, to_materialisation_request
from datasets_api.kinds.spec_id import resolve_served_definition, serve_mint
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.shape import resolve_shape
from datasets_api.timings import phase
from spec_registry.identity import recompute_identity


def mint_statvar_set(
    statvars: Sequence[str],
    *,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
    trace_id: str | None = None,
) -> Manifest:
    """Mints the pinned spec of the one StatVar in `statvars`, which `StatVarSetInput` caps at
    one. The identity is the document the registration stores, so the mint materialises what
    the registry reads back, and `kinds.spec_id.serve_mint` serves it.
    Resolution can raise `MemberNotResolvable`."""
    (stat_var,) = statvars
    with phase("resolve"):
        shape = resolve_shape(stat_var, graph=deps.graph)
        recomputed = recompute_identity(
            build_identity_document(shape, scope=scope, graph=deps.graph)
        )
        deps.registry.register_definition(
            identity_definition=recomputed.identity,
            schema_version=1,
            supplied_spec_id=recomputed.spec_id,
            trace_id=trace_id,
        )
        # No manifest returns before the registry write is read back.
        stored = deps.spec_reader.read_definition(recomputed.spec_id)
        assert stored is not None, "definition missing immediately after its own registration"
        stored = resolve_served_definition(
            stored,
            code_index=deps.code_index,
            registry=deps.registry,
            trace_id=trace_id,
        )
        request = to_materialisation_request(
            stored.identity, shape=shape, code_index=deps.code_index
        )

    return serve_mint(stored, requests=(request,), options=options, deps=deps, trace_id=trace_id)
