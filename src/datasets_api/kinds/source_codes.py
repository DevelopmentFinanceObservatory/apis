"""Resolves source codes through `member` arcs, then mints the StatVar set.

The lookup key is `{source}:{code}` when `source` is set, otherwise `code`.
An absent code raises `code_not_found`. Multiple matches raise `code_ambiguous` with every
candidate.
"""

from __future__ import annotations

from collections.abc import Sequence

from datasets_api.codes_materialise import CodeAmbiguous, CodeNotFoundError
from datasets_api.kinds.statvar_set import mint_statvar_set, preview_statvar_set
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.preview_models import PreviewEnvelope
from datasets_api.timings import phase
from materialise.graph import GraphReader

_STAT_VAR_PREFIX = "dcid:"
_MEMBER_PROPERTY = "member"


def _lookup_key(code: str, *, source: str | None) -> str:
    return f"{source}:{code}" if source is not None else code


def _resolve_code(
    code: str, *, source: str | None, graph: GraphReader
) -> str | tuple[str, tuple[tuple[str, str], ...]]:
    """Returns a `dcid:`-prefixed ref or the ambiguous code and its candidates.

    Raises `CodeNotFoundError` immediately. The caller aggregates ambiguity.
    """
    arcs = graph.node_arc_values(_lookup_key(code, source=source), properties=(_MEMBER_PROPERTY,))
    candidates = arcs.get(_MEMBER_PROPERTY, ())
    if not candidates:
        raise CodeNotFoundError(code=code, source=source)
    if len(candidates) > 1:
        labelled = tuple((c, graph.node_label(c) or c) for c in candidates)
        return code, labelled
    resolved = candidates[0]
    return resolved if resolved.startswith(_STAT_VAR_PREFIX) else f"{_STAT_VAR_PREFIX}{resolved}"


def _resolve_codes(codes: Sequence[str], *, source: str | None, graph: GraphReader) -> list[str]:
    """Resolves each code and aggregates ambiguous codes into one refusal.

    A missing code raises immediately. Both mint and preview use this resolver.
    """
    resolved: list[str] = []
    ambiguous: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    ambiguous_indices: list[int] = []
    for index, code in enumerate(codes):
        try:
            outcome = _resolve_code(code, source=source, graph=graph)
        except CodeNotFoundError as exc:
            # Attach the index to the exception. The mint classifier uses it for the pointer.
            exc.index = index
            raise
        if isinstance(outcome, str):
            resolved.append(outcome)
        else:
            ambiguous.append(outcome)
            ambiguous_indices.append(index)
    if ambiguous:
        # Keep request indexes aligned with ambiguous codes for the mint classifier.
        exc = CodeAmbiguous(ambiguous=tuple(ambiguous))
        exc.indices = tuple(ambiguous_indices)
        raise exc
    return resolved


def mint_source_codes(
    codes: Sequence[str],
    *,
    source: str | None,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
    trace_id: str | None = None,
) -> Manifest:
    with phase("resolve"):
        resolved = _resolve_codes(codes, source=source, graph=deps.graph)
    return mint_statvar_set(resolved, scope=scope, options=options, deps=deps, trace_id=trace_id)


def preview_source_codes(
    codes: Sequence[str], *, source: str | None, scope: Scope | None, limit: int, deps: MintDeps
) -> PreviewEnvelope:
    """Resolves codes as `mint_source_codes` does, without registering anything."""
    with phase("resolve"):
        resolved = _resolve_codes(codes, source=source, graph=deps.graph)
    return preview_statvar_set(resolved, scope=scope, limit=limit, deps=deps)
