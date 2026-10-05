"""Pure key builders and storage gates for the two-layer response cache.

Both layers hash RFC 8785 canonical JSON with SHA-256. Their shared scope
includes request options, the engine fingerprint, day, graph generation,
coverage stamp, and conversion catalogue revision; token changes invalidate
both layers.
Layer 1 adds the normalised query; layer 2 adds the effective extraction
and raw-text signals. The `l1_` and `l2_` prefixes separate keys.
Entry encoding and storability checks also live here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict

import rfc8785

from search.engine.envelope import make_diagnostics, make_query_echo, now_ms
from search.engine.extract import ExtractedDate, ExtractedEntity
from search.engine.qualifier_match import CodedQualifier, QualifierMatch, recall_text
from search.engine.region.codes import MEMBERSHIP_UNVERIFIED, UNIT_CONVERSION_UNAVAILABLE
from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    NoDataPart,
    Operation,
    Part,
    ResolveOptions,
    ResolveResponse,
)

_INTERNAL_WHITESPACE = re.compile(r"\s+")
# Warnings that mark an answer a dependency could not complete: the skills index, the UN
# System Data Commons, the member walk that settles a role-bearing answer's members, or the
# conversion catalogue its unit conversion offer reads.
_UNSTORABLE_WARNING_CODES = frozenset(
    {
        "skills_index_unavailable",
        "skills_index_incomplete",
        "un_system_unavailable",
        MEMBERSHIP_UNVERIFIED,
        UNIT_CONVERSION_UNAVAILABLE,
    }
)


def normalise_query(query: str) -> str:
    """NFKC-normalise, strip, and collapse whitespace while preserving case.

    Layer 1 preserves case because extraction can depend on it. Layer 2 can
    still share results when case variants produce the same extraction.
    """
    return _INTERNAL_WHITESPACE.sub(" ", unicodedata.normalize("NFKC", query).strip())


def _resolved_options(options: ResolveOptions | None, *, max_candidates: int) -> dict[str, object]:
    """Apply option defaults and use the server-clamped candidate cap.

    Requests with different raw caps that resolve to the same effective cap
    share a key; changing the server ceiling changes it.
    """
    if options is None:
        return {
            "max_candidates": max_candidates,
            "include_sentence": True,
            "include_option_coverage": False,
            "include_observations": False,
            "include_skills": False,
        }
    return {
        "max_candidates": max_candidates,
        "include_sentence": True if options.include_sentence is None else options.include_sentence,
        "include_option_coverage": bool(options.include_option_coverage),
        "include_observations": bool(options.include_observations),
        "include_skills": bool(options.include_skills),
    }


def scope_document(
    *,
    schema_version: int,
    options: ResolveOptions | None,
    max_candidates: int,
    seam_default: bool,
    build_commit: str,
    model: str,
    seed: int | None,
    today: str,
    graph_generation: str,
    coverage_stamp: str,
    conversion_catalogue: str | None,
) -> dict:
    """Build shared key material from caller-supplied values.

    Includes the schema, defaulted response options, effective candidate cap,
    place-as-constraint default, build/model/seed fingerprint, day, graph and
    coverage stamps, and the revision of the conversion catalogue the request
    bound (None when it bound none). The cap and place default can change answers
    without changing the build commit. A catalogue reload changes the revision
    and not the graph generation. Each stamp is captured once and invalidates both
    layers when it changes. Transport is omitted because option-coverage and
    observation requests are refused by `is_cacheable_request`.
    """
    return {
        "schema_version": schema_version,
        "response_options": _resolved_options(options, max_candidates=max_candidates),
        "seam_default": seam_default,
        "build_commit": build_commit,
        "model": model,
        "seed": seed,
        "today": today,
        "graph_generation": graph_generation,
        "coverage_stamp": coverage_stamp,
        "conversion_catalogue": conversion_catalogue,
    }


def _digest(prefix: str, document: dict) -> str:
    canonical = rfc8785.dumps(document)
    return f"{prefix}_{hashlib.sha256(canonical).hexdigest()}"


def layer1_key(*, query: str, scope: dict) -> str:
    """Key on the normalised query string alone, on top of the shared scope."""
    document = {"scope": scope, "query": normalise_query(query)}
    return _digest("l1", document)


def _qualifier_match_sort_key(match: QualifierMatch) -> tuple[str, str, str, str]:
    """Sort matches by their four string fields for stable cache keys."""
    return (match.code_system, match.code_dcid, match.source, match.matched_text)


def _coded_qualifier_sort_key(
    qualifier: CodedQualifier,
) -> tuple[str, tuple[tuple[str, str, str, str], ...]]:
    """Sort by phrase and sorted match content, independent of extraction order."""
    return (
        qualifier.phrase,
        tuple(_qualifier_match_sort_key(match) for match in qualifier.matches),
    )


def _variable_qualifiers_document(
    variables: Sequence[str], variable_qualifiers: Sequence[tuple[CodedQualifier, ...]]
) -> dict[str, dict[str, object]]:
    """Build each variable's qualifier sidecar for the layer-2 key.

    `recall_text` preserves extraction order because that order can change the
    composed retrieval input. The qualifier list is sorted by content because
    binding and warnings depend on its set, not extraction order. Variables
    without qualifiers are omitted; both input sequences are parallel.
    """
    return {
        variable: {
            "recall_text": recall_text(variable, qualifiers),
            "qualifiers": [
                {
                    "phrase": qualifier.phrase,
                    "matches": [
                        asdict(match)
                        for match in sorted(qualifier.matches, key=_qualifier_match_sort_key)
                    ],
                }
                for qualifier in sorted(qualifiers, key=_coded_qualifier_sort_key)
            ],
        }
        for variable, qualifiers in zip(variables, variable_qualifiers, strict=True)
        if qualifiers
    }


def _operations_document(
    variables: Sequence[str], operations: Sequence[tuple[Operation, ...]]
) -> dict[str, list[dict[str, object]]]:
    """Build each variable's operations for the layer-2 key, in query order.

    Order is kept because each part echoes its operations in that order. Variables
    without operations are omitted; both input sequences are parallel.
    """
    return {
        variable: [operation.model_dump(mode="json") for operation in variable_operations]
        for variable, variable_operations in zip(variables, operations, strict=True)
        if variable_operations
    }


def layer2_key(
    *,
    variables: Sequence[str],
    entities: Sequence[ExtractedEntity],
    dates: Sequence[ExtractedDate],
    dropped_qualifiers: Sequence[str],
    variable_qualifiers: Sequence[tuple[CodedQualifier, ...]],
    operations: Sequence[tuple[Operation, ...]],
    role_contradictions: Sequence[Mapping[str, object]],
    query_text_qualifier_matches: Sequence[Mapping[str, object]],
    query_text_qualifier_ties: Sequence[Mapping[str, object]],
    query_text_qualifier_readings: Sequence[Mapping[str, object]],
    unsupported_qualifier_phrases: Sequence[str],
    declared_default_trigger_words: Sequence[Mapping[str, str]],
    declared_default_slots_named: Sequence[str],
    scope: dict,
) -> str:
    """Key the effective extraction and raw-text signals over the shared scope.

    Full entity objects preserve directional roles. `variable_qualifiers`
    retains typed phrases absent from `variables`; phrase text reaches warnings
    and resolved qualifiers. Its `recall_text` stays in extraction order because
    phrase containment can change the retrieval input, while qualifier records
    are sorted because binding depends on their set. `operations` runs parallel to
    `variables`: each part echoes its own, and their phrases reach its bind.

    The raw-text signals cover role contradictions, query-text qualifier matches,
    ties and two-reading spans, unsupported phrases, declared-default trigger words, and the
    slots with a declared default whose plain name the query names. They keep
    layer 2 sound when distinct queries produce the same extraction without
    hashing raw query text, preserving reuse when those readers agree.
    """
    document = {
        "scope": scope,
        "variables": list(variables),
        "entities": [entity.model_dump(mode="json") for entity in entities],
        "dates": [date.model_dump(mode="json") for date in dates],
        "dropped_qualifiers": list(dropped_qualifiers),
        "variable_qualifiers": _variable_qualifiers_document(variables, variable_qualifiers),
        "operations": _operations_document(variables, operations),
        "role_contradictions": list(role_contradictions),
        "query_text_qualifier_matches": list(query_text_qualifier_matches),
        "query_text_qualifier_ties": list(query_text_qualifier_ties),
        "query_text_qualifier_readings": list(query_text_qualifier_readings),
        "unsupported_qualifier_phrases": list(unsupported_qualifier_phrases),
        "declared_default_trigger_words": [dict(word) for word in declared_default_trigger_words],
        "declared_default_slots_named": list(declared_default_slots_named),
    }
    return _digest("l2", document)


def is_cacheable_request(options: ResolveOptions | None) -> bool:
    """Reject option-coverage and observation requests; `None` is cacheable.

    Probe failures can appear in SSE `done` warnings without changing the
    response body, so these requests are uncacheable on every transport.
    """
    if options is None:
        return True
    return not (options.include_option_coverage or options.include_observations)


# ---------------------------------------------------------------------------
# Entry serialisation and the write-time storability gate
# ---------------------------------------------------------------------------


def to_entry(response: ResolveResponse) -> tuple[bytes, int]:
    """Encode a finished response of any kind and return its retained byte length.

    The bytes carry `kind`, so `from_entry` restores the same kind. The store
    keeps these bytes. A parsed-dict size estimate would undercount
    memory by 2.65 to 2.75 times on recorded responses.
    """
    encoded = response.model_dump_json().encode("utf-8")
    return encoded, len(encoded)


def from_entry(
    entry: bytes,
    *,
    query: str,
    extract_skipped: bool,
    start_ms: int,
    llm_usage: dict | None,
    timing_by_step: dict[str, int] | None = None,
) -> ResolveResponse:
    """Revalidate stored JSON and restore request-specific response fields.

    Replace `query_echo` and `diagnostics` so wording, trace id, timing, and
    usage describe this hit; stored warnings carry over unchanged. JSON
    validation reads the retained bytes directly. `timing_by_step` is omitted
    when the hit has no named cache step.
    """
    restored = RESOLVE_RESPONSE_ADAPTER.validate_json(entry)
    echo = make_query_echo(query, extract_skipped=extract_skipped)
    diagnostics = make_diagnostics(
        restored.diagnostics.engine_build,
        restored.diagnostics.warnings,
        timing_by_step or {},
        now_ms() - start_ms,
        llm_usage=llm_usage,
    )
    return restored.model_copy(update={"query_echo": echo, "diagnostics": diagnostics})


def _is_unresolved_placeholder(part: Part) -> bool:
    return isinstance(part, NoDataPart) and part.no_data.reason == "variable_not_resolved"


def is_storable(response: ResolveResponse, *, timed_out: bool, degraded_leg: bool) -> bool:
    """Return whether the response is complete and safe to cache.

    Reject timeouts, degraded legs, unavailable-skills, UN and unverified-membership
    warnings, and an unresolved-variable placeholder alongside a resolved part. A catalogue or
    declined answer has no parts, so only the first three checks apply to it.
    """
    if timed_out or degraded_leg:
        return False

    parts = response.parts if response.kind == "resolution" else []
    all_warnings = list(response.diagnostics.warnings)
    for part in parts:
        all_warnings.extend(part.warnings)
    if any(w.code in _UNSTORABLE_WARNING_CODES for w in all_warnings):
        return False

    placeholder = [p for p in parts if _is_unresolved_placeholder(p)]
    resolved = [p for p in parts if not _is_unresolved_placeholder(p)]
    return not (placeholder and resolved)
