"""Pure key builders and storage gates for the two-layer response cache and its part memo.

Every key hashes RFC 8785 canonical JSON with SHA-256. The layers' shared scope
includes request options, the engine fingerprint, day, graph generation,
coverage stamp, and conversion catalogue revision; token changes invalidate
both layers.
Layer 1 adds the normalised query; layer 2 adds the effective extraction
and raw-text signals, or a parsed request's whole input. The part memo keys a
resubmitted identity on its whole payload, the engine options and the same stamps.
The `l1_`, `l2_` and `m_` prefixes separate keys.
Entry encoding and storability checks also live here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Literal

import rfc8785

from search.engine.envelope import make_diagnostics, make_query_echo, now_ms
from search.engine.extract import ExtractedDate, ExtractedEntity
from search.engine.qualifier_match import CodedQualifier, QualifierMatch, recall_text
from search.engine.region.codes import (
    AVAILABILITY_PROBE_FAILED,
    MEMBERS_ABSENT_FROM_GRAPH,
    MEMBERSHIP_UNVERIFIED,
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_TIMED_OUT,
    OPTION_COVERAGE_PROBE_FAILED,
    OPTION_COVERAGE_TIMED_OUT,
    ROLE_DECLARATION_UNVERIFIED,
    UN_SYSTEM_UNAVAILABLE,
    UNIT_CONVERSION_UNAVAILABLE,
)
from search.engine.region.result import RegionResult
from search.models import (
    RESOLVE_RESPONSE_ADAPTER,
    NoDataPart,
    Operation,
    ParsedInput,
    Part,
    ResolveOptions,
    ResolveResponse,
    ResubmitIdentityPayload,
    SourceMention,
    Warning,
)

_INTERNAL_WHITESPACE = re.compile(r"\s+")
# Warnings that mark an answer a dependency could not complete: the skills index, the UN
# System Data Commons, the member walk that settles a role-bearing answer's members, the
# role declarations a population re-reads, the conversion catalogue its unit conversion
# offer reads, and the option-coverage, availability and observation reads that failed or
# ran past the deadline. An elected member the graph read did not return
# (`members_absent_from_graph`) can be a transient drop, so that answer is not kept either.
_UNSTORABLE_WARNING_CODES = frozenset(
    {
        "skills_index_unavailable",
        "skills_index_incomplete",
        UN_SYSTEM_UNAVAILABLE,
        MEMBERSHIP_UNVERIFIED,
        MEMBERS_ABSENT_FROM_GRAPH,
        ROLE_DECLARATION_UNVERIFIED,
        UNIT_CONVERSION_UNAVAILABLE,
        OPTION_COVERAGE_PROBE_FAILED,
        OPTION_COVERAGE_TIMED_OUT,
        AVAILABILITY_PROBE_FAILED,
        OBSERVATIONS_PROBE_FAILED,
        OBSERVATIONS_TIMED_OUT,
        OBSERVATIONS_BULK_FETCH_FAILED,
    }
)


def normalise_query(query: str) -> str:
    """NFKC-normalise, strip, and collapse whitespace while preserving case.

    Layer 1 preserves case because extraction can depend on it. Layer 2 can
    still share results when case variants produce the same extraction.
    """
    return _INTERNAL_WHITESPACE.sub(" ", unicodedata.normalize("NFKC", query).strip())


def _resolved_options(options: ResolveOptions | None, *, max_candidates: int) -> dict[str, object]:
    """Apply option defaults beside the server's candidate cap, so changing the cap
    changes the key."""
    if options is None:
        return {
            "max_candidates": max_candidates,
            "include_option_coverage": False,
            "include_observations": False,
            "include_skills": False,
        }
    return {
        "max_candidates": max_candidates,
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
    qualifier_coverage: Literal["log", "refuse"],
    build_commit: str,
    model: str,
    seed: int | None,
    today: str,
    graph_generation: str,
    coverage_stamp: str,
    conversion_catalogue: str | None,
    origin_catalogue: str,
    un_system_enabled: bool,
) -> dict:
    """Build shared key material from caller-supplied values.

    Includes the schema, defaulted response options, effective candidate cap,
    place-as-constraint default, qualifier coverage mode, build/model/seed fingerprint,
    day, graph and coverage stamps, the revision of the conversion catalogue the request
    bound (None when it bound none), and the build of the origin catalogue it bound with
    its UN System setting, which decide what provenances an answer admits. The cap, the
    place default and the coverage mode can change answers without changing the build
    commit. A catalogue
    reload changes the revision or build and not the graph generation. Each stamp is
    captured once and invalidates both layers when it changes. Transport is omitted because a request with trailing
    events uses neither layer, and without them both transports answer the same
    response.
    """
    return {
        "schema_version": schema_version,
        "response_options": _resolved_options(options, max_candidates=max_candidates),
        "seam_default": seam_default,
        "qualifier_coverage": qualifier_coverage,
        "build_commit": build_commit,
        "model": model,
        "seed": seed,
        "today": today,
        "graph_generation": graph_generation,
        "coverage_stamp": coverage_stamp,
        "conversion_catalogue": conversion_catalogue,
        "origin_catalogue": origin_catalogue,
        "un_system_enabled": un_system_enabled,
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


def _per_variable_document(
    variables: Sequence[str], items: Sequence[Sequence[Operation | SourceMention]]
) -> dict[str, list[dict[str, object]]]:
    """Build each variable's operations or source mentions for the layer-2 key, in query
    order.

    Order is kept because each part echoes them in that order. Variables without any are
    omitted. Both input sequences are parallel.
    """
    return {
        variable: [item.model_dump(mode="json") for item in variable_items]
        for variable, variable_items in zip(variables, items, strict=True)
        if variable_items
    }


def layer2_key(
    *,
    variables: Sequence[str],
    entities: Sequence[ExtractedEntity],
    dates: Sequence[ExtractedDate],
    dropped_qualifiers: Sequence[str],
    variable_qualifiers: Sequence[tuple[CodedQualifier, ...]],
    operations: Sequence[tuple[Operation, ...]],
    sources: Sequence[tuple[SourceMention, ...]],
    role_contradictions: Sequence[Mapping[str, object]],
    query_text_qualifier_matches: Sequence[Mapping[str, object]],
    query_text_qualifier_ties: Sequence[Mapping[str, object]],
    query_text_qualifier_readings: Sequence[Mapping[str, object]],
    query_text_qualifier_occurrences: Sequence[Mapping[str, object]],
    unsupported_qualifier_phrases: Sequence[str],
    declared_default_trigger_words: Sequence[Mapping[str, str]],
    declared_default_slots_named: Sequence[str],
    scope: dict,
    separate: Sequence[bool] = (),
) -> str:
    """Key the effective extraction and raw-text signals over the shared scope.

    Full entity objects preserve directional roles. `variable_qualifiers`
    retains typed phrases absent from `variables`; phrase text reaches warnings
    and resolved qualifiers. Its `recall_text` stays in extraction order because
    phrase containment can change the retrieval input, while qualifier records
    are sorted because binding depends on their set. `operations` runs parallel to
    `variables`: each part echoes its own, and their phrases reach its bind. `sources` runs
    parallel to `variables` too: the source mentions each part compiles, in query order, as
    the part echoes them, so "aid to Kenya" and "aid to Kenya excluding the CRS" keep
    separate entries.
    `separate` runs parallel to `variables` when given and marks each variable compared
    with the others, which never folds (`same_shape.fold_key`). The key states the marked
    positions and adds nothing when none is marked.

    The raw-text signals cover role contradictions, query-text qualifier matches,
    ties and two-reading spans, every place a row of any system names a query-text word
    (`qualifier_match.qualifier_occurrences`, which decides each shape vocabulary's
    reading), unsupported phrases, declared-default trigger words, and the
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
        "operations": _per_variable_document(variables, operations),
        "sources": _per_variable_document(variables, sources),
        "role_contradictions": list(role_contradictions),
        "query_text_qualifier_matches": list(query_text_qualifier_matches),
        "query_text_qualifier_ties": list(query_text_qualifier_ties),
        "query_text_qualifier_readings": list(query_text_qualifier_readings),
        "query_text_qualifier_occurrences": list(query_text_qualifier_occurrences),
        "unsupported_qualifier_phrases": list(unsupported_qualifier_phrases),
        "declared_default_trigger_words": [dict(word) for word in declared_default_trigger_words],
        "declared_default_slots_named": list(declared_default_slots_named),
    }
    if separate_positions := [i for i, marked in enumerate(separate) if marked]:
        document["separate"] = separate_positions
    return _digest("l2", document)


def parsed_key(*, parsed: ParsedInput, scope: dict) -> str:
    """Key a parsed request's whole input over the shared scope, on layer 2.

    The posted parts are the extraction a raw-text request derives, and every field of
    them, and the raw query its echo carries, reaches the answer."""
    document = {"scope": scope, "parsed": parsed.model_dump(mode="json")}
    return _digest("l2", document)


def part_memo_key(
    *,
    payload: ResubmitIdentityPayload,
    engine_flags: Mapping[str, bool],
    build_commit: str,
    graph_generation: str,
    coverage_stamp: str,
    conversion_catalogue: str | None,
    origin_catalogue: str,
    un_system_enabled: bool,
    today: str,
) -> str:
    """Key one resubmitted identity's region in the part memo.

    The payload is the identity and the named groups posted beside it, and the whole of it
    enters, since each field changes the answer.
    `engine_flags` are the four enrichment flags of the call's `EngineOptions`. The build commit covers the
    code index, and the graph generation, coverage stamp, conversion catalogue revision,
    origin catalogue build with its UN System setting, and UTC day cover what the graph,
    coverage and conversion reads return and what the answer admits."""
    document = {
        "payload": payload.model_dump(mode="json"),
        "engine_flags": dict(engine_flags),
        "build_commit": build_commit,
        "graph_generation": graph_generation,
        "coverage_stamp": coverage_stamp,
        "conversion_catalogue": conversion_catalogue,
        "origin_catalogue": origin_catalogue,
        "un_system_enabled": un_system_enabled,
        "today": today,
    }
    return _digest("m", document)


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
    query: str | None,
    extract_skipped: bool,
    start_ms: int,
    llm_usage: dict | None,
    timing_by_step: dict[str, int] | None = None,
) -> ResolveResponse:
    """Revalidate stored JSON and restore request-specific response fields.

    Replace `query_echo` and `diagnostics` so wording, trace id, timing, and
    usage describe this hit; stored warnings carry over unchanged. A `None`
    query keeps the stored echo, which a parsed request's key covers whole. JSON
    validation reads the retained bytes directly. `timing_by_step` is omitted
    when the hit has no named cache step.
    """
    restored = RESOLVE_RESPONSE_ADAPTER.validate_json(entry)
    echo = (
        restored.query_echo
        if query is None
        else make_query_echo(query, extract_skipped=extract_skipped)
    )
    diagnostics = make_diagnostics(
        restored.diagnostics.engine_build,
        restored.diagnostics.warnings,
        timing_by_step or {},
        now_ms() - start_ms,
        llm_usage=llm_usage,
    )
    return restored.model_copy(update={"query_echo": echo, "diagnostics": diagnostics})


def _has_unstorable_warning(warnings: Sequence[Warning]) -> bool:
    return any(w.code in _UNSTORABLE_WARNING_CODES for w in warnings)


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
    if _has_unstorable_warning(all_warnings):
        return False

    placeholder = [p for p in parts if _is_unresolved_placeholder(p)]
    resolved = [p for p in parts if not _is_unresolved_placeholder(p)]
    return not (placeholder and resolved)


def is_region_storable(region: RegionResult) -> bool:
    """Return whether a resubmitted identity's region is complete and safe to memoise.

    Reject a region that is a degraded leg (`RegionResult.leg_degraded`: it swallowed a read
    failure, a presence read left its default scope unsettled, or it left an alternate
    unevaluated), or that carries a warning `is_storable` rejects."""
    return not (region.leg_degraded or _has_unstorable_warning(region.warnings))
