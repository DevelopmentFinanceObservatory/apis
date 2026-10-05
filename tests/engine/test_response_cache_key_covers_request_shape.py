"""Freezes request field sets so new caller or response dimensions cannot bypass cache coverage.

New per-caller fields must enter `scope_document` or make the request uncacheable.
"""

from __future__ import annotations

from search.models import ResolveOptions, ResolveRequest

_EXPECTED_RESOLVE_REQUEST_FIELDS = frozenset({"schema_version", "input", "response_options"})

_EXPECTED_RESOLVE_OPTIONS_FIELDS = frozenset(
    {
        "max_candidates",
        "include_sentence",
        "include_option_coverage",
        "include_observations",
        "include_skills",
    }
)

_GUIDANCE = (
    "ResolveRequest/ResolveOptions gained or lost a field since this guard was written. "
    "Before touching the expected set below, decide whether the new field carries a "
    "per-caller dimension (auth, API key, tenant, locale, role, or anything else that "
    "could make one caller's cached answer reach another caller) or otherwise changes "
    "what answer a request resolves to. If it does either, fold it into "
    "search.engine.response_cache_keys.scope_document, or refuse the request from "
    "caching entirely via is_cacheable_request, before this expected set is updated to "
    "include it. Only once that decision is made and applied should the set below move."
)


def test_resolve_request_field_set_is_frozen():
    assert set(ResolveRequest.model_fields) == _EXPECTED_RESOLVE_REQUEST_FIELDS, _GUIDANCE


def test_resolve_options_field_set_is_frozen():
    assert set(ResolveOptions.model_fields) == _EXPECTED_RESOLVE_OPTIONS_FIELDS, _GUIDANCE
