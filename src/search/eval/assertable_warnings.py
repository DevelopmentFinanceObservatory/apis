"""Allow warning assertions only where firing is deterministic for a case's door.

A case's ``warnings`` lists codes the answer must carry. The allowlist uses
``(code, door)`` because the same warning can depend on model output for one door and on
request or graph data for another. Tests check emission sites, and the registry check
confirms each pair is registered for ``search.resolve`` on that door.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import get_args

from dfo_spec_kit.registry import load_registry

from search.engine.codes import WARNING_CODES
from search.eval.cases import Case
from search.models import EntryPath

CASE_DOORS: frozenset[str] = frozenset(get_args(EntryPath))
"""The doors a case may name, the request's own input kinds."""

# (warning code, door) pairs a case may list under warnings. Each listed pair passed an
# emission-site check for request/graph-only behaviour.
ASSERTABLE_WARNINGS: frozenset[tuple[str, str]] = frozenset(
    {
        # role_scope_defaulted: parsed and dataset_resubmit read bound entities from
        # the request. raw_text depends on extraction output.
        ("role_scope_defaulted", "parsed"),
        ("role_scope_defaulted", "dataset_resubmit"),
        # conjoin.combine_regions emits binding_folded when same-shape regions fold.
        # Parsed doors post each leg. raw_text depends on extraction.
        ("binding_folded", "parsed"),
        ("binding_folded", "dataset_resubmit"),
        # assemble.py emits observations_omitted from the request's
        # include_observations flag on candidate parts.
        ("observations_omitted", "raw_text"),
        ("observations_omitted", "parsed"),
        ("observations_omitted", "dataset_resubmit"),
        # _ground_answer resolves labels from graph content. Parsed and resubmit
        # entity dcids come from the request. raw_text depends on extraction.
        ("entity_label_unresolved", "parsed"),
        ("entity_label_unresolved", "dataset_resubmit"),
        # dispatch._deflator_projected_warning fires when a converted series has projected
        # points, and only while the request sets include_observations, which a case
        # opts into with its own include_observations field. A resubmit posts the
        # identity's unit and window, and the projected rate or deflator facets come from
        # the graph, so no model output decides it.
        ("deflator_projected", "dataset_resubmit"),
        # The bound-unit gate (convertibility.unit_coverage) emits currency_unmapped_omitted
        # when a conversion into each entity's own currency leaves out entities no currency
        # names. A resubmit posts the identity's unit and roles, and nationalCurrencyOf comes
        # from the graph, so no model output decides it.
        ("currency_unmapped_omitted", "dataset_resubmit"),
        # The answer's deflator omission (convertibility.deflator_omissions) reads the
        # posted unit and base year and the deflator tables, so no model output decides it.
        ("deflator_missing_omitted", "dataset_resubmit"),
    }
)


def _required_codes(case: Case) -> Iterator[str]:
    """Every warning code any status block of ``case`` requires, parts included."""
    expectations = case.expect.parts or [case.expect]
    for expectation in expectations:
        for status in expectation.accepted_statuses:
            block = expectation.for_status(status)
            for code in (block.warnings if block is not None else None) or []:
                if isinstance(code, str):
                    yield code


def warning_violations(case: Case) -> list[str]:
    """One message per required warning code outside the allowlist for the case's door."""
    return [
        f"{case.id}: warnings names {code!r} on door {case.door!r}, "
        "which is not in ASSERTABLE_WARNINGS"
        for code in dict.fromkeys(_required_codes(case))
        if (code, case.door) not in ASSERTABLE_WARNINGS
    ]


def check_cases_assertable_warnings(cases: Iterable[Case]) -> list[str]:
    """Every assertable-warnings violation across ``cases``, in case order."""
    return [violation for case in cases for violation in warning_violations(case)]


def _registry_resolve_doors_by_code() -> dict[str, frozenset[str] | None]:
    """Every door a `search.resolve` mapping in the pinned dfo-spec-kit registry
    names for each active code, unioned across every such mapping. `None` means
    at least one matching mapping named no `doors` of its own, which the
    registry format treats as applying to every door (the same convention
    `tests/platform/test_registry_binding.py`'s `_registered_resolve_warning_doors`
    reads).
    """
    doors_by_code: dict[str, set[str] | None] = {}
    for entry in load_registry().entries:
        if entry.lifecycle != "active":
            continue
        for mapping in entry.mappings:
            if mapping.endpoint != "search.resolve":
                continue
            existing = doors_by_code.get(entry.code, set())
            if existing is None or mapping.doors is None:
                doors_by_code[entry.code] = None
            else:
                doors_by_code[entry.code] = existing | set(mapping.doors)
    return {
        code: (None if doors is None else frozenset(doors)) for code, doors in doors_by_code.items()
    }


def _pairs_not_registered(
    pairs: Iterable[tuple[str, str]], doors_by_code: Mapping[str, frozenset[str] | None]
) -> list[tuple[str, str]]:
    """Return pairs without an active ``search.resolve`` door registration.

    Codes absent from the map through missing/inactive entries or no resolve mapping
    are missing for every door. A registered code mapped to None applies to all doors.
    """
    missing = []
    for code, door in sorted(pairs):
        if code not in doors_by_code:
            missing.append((code, door))
            continue
        doors = doors_by_code[code]
        if doors is not None and door not in doors:
            missing.append((code, door))
    return missing


def doors_not_registered() -> list[tuple[str, str]]:
    """Return allowlisted pairs absent from the pinned search.resolve registry mappings."""
    return _pairs_not_registered(ASSERTABLE_WARNINGS, _registry_resolve_doors_by_code())


# Validate warning codes and doors against their registries at import time.
for _code, _door in ASSERTABLE_WARNINGS:
    if _code not in WARNING_CODES:
        raise ValueError(f"ASSERTABLE_WARNINGS names {_code!r}, not in WARNING_CODES")
    if _door not in CASE_DOORS:
        raise ValueError(f"ASSERTABLE_WARNINGS names door {_door!r}, not in CASE_DOORS")
del _code, _door

_unregistered = doors_not_registered()
if _unregistered:
    raise ValueError(
        "ASSERTABLE_WARNINGS names (code, door) pair(s) the pinned registry does "
        f"not register on search.resolve for that door: {_unregistered}"
    )
del _unregistered
