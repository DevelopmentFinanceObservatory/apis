"""The registered axis rules, the conflicts between their options, and the rule version.

`AXIS_RULES` maps each live axis to its rule, in `models.vocab.SCOPE_AXES` order; an axis
with no rule is left unsettled. A rule reads the base identity, the record the part's binder
wrote for its axis (None on a dataset_resubmit, where no binder runs) and the engine facts,
and returns the axis's table, or None when the axis is absent from the part.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from collections.abc import Callable, Mapping

from code_index import declarations, scope_conventions
from search.models import BaseIdentity, ScopeAxis
from search.scope.basis import HEADLINE_OPTION, basis_rule
from search.scope.provider import INCLUSIVE, provider_rule
from search.scope.region import region_rule
from search.scope.sector import sector_rule
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, Conflict, ScopeFacts

type AxisRule = Callable[[BaseIdentity, AxisRecord | None, ScopeFacts], AxisTable | None]

AXIS_RULES: Mapping[ScopeAxis, AxisRule] = {
    "sector": sector_rule,
    "region": region_rule,
    "measure_basis": basis_rule,
    "provider": provider_rule,
}
"""The rule of every live axis."""

CONFLICTS: tuple[Conflict, ...] = (Conflict(INCLUSIVE, HEADLINE_OPTION, "both_composite_sources"),)
"""The pairs of options of different axes that cannot hold together on every part: the
inclusive provider scope and the OECD headline both bind the identity's source. A table adds
the conflicts of its own part (`AxisTable.conflicts`), as the provider table does for the
measure bases its inclusive source does not add on (`provider.provider_rule`)."""


def rule_version(rules: Mapping[ScopeAxis, AxisRule], conflicts: tuple[Conflict, ...]) -> str:
    """The hash of what a compile reads besides the request: the declared scope
    conventions, the containing totals a part's open side binds
    (`code_index.declarations.CONTAINING_TOTALS`), the live axes and the conflicts. A stored scope
    whose rule version differs is recompiled from its entries. Hashed once per distinct
    input, since the conventions are fixed for the life of the process."""
    return _version(
        tuple(sorted(rules)),
        conflicts,
        scope_conventions.SECTOR_GROUPINGS,
        scope_conventions.MEASURE_BASIS_CONVENTIONS,
        scope_conventions.PROVIDER_CONVENTIONS,
        scope_conventions.PROVIDER_SCOPE_RULES,
        scope_conventions.REDISTRIBUTED_PROVIDERS,
        scope_conventions.PUBLISHED_PROVIDER_SCOPES,
        scope_conventions.INCLUSIVE_SOURCES,
        scope_conventions.DONOR_ORGANISATIONS,
        scope_conventions.BASIS_MEASURES,
        scope_conventions.BASIS_RESTRICTIONS,
        scope_conventions.OPTION_LABELS,
        scope_conventions.REGION_NAMES,
        scope_conventions.CONVENTION_CURRENCY,
        scope_conventions.REGION_READINGS,
        tuple(
            (dataset, axis, row)
            for (dataset, axis), row in sorted(declarations.CONTAINING_TOTALS.items())
        ),
    )


@functools.cache
def _version(
    axes: tuple[str, ...],
    conflicts: tuple[Conflict, ...],
    sector_groupings: tuple[scope_conventions.SectorGrouping, ...],
    basis_conventions: tuple[scope_conventions.BasisConvention, ...],
    provider_conventions: tuple[scope_conventions.ProviderConvention, ...],
    provider_scope_rules: tuple[scope_conventions.ProviderScopeRule, ...],
    redistributed_providers: tuple[scope_conventions.RedistributedProvider, ...],
    published_provider_scopes: tuple[scope_conventions.PublishedProviderScope, ...],
    inclusive_sources: tuple[scope_conventions.InclusiveSource, ...],
    donor_organisations: tuple[str, ...],
    basis_measures: tuple[scope_conventions.BasisMeasure, ...],
    basis_restrictions: tuple[str, ...],
    option_labels: scope_conventions.OptionLabels,
    region_names: tuple[tuple[str, str], ...],
    convention_currency: str,
    region_readings: tuple[scope_conventions.RegionReading, ...],
    containing_totals: tuple[tuple[str, str, declarations.ContainingTotal], ...],
) -> str:
    document = {
        "sector_groupings": [dataclasses.asdict(row) for row in sector_groupings],
        "measure_basis_conventions": [dataclasses.asdict(row) for row in basis_conventions],
        "provider_conventions": [dataclasses.asdict(row) for row in provider_conventions],
        "provider_scope_rules": [dataclasses.asdict(row) for row in provider_scope_rules],
        "redistributed_providers": [dataclasses.asdict(row) for row in redistributed_providers],
        "published_provider_scopes": [dataclasses.asdict(row) for row in published_provider_scopes],
        "inclusive_sources": [dataclasses.asdict(row) for row in inclusive_sources],
        "donor_organisations": list(donor_organisations),
        "basis_measures": [dataclasses.asdict(row) for row in basis_measures],
        "basis_restrictions": list(basis_restrictions),
        "option_labels": dataclasses.asdict(option_labels),
        "region_names": [list(pair) for pair in region_names],
        "convention_currency": convention_currency,
        "region_readings": [dataclasses.asdict(row) for row in region_readings],
        "containing_totals": [
            [dataset, axis, dataclasses.asdict(row)] for dataset, axis, row in containing_totals
        ],
        "axes": list(axes),
        "conflicts": [dataclasses.asdict(conflict) for conflict in conflicts],
    }
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return "scope1:" + hashlib.sha256(text.encode()).hexdigest()[:32]
