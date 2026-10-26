"""The registered axis rules, the conflicts between their options, and the rule version.

`AXIS_RULES` maps each live axis to its rule, in `models.vocab.SCOPE_AXES` order; an axis
with no rule is never settled. A rule reads the base identity, the record the part's binder
wrote for its axis (None on a dataset_resubmit, where no binder runs) and the engine facts,
and returns the axis's table, or None when the axis is absent from the part.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from collections.abc import Callable, Mapping

from code_index import scope_conventions
from search.models import BaseIdentity, ScopeAxis
from search.scope.basis import HEADLINE_OPTION, basis_rule
from search.scope.provider import INCLUSIVE, provider_rule
from search.scope.region import region_rule
from search.scope.sector import sector_rule
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, Conflict, ScopeFacts
from search.scope.vocabulary import measure_basis_option_id

type AxisRule = Callable[[BaseIdentity, AxisRecord | None, ScopeFacts], AxisTable | None]

AXIS_RULES: Mapping[ScopeAxis, AxisRule] = {
    "sector": sector_rule,
    "region": region_rule,
    "measure_basis": basis_rule,
    "provider": provider_rule,
}
"""The rule of every live axis."""

CONFLICTS: tuple[Conflict, ...] = (
    *(
        Conflict(INCLUSIVE, measure_basis_option_id(basis), "imputed_gross_disbursements_only")
        for basis in ("net", "grant_equivalent", "commitments")
    ),
    Conflict(INCLUSIVE, HEADLINE_OPTION, "both_composite_sources"),
)
"""The pairs of options of different axes that cannot hold together. The inclusive provider
scope holds only with gross disbursements, the imputed table's one basis, and both it and the
OECD headline bind the identity's source."""


def rule_version(rules: Mapping[ScopeAxis, AxisRule], conflicts: tuple[Conflict, ...]) -> str:
    """The hash of what a compile reads besides the request: the declared scope
    conventions, the live axes and the conflicts. A stored scope
    whose rule version differs is recompiled from its entries. Hashed once per distinct
    input, since the conventions are fixed for the life of the process."""
    return _version(
        tuple(sorted(rules)),
        conflicts,
        scope_conventions.SECTOR_GROUPINGS,
        scope_conventions.MEASURE_BASIS_CONVENTIONS,
        scope_conventions.PROVIDER_CONVENTIONS,
        scope_conventions.BASIS_MEASURES,
        scope_conventions.BASIS_RESTRICTIONS,
        scope_conventions.OPTION_LABELS,
        scope_conventions.REGION_NAMES,
        scope_conventions.CONVENTION_CURRENCY,
    )


@functools.cache
def _version(
    axes: tuple[str, ...],
    conflicts: tuple[Conflict, ...],
    sector_groupings: tuple[scope_conventions.SectorGrouping, ...],
    basis_conventions: tuple[scope_conventions.BasisConvention, ...],
    provider_conventions: tuple[scope_conventions.ProviderConvention, ...],
    basis_measures: tuple[scope_conventions.BasisMeasure, ...],
    basis_restrictions: tuple[str, ...],
    option_labels: scope_conventions.OptionLabels,
    region_names: tuple[tuple[str, str], ...],
    convention_currency: str,
) -> str:
    document = {
        "sector_groupings": [dataclasses.asdict(row) for row in sector_groupings],
        "measure_basis_conventions": [dataclasses.asdict(row) for row in basis_conventions],
        "provider_conventions": [dataclasses.asdict(row) for row in provider_conventions],
        "basis_measures": [dataclasses.asdict(row) for row in basis_measures],
        "basis_restrictions": list(basis_restrictions),
        "option_labels": dataclasses.asdict(option_labels),
        "region_names": [list(pair) for pair in region_names],
        "convention_currency": convention_currency,
        "axes": list(axes),
        "conflicts": [dataclasses.asdict(conflict) for conflict in conflicts],
    }
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return "scope1:" + hashlib.sha256(text.encode()).hexdigest()[:32]
