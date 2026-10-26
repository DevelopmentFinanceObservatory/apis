"""The registered axis rules, the conflicts between their options, and the rule version.

`AXIS_RULES` maps each live axis to its rule, in `models.vocab.SCOPE_AXES` order; an axis
with no rule is never settled. A rule reads the base identity, the record the part's binder
wrote for its axis (None on a dataset_resubmit, where no binder runs) and the engine facts,
and returns the axis's table, or None when the axis is absent from the part.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping

from code_index import scope_conventions
from search.models import BaseIdentity, ScopeAxis
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, Conflict, ScopeFacts

type AxisRule = Callable[[BaseIdentity, AxisRecord | None, ScopeFacts], AxisTable | None]

AXIS_RULES: Mapping[ScopeAxis, AxisRule] = {}
"""The rule of every live axis."""

CONFLICTS: tuple[Conflict, ...] = ()
"""The pairs of options of different axes that cannot hold together."""


def rule_version(rules: Mapping[ScopeAxis, AxisRule], conflicts: tuple[Conflict, ...]) -> str:
    """The hash of what a compile reads besides the request: the declared scope
    conventions, the live axes and the conflicts. A stored scope
    whose rule version differs is recompiled from its entries."""
    document = {
        "sector_groupings": [dataclasses.asdict(row) for row in scope_conventions.SECTOR_GROUPINGS],
        "measure_basis_conventions": [
            dataclasses.asdict(row) for row in scope_conventions.MEASURE_BASIS_CONVENTIONS
        ],
        "axes": sorted(rules),
        "conflicts": [dataclasses.asdict(conflict) for conflict in conflicts],
    }
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return "scope1:" + hashlib.sha256(text.encode()).hexdigest()[:32]
