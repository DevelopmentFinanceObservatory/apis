"""Read the default facet each admitted StatVar declares in the graph.

A mint serves a StatVar's declared default on an unbound axis ahead of the deployment
ranking (`materialise.facets.select_facet`, `materialise.group_facets.elect_group`). Search
reads the same declarations once per admitted StatVar, and every read plan built over them
elects under them (`read_plan.in_elected_datasets`), so a preview, the answer's facets and a
download serve one method.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from materialise.facets import FACET_AXES

if TYPE_CHECKING:
    from materialise.graph import FacetDefaultPolicy
    from search.engine.clients.graph import EngineGraphClient

type DeclaredPolicies = Mapping[str, Mapping[str, FacetDefaultPolicy]]
"""The defaults each StatVar declares, by StatVar and then by facet axis. A StatVar or axis
declaring nothing is absent."""

NO_DECLARED_POLICIES: DeclaredPolicies = MappingProxyType({})
"""The declarations of StatVars that declare nothing."""

# Materialise asks for policies by the `dcid:`-prefixed ref its requests carry.
_STAT_VAR_REF_PREFIX = "dcid:"


def read_declared_policies(
    graph: EngineGraphClient, stat_vars: Iterable[str]
) -> dict[str, dict[str, FacetDefaultPolicy]]:
    """The default each of `stat_vars` declares on each facet axis
    (`EngineGraphClient.default_facet_policy`), the read a mint of each takes. The origin
    client answers a held StatVar from the catalogue's arc rows with no graph request."""
    declared: dict[str, dict[str, FacetDefaultPolicy]] = {}
    for stat_var in dict.fromkeys(stat_vars):
        axes = {
            axis: policy
            for axis in FACET_AXES
            if (
                policy := graph.default_facet_policy(
                    stat_var=f"{_STAT_VAR_REF_PREFIX}{stat_var}", axis=axis
                )
            )
            is not None
        }
        if axes:
            declared[stat_var] = axes
    return declared
