"""Read the default facet each admitted StatVar declares in the graph.

A mint serves a StatVar's declared default on an unbound axis ahead of the deployment
ranking (`materialise.facets.select_facet`, `materialise.group_facets.elect_group`). Search
reads the same declarations once for every admitted StatVar, and every read plan elects under
them from its first election (`read_plan.plan_pairs`, `read_plan.role_election`), so a
preview, the answer's facets, an option's coverage and a download serve one method.

A failed read raises. An answer elected without its members' declarations would name a Spec
that a mint, reading them once the graph answers, serves under another method, so the
region fails as on any other graph read its answer needs. The options door, whose coverage
is optional, returns its options without coverage instead
(`options.surface._observation_about_options_set`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from materialise.graph import FacetDefaultPolicy
    from search.engine.clients.graph import EngineGraphClient

type DeclaredPolicies = Mapping[str, Mapping[str, FacetDefaultPolicy]]
"""The defaults each StatVar declares, by StatVar and then by facet axis. A StatVar or axis
declaring nothing is absent."""

NO_DECLARED_POLICIES: DeclaredPolicies = MappingProxyType({})
"""The declarations of StatVars that declare nothing."""


def read_declared_policies(
    graph: EngineGraphClient, stat_vars: Iterable[str], *, deadline: float | None
) -> DeclaredPolicies:
    """The default each of `stat_vars` declares on each facet axis, the declaration a mint of
    each reads (`GraphReader.default_facet_policy`), in one read
    (`EngineGraphClient.default_facet_policies`). The origin client answers a variable the
    catalogue holds from its arc rows and reads the rest in one batch, by `deadline` and the
    request deadline bound to the context.

    Raises:
        GraphInfraError: the graph failed the read. No ranking stands in for the
            declarations, since the answer would then serve a method its Spec's mint does not.
    """
    dcids = list(dict.fromkeys(stat_vars))
    if not dcids:
        return NO_DECLARED_POLICIES
    return graph.default_facet_policies(dcids, deadline=deadline)
